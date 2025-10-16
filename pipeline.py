"""
PII Pipeline Module
=================

Complete end-to-end pipeline for PII redaction including:
1. OCR text extraction and annotation
2. Text PII detection and redaction  
3. Face detection and redaction

Main Endpoint: process_document() -> Complete pipeline processing
"""

import os
from typing import Dict, Any, Optional
from .ocr_processor import OCRProcessor
from .text_pii_redactor import TextPIIRedactor
from .face_redactor import FaceRedactor
from .config import PIIConfig

class PIIPipeline:
    """
    Complete PII redaction pipeline
    
    Main Endpoint:
    - process_document(image_path) -> Complete processing with all outputs
    """
    
    def __init__(self, config: PIIConfig = None):

        self.config = config or PIIConfig()
        # Initialize all processors with the respective configuration as defined in config.py
        self.ocr_processor = OCRProcessor(self.config)
        self.text_redactor = TextPIIRedactor(self.config)
        self.face_redactor = FaceRedactor(self.config)
        
    
    def process_document(self, input_path: str, save_intermediate: bool = True) -> Dict[str, Any]:
        """
        Main pipeline: handles image and PDF input, runs text and face redaction.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        _, ext = os.path.splitext(input_path.lower())
        results = {'input_path': input_path, 'processing_steps': []}

        if ext == '.pdf':
            # PDF: text redaction, then face redaction on page images + embedded images
            from .text_processor import redact_pdf_preserve_layout
            output_pdf = os.path.join(os.path.dirname(input_path), 'redacted_' + os.path.basename(input_path))
            pdf_redact_success = redact_pdf_preserve_layout(input_path, output_pdf)
            results['pdf_redacted_path'] = output_pdf
            results['processing_steps'].append('PDF_TEXT_REDACTED')
            if not pdf_redact_success:
                results['summary'] = {'status': 'PDF redaction failed'}
                return results
            
            # Step 1: Extract embedded images from PDF for face detection
            print("\n📷 Extracting embedded images from PDF...")
            embedded_images = self._extract_pdf_images(output_pdf)
            results['embedded_images'] = embedded_images
            
            # Step 2: Convert PDF pages to images
            try:
                from pdf2image import convert_from_path
                print(f"\n📄 Converting PDF pages to images (DPI: {self.config.IMAGE_CONFIG.get('dpi', 200)})...")
                page_images = convert_from_path(output_pdf, dpi=self.config.IMAGE_CONFIG.get('dpi', 200))
                image_paths = []
                for i, img in enumerate(page_images):
                    img_path = os.path.join(os.path.dirname(output_pdf), f'redacted_page_{i+1}.jpg')
                    img.save(img_path)
                    image_paths.append(img_path)
                results['pdf_page_images'] = image_paths
                print(f"✓ Converted {len(image_paths)} pages to images")
            except Exception as e:
                results['summary'] = {'status': f'PDF to image conversion failed: {e}'}
                return results
            
            # Step 3: Face redaction on page images
            print(f"\n👤 Running face detection on {len(image_paths)} page images...")
            face_results_all = []
            face_redacted_page_images = []
            total_faces = 0
            
            for i, img_path in enumerate(image_paths):
                face_results = self.face_redactor.redact_faces(img_path)
                face_results_all.append(face_results)
                total_faces += face_results['num_faces_detected']
                
                # Store the face-redacted image path
                face_redacted_page_images.append(face_results['redacted_image_path'])
                
                if face_results['num_faces_detected'] > 0:
                    print(f"   Page {i+1}: {face_results['num_faces_detected']} face(s) detected and redacted")
            
            # Step 4: Face redaction on embedded images
            embedded_face_results = []
            if embedded_images:
                print(f"\n👤 Running face detection on {len(embedded_images)} embedded images...")
                for i, img_path in enumerate(embedded_images):
                    face_results = self.face_redactor.redact_faces(img_path)
                    embedded_face_results.append(face_results)
                    total_faces += face_results['num_faces_detected']
                    if face_results['num_faces_detected'] > 0:
                        print(f"   Embedded image {i+1}: {face_results['num_faces_detected']} face(s) detected and redacted")
            
            # Step 5: Create final PDF from face-redacted page images (if faces were detected)
            final_pdf_path = output_pdf
            if total_faces > 0:
                print(f"\n📄 Creating final PDF with face redactions...")
                final_pdf_path = self._create_pdf_from_images(
                    face_redacted_page_images, 
                    output_pdf.replace('.pdf', '_fully_redacted.pdf')
                )
                if final_pdf_path:
                    print(f"✓ Final PDF with text + face redactions: {final_pdf_path}")
                    results['final_pdf_path'] = final_pdf_path
                else:
                    print("⚠ Warning: Could not create final PDF, text-only redacted PDF available")
            
            results['face_results'] = face_results_all
            results['embedded_face_results'] = embedded_face_results
            results['face_redacted_page_images'] = face_redacted_page_images
            results['processing_steps'].append('FACE_REDACTED')
            results['summary'] = {
                'status': 'PDF fully processed', 
                'num_pages': len(image_paths),
                'num_embedded_images': len(embedded_images),
                'total_faces_detected': total_faces,
                'page_face_redactions': [fr['num_faces_detected'] for fr in face_results_all],
                'embedded_face_redactions': [fr['num_faces_detected'] for fr in embedded_face_results] if embedded_face_results else [],
                'final_pdf': final_pdf_path
            }
            print(f"\n✓ PDF Processing Complete: {total_faces} total face(s) detected and redacted")
            print(f"✓ Final output: {final_pdf_path}")
            return results

        if ext not in self.config.IMAGE_CONFIG['supported_formats']:
            raise ValueError(f"Unsupported image format: {ext}")
        # Image: OCR, text redaction, face redaction
        ocr_results = self.ocr_processor.process_image(input_path)
        results['ocr_results'] = ocr_results['ocr_results']
        results['ocr_annotated_path'] = ocr_results['annotated_image_path']
        text_pii_results = self.text_redactor.redact_pii(ocr_results)
        results['text_pii_results'] = text_pii_results
        results['text_redacted_path'] = text_pii_results['redacted_image_path']
        face_results = self.face_redactor.redact_faces(text_pii_results['redacted_image_path'])
        results['face_results'] = face_results
        results['final_redacted_path'] = face_results['redacted_image_path']
        results['processing_steps'].append('FACE_REDACTED')
        results['summary'] = self._generate_summary(results)
        return results
    
    def process_text_only(self, image_path: str) -> Dict[str, Any]:
        """Process only text PII redaction (skip face detection)"""
        print(f"📄 Processing text PII only: {image_path}")
        
        # Step 1: OCR
        ocr_results = self.ocr_processor.process_image(image_path)
        
        # Step 2: Text PII
        text_pii_results = self.text_redactor.redact_pii(ocr_results)
        
        return {
            'input_image_path': image_path,
            'ocr_results': ocr_results,
            'text_pii_results': text_pii_results,
            'redacted_image_path': text_pii_results['redacted_image_path'],
            'processing_steps': ['OCR_COMPLETED', 'TEXT_PII_REDACTED']
        }
    
    def process_faces_only(self, image_path: str) -> Dict[str, Any]:
        """Process only face redaction (skip text processing)"""
        print(f"👤 Processing faces only: {image_path}")
        
        face_results = self.face_redactor.redact_faces(image_path)
        
        return {
            'input_image_path': image_path,
            'face_results': face_results,
            'redacted_image_path': face_results['redacted_image_path'],
            'processing_steps': ['FACE_REDACTED']
        }
    
    def _extract_pdf_images(self, pdf_path: str) -> list:
        """
        Extract embedded images from PDF for face detection.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of paths to extracted images
        """
        import fitz  # PyMuPDF
        from PIL import Image
        import io
        
        extracted_images = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    
                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Save extracted image
                    output_dir = os.path.dirname(pdf_path)
                    img_filename = f"extracted_page{page_num+1}_img{img_index+1}.{image_ext}"
                    img_path = os.path.join(output_dir, img_filename)
                    
                    # Convert to RGB if needed and save as JPG
                    try:
                        img = Image.open(io.BytesIO(image_bytes))
                        
                        # Skip very small images (likely icons/logos)
                        if img.width < 100 or img.height < 100:
                            continue
                        
                        # Convert to RGB if necessary
                        if img.mode in ('RGBA', 'LA', 'P'):
                            img = img.convert('RGB')
                        
                        # Save as JPG for consistency
                        img_path = img_path.rsplit('.', 1)[0] + '.jpg'
                        img.save(img_path, 'JPEG')
                        extracted_images.append(img_path)
                        
                    except Exception as e:
                        print(f"   Warning: Could not process image {img_filename}: {e}")
                        continue
            
            doc.close()
            
            if extracted_images:
                print(f"✓ Extracted {len(extracted_images)} embedded images from PDF")
            else:
                print("ℹ No embedded images found in PDF")
                
        except Exception as e:
            print(f"Warning: Could not extract images from PDF: {e}")
        
        return extracted_images
    
    def _create_pdf_from_images(self, image_paths: list, output_pdf_path: str) -> str:
        """
        Create a PDF from a list of images (for face-redacted pages).
        
        Args:
            image_paths: List of paths to face-redacted page images
            output_pdf_path: Path to save the final PDF
            
        Returns:
            Path to created PDF, or None if failed
        """
        try:
            from PIL import Image
            
            if not image_paths:
                print("⚠ No images to convert to PDF")
                return None
            
            # Open all images and convert to RGB
            images = []
            for img_path in image_paths:
                try:
                    img = Image.open(img_path)
                    # Convert to RGB if needed (PDF requires RGB)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"⚠ Warning: Could not open image {img_path}: {e}")
                    continue
            
            if not images:
                print("⚠ No valid images to create PDF")
                return None
            
            # Save first image as PDF, append rest
            images[0].save(
                output_pdf_path, 
                "PDF", 
                resolution=100.0, 
                save_all=True, 
                append_images=images[1:] if len(images) > 1 else []
            )
            
            # Close all images
            for img in images:
                img.close()
            
            print(f"✓ Created PDF with {len(images)} face-redacted pages")
            return output_pdf_path
            
        except Exception as e:
            print(f"✗ Error creating PDF from images: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate processing summary"""
        ocr_summary = self.ocr_processor.get_text_summary(results['ocr_results'])
        face_summary = self.face_redactor.get_face_summary(results['face_results'])
        
        return {
            'input_file': results.get('input_image_path') or results.get('input_path', 'Unknown'),
            'output_files': {
                'ocr_annotated': results['ocr_annotated_path'],
                'text_redacted': results['text_redacted_path'], 
                'final_redacted': results['final_redacted_path']
            },
            'ocr_stats': ocr_summary,
            'pii_stats': {
                'total_pii_detected': results['text_pii_results']['num_pii_detected'],
                'pii_types': list(set([item[1] for item in results['text_pii_results']['pii_detections']])) if results['text_pii_results']['pii_detections'] else []
            },
            'face_stats': face_summary,
            'processing_steps': results['processing_steps'],
            'pipeline_status': 'COMPLETED'
        }
    
    def get_supported_entities(self) -> list:
        """Get list of supported PII entity types"""
        return self.text_redactor.entities
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate all components are properly configured"""
        validation = {
            'ocr_processor': self.ocr_processor.reader is not None,
            'text_redactor': self.text_redactor.analyzer is not None,
            'face_redactor': self.face_redactor.mtcnn is not None
        }
        
        validation['pipeline_ready'] = all(validation.values())
        
        return validation
