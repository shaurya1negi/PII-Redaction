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
            # PDF: text redaction, then face redaction on page images
            from .text_processor import redact_pdf_preserve_layout
            output_pdf = os.path.join(os.path.dirname(input_path), 'redacted_' + os.path.basename(input_path))
            pdf_redact_success = redact_pdf_preserve_layout(input_path, output_pdf)
            results['pdf_redacted_path'] = output_pdf
            results['processing_steps'].append('PDF_TEXT_REDACTED')
            if not pdf_redact_success:
                results['summary'] = {'status': 'PDF redaction failed'}
                return results
            try:
                from pdf2image import convert_from_path
                page_images = convert_from_path(output_pdf, dpi=self.config.IMAGE_CONFIG.get('dpi', 200))
                image_paths = []
                for i, img in enumerate(page_images):
                    img_path = os.path.join(os.path.dirname(output_pdf), f'redacted_page_{i+1}.jpg')
                    img.save(img_path)
                    image_paths.append(img_path)
                results['pdf_page_images'] = image_paths
            except Exception as e:
                results['summary'] = {'status': f'PDF to image conversion failed: {e}'}
                return results
            face_results_all = []
            for img_path in image_paths:
                face_results = self.face_redactor.redact_faces(img_path)
                face_results_all.append(face_results)
            results['face_results'] = face_results_all
            results['processing_steps'].append('FACE_REDACTED')
            results['summary'] = {'status': 'PDF processed', 'num_pages': len(image_paths), 'face_redactions': [fr['num_faces_detected'] for fr in face_results_all]}
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
