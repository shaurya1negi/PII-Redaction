import sys

sys.path.append('/home/shaurya/Desktop/myenvPython3.12/PII_redaction')
from PII_modules import PIIPipeline, OCRProcessor, TextPIIRedactor, FaceRedactor, PIIConfig

def example_complete_pipeline():

    pipeline = PIIPipeline()
    image_path = "/home/shaurya/Desktop/myenvPython3.12/PII_redaction/facenet_env/sample_documents/images.jpeg"
    
    results = pipeline.process_document(image_path)
    
    print("\\n PROCESSING SUMMARY:")
    print(f"   Input: {results['summary']['input_file']}")
    print(f"   OCR detections: {results['summary']['ocr_stats']['total_detections']}")
    print(f"   PII entities found: {results['summary']['pii_stats']['total_pii_detected']}")
    print(f"   Faces detected: {results['summary']['face_stats']['total_faces']}")
    print(f"   Final output: {results['final_redacted_path']}")
    
    return results

def example_individual_components():
    print("\\nINDIVIDUAL COMPONENT USAGE")
    
    image_path = "/home/shaurya/Desktop/myenvPython3.12/PII_redaction/abd3f33f-ea9b-467c-918c-b5248fc42e7d.jpeg"
    
    ocr_processor = OCRProcessor()
    ocr_results = ocr_processor.process_image(image_path)
    print(f"   Text regions detected: {len(ocr_results['ocr_results'])}")
    print(f"   Annotated image: {ocr_results['annotated_image_path']}")
    
    text_redactor = TextPIIRedactor()
    pii_results = text_redactor.redact_pii(ocr_results)
    print(f"   PII entities detected: {pii_results['num_pii_detected']}")
    print(f"   Text redacted image: {pii_results['redacted_image_path']}")
    
    face_redactor = FaceRedactor()
    face_results = face_redactor.redact_faces(pii_results['redacted_image_path'])
    print(f"   Faces detected: {face_results['num_faces_detected']}")
    print(f"   Final redacted image: {face_results['redacted_image_path']}")

def example_website_integration():
    print("\\nWEBSITE/APPLICATION INTEGRATION")
    
    print("""
    from PII_modules import PIIPipeline
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    pii_pipeline = PIIPipeline()
    
    @app.route('/redact_document', methods=['POST'])
    def redact_document():
        file = request.files['document']
        file_path = f"uploads/{file.filename}"
        file.save(file_path)
        results = pii_pipeline.process_document(file_path)
        return jsonify({'status': 'success', 'results': results})
    """)

def example_configuration_customization():
    print("\\nCUSTOM CONFIGURATION")
    
    custom_config = PIIConfig()
    custom_config.OCR_CONFIG['parameters']['canvas_size'] = 2000
    custom_config.PII_CONFIG['threshold'] = 0.15
    custom_config.FACE_CONFIG['blur_radius'] = 20
    
    pipeline = PIIPipeline(config=custom_config)
    print("   ✅ Custom configuration applied")

if __name__ == "__main__":
    example_complete_pipeline()
