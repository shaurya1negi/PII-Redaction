"""
Configuration module for PII Redaction
"""

class PIIConfig:
    
    OCR_CONFIG = {
        'languages': ['en'],
        'gpu': True,
        'parameters': {
            'paragraph': False,
            'width_ths': 0.001,
            'height_ths': 0.001,
            'text_threshold': 0.30,
            'low_text': 0.25,
            'link_threshold': 0.20,
            'canvas_size': 2000,
            'mag_ratio': 1.5,
            'slope_ths': 0.2,
            'ycenter_ths': 0.5,
            'add_margin': 0.04,
        }
    }
    
    NLP_CONFIG = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_trf"}],
    }
    
    PII_CONFIG = {
        'threshold': 0.1,
        'supported_entities': [
            'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'CREDIT_CARD',
            'IBAN_CODE', 'IP_ADDRESS', 'DATE_TIME', 'LOCATION',
            'IN_AADHAAR', 'IN_PAN', 'IN_VEHICLE_REGISTRATION', 'IN_PASSPORT'
        ],
        'excluded_entities': [
            'US_PASSPORT', 'UK_NHS', 'US_ITIN', 'US_DRIVER_LICENSE',
            'US_BANK_NUMBER', 'ORGANIZATION'
        ]
    }
    
    FACE_CONFIG = {
        'image_size': (740, 843),
        'select_largest': False,
        'keep_all': True,
        'device': 'cuda',
        'blur_radius': 15
    }
    
    IMAGE_CONFIG = {
        'blur_radius': 8,
        'dpi': 200,
        'supported_formats': ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    }
    
    # Output Configuration
    OUTPUT_CONFIG = {
        'ocr_annotated_suffix': '_ocr_annotated',
        'text_redacted_suffix': '_text_redacted', 
        'face_redacted_suffix': '_face_redacted',
        'final_redacted_suffix': '_fully_redacted'
    }
