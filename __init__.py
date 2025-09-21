"""
PII Redaction Module Suite

Components:
- OCRProcessor: Text extraction and annotation
- TextPIIRedactor: Text-based PII detection and redaction
- FaceRedactor: Face detection and redaction  
- PIIPipeline: Complete end-to-end processing

Usage:
    from PII_modules import PIIPipeline
    
    pipeline = PIIPipeline()
    results = pipeline.process_document("image.jpg")
"""

# Import configuration first
from .config import PIIConfig

# Import main modules
from .ocr_processor import OCRProcessor
from .text_pii_redactor import TextPIIRedactor
from .face_redactor import FaceRedactor
from .pipeline import PIIPipeline

__all__ = [
    'OCRProcessor',
    'TextPIIRedactor', 
    'FaceRedactor',
    'PIIPipeline',
    'PIIConfig'
]

__version__ = "1.0.0"

'''
Without __init__.py
# You would need to do this (messy):
from PII_modules.pipeline import PIIPipeline
from PII_modules.ocr_processor import OCRProcessor
from PII_modules.text_pii_redactor import TextPIIRedactor
from PII_modules.face_redactor import FaceRedactor

with __init__.py 
# Clean and simple:
from PII_modules import PIIPipeline, OCRProcessor, TextPIIRedactor, FaceRedactor
'''

"""
from PII_modules import PIIPipeline, OCRProcessor, TextPIIRedactor, FaceRedactor, PIIConfig
Python sees from PII_modules import ...
Finds the PII_modules folder
Executes __init__.py automatically
Runs all the imports in __init__.py
"""