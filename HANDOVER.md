# PIIshield - Handover Document

**To:** Availity Development Team  
**Date:** 16 October 2025  
**Project:** PIIshield - PII Redaction System  
**Status:** Functional POC  
**Repository:** [shaurya1negi/PIIshield](https://github.com/shaurya1negi/PIIshield)

---

## Executive Summary

PIIshield is a complete PII redaction solution that automatically detects and redacts sensitive information from PDFs and images. It combines OCR, NLP transformers, and face detection to provide comprehensive data protection.

### What It Does

- **Detects & Redacts Text PII**: Names, emails, phone numbers, SSN, credit cards, Indian documents (Aadhaar, PAN)
- **Detects & Redacts Faces**: Automatic face detection with blur
- **Supports PDF & Images**: Maintains PDF layout while redacting
- **Contextual Analysis**: Uses transformer models to reduce false positives

---

## Quick Start

### Setup (5 minutes)

```bash
# 1. Navigate to project
cd /home/your_username/Availty/PIIshield

# 2. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download NLP model (automatic on first run, or manual)
python -m spacy download en_core_web_trf

# 5. Verify
python -c "from PIIshield import PIIPipeline; print('✓ Ready')"
```

### Run

```bash
python3 redact_file_cli.py
# Enter file path when prompted
```

### Programmatic Usage

```python
from PIIshield import PIIPipeline

pipeline = PIIPipeline()
results = pipeline.process_document("document.pdf")
print(f"Output: {results.get('pdf_redacted_path')}")
```

---

## System Architecture

```
Input File
    ↓
PDF → text_processor.py → PyMuPDF + Presidio → Text Redacted PDF
    ↓
pdf2image → Page Images → face_redactor.py → MTCNN → Fully Redacted
    
Image → ocr_processor.py → EasyOCR → Text Extraction
    ↓
text_pii_redactor.py → Presidio + Transformer → Text Redacted
    ↓
face_redactor.py → MTCNN → Fully Redacted
```

---

## Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| `pipeline.py` | Orchestrator | Main entry point, coordinates all processing |
| `ocr_processor.py` | EasyOCR | Extract text from images |
| `text_pii_redactor.py` | Presidio + spaCy | Detect & redact text PII |
| `face_redactor.py` | MTCNN | Detect & redact faces |
| `text_processor.py` | PyMuPDF + Presidio | PDF contextual redaction |
| `config.py` | Config | Centralized configuration |
| `redact_file_cli.py` | CLI | Command-line interface |

---

## Dependencies

### Core Technologies

- **Python**: 3.12.3 (recommended)
- **EasyOCR**: Text extraction (GPU-accelerated)
- **Microsoft Presidio**: PII detection with contextual analysis
- **spaCy + en_core_web_trf**: Transformer-based NLP model
- **MTCNN (facenet-pytorch)**: Face detection
- **PyMuPDF**: PDF processing with layout preservation
- **pdf2image + Poppler**: PDF to image conversion

### System Requirements

- **Poppler**: Required for PDF processing
  ```bash
  # Ubuntu/Debian
  sudo apt-get install poppler-utils
  ```
- **CUDA** (optional): For GPU acceleration
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~3GB for models and dependencies

### No Database Required
All processing is in-memory with file-based outputs.

---

## Configuration

All settings in `config.py`:

```python
# Adjust PII detection sensitivity
PII_CONFIG = {
    'threshold': 0.1,  # Lower = more sensitive (more detections)
}

# GPU vs CPU
FACE_CONFIG = {'device': 'cuda'}  # or 'cpu'
OCR_CONFIG = {'gpu': True}         # or False

# Blur intensity
IMAGE_CONFIG = {'blur_radius': 8}
FACE_CONFIG = {'blur_radius': 15}
```

---

## API Reference

### Main Pipeline

```python
from PIIshield import PIIPipeline

pipeline = PIIPipeline()

# Process any file (PDF or image)
results = pipeline.process_document("file.pdf")

# Process only text PII
results = pipeline.process_text_only("image.jpg")

# Process only faces
results = pipeline.process_faces_only("image.jpg")

# Get supported PII types
entities = pipeline.get_supported_entities()

# Validate setup
status = pipeline.validate_configuration()
```

### Supported PII Entities

- **General**: PERSON, LOCATION, DATE_TIME, EMAIL_ADDRESS, PHONE_NUMBER, IP_ADDRESS
- **Financial**: CREDIT_CARD, IBAN_CODE
- **Indian Documents**: IN_AADHAAR, IN_PAN, IN_VEHICLE_REGISTRATION, IN_PASSPORT

---

## Output Files

### For Images
- `*_ocr_annotated.jpg` - OCR bounding boxes (debug)
- `*_text_redacted.jpg` - Text PII blurred
- `*_fully_redacted.jpg` - **Final output** (text + faces redacted)

### For PDFs
- `redacted_*.pdf` - **Final PDF** with text redacted
- `redacted_page_*.jpg` - Individual pages as images
- `redacted_page_*_fully_redacted.jpg` - Pages with faces redacted

---

## Known Issues & Solutions

### 1. GPU Memory Error
```python
# In config.py
FACE_CONFIG = {'device': 'cpu'}
OCR_CONFIG = {'gpu': False}
```

### 2. spaCy Model Not Found
```bash
python -m spacy download en_core_web_trf
```

### 3. PDF Conversion Failed
```bash
# Install Poppler
sudo apt-get install poppler-utils  # Ubuntu
brew install poppler                 # macOS
```

### 4. False Positives (common words detected as PII)
```python
# In config.py, increase threshold
PII_CONFIG = {'threshold': 0.3}  # Default: 0.1
```

### 5. Module Not Found
```bash
# Ensure virtual environment is activated
source venv/bin/activate
```

---

## Testing

### Validate Installation

```python
from PIIshield import PIIPipeline

pipeline = PIIPipeline()
status = pipeline.validate_configuration()

# Should return:
# {'ocr_processor': True, 'text_redactor': True, 
#  'face_redactor': True, 'pipeline_ready': True}
```

### Test Individual Components

```python
# Test OCR
from PIIshield import OCRProcessor
ocr = OCRProcessor()
results = ocr.process_image("test.jpg")
print(f"Detected {len(results['ocr_results'])} text regions")

# Test Face Detection
from PIIshield import FaceRedactor
face = FaceRedactor()
results = face.detect_faces_only("test.jpg")
print(f"Detected {results['num_faces_detected']} faces")

# Test PII Detection
from PIIshield import TextPIIRedactor
redactor = TextPIIRedactor()
print(f"Entities: {redactor.entities}")
```

---

## Performance

| Environment | Speed per Image | Speed per PDF Page |
|-------------|----------------|-------------------|
| GPU (CUDA) | 2-5 seconds | 3-7 seconds |
| CPU only | 10-30 seconds | 15-45 seconds |

**Note**: First run downloads models (~2GB), subsequent runs are faster.

---

## Security Considerations

⚠️ **Important**:
- Intermediate files contain partially redacted data - delete after processing
- Store input/output files in secure, access-controlled locations
- PIIshield does not log or persist any PII data
- Models cached in `~/.cache/huggingface/` and `~/.EasyOCR/`

---

## Production Recommendations

1. **Use GPU**: 5-10x faster processing
2. **Delete Intermediate Files**: Only keep final outputs
3. **Adjust Threshold**: Balance sensitivity vs false positives
4. **Batch Processing**: Process multiple files in loop
5. **Error Handling**: Wrap in try-except for production
6. **Monitoring**: Log processing times and success rates

### Example Production Script

```python
from PIIshield import PIIPipeline
import glob
import logging

logging.basicConfig(level=logging.INFO)
pipeline = PIIPipeline()

for file_path in glob.glob("/input/*.pdf"):
    try:
        results = pipeline.process_document(file_path)
        logging.info(f"✓ {file_path} → {results['pdf_redacted_path']}")
    except Exception as e:
        logging.error(f"✗ {file_path}: {e}")
```

---

## Troubleshooting

### Check CUDA Availability
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Dependencies
```bash
pip list | grep -E "easyocr|presidio|facenet|spacy|PyMuPDF"
```

---

## Future Enhancements

**Ready for Implementation**:
- [ ] RESTful API wrapper (Flask/FastAPI)
- [ ] Docker containerization
- [ ] Batch processing API endpoint
- [ ] Confidence score per entity type
- [ ] Additional language support
- [ ] Custom entity training
- [ ] Audit logging
- [ ] Web UI for file upload

---

## Support & Documentation

- **Full Documentation**: See `README.md`
- **Repository**: https://github.com/shaurya1negi/PIIshield
- **Issues**: Create GitHub issue for bugs/questions
- **Configuration**: All settings in `config.py`

---

## Contact & Handover

**Original Developer**: shaurya1negi  
**Handover Date**: 16 October 2025  
**Status**: ✅ Production-Ready POC  
**Testing**: ✅ All components validated  
**Documentation**: ✅ Complete

### Handover Checklist

- [x] Source code committed to repository
- [x] All dependencies documented in `requirements.txt`
- [x] Configuration centralized in `config.py`
- [x] CLI interface functional (`redact_file_cli.py`)
- [x] API endpoints documented
- [x] Known issues documented with solutions
- [x] Installation guide tested
- [x] Example usage provided
- [x] Security considerations documented

---

## Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate

# Run CLI
python3 redact_file_cli.py

# Run programmatically
python3 -c "from PIIshield import PIIPipeline; \
            p = PIIPipeline(); \
            p.process_document('input.pdf')"

# Validate setup
python3 -c "from PIIshield import PIIPipeline; \
            print(PIIPipeline().validate_configuration())"

# Check GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

For detailed information, refer to `README.md`.
