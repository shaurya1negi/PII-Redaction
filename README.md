# PIIshield - PII Redaction System

## Overview

PIIshield is a comprehensive PII (Personally Identifiable Information) redaction system that automatically detects and redacts sensitive information from documents and images. It supports both PDF and image formats, using advanced OCR, NLP transformers, and face detection technologies.

**Repository:** [shaurya1negi/PIIshield](https://github.com/shaurya1negi/PIIshield)  
**Version:** 1.0.0  
**Date:** October 2025

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Execution Flow](#execution-flow)
- [Output Files](#output-files)
- [Known Issues](#known-issues)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## Features

- **Multi-format Support**: PDF and image formats (.jpg, .jpeg, .png, .tiff, .bmp)
- **Text PII Detection**: Uses Microsoft Presidio with transformer models (en_core_web_trf)
- **Face Detection & Redaction**: MTCNN-based face detection with Gaussian blur
- **OCR Processing**: EasyOCR with GPU acceleration support
- **Contextual Analysis**: Advanced contextual validation for PII detection
- **Indian Document Support**: Aadhaar, PAN, Vehicle Registration, Passport
- **Layout Preservation**: Maintains document layout during PDF redaction

### Supported PII Entities

- **Personal Data**: PERSON, LOCATION, DATE_TIME
- **Contact Info**: EMAIL_ADDRESS, PHONE_NUMBER, IP_ADDRESS
- **Financial**: CREDIT_CARD, IBAN_CODE
- **Indian Documents**: IN_AADHAAR, IN_PAN, IN_VEHICLE_REGISTRATION, IN_PASSPORT

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   PIIshield Pipeline                 │
├─────────────────────────────────────────────────────┤
│                                                       │
│  Input (PDF/Image)                                   │
│         │                                            │
│         ├──► PDF? ──Yes──► text_processor.py        │
│         │              │                             │
│         │              └──► PyMuPDF + Presidio      │
│         │                   (Contextual Redaction)  │
│         │                                            │
│         └──► Image? ──Yes──► OCR Pipeline           │
│                        │                             │
│                        ├──► ocr_processor.py         │
│                        │    (EasyOCR)                │
│                        │                             │
│                        ├──► text_pii_redactor.py    │
│                        │    (Presidio + Transformer) │
│                        │                             │
│                        └──► face_redactor.py         │
│                             (MTCNN)                  │
│                                                       │
│  Output (Redacted Files)                             │
└─────────────────────────────────────────────────────┘
```

---

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python Version**: 3.12.3 (recommended)
- **pip Version**: 24.3.1 or higher
- **RAM**: Minimum 8GB (16GB recommended for GPU processing)
- **GPU**: CUDA-compatible GPU (optional, for acceleration)

### Software Dependencies

- **CUDA Toolkit** (optional): For GPU acceleration
- **Poppler** (for PDF processing): Required for pdf2image

#### Install Poppler:

**Ubuntu/Debian**:
```bash
sudo apt-get install poppler-utils
```

**macOS**:
```bash
brew install poppler
```

**Windows**:
Download from: http://blog.alivate.com.au/poppler-windows/

---

## Installation

### Step 1: Clone the Repository

```bash
cd /home/your_username/workspace
git clone https://github.com/shaurya1negi/PIIshield.git
cd PIIshield
```

### Step 2: Create Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Important**: Dependencies are installed in the following order:
1. facenet-pytorch (with CUDA support if available)
2. easyocr & opencv-python
3. presidio-analyzer with transformers
4. spaCy → spacy-transformers (in this order, as spacy-transformers is a plugin)
5. pdf2image, PyMuPDF, pandas, unidecode, accelerate

### Step 5: Download spaCy Transformer Model

The transformer model will be automatically downloaded on first run, or you can manually install it:

```bash
python -m spacy download en_core_web_trf
```

### Step 6: Verify Installation

```bash
python -c "from PIIshield import PIIPipeline; p = PIIPipeline(); print(p.validate_configuration())"
```

Expected output:
```python
{
    'ocr_processor': True,
    'text_redactor': True,
    'face_redactor': True,
    'pipeline_ready': True
}
```

---

## Configuration

Configuration is centralized in `config.py`. Modify settings as needed:

### OCR Configuration

```python
OCR_CONFIG = {
    'languages': ['en'],      # Language support
    'gpu': True,              # GPU acceleration
    'parameters': {           # EasyOCR parameters
        'text_threshold': 0.30,
        'low_text': 0.25,
        'link_threshold': 0.20,
        'canvas_size': 2000,
        'mag_ratio': 1.5,
        # ... more parameters
    }
}
```

### NLP Configuration

```python
NLP_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_trf"}]
}
```

### PII Detection Configuration

```python
PII_CONFIG = {
    'threshold': 0.1,         # Confidence threshold (0.0-1.0)
    'supported_entities': [   # PII types to detect
        'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER',
        'CREDIT_CARD', 'IBAN_CODE', 'IP_ADDRESS', 'DATE_TIME', 'LOCATION',
        'IN_AADHAAR', 'IN_PAN', 'IN_VEHICLE_REGISTRATION', 'IN_PASSPORT'
    ],
    'excluded_entities': [
        'US_PASSPORT', 'UK_NHS', 'US_ITIN', 'US_DRIVER_LICENSE',
        'US_BANK_NUMBER', 'ORGANIZATION'
    ]
}
```

### Face Detection Configuration

```python
FACE_CONFIG = {
    'image_size': (740, 843),
    'select_largest': False,
    'keep_all': True,
    'device': 'cuda',         # 'cuda' or 'cpu'
    'blur_radius': 15
}
```

### Image Processing Configuration

```python
IMAGE_CONFIG = {
    'blur_radius': 8,
    'dpi': 200,
    'supported_formats': ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
}
```

---

## Usage

### Command Line Interface

#### Basic Usage

```bash
python3 redact_file_cli.py
```

When prompted, enter the file path:
```
Enter the path to the file (PDF or image): /path/to/your/document.pdf
```

#### Direct Invocation

```bash
# For PDFs
echo "/path/to/document.pdf" | python3 redact_file_cli.py

# For Images
echo "/path/to/image.jpg" | python3 redact_file_cli.py
```

### Programmatic Usage

```python
from PIIshield import PIIPipeline

# Initialize pipeline
pipeline = PIIPipeline()

# Process document (PDF or image)
results = pipeline.process_document("input_document.pdf")

# Access results
print(f"Redacted file: {results.get('final_redacted_path') or results.get('pdf_redacted_path')}")
print(f"Summary: {results['summary']}")
```

### Advanced Usage Examples

#### Process Only Text PII (Skip Face Detection)

```python
from PIIshield import PIIPipeline

pipeline = PIIPipeline()
results = pipeline.process_text_only("image.jpg")
print(f"Text redacted: {results['redacted_image_path']}")
```

#### Process Only Faces (Skip Text)

```python
from PIIshield import PIIPipeline

pipeline = PIIPipeline()
results = pipeline.process_faces_only("image.jpg")
print(f"Faces redacted: {results['redacted_image_path']}")
```

#### Custom Configuration

```python
from PIIshield import PIIPipeline, PIIConfig

# Customize configuration
config = PIIConfig()
config.PII_CONFIG['threshold'] = 0.3  # Higher threshold
config.FACE_CONFIG['device'] = 'cpu'  # Force CPU

pipeline = PIIPipeline(config=config)
results = pipeline.process_document("image.jpg")
```

---

## API Endpoints

### `PIIPipeline` (pipeline.py)

Main orchestrator for complete PII redaction workflow.

#### `process_document(input_path: str, save_intermediate: bool = True) -> Dict`

**Description**: Complete end-to-end processing for PDF or image.

**Parameters**:
- `input_path` (str): Path to input file (PDF or image)
- `save_intermediate` (bool): Save intermediate outputs (default: True)

**Returns**:
```python
{
    'input_path': str,
    'processing_steps': List[str],
    'ocr_results': List,              # For images only
    'text_pii_results': Dict,         # For images only
    'face_results': Dict,
    'pdf_redacted_path': str,         # For PDFs only
    'final_redacted_path': str,       # For images only
    'summary': Dict
}
```

#### `process_text_only(image_path: str) -> Dict`

**Description**: Process only text PII redaction.

**Returns**:
```python
{
    'ocr_results': Dict,
    'text_pii_results': Dict,
    'redacted_image_path': str
}
```

#### `process_faces_only(image_path: str) -> Dict`

**Description**: Process only face redaction.

**Returns**:
```python
{
    'face_results': Dict,
    'redacted_image_path': str
}
```

#### `get_supported_entities() -> List[str]`

**Description**: Get list of supported PII entity types.

#### `validate_configuration() -> Dict[str, bool]`

**Description**: Validate all components are properly initialized.

---

### `OCRProcessor` (ocr_processor.py)

Text extraction and annotation using EasyOCR.

#### `process_image(image_path: str) -> Dict`

**Description**: Extract text from image with bounding boxes.

**Returns**:
```python
{
    'ocr_results': List[Tuple],    # [(index, bbox, text, confidence), ...]
    'ocr_dict': Dict,
    'annotated_image_path': str,
    'original_image': PIL.Image,
    'image_path': str
}
```

#### `get_text_summary(ocr_results: List) -> Dict`

**Description**: Get OCR statistics including total detections and confidence scores.

---

### `TextPIIRedactor` (text_pii_redactor.py)

PII detection and redaction using Presidio with transformer models.

#### `redact_pii(ocr_data: Dict) -> Dict`

**Description**: Detect and redact PII from OCR results.

**Returns**:
```python
{
    'pii_detections': List[List],    # [index, entity_type, text, score]
    'pii_indices': np.ndarray,
    'pii_bounding_boxes': List,
    'redacted_image_path': str,
    'num_pii_detected': int
}
```

#### `analyze_single_text(text: str) -> List[Dict]`

**Description**: Analyze single text string for PII.

---

### `FaceRedactor` (face_redactor.py)

Face detection and redaction using MTCNN.

#### `redact_faces(image_path: str) -> Dict`

**Description**: Detect and redact faces.

**Returns**:
```python
{
    'face_boxes': List,
    'face_probabilities': List,
    'face_landmarks': List,
    'redacted_image_path': str,
    'num_faces_detected': int
}
```

#### `detect_faces_only(image_path: str) -> Dict`

**Description**: Detect faces without redaction.

#### `get_face_summary(face_data: Dict) -> Dict`

**Description**: Get summary statistics of face detection results.

---

### `text_processor` (text_processor.py)

Contextual PDF text redaction using PyMuPDF and Presidio.

#### `redact_pdf_preserve_layout(input_path: str, output_path: str) -> bool`

**Description**: Redact PII from PDF while preserving layout.

**Returns**: `True` if successful, `False` otherwise.

#### `detect_pii_contextual(text: str) -> List[str]`

**Description**: Detect PII with advanced contextual validation. Uses Presidio with confidence threshold of 0.6 and includes regex pattern matching for critical PII.

---

## Execution Flow

### PDF Processing Flow

```
1. Input PDF → text_processor.py
   ↓
2. For each page:
   a. Extract text (PyMuPDF)
   b. Detect PII contextually (Presidio + Transformer)
   c. Apply redaction annotations (black boxes)
   d. Save redacted page
   ↓
3. Convert PDF to images (pdf2image @ 200 DPI)
   ↓
4. For each page image:
   a. Detect faces (MTCNN)
   b. Apply face redaction (Gaussian blur)
   ↓
5. Output:
   - redacted_*.pdf (text redacted)
   - redacted_page_*.jpg (page images)
   - redacted_page_*_fully_redacted.jpg (with faces redacted)
```

### Image Processing Flow

```
1. Input Image → ocr_processor.py
   ↓
2. OCR Text Extraction (EasyOCR)
   - Detects text with bounding boxes
   - Confidence scoring
   - Output: *_ocr_annotated.jpg
   ↓
3. text_pii_redactor.py
   - Presidio analysis with contextual validation
   - 5-word context window (2 before, 2 after)
   - Threshold: 0.1 confidence score
   - Blur PII regions (Gaussian blur radius: 8)
   - Output: *_text_redacted.jpg
   ↓
4. face_redactor.py
   - MTCNN face detection
   - Detects all faces (keep_all: True)
   - Gaussian blur faces (radius: 15)
   - Output: *_fully_redacted.jpg
```

---

## Output Files

### For Images

| File | Description |
|------|-------------|
| `*_ocr_annotated.jpg` | Original image with OCR bounding boxes (red polygons) |
| `*_text_redacted.jpg` | Image with PII text regions blurred |
| `*_fully_redacted.jpg` | Final output with text + faces redacted |

### For PDFs

| File | Description |
|------|-------------|
| `redacted_*.pdf` | PDF with text PII redacted (black boxes preserving layout) |
| `redacted_page_*.jpg` | Individual pages converted to images (200 DPI) |
| `redacted_page_*_fully_redacted.jpg` | Pages with faces also redacted |

---

## Known Issues

### 1. **GPU Memory Issues**

**Symptom**: `CUDA out of memory` error during face detection or OCR processing

**Solutions**:
```python
# Option 1: Force CPU in config.py
FACE_CONFIG = {'device': 'cpu'}
OCR_CONFIG = {'gpu': False}

# Option 2: Reduce image size
IMAGE_CONFIG = {'dpi': 150}  # Default: 200
```

### 2. **spaCy Model Download Failures**

**Symptom**: `Can't find model 'en_core_web_trf'` or transformer model not loading

**Solution**:
```bash
# Manual download
python -m spacy download en_core_web_trf

# Verify installation
python -c "import spacy; nlp = spacy.load('en_core_web_trf'); print('OK')"

# If still failing, reinstall spacy-transformers
pip uninstall spacy-transformers
pip install spacy-transformers
```

### 3. **PDF Conversion Failures**

**Symptom**: `pdf2image conversion failed` or `Unable to locate Poppler`

**Solution**:
```bash
# Ensure Poppler is installed
which pdftoppm  # Should return path

# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Windows - Add poppler bin to PATH
```

### 4. **EasyOCR Language Packs**

**Symptom**: First run downloads large language models (~500MB for English)

**Solution**: Language models are downloaded automatically on first run. Ensure:
- Stable internet connection
- Sufficient disk space (~2GB for models)
- Models are cached in `~/.EasyOCR/`

### 5. **False Positives in PII Detection**

**Symptom**: Common words flagged as PII (e.g., "May", "Will", "John")

**Solution**:
```python
# Adjust threshold in config.py
PII_CONFIG = {
    'threshold': 0.3,  # Increase from 0.1 for fewer false positives
}
```

**Note**: The system includes contextual validation with common word filtering. Words like "the", "and", "for", etc., are automatically excluded.

### 6. **Face Detection Misses**

**Symptom**: Some faces not detected, especially at angles or with occlusion

**Solution**:
```python
# Adjust MTCNN parameters in config.py
FACE_CONFIG = {
    'image_size': (1024, 1024),  # Larger size for better detection
    'select_largest': False,
    'keep_all': True,
}
```

**Limitations**: MTCNN may miss:
- Severely occluded faces
- Side profiles (>60° angle)
- Very small faces (<20x20 pixels)

### 7. **Import Errors**

**Symptom**: `ModuleNotFoundError: No module named 'PIIshield'`

**Solution**:
```bash
# Ensure you're in the parent directory
cd /home/your_username/workspace
python3 PIIshield/redact_file_cli.py

# Or activate virtual environment
cd PIIshield
source venv/bin/activate
python3 redact_file_cli.py
```

### 8. **NumPy Version Conflicts**

**Symptom**: `AttributeError: module 'numpy' has no attribute 'float'` or similar

**Solution**:
```bash
# Install specific numpy version
pip install "numpy<2.0.0,>=1.24.0"
```

This is specified in `requirements.txt` to mitigate dependency issues with facenet-pytorch.

### 9. **PDF Layout Corruption**

**Symptom**: Redacted PDF has misaligned text or broken formatting

**Solution**: 
- The system uses PyMuPDF for layout preservation
- If issues persist, non-standard PDFs are automatically routed through OCR pipeline
- Ensure PyMuPDF is up to date: `pip install --upgrade PyMuPDF`

---

## Troubleshooting

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from PIIshield import PIIPipeline
pipeline = PIIPipeline()
```

### Validate Installation

```python
from PIIshield import PIIPipeline

pipeline = PIIPipeline()
status = pipeline.validate_configuration()

if not status['pipeline_ready']:
    print("Issues found:")
    for component, is_ready in status.items():
        if not is_ready:
            print(f"  - {component}: NOT READY")
```

### Check Dependencies

```bash
pip list | grep -E "easyocr|presidio|facenet|spacy|torch|PyMuPDF|pdf2image"
```

### Test Individual Components

```python
# Test OCR
from PIIshield import OCRProcessor
ocr = OCRProcessor()
results = ocr.process_image("test.jpg")
print(f"Detected {len(results['ocr_results'])} text regions")

# Test PII Detection
from PIIshield import TextPIIRedactor
redactor = TextPIIRedactor()
print(f"Supported entities: {redactor.entities}")

# Test Face Detection
from PIIshield import FaceRedactor
face_redactor = FaceRedactor()
results = face_redactor.detect_faces_only("test.jpg")
print(f"Detected {results['num_faces_detected']} faces")
```

### Check CUDA Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
```

---

## Project Structure

```
PIIshield/
├── __init__.py                 # Package initialization and exports
├── config.py                   # Centralized configuration
├── pipeline.py                 # Main orchestrator (PIIPipeline)
├── ocr_processor.py            # OCR text extraction (EasyOCR)
├── text_pii_redactor.py        # PII detection & redaction (Presidio)
├── face_redactor.py            # Face detection & redaction (MTCNN)
├── text_processor.py           # PDF contextual redaction (PyMuPDF)
├── redact_file_cli.py          # CLI entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── Readme                      # Quick start guide
└── __pycache__/                # Python cache (auto-generated)
```

### Component Dependencies

```
PIIPipeline (Main Orchestrator)
    ├── OCRProcessor (EasyOCR)
    ├── TextPIIRedactor (Presidio + spaCy Transformer)
    ├── FaceRedactor (MTCNN + facenet-pytorch)
    └── text_processor (PyMuPDF + Presidio)

External Dependencies:
    ├── easyocr (OCR engine)
    ├── presidio-analyzer (PII detection)
    ├── spacy + en_core_web_trf (NLP transformer model)
    ├── facenet-pytorch (Face detection)
    ├── PyMuPDF (PDF processing)
    ├── pdf2image + Poppler (PDF to image conversion)
    └── PIL, numpy, pandas (Image and data processing)
```

### No Database Required

PIIshield is a **stateless processing pipeline** with no database dependencies. All processing is done in-memory with file-based outputs.

---

## Performance Optimization

### GPU Acceleration

```python
# Verify CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Check which components use GPU
from PIIshield import PIIPipeline, PIIConfig
config = PIIConfig()
print(f"OCR GPU: {config.OCR_CONFIG['gpu']}")
print(f"Face Detection Device: {config.FACE_CONFIG['device']}")
```

**Expected Performance**:
- **With GPU**: 2-5 seconds per image (depending on resolution)
- **CPU only**: 10-30 seconds per image

### Batch Processing

```python
from PIIshield import PIIPipeline
import glob

pipeline = PIIPipeline()

# Process multiple files
for file_path in glob.glob("/path/to/images/*.jpg"):
    try:
        results = pipeline.process_document(file_path)
        print(f"✓ Processed: {file_path}")
    except Exception as e:
        print(f"✗ Failed: {file_path} - {e}")
```

### Memory Management

For large PDFs or high-resolution images:

```python
# Reduce DPI for PDF conversion
config.IMAGE_CONFIG['dpi'] = 150  # Default: 200

# Process pages individually instead of all at once
# (already implemented in pipeline.py)
```

---

## Security Considerations

1. **Sensitive Data Handling**: Input files contain PII - ensure secure file transfer and storage
2. **Intermediate Files**: Temporary files (*_ocr_annotated.jpg, *_text_redacted.jpg) contain partially redacted data
3. **Model Files**: Transformer models cached in `~/.cache/huggingface/` and `~/.EasyOCR/`
4. **Output Storage**: Store redacted files in secure, access-controlled locations
5. **No Data Persistence**: PIIshield does not store or log any PII data

### Recommendations for Production

- Delete intermediate files after processing
- Use encrypted storage for input/output files
- Run in isolated environments (containers/VMs)
- Implement access logging for audit trails
- Regular security updates for all dependencies

---

## Developer Setup at Availity Workspace

### Quick Setup for Availity Developers

```bash
# 1. Navigate to your workspace
cd /home/your_username/Availty

# 2. Clone the repository
git clone https://github.com/shaurya1negi/PIIshield.git
cd PIIshield

# 3. Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Download spaCy model
python -m spacy download en_core_web_trf

# 6. Verify installation
python -c "from PIIshield import PIIPipeline; print('Installation successful')"

# 7. Run CLI
python3 redact_file_cli.py
```

### Dependency Checklist

- [x] Python 3.12.3
- [x] pip 24.3.1+
- [x] Poppler (for PDF processing)
- [x] CUDA Toolkit (optional, for GPU)
- [x] Virtual environment activated
- [x] All packages from requirements.txt
- [x] en_core_web_trf model downloaded

---

## Handover Notes

### POC Status

**Complete and Functional**

- All modules tested and working
- PDF and image processing verified
- Face detection and text PII redaction operational
- Contextual validation implemented
- Indian document recognition (Aadhaar, PAN, etc.)

### Key Technical Decisions

1. **Presidio with Transformers**: Chosen for high accuracy in PII detection with contextual understanding
2. **MTCNN for Face Detection**: Reliable multi-face detection with landmark support
3. **EasyOCR**: Multi-language support with GPU acceleration
4. **PyMuPDF for PDFs**: Layout preservation while redacting
5. **Modular Design**: Independent components for flexibility

### Future Enhancements

- [ ] Support for additional languages
- [ ] Batch API for multiple files
- [ ] RESTful API wrapper
- [ ] Confidence score tuning per entity type
- [ ] Custom entity training
- [ ] Redaction audit logs
- [ ] Docker containerization

---

## Support & Contact

For issues or questions:
1. Check logs for detailed error messages
2. Validate configuration with `validate_configuration()`
3. Contact: [Repository Issues](https://github.com/shaurya1negi/PIIshield/issues)

---

## Version History

- **v1.0.0** (October 2025): Initial release
  - PDF and image support
  - Text PII redaction with Presidio
  - Face redaction with MTCNN
  - Indian document recognition
  - Contextual validation

---

## Acknowledgments

- **Microsoft Presidio**: PII detection framework
- **EasyOCR**: OCR engine
- **facenet-pytorch**: Face detection
- **spaCy**: NLP processing
- **PyMuPDF**: PDF manipulation

---

**End of README**
