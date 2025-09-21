"""
Text PII Redactor Module
======================

Handles PII detection and redaction using Presidio with transformer spaCy model
exactly as configured in your original solution.

Endpoint: redact_pii() -> Returns PII detections and redacted image
"""

import re
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Dict, Tuple, Any
from unidecode import unidecode
import os

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    InAadhaarRecognizer, InPanRecognizer, 
    InVehicleRegistrationRecognizer, InPassportRecognizer
)

from .config import PIIConfig

class TextPIIRedactor:
    """
    Text PII Redactor using Presidio with transformer spaCy model
    
    Endpoints:
    - redact_pii(ocr_data, image_path) -> PII detections + redacted image
    """
    
    def __init__(self, config: PIIConfig = None):
        """Initialize PII redactor with exact configuration from your solution"""
        self.config = config or PIIConfig()
        self.analyzer = None
        self.entities = None
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """Initialize Presidio analyzer with exact configuration from your solution"""
        try:
            # Configure NLP engine to use transformer model (exact from your solution)
            nlp_configuration = self.config.NLP_CONFIG
            
            try:
                # Create NLP engine with transformer model
                nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
                print("Using transformer model (en_core_web_trf)")
            except Exception as e:
                print(f"Transformer model failed: {e}")
                print("Falling back to default model...")
                nlp_engine = NlpEngineProvider().create_engine()
            
            # Setup registry 
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(nlp_engine=nlp_engine)
            registry.add_recognizer(InAadhaarRecognizer())
            registry.add_recognizer(InPanRecognizer())
            registry.add_recognizer(InVehicleRegistrationRecognizer())
            registry.add_recognizer(InPassportRecognizer())
            
            self.analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)
            
            # Geting supported entities 
            entities = self.analyzer.get_supported_entities()
            for excluded in self.config.PII_CONFIG['excluded_entities']:
                if excluded in entities:
                    entities.remove(excluded)
            
            self.entities = entities
            print(f"Presidio analyzer initialized with entities: {entities}")
            
        except Exception as e:
            print(f"Failed to initialize Presidio analyzer: {e}")
            raise
    
    def tokens(self, s: str) -> List[str]:
        """Lightweight tokenizer """
        if not s:
            return []
        s_norm = unidecode(s).lower()
        return re.findall(r"[a-z0-9@.+-]+", s_norm)
    
    def redact_pii(self, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main endpoint: Detect and redact PII from OCR results
        
        Args:
            ocr_data: Dictionary from OCRProcessor.process_image()
            
        Returns:
            Dict containing:
            - 'pii_detections': List of detected PII entities
            - 'pii_dataframe': DataFrame of PII detections
            - 'redacted_image_path': Path to text-redacted image
            - 'pii_bounding_boxes': Bounding boxes of PII text
        """
        ocr_results = ocr_data['ocr_results']
        ocr_dict = ocr_data['ocr_dict']
        image_path = ocr_data['image_path']
        
        # Detect PII exactly like your solution
        PII_identified_entities = []
        threshold = self.config.PII_CONFIG['threshold']
        
        for i, items in enumerate(ocr_results):
            # Build context exactly like your solution
            prev_to_prev_text = ocr_results[i-2][2] if i > 1 else ""
            prev_text = ocr_results[i-1][2] if i > 0 else ""
            next_text = ocr_results[i+1][2] if i < len(ocr_results)-1 else ""
            next_to_next_text = ocr_results[i+2][2] if i < len(ocr_results)-2 else ""
            orig_item = items[2]
            
            text_index = items[0]
            target_text = unidecode(prev_to_prev_text+" "+prev_text+" "+orig_item+" "+next_text+" "+next_to_next_text)
            
            # Analyze with Presidio exactly like your solution
            presidio_results = self.analyzer.analyze(
                text=target_text, 
                entities=self.entities, 
                language="en"
            )
            
            if presidio_results:
                item_tokens = self.tokens(unidecode(orig_item))
                for result in presidio_results:
                    span = target_text[result.start:result.end]
                    span_tokens = self.tokens(span)
                    if any(tok for tok in item_tokens if tok and tok in span_tokens) and result.score > threshold:
                
                        PII_identified_entities.append([
                            text_index, result.entity_type, items[2], result.score
                        ])
        
        # Get PII bounding boxes exactly like your solution
        if PII_identified_entities:
            arr = np.array(PII_identified_entities, dtype=object)
            PII_entities_dict = {
                "text index": arr[:,0].tolist(), 
                "entity type": arr[:,1].tolist(), 
                "text": arr[:,2].tolist()
            }
            PII_indices = np.unique(np.array(PII_entities_dict["text index"]))
            PII_boundingboxes = [ocr_dict["bbox"][i] for i in PII_indices]
        else:
            PII_boundingboxes = []
            PII_indices = []
        
        # Create redacted image
        redacted_image_path = self._create_redacted_image(image_path, PII_boundingboxes)
        
        return {
            'pii_detections': PII_identified_entities,
            'pii_indices': PII_indices,
            'pii_bounding_boxes': PII_boundingboxes,
            'redacted_image_path': redacted_image_path,
            'num_pii_detected': len(PII_identified_entities)
        }
    
    def _create_redacted_image(self, image_path: str, pii_bounding_boxes: List) -> str:
        """Create redacted image exactly like your solution"""
        # Load the image
        img_ = Image.open(image_path).convert("RGB")
        w, h = img_.size
        
        # Create a fully blurred version of the image
        blurred_img = img_.filter(ImageFilter.GaussianBlur(
            radius=self.config.IMAGE_CONFIG['blur_radius']
        ))
        
        # Create a mask
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw all PII bounding boxes onto the mask exactly like your solution
        for box in pii_bounding_boxes:
            if not box:
                continue
            
            # Convert numpy array to list of tuples if necessary
            poly_points = [tuple(p) for p in np.array(box, dtype=int)]
            
            # Draw the polygon on the mask
            draw.polygon(poly_points, fill=255)
        
        # Composite the blurred image onto the original using the mask
        img_.paste(blurred_img, (0, 0), mask)
        
        # Save redacted image
        base_name = os.path.splitext(image_path)[0]
        redacted_path = f"{base_name}{self.config.OUTPUT_CONFIG['text_redacted_suffix']}.jpg"
        img_.save(redacted_path)
        
        return redacted_path
    
    def analyze_single_text(self, text: str) -> List[Dict]:
        """Analyze single text for PII (utility function)"""
        text_normalized = unidecode(text)
        presidio_results = self.analyzer.analyze(
            text=text_normalized, 
            entities=self.entities, 
            language="en"
        )
        
        results = []
        for result in presidio_results:
            results.append({
                'entity_type': result.entity_type,
                'score': result.score,
                'text': text[result.start:result.end],
                'start': result.start,
                'end': result.end
            })
        
        return results
