"""
OCR Processor: text extraction and annotation
"""

import easyocr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from typing import List, Dict, Tuple, Any
import os
from .config import PIIConfig

class OCRProcessor:
    
    def __init__(self, config: PIIConfig = None):
        self.config = config or PIIConfig()
        self.reader = None
        self._initialize_reader()
    
    def _initialize_reader(self):
        self.reader = easyocr.Reader(
            self.config.OCR_CONFIG['languages'], 
            gpu=self.config.OCR_CONFIG['gpu']
        )
    
    def process_image(self, image_path: str) -> Dict[str, Any]: # -> Dict[str, Any]: is the return type of the function
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        ocr_result = self.reader.readtext(
            image_path,
            **self.config.OCR_CONFIG['parameters']
        )
        
        # Handle empty OCR results
        if not ocr_result:
            print(f"Warning: No text detected in image: {image_path}")
            original_image = Image.open(image_path).convert("RGB")
            return {
                'ocr_results': [],
                'ocr_dict': {
                    "text index": [],
                    "bbox": [],
                    "text": [],
                    "confidence": []
                },
                'annotated_image_path': None,
                'original_image': original_image,
                'image_path': image_path
            }
        
        # Index OCR results 
        ocr_indexed = []
        for idx, item in enumerate(ocr_result):
            bbox = item[0] if len(item) > 0 else None
            text = item[1] if len(item) > 1 else ""
            confidence = item[2] if len(item) > 2 else None
            ocr_indexed.append((idx, bbox, text, confidence))
        
        # Create DataFrame exactly like your solution
        ocr_result_array = np.array(ocr_indexed, dtype=object)
        ocr_result_dict = {
            "text index": ocr_result_array[:,0].tolist(),
            "bbox": ocr_result_array[:,1].tolist(),
            "text": ocr_result_array[:,2].tolist(),
            "confidence": ocr_result_array[:,3].tolist()
        }
        
        # Create annotated image
        annotated_image_path = self._create_annotated_image(image_path, ocr_indexed)
        
        # Load original image
        original_image = Image.open(image_path).convert("RGB")
        
        return {
            'ocr_results': ocr_indexed,
            'ocr_dict': ocr_result_dict,
            'annotated_image_path': annotated_image_path,
            'original_image': original_image,
            'image_path': image_path
        }
    
    def _create_annotated_image(self, image_path: str, ocr_results: List) -> str:
        """Create annotated image with OCR bounding boxes"""
        img = Image.open(image_path).convert("RGB")
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        

        for item in ocr_results:
            coordinates = item[1]  # List of 4 points
            if coordinates:
                polygon = [tuple(point) for point in coordinates]
                draw.polygon(polygon, outline="red", width=2)
        
        # Save annotated image
        base_name = os.path.splitext(image_path)[0]
        annotated_path = f"{base_name}{self.config.OUTPUT_CONFIG['ocr_annotated_suffix']}.jpg"
        img_draw.save(annotated_path)
        
        return annotated_path
    
    def get_text_summary(self, ocr_results: List) -> Dict[str, Any]:
        """Get summary statistics of OCR results"""
        if not ocr_results:
            return {'total_detections': 0, 'avg_confidence': 0, 'text_items': []}
        
        confidences = [item[3] for item in ocr_results if item[3] is not None]
        texts = [item[2] for item in ocr_results if item[2]]
        
        return {
            'total_detections': len(ocr_results),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'min_confidence': np.min(confidences) if confidences else 0,
            'max_confidence': np.max(confidences) if confidences else 0,
            'text_items': texts
        }
