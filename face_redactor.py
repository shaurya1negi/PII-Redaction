"""
Face Redactor Module
==================

Handles face detection and redaction using MTCNN with exact parameters
from your original solution.

Endpoint: redact_faces() -> Returns face detections and fully redacted image
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import List, Dict, Tuple, Any
import os
from facenet_pytorch import MTCNN

from .config import PIIConfig

class FaceRedactor:
    """
    Face Redactor using MTCNN with exact configuration from your solution
    
    Endpoints:
    - redact_faces(image_path) -> Face detections + fully redacted image
    """
    
    def __init__(self, config: PIIConfig = None):
        """Initialize face redactor with exact configuration from your solution"""
        self.config = config or PIIConfig()
        self.mtcnn = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize MTCNN detector with exact parameters from your solution"""
        try:
            self.mtcnn = MTCNN(
                image_size=self.config.FACE_CONFIG['image_size'],
                select_largest=self.config.FACE_CONFIG['select_largest'],
                keep_all=self.config.FACE_CONFIG['keep_all'],
                device=self.config.FACE_CONFIG['device']
            )
            print("MTCNN face detector initialized.")
        except Exception as e:
            print(f"❌ Failed to initialize MTCNN: {e}")
            # Fallback to CPU if CUDA fails
            try:
                self.mtcnn = MTCNN(
                    image_size=self.config.FACE_CONFIG['image_size'],
                    select_largest=self.config.FACE_CONFIG['select_largest'],
                    keep_all=self.config.FACE_CONFIG['keep_all'],
                    device='cpu'
                )
                print("MTCNN initialized with CPU fallback.")
            except Exception as e2:
                print(f"❌ Failed to initialize MTCNN even with CPU: {e2}")
                raise
    
    def redact_faces(self, image_path: str) -> Dict[str, Any]:
        """
        Main endpoint: Detect and redact faces from image
        
        Args:
            image_path (str): Path to input image (can be text-redacted image)
            
        Returns:
            Dict containing:
            - 'face_boxes': Detected face bounding boxes
            - 'face_probabilities': Detection probabilities
            - 'face_landmarks': Facial landmarks
            - 'redacted_image_path': Path to fully redacted image
            - 'num_faces_detected': Number of faces found
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image and convert to RGB
        img = Image.open(image_path).convert("RGB")
        
        # Detect faces exactly like your solution
        boxes, probs, points = self.mtcnn.detect(img, landmarks=True)
        
        # Create face-redacted image
        redacted_image_path = self._create_face_redacted_image(image_path, img, boxes)
        
        num_faces = len(boxes) if boxes is not None else 0
        
        return {
            'face_boxes': boxes.tolist() if boxes is not None else [],
            'face_probabilities': probs.tolist() if probs is not None else [],
            'face_landmarks': points.tolist() if points is not None else [],
            'redacted_image_path': redacted_image_path,
            'num_faces_detected': num_faces,
            'original_image_path': image_path
        }
    
    def _create_face_redacted_image(self, image_path: str, img: Image.Image, boxes) -> str:
        """Create face-redacted image exactly like your solution"""
        img_draw = img.copy()
        
        if boxes is None:
            print("No faces detected.")
        else:
            # Redact faces exactly like your solution
            for box in boxes:
                # box may be a numpy array; take first 4 values (x1,y1,x2,y2)
                x1, y1, x2, y2 = map(int, box[:4])
                
                # clamp to image bounds
                x1 = max(0, min(img_draw.width, x1))
                x2 = max(0, min(img_draw.width, x2))
                y1 = max(0, min(img_draw.height, y1))
                y2 = max(0, min(img_draw.height, y2))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # crop region, blur and paste back exactly like your solution
                region = img_draw.crop((x1, y1, x2, y2))
                blurred = region.filter(ImageFilter.GaussianBlur(
                    radius=self.config.FACE_CONFIG['blur_radius']
                ))
                img_draw.paste(blurred, (x1, y1))
        
        # Save fully redacted image
        base_name = os.path.splitext(image_path)[0]
        
        # If this is already a text-redacted image, create final output
        if self.config.OUTPUT_CONFIG['text_redacted_suffix'] in base_name:
            redacted_path = base_name.replace(
                self.config.OUTPUT_CONFIG['text_redacted_suffix'],
                self.config.OUTPUT_CONFIG['final_redacted_suffix']
            ) + '.jpg'
        else:
            redacted_path = f"{base_name}{self.config.OUTPUT_CONFIG['face_redacted_suffix']}.jpg"
        
        img_draw.save(redacted_path)
        
        return redacted_path
    
    def detect_faces_only(self, image_path: str) -> Dict[str, Any]:
        """Detect faces without redaction (utility function)"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = Image.open(image_path).convert("RGB")
        boxes, probs, points = self.mtcnn.detect(img, landmarks=True)
        
        num_faces = len(boxes) if boxes is not None else 0
        
        return {
            'face_boxes': boxes.tolist() if boxes is not None else [],
            'face_probabilities': probs.tolist() if probs is not None else [],
            'face_landmarks': points.tolist() if points is not None else [],
            'num_faces_detected': num_faces
        }
    
    def get_face_summary(self, face_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary statistics of face detection results"""
        return {
            'total_faces': face_data['num_faces_detected'],
            'avg_confidence': np.mean(face_data['face_probabilities']) if face_data['face_probabilities'] else 0,
            'min_confidence': np.min(face_data['face_probabilities']) if face_data['face_probabilities'] else 0,
            'max_confidence': np.max(face_data['face_probabilities']) if face_data['face_probabilities'] else 0,
            'face_boxes': face_data['face_boxes']
        }
