"""
Sửa ảnh lộn ngược (orientation correction)
"""
import cv2
import numpy as np
import pytesseract
from ..config import Config

class OrientationCorrector:
    def __init__(self, confidence_threshold=None):
        self.confidence_threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD
    
    def detect_orientation(self, image, method='ocr'):
        """Phát hiện hướng của ảnh (0, 90, 180, 270 độ)."""
        if image is None:
            raise ValueError("Input image is None")
        
        if method == 'auto':
            try:
                angle, conf = self._detect_orientation_ocr(image)
                if conf > self.confidence_threshold:
                    return angle, conf
            except:
                pass
            
            try:
                return self._detect_orientation_text_direction(image), 50
            except:
                return 0, 0
                
        elif method == 'ocr':
            return self._detect_orientation_ocr(image)
        elif method == 'text_direction':
            angle = self._detect_orientation_text_direction(image)
            return angle, 50
        else:
            raise ValueError("Invalid method. Choose from: 'ocr', 'text_direction', 'auto'")
    
    def _detect_orientation_ocr(self, image):
        """Detect orientation bằng OCR confidence"""
        orientations = [0, 90, 180, 270]
        best_orientation = 0
        best_confidence = 0
        
        for angle in orientations:
            if angle != 0:
                rotated = self._rotate_image_simple(image, -angle)
            else:
                rotated = image
            
            try:
                confidence = self._get_ocr_confidence(rotated)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_orientation = angle
            except:
                continue
        
        return best_orientation, best_confidence
    
    def _detect_orientation_text_direction(self, image):
        """Detect orientation dựa trên text direction"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horizontal)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vertical)
        
        h_count = cv2.countNonZero(horizontal_lines)
        v_count = cv2.countNonZero(vertical_lines)
        
        if h_count > v_count * 1.5:
            return 0
        elif v_count > h_count * 1.5:
            return 90
        else:
            return 0
    
    def correct_orientation(self, image, method='auto', force_correction=False):
        """Sửa hướng ảnh lộn ngược."""
        if image is None:
            raise ValueError("Input image is None")
        
        original_image = image.copy()
        orientation_angle, confidence = self.detect_orientation(image, method)
        
        processing_info = {
            'orientation_angle': orientation_angle,
            'confidence': confidence,
            'method_used': method,
            'corrected': False,
            'validation_passed': False
        }
        
        if (confidence > self.confidence_threshold and orientation_angle != 0) or force_correction:
            corrected_image = self._rotate_image_simple(image, -orientation_angle)
            
            if not force_correction:
                validation_passed = self._validate_orientation_correction(original_image, corrected_image)
                processing_info['validation_passed'] = validation_passed
                
                if validation_passed:
                    processing_info['corrected'] = True
                    return corrected_image, orientation_angle, processing_info
                else:
                    return original_image, 0, processing_info
            else:
                processing_info['corrected'] = True
                processing_info['validation_passed'] = True
                return corrected_image, orientation_angle, processing_info
        
        return original_image, 0, processing_info
    
    def _rotate_image_simple(self, image, angle):
        """Xoay ảnh đơn giản (90, 180, 270 độ)"""
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270 or angle == -90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, rotation_matrix, (w, h))
    
    def _validate_orientation_correction(self, original, corrected):
        """Validate việc sửa orientation"""
        try:
            orig_conf = self._get_ocr_confidence(original)
            corr_conf = self._get_ocr_confidence(corrected)
            return corr_conf > orig_conf * 1.2
        except:
            return True
    
    def _get_ocr_confidence(self, image):
        """Lấy confidence score từ OCR"""
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            return np.mean(confidences) if confidences else 0
        except:
            return 0
