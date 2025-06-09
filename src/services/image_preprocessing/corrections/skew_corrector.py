"""
Sửa ảnh nghiêng (skew correction)
"""
import cv2
import numpy as np
from scipy import ndimage
import pytesseract
from ..config import Config

class SkewCorrector:
    def __init__(self, skew_threshold=None, confidence_threshold=None):
        self.skew_threshold = skew_threshold or Config.SKEW_THRESHOLD
        self.confidence_threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD
    
    def detect_skew_advanced(self, image, method='hough'):
        """Phát hiện góc nghiêng của ảnh."""
        if image is None:
            raise ValueError("Input image is None")
            
        if method == 'auto':
            angles = []
            
            try:
                angle_hough = self._detect_skew_hough(image)
                angles.append(angle_hough)
            except:
                pass
                
            try:
                angle_projection = self._detect_skew_projection(image)
                angles.append(angle_projection)
            except:
                pass
                
            try:
                angle_textlines = self._detect_skew_textlines(image)
                angles.append(angle_textlines)
            except:
                pass
            
            if angles:
                return np.median(angles)
            return 0
            
        elif method == 'hough':
            return self._detect_skew_hough(image)
        elif method == 'projection':
            return self._detect_skew_projection(image)
        elif method == 'textlines':
            return self._detect_skew_textlines(image)
        else:
            raise ValueError("Invalid method. Choose from: 'hough', 'projection', 'textlines', 'auto'")
    
    def _detect_skew_hough(self, image):
        """Phát hiện skew bằng Hough Transform"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        angles = []
        if lines is not None:
            for rho, theta in lines[:, 0]:
                angle = theta * 180 / np.pi - 90
                if -45 <= angle <= 45:
                    angles.append(angle)
        
        if angles:
            return np.median(angles)
        return 0
    
    def _detect_skew_projection(self, image, angle_range=(-45, 45), angle_step=0.5):
        """Phát hiện skew bằng Projection Profile"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        best_angle = 0
        max_variance = 0
        
        for angle in np.arange(angle_range[0], angle_range[1], angle_step):
            rotated = ndimage.rotate(binary, angle, reshape=False, cval=255)
            horizontal_profile = np.sum(rotated, axis=1)
            variance = np.var(horizontal_profile)
            
            if variance > max_variance:
                max_variance = variance
                best_angle = angle
        
        return best_angle
    
    def _detect_skew_textlines(self, image):
        """Phát hiện skew dựa trên text lines"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        angles = []
        for contour in contours:
            if len(contour) >= 5:
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                    
                angles.append(angle)
        
        if angles:
            return np.median(angles)
        return 0
    
    def correct_skew_advanced(self, image, method='auto', force_correction=False):
        """Sửa ảnh nghiêng với validation."""
        if image is None:
            raise ValueError("Input image is None")
        
        original_image = image.copy()
        skew_angle = self.detect_skew_advanced(image, method)
        
        processing_info = {
            'skew_angle': skew_angle,
            'method_used': method,
            'corrected': False,
            'validation_passed': False
        }
        
        if abs(skew_angle) > self.skew_threshold or force_correction:
            corrected_image = self._rotate_image_advanced(image, skew_angle)
            
            if not force_correction:
                validation_passed = self._validate_skew_correction(original_image, corrected_image)
                processing_info['validation_passed'] = validation_passed
                
                if validation_passed:
                    processing_info['corrected'] = True
                    return corrected_image, skew_angle, processing_info
                else:
                    return original_image, 0, processing_info
            else:
                processing_info['corrected'] = True
                processing_info['validation_passed'] = True
                return corrected_image, skew_angle, processing_info
        
        return original_image, 0, processing_info
    
    def _rotate_image_advanced(self, image, angle):
        """Xoay ảnh với tính toán kích thước mới chính xác"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def _validate_skew_correction(self, original, corrected):
        """Validate việc sửa skew"""
        try:
            orig_conf = self._get_ocr_confidence(original)
            corr_conf = self._get_ocr_confidence(corrected)
            return corr_conf > orig_conf * 1.1
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
