"""
Xử lý ảnh cho OCR
"""
import cv2
import logging
from PIL import Image
from ..converters.format_converter import FormatConverter
from ..corrections.orientation_corrector import OrientationCorrector
from ..corrections.skew_corrector import SkewCorrector
from ..enhancers.image_enhancer import ImageEnhancer
from ..enhancers.noise_reducer import NoiseReducer

class OCRProcessor:
    def __init__(self):
        self.format_converter = FormatConverter()
        self.orientation_corrector = OrientationCorrector()
        self.skew_corrector = SkewCorrector()
        self.image_enhancer = ImageEnhancer()
        self.noise_reducer = NoiseReducer()
    
    def auto_correct_document(self, image, correct_orientation=True, correct_skew=True):
        """Tự động sửa cả orientation và skew của document."""
        if image is None:
            raise ValueError("Input image is None")
        
        result = {
            'original': image.copy(),
            'final': image.copy(),
            'orientation_info': None,
            'skew_info': None,
            'processing_steps': []
        }
        
        current_image = image.copy()
        
        # Bước 1: Sửa orientation
        if correct_orientation:
            corrected_img, angle, info = self.orientation_corrector.correct_orientation(
                current_image, method='auto'
            )
            if info['corrected']:
                current_image = corrected_img
                result['processing_steps'].append(f"Corrected orientation: {angle}°")
            result['orientation_info'] = info
        
        # Bước 2: Sửa skew
        if correct_skew:
            corrected_img, angle, info = self.skew_corrector.correct_skew_advanced(
                current_image, method='auto'
            )
            if info['corrected']:
                current_image = corrected_img
                result['processing_steps'].append(f"Corrected skew: {angle:.2f}°")
            result['skew_info'] = info
        
        result['final'] = current_image
        return result
    
    def process_image_for_ocr(self, image, auto_correct=True):
        """Xử lý ảnh hoàn chỉnh cho OCR."""
        # Chuyển đổi sang định dạng OpenCV
        if isinstance(image, Image.Image):
            cv_image = self.format_converter.pil_to_cv(image)
        elif isinstance(image, str):
            cv_image = cv2.imread(image)
        else:
            cv_image = image
            
        if cv_image is None or cv_image.size == 0:
            raise ValueError("Invalid input image")
        
        result = {
            'original': cv_image.copy(),
            'processed': None,
            'corrections_applied': [],
            'processing_info': {}
        }
        
        current_image = cv_image.copy()
        
        # Tự động sửa orientation và skew
        if auto_correct:
            correction_result = self.auto_correct_document(current_image)
            current_image = correction_result['final']
            result['corrections_applied'] = correction_result['processing_steps']
            result['processing_info']['orientation'] = correction_result['orientation_info']
            result['processing_info']['skew'] = correction_result['skew_info']
        
        # Chuẩn hóa ảnh
        current_image = self.format_converter.normalize(current_image)
             
        # Điều chỉnh DPI
        pil_image = self.format_converter.cv_to_pil(current_image)
        pil_image = self.image_enhancer.set_image_dpi(pil_image)
        current_image = self.format_converter.pil_to_cv(pil_image)
        
        # Loại bỏ nhiễu
        current_image = self.noise_reducer.remove_noise(current_image)
        
        # Chuyển sang ảnh xám
        current_image = self.image_enhancer.get_grayscale(current_image)
        
        result['processed'] = current_image
        return result
    
    def preprocess_for_detection(self, image_path: str) -> Image.Image:
        """Xử lý ảnh cho detection."""
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            raise ValueError(f"Failed to read image: {image_path}")
            
        try:
            # Sửa skew trước
            corrected_img, skew_angle, skew_info = self.skew_corrector.correct_skew_advanced(
                cv_img, method='auto', force_correction=True
            )
            
            if skew_info['corrected']:
                logging.info(f"Skew corrected by {skew_angle:.2f} degrees")
                cv_img = corrected_img
            
            # Sau đó sửa orientation
            corrected_img, orientation_angle, orientation_info = self.orientation_corrector.correct_orientation(
                cv_img, method='auto', force_correction=True
            )
            
            if orientation_info['corrected']:
                logging.info(f"Orientation corrected by {orientation_angle} degrees")
                cv_img = corrected_img
            
            # Chuyển sang PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            
            return pil_img
            
        except Exception as e:
            logging.error(f"Error during preprocessing: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")
