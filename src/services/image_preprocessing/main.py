"""
Main class giữ nguyên interface cũ để backward compatibility
"""
from .processors.ocr_processor import OCRProcessor
from .converters.format_converter import FormatConverter
from .corrections.orientation_corrector import OrientationCorrector
from .corrections.skew_corrector import SkewCorrector
from .enhancers.image_enhancer import ImageEnhancer
from .enhancers.noise_reducer import NoiseReducer

class PreprocessImage(OCRProcessor):
    """
    Lớp chính kế thừa từ OCRProcessor để giữ backward compatibility
    """
    def __init__(self):
        super().__init__()
    
    # Expose các method từ các component
    def cv_to_pil(self, cv_image):
        return self.format_converter.cv_to_pil(cv_image)
    
    def pil_to_cv(self, pil_image):
        return self.format_converter.pil_to_cv(pil_image)
    
    def normalize(self, img):
        return self.format_converter.normalize(img)
    
    def detect_skew_advanced(self, image, method='hough'):
        return self.skew_corrector.detect_skew_advanced(image, method)
    
    def correct_skew_advanced(self, image, method='auto', force_correction=False):
        return self.skew_corrector.correct_skew_advanced(image, method, force_correction)
    
    def detect_orientation(self, image, method='ocr'):
        return self.orientation_corrector.detect_orientation(image, method)
    
    def correct_orientation(self, image, method='auto', force_correction=False):
        return self.orientation_corrector.correct_orientation(image, method, force_correction)
    
    def set_image_dpi(self, im):
        return self.image_enhancer.set_image_dpi(im)
    
    def remove_noise(self, image):
        return self.noise_reducer.remove_noise(image)
    
    def get_grayscale(self, image):
        return self
