"""
Loại bỏ nhiễu từ ảnh
"""
import cv2
import numpy as np
from ..config import Config

class NoiseReducer:
    @staticmethod
    def remove_noise(image):
        """Loại bỏ nhiễu từ ảnh màu."""
        if image is None:
            raise ValueError("Input image is None")
            
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        if len(image.shape) != 3 or image.shape[2] != 3:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError("Input image must be a color image or grayscale image")
        
        params = Config.NOISE_REDUCTION_KERNEL
        return cv2.fastNlMeansDenoisingColored(image, None, *params)
