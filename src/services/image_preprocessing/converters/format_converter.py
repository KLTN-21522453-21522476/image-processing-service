"""
Chuyển đổi giữa các định dạng ảnh
"""
import cv2
import numpy as np
from PIL import Image
import os

class FormatConverter:
    @staticmethod
    def cv_to_pil(cv_image):
        """Chuyển đổi ảnh OpenCV sang PIL Image."""
        if cv_image is None:
            raise ValueError("Input image is None")
            
        # Handle dictionary result from process_image_for_ocr
        if isinstance(cv_image, dict) and 'processed' in cv_image:
            cv_image = cv_image['processed']
            
        if len(cv_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_image)
        else:
            return Image.fromarray(cv_image)

    @staticmethod
    def pil_to_cv(pil_image):
        """Chuyển đổi PIL Image sang định dạng OpenCV."""
        if isinstance(pil_image, str):
            if os.path.isfile(pil_image):
                return cv2.imread(pil_image)
            else:
                raise FileNotFoundError(f"File not found: {pil_image}")
                
        if isinstance(pil_image, Image.Image):
            img_np = np.array(pil_image)
            
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                return img_np
        
        if isinstance(pil_image, np.ndarray):
            return pil_image
            
        raise TypeError("Input must be PIL Image, file path, or numpy array")
    
    @staticmethod
    def normalize(img):
        """Chuẩn hóa ảnh về phạm vi 0-255."""
        if img is None:
            raise ValueError("Input image is None")
            
        if len(img.shape) == 3:
            norm_img = np.zeros_like(img)
        else:
            norm_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            
        return cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
