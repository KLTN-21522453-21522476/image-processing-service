"""
Cải thiện chất lượng ảnh
"""
import cv2
import numpy as np
from PIL import Image
from ..config import Config

class ImageEnhancer:
    @staticmethod
    def set_image_dpi(im, dpi=None):
        """Tăng độ phân giải của ảnh."""
        if not isinstance(im, Image.Image):
            raise TypeError("Input must be a PIL Image")
        
        dpi = dpi or Config.DEFAULT_DPI
        max_size = Config.MAX_IMAGE_SIZE
        
        length_x, width_y = im.size
        factor = min(1, float(max_size / length_x))
        size = int(factor * length_x), int(factor * width_y)
        im_resized = im.resize(size, Image.LANCZOS)
        
        im_resized.info['dpi'] = (dpi, dpi)
        return im_resized
    
    @staticmethod
    def get_grayscale(image):
        """Chuyển đổi ảnh màu sang ảnh xám."""
        if image is None:
            raise ValueError("Input image is None")
            
        if len(image.shape) == 2:
            return image
            
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def thresholding(image):
        """Áp dụng ngưỡng để chuyển đổi ảnh xám thành ảnh nhị phân."""
        if image is None:
            raise ValueError("Input image is None")
            
        if len(image.shape) == 3:
            image = ImageEnhancer.get_grayscale(image)
            
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    @staticmethod
    def sharpen_text(image, kernel_size=None, amount=None):
        """Làm rõ nét chữ trong ảnh."""
        if image is None:
            raise ValueError("Input image is None")
        
        kernel_size = kernel_size or Config.DEFAULT_SHARPEN_KERNEL
        amount = amount or Config.DEFAULT_SHARPEN_AMOUNT
        
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    @staticmethod
    def thin_and_skeletonize(image):
        """Làm mỏng và tạo bộ khung cho ảnh."""
        if image is None:
            raise ValueError("Input image is None")
        
        kernel_size = Config.EROSION_KERNEL_SIZE
        kernel = np.ones(kernel_size, np.uint8)
        return cv2.erode(image, kernel, iterations=1)
