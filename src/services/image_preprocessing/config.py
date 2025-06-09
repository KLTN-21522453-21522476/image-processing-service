"""
Cấu hình chung cho image preprocessing
"""

class Config:
    # Ngưỡng cho skew correction
    SKEW_THRESHOLD = 0.5
    
    # Ngưỡng confidence cho OCR
    CONFIDENCE_THRESHOLD = 30
    
    # Cài đặt DPI
    DEFAULT_DPI = 300
    MAX_IMAGE_SIZE = 1024
    
    # Kernel sizes cho các operations
    NOISE_REDUCTION_KERNEL = (10, 10, 7, 15)
    EROSION_KERNEL_SIZE = (5, 5)
    
    # Sharpening parameters
    DEFAULT_SHARPEN_KERNEL = 3
    DEFAULT_SHARPEN_AMOUNT = 1.5
