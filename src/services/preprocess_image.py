import cv2
import numpy as np
import tempfile
from PIL import Image
import os

class PreprocessImage:
    def __init__(self):
        pass
    
    def cv_to_pil(self, cv_image):
        """
        Chuyển đổi ảnh OpenCV sang PIL Image.
        
        Args:
            cv_image: Ảnh định dạng OpenCV (numpy array)
            
        Returns:
            Đối tượng PIL Image
        """
        if cv_image is None:
            raise ValueError("Input image is None")
            
        # Kiểm tra xem ảnh có phải là ảnh màu không
        if len(cv_image.shape) == 3:
            # Chuyển BGR sang RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_image)
        else:
            # Ảnh xám
            return Image.fromarray(cv_image)

    def pil_to_cv(self, pil_image):
        """
        Chuyển đổi PIL Image sang định dạng OpenCV.
        
        Args:
            pil_image: Đối tượng PIL Image hoặc đường dẫn đến file ảnh
            
        Returns:
            Ảnh định dạng OpenCV (numpy array)
        """
        # Nếu đầu vào là đường dẫn file
        if isinstance(pil_image, str):
            if os.path.isfile(pil_image):
                # Đọc file trực tiếp bằng OpenCV
                return cv2.imread(pil_image)
            else:
                raise FileNotFoundError(f"File not found: {pil_image}")
                
        # Nếu đầu vào là PIL Image
        if isinstance(pil_image, Image.Image):
            # Chuyển PIL Image sang mảng numpy
            img_np = np.array(pil_image)
            
            # Nếu ảnh là RGB, chuyển sang BGR (định dạng OpenCV)
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                return img_np
        
        # Nếu đầu vào đã là mảng numpy, trả về nguyên bản
        if isinstance(pil_image, np.ndarray):
            return pil_image
            
        raise TypeError("Input must be PIL Image, file path, or numpy array")
    
    def normalize(self, img):
        """
        Chuẩn hóa ảnh để đưa giá trị pixel về phạm vi bình thường.
        
        Args:
            img: Ảnh đầu vào (numpy array)
            
        Returns:
            Ảnh đã được chuẩn hóa
        """
        if img is None:
            raise ValueError("Input image is None")
            
        # Tạo mảng output với cùng kích thước và số kênh màu
        if len(img.shape) == 3:
            norm_img = np.zeros_like(img)
        else:
            norm_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            
        return cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    
    def deskew(self, image):
        """
        Hiệu chỉnh độ nghiêng của ảnh để cải thiện hiệu suất OCR.
        
        Args:
            image: Ảnh đầu vào (nên là ảnh nhị phân hoặc ảnh xám)
            
        Returns:
            Ảnh đã được hiệu chỉnh độ nghiêng
        """
        if image is None:
            raise ValueError("Input image is None")
            
        # Chuyển sang ảnh xám nếu là ảnh màu
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Tìm tọa độ các điểm không đen
        co_ords = np.column_stack(np.where(gray > 0))
        if len(co_ords) == 0:
            return image  # Không có điểm nào để xử lý
            
        angle = cv2.minAreaRect(co_ords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Xoay ảnh gốc (có thể là ảnh màu hoặc ảnh xám)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    
    def set_image_dpi(self, im):
        """
        Tăng độ phân giải của ảnh lên 300 PPI để cải thiện hiệu suất OCR.
        
        Args:
            im: Đối tượng PIL Image
            
        Returns:
            Đối tượng PIL Image đã được điều chỉnh
        """
        if not isinstance(im, Image.Image):
            raise TypeError("Input must be a PIL Image")
            
        length_x, width_y = im.size
        factor = min(1, float(1024.0 / length_x))
        size = int(factor * length_x), int(factor * width_y)
        im_resized = im.resize(size, Image.LANCZOS)
        
        im_resized.info['dpi'] = (300, 300)
        return im_resized
    
    def remove_noise(self, image):
        """
        Loại bỏ nhiễu từ ảnh để làm mịn ảnh.
        
        Args:
            image: Ảnh đầu vào (numpy array, ảnh màu)
            
        Returns:
            Ảnh đã được loại bỏ nhiễu
        """
        if image is None:
            raise ValueError("Input image is None")
            
        # Đảm bảo ảnh là numpy array và có 3 kênh màu
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        # Kiểm tra xem ảnh có phải là ảnh màu không
        if len(image.shape) != 3 or image.shape[2] != 3:
            # Nếu là ảnh xám, chuyển sang ảnh màu
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError("Input image must be a color image or grayscale image")
                
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    
    def thin_and_skeletonize(self, image):
        """
        Làm mỏng và tạo bộ khung cho ảnh, đặc biệt hữu ích cho văn bản viết tay.
        
        Args:
            image: Ảnh đầu vào (nên là ảnh nhị phân hoặc ảnh xám)
            
        Returns:
            Ảnh đã được làm mỏng
        """
        if image is None:
            raise ValueError("Input image is None")
            
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)
    
    def get_grayscale(self, image):
        """
        Chuyển đổi ảnh màu sang ảnh xám.
        
        Args:
            image: Ảnh đầu vào (ảnh màu)
            
        Returns:
            Ảnh xám
        """
        if image is None:
            raise ValueError("Input image is None")
            
        # Nếu ảnh đã là ảnh xám, trả về nguyên bản
        if len(image.shape) == 2:
            return image
            
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def thresholding(self, image):
        """
        Áp dụng ngưỡng để chuyển đổi ảnh xám thành ảnh nhị phân.
        Sử dụng phương pháp Otsu để tự động xác định ngưỡng tối ưu.
        
        Args:
            image: Ảnh đầu vào (ảnh xám)
            
        Returns:
            Ảnh nhị phân
        """
        if image is None:
            raise ValueError("Input image is None")
            
        # Đảm bảo ảnh là ảnh xám
        if len(image.shape) == 3:
            image = self.get_grayscale(image)
            
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    def sharpen_text(self, image, kernel_size=3, amount=1.5):
        """
        Làm rõ nét chữ trong ảnh bằng cách áp dụng bộ lọc làm sắc nét.
        
        Args:
            image: Ảnh đầu vào (numpy array)
            kernel_size: Kích thước kernel cho bộ lọc làm sắc nét (số lẻ)
            amount: Mức độ làm sắc nét, giá trị cao hơn cho hiệu ứng mạnh hơn
            
        Returns:
            Ảnh đã được làm rõ nét chữ
        """
        if image is None:
            raise ValueError("Input image is None")
            
        # Đảm bảo kernel_size là số lẻ
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Làm mờ ảnh với Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Áp dụng kỹ thuật unsharp masking
        if len(image.shape) == 3:
            # Ảnh màu
            sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        else:
            # Ảnh xám
            sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
            
        # Đảm bảo giá trị pixel nằm trong khoảng hợp lệ
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened

    
    def process_image_for_ocr(self, image):
        """
        Xử lý ảnh hoàn chỉnh cho OCR.
        
        Args:
            image: Đối tượng PIL Image, đường dẫn file, hoặc numpy array
            
        Returns:
            Ảnh đã được xử lý hoàn chỉnh (định dạng OpenCV)
        """
        # Chuyển đổi sang định dạng OpenCV nếu cần
        if isinstance(image, Image.Image):
            cv_image = self.pil_to_cv(image)
        elif isinstance(image, str):
            cv_image = cv2.imread(image)
        else:
            cv_image = image
            
        # Kiểm tra ảnh đầu vào
        if cv_image is None or cv_image.size == 0:
            raise ValueError("Invalid input image")
            
        # Chuẩn hóa ảnh
        cv_image = self.normalize(cv_image)
             
        # Chuyển sang PIL, điều chỉnh DPI và chuyển lại OpenCV
        pil_image = self.cv_to_pil(cv_image)
        pil_image = self.set_image_dpi(pil_image)
        cv_image = self.pil_to_cv(pil_image)
        
        # Loại bỏ nhiễu
        cv_image = self.remove_noise(cv_image)
        
        # Chuyển sang ảnh xám
        cv_image = self.get_grayscale(cv_image)
        
        return cv_image
