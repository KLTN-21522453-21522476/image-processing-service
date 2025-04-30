import os
import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
from PIL import Image
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from config import Config
from models.invoice_data import Item, Invoice
from utils.helper import group_aligned_labels, cleanning_text, cleanning_num, group_invoice_items
from services.preprocess_image import PreprocessImage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModelImageProcessingService:
    """Service for processing invoice images using YOLO and VietOCR models."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the image processing service with specified model.
        
        Args:
            model_name: Name of the YOLO model to use. If None, uses default model.
        """
        # Define available models
        self.available_models = {
            'yolo8': 'yolo8.pt',
            'yolo10': 'yolo10.pt',
            'yolo11': 'yolo11.pt',
            'default': 'yolo11.pt'
        }
        
        # Select model file
        model_file = self.available_models.get(model_name, self.available_models['default'])
        
        # Load YOLO model
        model_path = os.path.join(Config.MODELS_FOLDER, model_file, model_file)
        self.model = YOLO(model_path)
        
        # Load VietOCR model
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cpu'
        self.detector = Predictor(config)
    
    def process_image(self, image_path: str, file_name: str) -> Dict:
        """
        Process an invoice image to extract structured data.
        
        Args:
            image_path: Path to the input image
            file_name: Name for the processed output file
            
        Returns:
            Dictionary containing extracted invoice data
        """
        # Result file path
        processed_image_path = os.path.join(Config.RESULT_FOLDER, file_name)
        
        # Initialize invoice data
        store_name = ""
        created_date = ""
        invoice_id = ""
        address = ""
        total_amount = 0
        items = []
        
        try:
            # Read image
            img = Image.open(image_path)
            
            # Run YOLO detection
            results = self.model(img, conf=0.4, iou=0.45, max_det=50)
            
            # Process results
            for result in results:
                # Save result image
                result.save(filename=processed_image_path)
                
                # Group aligned bounding boxes
                boxes = result.boxes
                groups = group_invoice_items(boxes, 30)
                
                # Process each group of aligned boxes
                for group in groups:
                    item_data = self._process_box_group(group, img)
                    
                    # Update invoice data
                    if item_data.get('store_name'):
                        store_name = item_data['store_name']
                    if item_data.get('created_date'):
                        created_date = item_data['created_date']
                    if item_data.get('invoice_id'):
                        invoice_id = item_data['invoice_id']
                    if item_data.get('address'):
                        address = item_data['address']
                    
                    # Add item if relevant data exists
                    if any([item_data.get('item_name'), item_data.get('price'), item_data.get('quantity')]):
                        item = Item(
                            item=item_data.get('item_name', ''),
                            price=item_data.get('price', 0),
                            quantity=item_data.get('quantity', 0)
                        )
                        total_amount += int(item.price) * int(item.quantity)
                        items.append(item)
            
            # Create invoice object
            invoice = Invoice(
                fileName=file_name,
                storeName=store_name,
                createdDate=created_date,
                id=invoice_id,
                status="",
                approvedBy="",
                submittedBy="",
                items=items,
                address=address,
                totalAmount=total_amount
            )
            
            return invoice.model_dump()
        
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _process_box_group(self, group: List, img: Image.Image) -> Dict:
        """
        Process a group of bounding boxes to extract relevant data.
        
        Args:
            group: List of bounding boxes
            img: Original image
            
        Returns:
            Dictionary with extracted data
        """
        result = {
            'item_name': '',
            'store_name': '',
            'address': '',
            'invoice_id': '',
            'created_date': '',
            'price': 0,
            'quantity': 0
        }
        
        for bbox in group:
            xmin, ymin, xmax, ymax = bbox.xyxy[0]
            cls = int(bbox.cls)
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            
            # Extract text based on class
            text = self._extract_text_from_box(img, xmin, ymin, xmax, ymax, cls)
            
            # Update result based on class
            if cls == 0 and text:  # Item name
                result['item_name'] = text
            elif cls == 1 and text:  # Store name
                result['store_name'] = text
            elif cls == 2 and text:  # Address
                result['address'] = text
            elif cls == 4 and text:  # Invoice ID
                result['invoice_id'] = text
            elif cls == 5 and text:  # Created date
                result['created_date'] = text
            elif cls == 6 and text:  # Price
                result['price'] = text
            elif cls == 7 and text:  # Quantity
                result['quantity'] = text
        
        return result
    
    def _extract_text_from_box(self, img: Image.Image, xmin: int, ymin: int, xmax: int, ymax: int, cls: int) -> Union[str, int]:
        """
        Extract text from a bounding box in the image.
        
        Args:
            img: Original image
            xmin, ymin, xmax, ymax: Bounding box coordinates
            cls: Class of the detected object
            
        Returns:
            Extracted and cleaned text or number
        """
        # Determine padding based on class
        offset = 8 if cls == 0 else 8
        
        # Crop image with padding
        img_temp = img.copy()
        
        
        cropped_img = img_temp.crop((xmin-offset, ymin-offset, xmax+offset, ymax+offset))
        
        preprocessor = PreprocessImage()
        processed_cv_image = preprocessor.process_image_for_ocr(cropped_img)
        preprocessed_cropped_image = preprocessor.cv_to_pil(processed_cv_image)
        
        # Extract text using OCR
        raw_text = self.detector.predict(preprocessed_cropped_image)
        
        if not raw_text:
            return '' if cls not in [6, 7] else 0
        
        # Clean text based on class
        if cls in [6, 7]:  # Price or quantity
            return cleanning_num(raw_text, cls)
        elif cls in [0, 1]:  # Text fields
            return cleanning_text(raw_text, cls)
        else:
            return raw_text

