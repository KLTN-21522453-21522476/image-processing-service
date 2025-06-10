import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import datetime

import cv2
from PIL import Image
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from config import Config
from models.invoice_data import Item, Invoice
from utils.helper import group_aligned_labels, cleanning_text, cleanning_num, group_invoice_items, save_result_file, handle_overlapping_boxes
from services.image_preprocessing.main import PreprocessImage
from models_config import ModelsConfig

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
        # Use centralized model configuration
        model_file = ModelsConfig.get_model_file(model_name)
        
        # Load YOLO model
        model_path = os.path.join(Config.MODELS_FOLDER, model_file, model_file)
        self.model = YOLO(model_path)
        
        # Load VietOCR model
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cpu'
        self.detector = Predictor(config)
        
        # Store current model info
        self.current_model = model_name or 'default'
        self.current_model_file = model_file
        
        # Define class mapping after loading model
        self.class_field_mapping = {
            'Item': 'item_name',
            'Store_name': 'store_name', 
            'address': 'address',
            'bill_id': 'invoice_id',
            'date': 'created_date',
            'price': 'price',
            'quantity': 'quantity',
            'amount': 'total_amount',
            'total': 'total_amount',
        }
        
        # Classes that need numeric processing
        self.numeric_classes = {'price', 'quantity', 'amount', 'total'}
        
        logger.info(f"Initialized with model: {self.current_model} ({self.current_model_file})")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return ModelsConfig.get_available_model_names()
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different YOLO model.
        
        Args:
            model_name: Name of the new model to use
            
        Returns:
            True if successful, False otherwise
        """
        if not ModelsConfig.is_valid_model(model_name):
            logger.error(f"Invalid model name: {model_name}")
            return False
            
        try:
            model_file = ModelsConfig.get_model_file(model_name)
            model_path = os.path.join(Config.MODELS_FOLDER, model_file, model_file)
            
            self.model = YOLO(model_path)
            self.current_model = model_name
            self.current_model_file = model_file
            
            logger.info(f"Switched to model: {model_name} ({model_file})")
            return True
        except Exception as e:
            logger.error(f"Error switching to model {model_name}: {str(e)}")
            return False
    
    def process_image(self, image_path: str, file_name: str, model_name: str) -> Dict:
        """
        Process an invoice image to extract structured data.
        
        Args:
            image_path: Path to the input image
            file_name: Name for the processed output file
            
        Returns:
            Dictionary containing extracted invoice data
        """
        # Initialize invoice data
        store_name = ""
        created_date = ""
        invoice_id = ""
        address = ""
        total_amount = 0
        items = []
        
        try:
            # Initialize preprocessor and preprocess image
            preprocessor = PreprocessImage()
            img = preprocessor.preprocess_for_detection(image_path)

            # Run YOLO detection
            results = self.model(img, conf=0.2, iou=0.45, max_det=50)
            
            class_names = {}
            for cls_id, cls_name in self.model.names.items():
                class_names[cls_id] = cls_name
            
            # Process results
            for result in results:
                boxes = result.boxes
                filtered_boxes = handle_overlapping_boxes(boxes, class_names, iou_threshold=0.9)
                # Save result image with annotations
                timestamped_filename = save_result_file(file_name, result, "yolo")
                
                # Group aligned bounding boxes
                groups = group_invoice_items(filtered_boxes, class_names=class_names, y_tolerance=70)

                
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
                    if item_data.get('total_amount'):
                        total_amount = item_data['total_amount']
                        
                    # Add item if relevant data exists
                    if any([item_data.get('item_name'), item_data.get('price'), item_data.get('quantity')]):
                        item = Item(
                            item=item_data.get('item_name', ''),
                            price=item_data.get('price', 0),
                            quantity=item_data.get('quantity', 0)
                        )
                        items.append(item)
            
            # Create invoice object
            invoice = Invoice(
                model=model_name,
                fileName=timestamped_filename,
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
        """
        result = {
            'item_name': '',
            'store_name': '',
            'address': '',
            'invoice_id': '',
            'created_date': '',
            'price': 0,
            'quantity': 0,
            'total_amount' : 0
        }
        
        for bbox in group:
            xmin, ymin, xmax, ymax = bbox.xyxy[0]
            cls_id = int(bbox.cls)
            cls_name = self.model.names[cls_id]  # Lấy tên class từ model
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            
            if cls_name not in self.class_field_mapping:
                continue
            # Extract text based on class name
            text = self._extract_text_from_box(img, xmin, ymin, xmax, ymax, cls_id)
            
            if text:
                field_name = self.class_field_mapping[cls_name]
                result[field_name] = text
        
        return result

    def _extract_text_from_box(self, img: Image.Image, xmin: int, ymin: int, xmax: int, ymax: int, cls_id: int) -> Union[str, int]:
        """Extract text from a bounding box in the image."""
        cls_name = self.model.names[cls_id]
        
        # Determine padding based on class name
        offset = 8  # You can customize this based on class name if needed
        
        # Crop image with padding
        img_temp = img.copy()
        cropped_img = img_temp.crop((xmin-offset, ymin-offset, xmax+offset, ymax+offset))
        
        # preprocessor = PreprocessImage()
        # processed_cv_image = preprocessor.process_image_for_ocr(cropped_img)
        # preprocessed_cropped_image = preprocessor.cv_to_pil(processed_cv_image)
        
        # Extract text using OCR
        raw_text = self.detector.predict(cropped_img)
        
        if not raw_text:
            return '' if cls_name not in self.numeric_classes else 0
        
        # Clean text based on class name
        if cls_name in self.numeric_classes:
            return cleanning_num(raw_text, cls_id)  
        else:
            return raw_text