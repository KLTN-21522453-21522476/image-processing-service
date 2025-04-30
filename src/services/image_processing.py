import os
import cv2
from PIL import Image
import logging
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from config import Config
from models.invoice_data import Item, Invoice
from utils.helper import handle_overlapping_boxes, group_aligned_labels, cleanning_text, cleanning_num

logging.basicConfig(level=logging.DEBUG)

class ImageProcessingService:
    def __init__(self):
        # Load YOLO model
        model_path = os.path.join(Config.MODELS_FOLDER, "yolo8v6.pt", "yolo8v6.pt")
        self.model = YOLO(model_path)
        
        # Load VietOCR model
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cpu'
        self.detector = Predictor(config)
          
    def process_image(self, image_path, file_name):
        store_name = ""
        created_date = ""
        id = ""
        items = []
        try:
             # Read image and change to RGB
            img = Image.open(image_path)    
            results = self.model.predict(img)
            
            img_copy = img.copy()
                    
            # Extract bounding boxes from YOLO's results
            for result in results:
                annotator = Annotator(img_copy)
                bboxes = handle_overlapping_boxes(result.boxes)
                logging.debug(bboxes)
                groups = group_aligned_labels(bboxes, 50)
                
                for group in groups:     
                    item_name = "" 
                    price = 0
                    quantity = 0                       
                    
                    for bbox in group:
                        xmin, ymin, xmax, ymax = bbox.xyxy[0]
                        cls = bbox.cls
                        xmin, ymin, xmax, ymax, cls = int(xmin), int(ymin), int(xmax), int(ymax), int(cls)
                        
                        # Text processing        
                        match cls:
                            case 0:
                                offset = int(8)
                                b = [xmin-offset, ymin-offset, xmax+offset, ymax+offset]
                                imgTemp = img.copy()
                                cropped_img = imgTemp.crop((xmin-offset, ymin-offset, xmax+offset, ymax+offset))
                                itemNameRaw = self.detector.predict(cropped_img)
                                if itemNameRaw:
                                    item_name = cleanning_text(itemNameRaw, cls)
                                else:
                                    continue
                            
                            case 1:
                                offset = int(4)
                                b = [xmin-offset, ymin-offset, xmax+offset, ymax+offset]
                                imgTemp = img.copy()
                                cropped_img = imgTemp.crop((xmin-offset, ymin-offset, xmax+offset, ymax+offset))                           
                                storeNameRaw = self.detector.predict(cropped_img)
                                if storeNameRaw:
                                    store_name = cleanning_text(storeNameRaw, cls)

                            case 2:
                                offset = int(4)
                                b = [xmin-offset, ymin-offset, xmax+offset, ymax+offset]
                                imgTemp = img.copy()
                                cropped_img = imgTemp.crop((xmin-offset, ymin-offset, xmax+offset, ymax+offset))   
                                priceValueRaw = self.detector.predict(cropped_img)
                                if priceValueRaw:
                                    price = cleanning_num(priceValueRaw, cls)
            
                            case 3:
                                offset = int(10)
                                b = [xmin-offset, ymin-offset, xmax+offset, ymax+offset]
                                imgTemp = img.copy()
                                cropped_img = imgTemp.crop((xmin-offset, ymin-offset, xmax+offset, ymax+offset))   
                                quantityValueRaw = self.detector.predict(cropped_img)
                                
                                if quantityValueRaw:
                                    quantity = cleanning_num(quantityValueRaw, cls)
     
                            case _:
                                continue
                                
                        annotator.box_label(b, self.model.names[cls])
                    
                    if item_name != "" or price != 0 or quantity != 0:
                        item = Item(item=item_name, price=price, quantity=quantity)
                        items.append(item)
                
                
            invoice = Invoice(
                fileName=file_name,
                storeName=store_name,
                createdDate="",
                id="",
                status="",
                approvedBy="",
                submittedBy="",
                items=items
            )   
            # Save the image with bounding boxes
            processed_image_path = os.path.join(Config.RESULT_FOLDER, file_name)
            img = annotator.result()  
            cv2.imwrite(processed_image_path, img)

            return invoice.model_dump()
        
        except Exception as e:
            logging.debug(f"{str(e)}")

