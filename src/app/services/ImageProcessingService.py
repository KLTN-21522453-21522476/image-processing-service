import os
import json
import cv2
from PIL import Image
import logging
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
from app.utils.Helper import handle_overlapping_boxes, group_aligned_labels, cleanning_text, cleanning_num
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from app.configurations.Config import Config
from app.dtos.InvoiceDataResponse import Item, StoreData, Bill


logging.basicConfig(level=logging.DEBUG)

class ImageProcessingService:
    def __init__(self):
        # Load YOLO model
        model_path = os.path.join(Config.MODELS_FOLDER, "yolo8v6.pt")
        self.model = YOLO(model_path)
        
        # Load VietOCR model
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cpu'
        self.detector = Predictor(config)
          
    def process_image(self, image_path, filename):
        try:
             # Read image and change to RGB
            img = Image.open(image_path)    
            results = self.model.predict(img)
            
            img_copy = img.copy()
                    
            # Store data
            storeData = StoreData("", [])
            bill = Bill(filename, [])
            itemList = []  
            # Extract bounding boxes from YOLO's results
            for result in results:
                annotator = Annotator(img_copy)
                bboxes = handle_overlapping_boxes(result.boxes)
                logging.debug(bboxes)
                groups = group_aligned_labels(bboxes, 50)
                
                for group in groups:     
                    item = Item("", None, None)
                                           
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
                                    item._item = cleanning_text(itemNameRaw, cls)
                                else:
                                    item._item = ""
                                    continue
                            
                            case 1:
                                offset = int(4)
                                b = [xmin-offset, ymin-offset, xmax+offset, ymax+offset]
                                imgTemp = img.copy()
                                cropped_img = imgTemp.crop((xmin-offset, ymin-offset, xmax+offset, ymax+offset))                           
                                storeNameRaw = self.detector.predict(cropped_img)
                                if storeNameRaw:
                                    storeData._storeName = cleanning_text(storeNameRaw, cls)
                                else:
                                    storeData._storeName = "Error"

                            case 2:
                                offset = int(4)
                                b = [xmin-offset, ymin-offset, xmax+offset, ymax+offset]
                                imgTemp = img.copy()
                                cropped_img = imgTemp.crop((xmin-offset, ymin-offset, xmax+offset, ymax+offset))   
                                priceValueRaw = self.detector.predict(cropped_img)
                                if priceValueRaw:
                                    item._price = cleanning_num(priceValueRaw, cls)
                                else:
                                    item._price = None
            
                            case 3:
                                offset = int(10)
                                b = [xmin-offset, ymin-offset, xmax+offset, ymax+offset]
                                imgTemp = img.copy()
                                cropped_img = imgTemp.crop((xmin-offset, ymin-offset, xmax+offset, ymax+offset))   
                                quantityValueRaw = self.detector.predict(cropped_img)
                                
                                if quantityValueRaw:
                                    item._quantity = cleanning_num(quantityValueRaw, cls)
                                else:
                                    item._quantity = None
                            
                            case _:
                                continue
                                
                        annotator.box_label(b, self.model.names[cls])
                    itemList.append(item)
                
                
            storeData._items = itemList
            bill._storeData = storeData
            result_json = json.dumps(bill.to_dict(), ensure_ascii=False, indent=4)
            
            # Save the image with bounding boxes
            processed_image_path = os.path.join(Config.RESULT_FOLDER, filename)
            img = annotator.result()  
            cv2.imwrite(processed_image_path, img)

            return result_json
        
        except Exception as e:
            logging.debug(f"{str(e)}")

