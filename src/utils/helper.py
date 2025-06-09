import re
import os
import datetime
import cv2
from config import Config


# Iou
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    
    return iou

# Overlap labels handle
def handle_overlapping_boxes(boxes, iou_threshold=0.5):
    filtered_boxes = []
    
    for i, box1 in enumerate(boxes):
        keep = True
        for j, box2 in enumerate(boxes):
            if i != j:
                iou = calculate_iou(box1.xyxy[0], box2.xyxy[0])
                if iou > iou_threshold:
                    if box1.conf < box2.conf:
                        keep = False
                        break
        if keep:
            filtered_boxes.append(box1)
    
    return filtered_boxes

# Group labels with the same column (represent for each item)
def group_aligned_labels(boxes, tolerance=15):
    groups = []
    used = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if not used[i]:
            group = [boxes[i]]
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if not used[j]:
                    y1_i = (boxes[i].xyxy[0][1] + boxes[i].xyxy[0][3]) / 2  # Calculate the average y-coordinate of box i
                    y1_j = (boxes[j].xyxy[0][1] + boxes[j].xyxy[0][3]) / 2  # Calculate the average y-coordinate of box j
                    if abs(y1_i - y1_j) <= tolerance:
                        group.append(boxes[j])
                        used[j] = True
            groups.append(group)
    
    return groups

# Clean Data
def cleanning_text(text, cls):
    clean_text = ""
    
    # Remove newline characters
    text_without_space = text.replace('\n', ' ').strip()
    
    # Convert text to lowercase
    final_lower = text_without_space.lower()
    clean_text = re.sub(r'[^a-zA-ZâấầẩẫậăắằẳẵặáàảãạăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĂÂĐÊÔƠƯƵÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ() ]+', '', final_lower)

    return clean_text.strip()

def cleanning_num(num, cls) :
    clean_num = ""
    
    # Remove newline characters
    text_without_space = num.replace('\n', ' ').strip()
    
    # Convert text to lowercase
    final_lower = text_without_space.lower()
    clean_num = re.sub(r'[^0-9]+', '', final_lower)
        
    return clean_num.strip()

def group_invoice_items(boxes, y_tolerance=15, max_horizontal_gap=None):
    if not boxes:
        return []
    
    # Get class indices from the first box's model
    item_class_idx = 0  # Item class index
    price_class_idx = 6  # Price class index
    quantity_class_idx = 7  # Quantity class index
    
    # Separate boxes by type
    item_boxes = [box for box in boxes if int(box.cls) == item_class_idx]
    price_boxes = [box for box in boxes if int(box.cls) == price_class_idx]
    quantity_boxes = [box for box in boxes if int(box.cls) == quantity_class_idx]
    other_boxes = [box for box in boxes if int(box.cls) not in [item_class_idx, price_class_idx, quantity_class_idx]]
    
    # If no item boxes, return other boxes or empty list
    if not item_boxes:
        return [other_boxes] if other_boxes else []
    
    # Sort item boxes by y-coordinate (top to bottom)
    item_boxes.sort(key=lambda box: (box.xyxy[0][1] + box.xyxy[0][3]) / 2)
    
    # Create result groups
    result_groups = []
    
    # Process each item box
    for i, current_item in enumerate(item_boxes):
        current_y = (current_item.xyxy[0][1] + current_item.xyxy[0][3]) / 2
        
        # Find distance to next item (if any)
        next_item_distance = float('inf')
        if i < len(item_boxes) - 1:
            next_item = item_boxes[i + 1]
            next_item_y = (next_item.xyxy[0][1] + next_item.xyxy[0][3]) / 2
            next_item_distance = abs(next_item_y - current_y)
        
        # Create new group with current item
        group = [current_item]
        
        # Find matching price and quantity
        # Use smaller y_tolerance to ensure tighter grouping
        y_tolerance = min(y_tolerance, next_item_distance * 0.5 if next_item_distance < float('inf') else y_tolerance)
        
        # Find closest price box within tolerance
        matching_price = None
        min_price_distance = float('inf')
        for price_box in price_boxes:
            price_y = (price_box.xyxy[0][1] + price_box.xyxy[0][3]) / 2
            distance = abs(price_y - current_y)
            if distance <= y_tolerance and distance < min_price_distance:
                matching_price = price_box
                min_price_distance = distance
        
        # Find closest quantity box within tolerance
        matching_quantity = None
        min_quantity_distance = float('inf')
        for quantity_box in quantity_boxes:
            quantity_y = (quantity_box.xyxy[0][1] + quantity_box.xyxy[0][3]) / 2
            distance = abs(quantity_y - current_y)
            if distance <= y_tolerance and distance < min_quantity_distance:
                matching_quantity = quantity_box
                min_quantity_distance = distance
        
        # Add matching boxes to group
        if matching_price:
            group.append(matching_price)
            # Remove used price box to prevent reuse
            price_boxes.remove(matching_price)
            
        if matching_quantity:
            group.append(matching_quantity)
            # Remove used quantity box to prevent reuse
            quantity_boxes.remove(matching_quantity)
        
        result_groups.append(group)
    
    # Add remaining boxes as a separate group
    remaining_boxes = price_boxes + quantity_boxes + other_boxes
    if remaining_boxes:
        result_groups.append(remaining_boxes)
    
    return result_groups

def save_result_file(file_name: str, image_data, save_format: str = 'cv2') -> str:
    """
    Save a result file with timestamp in the filename.
    
    Args:
        file_name: Original filename
        image_data: Image data to save (can be cv2 image or YOLO result)
        save_format: Format to save in ('cv2' or 'yolo')
        
    Returns:
        str: The timestamped filename that was used
    """
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add timestamp to result filename
    filename_without_ext = os.path.splitext(file_name)[0]
    file_ext = os.path.splitext(file_name)[1] or '.jpg'  # Default to .jpg if no extension
    timestamped_filename = f"{filename_without_ext}_{timestamp}{file_ext}"
    
    # Create full path
    result_path = os.path.join(Config.RESULT_FOLDER, timestamped_filename)
    
    # Ensure the result directory exists
    os.makedirs(Config.RESULT_FOLDER, exist_ok=True)
    
    # Save the file based on format
    try:
        if save_format == 'cv2':
            if isinstance(image_data, type(cv2.imread('dummy.jpg'))):
                # Direct OpenCV image
                cv2.imwrite(result_path, image_data)
            else:
                # YOLO result - get the plotted image
                result_image = image_data.plot()
                cv2.imwrite(result_path, result_image)
        elif save_format == 'yolo':
            # Save raw YOLO result
            image_data.save(result_path)
    except Exception as e:
        print(f"Error saving result file: {str(e)}")
        return file_name  # Return original filename if save fails
        
    return timestamped_filename
