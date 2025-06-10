import re
import os
import datetime
import cv2
from config import Config

# Overlap labels handle
def handle_overlapping_boxes(boxes, class_names, iou_threshold=0.5):
    """
    Handle overlapping boxes by filtering out smaller boxes when they overlap with larger ones of the same class.
    
    Args:
        boxes: YOLO detection boxes
        class_names: Dict mapping class IDs to class names {0: "Items", 1: "Store_name", ...}
        iou_threshold: IoU threshold for considering boxes as overlapping
    
    Returns:
        List of non-overlapping boxes (largest boxes kept for each overlapping group)
    """
    if not boxes or len(boxes) == 0:
        return []
    
    # Validate class_names parameter
    if not isinstance(class_names, dict):
        raise TypeError(f"class_names must be a dictionary, got {type(class_names)}")
    
    # Helper function to get class name from box
    def get_class_name(box):
        try:
            cls_idx = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
            return class_names.get(cls_idx, f"unknown_{cls_idx}")
        except Exception as e:
            print(f"Error getting class name: {e}")
            return "unknown"
    
    # Helper function to get class ID from box
    def get_class_id(box):
        try:
            return int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
        except Exception as e:
            print(f"Error getting class ID: {e}")
            return -1
    
    # Helper function to get box area
    def get_box_area(box):
        try:
            # Handle different xyxy formats
            if hasattr(box.xyxy, 'shape') and len(box.xyxy.shape) > 1:
                coords = box.xyxy[0] if box.xyxy.shape[0] > 0 else box.xyxy
            else:
                coords = box.xyxy
            
            # Extract coordinates
            if hasattr(coords, 'item'):
                x1, y1, x2, y2 = coords[0].item(), coords[1].item(), coords[2].item(), coords[3].item()
            elif hasattr(coords, '__getitem__'):
                x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
            else:
                x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
            
            return (x2 - x1) * (y2 - y1)
        except Exception as e:
            print(f"Error calculating box area: {e}")
            return 0
    
    # Helper function to get box coordinates
    def get_box_coords(box):
        try:
            if hasattr(box.xyxy, 'shape') and len(box.xyxy.shape) > 1:
                coords = box.xyxy[0] if box.xyxy.shape[0] > 0 else box.xyxy
            else:
                coords = box.xyxy
            
            if hasattr(coords, 'item'):
                return [coords[i].item() for i in range(4)]
            elif hasattr(coords, '__getitem__'):
                return [float(coords[i]) for i in range(4)]
            else:
                return [float(coords[i]) for i in range(4)]
        except Exception as e:
            print(f"Error getting box coordinates: {e}")
            return [0, 0, 0, 0]
    
    # Helper function to calculate IoU between two boxes
    def calculate_iou(box1, box2):
        try:
            coords1 = get_box_coords(box1)
            coords2 = get_box_coords(box2)
            
            x1_1, y1_1, x2_1, y2_1 = coords1
            x1_2, y1_2, x2_2, y2_2 = coords2
            
            # Calculate intersection area
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # Calculate union area
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - intersection_area
            
            if union_area <= 0:
                return 0.0
            
            return intersection_area / union_area
        except Exception as e:
            print(f"Error calculating IoU: {e}")
            return 0.0
    
    # Group boxes by class ID and class name
    class_groups = {}
    for box in boxes:
        class_id = get_class_id(box)
        class_name = get_class_name(box)
        
        # Use both class_id and class_name as key to handle cases where different IDs map to same name
        key = (class_id, class_name)
        
        if key not in class_groups:
            class_groups[key] = []
        class_groups[key].append(box)
    
    print(f"Found {len(class_groups)} different classes:")
    for (class_id, class_name), group_boxes in class_groups.items():
        print(f"  Class {class_id} ({class_name}): {len(group_boxes)} boxes")
    
    # Process each class group separately
    filtered_boxes = []
    
    for (class_id, class_name), group_boxes in class_groups.items():
        if len(group_boxes) == 1:
            # Only one box in this class, keep it
            filtered_boxes.extend(group_boxes)
            continue
        
        # Sort boxes by area (largest first)
        group_boxes.sort(key=get_box_area, reverse=True)
        
        # Keep track of boxes to keep
        boxes_to_keep = []
        
        for current_box in group_boxes:
            # Check if current box overlaps significantly with any already kept box
            should_keep = True
            
            for kept_box in boxes_to_keep:
                iou = calculate_iou(current_box, kept_box)
                if iou > iou_threshold:
                    # Current box overlaps with a larger box that's already kept
                    should_keep = False
                    print(f"Removing overlapping box (IoU: {iou:.3f}) for class {class_name}")
                    break
            
            if should_keep:
                boxes_to_keep.append(current_box)
        
        print(f"Class {class_name}: kept {len(boxes_to_keep)} out of {len(group_boxes)} boxes")
        filtered_boxes.extend(boxes_to_keep)
    
    print(f"Total boxes after filtering: {len(filtered_boxes)} (was {len(boxes)})")
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

def group_invoice_items(boxes, class_names, y_tolerance=15, max_horizontal_gap=None):
    """
    Group invoice items based on class names and spatial proximity.
    
    Args:
        boxes: YOLO detection boxes
        class_names: Dict mapping class IDs to class names {0: "Items", 1: "Store_name", ...}
        y_tolerance: Y-axis tolerance for grouping
        max_horizontal_gap: Maximum horizontal gap (unused for now)
    """
    if not boxes:
        return []
    
    # Validate class_names parameter
    if not isinstance(class_names, dict):
        raise TypeError(f"class_names must be a dictionary, got {type(class_names)}")
    
    # Define target class names to look for (adjust these to match your actual class names)
    item_class_names = ["Items", "Item", "items", "item"]  # Flexible matching
    price_class_names = ["prices", "price", "Price", "Prices"]
    quantity_class_names = ["quantitys", "quantity", "quantities", "qty", "Qty"]
    
    # Helper function to get class name from box
    def get_class_name(box):
        try:
            cls_idx = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
            return class_names.get(cls_idx, f"unknown_{cls_idx}")
        except Exception as e:
            print(f"Error getting class name: {e}")
            return "unknown"
    
    # Helper function to categorize boxes
    def categorize_box(box):
        class_name = get_class_name(box)
        
        if class_name in item_class_names:
            return 'item'
        elif class_name in price_class_names:
            return 'price'
        elif class_name in quantity_class_names:
            return 'quantity'
        else:
            return 'other'
    
    # Helper function to get y coordinate from box
    def get_y_center(box):
        try:
            # Handle different xyxy formats
            if hasattr(box.xyxy, 'shape') and len(box.xyxy.shape) > 1:
                coords = box.xyxy[0] if box.xyxy.shape[0] > 0 else box.xyxy
            else:
                coords = box.xyxy
            
            # Extract coordinates
            if hasattr(coords, 'item'):
                y1 = coords[1].item()
                y2 = coords[3].item()
            elif hasattr(coords, '__getitem__'):
                y1 = float(coords[1])
                y2 = float(coords[3])
            else:
                y1 = float(coords[1])
                y2 = float(coords[3])
                
            return (y1 + y2) / 2
        except Exception as e:
            print(f"Error getting y center: {e}")
            return 0
    
    # Separate boxes by category
    item_boxes = [box for box in boxes if categorize_box(box) == 'item']
    price_boxes = [box for box in boxes if categorize_box(box) == 'price']
    quantity_boxes = [box for box in boxes if categorize_box(box) == 'quantity']
    other_boxes = [box for box in boxes if categorize_box(box) == 'other']
    
    # Debug: Print detected classes and their categories
    detected_classes = {}
    for box in boxes:
        class_name = get_class_name(box)
        category = categorize_box(box)
        if class_name not in detected_classes:
            detected_classes[class_name] = category
    
    print(f"Detected classes and categories: {detected_classes}")
    print(f"Item boxes: {len(item_boxes)}, Price boxes: {len(price_boxes)}, Quantity boxes: {len(quantity_boxes)}, Other boxes: {len(other_boxes)}")
    
    # If no item boxes, return other boxes or empty list
    if not item_boxes:
        return [other_boxes] if other_boxes else []
    
    # Sort item boxes by y-coordinate (top to bottom)
    item_boxes.sort(key=lambda box: get_y_center(box))
    
    # Create result groups
    result_groups = []
    
    # Process each item box
    for i, current_item in enumerate(item_boxes):
        current_y = get_y_center(current_item)
        
        # Find distance to next item (if any)
        next_item_distance = float('inf')
        if i < len(item_boxes) - 1:
            next_item = item_boxes[i + 1]
            next_item_y = get_y_center(next_item)
            next_item_distance = abs(next_item_y - current_y)
        
        # Create new group with current item
        group = [current_item]
        
        # Use adaptive tolerance based on distance to next item
        current_y_tolerance = min(y_tolerance, next_item_distance * 0.5 if next_item_distance < float('inf') else y_tolerance)
        
        # Find closest price box within tolerance
        matching_price = None
        min_price_distance = float('inf')
        for price_box in price_boxes:
            price_y = get_y_center(price_box)
            distance = abs(price_y - current_y)
            if distance <= current_y_tolerance and distance < min_price_distance:
                matching_price = price_box
                min_price_distance = distance
        
        # Find closest quantity box within tolerance
        matching_quantity = None
        min_quantity_distance = float('inf')
        for quantity_box in quantity_boxes:
            quantity_y = get_y_center(quantity_box)
            distance = abs(quantity_y - current_y)
            if distance <= current_y_tolerance and distance < min_quantity_distance:
                matching_quantity = quantity_box
                min_quantity_distance = distance
        
        # Add matching boxes to group
        if matching_price:
            group.append(matching_price)
            price_boxes.remove(matching_price)
            
        if matching_quantity:
            group.append(matching_quantity)
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
