import re


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
    
    # Tách boxes theo loại
    item_boxes = [box for box in boxes if int(box.cls) == 0]
    price_boxes = [box for box in boxes if int(box.cls) == 6]
    quantity_boxes = [box for box in boxes if int(box.cls) == 7]
    other_boxes = [box for box in boxes if int(box.cls) not in [0, 6, 7]]
    
    # Nếu không có item boxes, trả về kết quả rỗng hoặc xử lý khác
    if not item_boxes:
        return [other_boxes] if other_boxes else []
    
    # Sắp xếp các item boxes theo tọa độ y (từ trên xuống dưới)
    item_boxes.sort(key=lambda box: (box.xyxy[0][1] + box.xyxy[0][3]) / 2)
    
    # Tạo các nhóm kết quả
    result_groups = []
    
    # Xử lý từng item box
    for i in range(len(item_boxes)):
        current_item = item_boxes[i]
        current_y = (current_item.xyxy[0][1] + current_item.xyxy[0][3]) / 2
        
        # Tìm item tiếp theo gần nhất (nếu có)
        next_item_distance = float('inf')
        if i < len(item_boxes) - 1:
            next_item = item_boxes[i + 1]
            next_item_y = (next_item.xyxy[0][1] + next_item.xyxy[0][3]) / 2
            next_item_distance = abs(next_item_y - current_y)
        
        # Tạo nhóm mới với item hiện tại
        group = [current_item]
        
        # Tìm price và quantity phù hợp cho item hiện tại
        matching_price = None
        matching_quantity = None
        
        # Lọc các box price và quantity có y-coordinate không cao hơn item hiện tại
        # và thấp hơn item tiếp theo (nếu có)
        candidate_prices = []
        candidate_quantities = []
        
        for price_box in price_boxes:
            price_y = (price_box.xyxy[0][1] + price_box.xyxy[0][3]) / 2
            # Price phải không cao hơn item hiện tại và thấp hơn item tiếp theo
            if abs(price_y - current_y) <= next_item_distance and (i == len(item_boxes) - 1 or price_y < next_item_y):
                candidate_prices.append(price_box)
        
        for quantity_box in quantity_boxes:
            quantity_y = (quantity_box.xyxy[0][1] + quantity_box.xyxy[0][3]) / 2
            # Quantity phải không cao hơn item hiện tại và thấp hơn item tiếp theo
            if abs(quantity_y - current_y) <= next_item_distance and (i == len(item_boxes) - 1 or quantity_y < next_item_y):
                candidate_quantities.append(quantity_box)
        
        # Nếu có các ứng viên, chọn price và quantity gần nhất với item hiện tại
        if candidate_prices:
            matching_price = min(candidate_prices, 
                               key=lambda box: abs((box.xyxy[0][1] + box.xyxy[0][3])/2 - current_y))
            group.append(matching_price)
        
        if candidate_quantities:
            matching_quantity = min(candidate_quantities, 
                                  key=lambda box: abs((box.xyxy[0][1] + box.xyxy[0][3])/2 - current_y))
            group.append(matching_quantity)
        
        # Nếu có cả price và quantity, tính khoảng cách B giữa chúng
        if matching_price and matching_quantity:
            price_y = (matching_price.xyxy[0][1] + matching_price.xyxy[0][3]) / 2
            quantity_y = (matching_quantity.xyxy[0][1] + matching_quantity.xyxy[0][3]) / 2
            distance_B = abs(price_y - quantity_y)
            
            # Nếu khoảng cách B quá lớn so với khoảng cách A (giữa các item), 
            # có thể loại bỏ price hoặc quantity không phù hợp
            if next_item_distance != float('inf') and distance_B > next_item_distance * 0.8:
                # Loại bỏ box xa nhất so với item
                price_distance = abs(price_y - current_y)
                quantity_distance = abs(quantity_y - current_y)
                
                if price_distance > quantity_distance:
                    group.remove(matching_price)
                else:
                    group.remove(matching_quantity)
        
        result_groups.append(group)
    
    # Thêm các box khác vào kết quả
    if other_boxes:
        result_groups.append(other_boxes)
    
    return result_groups
