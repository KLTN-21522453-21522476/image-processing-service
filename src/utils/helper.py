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
    
    # Bước 1: Nhóm theo vị trí dọc (hàng)
    rows = {}
    for i, box in enumerate(boxes):
        y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
        assigned = False
        
        for row_y in sorted(rows.keys()):
            if abs(y_center - row_y) <= y_tolerance:
                rows[row_y].append(box)
                assigned = True
                break
                
        if not assigned:
            rows[y_center] = [box]
    
    # Bước 2: Xác định hàng chứa mục vs hàng tiêu đề/chân trang
    item_rows = []
    header_footer_rows = []
    
    for y, row_boxes in rows.items():
        # Kiểm tra xem hàng này có chứa các lớp liên quan đến mục (0, 6, 7) không
        classes = [int(box.cls) for box in row_boxes]
        if any(cls in [0, 6, 7] for cls in classes):
            item_rows.append((y, row_boxes))
        else:
            header_footer_rows.append((y, row_boxes))
    
    # Sắp xếp các hàng theo vị trí dọc
    item_rows.sort(key=lambda x: x[0])
    
    # Bước 3: Với mỗi hàng mục, đảm bảo nó có các thành phần cần thiết
    result_groups = []
    
    for _, row_boxes in item_rows:
        # Sắp xếp các box trong hàng theo vị trí ngang
        row_boxes.sort(key=lambda box: box.xyxy[0][0])
        
        # Xác định các thành phần mục theo lớp
        item_name_boxes = [box for box in row_boxes if int(box.cls) == 0]
        price_boxes = [box for box in row_boxes if int(box.cls) == 6]
        quantity_boxes = [box for box in row_boxes if int(box.cls) == 7]
        
        # Tạo nhóm dựa trên các thành phần hiện có
        if item_name_boxes:
            for item_box in item_name_boxes:
                group = [item_box]
                
                # Tìm giá và số lượng gần nhất
                if price_boxes:
                    closest_price = min(price_boxes, 
                                       key=lambda box: abs((box.xyxy[0][0] + box.xyxy[0][2])/2 - 
                                                          (item_box.xyxy[0][0] + item_box.xyxy[0][2])/2))
                    group.append(closest_price)
                
                if quantity_boxes:
                    closest_quantity = min(quantity_boxes, 
                                          key=lambda box: abs((box.xyxy[0][0] + box.xyxy[0][2])/2 - 
                                                             (item_box.xyxy[0][0] + item_box.xyxy[0][2])/2))
                    group.append(closest_quantity)
                
                result_groups.append(group)
        else:
            # Xử lý các hàng không có tên mục
            result_groups.append(row_boxes)
    
    # Thêm thông tin tiêu đề/chân trang dưới dạng các nhóm riêng biệt
    for _, row_boxes in header_footer_rows:
        result_groups.append(row_boxes)
    
    return result_groups
