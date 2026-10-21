import cv2
import numpy as np
import math
from ultralytics import YOLO # Import YOLO

def draw_triangle(image, top_center_xy, color, size=10):
    """
    Draws an inverted triangle pointing down at the object.
    top_center_xy: The (x, y) coordinate of the top-middle of the ball's bounding box.
    """
    x, y = top_center_xy
    
    triangle_cnt = np.array( [
        (x, y - 10), 
        (x - size, y - 10 - size*2), 
        (x + size, y - 10 - size*2) 
    ] )
    cv2.drawContours(image, [triangle_cnt], 0, color, -1) # -1 fills the shape


def detect_pitch_lines(image_path, output_path_tracking, output_path_lines):
    model = YOLO('yolov8m.pt') 
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return

    lines_img = img.copy()
    final_img = img.copy()
    img_height, img_width = img.shape[:2]

    # ==========================================
    # STEP 1: DETECT PLAYERS
    # ==========================================
    results = model.track(img, persist=True, classes=[0, 32], conf=0.1)
    player_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    detected_objects = []

    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            detected_objects.append({'box': (x1, y1, x2, y2), 'cls': cls_id})

            if cls_id == 0: 
                pad = 10
                cv2.rectangle(player_mask, (max(0, x1-pad), max(0, y1-pad)), 
                              (min(img_width, x2+pad), min(img_height, y2+pad)), 255, -1)

    # ==========================================
    # STEP 2: FIELD SEGMENTATION
    # ==========================================
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 60, 60]) 
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    field_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    field_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.drawContours(field_mask, [field_polygon], -1, 255, thickness=cv2.FILLED)
    field_mask_eroded = cv2.erode(field_mask, np.ones((10, 10), np.uint8), iterations=3)

    # ==========================================
    # STEP 3: PREPARE MASKS FOR LINES
    # ==========================================
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 130, 0]) 
    upper_white = np.array([180, 255, 255]) 
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    white_mask = cv2.dilate(white_mask, np.ones((3,3), np.uint8), iterations=2)
    edges = cv2.Canny(white_mask, 50, 150)
    
    masked_edges = cv2.bitwise_and(edges, edges, mask=field_mask_eroded)
    inverted_player_mask = cv2.bitwise_not(player_mask)
    masked_edges = cv2.bitwise_and(masked_edges, masked_edges, mask=inverted_player_mask)

    # ==========================================
    # STEP 4: HOUGH DETECTION
    # ==========================================
    all_lines = []
    lines_std = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=40, minLineLength=100, maxLineGap=200)
    if lines_std is not None:
        all_lines.extend(lines_std)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    _, white_sensitive = cv2.threshold(enhanced_gray, 210, 255, cv2.THRESH_BINARY)
    white_sensitive = cv2.bitwise_and(white_sensitive, white_sensitive, mask=field_mask_eroded)
    white_sensitive = cv2.bitwise_and(white_sensitive, white_sensitive, mask=inverted_player_mask)
    edges_sensitive = cv2.Canny(white_sensitive, 30, 100)
    
    lines_sensitive = cv2.HoughLinesP(edges_sensitive, 1, np.pi/180, threshold=20, minLineLength=80, maxLineGap=200)

    if lines_sensitive is not None:
        for line in lines_sensitive:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 15 < angle < 35:
                all_lines.append(line)

    # ==========================================
    # STEP 5: FIND THE EXACT "MASTER" LINE & SAVE ALL HOUGH LINES
    # ==========================================
    master_slope = 0.4 
    max_len = 0
    master_line_coords = None
    
    # Πίνακας για να αποθηκεύσουμε τις μπλε γραμμές (Hough lines)
    valid_blue_lines = []

    if all_lines:
        for line in all_lines:
            if isinstance(line, np.ndarray): line = line.flatten()
            if len(line) == 1: line = line[0]
            x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            
            if y1 < img_height * 0.15: continue
            
            # Προσθήκη της γραμμής στον πίνακα
            valid_blue_lines.append((x1, y1, x2, y2))
            cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 1)

            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                angle_deg = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if 15 < abs(angle_deg) < 35:
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if length > max_len:
                        max_len = length
                        master_slope = slope
                        master_line_coords = (x1, y1, x2, y2)
    
    if master_line_coords is not None:
        mx1, my1, mx2, my2 = master_line_coords
        cv2.line(lines_img, (mx1, my1), (mx2, my2), (0, 0, 255), 4)

    # ==============================================================================
    # STEP 6: TEAM CLASSIFICATION
    # ==============================================================================
    all_team1_boxes = [] 
    all_team2_boxes = [] 

    for obj in detected_objects:
        x1, y1, x2, y2 = obj['box']
        cls_id = obj['cls']
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        # --- GEOMETRIC CHECKS ---
        dist = cv2.pointPolygonTest(field_polygon, (cx, y2), True)
        if dist < -5.0: continue
        if cy > img_height * 0.95: continue 
        if cx < img_width * 0.10 and cy > img_height * 0.60: continue
        
        # --- Remove Cameraman area (far right) ---
        if cx > img_width * 0.93: continue

        if cls_id == 32: 
            pass
        
        elif cls_id == 0: 
            h = y2 - y1
            w = x2 - x1
            center_x1 = int(x1 + w * 0.35)
            center_x2 = int(x2 - w * 0.35)
            if center_x2 <= center_x1: center_x2 = x1 + 1
            
            top_roi = img[y1 : y1 + int(h*0.6), center_x1:center_x2]
            if top_roi.size == 0: continue

            hsv_roi = cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV)

            mask_bg = cv2.inRange(hsv_roi, np.array([35, 40, 40]), np.array([85, 255, 255]))
            total_pixels = top_roi.shape[0] * top_roi.shape[1]
            bg_pixels = cv2.countNonZero(mask_bg)
            player_pixels = total_pixels - bg_pixels
            if player_pixels < 20: continue 

            mask_cyan = cv2.inRange(hsv_roi, np.array([80, 50, 50]), np.array([115, 255, 255]))
            mask_red1 = cv2.inRange(hsv_roi, np.array([0, 50, 50]), np.array([10, 255, 255]))
            mask_red2 = cv2.inRange(hsv_roi, np.array([170, 50, 50]), np.array([180, 255, 255]))
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_yellow = cv2.inRange(hsv_roi, np.array([20, 50, 60]), np.array([35, 255, 255]))
            mask_black = cv2.inRange(hsv_roi, np.array([0, 0, 0]), np.array([180, 255, 60]))

            score_cyan = cv2.countNonZero(mask_cyan) / player_pixels
            score_red = cv2.countNonZero(mask_red) / player_pixels
            score_yellow = cv2.countNonZero(mask_yellow) / player_pixels
            score_black = cv2.countNonZero(mask_black) / player_pixels

            label, color = "", (0,0,0)

            if score_cyan > score_red and score_cyan > 0.05:
                label = "Team 1"
                color = (255, 255, 0)
                all_team1_boxes.append((x1,y1,x2,y2))
            elif score_black > 0.20 and score_black > score_red:
                label = "Ref"
                color = (0, 0, 0)
            elif (score_red > 0.05) or (score_yellow > 0.10):
                label = "Team 2"
                color = (0, 0, 255)
                all_team2_boxes.append((x1,y1,x2,y2))

            # Εμφάνιση bounding box ΜΟΝΟ για τον διαιτητή
            if label == "Ref":
                cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 2)

    # ==========================================
    # STEP 7: GEOMETRIC FILTERING 
    # ==========================================
    lines_team1 = list(all_team1_boxes)
    lines_team2 = list(all_team2_boxes)
    final_line_boxes_t1 = []
    final_line_boxes_t2 = []

    def get_offside_metric(box):
        x1, y1, x2, y2 = box
        return y2 - (master_slope * x2) 

    # --- TEAM 1 (CYAN) ---
    if lines_team1:
        if len(lines_team1) > 1:
            lines_team1.sort(key=lambda b: b[0]) 
            lines_team1.pop() # Remove Goalkeeper
        
        if lines_team1:
            final_line_boxes_t1 = [min(lines_team1, key=get_offside_metric)]

    # --- TEAM 2 (RED) ---
    if len(lines_team2) >= 2:
        sorted_team2 = sorted(lines_team2, key=get_offside_metric)
        final_line_boxes_t2 = [sorted_team2[1]] 
    elif lines_team2:
        final_line_boxes_t2 = [min(lines_team2, key=get_offside_metric)] 

    # ==========================================
    # STEP 8: FINAL DRAW
    # ==========================================
    overlay = final_img.copy()
    alpha = 0.5 

    # Σχεδιασμός Γραμμών Οφσάιντ
    for (x1, y1, x2, y2) in final_line_boxes_t1:
        c_p = y2 - master_slope * x2
        y_start = int(c_p)
        y_end = int(master_slope * img_width + c_p)
        cv2.line(overlay, (0, y_start), (img_width, y_end), (255, 255, 0), 2)
        
        # Bounding box Γαλάζιου Παίκτη
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

    for (x1, y1, x2, y2) in final_line_boxes_t2:
        c_p = y2 - master_slope * x2
        y_start = int(c_p)
        y_end = int(master_slope * img_width + c_p)
        cv2.line(overlay, (0, y_start), (img_width, y_end), (0, 0, 255), 2)
        
        # Bounding box Κόκκινου Παίκτη
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.addWeighted(overlay, alpha, final_img, 1 - alpha, 0, final_img)

    # --- ΝΕΟ: Εμφάνιση ΜΟΝΟ των γραμμών 28 και 54 (χωρίς αριθμούς) ---
    for i, (x1, y1, x2, y2) in enumerate(valid_blue_lines):
        if i in [28, 54]:
            # Σκούρο μπλε χρώμα (BGR) και αυξημένο πάχος (4)
            dark_blue = (139, 0, 0)
            cv2.line(final_img, (x1, y1), (x2, y2), dark_blue, 4) 

    scale = 0.6
    dim = (int(img_width * scale), int(img_height * scale))
    
    cv2.imshow("Red Lines (Debug)", cv2.resize(lines_img, dim))
    cv2.imshow("Final Result", cv2.resize(final_img, dim))
    
    cv2.imwrite(output_path_lines, lines_img)
    cv2.imwrite(output_path_tracking, final_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
detect_pitch_lines('Input/frame_7.jpg', 'Output/final_tracking.jpg', 'Output/hough_lines.jpg')