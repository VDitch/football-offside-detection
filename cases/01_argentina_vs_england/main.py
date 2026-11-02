import cv2
import numpy as np
import math
from ultralytics import YOLO 

def draw_triangle(image, top_center_xy, color, size=10):
    x, y = top_center_xy
    triangle_cnt = np.array([
        (x, y - 10), 
        (x - size, y - 10 - size*2), 
        (x + size, y - 10 - size*2) 
    ])
    cv2.drawContours(image, [triangle_cnt], 0, color, -1)

def detect_pitch_lines(image_path, output_path_tracking, output_path_lines, show_gui=True):
    model = YOLO('yolov8m.pt') 
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return

    lines_img = img.copy()
    final_img = img.copy()
    img_height, img_width = img.shape[:2]

    # STEP 1: DETECT PLAYERS
    results = model.track(img, persist=True, classes=[0, 32], conf=0.08)
    player_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    detected_objects = []

    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            detected_objects.append({'box': (x1, y1, x2, y2), 'cls': cls_id})

            if cls_id == 0: 
                pad = 5 
                cv2.rectangle(player_mask, (max(0, x1-pad), max(0, y1-pad)), 
                              (min(img_width, x2+pad), min(img_height, y2+pad)), 255, -1)

    # STEP 2 & 3: FIELD SEGMENTATION & MASKS
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40]); upper_green = np.array([90, 255, 255])
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

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 110, 0]); upper_white = np.array([180, 255, 255]) 
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    edges = cv2.Canny(white_mask, 50, 150)
    masked_edges = cv2.bitwise_and(edges, edges, mask=field_mask_eroded)
    inverted_player_mask = cv2.bitwise_not(player_mask)
    masked_edges = cv2.bitwise_and(masked_edges, masked_edges, mask=inverted_player_mask)

    # ==============================================================================
    # STEP 4: HOUGH DETECTION (STANDARD WHITE LINES)
    # ==============================================================================
    all_lines = [] 
    lines_std = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=150)
    if lines_std is not None:
        for idx, line in enumerate(lines_std):
            x1_l, y1_l, x2_l, y2_l = line[0]
            all_lines.append([x1_l, y1_l, x2_l, y2_l])
            
            cv2.line(lines_img, (x1_l, y1_l), (x2_l, y2_l), (0, 255, 0), 2)
            
            mid_x = int((x1_l + x2_l) / 2)
            mid_y = int((y1_l + y2_l) / 2)
            cv2.putText(lines_img, str(idx), (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # ==============================================================================
    # STEP 4B: GRASS STRIPE (MOWING PATTERN) DETECTION
    # ==============================================================================
    _, lightness, _ = cv2.split(hls)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_l = clahe.apply(lightness)
    masked_l = cv2.bitwise_and(enhanced_l, enhanced_l, mask=field_mask_eroded)
    
    blur_l = cv2.GaussianBlur(masked_l, (5, 5), 0)
    sobelx = cv2.Sobel(blur_l, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    _, grass_edges = cv2.threshold(scaled_sobel, 12, 255, cv2.THRESH_BINARY)
    
    grass_edges = cv2.bitwise_and(grass_edges, grass_edges, mask=inverted_player_mask)
    inverted_white_mask = cv2.bitwise_not(white_mask)
    grass_edges = cv2.bitwise_and(grass_edges, grass_edges, mask=inverted_white_mask)
    grass_edges = cv2.dilate(grass_edges, np.ones((3,3), np.uint8), iterations=1)
    
    grass_lines_matrix = [] 
    grass_lines_std = cv2.HoughLinesP(grass_edges, 1, np.pi/180, threshold=25, minLineLength=40, maxLineGap=250)
    
    if grass_lines_std is not None:
        for line in grass_lines_std:
            gx1, gy1, gx2, gy2 = line[0]
            if gx2 != gx1:
                slope = (gy2 - gy1) / (gx2 - gx1)
                angle = np.abs(np.arctan(slope) * 180 / np.pi)
                if 15 < angle < 80 and slope > 0:
                    grass_lines_matrix.append([gx1, gy1, gx2, gy2])

    # STEP 5: MASTER LINE CALCULATION (Fallback)
    master_slope = 0.4 
    if all_lines:
        max_len = 0
        for line in all_lines:
            lx1, ly1, lx2, ly2 = line
            if lx2 != lx1:
                slope = (ly2 - ly1) / (lx2 - lx1)
                length = np.sqrt((lx2 - lx1)**2 + (ly2 - ly1)**2)
                if 15 < abs(np.arctan(slope)*180/np.pi) < 40 and length > max_len:
                    max_len = length; master_slope = slope

    # ==============================================================================
    # STEP 6: TEAM CLASSIFICATION (Υπολογισμός χωρίς σχεδίαση)
    # ==============================================================================
    ref = []
    team1 = []
    team2 = []

    for obj in detected_objects:
        x1, y1, x2, y2 = obj['box']
        cls_id = obj['cls']
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        dist = cv2.pointPolygonTest(field_polygon, (cx, y2), True)
        if dist < -15.0 or cy > img_height * 0.98 or cx > img_width * 0.98: continue

        if cls_id == 0: # Players
            h, w = y2 - y1, x2 - x1
            top_roi = img[y1 + int(h*0.2) : y1 + int(h*0.7), int(x1 + w*0.2):int(x2 - w*0.2)]
            if top_roi.size == 0: continue
            hsv_roi = cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV)

            mask_red1 = cv2.inRange(hsv_roi, np.array([0, 50, 40]), np.array([15, 255, 255]))
            mask_red2 = cv2.inRange(hsv_roi, np.array([160, 50, 40]), np.array([180, 255, 255]))
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            mask_dark_blue = cv2.inRange(hsv_roi, np.array([90, 30, 10]), np.array([135, 255, 120]))
            mask_white = cv2.inRange(hsv_roi, np.array([0, 0, 180]), np.array([180, 50, 255]))

            s_red = cv2.countNonZero(mask_red)
            s_blue = cv2.countNonZero(mask_dark_blue)
            s_white = cv2.countNonZero(mask_white)

            if s_red > s_blue and s_red > s_white and s_red > 5:
                ref.append([x1, y1, x2, y2])
            elif s_blue > s_red and s_blue > s_white and s_blue > 5:
                team1.append([x1, y1, x2, y2])
            elif s_white > s_red and s_white > s_blue and s_white > 5:
                team2.append([x1, y1, x2, y2])

    # Μεταφορά του index 1 από τον πίνακα ref στον team 2 
    if len(ref) > 1:
        box_to_move = ref.pop(1)
        team2.append(box_to_move)

    # (Οι εντολές σχεδίασης των Bounding Boxes αφαιρέθηκαν εντελώς από εδώ)

    # ==============================================================================
    # STEP 7 & 8: VANISHING POINT INTERPOLATION & DRAW LINES
    # ==============================================================================
    
    # Εντοπισμός των κουτιών για τα οποία θέλουμε να τραβήξουμε γραμμές
    final_line_boxes_t1_3 = [team1[3]] if len(team1) > 3 else []
    final_line_boxes_t2_2 = [team2[2]] if len(team2) > 2 else []
    final_line_boxes_t2_5 = [team2[5]] if len(team2) > 5 else []

    vp = None
    
    # Υπολογισμός Vanishing Point με βάση τις 3 συγκεκριμένες Hough γραμμές
    target_line_indices = [48, 63, 265] 
    valid_lines = []
    
    for i in target_line_indices:
        if i < len(all_lines):
            valid_lines.append(all_lines[i])
            gx1, gy1, gx2, gy2 = all_lines[i]
            cv2.line(lines_img, (gx1, gy1), (gx2, gy2), (0, 255, 255), 3) 
            mid_x, mid_y = int((gx1 + gx2) / 2), int((gy1 + gy2) / 2)
            cv2.putText(lines_img, str(i), (mid_x, mid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if len(valid_lines) >= 2:
        M = []
        for line in valid_lines:
            gx1, gy1, gx2, gy2 = line
            a = gy1 - gy2
            b = gx2 - gx1
            c = gx1 * gy2 - gx2 * gy1
            M.append([a, b, c])
            
        M = np.array(M)
        U, S, Vt = np.linalg.svd(M)
        V = Vt[-1] 
        
        if abs(V[2]) > 1e-6: 
            vp = (V[0] / V[2], V[1] / V[2])
    
    overlay = final_img.copy()

    # Βοηθητική συνάρτηση για τη σχεδίαση των γραμμών 
    def draw_perspective_line(img, box, vp_point, color):
        # Αγκύρωση στο ΚΑΤΩ ΔΕΞΙ άκρο (x2, y2) του Bounding Box του παίκτη
        px, py = box[2], box[3] 
        
        if vp_point is not None:
            vx, vy = vp_point
            if vx != px: 
                m = (vy - py) / (vx - px)
                c_p = py - m * px
                y_start = int(c_p)
                y_end = int(m * img_width + c_p)
                cv2.line(img, (0, y_start), (img_width, y_end), color, 2)
        else:
            c_p = py - master_slope * px 
            cv2.line(img, (0, int(c_p)), (img_width, int(master_slope * img_width + c_p)), color, 2)

    # Σχεδίαση των 3 ζητούμενων γραμμών (1 Μπλε, 2 Λευκές)
    if final_line_boxes_t1_3: 
        draw_perspective_line(overlay, final_line_boxes_t1_3[0], vp, (139, 0, 0)) # Μπλε (Team 1, index 3)
        
    if final_line_boxes_t2_2: 
        draw_perspective_line(overlay, final_line_boxes_t2_2[0], vp, (255, 255, 255)) # Λευκή (Team 2, index 2)
        
    if final_line_boxes_t2_5: 
        draw_perspective_line(overlay, final_line_boxes_t2_5[0], vp, (255, 255, 255)) # Λευκή (Team 2, index 5)

    cv2.addWeighted(overlay, 0.5, final_img, 0.5, 0, final_img)
    
        import os
    os.makedirs(os.path.dirname(os.path.abspath(output_path_lines)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_path_tracking)), exist_ok=True)
    
    cv2.imwrite(output_path_lines, lines_img)
    cv2.imwrite(output_path_tracking, final_img)
    print(f"[+] Saved Hough lines to: {output_path_lines}")
    print(f"[+] Saved tracking result to: {output_path_tracking}")

    if show_gui:
        try:
            scale = 0.6
            cv2.imshow("Hough Lines Result", cv2.resize(lines_img, (int(img_width * scale), int(img_height * scale))))
            cv2.imshow("Final Result", cv2.resize(final_img, (int(img_width * scale), int(img_height * scale))))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Soccer Offside Detection - Argentina vs England")
    parser.add_argument("--input", type=str, default=str(base_dir / "Input" / "frame_2.jpg"), help="Input image path")
    parser.add_argument("--output-tracking", type=str, default=str(base_dir / "Output" / "final_tracking.jpg"), help="Tracking output path")
    parser.add_argument("--output-lines", type=str, default=str(base_dir / "Output" / "hough_lines.jpg"), help="Lines output path")
    parser.add_argument("--no-gui", action="store_true", help="Run headless without cv2.imshow")
    args = parser.parse_args()

    detect_pitch_lines(args.input, args.output_tracking, args.output_lines, show_gui=not args.no_gui)
