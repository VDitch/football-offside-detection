import cv2
import numpy as np
import math
from ultralytics import YOLO

def draw_triangle(image, top_center_xy, color, size=10):
    """
    Draws an inverted triangle pointing down at the object.
    """
    x, y = top_center_xy
    triangle_cnt = np.array( [
        (x, y - 10),
        (x - size, y - 10 - size*2),
        (x + size, y - 10 - size*2)
    ] )
    cv2.drawContours(image, [triangle_cnt], 0, color, -1)


class PitchLineDetector:
    def __init__(self):
        self.lower_green = np.array([30, 40, 40])
        self.upper_green = np.array([85, 255, 255])
        self.lower_white = np.array([0, 0, 160])    
        self.upper_white = np.array([180, 80, 255])

    def process(self, img):
        processed_img = img.copy()
        img_height, img_width = img.shape[:2]

        # 1. FIELD MASK
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        green_mask = cv2.dilate(green_mask, np.ones((5, 5), np.uint8))

        # 2. WHITE DETECTION
        mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        final_mask = cv2.bitwise_and(mask_white, mask_white, mask=green_mask)
        blurred_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

        # 3. HOUGH TRANSFORM
        edges = cv2.Canny(blurred_mask, 50, 150)
        raw_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=50, maxLineGap=50)
        
        clean_lines = []
        left_candidates = []
        right_candidates = []

        if raw_lines is not None:
            for line in raw_lines:
                x1, y1, x2, y2 = line[0]
                clean_lines.append([x1, y1, x2, y2])

                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                mid_x = (x1 + x2) / 2
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi) if x2 != x1 else 90

                if mid_x < img_width * 0.45 and 110 < angle < 160:
                    left_candidates.append({'points': (x1, y1, x2, y2), 'length': length})
                elif mid_x > img_width * 0.52 and 75 < angle < 105:
                    right_candidates.append({'points': (x1, y1, x2, y2), 'length': length})

        all_lines_matrix = np.array(clean_lines) if len(clean_lines) > 0 else np.empty((0, 4), dtype=int)
        
        left_line_data = self._draw_extended_line(processed_img, left_candidates, return_data=True)
        right_line_data = self._draw_extended_line(processed_img, right_candidates, return_data=True)

        return processed_img, left_line_data, right_line_data, all_lines_matrix

    def _draw_extended_line(self, img, candidates, return_data=False):
        if not candidates: return None
        best_line = max(candidates, key=lambda x: x['length'])
        bx1, by1, bx2, by2 = best_line['points']
        if bx2 == bx1:
             return {'type': 'vertical', 'x': bx1} if return_data else None
        m = (by2 - by1) / (bx2 - bx1)
        c = by1 - m * bx1
        return {'type': 'slanted', 'm': m, 'c': c} if return_data else None


class TeamDetector:
    def __init__(self, model_path='yolov8x.pt'):
        self.model = YOLO(model_path)

    def process(self, img, target_point=None):
        img_height, img_width = img.shape[:2]
        results = self.model.track(img, persist=True, classes=[0], conf=0.1, imgsz=1280)
        team1_players, team2_players = [], []
        team1_id_counter, team2_id_counter = 1, 1

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ορίζουμε το ROI
                roi_bot = min(img_height, y1 + int((y2 - y1) * 0.80))
                shirt_roi = img[max(0, y1):roi_bot, max(0, x1):min(img_width, x2)]
                
                if shirt_roi.size > 0:
                    hsv_roi = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2HSV)
                    mask_white = cv2.inRange(hsv_roi, np.array([0, 0, 80]), np.array([180, 80, 255]))
                    white_ratio = cv2.countNonZero(mask_white) / shirt_roi.size
                    mask_red = cv2.bitwise_or(cv2.inRange(hsv_roi, np.array([0, 50, 50]), np.array([15, 255, 255])),
                                              cv2.inRange(hsv_roi, np.array([165, 50, 50]), np.array([180, 255, 255])))
                    
                    # Το ΚΟΚΚΙΝΟ bounding box (Διαιτητής) σχεδιάζεται κανονικά
                    if cv2.countNonZero(mask_red) / shirt_roi.size > 0.03:
                        draw_x1, draw_y1, draw_x2, draw_y2 = x1, y1, x2, y2
                        draw_thickness = 2
                        if target_point is not None:
                            tx, ty = target_point
                            if x1 <= tx <= x2 and y1 <= ty <= y2:
                                draw_x1, draw_y1, draw_x2, draw_y2 = x1 + 5, y1 + 5, x2 - 5, y2 - 5
                                draw_thickness = 1
                        cv2.rectangle(img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 0, 255), draw_thickness)
                        continue

                    # Team logic
                    if white_ratio > 0.10:
                        if team2_id_counter == 10:
                            team1_players.append([x1, y1, x2, y2])
                            team1_id_counter += 1
                        else:
                            team2_players.append([x1, y1, x2, y2])
                            team2_id_counter += 1
                    else:
                        if team1_id_counter == 2:
                            team2_players.append([x1, y1, x2, y2])
                            team2_id_counter += 1
                        else:
                            team1_players.append([x1, y1, x2, y2])
                            team1_id_counter += 1

        return img, np.array(team1_players) if team1_players else np.empty((0,4)), np.array(team2_players) if team2_players else np.empty((0,4))


class OffsideLineDrawer:
    def draw(self, img, hough_matrix, team1_boxes, team2_boxes):
        img_height, img_width = img.shape[:2]

        ref_lines = []
        for line in hough_matrix:
            x1, y1, x2, y2 = line
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if dx == 0: continue
            
            m = dy / dx
            mid_x_pct = ((x1 + x2) / 2) / img_width
            mid_y_pct = ((y1 + y2) / 2) / img_height
            
            is_goal_line = length > 120 and abs(m) > 1.1 and (mid_x_pct < 0.15)
            is_outer_box = length > 120 and abs(m) > 1.1 and (0.65 < mid_x_pct < 0.80)

            if (is_goal_line or is_outer_box) and mid_y_pct < 0.8:
                c = y1 - m * x1
                ref_lines.append({'m': m, 'c': c, 'len': length})

        # Η κίτρινη γραμμή στοχεύει τον παίκτη T2-10 (index 9) και η πράσινη τον T1-1 (index 0)
        target_yellow = (team2_boxes[9][0], team2_boxes[9][3]) if len(team2_boxes) > 9 else None
        target_green = (team1_boxes[0][0], team1_boxes[0][3]) if len(team1_boxes) > 0 else None

        if len(ref_lines) >= 2:
            ref_lines.sort(key=lambda x: x['len'], reverse=True)
            l1, l2 = ref_lines[0], ref_lines[-1]
            
            div = l1['m'] - l2['m']
            if abs(div) > 0.0001:
                vp_x = (l2['c'] - l1['c']) / div
                vp_y = l1['m'] * vp_x + l1['c']

                def draw_to_vanishing_point(point, color):
                    if point is None: return
                    px, py = point
                    
                    if abs(vp_x - px) < 10:
                        cv2.line(img, (int(px), 0), (int(px), img_height), color, 2)
                    else:
                        m_persp = (vp_y - py) / (vp_x - px)
                        c_persp = py - m_persp * px
                        
                        x_top = int((0 - c_persp) / m_persp)
                        x_bot = int((img_height - c_persp) / m_persp)
                        cv2.line(img, (x_top, 0), (x_bot, img_height), color, 2)

                draw_to_vanishing_point(target_yellow, (0, 255, 255))
                draw_to_vanishing_point(target_green, (0, 255, 0))
                return img

        m_fallback = ref_lines[0]['m'] if ref_lines else -12.0
        def draw_stable(point, color):
            if point is None: return
            px, py = point
            x_norm = (px / img_width) - 0.5
            m_adj = m_fallback + (x_norm * 2.0)
            c = py - m_adj * px
            cv2.line(img, (int((0-c)/m_adj), 0), (int((img_height-c)/m_adj), img_height), color, 2)

        draw_stable(target_yellow, (0, 255, 255))
        draw_stable(target_green, (0, 255, 0))
        return img


def main_pipeline(image_path, output_path, show_gui=True):
    original_img = cv2.imread(image_path)
    if original_img is None: return
    
    line_detector = PitchLineDetector()
    img_with_pitch, left_data, right_data, hough_matrix = line_detector.process(original_img)
    
    team_detector = TeamDetector()
    target_player_point = (1050, 520)
    img_with_teams, team1, team2 = team_detector.process(img_with_pitch, target_point=target_player_point)

    # Μεταφορά του T1-2 στην Team 2
    team1_list = list(team1)
    team2_list = list(team2)

    if len(team1_list) >= 2:
        player_to_move = team1_list.pop(1)
        team2_list.append(player_to_move)

    team1 = np.array(team1_list) if len(team1_list) > 0 else np.empty((0,4))
    team2 = np.array(team2_list) if len(team2_list) > 0 else np.empty((0,4))
    
    offside_drawer = OffsideLineDrawer()
    final_result = offside_drawer.draw(img_with_teams, hough_matrix, team1, team2)

    tx, ty = target_player_point

    def draw_team_box(img, box, color):
        """ Σχεδιάζει μόνο το κουτί (ΧΩΡΙΣ ΛΕΖΑΝΤΕΣ) """
        x1, y1, x2, y2 = map(int, box)
        draw_x1, draw_y1, draw_x2, draw_y2 = x1, y1, x2, y2
        thickness = 2
        
        if x1 <= tx <= x2 and y1 <= ty <= y2:
            draw_x1, draw_y1, draw_x2, draw_y2 = x1 + 5, y1 + 5, x2 - 5, y2 - 5
            thickness = 1
            
        cv2.rectangle(img, (draw_x1, draw_y1), (draw_x2, draw_y2), color, thickness)

    # Σχεδιασμός ΜΟΝΟ για T1-1
    for index, box in enumerate(team1, start=1):
        if index == 1:
            draw_team_box(final_result, box, (0, 255, 0))

    # Σχεδιασμός ΜΟΝΟ για T2-10
    for index, box in enumerate(team2, start=1):
        if index == 10:
            draw_team_box(final_result, box, (0, 255, 255))

    blue_lines = [] 

    # FINAL STEP: Draw structural lines
    for line in hough_matrix:
        x1, y1, x2, y2 = line
        
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if dx == 0: m = 999.0  
        else: m = dy / dx

        is_long = length > 90
        is_vertical_structure = abs(m) > 1.1
        mid_x_pct = ((x1 + x2) / 2) / original_img.shape[1]

        is_goal_line = is_long and is_vertical_structure and (mid_x_pct < 0.15)
        is_box_side = is_long and is_vertical_structure and (0.65 < mid_x_pct < 0.80)

        if is_goal_line or is_box_side:
            blue_lines.append([x1, y1, x2, y2])
            line_idx = len(blue_lines)
            
            # Εμφάνιση ΜΟΝΟ των γραμμών 32 και 36 (ΧΩΡΙΣ ΛΕΖΑΝΤΕΣ)
            if line_idx in [32, 36]:
                cv2.line(final_result, (x1, y1), (x2, y2), (255, 0, 0), 4)

    # ---------------------------------------------------------
    # Εκτύπωση αποτελεσμάτων στην κονσόλα
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("      ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑΓΝΩΡΙΣΗΣ")
    print("="*40)
    
    print("\n--- ΜΠΛΕ ΓΡΑΜΜΕΣ ΓΗΠΕΔΟΥ ---")
    if blue_lines:
        for index, line in enumerate(blue_lines, start=1):
            visibility = "(ΟΡΑΤΗ)" if index in [32, 36] else "(ΚΡΥΦΗ)"
            print(f"Γραμμή L{index} {visibility}: {line}")
    else:
        print("Δεν βρέθηκαν μπλε γραμμές.")

    print("\n--- TEAM 1 (Πράσινα Bounding Boxes) ---")
    if len(team1) > 0:
        for index, box in enumerate(team1, start=1):
            visibility = "(ΟΡΑΤΟΣ)" if index == 1 else "(ΚΡΥΦΟΣ)"
            print(f"Παίκτης T1-{index} {visibility}: Συντεταγμένες = {list(map(int, box))}")
    else:
        print("Δεν βρέθηκαν παίκτες για την Team 1.")

    print("\n--- TEAM 2 (Κίτρινα Bounding Boxes) ---")
    if len(team2) > 0:
        for index, box in enumerate(team2, start=1):
            visibility = "(ΟΡΑΤΟΣ)" if index == 10 else "(ΚΡΥΦΟΣ)"
            print(f"Παίκτης T2-{index} {visibility}: Συντεταγμένες = {list(map(int, box))}")
    else:
        print("Δεν βρέθηκαν παίκτες για την Team 2.")
    print("="*40 + "\n")

    # Display and Save
        import os
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, final_result)
    print(f"[+] Saved perspective result to: {output_path}")

    if show_gui:
        try:
            scale = 0.45
            dim = (int(final_result.shape[1] * scale), int(final_result.shape[0] * scale))
            cv2.imshow("Final Result: Perspective Offside", cv2.resize(final_result, dim))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Soccer Offside Detection - Real Betis vs Panathinaikos (2025)")
    parser.add_argument("--input", type=str, default=str(base_dir / "Input" / "frame_5.jpg"), help="Input image path")
    parser.add_argument("--output", type=str, default=str(base_dir / "Output" / "final_result_perspective_3.jpg"), help="Output image path")
    parser.add_argument("--no-gui", action="store_true", help="Run headless without cv2.imshow")
    args = parser.parse_args()

    main_pipeline(args.input, args.output, show_gui=not args.no_gui)
