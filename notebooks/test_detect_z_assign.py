"""
Stage 4: Zone Assignment
- Loads zone polygons from JSON config
- Detects persons using YOLO
- Checks if foot point is inside any zone polygon
- Occupied zone = green outline, Unoccupied zone = translucent red fill
- Clean view: foot dots + zone polygons + zone labels only
"""
"""
Stage 4: Zone Assignment (Modular)
Reusable functions: load_zones, detect_people, assign_zones, draw_zones
"""

import cv2
import json
import numpy as np
from ultralytics import YOLO
import os
import sys
from typing import Dict, List, Tuple, Optional

# from test_detect_z_assign import (
#     load_zones,
#     detect_people,
# )


# ------------------ Reusable functions ------------------

def load_zones(json_path: str) -> Dict[str, np.ndarray]:
    """
    Load zone polygons from a JSON file.
    JSON format: {zone_name: {top_left: [x,y], top_right, bottom_right, bottom_left}}
    Returns: dict mapping zone_name -> numpy array of points (4x2) in correct order.
    """
    with open(json_path, 'r') as f:
        zones_data = json.load(f)

    zone_polygons = {}
    for zone_name, coords in zones_data.items():
        pts = np.array([
            coords["top_left"],
            coords["top_right"],
            coords["bottom_right"],
            coords["bottom_left"]
        ], dtype=np.int32)
        zone_polygons[zone_name] = pts
    return zone_polygons


def detect_people(model: YOLO, frame: np.ndarray, conf: float = 0.25) -> List[Tuple[int, int, int, int]]:
    """
    Run YOLO detection and return a list of bounding boxes for persons (class 0).
    Each element: (x1, y1, x2, y2)
    """
    results = model(frame, conf=conf)
    persons = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:   # person class (COCO dataset)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append((x1, y1, x2, y2))
    return persons


def assign_zones(persons: List[Tuple[int, int, int, int]],
                 zone_polygons: Dict[str, np.ndarray]) -> Dict[str, bool]:
    """
    Determine which zones are occupied by at least one person.
    Foot point = bottom-centre of the bounding box.
    Uses cv2.pointPolygonTest for containment check.
    Returns: dict zone_name -> occupied (True/False)
    """
    occupied = {name: False for name in zone_polygons}

    for (x1, y1, x2, y2) in persons:
        foot_x = (x1 + x2) // 2
        foot_y = y2   # bottom edge
        for zone_name, polygon in zone_polygons.items():
            if cv2.pointPolygonTest(polygon, (float(foot_x), float(foot_y)), False) >= 0:
                occupied[zone_name] = True
                break   # a person can only occupy one zone at a time
    return occupied


def draw_zones(frame: np.ndarray,
               zone_polygons: Dict[str, np.ndarray],
               occupied: Dict[str, bool],
               draw_foot_dots: bool = True) -> np.ndarray:
    """
    Draw zones on the frame:
    - occupied: green outline
    - unoccupied: translucent red fill + red outline
    Optionally draw foot points of detected persons (provided persons list).
    Returns the annotated frame.
    """
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    RED_FILL = (0, 0, 255)
    # Draw zones
    for zone_name, polygon in zone_polygons.items():
        if occupied.get(zone_name, False):
            # occupied
            cv2.polylines(frame, [polygon], True, GREEN, 2)
            status_text = f"{zone_name}: ON"
            text_color = GREEN
        else:
            # unoccupied
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], RED_FILL)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            cv2.polylines(frame, [polygon], True, RED, 2)
            status_text = f"{zone_name}: OFF"
            text_color = RED

        # Label at top-left corner of polygon
        label_x = polygon[0][0] + 5
        label_y = polygon[0][1] + 25
        cv2.putText(frame, status_text, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    return frame


def assign_people_to_zones(persons, zone_polygons):
    """
    Assign each detected person to a zone.

    Returns:
        assignments: list of zone names (or None if outside all zones)

    Example:
        ["zone_1", "zone_1", "zone_3", None]
    """

    assignments = []

    for (x1, y1, x2, y2) in persons:
        foot_x = (x1 + x2) // 2
        foot_y = y2

        assigned_zone = None

        for zone_name, polygon in zone_polygons.items():
            if cv2.pointPolygonTest(
                polygon,
                (float(foot_x), float(foot_y)),
                False
            ) >= 0:
                assigned_zone = zone_name
                break

        assignments.append(assigned_zone)

    return assignments

# ------------------ Main script (when run directly) ------------------

def main():
    # ---- Configuration ----
    MODEL_PATH = '../models/yolov8n.pt'
    if len(sys.argv) < 2:
        print("Usage: python stage4_zone_assignment.py <video_filename>")
        sys.exit(1)

    video_name = sys.argv[1]
    VIDEO_PATH = os.path.join("././videos/raw", video_name)
    json_name = f"zones_{os.path.splitext(video_name)[0]}.json"
    ZONES_JSON = os.path.join("././config", json_name)
    FPS_TARGET = 5

    # ---- Load resources ----
    zone_polygons = load_zones(ZONES_JSON)
    model = YOLO(MODEL_PATH)
    print("Model loaded")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open video {VIDEO_PATH}")
        exit()

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(video_fps / FPS_TARGET))
    print(f"Video FPS: {video_fps:.0f}, Processing every {frame_skip} frame(s) -> ~{video_fps/frame_skip:.0f} FPS")

    frame_count = 0
    print("Video opened. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break

        frame_count += 1
        # Optional frame skipping – uncomment if needed
        if frame_count % frame_skip != 0:
            continue

        # ---- Modular pipeline ----
        persons = detect_people(model, frame)
        occupied = assign_zones(persons, zone_polygons)

        # Optionally draw foot dots for each person
        for (x1, y1, x2, y2) in persons:
            foot_x = (x1 + x2) // 2
            foot_y = y2
            cv2.circle(frame, (foot_x, foot_y), 5, (0, 0, 255), -1)

        frame = draw_zones(frame, zone_polygons, occupied)

        cv2.imshow("Zone Assignment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()





















# import cv2
# import json
# import numpy as np
# from ultralytics import YOLO
# import os
# import sys

# # ==================== CONFIGURATION ====================
# MODEL_PATH = '../models/yolov8n.pt'                           # YOLO model

# # Require a video filename argument
# if len(sys.argv) < 2:
#     print("Usage: python test1_detect.py <video_filename>")
#     sys.exit(1)
# # the video names are: test_0.mp4, test_1.mp4, test_2.mp4, library_0.mp4, market_0.mp4, market_1.mp4, market_2.mp4, office_0.mp4

# # ==================== CONFIGURATION ====================
# video_name = sys.argv[1]
# VIDEO_PATH = os.path.join("././videos/raw", video_name)

# # Match output JSON name to video name automatically
# json_name = f"zones_{os.path.splitext(video_name)[0]}.json"
# ZONES_JSON = os.path.join("././config", json_name)
                       

# FPS_TARGET = 5                                                # Frames to process per second (5 = every 6th frame)
# # =======================================================

# # Load zone data
# with open(ZONES_JSON, 'r') as f:
#     zones_data = json.load(f)

# # Convert JSON points to the correct order for cv2.pointPolygonTest
# # JSON has: top_left, top_right, bottom_right, bottom_left
# # cv2 expects contour points in order (any consistent order works)
# zone_polygons = {}
# for zone_name, coords in zones_data.items():
#     pts = np.array([
#         coords["top_left"],
#         coords["top_right"],
#         coords["bottom_right"],
#         coords["bottom_left"]
#     ], dtype=np.int32)
#     zone_polygons[zone_name] = pts

# # Colors (BGR)
# GREEN = (0, 255, 0)
# RED = (0, 0, 255)
# RED_FILL = (0, 0, 255)      # For translucent overlay
# WHITE = (255, 255, 255)

# # Load model and video
# print(f"Model exists: {os.path.exists(MODEL_PATH)}")
# print(f"Video exists: {os.path.exists(VIDEO_PATH)}")

# model = YOLO(MODEL_PATH)
# print("Model loaded successfully")

# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     print(f"ERROR: Could not open video {VIDEO_PATH}")
#     exit()

# # Get video FPS to calculate frame skip
# video_fps = cap.get(cv2.CAP_PROP_FPS)
# frame_skip = max(1, int(video_fps / FPS_TARGET))
# print(f"Video FPS: {video_fps:.0f}, Processing every {frame_skip} frame(s) → ~{video_fps/frame_skip:.0f} FPS")

# # Zone occupancy tracking (start all as occupied/True)
# zone_occupied = {zone_name: True for zone_name in zone_polygons}

# frame_count = 0
# print("Video opened. Press 'q' to quit.")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Video ended.")
#         break

#     frame_count += 1

#     # Skip frames for performance
#     # if frame_count % frame_skip != 0:
#         # continue

#     # --- Reset all zones to unoccupied ---
#     for zone_name in zone_occupied:
#         zone_occupied[zone_name] = False

#     # --- Run YOLO detection ---
#     results = model(frame)

#     # --- Process detections ---
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             cls = int(box.cls[0])

#             if cls == 0:  # Person
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])

#                 # Foot position
#                 center_x = (x1 + x2) // 2
#                 bottom_y = y2
#                 foot_point = (center_x, bottom_y)

#                 # Draw foot dot
#                 cv2.circle(frame, foot_point, 5, (0, 0, 255), -1)
#                 # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#                 # Check which zone this person is in
#                 for zone_name, polygon in zone_polygons.items():
#                     # pointPolygonTest: returns >0 if inside, =0 on edge, <0 outside
#                     result_pt = cv2.pointPolygonTest(polygon, (float(center_x), float(bottom_y)), False)
#                     if result_pt >= 0:
#                         zone_occupied[zone_name] = True
#                         break  # Person counted in one zone, move to next person

#     # --- Draw zones ---
#     for zone_name, polygon in zone_polygons.items():
#         if zone_occupied[zone_name]:
#             # Occupied: green outline only
#             cv2.polylines(frame, [polygon], True, GREEN, 2)
#             status_text = f"{zone_name}: ON"
#             text_color = GREEN
#         else:
#             # Unoccupied: translucent red fill + outline
#             overlay = frame.copy()
#             cv2.fillPoly(overlay, [polygon], RED_FILL)
#             cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
#             cv2.polylines(frame, [polygon], True, RED, 2)
#             status_text = f"{zone_name}: OFF"
#             text_color = RED

#         # Zone label (top-left corner of polygon)
#         label_x = polygon[0][0] + 5
#         label_y = polygon[0][1] + 25
#         cv2.putText(frame, status_text, (label_x, label_y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

#     # --- Display ---
#     cv2.imshow("Zone Assignment", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Done!")




