"""
Stage 4: Zone Assignment
- Loads zone polygons from JSON config
- Detects persons using YOLO
- Checks if foot point is inside any zone polygon
- Occupied zone = green outline, Unoccupied zone = translucent red fill
- Clean view: foot dots + zone polygons + zone labels only
"""

import cv2
import json
import numpy as np
from ultralytics import YOLO
import os
import sys

# ==================== CONFIGURATION ====================
MODEL_PATH = '../models/yolov8n.pt'                           # YOLO model

# Require a video filename argument
if len(sys.argv) < 2:
    print("Usage: python test1_detect.py <video_filename>")
    sys.exit(1)

# ==================== CONFIGURATION ====================
video_name = sys.argv[1]
VIDEO_PATH = os.path.join("././videos/raw", video_name)

# Match output JSON name to video name automatically
json_name = f"zones_{os.path.splitext(video_name)[0]}.json"
ZONES_JSON = os.path.join("././config", json_name)
                       

FPS_TARGET = 5                                                # Frames to process per second (5 = every 6th frame)
# =======================================================

# Load zone data
with open(ZONES_JSON, 'r') as f:
    zones_data = json.load(f)

# Convert JSON points to the correct order for cv2.pointPolygonTest
# JSON has: top_left, top_right, bottom_right, bottom_left
# cv2 expects contour points in order (any consistent order works)
zone_polygons = {}
for zone_name, coords in zones_data.items():
    pts = np.array([
        coords["top_left"],
        coords["top_right"],
        coords["bottom_right"],
        coords["bottom_left"]
    ], dtype=np.int32)
    zone_polygons[zone_name] = pts

# Colors (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
RED_FILL = (0, 0, 255)      # For translucent overlay
WHITE = (255, 255, 255)

# Load model and video
print(f"Model exists: {os.path.exists(MODEL_PATH)}")
print(f"Video exists: {os.path.exists(VIDEO_PATH)}")

model = YOLO(MODEL_PATH)
print("Model loaded successfully")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open video {VIDEO_PATH}")
    exit()

# Get video FPS to calculate frame skip
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_skip = max(1, int(video_fps / FPS_TARGET))
print(f"Video FPS: {video_fps:.0f}, Processing every {frame_skip} frame(s) → ~{video_fps/frame_skip:.0f} FPS")

# Zone occupancy tracking (start all as occupied/True)
zone_occupied = {zone_name: True for zone_name in zone_polygons}

frame_count = 0
print("Video opened. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    frame_count += 1

    # Skip frames for performance
    # if frame_count % frame_skip != 0:
        # continue

    # --- Reset all zones to unoccupied ---
    for zone_name in zone_occupied:
        zone_occupied[zone_name] = False

    # --- Run YOLO detection ---
    results = model(frame)

    # --- Process detections ---
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])

            if cls == 0:  # Person
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Foot position
                center_x = (x1 + x2) // 2
                bottom_y = y2
                foot_point = (center_x, bottom_y)

                # Draw foot dot
                cv2.circle(frame, foot_point, 5, (0, 0, 255), -1)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Check which zone this person is in
                for zone_name, polygon in zone_polygons.items():
                    # pointPolygonTest: returns >0 if inside, =0 on edge, <0 outside
                    result_pt = cv2.pointPolygonTest(polygon, (float(center_x), float(bottom_y)), False)
                    if result_pt >= 0:
                        zone_occupied[zone_name] = True
                        break  # Person counted in one zone, move to next person

    # --- Draw zones ---
    for zone_name, polygon in zone_polygons.items():
        if zone_occupied[zone_name]:
            # Occupied: green outline only
            cv2.polylines(frame, [polygon], True, GREEN, 2)
            status_text = f"{zone_name}: ON"
            text_color = GREEN
        else:
            # Unoccupied: translucent red fill + outline
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], RED_FILL)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            cv2.polylines(frame, [polygon], True, RED, 2)
            status_text = f"{zone_name}: OFF"
            text_color = RED

        # Zone label (top-left corner of polygon)
        label_x = polygon[0][0] + 5
        label_y = polygon[0][1] + 25
        cv2.putText(frame, status_text, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # --- Display ---
    cv2.imshow("Zone Assignment", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")



















# import cv2
# from ultralytics import YOLO
# import os

# # Paths - adjust these to match your actual file locations
# MODEL_PATH = '../models/yolov8n.pt'  # or '../models/yolov8n.pt' if you moved it
# VIDEO_PATH = 'E:/8 sem project/cv_endsem_project/videos/raw/test1.mp4'  # CHANGE THIS to your actual video filename

# # Check files exist
# print(f"Model exists: {os.path.exists(MODEL_PATH)}")
# print(f"Video exists: {os.path.exists(VIDEO_PATH)}")

# # Load model
# model = YOLO(MODEL_PATH)
# print("Model loaded successfully")

# # Load video
# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     print(f"ERROR: Could not open video {VIDEO_PATH}")
#     exit()

# print("Video opened. Press 'q' to quit.")

# # stage 2 : storing all stud positions
# person_positions = []

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Video ended.")
#         break
    
#     results = model(frame)
    
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             cls = int(box.cls[0])
#             confidence = float(box.conf[0])
#             label = f"Person {confidence:.2f}"
            
#             if cls == 0:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])

#                 center_x = (x1+x2) // 2     #Finding x coordinate of position of student
#                 bottom_y = y2               # y coordinate of pos of stud
#                 person_position = (center_x, bottom_y) # position of student
#                 # person_positions.append((center_x, bottom_y))

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.circle(frame, (center_x, bottom_y), 5, (0, 0, 255), -1)
#                 cv2.putText(frame, label, (x1, y1 - 10),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 cv2.putText(
#                     frame,
#                     f"({center_x},{bottom_y})",
#                     (center_x + 10, bottom_y),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (0, 255, 0),
#                     1
#                 )
    
#     cv2.imshow("Person Detection", frame)
    
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Done!")
# # print(person_positions)







