import cv2
from ultralytics import YOLO
import os

# Paths - adjust these to match your actual file locations
MODEL_PATH = '../models/yolov8n.pt'  # or '../models/yolov8n.pt' if you moved it
VIDEO_PATH = 'E:/8 sem project/cv_endsem_project/videos/raw/test1.mp4'  # CHANGE THIS to your actual video filename

# Check files exist
print(f"Model exists: {os.path.exists(MODEL_PATH)}")
print(f"Video exists: {os.path.exists(VIDEO_PATH)}")

# Load model
model = YOLO(MODEL_PATH)
print("Model loaded successfully")

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open video {VIDEO_PATH}")
    exit()

print("Video opened. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break
    
    results = model(frame)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            label = f"Person {confidence:.2f}"
            
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Person Detection", frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")




# import cv2
# from ultralytics import YOLO
# import os

# # Load model
# model = YOLO('yolov8n.pt')

# # Load video
# cap = cv2.VideoCapture('../videos/raw/test1.mp4.mp4')  # CHANGE THIS to your video path

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Run YOLO inference
#     results = model(frame)
    
#     # Process detections
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             cls = int(box.cls[0])
#             confidence = float(box.conf[0])
#             label = f"Person {confidence:.2f}"
            
#             if cls == 0:  # person
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#     # Display frame
#     cv2.imshow("Person Detection", frame)
    
#     # Press q to quit, or wait 25ms between frames
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()