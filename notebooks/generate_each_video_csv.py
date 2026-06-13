# ------------------------------------------------------------------
# This script processes all videos one by one, and forms csv of respective videos in outputs/metrics folder whose schema is

# schema: video_name; frame_number;	processed_frame; timestamp_sec;	detected_people_count; unassigned_people; total_zones; occupied_zones; zone_1; zone_2; zone_3; zone_4; zone_5; zone_6; 

# Data of all detected things in frame, #f zones, #people in zones, is stored; and every 6th frame's data is stored
# ------------------------------------------------------------------



import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import argparse as agp
import time

# Import reusable functions from your modular stage4 script
from test_detect_z_assign import (
    load_zones,
    detect_people,
    assign_people_to_zones
)

# ==================================================
# CONFIGURATION
# ==================================================

MODEL_NAME = "yolov8s"

parser = agp.ArgumentParser()
parser.add_argument(
    "--model",
    default="yolov8n"
)
args = parser.parse_args()

MODEL_NAME = args.model

MODEL_PATH = f"../models/{MODEL_NAME}.pt"

VIDEOS_DIR = "././videos/raw"
CONFIG_DIR = "././config"

OUTPUT_DIR = f"././outputs/{MODEL_NAME}/metrics"

FPS_TARGET = 5
MAX_ZONES = 6

# ==================================================


def create_metrics_row(
    video_name,
    # model_name,
    frame_number,
    processed_frame,
    timestamp_sec,
    total_zones,
    assignments,
    inference_time
):
    """
    Build one CSV row.
    """

    detected_people_count = len(assignments)

    unassigned_people = sum(
        1 for zone in assignments
        if zone is None
    )

    zone_counts = {}

    for zone in assignments:
        if zone is None:
            continue

        zone_counts[zone] = zone_counts.get(zone, 0) + 1

    occupied_zones = sum(
        1 for count in zone_counts.values()
        if count > 0
    )

    row = {
        "video_name": video_name,
        "model_name": MODEL_NAME,
        "frame_number": frame_number,
        "processed_frame": processed_frame,
        "timestamp_sec": round(timestamp_sec, 2),

        "detected_people_count": detected_people_count,
        "unassigned_people": unassigned_people,

        "total_zones": total_zones,
        "occupied_zones": occupied_zones,

        "inference_time_ms": round(
            inference_time * 1000,
            2
        )
    }

    # Fill zone columns
    for i in range(1, MAX_ZONES + 1):

        zone_name = f"zone_{i}"

        if i <= total_zones:

            row[zone_name] = (
                1 if zone_counts.get(zone_name, 0) > 0
                else 0
            )

        else:
            row[zone_name] = np.nan

    return row


def process_video(
    video_path,
    json_path,
    model
):
    """
    Process one video and return dataframe.
    """

    video_name = os.path.basename(video_path)

    print(f"\nProcessing: {video_name}")

    zone_polygons = load_zones(json_path)

    total_zones = len(zone_polygons)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open {video_name}")
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_skip = max(
        1,
        int(video_fps / FPS_TARGET)
    )

    rows = []

    frame_number = 0
    processed_frame = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Skip frames
        if frame_number % frame_skip != 0:
            continue

        processed_frame += 1

        timestamp_sec = frame_number / video_fps

        start = time.time()

        print("Type of model passed to detect_people:", type(model))
        persons = detect_people(
            model,
            frame
        )

        inference_time = time.time() - start

        assignments = assign_people_to_zones(
            persons,
            zone_polygons
        )

        row = create_metrics_row(
            video_name=video_name,
            # model_name=model,
            frame_number=frame_number,
            processed_frame=processed_frame,
            timestamp_sec=timestamp_sec,
            total_zones=total_zones,
            assignments=assignments,
            inference_time=inference_time
        )

        rows.append(row)

        if processed_frame % 100 == 0:
            print(
                f"  Processed Frames: {processed_frame}"
            )

    cap.release()

    return pd.DataFrame(rows)


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(f"Loading YOLO model ({MODEL_NAME})...")

    model = YOLO(MODEL_PATH)

    print("Model loaded.")
    print(f"Type of model: {type(model)}")

    video_files = sorted([
        f for f in os.listdir(VIDEOS_DIR)
        if f.lower().endswith(".mp4")
    ])

    print(
        f"\nFound {len(video_files)} videos."
    )

    for video_file in video_files:

        video_path = os.path.join(
            VIDEOS_DIR,
            video_file
        )

        json_name = (
            f"zones_{os.path.splitext(video_file)[0]}.json"
        )

        json_path = os.path.join(
            CONFIG_DIR,
            json_name
        )

        if not os.path.exists(json_path):

            print(
                f"Missing JSON: {json_name}"
            )

            continue

        df = process_video(
            video_path,
            json_path,
            model
        )

        if df is None:
            continue

        output_csv = os.path.join(
            OUTPUT_DIR,
            f"{os.path.splitext(video_file)[0]}_frame_metrics.csv"
        )

        df.to_csv(
            output_csv,
            index=False
        )

        print(
            f"Saved: {output_csv}"
        )

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()





















# import os
# import cv2
# import numpy as np
# import pandas as pd
# from ultralytics import YOLO
# import argparse as agp
# import time

# # Import reusable functions from your modular stage4 script
# from test_detect_z_assign import (
#     load_zones,
#     detect_people,
#     assign_people_to_zones
# )

# # ==================================================
# # CONFIGURATION
# # ==================================================

# MODEL_NAME = "yolov8s"

# parser = agp.ArgumentParser()
# parser.add_argument(
#     "--model",
#     default="yolov8n"
# )
# args = parser.parse_args()

# MODEL_NAME = args.model

# MODEL_PATH = f"../models/{MODEL_NAME}.pt"

# VIDEOS_DIR = "././videos/raw"
# CONFIG_DIR = "././config"

# OUTPUT_DIR = f"././outputs/{MODEL_NAME}/metrics"

# FPS_TARGET = 5
# MAX_ZONES = 6

# # ==================================================


# def create_metrics_row(
#     video_name,
#     model_name,
#     frame_number,
#     processed_frame,
#     timestamp_sec,
#     total_zones,
#     assignments
# ):
#     """
#     Build one CSV row.
#     """

#     detected_people_count = len(assignments)

#     unassigned_people = sum(
#         1 for zone in assignments
#         if zone is None
#     )

#     zone_counts = {}

#     for zone in assignments:
#         if zone is None:
#             continue

#         zone_counts[zone] = zone_counts.get(zone, 0) + 1

#     occupied_zones = sum(
#         1 for count in zone_counts.values()
#         if count > 0
#     )

#     row = {
#         "video_name": video_name,
#         "model_name": MODEL_NAME,
#         "frame_number": frame_number,
#         "processed_frame": processed_frame,
#         "timestamp_sec": round(timestamp_sec, 2),

#         "detected_people_count": detected_people_count,
#         "unassigned_people": unassigned_people,

#         "total_zones": total_zones,
#         "occupied_zones": occupied_zones
#     }

#     # Fill zone columns
#     for i in range(1, MAX_ZONES + 1):

#         zone_name = f"zone_{i}"

#         if i <= total_zones:

#             row[zone_name] = (
#                 1 if zone_counts.get(zone_name, 0) > 0
#                 else 0
#             )

#         else:
#             row[zone_name] = np.nan

#     return row


# def process_video(
#     video_path,
#     json_path,
#     model
# ):
#     """
#     Process one video and return dataframe.
#     """

#     video_name = os.path.basename(video_path)

#     print(f"\nProcessing: {video_name}")

#     zone_polygons = load_zones(json_path)

#     total_zones = len(zone_polygons)

#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"Failed to open {video_name}")
#         return None

#     video_fps = cap.get(cv2.CAP_PROP_FPS)

#     frame_skip = max(
#         1,
#         int(video_fps / FPS_TARGET)
#     )

#     rows = []

#     frame_number = 0
#     processed_frame = 0

#     while cap.isOpened():

#         ret, frame = cap.read()

#         if not ret:
#             break

#         frame_number += 1

#         # Skip frames
#         if frame_number % frame_skip != 0:
#             continue

#         processed_frame += 1

#         timestamp_sec = frame_number / video_fps

#         start = time.time()

#         persons = detect_people(
#             model,
#             frame
#         )

#         inference_time = time.time() - start

#         assignments = assign_people_to_zones(
#             persons,
#             zone_polygons
#         )

#         row = create_metrics_row(
#             video_name=video_name,
#             model_name=model_name,
#             frame_number=frame_number,
#             processed_frame=processed_frame,
#             timestamp_sec=timestamp_sec,
#             total_zones=total_zones,
#             assignments=assignments
#         )

#         rows.append(row)

#         if processed_frame % 100 == 0:
#             print(
#                 f"  Processed Frames: {processed_frame}"
#             )

#     cap.release()

#     return pd.DataFrame(rows)


# def main():

#     os.makedirs(
#         OUTPUT_DIR,
#         exist_ok=True
#     )

#     print("Loading YOLO model...")

#     model = YOLO(MODEL_PATH)

#     print("Model loaded.")

#     video_files = sorted([
#         f for f in os.listdir(VIDEOS_DIR)
#         if f.lower().endswith(".mp4")
#     ])

#     print(
#         f"\nFound {len(video_files)} videos."
#     )

#     for video_file in video_files:

#         video_path = os.path.join(
#             VIDEOS_DIR,
#             video_file
#         )

#         json_name = (
#             f"zones_{os.path.splitext(video_file)[0]}.json"
#         )

#         json_path = os.path.join(
#             CONFIG_DIR,
#             json_name
#         )

#         if not os.path.exists(json_path):

#             print(
#                 f"Missing JSON: {json_name}"
#             )

#             continue

#         df = process_video(
#             video_path,
#             json_path,
#             model
#         )

#         if df is None:
#             continue

#         output_csv = os.path.join(
#             OUTPUT_DIR,
#             f"{os.path.splitext(video_file)[0]}_frame_metrics.csv"
#         )

#         df.to_csv(
#             output_csv,
#             index=False
#         )

#         print(
#             f"Saved: {output_csv}"
#         )

#     print("\nAll videos processed.")


# if __name__ == "__main__":
#     main()