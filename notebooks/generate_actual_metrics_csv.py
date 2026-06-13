# ------------------------------------------------------------------
# This file generates 20 samples from each video, saves those snapshots to outputs/images folder
# appends the data in actual_metrics.csv: having schema as:
# video_name	frame_number	processed_frame	timestamp_sec	detected_people_count	unassigned_people	total_zones	occupied_zones	zone_1	zone_2	zone_3	zone_4	zone_5	zone_6	Actual_people	actual_zone_1	actual_zone_2	actual_zone_3	actual_zone_4	actual_zone_5	actual_zone_6
# iss csv ka adha data repeated hai (from their respective zone video) and baki manually fill kia jayega (human perspective wala)
# ------------------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# Hardcoded video names (without extension)
VIDEO_NAMES = [
    "library_0",
    "test_0", 
    "test_1",
    "test_2",
    "market_0",
    "market_1",
    "market_2",
    "office_0"
]

NUM_SAMPLES = 20

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def get_metrics_csv_path(video_name: str, model_name: str) -> str:
    """Get path to metrics CSV for a specific video and model."""
    return os.path.join(
        f"E:/8 sem project/cv_endsem_project/outputs/{model_name}/metrics",
        f"{video_name}_frame_metrics.csv"
    )

def stratified_sample(df: pd.DataFrame, n_samples: int = 20) -> pd.DataFrame:
    """Take n_samples evenly spaced rows from the dataframe."""
    total_rows = len(df)
    if total_rows <= n_samples:
        return df.copy()
    indices = np.linspace(0, total_rows - 1, n_samples, dtype=int)
    return df.iloc[indices].copy()

def create_actual_metrics_schema(df: pd.DataFrame, video_name: str) -> pd.DataFrame:
    """
    Add columns for ground truth counts.
    actual_zone_X will store the number of people in that zone.
    Actual_people stores total people in the frame.
    """
    # Add video_name column only if it doesn't exist
    if "video_name" not in df.columns:
        df.insert(0, "video_name", video_name)
    else:
        # If it exists, ensure it has the correct value
        df["video_name"] = video_name
    
    zone_cols = [col for col in df.columns if col.startswith("zone_")]
    
    # Add actual_zone columns if they don't exist
    for zone in zone_cols:
        if f"actual_{zone}" not in df.columns:
            df[f"actual_{zone}"] = -1
    
    # Add Actual_people column if it doesn't exist
    if "Actual_people" not in df.columns:
        df["Actual_people"] = -1

    # Reorder columns: video_name, original zones, then Actual_people, then actual_zones
    ordered = ["video_name"] if "video_name" in df.columns else []
    ordered.extend([c for c in df.columns if c.startswith("zone_") and not c.startswith("actual_")])
    
    if "Actual_people" in df.columns:
        ordered.append("Actual_people")
    
    ordered.extend([c for c in df.columns if c.startswith("actual_")])
    
    # Add any remaining columns not yet ordered
    ordered.extend([c for c in df.columns if c not in ordered])
    
    return df[ordered]

def process_video(video_name: str, model_name: str, actual_metrics_path: str, is_first_video: bool):
    """Process a single video: sample frames and append to actual_metrics.csv."""
    
    # 1. Get metrics CSV path
    metrics_csv = get_metrics_csv_path(video_name, model_name)
    
    if not os.path.exists(metrics_csv):
        print(f"Warning: Metrics CSV not found for {video_name} at {metrics_csv}")
        return False
    
    print(f"\nProcessing {video_name}...")
    print(f"Loading metrics: {metrics_csv}")
    
    # 2. Load and sample the dataframe
    df = pd.read_csv(metrics_csv)
    sampled_df = stratified_sample(df, NUM_SAMPLES)
    print(f"Selected {len(sampled_df)} stratified samples from {video_name}.")
    
    # 3. Create validation table with actual_* columns
    validation_df = create_actual_metrics_schema(sampled_df, video_name)
    
    # 4. Append to actual_metrics.csv
    if not is_first_video and os.path.exists(actual_metrics_path):
        # Append without header for subsequent videos
        validation_df.to_csv(actual_metrics_path, mode='a', index=False, header=False)
    else:
        # Write with header for first video (or if file doesn't exist)
        validation_df.to_csv(actual_metrics_path, mode='a', index=False, header=True)
    
    print(f"Added {len(validation_df)} rows for {video_name}")
    return True

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main(model_name: str):
    """Process all videos for a given model."""
    
    # Set up paths
    actual_metrics_path = f"E:/8 sem project/cv_endsem_project/outputs/{model_name}/metrics/{model_name}_actual_metrics.csv"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(actual_metrics_path), exist_ok=True)
    
    # Remove existing file if it exists to start fresh
    if os.path.exists(actual_metrics_path):
        print(f"Removing existing file: {actual_metrics_path}")
        os.remove(actual_metrics_path)
    
    print(f"Processing model: {model_name}")
    print(f"Output will be saved to: {actual_metrics_path}")
    print(f"Videos to process: {', '.join(VIDEO_NAMES)}")
    
    # Process each video
    first_video = True
    successful_videos = 0
    
    for video_name in VIDEO_NAMES:
        if process_video(video_name, model_name, actual_metrics_path, first_video):
            first_video = False
            successful_videos += 1
    
    print(f"\n{'='*50}")
    print(f"Completed processing for model: {model_name}")
    print(f"Successfully processed {successful_videos} out of {len(VIDEO_NAMES)} videos")
    print(f"Results saved to: {actual_metrics_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate validation samples for all videos using stratified sampling."
    )
    parser.add_argument(
        "--model", 
        required=True,
        help="Model name (e.g., yolov8n, yolov8s, etc.)"
    )
    args = parser.parse_args()
    
    main(args.model)



















# import os
# import cv2
# import argparse
# import json
# import numpy as np
# import pandas as pd
# from typing import Dict

# # ------------------------------------------------------------------
# # CONFIGURATION (adjust paths if needed)
# # ------------------------------------------------------------------
# import argparse as agp
# MODEL_NAME = "yolov8s"

# parser = agp.ArgumentParser()
# parser.add_argument(
#     "--model",
#     default="yolov8n"
# )
# args = parser.parse_args()

# MODEL_NAME = args.model
# CSV_DIR = f"E:/8 sem project/cv_endsem_project/outputs/{MODEL_NAME}/metrics"
# IMAGE_DIR = r"E:\8 sem project\cv_endsem_project\outputs\images"
# ACTUAL_METRICS_CSV = f"E:/8 sem project/cv_endsem_project/outputs/{MODEL_NAME}/metrics/actual_metrics.csv"
# NUM_SAMPLES = 20

# # ------------------------------------------------------------------
# # HELPER FUNCTIONS
# # ------------------------------------------------------------------
# def get_metrics_csv_path(video_path: str) -> str:
#     """Convert video path -> metrics CSV path."""
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     return os.path.join(CSV_DIR, f"{video_name}_frame_metrics.csv")

# def stratified_sample(df: pd.DataFrame, n_samples: int = 20) -> pd.DataFrame:
#     """Take n_samples evenly spaced rows from the dataframe."""
#     total_rows = len(df)
#     if total_rows <= n_samples:
#         return df.copy()
#     indices = np.linspace(0, total_rows - 1, n_samples, dtype=int)
#     return df.iloc[indices].copy()

# def create_actual_metrics_schema(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add columns for ground truth counts.
#     actual_zone_X will store the number of people in that zone.
#     Actual_people stores total people in the frame.
#     """
#     zone_cols = [col for col in df.columns if col.startswith("zone_")]
#     for zone in zone_cols:
#         df[f"actual_{zone}"] = -1
#     df["Actual_people"] = -1

#     # Reorder columns: original zones, then Actual_people, then actual_zones
#     ordered = [c for c in df.columns if not c.startswith("actual_") and c != "Actual_people"]
#     actual_zones = [c for c in df.columns if c.startswith("actual_") and c != "Actual_people"]
#     ordered.append("Actual_people")
#     ordered.extend(actual_zones)
#     return df[ordered]

# def draw_zones_simple(frame: np.ndarray,
#                       zone_polygons: Dict[str, np.ndarray],
#                       occupied: Dict[str, bool] = None) -> np.ndarray:
#     """
#     Draw zones on frame.
#     If occupied dict is given, green = occupied, red + fill = unoccupied.
#     Without occupied dict, all zones are drawn as translucent red (unoccupied style).
#     """
#     if occupied is None:
#         occupied = {}
#     GREEN = (0, 255, 0)
#     RED = (0, 0, 255)
#     RED_FILL = (0, 0, 255)

#     for zone_name, polygon in zone_polygons.items():
#         if occupied.get(zone_name, False):
#             # occupied -> green outline
#             cv2.polylines(frame, [polygon], True, GREEN, 2)
#             status = f"{zone_name}: ON"
#             color = GREEN
#         else:
#             # unoccupied -> red fill + red outline
#             overlay = frame.copy()
#             cv2.fillPoly(overlay, [polygon], RED_FILL)
#             cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
#             cv2.polylines(frame, [polygon], True, RED, 2)
#             status = f"{zone_name}: OFF"
#             color = RED

#         # Label near the first point of the polygon
#         x = polygon[0][0] + 5
#         y = polygon[0][1] + 25
#         cv2.putText(frame, status, (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#     return frame

# def save_sample_images(video_path: str,
#                        sampled_df: pd.DataFrame,
#                        zone_polygons: Dict[str, np.ndarray]) -> None:
#     """Extract sampled frames, draw zones, save as images."""
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {video_path}")

#     for _, row in sampled_df.iterrows():
#         frame_num = int(row["frame_number"])
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Warning: could not read frame {frame_num}")
#             continue

#         annotated = frame.copy()
#         # No occupancy info during validation -> all zones drawn as unoccupied
#         draw_zones_simple(annotated, zone_polygons)

#         video_name = os.path.splitext(os.path.basename(video_path))[0]
#         img_name = f"{video_name}_frame_{frame_num}.jpg"
#         img_path = os.path.join(IMAGE_DIR, img_name)
#         cv2.imwrite(img_path, annotated)
#         print(f"Saved {img_path}")

#     cap.release()

# # ------------------------------------------------------------------
# # MAIN
# # ------------------------------------------------------------------
# def main(video_path: str, zones_json: str):
#     # 1. Load metrics CSV
#     metrics_csv = get_metrics_csv_path(video_path)
#     if not os.path.exists(metrics_csv):
#         raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

#     print(f"Loading metrics: {metrics_csv}")
#     df = pd.read_csv(metrics_csv)
#     sampled_df = stratified_sample(df, NUM_SAMPLES)
#     print(f"Selected {len(sampled_df)} stratified samples.")

#     # 2. Load zones from JSON
#     if not os.path.exists(zones_json):
#         raise FileNotFoundError(f"Zones JSON not found: {zones_json}")
#     with open(zones_json, 'r') as f:
#         zones_data = json.load(f)

#     # zone_polygons = {}
#     # for zone_name, coords in zones_data.items():
#         # pts = np.array([
#         #     coords["top_left"],
#         #     coords["top_right"],
#         #     coords["bottom_right"],
#         #     coords["bottom_left"]
#         # ], dtype=np.int32)
#         # zone_polygons[zone_name] = pts

#     # 3. Save annotated images
#     # os.makedirs(IMAGE_DIR, exist_ok=True)
#     # save_sample_images(video_path, sampled_df, zone_polygons)

#     # 4. Create validation table
#     validation_df = create_actual_metrics_schema(sampled_df.copy())

#     # 5. Append to actual_metrics.csv
#     if os.path.exists(ACTUAL_METRICS_CSV):
#         validation_df.to_csv(ACTUAL_METRICS_CSV, mode='a', index=False, header=False)
#     else:
#         validation_df.to_csv(ACTUAL_METRICS_CSV, index=False)

#     print(f"\nSaved {len(validation_df)} rows to {ACTUAL_METRICS_CSV}")
#     # print(f"Annotated images saved to {IMAGE_DIR}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate validation samples with zone overlays.")
#     parser.add_argument("video_path", help="Path to the source video (e.g., library_0.mp4)")
#     parser.add_argument("--zones", required=True, help="Path to JSON file containing zone polygons")
#     args = parser.parse_args()
#     main(args.video_path, args.zones)









# # import os
# # import cv2
# # import argparse
# # import numpy as np
# # import pandas as pd

# # from test_detect_z_assign import load_zones, draw_zones


# # # ============================================================
# # # CONFIG
# # # ============================================================

# # CSV_DIR = r"E:\8 sem project\cv_endsem_project\outputs\metrics"

# # IMAGE_DIR = r"E:\8 sem project\cv_endsem_project\outputs\images"

# # ACTUAL_METRICS_CSV = (
# #     r"E:\8 sem project\cv_endsem_project\outputs\metrics\actual_metrics.csv"
# # )

# # NUM_SAMPLES = 20


# # # ============================================================
# # # HELPERS
# # # ============================================================

# # def get_metrics_csv_path(video_path):
# #     """
# #     library_0.mp4
# #     ->
# #     library_0_frame_metrics.csv
# #     """

# #     video_name = os.path.splitext(
# #         os.path.basename(video_path)
# #     )[0]

# #     return os.path.join(
# #         CSV_DIR,
# #         f"{video_name}_frame_metrics.csv"
# #     )


# # def stratified_sample(df, n_samples=20):

# #     total_rows = len(df)

# #     if total_rows <= n_samples:
# #         return df.copy()

# #     indices = np.linspace(
# #         0,
# #         total_rows - 1,
# #         n_samples,
# #         dtype=int
# #     )

# #     return df.iloc[indices].copy()


# # def create_actual_metrics_schema(df):

# #     zone_cols = [
# #         col for col in df.columns
# #         if col.startswith("zone_")
# #     ]

# #     for zone_col in zone_cols:
# #         df[f"actual_{zone_col}"] = -1

# #     df["Actual_people"] = -1

# #     ordered_cols = list(df.columns)

# #     ordered_cols.remove("Actual_people")

# #     insert_idx = ordered_cols.index(zone_cols[-1]) + 1

# #     ordered_cols.insert(insert_idx, "Actual_people")

# #     return df[ordered_cols]


# # def save_sample_images(
# #     video_path,
# #     sampled_df,
# #     zones
# # ):

# #     cap = cv2.VideoCapture(video_path)

# #     if not cap.isOpened():
# #         raise RuntimeError(
# #             f"Cannot open video:\n{video_path}"
# #         )

# #     for _, row in sampled_df.iterrows():

# #         frame_number = int(row["frame_number"])

# #         cap.set(
# #             cv2.CAP_PROP_POS_FRAMES,
# #             frame_number
# #         )

# #         success, frame = cap.read()

# #         if not success:
# #             print(
# #                 f"Could not read frame "
# #                 f"{frame_number}"
# #             )
# #             continue

# #         annotated = frame.copy()

# #         draw_zones(
# #             annotated,
# #             zones
# #         )

# #         video_name = os.path.splitext(
# #             os.path.basename(video_path)
# #         )[0]

# #         image_name = (
# #             f"{video_name}_frame_{frame_number}.jpg"
# #         )

# #         image_path = os.path.join(
# #             IMAGE_DIR,
# #             image_name
# #         )

# #         cv2.imwrite(
# #             image_path,
# #             annotated
# #         )

# #     cap.release()


# # # ============================================================
# # # MAIN
# # # ============================================================

# # def main(video_path):

# #     video_name = os.path.splitext(
# #         os.path.basename(video_path)
# #     )[0]

# #     metrics_csv = get_metrics_csv_path(
# #         video_path
# #     )

# #     if not os.path.exists(metrics_csv):
# #         raise FileNotFoundError(
# #             f"Metrics CSV not found:\n{metrics_csv}"
# #         )

# #     print(
# #         f"Loading metrics:\n{metrics_csv}"
# #     )

# #     df = pd.read_csv(metrics_csv)

# #     sampled_df = stratified_sample(
# #         df,
# #         NUM_SAMPLES
# #     )

# #     print(
# #         f"Selected "
# #         f"{len(sampled_df)} "
# #         f"stratified samples."
# #     )

# #     # --------------------------------
# #     # load zones
# #     # --------------------------------

# #     zones = load_zones(video_path)

# #     # --------------------------------
# #     # save annotated images
# #     # --------------------------------

# #     save_sample_images(
# #         video_path,
# #         sampled_df,
# #         zones
# #     )

# #     # --------------------------------
# #     # prepare validation rows
# #     # --------------------------------

# #     validation_df = create_actual_metrics_schema(
# #         sampled_df
# #     )

# #     # --------------------------------
# #     # append to actual_metrics.csv
# #     # --------------------------------

# #     if os.path.exists(
# #         ACTUAL_METRICS_CSV
# #     ):

# #         validation_df.to_csv(
# #             ACTUAL_METRICS_CSV,
# #             mode="a",
# #             index=False,
# #             header=False
# #         )

# #     else:

# #         validation_df.to_csv(
# #             ACTUAL_METRICS_CSV,
# #             index=False
# #         )

# #     print(
# #         f"\nSaved {len(validation_df)} rows "
# #         f"to actual_metrics.csv"
# #     )

# #     print(
# #         f"Saved annotated images to:\n"
# #         f"{IMAGE_DIR}"
# #     )


# # if __name__ == "__main__":

# #     parser = argparse.ArgumentParser()

# #     parser.add_argument(
# #         "video_path",
# #         help="Path to source video"
# #     )

# #     args = parser.parse_args()

# #     main(args.video_path)