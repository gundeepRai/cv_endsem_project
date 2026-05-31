"""
Zone Definition Tool
Click 4 corners per zone in order: TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT
Press 'q' to finish early, 'r' to reset current zone, 'u' to undo last point.
"""

""" ++++++++++++++++++++++++++++++++++ STAGE:3 FINDING COORDINATES OF ZONES ++++++++++++++++++++++++++++++++++++++++"""

import cv2
import json
import numpy as np

# ==================== CONFIGURATION ====================
VIDEO_PATH = "././videos/raw/test1.mp4"          # Change per video
OUTPUT_JSON = "././config/zones_test_video1.json"   # Change per video
NUM_ZONES = 4                              # Number of zones to define
POLYGON_COLOR = (100, 100, 255)            # Dull red/pink (BGR)
POLYGON_ALPHA = 0.35                       # Transparency level
# =======================================================

# Load first frame
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"ERROR: Cannot read video {VIDEO_PATH}")
    exit()

# Work on a copy for drawing
original = frame.copy()
display = frame.copy()

# Data storage
points = []          # all clicked points (flattened)
zones = {}           # final zone data
current_zone = []    # points of zone being clicked
zone_count = 1

# Instructions
INSTRUCTIONS = [
    f"ZONE {zone_count}: Click TOP-LEFT corner",
    f"ZONE {zone_count}: Click TOP-RIGHT corner",
    f"ZONE {zone_count}: Click BOTTOM-RIGHT corner",
    f"ZONE {zone_count}: Click BOTTOM-LEFT corner"
]
current_instruction = INSTRUCTIONS[0]

def draw_polygon(img, pts, color, alpha=0.35):
    """Draw semi-transparent polygon on image."""
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # Draw outline
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, color, 2)

def mouse_callback(event, x, y, flags, param):
    global points, current_zone, zone_count, current_instruction, display

    if event == cv2.EVENT_LBUTTONDOWN:
        # Store point
        current_zone.append((x, y))
        points.append((x, y))

        # Draw point on display
        cv2.circle(display, (x, y), 4, (0, 255, 0), -1)

        # If 4 points collected for this zone
        if len(current_zone) == 4:
            # Save zone
            zone_name = f"zone_{zone_count}"
            zones[zone_name] = {
                "top_left": current_zone[0],
                "top_right": current_zone[1],
                "bottom_right": current_zone[2],
                "bottom_left": current_zone[3]
            }

            # Draw polygon on display
            draw_polygon(display, current_zone, POLYGON_COLOR, POLYGON_ALPHA)

            # Move to next zone
            zone_count += 1
            current_zone = []

            if zone_count <= NUM_ZONES:
                current_instruction = f"ZONE {zone_count}: Click TOP-LEFT corner"
            else:
                current_instruction = "All zones defined. Press 'q' to save and quit."

        else:
            # Update instruction for next corner
            corners = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
            current_instruction = f"ZONE {zone_count}: Click {corners[len(current_zone)]} corner"

        # Show updated display
        cv2.imshow("Define Zones", display)

# Setup window
cv2.namedWindow("Define Zones", cv2.WINDOW_NORMAL)
cv2.imshow("Define Zones", display)
cv2.setMouseCallback("Define Zones", mouse_callback)

print("=" * 60)
print("  ZONE DEFINITION TOOL")
print("=" * 60)
print(f"  Video: {VIDEO_PATH}")
print(f"  Number of zones to define: {NUM_ZONES}")
print(f"  Output: {OUTPUT_JSON}")
print()
print("  Click corners in this order:")
print("    1. TOP-LEFT")
print("    2. TOP-RIGHT")
print("    3. BOTTOM-RIGHT")
print("    4. BOTTOM-LEFT")
print()
print("  Press 'q' to save and quit early.")
print("  Press 'r' to reset current zone.")
print("  Press 'u' to undo last point.")
print("=" * 60)

# Event loop
while True:
    # Update instruction text on display (bottom-left corner)
    temp_display = display.copy()
    cv2.putText(temp_display, current_instruction, (20, display.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imshow("Define Zones", temp_display)

    key = cv2.waitKey(1) & 0xFF

    # Quit and save
    if key == ord('q'):
        break

    # Reset current zone
    if key == ord('r') and len(current_zone) > 0:
        # Remove points of current zone from points list
        for _ in range(len(current_zone)):
            if points:
                points.pop()
        current_zone.clear()
        # Redraw display from original + saved zones
        display = original.copy()
        for zname, zdata in zones.items():
            pts = [zdata["top_left"], zdata["top_right"],
                   zdata["bottom_right"], zdata["bottom_left"]]
            draw_polygon(display, pts, POLYGON_COLOR, POLYGON_ALPHA)
        # Redraw all points still in points (from previous zones)
        for p in points:
            cv2.circle(display, p, 4, (0, 255, 0), -1)
        current_instruction = f"ZONE {zone_count}: Click TOP-LEFT corner"
        cv2.imshow("Define Zones", display)

    # Undo last point
    if key == ord('u') and len(current_zone) > 0:
        # Remove last point from current_zone and points
        current_zone.pop()
        points.pop()
        # Redraw (similar to reset but keep current_zone partial)
        display = original.copy()
        for zname, zdata in zones.items():
            pts = [zdata["top_left"], zdata["top_right"],
                   zdata["bottom_right"], zdata["bottom_left"]]
            draw_polygon(display, pts, POLYGON_COLOR, POLYGON_ALPHA)
        for p in points:
            cv2.circle(display, p, 4, (0, 255, 0), -1)
        corners = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
        current_instruction = f"ZONE {zone_count}: Click {corners[len(current_zone)]} corner"
        cv2.imshow("Define Zones", display)

cv2.destroyAllWindows()

# Save to JSON (overwrite mode)
import os
os.makedirs(os.path.dirname(OUTPUT_JSON) if os.path.dirname(OUTPUT_JSON) else '.', exist_ok=True)

# Convert tuple keys to lists for JSON
zones_json = {}
for zname, zdata in zones.items():
    zones_json[zname] = {
        "top_left": list(zdata["top_left"]),
        "top_right": list(zdata["top_right"]),
        "bottom_right": list(zdata["bottom_right"]),
        "bottom_left": list(zdata["bottom_left"])
    }

with open(OUTPUT_JSON, 'w') as f:
    json.dump(zones_json, f, indent=2)

print(f"\n✅ Zones saved to {OUTPUT_JSON}")
print(f"   Total zones defined: {len(zones)}")