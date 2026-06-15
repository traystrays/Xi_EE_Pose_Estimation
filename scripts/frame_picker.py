"""
Frame Picker
------------
- Reads a video and prints the total frame count.
- Opens a window with a slider (trackbar) to scrub through frames.
- Press 'S' to save the currently displayed frame as an image
  (in ./saved_frames/) and log the frame number to saved_frames.csv.
- Press 'Q' or ESC to quit.

Usage:
    python frame_picker.py path/to/video.mp4
"""

import sys
import os
import csv
import cv2
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "si_data_0612"
IMAGE_DIR = REPO_ROOT / "checkerboard_images" / "si_20260612" / "calibration"
CSV_PATH = DATA_DIR / "saved_frames.csv"

def main(video_path):
    if not os.path.exists(video_path):
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # Output setup
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Create CSV with header if it doesn't exist
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_number", "image_filename"])

    saved_frame_numbers = set()
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                saved_frame_numbers.add(int(row["frame_number"]))
            except (KeyError, TypeError, ValueError):
                continue

    print(f"Existing saved frames: {len(saved_frame_numbers)}")
    print(f"Saving images to: {IMAGE_DIR}")
    print(f"Logging frames to: {CSV_PATH}")

    window_name = "Frame Picker - drag slider, press S to save, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Trackbar callback (required by OpenCV, can be a no-op)
    def on_trackbar(val):
        pass

    max_frame = max(total_frames - 1, 0)
    cv2.createTrackbar("Frame", window_name, 0, max_frame, on_trackbar)

    current_frame = -1  # force initial read

    while True:
        try:
            slider_pos = cv2.getTrackbarPos("Frame", window_name)
        except cv2.error:
            print("Frame picker window was closed.")
            break

        if slider_pos != current_frame:
            current_frame = slider_pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: could not read frame {current_frame}")
                continue

            # Overlay frame number text
            display_frame = frame.copy()
            text = f"Frame: {current_frame} / {max_frame}"
            cv2.putText(
                display_frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
            )

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            break

        elif key in (ord("a"), ord("A")):  # A = previous frame
            new_pos = max(current_frame - 1, 0)
            cv2.setTrackbarPos("Frame", window_name, new_pos)

        elif key in (ord("d"), ord("D")):  # D = next frame
            new_pos = min(current_frame + 1, max_frame)
            cv2.setTrackbarPos("Frame", window_name, new_pos)

        elif key in (ord("s"), ord("S")):
            if current_frame in saved_frame_numbers:
                print(f"Frame {current_frame} is already saved; skipping duplicate")
                continue

            img_filename = f"frame_{current_frame:06d}.png"
            img_path = IMAGE_DIR / img_filename
            cv2.imwrite(str(img_path), frame)

            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([current_frame, img_filename])

            saved_frame_numbers.add(current_frame)
            print(f"Saved frame {current_frame} -> {img_path}")
            print(f"Total saved frames: {len(saved_frame_numbers)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python frame_picker.py <video_path>")
        sys.exit(1)

    main(sys.argv[1])
