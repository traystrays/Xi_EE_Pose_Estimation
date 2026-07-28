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
import argparse
import pandas as pd
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "xi_data_0616"
IMAGE_DIR = REPO_ROOT / "checkerboard_images" / "xi_20260616" / "hand_eye_calibration"
CSV_PATH = DATA_DIR / "saved_frames.csv"
FRAME_TRANSFORMS_CSV = DATA_DIR / "frame_transforms.csv"


def load_valid_frame_ids(csv_path=FRAME_TRANSFORMS_CSV):
    frame_transforms = pd.read_csv(csv_path)
    if "frame_id" not in frame_transforms:
        raise ValueError(f"No frame_id column found in {csv_path}")
    if "Transform" in frame_transforms:
        frame_transforms = frame_transforms[
            ~frame_transforms["Transform"]
            .astype(str)
            .str.contains(r"\bnan\b", case=False, regex=True)
        ]
    frame_ids = (
        pd.to_numeric(frame_transforms["frame_id"], errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not frame_ids:
        raise ValueError(f"No valid frame_id values found in {csv_path}")
    
    print(frame_ids)

    return frame_ids


def main(
    video_path,
    valid_only=False,
    data_dir=DATA_DIR,
    image_dir=IMAGE_DIR,
    valid_frames_csv=FRAME_TRANSFORMS_CSV,
):
    data_dir = Path(data_dir)
    image_dir = Path(image_dir)
    csv_path = data_dir / "saved_frames.csv"
    if not os.path.exists(video_path):
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    valid_frame_ids = None
    if valid_only:
        valid_frame_ids = load_valid_frame_ids(valid_frames_csv)
        valid_frame_ids = [
            frame_id for frame_id in valid_frame_ids
            if 0 <= frame_id < total_frames
        ]
        if not valid_frame_ids:
            print("Error: no frame_transforms.csv frame IDs are inside the video frame range.")
            sys.exit(1)
        print(f"Showing only valid transform frames: {len(valid_frame_ids)}")

    # Output setup
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Reset CSV with header at the start of each picker run.
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_number", "image_filename"])

    saved_frame_numbers = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                saved_frame_numbers.add(int(row["frame_number"]))
            except (KeyError, TypeError, ValueError):
                continue

    print(f"Existing saved frames: {len(saved_frame_numbers)}")
    print(f"Saving images to: {image_dir}")
    print(f"Logging frames to: {csv_path}")

    window_name = "Frame Picker - drag slider, press S to save, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Trackbar callback (required by OpenCV, can be a no-op)
    def on_trackbar(val):
        pass

    if valid_only:
        max_slider = len(valid_frame_ids) - 1
        trackbar_name = "Valid Frame Index"
    else:
        max_slider = max(total_frames - 1, 0)
        trackbar_name = "Frame"

    cv2.createTrackbar(trackbar_name, window_name, 0, max_slider, on_trackbar)

    current_slider_pos = -1  # force initial read
    current_frame = -1

    while True:
        try:
            slider_pos = cv2.getTrackbarPos(trackbar_name, window_name)
        except cv2.error:
            print("Frame picker window was closed.")
            break

        if slider_pos != current_slider_pos:
            current_slider_pos = slider_pos
            current_frame = (
                valid_frame_ids[slider_pos]
                if valid_only
                else slider_pos
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: could not read frame {current_frame}")
                continue

            # Overlay frame number text
            display_frame = frame.copy()
            if valid_only:
                text = f"Valid index: {slider_pos + 1}/{len(valid_frame_ids)} | Frame: {current_frame}"
            else:
                text = f"Frame: {current_frame} / {max_slider}"
            cv2.putText(
                display_frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
            )

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            break

        elif key in (ord("a"), ord("A")):  # A = previous frame
            new_pos = max(current_slider_pos - 1, 0)
            cv2.setTrackbarPos(trackbar_name, window_name, new_pos)

        elif key in (ord("d"), ord("D")):  # D = next frame
            new_pos = min(current_slider_pos + 1, max_slider)
            cv2.setTrackbarPos(trackbar_name, window_name, new_pos)

        elif key in (ord("s"), ord("S")):
            if current_frame in saved_frame_numbers:
                print(f"Frame {current_frame} is already saved; skipping duplicate")
                continue

            img_filename = f"frame_{current_frame:06d}.png"
            img_path = image_dir / img_filename
            cv2.imwrite(str(img_path), frame)

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([current_frame, img_filename])

            saved_frame_numbers.add(current_frame)
            print(f"Saved frame {current_frame} -> {img_path}")
            print(f"Total saved frames: {len(saved_frame_numbers)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Show only frame IDs present in --valid-frames-csv.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument(
        "--valid-frames-csv",
        type=Path,
        default=FRAME_TRANSFORMS_CSV,
    )
    args = parser.parse_args()

    main(
        args.video_path,
        valid_only=args.valid_only,
        data_dir=args.data_dir,
        image_dir=args.image_dir,
        valid_frames_csv=args.valid_frames_csv,
    )
