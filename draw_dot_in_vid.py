import re
from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd


DEFAULT_VIDEO = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\xi_api_data 05142026\endo_video\video_stream_1_20260514_201037.mp4")
DEFAULT_CSV = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\full_2D_projection.csv")
DEFAULT_OUTPUT = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\video_stream_with_dot.mp4")


def parse_2d_point(value):
    if pd.isna(value):
        return None

    if isinstance(value, str):
        nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", value)
        if len(nums) >= 2:
            return np.array([float(nums[0]), float(nums[1])], dtype=np.float32)
        return None

    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float32)
        if arr.size >= 2:
            return arr[:2]

    try:
        arr = np.asarray(value, dtype=np.float32).flatten()
        if arr.size >= 2:
            return arr[:2]
    except Exception:
        return None

    return None


def load_points(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "p_2D" not in df.columns:
        raise ValueError(f"CSV must contain a 'p_2D' column. Found: {list(df.columns)}")

    points = [parse_2d_point(v) for v in df["p_2D"]]
    return points


def draw_over_video(video_path: Path, points, output_path: Path | None = None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    if output_path is not None:
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open video writer: {output_path}")

    window_name = "Video with Red Dot"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, width), min(720, height))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index < len(points) and points[frame_index] is not None:
            x, y = points[frame_index]
            center = (int(round(x)), int(round(y)))
            cv2.circle(frame, center, radius=8, color=(0, 0, 255), thickness=-1)
 
        if writer is not None:
            writer.write(frame)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay 2D projected points on a video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Path to the input video file")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the CSV file with p_2D points")
    parser.add_argument("--output", type=Path, default=None, help="Optional output video file path")
    return parser.parse_args()


def main():
    args = parse_args()
    points = load_points(args.csv)
    print(f"Loaded {len(points)} points from {args.csv}")
    print(f"Opening video {args.video}")
    draw_over_video(args.video, points, args.output)


if __name__ == "__main__":
    main()
