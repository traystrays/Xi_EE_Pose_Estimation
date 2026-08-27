"""Process a requested interval shared by the 2026-06-23 video and robot logs.

The video clock is one hour behind the robot clock. Video timestamps are shifted
by ``--video-clock-offset-hours`` before matching. Manipulator 2's direct
translation-first tip transform is already expressed in the Manipulator 1
camera frame and is therefore projected directly into the image.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SURGERY_DIR = Path(r"D:\Tracy-RCL-2026\20260623_OR_Surgery")
DEFAULT_LOG_DIR = DEFAULT_SURGERY_DIR / "20260623_162611.969"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "xi_ee_0623_162612"
DEFAULT_K = Path(r"C:\Users\Tracy\Documents\MATLAB\20260616_xi_data\K_left.mat")


def parse_translation_first(value: object) -> np.ndarray:
    values = np.fromstring(str(value).strip().strip("[]"), sep=" ")
    if values.size != 12 or not np.isfinite(values).all():
        raise ValueError(f"Expected 12 finite values, got {values.size}")
    transform = np.eye(4)
    transform[:3, 3] = values[:3]
    transform[:3, :3] = values[3:].reshape(3, 3)
    rotation = transform[:3, :3]
    if abs(np.linalg.det(rotation) - 1.0) > 0.05:
        raise ValueError("Rotation determinant is not near +1")
    if np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 0.05:
        raise ValueError("Rotation is not orthonormal")
    if np.linalg.norm(transform[:3, 3]) < 1e-8:
        raise ValueError("Zero translation is an invalid/unpopulated transform")
    if np.count_nonzero(np.abs(transform[:3, 3]) < 1e-6) >= 2:
        raise ValueError("At least two translation components are unpopulated")
    if np.max(np.abs(transform[:3, 3])) > 10.0:
        raise ValueError("Translation is outside the plausible range")
    return transform


def format_array(array: np.ndarray) -> str:
    return np.array2string(
        np.asarray(array), precision=10, suppress_small=False, separator=" "
    )


def load_tip_transforms(csv_path: Path, prefix: str) -> pd.DataFrame:
    raw = pd.read_csv(
        csv_path, usecols=["system_timestamp_ms", "field", "manip", "data"]
    )
    raw = raw[raw["field"] == "ISI_TIP_TRANSFORM"].copy()
    raw[f"{prefix}_robot_ts"] = pd.to_datetime(
        raw["system_timestamp_ms"], unit="ms", utc=True
    ).dt.tz_convert("America/Vancouver")

    transforms = []
    valid = []
    for value in raw["data"]:
        try:
            transforms.append(parse_translation_first(value))
            valid.append(True)
        except ValueError:
            transforms.append(None)
            valid.append(False)
    raw[f"{prefix}_Transform"] = transforms
    raw[f"{prefix}_source_valid"] = valid
    return raw[
        [
            f"{prefix}_robot_ts",
            f"{prefix}_Transform",
            f"{prefix}_source_valid",
            "data",
        ]
    ].rename(columns={"data": f"{prefix}_Raw_Transform"})


def choose_overlap_frames(
    timestamp_path: Path,
    robot_start: pd.Timestamp,
    offset_hours: float,
    start_minutes: float,
    duration_minutes: float,
) -> pd.DataFrame:
    original = pd.Series(timestamp_path.read_text().splitlines(), name="video_ts_raw")
    original = original[original.str.len() > 0].reset_index(drop=True)
    corrected = pd.to_datetime(original) + pd.Timedelta(hours=offset_hours)
    corrected = corrected.dt.tz_localize("America/Vancouver")
    start = robot_start + pd.Timedelta(minutes=start_minutes)
    end = start + pd.Timedelta(minutes=duration_minutes)
    mask = (corrected >= start) & (corrected < end)
    selected = pd.DataFrame(
        {
            "source_frame_id": np.flatnonzero(mask.to_numpy()),
            "video_ts_raw": original[mask].to_numpy(),
            "video_ts_corrected": corrected[mask].to_numpy(),
        }
    )
    selected["clip_frame_id"] = np.arange(len(selected))
    return selected


def extract_clip(source: Path, selected: pd.DataFrame, output: Path) -> None:
    if selected.empty:
        raise RuntimeError("No video timestamps overlap the requested robot interval")
    first = int(selected["source_frame_id"].iloc[0])
    count = len(selected)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output}")
    written = 0
    while written < count:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    if written != count:
        raise RuntimeError(f"Expected {count} clip frames, wrote {written}")


def match_arm(
    frames: pd.DataFrame, arm: pd.DataFrame, prefix: str, tolerance_ms: float
) -> pd.DataFrame:
    frames = frames.copy()
    arm = arm.copy()
    timestamp_dtype = "datetime64[ns, America/Vancouver]"
    frames["video_ts_corrected"] = frames["video_ts_corrected"].astype(
        timestamp_dtype
    )
    arm[f"{prefix}_robot_ts"] = arm[f"{prefix}_robot_ts"].astype(timestamp_dtype)
    result = pd.merge_asof(
        frames.sort_values("video_ts_corrected"),
        arm.sort_values(f"{prefix}_robot_ts"),
        left_on="video_ts_corrected",
        right_on=f"{prefix}_robot_ts",
        direction="nearest",
        tolerance=pd.Timedelta(milliseconds=tolerance_ms),
    )
    result[f"{prefix}_time_error_ms"] = (
        result["video_ts_corrected"] - result[f"{prefix}_robot_ts"]
    ).dt.total_seconds().abs() * 1000.0
    return result


def save_transform_csv(matched: pd.DataFrame, prefix: str, output: Path) -> int:
    valid = matched[
        matched[f"{prefix}_robot_ts"].notna()
        & matched[f"{prefix}_source_valid"].fillna(False)
        & matched[f"{prefix}_Transform"].notna()
    ].copy()
    valid["Transform"] = valid[f"{prefix}_Transform"].map(format_array)
    valid["Rotation"] = valid[f"{prefix}_Transform"].map(
        lambda transform: format_array(transform[:3, :3])
    )
    valid["Translation"] = valid[f"{prefix}_Transform"].map(
        lambda transform: format_array(transform[:3, 3])
    )
    valid[
        [
            "clip_frame_id",
            "source_frame_id",
            "video_ts_raw",
            "video_ts_corrected",
            f"{prefix}_robot_ts",
            f"{prefix}_time_error_ms",
            f"{prefix}_Raw_Transform",
            "Transform",
            "Rotation",
            "Translation",
        ]
    ].to_csv(output, index=False)
    return len(valid)


def reproject_video(
    clip_path: Path,
    matched: pd.DataFrame,
    intrinsic_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    intrinsic = loadmat(intrinsic_path)["K_left"].astype(np.float64)
    positions = {}
    for row in matched.itertuples(index=False):
        transform = getattr(row, "M2_Transform")
        valid = isinstance(transform, np.ndarray) and bool(
            getattr(row, "M2_source_valid")
        )
        if valid:
            positions[int(row.clip_frame_id)] = transform[:3, 3]

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    projected = 0
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        point = positions.get(frame_id)
        # This Xi recording uses a camera convention in which visible points
        # have negative Z. Homogeneous projection remains valid for either sign.
        if point is not None and abs(point[2]) > 1e-8:
            pixel = intrinsic @ point
            u, v = pixel[0] / pixel[2], pixel[1] / pixel[2]
            if 0 <= u < width and 0 <= v < height:
                cv2.circle(frame, (round(u), round(v)), 8, (0, 0, 255), -1)
                projected += 1
        writer.write(frame)
        frame_id += 1
    cap.release()
    writer.release()
    return frame_id, projected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surgery-dir", type=Path, default=DEFAULT_SURGERY_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_K)
    parser.add_argument("--video-clock-offset-hours", type=float, default=1.0)
    parser.add_argument("--tolerance-ms", type=float, default=20.0)
    parser.add_argument(
        "--start-minutes",
        type=float,
        default=0.0,
        help="Interval start in minutes after the robot/video overlap begins.",
    )
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=1.0,
        help="Length of the requested interval in minutes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video = args.surgery_dir / "video_stream_1_20260623_145911.mp4"
    timestamps = args.surgery_dir / "video_stream_1_20260623_145911_timestamps.txt"
    m1_path = next(args.log_dir.glob("KinManip1_*.csv"))
    m2_path = next(args.log_dir.glob("KinManip2_*.csv"))

    m1 = load_tip_transforms(m1_path, "M1")
    m2 = load_tip_transforms(m2_path, "M2")
    robot_start = max(m1["M1_robot_ts"].min(), m2["M2_robot_ts"].min())
    frames = choose_overlap_frames(
        timestamps,
        robot_start,
        args.video_clock_offset_hours,
        args.start_minutes,
        args.duration_minutes,
    )
    clip = args.output_dir / "video_stream_1_robot_overlap_clip.mp4"
    extract_clip(video, frames, clip)
    frames.to_csv(args.output_dir / "video_frames.csv", index=False)

    matched = match_arm(frames, m1, "M1", args.tolerance_ms)
    matched = match_arm(matched, m2, "M2", args.tolerance_ms)
    m1_count = save_transform_csv(matched, "M1", args.output_dir / "ECM_transforms.csv")
    m2_count = save_transform_csv(matched, "M2", args.output_dir / "EE_transforms.csv")

    both = matched[
        matched["M1_Transform"].notna()
        & matched["M2_Transform"].notna()
        & matched["M1_source_valid"].fillna(False)
        & matched["M2_source_valid"].fillna(False)
    ].copy()
    both["Camera_T_EE"] = both["M2_Transform"].map(format_array)
    both["EE_position_in_camera"] = both["M2_Transform"].map(
        lambda transform: format_array(transform[:3, 3])
    )
    both[
        [
            "clip_frame_id",
            "source_frame_id",
            "video_ts_raw",
            "video_ts_corrected",
            "M1_robot_ts",
            "M2_robot_ts",
            "M1_time_error_ms",
            "M2_time_error_ms",
            "Camera_T_EE",
            "EE_position_in_camera",
        ]
    ].to_csv(args.output_dir / "EE_in_camera.csv", index=False)

    reprojection = args.output_dir / "EE_red_dot_reprojection.mp4"
    video_frames, projected = reproject_video(
        clip, matched, args.intrinsics, reprojection
    )
    print(f"Clock correction applied to video timestamps: +{args.video_clock_offset_hours} h")
    print(
        "Requested overlap interval: "
        f"{args.start_minutes:.3f} to "
        f"{args.start_minutes + args.duration_minutes:.3f} minutes"
    )
    print(f"Overlap video frames: {len(frames)}")
    print(f"Valid matched Manipulator 1 transforms: {m1_count}")
    print(f"Valid matched Manipulator 2 transforms: {m2_count}")
    print(f"Frames valid for both arms: {len(both)}")
    print(f"Reprojection frames written: {video_frames}")
    print(f"Frames with visible projected point: {projected}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
