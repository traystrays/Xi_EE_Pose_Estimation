"""Apply ECM-to-camera hand-eye calibration to direct and FK EE poses."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

from process_xi_0623_overlap import (
    format_array,
    load_tip_transforms,
    match_arm,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "xi_ee_0623_162612"
HAND_EYE = REPO_ROOT / "data" / "xi_data_0616" / "xi_hand_eye_0616_translation_first.npz"
INTRINSICS = Path(r"C:\Users\Tracy\Documents\MATLAB\20260616_xi_data\K_left.mat")
NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+\.?\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def parse_matrix(value: object) -> np.ndarray:
    values = np.asarray(NUMBER_PATTERN.findall(str(value)), dtype=np.float64)
    if values.size != 16:
        raise ValueError(f"Expected 16 matrix values, got {values.size}")
    return values.reshape(4, 4)


def save_pose_csv(rows: list[dict], path: Path) -> None:
    output = []
    for row in rows:
        transform = row.pop("transform")
        output.append(
            {
                **row,
                "Camera_T_EE": format_array(transform),
                "Rotation": format_array(transform[:3, :3]),
                "Translation": format_array(transform[:3, 3]),
            }
        )
    pd.DataFrame(output).to_csv(path, index=False)


def reproject_comparison(
    clip: Path,
    direct: dict[int, np.ndarray],
    fk: dict[int, np.ndarray],
    output: Path,
) -> tuple[int, int, int]:
    intrinsic = loadmat(INTRINSICS)["K_left"].astype(np.float64)
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frame_id = direct_visible = fk_visible = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for poses, color, label in (
            (direct, (0, 0, 255), "Direct"),
            (fk, (255, 255, 0), "FK"),
        ):
            transform = poses.get(frame_id)
            if transform is None or abs(transform[2, 3]) < 1e-8:
                continue
            point = transform[:3, 3]
            pixel = intrinsic @ point
            u, v = pixel[0] / pixel[2], pixel[1] / pixel[2]
            if 0 <= u < width and 0 <= v < height:
                cv2.circle(frame, (round(u), round(v)), 8, color, -1)
                cv2.putText(
                    frame,
                    label,
                    (round(u) + 10, round(v) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                if poses is direct:
                    direct_visible += 1
                else:
                    fk_visible += 1
        writer.write(frame)
        frame_id += 1
    cap.release()
    writer.release()
    return frame_id, direct_visible, fk_visible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--hand-eye", type=Path, default=HAND_EYE)
    parser.add_argument("--intrinsics", type=Path, default=INTRINSICS)
    parser.add_argument("--tolerance-ms", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    global INTRINSICS
    INTRINSICS = args.intrinsics
    calibration = np.load(args.hand_eye)
    ecm_T_cam = calibration["X"]
    cam_T_ecm = np.linalg.inv(ecm_T_cam)
    print("ECM_T_Cam:")
    print(ecm_T_cam)

    direct_csv = pd.read_csv(data_dir / "EE_transforms.csv")
    direct_rows = []
    direct_poses = {}
    for row in direct_csv.itertuples(index=False):
        ecm_T_ee = parse_matrix(row.Transform)
        cam_T_ee = cam_T_ecm @ ecm_T_ee
        frame_id = int(row.clip_frame_id)
        direct_poses[frame_id] = cam_T_ee
        direct_rows.append(
            {
                "clip_frame_id": frame_id,
                "source_frame_id": int(row.source_frame_id),
                "video_ts_corrected": row.video_ts_corrected,
                "transform": cam_T_ee,
            }
        )
    save_pose_csv(direct_rows, data_dir / "EE_in_camera_hand_eye.csv")

    frames = pd.read_csv(data_dir / "video_frames.csv")
    frames["video_ts_corrected"] = pd.to_datetime(
        frames["video_ts_corrected"], utc=True
    ).dt.tz_convert("America/Vancouver")
    f1 = load_tip_transforms(data_dir / "KinManip1_FK.csv", "F1")
    f2 = load_tip_transforms(data_dir / "KinManip2_FK.csv", "F2")
    matched = match_arm(frames, f1, "F1", args.tolerance_ms)
    matched = match_arm(matched, f2, "F2", args.tolerance_ms)
    valid = matched[
        matched["F1_Transform"].notna()
        & matched["F2_Transform"].notna()
        & matched["F1_source_valid"].fillna(False)
        & matched["F2_source_valid"].fillna(False)
    ]
    fk_rows = []
    fk_poses = {}
    for row in valid.itertuples(index=False):
        base_T_ecm = row.F1_Transform
        base_T_ee = row.F2_Transform
        ecm_T_ee = np.linalg.inv(base_T_ecm) @ base_T_ee
        cam_T_ee = cam_T_ecm @ ecm_T_ee
        frame_id = int(row.clip_frame_id)
        fk_poses[frame_id] = cam_T_ee
        fk_rows.append(
            {
                "clip_frame_id": frame_id,
                "source_frame_id": int(row.source_frame_id),
                "video_ts_corrected": row.video_ts_corrected,
                "F1_time_error_ms": row.F1_time_error_ms,
                "F2_time_error_ms": row.F2_time_error_ms,
                "Base_T_ECM_FK": format_array(base_T_ecm),
                "Base_T_EE_FK": format_array(base_T_ee),
                "ECM_T_EE_FK": format_array(ecm_T_ee),
                "transform": cam_T_ee,
            }
        )
    save_pose_csv(fk_rows, data_dir / "EE_in_camera_FK_hand_eye.csv")

    clip = data_dir / "video_stream_1_robot_overlap_clip.mp4"
    legacy_clip = data_dir / "video_stream_1_robot_overlap_first_1min.mp4"
    if not clip.exists() and legacy_clip.exists():
        clip = legacy_clip
    comparison = data_dir / "EE_reprojection_direct_vs_FK_hand_eye.mp4"
    written, direct_visible, fk_visible = reproject_comparison(
        clip, direct_poses, fk_poses, comparison
    )
    print(f"Direct hand-eye poses: {len(direct_poses)}")
    print(f"FK hand-eye poses: {len(fk_poses)}")
    print(f"Video frames written: {written}")
    print(f"Direct visible projections: {direct_visible}")
    print(f"FK visible projections: {fk_visible}")
    print(f"Output: {comparison}")


if __name__ == "__main__":
    main()
