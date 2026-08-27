"""Reproject FK end-effector positions without reading or writing video.

The two input CSVs must contain matching ``frame_id`` values and a transform
column named either ``Raw_Transform`` (12-value, translation-first Xi layout)
or ``Transform`` (a formatted 4x4 matrix).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
import cv2


REPO_ROOT = Path(__file__).resolve().parents[2]


BASE_T_EE_CSV = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\xi_ee_0722\EE2_transforms_FK.csv")
BASE_T_ECM_CSV = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\xi_ee_0722\ECM_transforms_FK.csv")

VIDEO = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\OR_session_data\session_195518\video_stream_1_session_195518.mp4")

Start_sec = 22
End_sec = 30

OUTPUT_VIDEO = REPO_ROOT / "data" / "xi_ee_0722" / "EE_red_dot_projection_22s_to_28s.mp4"

HAND_EYE_NPZ = (
    REPO_ROOT
    / "data"
    / "xi_data_0616"
    / "xi_hand_eye_0616_translation_first.npz"
)


INTRINSICS_MAT = Path(
    r"C:\Users\Tracy\Documents\MATLAB\20260616_xi_data\K_left.mat"
)
OUTPUT_CSV = REPO_ROOT / "data" / "xi_ee_0722" / "EE_pixel_projection.csv"

# Confirmed from video_stream_1_session_195518.mp4 metadata.
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

IMAGE_WIDTH = 1264
IMAGE_HEIGHT = 1010


def parse_numbers(value: object) -> np.ndarray:
    """Extract whitespace/comma-separated numbers from an array string."""
    return np.fromstring(
        str(value).replace("[", " ").replace("]", " ").replace(",", " "),
        sep=" ",
        dtype=np.float64,
    )


def parse_transform(value: object) -> np.ndarray:
    """Parse either a 12-value Xi transform or a formatted 4x4 transform."""
    values = parse_numbers(value)

    if values.size == 12:
        # Xi translation-first layout:
        # [tx ty tz R00 R01 R02 R10 R11 R12 R20 R21 R22]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = values[:3]
        transform[:3, :3] = values[3:].reshape(3, 3)
    elif values.size == 16:
        transform = values.reshape(4, 4)
    else:
        raise ValueError(f"expected 12 or 16 transform values, got {values.size}")

    rotation = transform[:3, :3]
    if not np.isfinite(transform).all():
        raise ValueError("transform contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("invalid homogeneous-transform bottom row")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError("rotation determinant is not +1")

    return transform


def load_transforms(csv_path: Path, transform_name: str) -> pd.DataFrame:
    """Load valid transforms and return frame_id plus the named matrix column."""
    df = pd.read_csv(csv_path)
    if "frame_id" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a frame_id column")

    if "Raw_Transform" in df.columns:
        source_column = "Raw_Transform"
    elif "Transform" in df.columns:
        source_column = "Transform"
    else:
        raise ValueError(
            f"{csv_path} must contain Raw_Transform or Transform"
        )

    rows = []
    for row in df.itertuples(index=False):
        frame_id = int(row.frame_id)
        try:
            transform = parse_transform(getattr(row, source_column))
        except ValueError as exc:
            print(f"{csv_path.name}: skipping frame {frame_id}: {exc}")
            continue
        rows.append({"frame_id": frame_id, transform_name: transform})

    result = pd.DataFrame(rows)
    print(f"{csv_path.name}: loaded {len(result)} valid transforms")
    return result


def format_array(value: np.ndarray) -> str:
    return np.array2string(
        np.asarray(value, dtype=np.float64),
        precision=10,
        suppress_small=False,
        separator=" ",
    )

def rotz(theta, degrees=False):
    """Rotation about the z-axis by theta (in radians)"""
    if degrees:
        theta = np.radians(theta)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def save_red_dot_video(
    video_path,
    projections_df,
    output_video,
    start_sec,
    end_sec,
):
    """Save a red-dot clip for the half-open interval [start_sec, end_sec)."""
    if start_sec < 0:
        raise ValueError("start_sec must be non-negative")
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Video reports an invalid FPS: {fps}")

    start_frame = int(np.ceil(start_sec * fps))
    end_frame = int(np.ceil(end_sec * fps))

    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output_video}")

    points_by_frame = {}
    for row_idx, row in projections_df.iterrows():
        if not np.isfinite([row["pixel_u"], row["pixel_v"]]).all():
            continue

        frame_id = (
            int(row["frame_id"])
            if "frame_id" in projections_df.columns
            else int(row_idx)
        )
        if not start_frame <= frame_id < end_frame:
            continue

        point = (
            int(round(float(row["pixel_u"]))),
            int(round(float(row["pixel_v"]))),
        )
        # Keep diagnostic dots that land on the encoded black border.
        if 0 <= point[0] < width and 0 <= point[1] < height:
            points_by_frame[frame_id] = point

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_id = start_frame
    frames_written = 0
    drawn = 0
    try:
        while frame_id < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            point = points_by_frame.get(frame_id)
            if point is not None:
                cv2.circle(frame, point, 6, (0, 0, 255), -1)
                drawn += 1

            writer.write(frame)
            frames_written += 1
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    print(f"Saved red dot video: {output_video}")
    print(
        f"Video interval: [{start_sec}, {end_sec}) seconds; "
        f"source frames: {start_frame}-{end_frame - 1}"
    )
    print(f"Video frames written: {frames_written}")
    print(f"Frames with red dot drawn: {drawn}")



def main() -> None:
    base_to_ee = load_transforms(BASE_T_EE_CSV, "Base_T_EE")
    base_to_ecm = load_transforms(BASE_T_ECM_CSV, "Base_T_ECM")

    matched = base_to_ee.merge(
        base_to_ecm,
        on="frame_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("frame_id")
    print(f"Frames common to both CSVs: {len(matched)}")

    calibration = np.load(HAND_EYE_NPZ)
    ECM_T_camera = np.eye(4)
    ECM_T_camera[0][0] = -1
    ECM_T_camera[1][1] = -1


    print(ECM_T_camera)

    
    # ECM_T_camera = calibration["X"].astype(np.float64)
    Camera_T_ECM = np.linalg.inv(ECM_T_camera)
    K = loadmat(INTRINSICS_MAT)["K_left"].astype(np.float64)

    output_rows = []
    for row in matched.itertuples(index=False):
        Base_T_EE = row.Base_T_EE
        Base_T_ECM = row.Base_T_ECM

        ECM_T_Base = np.linalg.inv(Base_T_ECM)
        ECM_T_EE = ECM_T_Base @ Base_T_EE
        Camera_T_EE = Camera_T_ECM @ ECM_T_EE
        point = Camera_T_EE[:3, 3]

        X, Y, Z = point

        if np.isfinite(point).all() and Z > 1e-9:
            homogeneous_pixel = K @ point
            # pixel_u = homogeneous_pixel[0] / homogeneous_pixel[2]
            # pixel_v = homogeneous_pixel[1] / homogeneous_pixel[2]
            pixel_v = homogeneous_pixel[0] / homogeneous_pixel[2]
            pixel_u = homogeneous_pixel[1] / homogeneous_pixel[2]
            
            in_front = True
            in_image = (
                0.0 <= pixel_u < IMAGE_WIDTH
                and 0.0 <= pixel_v < IMAGE_HEIGHT
            )
        else:
            pixel_u = np.nan
            pixel_v = np.nan
            in_front = False
            in_image = False

        output_rows.append(
            {
                "frame_id": int(row.frame_id),
                "Base_T_EE": format_array(Base_T_EE),
                "Base_T_ECM": format_array(Base_T_ECM),
                "ECM_T_EE": format_array(ECM_T_EE),
                "Camera_T_EE": format_array(Camera_T_EE),
                "camera_x": X,
                "camera_y": Y,
                "camera_z": Z,
                "x_over_z": X / Z if abs(Z) > 1e-9 else np.nan,
                "y_over_z": Y / Z if abs(Z) > 1e-9 else np.nan,
                "pixel_u": pixel_u,
                "pixel_v": pixel_v,
                "in_front": in_front,
                "in_image": in_image,
            }
        )

    output = pd.DataFrame(output_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(output)} projected poses to: {OUTPUT_CSV}")
    if not output.empty:
        print(output[["frame_id", "camera_x", "camera_y", "camera_z",
                      "pixel_u", "pixel_v", "in_front", "in_image"]].head())
        print(f"Points in front of camera: {int(output['in_front'].sum())}")
        print(f"Points inside image bounds: {int(output['in_image'].sum())}")

    save_red_dot_video(
        VIDEO,
        output,
        OUTPUT_VIDEO,
        Start_sec,
        End_sec,
    )


if __name__ == "__main__":
    main()
