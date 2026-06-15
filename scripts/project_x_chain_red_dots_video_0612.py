from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

from clean_si_api import checkerboard_inner_corners


REPO_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = Path(
    r"C:\Users\Tracy\Documents\2026 RCL UBC\si_data_06122026\video_stream_1_20260612_170858.mp4"
)
LEFT_K_PATH = Path(
    r"C:\Users\Tracy\Documents\MATLAB\20260612_si_data\cameraParams_5mm_leftK.mat"
)
ROBOT_TRANSFORMS_CSV = (
    REPO_ROOT / "data" / "si_data_0612" / "hand_eye_calibration_data_METERS.csv"
)
HAND_EYE_NPZ = REPO_ROOT / "data" / "si_data_0612" / "hand_eye_0612_python.npz"
OUTPUT_VIDEO = (
    REPO_ROOT
    / "checkerboard_images"
    / "si_20260612"
    / "x_chain_red_dots_video.mp4"
)

COLUMNS = 7
ROWS = 6
SQUARE_SIZE_M = 0.005
PATTERN_SIZE = (COLUMNS, ROWS)

DIST_COEFFS = np.array(
    [-0.306143873919413, 0.4667777390935303, 0.0, 0.0, 0.0],
    dtype=np.float64,
)

# Set these to process a shorter video segment during debugging.
START_FRAME = 0
END_FRAME = None
FRAME_STEP = 1


def parse_transform_string(transform_str):
    values = np.fromstring(
        transform_str.replace("[", " ").replace("]", " ").replace(",", " "),
        sep=" ",
    )

    if values.size != 16:
        raise ValueError(f"Expected 16 transform values, got {values.size}")

    return values.reshape(4, 4)


def is_finite_transform(T):
    return T.shape == (4, 4) and np.isfinite(T).all()


def project_to_rotation(R_mat):

    if not np.isfinite(R_mat).all():
        raise ValueError("Rotation matrix contains NaN or Inf")

    U, _, Vt = np.linalg.svd(R_mat)
    R_fixed = U @ Vt

    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1
        R_fixed = U @ Vt

    return R_fixed


def draw_points(image, points, color=(0, 0, 255), radius=4):
    h, w = image.shape[:2]
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    for point in points:
        u = int(round(point[0]))
        v = int(round(point[1]))
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(image, (u, v), radius, color, -1)


def load_robot_transforms(csv_path):
    df = pd.read_csv(csv_path)

    transforms = {}
    for _, row in df.iterrows():
        if pd.isna(row["Transform"]):
            continue

        frame_id = int(row["frame_id"])
        try:
            transform = parse_transform_string(row["Transform"])
        except ValueError:
            continue

        if not is_finite_transform(transform):
            continue

        transforms[frame_id] = transform

    return transforms


def project_checkerboard_points(camera_T_checkerboard, object_points, K):
    R_pred = project_to_rotation(camera_T_checkerboard[:3, :3])
    t_pred = camera_T_checkerboard[:3, 3].reshape(3, 1)
    rvec_pred, _ = cv2.Rodrigues(R_pred)

    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec_pred,
        t_pred,
        K,
        DIST_COEFFS,
    )

    return projected_points.reshape(-1, 2)


def main():
    np.set_printoptions(precision=8, suppress=True)

    K = loadmat(LEFT_K_PATH)["cameraParams_5mmK"].astype(np.float64)
    object_points = checkerboard_inner_corners(
        columns=COLUMNS,
        rows=ROWS,
        square_size=SQUARE_SIZE_M,
    ).astype(np.float64)

    solved = np.load(HAND_EYE_NPZ)
    gripper_T_camera = solved["X"]
    base_T_checkerboard = solved["mean_base_T_checkerboard"]
    robot_transforms = load_robot_transforms(ROBOT_TRANSFORMS_CSV)

    print("Loaded K:")
    print(K)
    print("\nLoaded X = Gripper_T_Camera:")
    print(gripper_T_camera)
    print("\nLoaded fixed Base_T_Checkerboard:")
    print(base_T_checkerboard)
    print(f"\nLoaded robot transforms: {len(robot_transforms)}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    end_frame = total_frames if END_FRAME is None else min(END_FRAME, total_frames)

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {OUTPUT_VIDEO}")

    processed = 0
    missing_robot = 0
    invalid_projection = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    for frame_id in range(START_FRAME, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_id - START_FRAME) % FRAME_STEP != 0:
            continue

        overlay = frame.copy()
        base_T_gripper = robot_transforms.get(frame_id)

        if base_T_gripper is None:
            missing_robot += 1
            writer.write(overlay)
            processed += 1
            continue

        # Same verification chain as reproject_solved_base_checkerboard_0612.py:
        # Base_T_Camera_i = Base_T_Gripper_i @ X
        # Camera_T_Checkerboard_pred_i = inv(Base_T_Camera_i) @ fixed Base_T_Checkerboard
        base_T_camera = base_T_gripper @ gripper_T_camera
        camera_T_checkerboard_pred = np.linalg.inv(base_T_camera) @ base_T_checkerboard

        try:
            projected_points = project_checkerboard_points(
                camera_T_checkerboard_pred,
                object_points,
                K,
            )
        except (ValueError, np.linalg.LinAlgError):
            invalid_projection += 1
            writer.write(overlay)
            processed += 1
            continue

        draw_points(overlay, projected_points, color=(0, 0, 255), radius=4)

        cv2.putText(
            overlay,
            f"frame {frame_id}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(overlay)
        processed += 1

        if processed % 250 == 0:
            print(
                f"processed={processed} frame={frame_id} "
                f"missing_robot={missing_robot} invalid_projection={invalid_projection}"
            )

    cap.release()
    writer.release()

    print("\nDone.")
    print(f"Processed frames: {processed}")
    print(f"Frames missing robot transform: {missing_robot}")
    print(f"Frames skipped due to invalid projection: {invalid_projection}")
    print(f"Saved video: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
