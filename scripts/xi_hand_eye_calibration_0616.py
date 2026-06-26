from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from clean_si_api import estimate_checkerboard_pose, checkerboard_inner_corners
from hand_eye_calib import solve_X, load_transform_list, match_A_B


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "xi_data_0616"
IMAGE_FOLDER = REPO_ROOT / "checkerboard_images" / "xi_20260616" / "hand_eye_calibration"

MATCHED_TRANSFORMS_CSV = DATA_DIR / "matched_transforms.csv"
LEFT_K_PATH = Path(r"C:\Users\Tracy\Documents\MATLAB\20260616_xi_data\K_left.mat")

B_NPY_PATH = DATA_DIR / "B_matrices_METERS.npy"
B_CSV_PATH = DATA_DIR / "B_matrices_METERS.csv"
A_NPY_PATH = DATA_DIR / "A_matrices_METERS.npy"
A_CSV_PATH = DATA_DIR / "A_matrices_METERS.csv"
POSE_CSV_PATH = DATA_DIR / "checkerboard_pose_results.csv"

CHECKERBOARD_OUTPUT_DIR = IMAGE_FOLDER.parent / "checkerboard_pose_result"

COLUMNS = 7
ROWS = 6
SQUARE_SIZE_M = 0.010
PATTERN_SIZE = (COLUMNS, ROWS)

DIST_COEFFS = np.array(
    [-0.002994058031627, 0.013201245121718, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
MAX_TRANSLATION_NORM_M = 5.0


def parse_numbers(transform_str):
    if not isinstance(transform_str, str):
        raise ValueError("Transform value is not a string")

    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", transform_str)
    return np.array(numbers, dtype=np.float64)


def parse_xi_raw_transform(transform_str):
    values = parse_numbers(transform_str)

    if values.size != 12:
        raise ValueError(f"Expected 12 Xi transform values, got {values.size}")

    T = np.eye(4, dtype=np.float64)

    # Xi ISI_TIP_TRANSFORM format:
    # [tx ty tz R00 R01 R02 R10 R11 R12 R20 R21 R22]
    T[:3, 3] = values[:3]
    T[:3, :3] = values[3:].reshape(3, 3)

    validate_rigid_transform(T)
    return T


def parse_transform_string(transform_str):
    values = parse_numbers(transform_str)

    if values.size == 16:
        T = values.reshape(4, 4)
        validate_rigid_transform(T)
        return T

    if values.size == 12:
        return parse_xi_raw_transform(transform_str)

    raise ValueError(f"Expected 12 or 16 transform values, got {values.size}")


def validate_rigid_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]

    if not np.isfinite(T).all():
        raise ValueError("Transform contains nan or inf")

    det_R = np.linalg.det(R)
    orth_error = np.linalg.norm(R.T @ R - np.eye(3))
    translation_norm = np.linalg.norm(t)

    if not np.isclose(det_R, 1.0, atol=1e-3):
        raise ValueError(f"Rotation determinant is not near +1: {det_R}")

    if orth_error > 1e-3:
        raise ValueError(f"Rotation is not orthonormal: error={orth_error}")

    if translation_norm > MAX_TRANSLATION_NORM_M:
        raise ValueError(
            f"Translation norm is too large for meters: {translation_norm}"
        )


def flatten_matrix_record(prefix, T):
    record = {}
    for r in range(4):
        for c in range(4):
            record[f"{prefix}{r}{c}"] = T[r, c]
    return record


def image_path_for_row(row):
    if "image_filename" in row and isinstance(row["image_filename"], str):
        image_path = IMAGE_FOLDER / row["image_filename"]
        if image_path.exists():
            return image_path

    frame_id = int(row["frame_id"])
    candidates = [
        IMAGE_FOLDER / f"frame_{frame_id:06d}.png",
        IMAGE_FOLDER / f"frame_{frame_id:06d}.jpg",
        IMAGE_FOLDER / f"frame_{frame_id}.png",
        IMAGE_FOLDER / f"frame_{frame_id}.jpg",
    ]
    for image_path in candidates:
        if image_path.exists():
            return image_path

    return candidates[0]


def B_matrix():
    """
    Build camera-motion B matrices for Xi 0616.
    CAMERA PORTION.
    For adjacent accepted checkerboard frames:
        B_ij = Camera_T_Checkerboard_i @ inv(Camera_T_Checkerboard_j)
    """
    CHECKERBOARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    K = loadmat(LEFT_K_PATH)["K_left"].astype(np.float64)
    object_points = checkerboard_inner_corners(
        columns=COLUMNS,
        rows=ROWS,
        square_size=SQUARE_SIZE_M,
    ).astype(np.float64)

    matched = pd.read_csv(MATCHED_TRANSFORMS_CSV)
    pose_results = []
    pose_records = []

    for _, row in matched.iterrows():
        frame_id = int(row["frame_id"])
        image_path = image_path_for_row(row)

        if not image_path.exists():
            print(f"Frame {frame_id}: image not found: {image_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Frame {frame_id}: image could not be read: {image_path}")
            continue

        result = estimate_checkerboard_pose(
            image=image,
            object_points=object_points,
            K=K,
            dist_coeffs=DIST_COEFFS,
            pattern_size=PATTERN_SIZE,
        )
        if result is None:
            print(f"Frame {frame_id}: checkerboard not detected")
            continue

        T_cam_board, rvec, tvec, corners = result
        pose_results.append(
            {
                "frame_id": frame_id,
                "T_cam_board": T_cam_board,
            }
        )
        pose_records.append(
            {
                "frame_id": frame_id,
                "image_filename": image_path.name,
                "rvec": np.array2string(rvec.flatten(), separator=" "),
                "tvec": np.array2string(tvec.flatten(), separator=" "),
                "T_cam_board": np.array2string(T_cam_board, separator=" "),
            }
        )

        image_with_corners = image.copy()
        cv2.drawChessboardCorners(
            image_with_corners,
            PATTERN_SIZE,
            corners,
            True,
        )
        cv2.imwrite(
            str(CHECKERBOARD_OUTPUT_DIR / f"frame_{frame_id:06d}_corners.png"),
            image_with_corners,
        )

    if len(pose_results) < 2:
        raise RuntimeError(
            f"Need at least 2 detected checkerboard poses to build B; got {len(pose_results)}"
        )

    B_matrices = []
    B_records = []

    for i in range(len(pose_results) - 1):
        frame_i = pose_results[i]["frame_id"]
        frame_j = pose_results[i + 1]["frame_id"]
        T_i = pose_results[i]["T_cam_board"]
        T_j = pose_results[i + 1]["T_cam_board"]

        B_ij = T_i @ np.linalg.inv(T_j)
        B_matrices.append(B_ij)

        record = {
            "from_frame": frame_i,
            "to_frame": frame_j,
        }
        record.update(flatten_matrix_record("B", B_ij))
        B_records.append(record)

    B_matrices = np.array(B_matrices)
    np.save(B_NPY_PATH, B_matrices)
    pd.DataFrame(B_records).to_csv(B_CSV_PATH, index=False)
    pd.DataFrame(pose_records).to_csv(POSE_CSV_PATH, index=False)

    print(f"Saved B matrices: {B_CSV_PATH}")
    print(f"Saved B numpy array: {B_NPY_PATH}")
    print(f"Saved checkerboard poses: {POSE_CSV_PATH}")
    print(f"B_matrices shape: {B_matrices.shape}")

    return B_matrices


def A_matrix():
    """
    Build robot-motion A matrices for Xi 0616.
    ROBOT ARM PORTION
    Uses the same frame pairs already accepted by B_matrix:
        A_ij = inv(Base_T_Tip_i) @ Base_T_Tip_j
    """
    robot = pd.read_csv(MATCHED_TRANSFORMS_CSV, usecols=["frame_id", "Raw_Transform"])
    camera_pairs = pd.read_csv(B_CSV_PATH, usecols=["from_frame", "to_frame"])

    robot_by_frame = {}
    for _, row in robot.iterrows():
        frame_id = int(row["frame_id"])
        try:
            robot_by_frame[frame_id] = parse_xi_raw_transform(row["Raw_Transform"])
        except ValueError as exc:
            print(f"Frame {frame_id}: invalid robot transform skipped: {exc}")

    A_matrices = []
    A_records = []

    for _, row in camera_pairs.iterrows():
        frame_i = int(row["from_frame"])
        frame_j = int(row["to_frame"])

        if frame_i not in robot_by_frame or frame_j not in robot_by_frame:
            print(f"Missing robot transform for pair {frame_i}->{frame_j}; skipping")
            continue

        try:
            A_ij = np.linalg.inv(robot_by_frame[frame_i]) @ robot_by_frame[frame_j]
        except np.linalg.LinAlgError as exc:
            print(f"Cannot invert robot transform for frame {frame_i}: {exc}")
            continue

        A_matrices.append(A_ij)
        record = {
            "from_frame": frame_i,
            "to_frame": frame_j,
        }
        record.update(flatten_matrix_record("A", A_ij))
        A_records.append(record)

    if not A_matrices:
        raise RuntimeError("No A matrices were computed.")

    A_matrices = np.array(A_matrices)
    np.save(A_NPY_PATH, A_matrices)
    pd.DataFrame(A_records).to_csv(A_CSV_PATH, index=False)

    print(f"Saved A matrices: {A_CSV_PATH}")
    print(f"Saved A numpy array: {A_NPY_PATH}")
    print(f"A_matrices shape: {A_matrices.shape}")

    return A_matrices



if __name__ == "__main__":
    B_matrix()
    A_matrix()

    A_keys, A_list = load_transform_list(A_CSV_PATH)
    B_keys, B_list = load_transform_list(B_CSV_PATH)

    matched_keys, A_list, B_list = match_A_B(A_keys, A_list, B_keys, B_list)

    print(matched_keys)

    X, R_X, t_X, info = solve_X(A_list, B_list)

    print(X)
