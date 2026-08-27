from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
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
OUTPUT_NPZ = DATA_DIR / "xi_hand_eye_0616.npz"
TRANSFORM_FORMAT_REVIEW_CSV = DATA_DIR / "xi_transform_format_review.csv"

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


def build_xi_transform_from_values(values, layout):
    T = np.eye(4, dtype=np.float64)

    if layout == "row_major_3x4":
        T_temp = values.reshape(3, 4)
        T[:3, :3] = T_temp[:, :3]
        T[:3, 3] = T_temp[:, 3]
    elif layout == "translation_first":
        T[:3, 3] = values[:3]
        T[:3, :3] = values[3:].reshape(3, 3)
    else:
        raise ValueError(f"Unknown Xi transform layout: {layout}")

    return T


def transform_quality(T):
    R = T[:3, :3]
    t = T[:3, 3]

    if not np.isfinite(T).all():
        return {
            "det_R": np.nan,
            "orth_error": np.nan,
            "translation_norm": np.nan,
            "is_valid_rigid": False,
            "quality_score": np.inf,
        }

    det_R = float(np.linalg.det(R))
    orth_error = float(np.linalg.norm(R.T @ R - np.eye(3)))
    translation_norm = float(np.linalg.norm(t))
    is_valid_rigid = (
        np.isclose(det_R, 1.0, atol=1e-3)
        and orth_error <= 1e-3
        and translation_norm <= MAX_TRANSLATION_NORM_M
        and translation_norm >= 1e-6
    )

    return {
        "det_R": det_R,
        "orth_error": orth_error,
        "translation_norm": translation_norm,
        "is_valid_rigid": bool(is_valid_rigid),
        "quality_score": abs(det_R - 1.0) + orth_error,
    }


def parse_xi_raw_transform(transform_str, layout="translation_first", validate=True):
    values = parse_numbers(transform_str)

    if values.size != 12:
        raise ValueError(f"Expected 12 Xi transform values, got {values.size}")

    if layout == "auto":
        candidates = []
        for layout_name in ("row_major_3x4", "translation_first"):
            T_candidate = build_xi_transform_from_values(values, layout_name)
            quality = transform_quality(T_candidate)
            candidates.append((quality["quality_score"], layout_name, T_candidate))

        _, _, T = min(candidates, key=lambda item: item[0])
    else:
        T = build_xi_transform_from_values(values, layout)
        print(T)
        print("t")
        print(T[:3,3])
        print("R")
        print(T[:3,:3])

    if validate:
        validate_rigid_transform(T)

    return T


# def parse_transform_string(transform_str):
#     values = parse_numbers(transform_str)

#     if values.size == 16:
#         T = values.reshape(4, 4)
#         validate_rigid_transform(T)
#         return T

#     if values.size == 12:
#         return parse_xi_raw_transform(transform_str)

#     raise ValueError(f"Expected 12 or 16 transform values, got {values.size}")


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


def save_transform_format_review():
    df = pd.read_csv(MATCHED_TRANSFORMS_CSV, usecols=["frame_id", "Raw_Transform"])
    rows = []

    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])
        raw_transform = row["Raw_Transform"]

        record = {
            "frame_id": frame_id,
            "Raw_Transform": raw_transform,
        }

        try:
            values = parse_numbers(raw_transform)
            if values.size != 12:
                raise ValueError(f"Expected 12 values, got {values.size}")

            for layout in ("row_major_3x4", "translation_first"):
                T = build_xi_transform_from_values(values, layout)
                quality = transform_quality(T)
                record[f"{layout}_Transform"] = np.array2string(
                    T,
                    precision=10,
                    suppress_small=False,
                    separator=" ",
                )
                record[f"{layout}_det_R"] = quality["det_R"]
                record[f"{layout}_orth_error"] = quality["orth_error"]
                record[f"{layout}_translation_norm"] = quality["translation_norm"]
                record[f"{layout}_is_valid_rigid"] = quality["is_valid_rigid"]

        except ValueError as exc:
            record["parse_error"] = str(exc)

        rows.append(record)

    pd.DataFrame(rows).to_csv(TRANSFORM_FORMAT_REVIEW_CSV, index=False)
    print(f"Saved transform format review: {TRANSFORM_FORMAT_REVIEW_CSV}")


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
            # print(f"Frame {frame_id}: image not found: {image_path}")
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

    print(f"Detected checkerboard poses: {len(pose_results)}")

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


def A_matrix(layout="translation_first", validate=True):
    """
    Build robot-motion A matrices for Xi 0616.
    ROBOT ARM PORTION
    Uses the same frame pairs already accepted by B_matrix:
        A_ij = inv(Base_T_Tip_i) @ Base_T_Tip_j
    """
    robot = pd.read_csv(MATCHED_TRANSFORMS_CSV, usecols=["frame_id", "Raw_Transform"])
    camera_pairs = pd.read_csv(B_CSV_PATH, usecols=["from_frame", "to_frame"])

    if layout == "auto":
        A_csv_path = A_CSV_PATH
        A_npy_path = A_NPY_PATH
    else:
        A_csv_path = DATA_DIR / f"A_matrices_{layout}_METERS.csv"
        A_npy_path = DATA_DIR / f"A_matrices_{layout}_METERS.npy"

    robot_by_frame = {}
    for _, row in robot.iterrows():
        frame_id = int(row["frame_id"])
        try:
            robot_by_frame[frame_id] = parse_xi_raw_transform(
                row["Raw_Transform"],
                layout=layout,
                validate=validate,
            )
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
    np.save(A_npy_path, A_matrices)
    pd.DataFrame(A_records).to_csv(A_csv_path, index=False)

    print(f"Saved A matrices: {A_csv_path}")
    print(f"Saved A numpy array: {A_npy_path}")
    print(f"A_matrices shape: {A_matrices.shape}")

    return A_matrices, A_csv_path, A_npy_path


def project_to_rotation(R_mat):
    U, _, Vt = np.linalg.svd(R_mat)
    R_fixed = U @ Vt

    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1
        R_fixed = U @ Vt

    return R_fixed


def parse_pose_transform(transform_str):
    values = parse_numbers(transform_str)

    if values.size != 16:
        raise ValueError(f"Expected 16 checkerboard pose values, got {values.size}")

    return values.reshape(4, 4)


def load_absolute_pose_maps(layout="translation_first", validate=True):
    robot_df = pd.read_csv(MATCHED_TRANSFORMS_CSV, usecols=["frame_id", "Raw_Transform"])
    pose_df = pd.read_csv(POSE_CSV_PATH, usecols=["frame_id", "T_cam_board"])

    base_T_ecm_by_frame = {}
    for _, row in robot_df.iterrows():
        frame_id = int(row["frame_id"])
        try:
            base_T_ecm_by_frame[frame_id] = parse_xi_raw_transform(
                row["Raw_Transform"],
                layout=layout,
                validate=validate,
            )
        except ValueError as exc:
            print(f"Frame {frame_id}: skipped base_T_ECM for consistency check: {exc}")

    camera_T_checker_by_frame = {}
    for _, row in pose_df.iterrows():
        frame_id = int(row["frame_id"])
        try:
            camera_T_checker_by_frame[frame_id] = parse_pose_transform(
                row["T_cam_board"]
            )
        except ValueError as exc:
            print(f"Frame {frame_id}: skipped camera_T_checker for consistency check: {exc}")

    return base_T_ecm_by_frame, camera_T_checker_by_frame


def analyze_transform_deviation(T_list):
    T_list = np.asarray(T_list, dtype=np.float64)

    if T_list.ndim != 3 or T_list.shape[1:] != (4, 4):
        raise ValueError(f"T_list must have shape (N, 4, 4), got {T_list.shape}")

    R_list = np.array([project_to_rotation(T[:3, :3]) for T in T_list])
    t_list = T_list[:, :3, 3]

    T_mean = np.eye(4, dtype=np.float64)
    T_mean[:3, :3] = R.from_matrix(R_list).mean().as_matrix()
    T_mean[:3, 3] = np.mean(t_list, axis=0)

    T_mean_inv = np.linalg.inv(T_mean)
    translation_errors = []
    rotation_errors_deg = []

    for T_i in T_list:
        T_err = T_mean_inv @ T_i
        translation_errors.append(np.linalg.norm(T_err[:3, 3]))

        R_err = project_to_rotation(T_err[:3, :3])
        rotation_errors_deg.append(R.from_matrix(R_err).magnitude() * 180.0 / np.pi)

    return T_mean, np.array(translation_errors), np.array(rotation_errors_deg)


def find_base_T_checker(
    ECM_T_camera,
    matched_keys,
    layout="translation_first",
    validate=True,
):
    base_T_ecm_by_frame, camera_T_checker_by_frame = load_absolute_pose_maps(
        layout=layout,
        validate=validate,
    )

    frame_ids = sorted(
        {
            frame_id
            for pair in matched_keys
            for frame_id in pair
        }
    )

    base_T_checker_rows = []
    base_T_checker_list = []
    skipped_frames = []

    for frame_id in frame_ids:
        if frame_id not in base_T_ecm_by_frame or frame_id not in camera_T_checker_by_frame:
            skipped_frames.append(frame_id)
            continue

        base_T_checker = (
            base_T_ecm_by_frame[frame_id]
            @ ECM_T_camera
            @ camera_T_checker_by_frame[frame_id]
        )
        base_T_checker_list.append(base_T_checker)
        base_T_checker_rows.append(
            {
                "frame_id": frame_id,
                "Base_T_Checkerboard": np.array2string(
                    base_T_checker,
                    precision=10,
                    suppress_small=False,
                    separator=" ",
                ),
            }
        )

    if len(base_T_checker_list) < 2:
        raise RuntimeError(
            f"Need at least 2 Base_T_Checkerboard samples; got {len(base_T_checker_list)}"
        )

    mean_base_T_checker, translation_errors, rotation_errors_deg = (
        analyze_transform_deviation(base_T_checker_list)
    )

    if layout == "auto":
        base_checker_csv = DATA_DIR / "base_T_checkerboard_per_frame.csv"
    else:
        base_checker_csv = DATA_DIR / f"base_T_checkerboard_per_frame_{layout}.csv"
    pd.DataFrame(base_T_checker_rows).to_csv(base_checker_csv, index=False)

    print("\nBase_T_Checkerboard consistency")
    print(f"Frames used: {len(base_T_checker_list)}")
    if skipped_frames:
        print(f"Skipped frames: {skipped_frames}")

    print("\nMean Base_T_Checkerboard:")
    print(mean_base_T_checker)

    print("\nTranslation errors from mean:")
    print(f"Mean: {np.mean(translation_errors):.6f} m")
    print(f"RMS:  {np.sqrt(np.mean(translation_errors ** 2)):.6f} m")
    print(f"Max:  {np.max(translation_errors):.6f} m")

    print("\nRotation errors from mean:")
    print(f"Mean: {np.mean(rotation_errors_deg):.6f} deg")
    print(f"RMS:  {np.sqrt(np.mean(rotation_errors_deg ** 2)):.6f} deg")
    print(f"Max:  {np.max(rotation_errors_deg):.6f} deg")
    print(f"\nSaved per-frame Base_T_Checkerboard: {base_checker_csv}")

    return (
        mean_base_T_checker,
        np.array(base_T_checker_list),
        translation_errors,
        rotation_errors_deg,
    )


def solve_for_layout(layout="translation_first", validate=True):
    print(f"\n=== Solving Xi hand-eye with layout: {layout} ===")
    print(f"Robot transform validation: {validate}")

    _, a_csv_path, _ = A_matrix(layout=layout, validate=validate)

    A_keys, A_list = load_transform_list(a_csv_path)
    B_keys, B_list = load_transform_list(B_CSV_PATH)

    matched_keys, A_list, B_list = match_A_B(A_keys, A_list, B_keys, B_list)

    print(f"Matched A/B pairs for {layout}: {len(matched_keys)}")
    print(matched_keys)

    ECM_T_Camera, R_X, t_X, info = solve_X(A_list, B_list)

    print(f"\nECM_T_Camera for {layout}:")
    print(ECM_T_Camera)
    print(f"t_X norm for {layout}: {np.linalg.norm(t_X):.6f} m")
    print(info)

    (
        mean_base_T_checker,
        base_T_checker_list,
        base_T_checker_translation_errors,
        base_T_checker_rotation_errors_deg,
    ) = find_base_T_checker(
        ECM_T_Camera,
        matched_keys,
        layout=layout,
        validate=validate,
    )

    output_npz = DATA_DIR / f"xi_hand_eye_0616_{layout}.npz"
    np.savez(
        output_npz,
        X=ECM_T_Camera,
        R_X=R_X,
        t_X=t_X,
        matched_keys=np.array(matched_keys, dtype=int),
        mean_base_T_checker=mean_base_T_checker,
        base_T_checker_list=base_T_checker_list,
        base_T_checker_translation_errors=base_T_checker_translation_errors,
        base_T_checker_rotation_errors_deg=base_T_checker_rotation_errors_deg,
    )
    print(f"Saved hand-eye result for {layout}: {output_npz}")
    print("ECM_T_Camera incoming...")
    print(np.array2string(ECM_T_Camera, precision = 8, suppress_small=True))
    return {
        "layout": layout,
        "output_npz": output_npz,
        "matched_pair_count": len(matched_keys),
        "translation_rms": float(
            np.sqrt(np.mean(base_T_checker_translation_errors ** 2))
        ),
        "translation_mean": float(np.mean(base_T_checker_translation_errors)),
        "translation_max": float(np.max(base_T_checker_translation_errors)),
        "rotation_rms_deg": float(
            np.sqrt(np.mean(base_T_checker_rotation_errors_deg ** 2))
        ),
        "rotation_mean_deg": float(np.mean(base_T_checker_rotation_errors_deg)),
        "rotation_max_deg": float(np.max(base_T_checker_rotation_errors_deg)),
        "X_translation_norm": float(np.linalg.norm(t_X)),
    }

if __name__ == "__main__":

    print(DATA_DIR)
    B_matrix()
    save_transform_format_review()

    summaries = [
        solve_for_layout("translation_first", validate=True),
        # solve_for_layout("row_major_3x4", validate=False),
    ]

    summary_csv = DATA_DIR / "xi_hand_eye_format_comparison.csv"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    print(f"\nSaved format comparison summary: {summary_csv}")

