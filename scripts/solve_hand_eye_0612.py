from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R

from clean_si_api import estimate_checkerboard_pose, checkerboard_inner_corners
from hand_eye_calib import solve_X, residual_errors


REPO_ROOT = Path(__file__).resolve().parents[1]

LEFT_K_PATH = Path(r"C:\Users\Tracy\Documents\MATLAB\20260612_si_data\K_left.mat")
IMAGE_FOLDER = REPO_ROOT / "checkerboard_images" / "si_20260612" / "calibration"
MATCHED_TRANSFORMS_CSV = REPO_ROOT / "data" / "si_data_0612" / "matched_transforms.csv"
OUTPUT_NPZ = REPO_ROOT / "data" / "si_data_0612" / "hand_eye_0612_python.npz"

COLUMNS = 7
ROWS = 6
SQUARE_SIZE_M = 0.005
PATTERN_SIZE = (COLUMNS, ROWS)

DIST_COEFFS = np.array(
    [-0.297863212023482, 0.403564940306343, 0.0, 0.0, 0.0],
    dtype=np.float64,
)


def parse_transform_string(transform_str):
    values = np.fromstring(
        transform_str.replace("[", " ").replace("]", " ").replace(",", " "),
        sep=" ",
    )

    if values.size != 16:
        raise ValueError(f"Expected 16 transform values, got {values.size}")

    return values.reshape(4, 4)


def project_to_rotation(R_mat):
    U, _, Vt = np.linalg.svd(R_mat)
    R_fixed = U @ Vt

    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1
        R_fixed = U @ Vt

    return R_fixed


def analyze_transform_deviation(T_list):
    T_list = np.asarray(T_list, dtype=np.float64)

    if T_list.ndim != 3 or T_list.shape[1:] != (4, 4):
        raise ValueError(f"T_list must have shape (N, 4, 4), got {T_list.shape}")

    R_list = np.array([project_to_rotation(T[:3, :3]) for T in T_list])
    t_list = T_list[:, :3, 3]

    T_mean = np.eye(4)
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


def image_path_for_row(row):
    if "image_filename" in row and isinstance(row["image_filename"], str):
        path = IMAGE_FOLDER / row["image_filename"]
        if path.exists():
            return path

    frame_id = int(row["frame_id"])
    candidates = [
        IMAGE_FOLDER / f"frame_{frame_id:06d}.png",
        IMAGE_FOLDER / f"frame_00{frame_id}.png",
        IMAGE_FOLDER / f"frame_{frame_id}.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def load_absolute_poses():
    K = loadmat(LEFT_K_PATH)["K"].astype(np.float64)
    object_points = checkerboard_inner_corners(
        columns=COLUMNS,
        rows=ROWS,
        square_size=SQUARE_SIZE_M,
    ).astype(np.float64)

    df = pd.read_csv(MATCHED_TRANSFORMS_CSV)

    frame_ids = []
    base_T_gripper_list = []
    camera_T_checkerboard_list = []

    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])
        image_path = image_path_for_row(row)

        if not image_path.exists():
            print(f"frame {frame_id}: image not found at {image_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"frame {frame_id}: image could not be read")
            continue

        result = estimate_checkerboard_pose(
            image=image,
            object_points=object_points,
            K=K,
            dist_coeffs=DIST_COEFFS,
            pattern_size=PATTERN_SIZE,
        )

        if result is None:
            print(f"frame {frame_id}: checkerboard not detected")
            continue

        camera_T_checkerboard, _, _, _ = result
        base_T_gripper = parse_transform_string(row["Transform"])

        frame_ids.append(frame_id)
        base_T_gripper_list.append(base_T_gripper)
        camera_T_checkerboard_list.append(camera_T_checkerboard)

    if len(frame_ids) < 3:
        raise RuntimeError(f"Need at least 3 accepted frames; got {len(frame_ids)}")

    return (
        np.array(frame_ids, dtype=int),
        np.array(base_T_gripper_list),
        np.array(camera_T_checkerboard_list),
    )


def build_relative_motions(frame_ids, base_T_gripper_list, camera_T_checkerboard_list):
    keys = []
    A_list = []
    B_list = []

    for i in range(len(frame_ids) - 1):
        A_ij = np.linalg.inv(base_T_gripper_list[i]) @ base_T_gripper_list[i + 1]
        B_ij = camera_T_checkerboard_list[i] @ np.linalg.inv(
            camera_T_checkerboard_list[i + 1]
        )

        keys.append((int(frame_ids[i]), int(frame_ids[i + 1])))
        A_list.append(A_ij)
        B_list.append(B_ij)

    return keys, np.array(A_list), np.array(B_list)


def main():
    np.set_printoptions(precision=8, suppress=True)

    frame_ids, base_T_gripper_list, camera_T_checkerboard_list = load_absolute_poses()

    print(f"Accepted frames: {len(frame_ids)}")
    print(f"First frame: {frame_ids[0]}")
    print(f"Last frame: {frame_ids[-1]}")

    keys, A_list, B_list = build_relative_motions(
        frame_ids,
        base_T_gripper_list,
        camera_T_checkerboard_list,
    )

    gripper_T_camera, R_X, t_X, info = solve_X(A_list, B_list)

    print("\nSolved hand-eye X = Gripper_T_Camera:")
    print(gripper_T_camera)

    print("\nSolver info:")
    for key, value in info.items():
        print(f"{key}: {value}")

    residual_df = residual_errors(A_list, B_list, gripper_T_camera, keys=keys)

    print("\nAX = XB residual summary:")
    print(residual_df[["rotation_error_deg", "translation_error"]].describe())

    base_T_checkerboard_list = np.array(
        [
            base_T_gripper @ gripper_T_camera @ camera_T_checkerboard
            for base_T_gripper, camera_T_checkerboard in zip(
                base_T_gripper_list,
                camera_T_checkerboard_list,
            )
        ]
    )

    mean_base_T_checkerboard, translation_errors, rotation_errors_deg = (
        analyze_transform_deviation(base_T_checkerboard_list)
    )

    print("\nMean Base_T_Checkerboard:")
    print(mean_base_T_checkerboard)

    print("\nFixed checkerboard consistency:")
    print(f"Translation mean error: {np.mean(translation_errors):.6f} m")
    print(f"Translation RMS error: {np.sqrt(np.mean(translation_errors ** 2)):.6f} m")
    print(f"Translation max error: {np.max(translation_errors):.6f} m")
    print(f"Rotation mean error: {np.mean(rotation_errors_deg):.6f} deg")
    print(f"Rotation RMS error: {np.sqrt(np.mean(rotation_errors_deg ** 2)):.6f} deg")
    print(f"Rotation max error: {np.max(rotation_errors_deg):.6f} deg")

    np.savez(
        OUTPUT_NPZ,
        frame_ids=frame_ids,
        X=gripper_T_camera,
        R_X=R_X,
        t_X=t_X,
        mean_base_T_checkerboard=mean_base_T_checkerboard,
        translation_errors=translation_errors,
        rotation_errors_deg=rotation_errors_deg,
    )

    print(f"\nSaved result to: {OUTPUT_NPZ}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
