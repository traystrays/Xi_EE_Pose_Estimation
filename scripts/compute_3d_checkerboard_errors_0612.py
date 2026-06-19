from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R

from clean_si_api import estimate_checkerboard_pose, checkerboard_inner_corners


REPO_ROOT = Path(__file__).resolve().parents[1]

LEFT_K_PATH = Path(
    r"C:\Users\Tracy\Documents\MATLAB\20260612_si_data\cameraParams_5mm_leftK.mat"
)
IMAGE_FOLDER = REPO_ROOT / "checkerboard_images" / "si_20260612" / "calibration"
MATCHED_TRANSFORMS_CSV = REPO_ROOT / "data" / "si_data_0612" / "matched_transforms.csv"
HAND_EYE_NPZ = REPO_ROOT / "data" / "si_data_0612" / "hand_eye_0612_python.npz"
OUTPUT_FOLDER = REPO_ROOT / "data" / "si_data_0612" / "checkerboard_3d_errors"
FRAME_ERROR_CSV = OUTPUT_FOLDER / "frame_3d_errors.csv"
CORNER_ERROR_CSV = OUTPUT_FOLDER / "corner_3d_errors.csv"

COLUMNS = 7
ROWS = 6
SQUARE_SIZE_M = 0.005
PATTERN_SIZE = (COLUMNS, ROWS)

DIST_COEFFS = np.array(
    [-0.306143873919413, 0.4667777390935303, 0.0, 0.0, 0.0],
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


def transform_points(T, points_3d):
    points_3d = np.asarray(points_3d, dtype=np.float64)
    points_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    transformed_h = (T @ points_h.T).T
    return transformed_h[:, :3]


def project_to_rotation(R_mat):
    U, _, Vt = np.linalg.svd(R_mat)
    R_fixed = U @ Vt

    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1
        R_fixed = U @ Vt

    return R_fixed


def pose_error(reference_T_observed, reference_T_predicted):
    error_T = np.linalg.inv(reference_T_predicted) @ reference_T_observed
    translation_error_m = np.linalg.norm(error_T[:3, 3])
    rotation_error_deg = (
        R.from_matrix(project_to_rotation(error_T[:3, :3])).magnitude()
        * 180.0
        / np.pi
    )
    return translation_error_m, rotation_error_deg


def main():
    np.set_printoptions(precision=8, suppress=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    K = loadmat(LEFT_K_PATH)["cameraParams_5mmK"].astype(np.float64)
    object_points = checkerboard_inner_corners(
        columns=COLUMNS,
        rows=ROWS,
        square_size=SQUARE_SIZE_M,
    ).astype(np.float64)

    solved = np.load(HAND_EYE_NPZ)
    gripper_T_camera = solved["X"]
    fixed_base_T_checkerboard = solved["mean_base_T_checkerboard"]

    fixed_points_base = transform_points(fixed_base_T_checkerboard, object_points)

    frame_rows = []
    corner_rows = []

    df = pd.read_csv(MATCHED_TRANSFORMS_CSV)

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

        observed_pose = estimate_checkerboard_pose(
            image=image,
            object_points=object_points,
            K=K,
            dist_coeffs=DIST_COEFFS,
            pattern_size=PATTERN_SIZE,
        )

        if observed_pose is None:
            print(f"frame {frame_id}: checkerboard not detected")
            continue

        camera_T_checkerboard_observed, _, _, _ = observed_pose
        base_T_gripper = parse_transform_string(row["Transform"])
        base_T_camera = base_T_gripper @ gripper_T_camera
        observed_base_T_checkerboard = base_T_camera @ camera_T_checkerboard_observed

        observed_points_base = transform_points(
            observed_base_T_checkerboard,
            object_points,
        )

        corner_vectors_m = observed_points_base - fixed_points_base
        corner_errors_m = np.linalg.norm(corner_vectors_m, axis=1)
        mean_corner_vector_m = np.mean(corner_vectors_m, axis=0)
        rms_corner_vector_m = np.sqrt(np.mean(corner_vectors_m**2, axis=0))

        translation_error_m, rotation_error_deg = pose_error(
            observed_base_T_checkerboard,
            fixed_base_T_checkerboard,
        )

        frame_rows.append(
            {
                "frame_id": frame_id,
                "image_filename": image_path.name,
                "mean_corner_error_m": float(np.mean(corner_errors_m)),
                "rms_corner_error_m": float(np.sqrt(np.mean(corner_errors_m**2))),
                "max_corner_error_m": float(np.max(corner_errors_m)),
                "mean_error_x_m": float(mean_corner_vector_m[0]),
                "mean_error_y_m": float(mean_corner_vector_m[1]),
                "mean_error_z_m": float(mean_corner_vector_m[2]),
                "rms_error_x_m": float(rms_corner_vector_m[0]),
                "rms_error_y_m": float(rms_corner_vector_m[1]),
                "rms_error_z_m": float(rms_corner_vector_m[2]),
                "pose_translation_error_m": float(translation_error_m),
                "pose_rotation_error_deg": float(rotation_error_deg),
            }
        )

        for corner_index, (error_m, vector_m) in enumerate(
            zip(corner_errors_m, corner_vectors_m)
        ):
            corner_rows.append(
                {
                    "frame_id": frame_id,
                    "image_filename": image_path.name,
                    "corner_index": corner_index,
                    "error_m": float(error_m),
                    "error_x_m": float(vector_m[0]),
                    "error_y_m": float(vector_m[1]),
                    "error_z_m": float(vector_m[2]),
                }
            )

    if not frame_rows:
        raise RuntimeError("No 3D errors were computed.")

    frame_df = pd.DataFrame(frame_rows)
    corner_df = pd.DataFrame(corner_rows)

    frame_df.to_csv(FRAME_ERROR_CSV, index=False)
    corner_df.to_csv(CORNER_ERROR_CSV, index=False)

    all_corner_errors = corner_df["error_m"].to_numpy()
    all_error_vectors = corner_df[["error_x_m", "error_y_m", "error_z_m"]].to_numpy()
    rms_error_vector = np.sqrt(np.mean(all_error_vectors**2, axis=0))

    print("3D checkerboard corner error summary:")
    print(f"Frames evaluated: {len(frame_df)}")
    print(f"Mean corner error: {np.mean(all_corner_errors):.6f} m")
    print(f"RMS corner error: {np.sqrt(np.mean(all_corner_errors**2)):.6f} m")
    print(f"Max corner error: {np.max(all_corner_errors):.6f} m")
    print("\nAxis-wise RMS 3D error:")
    print(f"RMS X error: {rms_error_vector[0]:.6f} m")
    print(f"RMS Y error: {rms_error_vector[1]:.6f} m")
    print(f"RMS Z error: {rms_error_vector[2]:.6f} m")

    print("\nPose-level error summary:")
    print(
        "Translation RMS error: "
        f"{np.sqrt(np.mean(frame_df['pose_translation_error_m'].to_numpy() ** 2)):.6f} m"
    )
    print(
        "Rotation RMS error: "
        f"{np.sqrt(np.mean(frame_df['pose_rotation_error_deg'].to_numpy() ** 2)):.6f} deg"
    )

    print(f"\nSaved frame errors to: {FRAME_ERROR_CSV}")
    print(f"Saved corner errors to: {CORNER_ERROR_CSV}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
