"""Visually and numerically verify the October 20 AX=XB calibration."""

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

from clean_si_api import checkerboard_inner_corners, estimate_checkerboard_pose
from solve_hand_eye_0612 import parse_transform_string


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "si_data_1020"
IMAGE_DIR = REPO_ROOT / "checkerboard_images" / "si_20261020" / "calibration"
OUTPUT_DIR = DATA_DIR / "checkerboard_reprojection"
MATCHED_CSV = DATA_DIR / "matched_transforms.csv"
CALIBRATION_NPZ = DATA_DIR / "hand_eye_1020_python.npz"
CAMERA_MAT = DATA_DIR / "camera_parameters.mat"
ERROR_CSV = OUTPUT_DIR / "reprojection_errors.csv"

PATTERN_SIZE = (7, 6)
SQUARE_SIZE_M = 0.005


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    camera = loadmat(CAMERA_MAT)
    K = np.asarray(camera["K"], dtype=np.float64)
    distortion = np.asarray(camera["distCoeffs"], dtype=np.float64).reshape(-1)
    object_points = checkerboard_inner_corners(
        columns=PATTERN_SIZE[0],
        rows=PATTERN_SIZE[1],
        square_size=SQUARE_SIZE_M,
    ).astype(np.float64)

    with np.load(CALIBRATION_NPZ, allow_pickle=False) as calibration:
        gripper_T_camera = calibration["X"].copy()
        base_T_checkerboard = calibration["mean_base_T_checkerboard"].copy()

    rows = []
    for _, record in pd.read_csv(MATCHED_CSV).iterrows():
        frame_id = int(record["frame_id"])
        image_path = IMAGE_DIR / str(record["image_filename"])
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Frame {frame_id}: could not read {image_path}")
            continue

        detected = estimate_checkerboard_pose(
            image, object_points, K, distortion, PATTERN_SIZE
        )
        if detected is None:
            print(f"Frame {frame_id}: checkerboard not detected")
            continue
        detected_corners = detected[3].reshape(-1, 2)

        base_T_gripper = parse_transform_string(record["Transform"])
        base_T_camera = base_T_gripper @ gripper_T_camera
        camera_T_checkerboard_predicted = (
            np.linalg.inv(base_T_camera) @ base_T_checkerboard
        )
        rvec, _ = cv2.Rodrigues(
            camera_T_checkerboard_predicted[:3, :3]
        )
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            camera_T_checkerboard_predicted[:3, 3],
            K,
            distortion,
        )
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - detected_corners, axis=1)

        annotated = image.copy()
        for detected_point, projected_point in zip(
            detected_corners, projected
        ):
            detected_xy = tuple(np.rint(detected_point).astype(int))
            projected_xy = tuple(np.rint(projected_point).astype(int))
            cv2.circle(annotated, detected_xy, 5, (0, 255, 0), -1)
            cv2.circle(annotated, projected_xy, 5, (0, 0, 255), -1)
            cv2.line(
                annotated, detected_xy, projected_xy, (0, 255, 255), 1
            )

        mean_error = float(errors.mean())
        rms_error = float(np.sqrt(np.mean(errors**2)))
        cv2.putText(
            annotated,
            f"Frame {frame_id}: mean {mean_error:.2f}px, "
            f"RMS {rms_error:.2f}px",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "Green: detected  Red: AX=XB prediction",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        output_path = OUTPUT_DIR / f"frame_{frame_id:06d}_reprojection.png"
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Could not write {output_path}")

        rows.append(
            {
                "frame_id": frame_id,
                "mean_error_px": mean_error,
                "rms_error_px": rms_error,
                "max_error_px": float(errors.max()),
                "output_image": output_path.name,
            }
        )

    if not rows:
        raise RuntimeError("No checkerboard frames could be reprojected.")

    errors_df = pd.DataFrame(rows)
    errors_df.to_csv(ERROR_CSV, index=False)
    print(f"Frames evaluated: {len(errors_df)}")
    print(
        f"Mean per-corner error: "
        f"{errors_df['mean_error_px'].mean():.3f} px"
    )
    print(
        f"RMS per-corner error: "
        f"{np.sqrt(np.mean(errors_df['rms_error_px'] ** 2)):.3f} px"
    )
    print(f"Worst corner error: {errors_df['max_error_px'].max():.3f} px")
    worst = errors_df.loc[errors_df["rms_error_px"].idxmax()]
    print(
        f"Worst frame by RMS: {int(worst['frame_id'])} "
        f"({worst['rms_error_px']:.3f} px)"
    )
    print(f"Saved annotated images to: {OUTPUT_DIR}")
    print(f"Saved errors to: {ERROR_CSV}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise

