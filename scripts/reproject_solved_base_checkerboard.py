from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

from clean_si_api import estimate_checkerboard_pose, checkerboard_inner_corners


REPO_ROOT = Path(__file__).resolve().parents[1]

LEFT_K_PATH = Path(r"C:\Users\Tracy\Documents\MATLAB\20260616_xi_data\K_left.mat")
K = loadmat(LEFT_K_PATH)["K_left"].astype(np.float64)
print(K)
IMAGE_FOLDER = REPO_ROOT / "checkerboard_images" / "xi_20260616" / "hand_eye_calibration"
MATCHED_TRANSFORMS_CSV = REPO_ROOT / "data" / "xi_data_0616" / "matched_transforms.csv"
HAND_EYE_NPZ = REPO_ROOT / "data" / "xi_data_0616" / "xi_hand_eye_0616.npz"
OUTPUT_FOLDER = IMAGE_FOLDER.parent / "solved_base_checkerboard_reprojected"
ERROR_CSV = OUTPUT_FOLDER / "solved_base_checkerboard_reprojection_errors.csv"

COLUMNS = 7
ROWS = 6
SQUARE_SIZE_M = 0.010 # 10mm in meters
PATTERN_SIZE = (COLUMNS, ROWS)

DIST_COEFFS = np.array(
    [-0.002994058031627, 0.013201245121718, 0.0, 0.0, 0.0],
    dtype=np.float64,
)

DISPLAY_IMAGES = True


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


def draw_points(image, points, color, radius):
    h, w = image.shape[:2]
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    for point in points:
        u = int(round(point[0]))
        v = int(round(point[1]))
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(image, (u, v), radius, color, -1)


def main():
    np.set_printoptions(precision=8, suppress=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print(K)

    object_points = checkerboard_inner_corners(
        columns=COLUMNS,
        rows=ROWS,
        square_size=SQUARE_SIZE_M,
    ).astype(np.float64)

    solved = np.load(HAND_EYE_NPZ)
    ECM_T_camera = solved["X"]
    
    base_T_checkerboard = solved["mean_base_T_checker"]

    print("Loaded X = Gripper_T_Camera:")
    print(ECM_T_camera)

    print("\nLoaded fixed Base_T_Checkerboard:")
    print(base_T_checkerboard)

    df = pd.read_csv(MATCHED_TRANSFORMS_CSV)

    all_pixel_errors = []
    rows = []
    output_paths = []

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

        _, _, _, detected_corners = observed_pose
        detected_corners = detected_corners.reshape(-1, 2)

        base_T_gripper = parse_transform_string(row["Transform"])
        base_T_camera = base_T_gripper @ ECM_T_camera
        camera_T_base = np.linalg.inv(base_T_camera)
        camera_T_checkerboard_pred = camera_T_base @ base_T_checkerboard

        R_pred = project_to_rotation(camera_T_checkerboard_pred[:3, :3])
        t_pred = camera_T_checkerboard_pred[:3, 3].reshape(3, 1)
        rvec_pred, _ = cv2.Rodrigues(R_pred)

        projected_corners, _ = cv2.projectPoints(
            object_points,
            rvec_pred,
            t_pred,
            K,
            DIST_COEFFS,
        )
        projected_corners = projected_corners.reshape(-1, 2)

        pixel_errors = np.linalg.norm(projected_corners - detected_corners, axis=1)
        mean_px = float(np.mean(pixel_errors))
        rms_px = float(np.sqrt(np.mean(pixel_errors**2)))
        max_px = float(np.max(pixel_errors))

        all_pixel_errors.extend(pixel_errors.tolist())

        rows.append(
            {
                "frame_id": frame_id,
                "image_filename": image_path.name,
                "mean_pixel_error": mean_px,
                "rms_pixel_error": rms_px,
                "max_pixel_error": max_px,
            }
        )

        overlay = image.copy()

        # Green: detected checkerboard corners. Red: fixed-base prediction.
        draw_points(overlay, detected_corners, color=(0, 255, 0), radius=4)
        draw_points(overlay, projected_corners, color=(0, 0, 255), radius=4)

        h, w = overlay.shape[:2]
        for detected, projected in zip(detected_corners, projected_corners):
            detected = np.asarray(detected, dtype=np.float64).reshape(-1)
            projected = np.asarray(projected, dtype=np.float64).reshape(-1)

            if detected.size < 2 or projected.size < 2:
                continue

            if not (
                np.isfinite(detected[:2]).all()
                and np.isfinite(projected[:2]).all()
            ):
                continue

            p1 = (int(round(float(detected[0]))), int(round(float(detected[1]))))
            p2 = (int(round(float(projected[0]))), int(round(float(projected[1]))))

            p1_in_image = 0 <= p1[0] < w and 0 <= p1[1] < h
            p2_in_image = 0 <= p2[0] < w and 0 <= p2[1] < h

            print(
                f"frame {frame_id}: "
                f"detected=({p1[0]}, {p1[1]}) in_image={p1_in_image} | "
                f"projected=({p2[0]}, {p2[1]}) in_image={p2_in_image}"
            )

        label = (
            f"frame {frame_id} | mean {mean_px:.2f}px | "
            f"rms {rms_px:.2f}px | max {max_px:.2f}px"
        )
        cv2.putText(
            overlay,
            label,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "green=detected red=base-predicted",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        output_path = OUTPUT_FOLDER / f"frame_{frame_id:06d}_solved_base_reprojection.jpg"
        cv2.imwrite(str(output_path), overlay)
        output_paths.append(output_path)

        print(
            f"frame {frame_id}: mean={mean_px:.2f}px "
            f"rms={rms_px:.2f}px max={max_px:.2f}px"
        )

    if not rows:
        raise RuntimeError("No frames were reprojected.")

    error_df = pd.DataFrame(rows)
    error_df.to_csv(ERROR_CSV, index=False)

    all_pixel_errors = np.array(all_pixel_errors)

    print("\nOverall reprojection error:")
    print(f"Mean pixel error: {np.mean(all_pixel_errors):.2f} px")
    print(f"RMS pixel error: {np.sqrt(np.mean(all_pixel_errors**2)):.2f} px")
    print(f"Max pixel error: {np.max(all_pixel_errors):.2f} px")
    print(f"\nSaved overlays to: {OUTPUT_FOLDER}")
    print(f"Saved per-frame errors to: {ERROR_CSV}")

    if DISPLAY_IMAGES:
        print("\nDisplaying overlays. Press any key for next image; Esc or q exits.")

        for output_path in output_paths:
            image = cv2.imread(str(output_path))
            if image is None:
                print(f"Could not read display image: {output_path}")
                continue

            cv2.imshow("Solved Base Checkerboard Reprojection", image)
            key = cv2.waitKey(0) & 0xFF

            if key in (27, ord("q")):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
