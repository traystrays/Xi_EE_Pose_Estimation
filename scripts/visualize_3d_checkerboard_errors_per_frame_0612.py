from pathlib import Path
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from clean_si_api import estimate_checkerboard_pose, checkerboard_inner_corners
from compute_3d_checkerboard_errors_0612 import (
    DIST_COEFFS,
    HAND_EYE_NPZ,
    IMAGE_FOLDER,
    LEFT_K_PATH,
    MATCHED_TRANSFORMS_CSV,
    PATTERN_SIZE,
    image_path_for_row,
    parse_transform_string,
    transform_points,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FOLDER = (
    REPO_ROOT / "data" / "si_data_0612" / "checkerboard_3d_errors" / "per_frame_plots"
)

COLUMNS = 7
ROWS = 6
SQUARE_SIZE_M = 0.005


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    radius = 0.5 * max(
        abs(x_limits[1] - x_limits[0]),
        abs(y_limits[1] - y_limits[0]),
        abs(z_limits[1] - z_limits[0]),
    )

    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


def plot_frame(frame_id, actual_points, predicted_points, output_path):
    error_vectors = predicted_points - actual_points
    errors_m = np.linalg.norm(error_vectors, axis=1)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        actual_points[:, 0],
        actual_points[:, 1],
        actual_points[:, 2],
        c="tab:green",
        s=34,
        label="Ground truth OpenCV corners",
        depthshade=False,
    )
    ax.scatter(
        predicted_points[:, 0],
        predicted_points[:, 1],
        predicted_points[:, 2],
        c="tab:red",
        s=24,
        label="Predicted corners",
        depthshade=False,
    )

    for actual, predicted in zip(actual_points, predicted_points):
        ax.plot(
            [actual[0], predicted[0]],
            [actual[1], predicted[1]],
            [actual[2], predicted[2]],
            color="tab:blue",
            linewidth=0.8,
            alpha=0.55,
        )

    ax.set_title(
        f"Frame {frame_id}: 3D Checkerboard Error\n"
        f"mean={np.mean(errors_m):.4f} m, "
        f"rms={np.sqrt(np.mean(errors_m**2)):.4f} m, "
        f"max={np.max(errors_m):.4f} m"
    )
    ax.set_xlabel("Base X (m)")
    ax.set_ylabel("Base Y (m)")
    ax.set_zlabel("Base Z (m)")
    ax.legend()
    set_axes_equal(ax)
    ax.view_init(elev=25, azim=-60)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
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
    predicted_points_base = transform_points(fixed_base_T_checkerboard, object_points)

    df = pd.read_csv(MATCHED_TRANSFORMS_CSV)
    plotted = 0

    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])
        image_path = image_path_for_row(row)

        if not image_path.exists():
            print(f"frame {frame_id}: image not found")
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

        # Ground truth for this frame: OpenCV solvePnP checkerboard pose.
        actual_base_T_checkerboard = base_T_camera @ camera_T_checkerboard_observed
        actual_points_base = transform_points(actual_base_T_checkerboard, object_points)

        output_path = OUTPUT_FOLDER / f"frame_{frame_id:06d}_3d_actual_vs_predicted.png"
        plot_frame(frame_id, actual_points_base, predicted_points_base, output_path)
        plotted += 1

    print(f"Saved per-frame 3D plots: {plotted}")
    print(f"Output folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
