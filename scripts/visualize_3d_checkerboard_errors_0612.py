from pathlib import Path
import sys

import cv2
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
    parse_transform_string,
    transform_points,
    image_path_for_row,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FOLDER = REPO_ROOT / "data" / "si_data_0612" / "checkerboard_3d_errors"
OUTPUT_PNG = OUTPUT_FOLDER / "checkerboard_3d_actual_vs_predicted.png"

COLUMNS = 7
ROWS = 6
SQUARE_SIZE_M = 0.005
SHOW_PLOT = True


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


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

    observed_boards = []
    observed_origins = []
    frame_ids = []

    df = pd.read_csv(MATCHED_TRANSFORMS_CSV)

    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])
        image_path = image_path_for_row(row)

        if not image_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        observed_pose = estimate_checkerboard_pose(
            image=image,
            object_points=object_points,
            K=K,
            dist_coeffs=DIST_COEFFS,
            pattern_size=PATTERN_SIZE,
        )

        if observed_pose is None:
            continue

        camera_T_checkerboard_observed, _, _, _ = observed_pose
        base_T_gripper = parse_transform_string(row["Transform"])
        observed_base_T_checkerboard = (
            base_T_gripper @ gripper_T_camera @ camera_T_checkerboard_observed
        )

        observed_points_base = transform_points(
            observed_base_T_checkerboard,
            object_points,
        )

        observed_boards.append(observed_points_base)
        observed_origins.append(observed_base_T_checkerboard[:3, 3])
        frame_ids.append(frame_id)

    if not observed_boards:
        raise RuntimeError("No observed checkerboard poses were available to plot.")

    observed_boards = np.array(observed_boards)
    observed_origins = np.array(observed_origins)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    for board_points in observed_boards:
        ax.scatter(
            board_points[:, 0],
            board_points[:, 1],
            board_points[:, 2],
            c="tab:blue",
            s=8,
            alpha=0.25,
            depthshade=False,
        )

    ax.scatter(
        predicted_points_base[:, 0],
        predicted_points_base[:, 1],
        predicted_points_base[:, 2],
        c="tab:red",
        s=45,
        label="Predicted fixed checkerboard corners",
        depthshade=False,
    )

    ax.scatter(
        observed_origins[:, 0],
        observed_origins[:, 1],
        observed_origins[:, 2],
        c="tab:green",
        s=24,
        label="Observed checkerboard origins",
        depthshade=False,
    )

    centroid_pred = predicted_points_base.mean(axis=0)
    centroid_obs = observed_boards.reshape(-1, 3).mean(axis=0)
    ax.scatter(
        [centroid_pred[0]],
        [centroid_pred[1]],
        [centroid_pred[2]],
        c="black",
        s=90,
        marker="x",
        label="Predicted centroid",
        depthshade=False,
    )
    ax.scatter(
        [centroid_obs[0]],
        [centroid_obs[1]],
        [centroid_obs[2]],
        c="tab:purple",
        s=90,
        marker="x",
        label="Observed centroid",
        depthshade=False,
    )

    ax.set_title("Checkerboard 3D Position in Base Frame")
    ax.set_xlabel("Base X (m)")
    ax.set_ylabel("Base Y (m)")
    ax.set_zlabel("Base Z (m)")
    ax.legend()
    set_axes_equal(ax)
    ax.view_init(elev=25, azim=-60)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200)

    print(f"Observed frames plotted: {len(frame_ids)}")
    print(f"Saved plot to: {OUTPUT_PNG}")

    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
