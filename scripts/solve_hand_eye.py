"""Solve SI hand-eye calibration with the existing AX=XB code."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import savemat


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "si_data_1020"
IMAGE_DIR = REPO_ROOT / "checkerboard_images" / "si_20261020" / "calibration"
SAVED_FRAMES = DATA_DIR / "saved_frames.csv"
SYNCED_TRANSFORMS = DATA_DIR / "hand_eye_calibration_data_METERS.csv"
MATCHED_TRANSFORMS = DATA_DIR / "matched_transforms.csv"
CAMERA_PARAMETERS = DATA_DIR / "camera_parameters.mat"
OUTPUT_NPZ = DATA_DIR / "hand_eye_1020_python_1.npz"

print(f"Using data directory: {DATA_DIR}")


# CAMERA_MATRIX = np.array(
#     [
#         [1.549987410854589e3, 0.0, 9.475736024485624e2],
#         [0.0, 1.551038702981481e3, 6.681986905075408e2],
#         [0.0, 0.0, 1.0],
#     ],
#     dtype=np.float64,
# )

# # OpenCV order: k1, k2, p1, p2, k3.
# DIST_COEFFS = np.array(
#     [-0.289954027206988, 0.292060933726558, 0.0, 0.0, 0.0],
#     dtype=np.float64,
# )

CAMERA_MATRIX = np.array(
    [
        [1580.6597160812203, 0.0, 993.6383389647222],
        [0.0, 1582.3660553768693, 593.4610147347427],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# OpenCV order: k1, k2, p1, p2, k3.
DIST_COEFFS = np.array(
    [-0.28357122637446786, -0.28357122637446786, -0.006334614947363842, 0.0012504656853484606, 0.39038317041842135],
    dtype=np.float64,
)

def prepare_selected_transforms() -> int:
    """Match the hand-picked frame list to synchronized ECM transforms."""
    selected = pd.read_csv(SAVED_FRAMES)
    transforms = pd.read_csv(SYNCED_TRANSFORMS)
    selected["frame_number"] = selected["frame_number"].astype(int)
    transforms["frame_id"] = transforms["frame_id"].astype(int)

    matched = selected.merge(
        transforms,
        left_on="frame_number",
        right_on="frame_id",
        how="left",
        validate="one_to_one",
    )
    invalid = matched[
        matched["Transform"].isna()
        | matched["Transform"].astype(str).str.contains(
            r"\bnan\b", case=False, regex=True
        )
    ]
    if not invalid.empty:
        raise RuntimeError(
            "Selected frame(s) have no valid ECM transform: "
            f"{invalid['frame_number'].astype(int).tolist()}"
        )

    matched[
        [
            "frame_number",
            "image_filename",
            "frame_id",
            "frame_time",
            "Transform",
        ]
    ].to_csv(MATCHED_TRANSFORMS, index=False)
    return len(matched)


def main() -> None:
    count = prepare_selected_transforms()
    savemat(
        CAMERA_PARAMETERS,
        {
            "cameraParams_5mmK": CAMERA_MATRIX,
            "K": CAMERA_MATRIX,
            "radialDistortion": DIST_COEFFS[:2],
            "distCoeffs": DIST_COEFFS,
        },
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import solve_hand_eye_0612 as solver

    solver.LEFT_K_PATH = CAMERA_PARAMETERS
    solver.IMAGE_FOLDER = IMAGE_DIR
    solver.MATCHED_TRANSFORMS_CSV = MATCHED_TRANSFORMS
    solver.OUTPUT_NPZ = OUTPUT_NPZ
    solver.DIST_COEFFS = DIST_COEFFS

    print(f"Matched selected frames: {count}")
    print(f"Saved camera parameters: {CAMERA_PARAMETERS}")
    solver.main()
    print(f"Saved July 27 hand-eye calibration: {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
