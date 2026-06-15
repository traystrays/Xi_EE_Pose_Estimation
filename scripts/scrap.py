"""
Build matched_transforms.csv from saved_frames.csv.

For each selected frame in data/si_data_0612/saved_frames.csv, this script finds
the matching robot transform in hand_eye_calibration_data_METERS.csv and writes:

    data/si_data_0612/matched_transforms.csv

It also writes scripts/gripper_transforms.mat for MATLAB use.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "si_data_0612"

SAVED_FRAMES_CSV = DATA_DIR / "saved_frames.csv"
TRANSFORM_CSV = DATA_DIR / "hand_eye_calibration_data_METERS.csv"
OUTPUT_CSV = DATA_DIR / "matched_transforms.csv"
GRIPPER_TRANSFORMS_MAT = Path(__file__).resolve().with_name("gripper_transforms.mat")


def parse_transform_string(transform_str):
    values = np.fromstring(
        transform_str.replace("[", " ").replace("]", " ").replace(",", " "),
        sep=" ",
    )

    if values.size != 16:
        raise ValueError(f"Expected 16 transform values, got {values.size}")

    return values.reshape(4, 4)


def main():
    saved_frames = pd.read_csv(SAVED_FRAMES_CSV)
    transforms = pd.read_csv(TRANSFORM_CSV)

    saved_frames["frame_number"] = saved_frames["frame_number"].astype(int)
    transforms["frame_id"] = transforms["frame_id"].astype(int)

    saved_frames = (
        saved_frames.drop_duplicates(subset=["frame_number"], keep="first")
        .sort_values("frame_number")
        .reset_index(drop=True)
    )

    merged = saved_frames.merge(
        transforms,
        left_on="frame_number",
        right_on="frame_id",
        how="left",
    )

    missing = merged[merged["frame_id"].isna()]
    if not missing.empty:
        print("Warning: no robot transform found for frame(s):")
        print(missing["frame_number"].astype(int).tolist())

    merged = merged.dropna(subset=["frame_id", "Transform"]).copy()
    merged["frame_id"] = merged["frame_id"].astype(int)

    output_cols = [
        "frame_number",
        "image_filename",
        "frame_id",
        "frame_time",
        "Transform",
    ]
    merged[output_cols].to_csv(OUTPUT_CSV, index=False)

    transform_array = np.array(
        [parse_transform_string(transform_str) for transform_str in merged["Transform"]]
    )
    sio.savemat(GRIPPER_TRANSFORMS_MAT, {"gripperTforms": transform_array})

    print(f"Saved matched transforms: {OUTPUT_CSV}")
    print(f"Matched frame count: {len(merged)}")
    print(f"Saved MATLAB gripper transforms: {GRIPPER_TRANSFORMS_MAT}")
    print(f"gripperTforms shape: {transform_array.shape}")


if __name__ == "__main__":
    main()
