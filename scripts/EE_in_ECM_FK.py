"""Express FK end-effector poses in the FK ECM coordinate frame.

For each matching video frame, compute::

    ECM_T_EE_FK = inverse(Base_T_ECM_FK) @ Base_T_EE_FK

No hand-eye calibration or camera-frame transform is used.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "xi_ee_0616_185054"
NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+\.?\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def parse_matrix(value: object) -> np.ndarray:
    values = np.asarray(NUMBER_PATTERN.findall(str(value)), dtype=np.float64)
    if values.size != 16:
        raise ValueError(f"Expected 16 transform values, got {values.size}")
    matrix = values.reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError("Transform contains non-finite values")
    return matrix


def format_array(array: np.ndarray) -> str:
    return np.array2string(
        np.asarray(array, dtype=np.float64),
        precision=10,
        suppress_small=False,
        separator=" ",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ecm",
        type=Path,
        default=DATA_DIR / "ECMFK_transforms.csv",
        help="CSV containing Base_T_ECM FK transforms",
    )
    parser.add_argument(
        "--ee",
        type=Path,
        default=DATA_DIR / "EEFK_transforms.csv",
        help="CSV containing Base_T_EE FK transforms",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "EE_in_ECM_FK.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def load_transforms(path: Path, prefix: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"frame_id", "Transform"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if data["frame_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate frame_id values")

    columns = ["frame_id", "Transform"]
    for optional in ("video_ts", "robot_ts", "time_error_ms"):
        if optional in data.columns:
            columns.append(optional)
    return data[columns].rename(
        columns={name: f"{prefix}_{name}" for name in columns if name != "frame_id"}
    )


def main() -> None:
    args = parse_args()
    ecm = load_transforms(args.ecm, "ECM")
    ee = load_transforms(args.ee, "EE")
    matched = ecm.merge(ee, on="frame_id", how="inner", validate="one_to_one")
    if matched.empty:
        raise SystemExit("No matching frame_id values between the ECM and EE files")

    output_rows = []
    invalid_frames = []
    for row in matched.itertuples(index=False):
        try:
            base_t_ecm = parse_matrix(row.ECM_Transform)
            base_t_ee = parse_matrix(row.EE_Transform)
            ecm_t_ee = np.linalg.inv(base_t_ecm) @ base_t_ee
        except (ValueError, np.linalg.LinAlgError):
            invalid_frames.append(int(row.frame_id))
            continue

        output_row = {
            "frame_id": int(row.frame_id),
            "Base_T_ECM_FK": format_array(base_t_ecm),
            "Base_T_EE_FK": format_array(base_t_ee),
            "ECM_T_EE_FK": format_array(ecm_t_ee),
            "Transform": format_array(ecm_t_ee),
            "Rotation": format_array(ecm_t_ee[:3, :3]),
            "Translation": format_array(ecm_t_ee[:3, 3]),
        }
        for name in (
            "ECM_video_ts",
            "ECM_robot_ts",
            "ECM_time_error_ms",
            "EE_video_ts",
            "EE_robot_ts",
            "EE_time_error_ms",
        ):
            if hasattr(row, name):
                output_row[name] = getattr(row, name)
        output_rows.append(output_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(args.output, index=False)
    print(f"Matched frames: {len(matched)}")
    print(f"Valid ECM-relative EE poses: {len(output_rows)}")
    print(f"Invalid frames skipped: {len(invalid_frames)}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
