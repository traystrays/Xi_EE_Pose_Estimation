from pathlib import Path
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from threeD_trajectory import parse_3d_column


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data" / "xi_ee_0616_185054_new_measurements"

EE_CSV = OUTPUT_PATH / "EE_transforms_FK.csv"
ECM_CSV = OUTPUT_PATH / "ECM_transforms_FK.csv"
BASE_TO_EE_CSV = OUTPUT_PATH / "Base_T_EE_from_ECM_EE.csv"
BASE_TO_EEFK_CSV = OUTPUT_PATH / "Base_T_EEFK_from_inv_ECMFK_EEFK.csv"

EEFK_CSV = OUTPUT_PATH / "EEFK_corrected_pitch_transforms.csv"
ECMFK_CSV = OUTPUT_PATH / "ECMFK_transforms_FK.csv"

def parse_transform_matrix(transform_str):
    values = re.findall(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
        str(transform_str),
    )
    values = np.array(values, dtype=np.float64)

    if values.size != 16:
        raise ValueError(f"Expected 16 transform values, got {values.size}")

    return values.reshape(4, 4)


def format_array(array):
    return np.array2string(
        np.asarray(array, dtype=np.float64),
        precision=10,
        suppress_small=False,
        separator=" ",
    )


def format_position(position):
    position = np.asarray(position, dtype=np.float64).reshape(3)
    return f"[{position[0]:.10f} {position[1]:.10f} {position[2]:.10f}]"


def load_transform_csv(csv_path):
    df = pd.read_csv(csv_path, usecols=["frame_id", "Transform"])
    df["T"] = df["Transform"].apply(parse_transform_matrix)
    return df[["frame_id", "T"]]


def plot_positions(ax, position_df, col_name, title):
    x, y, z = parse_3d_column(position_df, col_name)

    if len(x) == 0:
        print(f"No points to plot for: {title}")
        return

    colors = np.linspace(1, 0, len(x))
    ax.scatter(x, y, z, c=colors, cmap="viridis", s=8)
    ax.scatter(x[0], y[0], z[0], c="blue", s=80, marker="o", label="start")
    ax.scatter(x[-1], y[-1], z[-1], c="red", s=80, marker="X", label="finish")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()


def print_first_transform_start_and_end(label, df, transform_col, position_col):
    if df.empty:
        print(f"{label}: no rows available")
        return

    first_row = df.iloc[0]
    last_row = df.iloc[-1]
    print(f"\n{label} first frame_id: {int(first_row['frame_id'])}")
    print(f"{label} first transform:")
    print(first_row[transform_col])
    print(f"{label} start point:")
    print(first_row[position_col])
    print(f"{label} end frame_id: {int(last_row['frame_id'])}")
    print(f"{label} end point:")
    print(last_row[position_col])


EE = load_transform_csv(EE_CSV)
ECM = load_transform_csv(ECM_CSV)
ECMFK = load_transform_csv(ECMFK_CSV)
EEFK = load_transform_csv(EEFK_CSV)


matched = EE.merge(
    ECM,
    on="frame_id",
    how="inner",
    suffixes=("_EE", "_ECM"),
    validate="one_to_one",
)

rows = []
for _, row in matched.iterrows():
    frame_id = int(row["frame_id"])
    EE_T = row["T_EE"]
    ECM_T = row["T_ECM"]
    # camera is the base
    Base_T_EE = EE_T

    rows.append(
        {
            "frame_id": frame_id,
            "ECM_T": format_array(ECM_T),
            "EE_T": format_array(EE_T),
            "Base_T_EE": format_array(Base_T_EE),
            "Base_EE_position": format_position(Base_T_EE[:3, 3]),
        }
    )

matched_FK = EEFK.merge(
    ECMFK,
    on="frame_id",
    how="inner",
    suffixes=("_EEFK", "_ECMFK"),
    validate="one_to_one",
)

rows_FK = []
for _, row in matched_FK.iterrows():
    frame_id = int(row["frame_id"])
    EEFK_T = row["T_EEFK"]
    ECMFK_T = row["T_ECMFK"]
    # camera is the base
    inv_ECMFK_T = np.linalg.inv(ECMFK_T)
    Base_T_EEFK = inv_ECMFK_T @ EEFK_T

    rows_FK.append(
        {
            "frame_id": frame_id,
            "ECMFK_T": format_array(ECMFK_T),
            "EEFK_T": format_array(EEFK_T),
            "Base_T_EEFK": format_array(Base_T_EEFK),
            "Base_EEFK_position": format_position(Base_T_EEFK[:3, 3]),
        }
    )


base_to_ee = pd.DataFrame(rows)
base_to_ee.to_csv(BASE_TO_EE_CSV, index=False)
base_to_eefk = pd.DataFrame(rows_FK)
base_to_eefk.to_csv(BASE_TO_EEFK_CSV, index=False)

print(f"EE frames loaded: {len(EE)}")
print(f"ECM frames loaded: {len(ECM)}")
print(f"EEFK frames loaded: {len(EEFK)}")
print(f"ECMFK frames loaded: {len(ECMFK)}")
print(f"Matched frames multiplied: {len(base_to_ee)}")
print(f"Matched FK frames multiplied: {len(base_to_eefk)}")
print(f"Saved Base_T_EE results to: {BASE_TO_EE_CSV}")
print(f"Saved Base_T_EEFK results to: {BASE_TO_EEFK_CSV}")

print_first_transform_start_and_end(
    "ECM @ EE",
    base_to_ee,
    "Base_T_EE",
    "Base_EE_position",
)
print_first_transform_start_and_end(
    "inv(ECMFK) @ EEFK",
    base_to_eefk,
    "Base_T_EEFK",
    "Base_EEFK_position",
)

fig = plt.figure(figsize=(14, 7))
base_ax = fig.add_subplot(121, projection="3d")
ecm_ax = fig.add_subplot(122, projection="3d")

plot_positions(
    base_ax,
    base_to_ee,
    "Base_EE_position",
    "Base to EE from ECM @ EE",
)
plot_positions(
    ecm_ax,
    base_to_eefk,
    "Base_EEFK_position",
    "inv(ECMFK) @ EEFK",
)

plt.tight_layout()

plt.show()
