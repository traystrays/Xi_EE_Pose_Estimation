import pandas as pd
import numpy as np
from pathlib import Path
from threeD_trajectory import parse_3d_column
import matplotlib.pyplot as plt

# Each input data file only contains 1 manip information
# Matches video frame time with timestamp of api data

# Change path for input and output files

# manip 1 -> ECM in this case, both absolute paths 
input_data = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\20260616_OR_Xi_checkerboard\20260616_183444.024\KinManip1_20260616_183444.049_2026-06-16_18.csv")
video_data = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\20260616_OR_Xi_checkerboard\video_stream_1_20260616_183326_timestamps.txt")
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT/"data"/"xi_data_0616"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

matched_path = OUTPUT_PATH/"frame_transforms.csv"

t_xyz_path = OUTPUT_PATH/"xyz_position_filtered.csv"


def format_frame_ranges(frame_ids):
    frame_ids = sorted(set(int(frame_id) for frame_id in frame_ids))
    if not frame_ids:
        return "none"

    ranges = []
    start = prev = frame_ids[0]

    for frame_id in frame_ids[1:]:
        if frame_id == prev + 1:
            prev = frame_id
            continue

        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = frame_id

    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(ranges)


def parse_xi_transform(transform_str):
    values = np.fromstring(
        str(transform_str)
        .replace("[", " ")
        .replace("]", " ")
        .replace(",", " "),
        sep=" ",
    )

    if values.size != 12:
        raise ValueError(f"Expected 12 values, got {values.size}")

    T = np.eye(4, dtype=np.float64)

    # Xi ISI_TIP_TRANSFORM format is row-major 3x4:
    # [R00 R01 R02 tx R10 R11 R12 ty R20 R21 R22 tz]
    T_temp = values.reshape(3, 4)
    T[:3, :3] = T_temp[:, :3]
    T[:3, 3] = T_temp[:, 3]

    t = T[:3,3]
    t_string = f"[{t[0]:.10f} {t[1]:.10f} {t[2]:.10f}]"

    return T, t_string


def parse_transform_values(transform_str):
    return np.fromstring(
        str(transform_str)
        .replace("[", " ")
        .replace("]", " ")
        .replace(",", " "),
        sep=" ",
    )


def format_array(array):
    return np.array2string(
        np.asarray(array, dtype=np.float64),
        precision=10,
        suppress_small=False,
        separator=" ",
    )


def remove_zero_positions(df, col_name="t_position", eps=1e-6):
    valid_rows = []
    removed_rows = []

    for idx, row in df.iterrows():
        values = parse_transform_values(row[col_name])
        is_valid = (
            values.size == 3
            and np.isfinite(values).all()
            and not np.any(np.abs(values) < eps)
        )

        if is_valid:
            valid_rows.append(idx)
        else:
            removed_rows.append(idx)

    cleaned = df.loc[valid_rows].copy()
    print(f"Removed zero/invalid t_xyz rows: {len(removed_rows)}")
    print(f"Remaining t_xyz rows: {len(cleaned)}")
    return cleaned


# read csv to dataframe
api_data = pd.read_csv(input_data, usecols=["system_timestamp_ms","field","data"]) 

# Keep only transform rows.
mask = api_data["field"] == "ISI_TIP_TRANSFORM"
api_data = api_data[mask].copy()

api_data["robot_ts"] = (
    pd.to_datetime(
        api_data["system_timestamp_ms"],
        unit="ms",  
        utc=True,
    )
    .dt.tz_convert("America/Vancouver")
)

video_frame = pd.read_csv(video_data, header=None, names=["video_ts"])
video_frame["video_ts"] = (
    pd.to_datetime(video_frame["video_ts"])
    .dt.tz_localize("America/Vancouver")
)

video_frame["frame_id"] = range(len(video_frame))

# Normalize both columns to the same datetime resolution for merge_asof.
timestamp_dtype = "datetime64[ns, America/Vancouver]"
api_data["robot_ts"] = api_data["robot_ts"].astype(timestamp_dtype)
video_frame["video_ts"] = video_frame["video_ts"].astype(timestamp_dtype)

# merge_asof requires both timestamp columns to be sorted.
api_data = api_data.sort_values("robot_ts")
video_frame = video_frame.sort_values("video_ts")

matched = pd.merge_asof(
    video_frame,
    api_data[["robot_ts", "data"]],
    left_on="video_ts",
    right_on="robot_ts",
    direction="nearest",
    tolerance=pd.Timedelta("20ms"),
)

unmatched_list = (
    matched.loc[matched["robot_ts"].isna(), "frame_id"]
    .astype(int)
    .tolist()
)
unmatched_count = int(matched["robot_ts"].isna().sum())
matched["time_error_ms"] = (
    matched["video_ts"] - matched["robot_ts"]
).dt.total_seconds().abs() * 1000

# Frames without a robot sample within 20 ms are not valid matches.
matched = matched.dropna(subset=["robot_ts", "data"]).copy()
matched = matched.rename(columns={"data": "Raw_Transform"})

parsed_transforms = []
invalid_frame_ids = []
t_rows = []

for idx, row in matched.iterrows():
    try:
        T, t_string = parse_xi_transform(row["Raw_Transform"])

        parsed_transforms.append(T)
        t_rows.append(t_string)

    except ValueError:
        parsed_transforms.append(None)
        invalid_frame_ids.append(int(row["frame_id"]))

t_xyz = pd.DataFrame({"t_position": t_rows})

matched["Transform_Matrix"] = parsed_transforms
matched = matched.dropna(subset=["Transform_Matrix"]).copy()

matched["Transform"] = matched["Transform_Matrix"].apply(format_array)
matched["Rotation"] = matched["Transform_Matrix"].apply(lambda T: format_array(T[:3, :3]))
matched["Translation"] = matched["Transform_Matrix"].apply(lambda T: format_array(T[:3, 3]))
matched = matched[
    [
        "frame_id",
        "video_ts",
        "robot_ts",
        "time_error_ms",
        "Raw_Transform",
        "Transform",
        "Rotation",
        "Translation",
    ]
]

matched.to_csv(matched_path, index=False)

print(f"Saved {len(matched)} matched frames to: {matched_path}")
print(f"Unmatched frames tolerance: {unmatched_count}")
print(f"Invalid transform strings skipped: {len(invalid_frame_ids)}")



def remove_outliers(csv_path):

    df = pd.read_csv(csv_path)
    valid_rows = []
    removed_frame_ids = []
    Rotation_col = "Rotation"
    Translation_col = "Translation"
    count = 0
    for idx, row in df.iterrows():
        try:
            Rotation = row[Rotation_col]

            # turn everything into a string
            Rotation = parse_transform_values(row[Rotation_col])
            Translation = parse_transform_values(row[Translation_col])


            is_valid = (
                np.isfinite(Rotation).all()
                and np.isfinite(Translation).all()
                and not np.any(Rotation > 1)
                and not np.any(np.abs(Rotation) < 1e-6) # to remove values < 1e-6
                and not np.any(Translation == 0) # to remove values = 0
            )

            if is_valid:
                valid_rows.append(idx)
                count+=1
            else:
                removed_frame_ids.append(int(row["frame_id"]))

        except (TypeError, ValueError):
            removed_frame_ids.append(int(row["frame_id"]))

    cleaned = df.loc[valid_rows].copy()
    cleaned.to_csv(csv_path, index=False)

    print(f"Removed invalid transforms: {len(removed_frame_ids)}")
    print(f"Remaining valid transforms: {count}")
    print(f"Removed frame IDs: {format_frame_ranges(removed_frame_ids)}")

    return cleaned

remove_outliers(matched_path)


# # t_xyz = remove_zero_positions(t_xyz)

# x, y, z = parse_3d_column(t_xyz, "t_position")
# t_xyz.to_csv(t_xyz_path, index=False)
# fig = plt.figure(figsize=(8, 7))
# ax = fig.add_subplot(111, projection="3d")

# colors = np.linspace(1, 0, len(x))

# ax.scatter(x, y, z, c=colors, cmap="viridis", s=8)
# ax.scatter(x[0], y[0], z[0], c="blue", s=80, marker="o", label="start")
# ax.scatter(x[-1], y[-1], z[-1], c="red", s=80, marker="X", label="finish")

# ax.set_title("Matched Xi Tip Positions")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.legend()

# plt.tight_layout()
# plt.show()
