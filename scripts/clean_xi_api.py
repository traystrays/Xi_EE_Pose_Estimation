import pandas as pd
import numpy as np
from pathlib import Path
from end_effector_calc import transform_to_matrix

# Each input data file only contains 1 manip information
# Matches video frame time with timestamp of api data

# Change path for input and output files

# manip 1 -> ECM in this case, both absolute paths 
input_data = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\20260616_OR_Xi_checkerboard\20260616_183444.024\KinManip1_20260616_183444.049_2026-06-16_18.csv")
video_data = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\20260616_OR_Xi_checkerboard\video_stream_1_20260616_183326_timestamps.txt")
REPO_ROOT = Path(__file__).resolve().parents[1]
print(REPO_ROOT)

OUTPUT_PATH = REPO_ROOT/"data"/"xi_data_0616"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

matched_path = OUTPUT_PATH/"matched_transforms.csv"


# read csv to dataframe
api_data = pd.read_csv(input_data, usecols=["system_timestamp_ms","field","data"]) 

# generates a boolean chain
filter = api_data["field"] == "ISI_TIP_TRANSFORM" 

api_data = api_data[filter].copy()

api_data["robot_ts"] = (
    pd.to_datetime(
        api_data["system_timestamp_ms"],
        unit="ms",  
        utc=True,
    )
    .dt.tz_convert("America/Vancouver")
)

print(api_data.head())


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

print(api_data.head())
print(video_frame.head())
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

# Frames without a robot sample within 20 ms are not valid matches.
matched = matched.dropna(subset=["robot_ts", "data"]).copy()
matched = matched.rename(columns={"data": "Transform"})
matched = matched[["frame_id", "robot_ts", "Transform"]]

matched.to_csv(matched_path, index=False)

print(f"Saved {len(matched)} matched frames to: {matched_path}")
print(f"Unmatched frames tolerance: {unmatched_count}")


def remove_outliers(csv_path):
    df = pd.read_csv(csv_path)
    valid_rows = []
    removed_frame_ids = []

    for idx, row in df.iterrows():
        try:
            values = np.fromstring(
                str(row["Transform"])
                .replace("[", " ")
                .replace("]", " ")
                .replace(",", " "),
                sep=" ",
            )

            is_valid = (
                values.size == 12
                and np.isfinite(values).all()
                and not np.any(values == 0.0)
            )

            if is_valid:
                valid_rows.append(idx)
            else:
                removed_frame_ids.append(int(row["frame_id"]))

        except (TypeError, ValueError):
            removed_frame_ids.append(int(row["frame_id"]))

    cleaned = df.loc[valid_rows].copy()
    cleaned.to_csv(csv_path, index=False)

    print(f"Removed invalid transforms: {len(removed_frame_ids)}")
    print(f"Remaining valid transforms: {len(cleaned)}")
    print(f"Removed frame IDs: {removed_frame_ids}")

    return cleaned

# remove_outliers(matched_path)
