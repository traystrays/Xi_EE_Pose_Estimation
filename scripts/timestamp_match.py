from pathlib import Path
import pandas as pd
import numpy as np

# Syncs API data to the video footage taken frame by frame

api_data_file = r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\cleaned_data.csv"
frame_file = r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\video_stream_1_20260514_201037_timestamps.txt"
output_file = r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\synced_frame_api.csv"

# USER INPUTS #
# frame timezone offset from API time
FRAME_TIMEZONE = 'America/Vancouver'

# Maximum allowed time difference between frame and API sample
MATCH_TOLERANCE = "50ms"

# Which API field to match
SIGNALS_TO_KEEP = [
    "ISI_SUJ_JOINT_VALUES",
    "ISI_JOINT_VALUES",
    "ISI_TIP_POSITION",
    "ISI_TIP_ROTATION",
    "ISI_TIP_TRANSFORM"
]

frames = pd.read_csv(frame_file, header=None, names=["timestamp"])

frames["frame_id"] = np.arange(0, len(frames))

frames["frame_time"] = (
        pd.to_datetime(frames["timestamp"])
        .dt.tz_localize(FRAME_TIMEZONE)
        .dt.tz_convert("UTC")
    )

frames = frames[["frame_id", "frame_time"]].sort_values("frame_time")
print(frames[["frame_id"]].head(), frames["frame_time"].head())

# api = pd.read_csv(api_data_file, usecols=[0])

# # API timestamp has timezone
# # 2026-05-14T20:09:43.790054-07:00
# api["api_time"] = pd.to_datetime(api["system_timestamp"], utc=True)

# print(api["api_time"].head())

api = pd.read_csv(api_data_file)

# Clean column names just in case there are spaces
api.columns = api.columns.str.strip()

print("\nAPI columns:")
print(api.columns.tolist())

TIME_COL = "system_timestamp"
FIELD_COL = "field"
MANIP_COL = "manip"
DATA_COL = "data"

required_cols = [TIME_COL, FIELD_COL, MANIP_COL, DATA_COL]

for col in required_cols:
    if col not in api.columns:
        raise ValueError(
            f"Column '{col}' not found in cleaned_data.csv. "
            f"Available columns are: {api.columns.tolist()}"
        )

# Convert API timestamp to UTC
api["api_time"] = pd.to_datetime(api[TIME_COL], utc=True)

# Clean manipulator ID
api[MANIP_COL] = pd.to_numeric(api[MANIP_COL], errors="coerce").astype("Int64")

# Keep only the signals you need for FIELD_COL, starts a new dataframe
api_filtered = api[api[FIELD_COL].isin(SIGNALS_TO_KEEP)].copy()
print(api_filtered.head())

# Keep only useful columns for now, in this order
api_filtered = api_filtered[
    [
        TIME_COL,
        "api_time",
        FIELD_COL,
        MANIP_COL,
        DATA_COL,
    ]
].copy()

# order by time
api_filtered = api_filtered.sort_values("api_time")

print("\nFiltered API data:")
print(api_filtered.head())

# =========================
# MATCH API DATA TO FRAMES
# =========================

all_matches = []

# Match separately for each signal and manipulator
for (field, manipulator), group in api_filtered.groupby([FIELD_COL, MANIP_COL]):

    group = group.sort_values("api_time").copy()

    # print(group.head())

    #merge_asof left side controls output size. So we are 
    matched = pd.merge_asof(
        frames.sort_values("frame_time"),
        group.sort_values("api_time"),
        left_on="frame_time",
        right_on="api_time",
        direction="nearest",
        tolerance=pd.Timedelta(MATCH_TOLERANCE) # calculate time difference
    )

    # print(matched[["frame_id"]].head())
    # Add field/manipulator back explicitly.
    # This is useful because unmatched rows may have NaN from merge_asof.
    matched[FIELD_COL] = field
    matched[MANIP_COL] = manipulator

    matched["time_error_s"] = (
        matched["frame_time"] - matched["api_time"]
    ).dt.total_seconds().abs()

    matched = matched.rename(columns={
        TIME_COL: "api_timestamp_raw",
        FIELD_COL: "field",
        MANIP_COL: "manipulator",
        DATA_COL: "data",
    })

    matched = matched[  
        [
            "frame_id",
            "frame_time",
            "api_time",
            "field",
            "manipulator",
            "api_timestamp_raw",
            # "time_error_s",
            "data",
        ]
    ]

    all_matches.append(matched)

synced = pd.concat(all_matches, ignore_index=True)

# sort and order everything here 
synced = synced.sort_values(
    ["frame_id", "manipulator", "field"]
).reset_index(drop=True)

# =========================
# SAVE RESULT
# =========================

synced.to_csv(output_file, index=False)

print(f"\nSaved synced data to:\n{output_file}")
print("\nSynced output preview:")
print(synced.head(20))

for i in range(0, 1603):
    if not synced[(synced["frame_id"] == i) & (synced["manipulator"] == 2) & (synced["field"] == "ISI_TIP_TRANSFORM")].empty:
        pass
    else:
        print(f"Frame {i} is missing manipulator 2 ISI_TIP_TRANSFORM")