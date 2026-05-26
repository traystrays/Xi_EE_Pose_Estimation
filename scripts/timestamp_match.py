from pathlib import Path
import pandas as pd
import numpy as np



api_data_file = r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\cleaned_data.csv"
frame_file = r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\video_stream_1_20260514_201037_timestamps.txt"
output_file = r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\synced_frame_api.csv"

# INPUTS

# frame timezone offset from API time
FRAME_TIMEZONE = 'America/Vancouver'


# Maximum allowed time difference between frame and API sample
MATCH_TOLERANCE = "50ms"

# Which API signal to match
SIGNAL_TO_MATCH = "ISI_TIP_TRANSFORM"



frames = pd.read_csv(frame_file, header=None, names=["timestamp"])

frames["frame_id"] = np.arange(1, len(frames) + 1)

frames["frame_time"] = (
        pd.to_datetime(frames["timestamp"])
        .dt.tz_localize(FRAME_TIMEZONE)
        .dt.tz_convert("UTC")
    )

print(frames["frame_time"].head())

api = pd.read_csv(api_data_file, usecols=[0])

# API timestamp has timezone
# 2026-05-14T20:09:43.790054-07:00
api["api_time"] = pd.to_datetime(api["system_timestamp"], utc=True)

print(api["api_time"].head())

# # =========================
# # READ API DATA
# # =========================



# # Clean manipulator ID
# api["manipulator"] = api["manipulator"].astype(float).astype(int)

# # Keep only the signal we care about
# api_signal = api[api["signal"] == SIGNAL_TO_MATCH].copy()


# # =========================
# # PARSE TRANSFORM VALUES
# # =========================

# def parse_transform(value_str):
#     """
#     Converts:
#     "[r11 r12 r13 x r21 r22 r23 y r31 r32 r33 z]"
#     into 12 floats.
#     """
#     value_str = str(value_str).strip()
#     value_str = value_str.replace("[", "").replace("]", "")
#     nums = [float(x) for x in value_str.split()]

#     if len(nums) != 12:
#         raise ValueError(f"Expected 12 values for transform, got {len(nums)}: {value_str}")

#     return nums


# transform_array = api_signal["value"].apply(parse_transform)

# transform_df = pd.DataFrame(
#     transform_array.to_list(),
#     index=api_signal.index,
#     columns=[
#         "r11", "r12", "r13", "x",
#         "r21", "r22", "r23", "y",
#         "r31", "r32", "r33", "z"
#     ]
# )

# api_signal = pd.concat([api_signal, transform_df], axis=1)


# # =========================
# # READ FRAME TIMESTAMPS
# # =========================

# frames_raw = pd.read_csv(frame_file)

# if "timestamp" in frames_raw.columns:
#     frames = frames_raw.copy()
# elif frames_raw.shape[1] == 1:
#     frames = frames_raw.copy()
#     frames.columns = ["timestamp"]
# else:
#     raise ValueError("frame_timestamps.csv should have either a 'timestamp' column or one timestamp column.")

# # Add frame_id if it does not exist
# if "frame_id" not in frames.columns:
#     frames["frame_id"] = np.arange(1, len(frames) + 1)

# # Convert frame timestamps
# if FRAME_TIMEZONE == "UTC":
#     frames["frame_time"] = pd.to_datetime(frames["timestamp"], utc=True)
# else:
#     frames["frame_time"] = (
#         pd.to_datetime(frames["timestamp"])
#         .dt.tz_localize(FRAME_TIMEZONE)
#         .dt.tz_convert("UTC")
#     )

# # Apply optional offset


# frames = frames[["frame_id", "frame_time"]].sort_values("frame_time").copy()


# # =========================
# # MATCH EACH FRAME TO API DATA
# # =========================

# synced = frames.copy()

# for manip_id in MANIPULATORS_TO_MATCH:
#     manip_data = api_signal[api_signal["manipulator"] == manip_id].copy()
#     manip_data = manip_data.sort_values("api_time")

#     if manip_data.empty:
#         print(f"Warning: no {SIGNAL_TO_MATCH} data found for manipulator {manip_id}")
#         continue

#     matched = pd.merge_asof(
#         frames.sort_values("frame_time"),
#         manip_data[[
#             "api_time",
#             "r11", "r12", "r13", "x",
#             "r21", "r22", "r23", "y",
#             "r31", "r32", "r33", "z"
#         ]],
#         left_on="frame_time",
#         right_on="api_time",
#         direction="nearest",
#         tolerance=pd.Timedelta(MATCH_TOLERANCE)
#     )

#     prefix = f"manip{manip_id}_"

#     matched[prefix + "time_error_s"] = (
#         matched["frame_time"] - matched["api_time"]
#     ).dt.total_seconds().abs()

#     matched = matched.rename(columns={
#         "api_time": prefix + "api_time",
#         "r11": prefix + "r11",
#         "r12": prefix + "r12",
#         "r13": prefix + "r13",
#         "x": prefix + "x",
#         "r21": prefix + "r21",
#         "r22": prefix + "r22",
#         "r23": prefix + "r23",
#         "y": prefix + "y",
#         "r31": prefix + "r31",
#         "r32": prefix + "r32",
#         "r33": prefix + "r33",
#         "z": prefix + "z",
#     })

#     cols_to_add = [
#         c for c in matched.columns
#         if c.startswith(prefix)
#     ]

#     synced = synced.merge(
#         matched[["frame_id"] + cols_to_add],
#         on="frame_id",
#         how="left"
#     )


# # =========================
# # SAVE RESULT
# # =========================

# synced.to_csv(output_file, index=False)

# print(f"Saved synced data to: {output_file}")
# print()
# print(synced.head())

# # Print matching quality
# for manip_id in MANIPULATORS_TO_MATCH:
#     col = f"manip{manip_id}_time_error_s"
#     if col in synced.columns:
#         valid = synced[col].dropna()
#         if len(valid) > 0:
#             print(f"\nManipulator {manip_id}:")
#             print(f"  matched frames: {len(valid)} / {len(synced)}")
#             print(f"  mean time error: {valid.mean():.6f} s")
#             print(f"  max time error:  {valid.max():.6f} s")
#         else:
#             print(f"\nManipulator {manip_id}: no valid matches")