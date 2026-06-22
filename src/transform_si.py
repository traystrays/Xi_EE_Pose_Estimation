import numpy as np
import pandas as pd
from pathlib import Path 


# Paths

Si_Data = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\si_data_06122026\data_local_part_1_12_6_2026_17_8_40.csv")
Left_cam_timestamp = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\si_data_06122026\video_stream_1_20260612_170858_timestamps.txt")
Output_Data = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0612")

left_video = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\si_data_06052026\video_stream_1_20260605_132941.mp4")
left_output_dir = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\checkerboard_images\si_20260612\left")
right_video = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\si_data_06052026\video_stream_2_20260605_132941.mp4")
right_output_dir = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\checkerboard_images\si_20260612\right")  

left_K_path = Path(r"C:\Users\Tracy\Documents\MATLAB\20260605_si_data\K_cam1_Si.mat")

Output_Data.mkdir(parents=True, exist_ok=True)
left_output_dir.mkdir(parents=True, exist_ok=True)
right_output_dir.mkdir(parents=True, exist_ok=True)

def Transform_Si(Si_csv: str | Path, arm: str):

    '''
    Parce through csv data from dv robot SI for a particular arm (eg: "mip Index (ECM)")
    Return a csv df with timestamp, transform matrix, rotational matrix, and translation
    '''


    FRAME_TIMEZONE = "America/Vancouver"
    ARM = "mip Index (ECM)"

    # read Si data, read camera timestamp
    df = pd.read_csv(Si_csv)
    frames = pd.read_csv(Left_cam_timestamp, header=None, names=["timestamp"])

    # select time stamp for Si data
    # timestamp = df.columns[0]

    frames["frame_id"] = np.arange(0, len(frames))

    frames["frame_time"] = (
            pd.to_datetime(frames["timestamp"])
            .dt.tz_localize(FRAME_TIMEZONE)
            .dt.tz_convert("UTC")
        )

    frames = frames[["frame_id", "frame_time"]].sort_values("frame_time")
    print(frames[["frame_id"]].head(), frames["frame_time"].head())

    si_timestamp = pd.to_datetime(
        df["Time_stamp"],
        unit="us",
        utc=True
    )
    # print(si_timestamp.head())

    # Find the column index of "mip Index (ECM)"
    ecm_start = df.columns.get_loc(arm)

    # The next three columns are ECM pos x, pos y, pos z
    ecm_pos = df.iloc[:, ecm_start + 1 : ecm_start + 13].copy()

    ecm_pos.insert(0, "timestamp", si_timestamp)

    # Rename them so they are clear
    ecm_pos.columns = [
            "timestamp", "pos x","pos y","pos z",
            "o0",
            "o1",
            "o2",
            "o3",
            "o4",
            "o5",
            "o6",
            "o7",
            "o8"
        ]
    
    ecm_pos = ecm_pos.sort_values("timestamp")
    print(ecm_pos.head())

    tolerance = pd.Timedelta("25ms")

    matched = pd.merge_asof(
        frames,
        ecm_pos,
        left_on="frame_time",
        right_on="timestamp",
        direction="nearest",
        tolerance=tolerance
    )

    print(matched.head())

    # Build transform matrices from matched rows when rotation (o0..o8) and position (pos x,y,z) exist

    transform_matrices = []

    for idx, row in matched.iterrows():
        try:
            t = np.array([row["pos x"], row["pos y"], row["pos z"]], dtype=float)

            R = np.array([
                [row["o0"], row["o1"], row["o2"]],
                [row["o3"], row["o4"], row["o5"]],
                [row["o6"], row["o7"], row["o8"]],
            ], dtype=float)

            if np.isnan(t).any() or np.isnan(R).any():
                transform_matrices.append(np.full((4, 4), np.nan, dtype=float))
                continue

            T = np.eye(4, dtype=float)
            T[:3, :3] = R
            T[:3, 3] = t

            transform_matrices.append(T)


        except KeyError:
            transform_matrices.append(np.full((4, 4), np.nan, dtype=float))
        except Exception as e:
            print(f"Error parsing row {idx}: {e}")
            transform_matrices.append(np.full((4, 4), np.nan, dtype=float))

    matched = matched.copy()
    matched["Transform"] = transform_matrices

    hand_eye_output = Output_Data / "hand_eye_calibration_data_METERS.csv"
    output_df = matched[["frame_id", "frame_time", "Transform"]]
    output_df.to_csv(hand_eye_output, index=False)
