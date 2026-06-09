import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.io import loadmat

# Paths
Si_Data = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\si_data_06052026\data_local_part_1_5_6_2026_13_28_44.csv")
Left_cam_timestamp = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0605\video_stream_1_20260605_132941_timestamps.txt")
Output_Data = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0605\hand_eye_calibration_data.csv")

left_video = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\si_data_06052026\video_stream_1_20260605_132941.mp4")
left_output_dir = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\checkerboard_images\left")
right_video = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\si_data_06052026\video_stream_2_20260605_132941.mp4")
right_output_dir = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\checkerboard_images\right")  

left_K_path = Path(r"C:\Users\Tracy\Documents\MATLAB\20260605_si_data\K_cam1_Si.mat")

def sync_si_data_with_video():
    FRAME_TIMEZONE = "America/Vancouver"
    ARM = "mip Index (ECM)"

    # read Si data, read camera timestamp
    df = pd.read_csv(Si_Data)
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
    ecm_start = df.columns.get_loc(ARM)

    # The next three columns are ECM pos x, pos y, pos z
    ecm_pos = df.iloc[:, ecm_start + 1 : ecm_start + 4].copy()
    ecm_pos.insert(0, "timestamp", si_timestamp)

    # Rename them so they are clear
    ecm_pos.columns = ["timestamp", "ECM_pos_x", "ECM_pos_y", "ECM_pos_z"]
    ecm_pos = ecm_pos.sort_values("timestamp")
    # print(ecm_pos.head())


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

    output_df = matched[["frame_id", "frame_time", "ECM_pos_x", "ECM_pos_y", "ECM_pos_z"]]
    output_df.to_csv(Output_Data, index=False)

    # 3D MATPLOT LIB CODE BELOW

    # x_coords = matched["ECM_pos_x"].tolist()
    # y_coords = matched["ECM_pos_y"].tolist()
    # z_coords = matched["ECM_pos_z"].tolist()

    # print(f"Total matched points: {len(x_coords)}")
    # # Create the 3D Plot
    # fig = plt.figure(figsize=(10, 8)) # figure size is 10 by 8 inches
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot only the points as individual dots
    # ax.scatter(x_coords, y_coords, z_coords, c=range(len(x_coords)), cmap='viridis', s=10, label='PSM2 Tip Positions')

    # # Customize plot labels and title
    # ax.set_title('3D Tip Positions', fontsize=14)
    # ax.set_xlabel('X Position')
    # ax.set_ylabel('Y Position')
    # ax.set_zlabel('Z Position')
    # ax.legend()

    # plt.show()

def select_frames():

    for i in range (0, 5984, 100):
        cap = cv2.VideoCapture(left_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {i} from left video")
        cap.release()
        cv2.imwrite(left_output_dir / f"frame_{i}_left.jpg", frame)

        cap = cv2.VideoCapture(right_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {i} from right video")
        cap.release()
        cv2.imwrite(right_output_dir / f"frame_{i}_right.jpg", frame)

def checkerboard_inner_corners(column, row, square_size):
    # Defines checkerboard pattern based on given pattern and square size
    # Columns and rows are INNER CORNERS
    objp = np.zeros((row*column, 3), np.float32)


# reproject back onto the image and see if the handeye calibration is correct
def checkerboard_position():
    left_K = loadmat(left_K_path)['K']
    print(left_K.dtype)



if __name__ == "__main__":
    checkerboard_position()





