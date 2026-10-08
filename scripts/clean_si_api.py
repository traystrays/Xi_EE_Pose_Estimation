import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.io import loadmat
import re
import math 

# Paths
Si_Data = Path(r"D:\20261005_calib\data_local_part_1_5_10_2026_15_10_10.csv")
Left_cam_timestamp = Path(r"D:\20261005_calib\video_stream_1_20261005_151006_timestamps.txt")
Output_Data = Path(r"D:\20261005_calib")

left_K_path = Path(r"C:\Users\Tracy\Documents\MATLAB\20261005_si_data\K_left.mat")

Output_Data.mkdir(parents=True, exist_ok=True)


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

    
    
    # 3D MATPLOT LIB CODE BELOW

    x_coords = ecm_pos["pos x"].tolist()
    y_coords = ecm_pos["pos y"].tolist()
    z_coords = ecm_pos["pos z"].tolist()

    print(f"Total matched points: {len(x_coords)}")
    # Create the 3D PlotS
    fig = plt.figure(figsize=(10, 8)) # figure size is 10 by 8 inches
    ax = fig.add_subplot(111, projection='3d')

    # Plot only the points as individual dots
    ax.scatter(x_coords, y_coords, z_coords, c=range(len(x_coords)), cmap='viridis', s=10, label='ECM')

    # Customize plot labels and title
    ax.set_title('3D Tip Positions', fontsize=14)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.legend()

    plt.show()




hand_eye_calibration_csv = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0605\hand_eye_calibration_data.csv")


MATLAB_transform = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0605\MATLAB.csv")
output_csv = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0605\R_t_parsed.csv")




def select_and_parse_frames():
    df = pd.read_csv(MATLAB_transform)
    
    results = []
    for i, row in df.iterrows():
        numbers = re.findall(r"[-+]?\d*\.?\d+", row["Transform"])
        T = np.array(numbers, dtype=float).reshape(4, 4)
        R = T[:3, :3]
        t = T[:3, 3]
        results.append({"index": i, "R": R, "t": t, "T": T})
    
    return results


def select_and_parse_frames():
    df = pd.read_csv(MATLAB_transform)
    
    rows = []
    for i, row in df.iterrows():
        numbers = re.findall(r"[-+]?\d*\.?\d+", row["Transform"])
        T = np.array(numbers, dtype=float).reshape(4, 4)
        R = T[:3, :3]
        t = T[:3, 3]
        
        print(f"Index {i}:")
        print(f"  R = {R}")
        print(f"  t = {t}")
        
        # Flatten R (9 values) and t (3 values) into one row
        rows.append({
            "index": i,
            "R11": R[0,0], "R12": R[0,1], "R13": R[0,2],
            "R21": R[1,0], "R22": R[1,1], "R23": R[1,2],
            "R31": R[2,0], "R32": R[2,1], "R33": R[2,2],
            "tx": t[0], "ty": t[1], "tz": t[2]
        })
    
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")
    return result_df


def checkerboard_inner_corners(columns, rows, square_size):
    # Defines checkerboard pattern based on given pattern and square size
    # Columns and rows are INNER CORNERS
    # square_size in mm
    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    return objp

def detect_checkerboard_keypoints(image, pattern_size):
    # detect refined locations for corners

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if not found:
        return None

    # Refine keypoints to sub-pixel accuracy
    # stops refinind ghwne it reaches 30 iterations or improvement is smaller than 0.001 pixels
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    corners_refined = cv2.cornerSubPix(
        gray,
        corners, # list of corners
        winSize=(11, 11), # total 11 x 11 pixels
        zeroZone=(-1, -1),
        criteria=criteria
    )

    return corners_refined

def estimate_checkerboard_pose(image, object_points, K, dist_coeffs, pattern_size):


    # detect key points
    image_points = detect_checkerboard_keypoints(image, pattern_size)

    if image_points is None:
        return None

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)

    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = R
    T_cam_board[:3, 3] = tvec.flatten()

    return T_cam_board, rvec, tvec, image_points



# reproject back onto the image and see if the handeye calibration is correct
def B_matrix():

    # obtain and save the B-matrices part of the classic AX = XB 
    # hand-eye calibration problem. 
    ## INPUTS 
    pattern_size = (7,6) # columns, rows of INNER corners
    square_size_m = 0.015 # in meters

    object_points = (checkerboard_inner_corners(7, 6, square_size_m))
    left_K = loadmat(left_K_path)['K'].astype(np.float64)

    print(left_K)

    # exit()

    image_folder = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\checkerboard_images\left")

    img_output_folder = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\checkerboard_images\checkerboard_pose_result")
    img_output_folder.mkdir(parents=True, exist_ok=True)

    # ======== Replace with your real distortion coefficients if you have them.

    dist_coeffs = np.array([-2.008552794758558e-06, -3.925424424248872e-12, 0, 0, 0], dtype=np.float64)

    pose_results = []

    for i in range(0, 5984, 100):
        image_path = image_folder / f"frame_{i}_left.jpg"

        if not image_path.exists():
            print(f"Image does not exist: {image_path}")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not load image: {image_path}")
            continue

        result = estimate_checkerboard_pose(
            image=image,
            object_points=object_points,
            K=left_K,
            dist_coeffs=dist_coeffs,
            pattern_size=pattern_size
        )

        if result is None:
            print(f"Checkerboard not detected in frame {i}")
            continue

        T_cam_board, rvec, tvec, corners = result

        pose_results.append({
            "frame_id": i,
            "T_cam_board": T_cam_board,
            "rvec": rvec,
            "tvec": tvec
        })

        # Optional: save image with detected corners drawn
        image_with_corners = image.copy()

        cv2.drawChessboardCorners(
            image_with_corners,
            pattern_size,
            corners,
            True
        )

        corner_output_path = img_output_folder / f"frame_{i}_left_corners.jpg"
        cv2.imwrite(str(corner_output_path), image_with_corners)

        print(f"Detected checkerboard pose in frame {i}")
        print("tvec:")
        print(tvec.flatten())

    if len(pose_results) < 2:
        print("Not enough valid checkerboard poses to compute B matrices.")
        return None


    # Compute B components for AX = XB
    B_matrices = []
    B_records = []

    for k in range(len(pose_results) - 1):
        frame_i = pose_results[k]["frame_id"]
        frame_j = pose_results[k + 1]["frame_id"]

        T_cam_board_i = pose_results[k]["T_cam_board"]
        T_cam_board_j = pose_results[k + 1]["T_cam_board"]

        # This convention matches:
        # A_ij = inv(T_base_tool_i) @ T_base_tool_j
        # B_ij = T_cam_board_i @ inv(T_cam_board_j)
        B_ij = T_cam_board_i @ np.linalg.inv(T_cam_board_j)

        B_matrices.append(B_ij)

        record = {
            "from_frame": frame_i,
            "to_frame": frame_j,
        }

        for r in range(4):
            for c in range(4):
                record[f"B{r}{c}"] = B_ij[r, c]

        B_records.append(record)

    B_matrices = np.array(B_matrices)


    # Save outputs
    B_npy_path = Output_Data / "B_matrices_METERS.npy"
    np.save(B_npy_path, B_matrices)

    B_csv_path = Output_Data / "B_matrices_METERS.csv"
    B_df = pd.DataFrame(B_records)
    B_df.to_csv(B_csv_path, index=False)

    # print("Saved B matrices to:")
    # print(B_npy_path)
    # print(B_csv_path)

    print("B_matrices shape:")
    print(B_matrices.shape)
    print(B_matrices)

    return B_matrices  

def A_matrix():

    def parse_transform_string(transform_str):
        if not isinstance(transform_str, str):
            raise ValueError("Transform value is not a string")

        flat = transform_str.replace("\n", " ")
        nums = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", flat)
        data = np.array([float(x) for x in nums], dtype=np.float64)

        if data.size == 16:
            return data.reshape(4, 4)
        if data.size == 12:
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = data.reshape(3, 4)
            return T

        raise ValueError(f"Transform string must contain 12 or 16 numbers; found {data.size}")

    robot = pd.read_csv(
        Output_Data / "hand_eye_calibration_data.csv",
        usecols=["frame_id", "Transform"],
    )
    camera = pd.read_csv(Output_Data / "B_matrices_METERS.csv", usecols=["from_frame", "to_frame"]).reset_index(drop=True)

    A_matrices = []
    A_records = []

    for idx, row in camera.iterrows():
        i = int(row["from_frame"])
        j = int(row["to_frame"])

        r_from = robot[robot["frame_id"] == i]
        r_to = robot[robot["frame_id"] == j]

        if r_from.empty or r_to.empty:
            print(f"Missing robot transform for frame {i} or {j}; skipping")
            continue

        try:
            robot_i_T = parse_transform_string(r_from.iloc[0]["Transform"])
            robot_j_T = parse_transform_string(r_to.iloc[0]["Transform"])

        except Exception as e:
            print(f"Skipping frame pair {i}->{j} due to parse error: {e}")
            continue

        try:
            A_ij = np.linalg.inv(robot_i_T) @ robot_j_T
            print(A_ij)
        except np.linalg.LinAlgError as e:
            print(f"Cannot invert robot transform for frame {i}: {e}")
            continue

        A_matrices.append(A_ij)

        record = {"from_frame": i, "to_frame": j}
        for r in range(4):
            for c in range(4):
                record[f"A{r}{c}"] = A_ij[r, c]
        A_records.append(record)

        print(f"Computed A matrix for frames {i}->{j}")

    if len(A_matrices) == 0:
        print("No A matrices were computed.")
        return None

    A_matrices = np.array(A_matrices)

    A_npy_path = Output_Data / "A_matrices_METERS.npy"
    np.save(A_npy_path, A_matrices)

    A_csv_path = Output_Data / "A_matrices_METERS.csv"
    A_df = pd.DataFrame(A_records)
    A_df.to_csv(A_csv_path, index=False)

    print("A_matrices shape:", A_matrices.shape)
    return A_matrices

        # robot_i_vals = r_from.iloc[0][transform_cols].to_numpy(dtype=np.float64)
        # robot_j_vals = r_to.iloc[0][transform_cols].to_numpy(dtype=np.float64)

        # robot_i_T = robot_i_vals.reshape(4, 4)
        # robot_j_T = robot_j_vals.reshape(4, 4)

        # try:
        #     A_ij = np.linalg.inv(robot_i_T) @ robot_j_T
        # except np.linalg.LinAlgError as e:
        #     print(f"Cannot invert robot transform for frame {i}: {e}")
        #     continue

        # A_matrices.append(A_ij)

        # record = {
        #     "from_frame": i,
        #     "to_frame": j,
    #     }
    #     for r in range(4):
    #         for c in range(4):
    #             record[f"A{r}{c}"] = A_ij[r, c]
    #     A_records.append(record)

    #     print(f"Computed A matrix for frames {i}->{j}")

    # if len(A_matrices) == 0:
    #     print("No A matrices were computed.")
    #     return None

    # A_matrices = np.array(A_matrices)

    # A_npy_path = Output_Data / "A_matrices.npy"
    # np.save(A_npy_path, A_matrices)

    # A_csv_path = Output_Data / "A_matrices.csv"
    # A_df = pd.DataFrame(A_records)
    # A_df.to_csv(A_csv_path, index=False)

    # print("A_matrices shape:", A_matrices.shape)
    # return A_matrices

if __name__ == "__main__":
    # select_and_parse_frames()
    # B_matrix()
    # A_matrix()
    # sync_si_data_with_video()
    sync_si_data_with_video()





