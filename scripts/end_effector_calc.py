from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from scipy.io import loadmat

video_path = r"C:\Users\Tracy\Documents\2026 RCL UBC\xi_api_data 05142026\endo_video\video_stream_1_20260514_201037.mp4"
data_path = r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\xi_data_0514\synced_frame_api.csv"
left_K = r"C:\Users\Tracy\Documents\MATLAB\20260514_xi_data\cam_left_K.mat"
right_K = r"C:\Users\Tracy\Documents\MATLAB\20260514_xi_data\cam_right_K.mat"
hand_eye_calib_csv = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\xi_data_0514\hand_eye_calib_data.csv")
full_2D_csv = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\xi_data_0514\full_2D_projection.csv")

calib = np.load("point_hand_eye_calib_result.npz")

random_csv = Path(r"D:\DataVGH\dvapi\KinCsvLog_09-09-2025_17-58-42.368.csv")

#=========== LOAD DATA ===========#

left_K = loadmat(left_K)['K']
print(left_K)

right_K = loadmat(right_K)['K2']
print(right_K)
cap = cv2.VideoCapture(video_path)

# fix later
frame_id = 1100
if not cap.isOpened():
    print("Error opening video file")

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

ret, frame = cap.read()

print(frame.shape)

if not ret:
    print(f"Error reading frame {frame_id}")

calib = np.load("point_hand_eye_calib_result.npz")
R_X, t_X = calib['R'], calib['t']
print(R_X,t_X)
CL_T_ECM = np.eye(4)
CL_T_ECM[:3, :3] = R_X
CL_T_ECM[:3, 3] = t_X.flatten()

# NEW Matrix

CL_T_ECM[:3, :3] = [
    [-0.5171981859, -0.8558511633, 0.0049822532],
    [0.8536669857, -0.5154437886, 0.0746349669],
    [-0.0613083518, 0.0428542545, 0.9971984752],
]

CL_T_ECM[:3, 3] = [-87.3955663331, 8.1365055571, 148.7597741708]
print(CL_T_ECM)

# exit()

#=============== funks ================#

# take 3D point and project on to 2D
def project_2d(K, point_3d):
    # divide to obtain 2D normalized coordinates
    x = point_3d[0] / point_3d[2]
    y = point_3d[1] / point_3d[2]

    # apply intrinsic matrix parameters 
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return u, v

# obtain 4x4 homogeneous transformation matrix from API string data
def transform_to_matrix(transform_str):
    T_temp = np.fromstring(transform_str.strip("[]").replace(",", " "), sep=" ").reshape(3, 4)
    R = T_temp[:3, :3]
    t = T_temp[:3, 3]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

synced_data = pd.read_csv(data_path, usecols=["frame_id", "field", "manipulator", "data"])

# start a list for p API
p_API_list = []
p_2D_list = []
np.set_printoptions(suppress=True, precision=8)


for i in range(0, 1602, 1):
    # filter for data of frame_id, manipulator 2, and field ISI_TIP_TRANSFORM
    frame_id = i
    print(f"\nFrame ID: {frame_id}")
    frame_data = synced_data[(synced_data["frame_id"] == frame_id)]

    PSM1_transform = frame_data.loc[(frame_data["manipulator"] == 0) & (frame_data["field"] == "ISI_TIP_TRANSFORM"), "data"].item()
    PSM2_transform = frame_data.loc[(frame_data["manipulator"] == 2) & (frame_data["field"] == "ISI_TIP_TRANSFORM"), "data"].item()
    ECM_transform = frame_data.loc[(frame_data["manipulator"] == 1) & (frame_data["field"] == "ISI_TIP_TRANSFORM"), "data"].item()

    # print(PSM2_transform)
    # Hardcode the ECM position because it is NOT moving, but there are many error messages
    ECM_transform = "[0.00616765767, -0.114661247, -0.947188497, -0.965364158, 0.188586876, 0.180290669, 0.205659389, 0.97525870, 0.0810634494, -0.160542935, 0.115335025, -0.980266452]"

    PSM1_T = transform_to_matrix(PSM1_transform)
    PSM2_T = transform_to_matrix(PSM2_transform)
    ECM_T = transform_to_matrix(ECM_transform)

    # If any element is NaN or exact integer zero, set the entire transform to zeros and skip
    if np.isnan(ECM_T).any():
        print(f"Frame {frame_id}: ECM_T contains NaN or zero; skipping frame")
        p_API_list.append("[0 0 0]")
        p_2D_list.append("[0 0]")
        continue

    # find inverse of base to ECM to get ECM to base
    try:
        inv_ECM_T = np.linalg.inv(ECM_T)
    except np.linalg.LinAlgError:
        print(f"Frame {frame_id}: ECM_T is singular and cannot be inverted; skipping frame")
        p_API_list.append("[0 0 0]")
        p_2D_list.append("[0 0]")
        continue
    # get PSM2 in cam frame


    print("multiplied by inverse ECM_T same for everything:")
    print(inv_ECM_T)

    PSM2_in_ECM = inv_ECM_T @ PSM2_T

    # print(PSM2_in_ECM)
    PSM2_in_CL = CL_T_ECM @ PSM2_in_ECM
    # print(PSM2_in_CL)

    p_G = np.array([0, 0, 0, 1]) # origin in PSM2 frame

    # point in camera frame in the end
    p_CAM = PSM2_in_CL @ p_G

    # print(p_CAM)

    # select the first 3 elements, last element is 1 anyways
    p_CAM = p_CAM[0:3]
    p_API_list.append(f"[{p_CAM[0]:.6f} {p_CAM[1]:.6f} {p_CAM[2]:.6f}]")
    # print("p_CAM:")
    # print(p_CAM)

    u, v = project_2d(left_K, p_CAM)

    print(u, v)
    p_2D_list.append(f"[{u:.6f} {v:.6f}]")


# df = pd.read_csv(hand_eye_calib_csv)


df = pd.DataFrame()
df["p_2D"] = p_2D_list
df.to_csv(full_2D_csv, index=False)
full_API_csv = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\full_API_projection.csv")
df["p_API"] = p_API_list
df.to_csv(full_API_csv, index=False)


# cv2.imshow("Frame", frame)
# while True:
#     key = cv2.waitKey(0)
#     if key & 0xFF == 27:  # ESC key
#         break
# cv2.destroyAllWindows()