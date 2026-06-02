from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from scipy.io import loadmat

video_path = r"C:\Users\Tracy\Documents\2026 RCL UBC\xi_api_data 05142026\endo_video\video_stream_1_20260514_201037.mp4"
data_path = r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\synced_frame_api.csv"
left_K = r"C:\Users\Tracy\Documents\MATLAB\20260514_xi_data\cam_left_K.mat"
right_K = r"C:\Users\Tracy\Documents\MATLAB\20260514_xi_data\cam_right_K.mat"
hand_eye_calib_csv = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\hand_eye_calib_data.csv")


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

if not ret:
    print(f"Error reading frame {frame_id}")

#=============== funks ================#

# take 3D point and project on to 2D
def project_2d(K, point_3d):
    # divide to obtain 2D normalized coordinates
    x = point_3d[0] / point_3d[2]
    y = point_3d[1] / point_3d[2]
    # apply intrinsic matrix parameters 
    # print( "u", x, K[0,0], x * K[0, 0], K[0, 2])
    # print( "v", y, K[1,1], y * K[1, 1], K[1, 2])
    print("z", point_3d[2])

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
for i in range(0, 1600, 50):
    # filter for data of frame_id, manipulator 2, and field ISI_TIP_TRANSFORM
    frame_id = i
    print(f"\nFrame ID: {frame_id}")
    frame_data = synced_data[(synced_data["frame_id"] == frame_id)]


    PSM1_transform = frame_data.loc[(frame_data["manipulator"] == 0) & (frame_data["field"] == "ISI_TIP_TRANSFORM"), "data"].item()
    PSM2_transform = frame_data.loc[(frame_data["manipulator"] == 1) & (frame_data["field"] == "ISI_TIP_TRANSFORM"), "data"].item()
    ECM_transform = frame_data.loc[(frame_data["manipulator"] == 2) & (frame_data["field"] == "ISI_TIP_TRANSFORM"), "data"].item()

    # Hardcode the ECM position because it is NOT moving
    ECM_transform = "[-0.00946542528 0.000198544934 0.0764139444 0.160584629 -0.127096385 0.978803575 0.902461052 -0.382696331 -0.197752804 0.399717778 0.915090442 0.0532469451]"

    PSM1_T = transform_to_matrix(PSM1_transform)
    PSM2_T = transform_to_matrix(PSM2_transform)
    ECM_T = transform_to_matrix(ECM_transform)
    print("hello",ECM_T)

    # find inverse of base to cam to get cam to base
    inv_ECM_T = np.linalg.inv(ECM_T)
    # get PSM1 in cam frame
    PSM1_in_ECM = inv_ECM_T @ PSM1_T

    p_G = np.array([0, 0, 0, 1]) # origin in PSM1 frame

    # point in ECM frame
    p_ECM = PSM1_in_ECM @ p_G
    p_ECM = p_ECM[0:3]
    p_API_list.append(f"[{p_ECM[0]:.6f} {p_ECM[1]:.6f} {p_ECM[2]:.6f}]")
    print("p_ECM:")
    print(p_ECM)

    u, v = project_2d(left_K, p_ECM)

    print(u, v)
    p_2D_list.append(f"[{u:.6f} {v:.6f}]")

# df = pd.read_csv(hand_eye_calib_csv)
# df["p_API"] = p_API_list
# df["p_2D"] = p_2D_list
# df.to_csv(hand_eye_calib_csv, index=False)


# cv2.imshow("Frame", frame)
# while True:
#     key = cv2.waitKey(0)
#     if key & 0xFF == 27:  # ESC key
#         break
# cv2.destroyAllWindows()