from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from scipy.io import loadmat

video_path = r"C:\Users\Tracy\Documents\2026 RCL UBC\xi_api_data 05142026\endo_video\video_stream_1_20260514_201037.mp4"
data_path = r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\synced_frame_api.csv"
left_K = r"C:\Users\Tracy\Documents\MATLAB\cam_left_K.mat"
right_K = r"C:\Users\Tracy\Documents\MATLAB\cam_right_K.mat"
#=========== LOAD DATA ===========#

left_K = loadmat(left_K)['K']
print(left_K)
right_K = loadmat(right_K)['K']
print(right_K)
cap = cv2.VideoCapture(video_path)

# fix later
frame_id = 710
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

for i in range(510, 550):
    # filter for data of frame_id, manipulator 2, and field ISI_TIP_TRANSFORM
    frame_id = i
    frame_data = synced_data[(synced_data["frame_id"] == frame_id)]


    PSM1_transform = frame_data.loc[(frame_data["manipulator"] == 0) & (frame_data["field"] == "ISI_TIP_TRANSFORM"), "data"].item()
    PSM2_transform = frame_data.loc[(frame_data["manipulator"] == 1) & (frame_data["field"] == "ISI_TIP_TRANSFORM"), "data"].item()
    ECM_transform = frame_data.loc[(frame_data["manipulator"] == 2) & (frame_data["field"] == "ISI_TIP_TRANSFORM"), "data"].item()

    PSM1_T = transform_to_matrix(PSM1_transform)
    PSM2_T = transform_to_matrix(PSM2_transform)
    ECM_T = transform_to_matrix(ECM_transform)

    # print(PSM1_T)
    # print(PSM2_T)
    # print(ECM_T)

    # find inverse of base to cam to get cam to base
    inv_ECM_T = np.linalg.inv(ECM_T)
    # get PSM1 in cam frame
    PSM1_in_ECM = inv_ECM_T @ PSM1_T

    p_G = np.array([0, 0, 0, 1]) # origin in PSM1 frame

    # point in ECM frame
    p_ECM = PSM1_in_ECM @ p_G
    # print("p_ECM:")
    # print(p_ECM)

    u, v = project_2d(left_K, p_ECM)
    print(u, v)


cv2.imshow("Frame", frame)
while True:
    key = cv2.waitKey(0)
    if key & 0xFF == 27:  # ESC key
        break
cv2.destroyAllWindows()