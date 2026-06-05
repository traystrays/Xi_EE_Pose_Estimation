from pathlib import Path
import cv2
import pandas as pd
from scipy.io import loadmat
import numpy as np
import sys
import re


LEFT_VIDEO = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\xi_api_data 05142026\endo_video\video_stream_1_20260514_201037.mp4")
RIGHT_VIDEO = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\xi_api_data 05142026\endo_video\video_stream_2_20260514_201037.mp4")
OUTPUT_CSV = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\hand_eye_calib_data.csv")
OUTPUT_CSV_2 = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\hand_eye_calib_data_2.csv")

R_right = Path(r"C:\Users\Tracy\Documents\MATLAB\20260514_xi_data\R_cam2.mat")
t_right = Path(r"C:\Users\Tracy\Documents\MATLAB\20260514_xi_data\t_cam2.mat")
left_K = r"C:\Users\Tracy\Documents\MATLAB\20260514_xi_data\cam_left_K.mat"
right_K = r"C:\Users\Tracy\Documents\MATLAB\20260514_xi_data\cam_right_K.mat"

clicks = []
current_points = {"left": None, "right": None}


def get_frame(video_path, frame_id):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Your frame_id starts at 1, OpenCV starts at 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_id} from {video_path}")

    return frame


def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    width_left = param["width_left"]
    scale = param.get("scale", 1)
    
    # Scale up coordinates if display was resized
    x = int(x * scale)
    y = int(y * scale)

    if x < width_left:
        current_points["left"] = (x, y)
        print(f"Left click:  uL={x}, vL={y}")
    else:
        current_points["right"] = (x - width_left, y)
        print(f"Right click: uR={x - width_left}, vR={y}")

EMPTY = True

# checking to see if output csv already exists and is populated with data
if OUTPUT_CSV_2.exists():
    if OUTPUT_CSV_2.stat().st_size <=5: # file is empty or only has header
        print(f"Output CSV {OUTPUT_CSV_2} exists but is empty. Starting fresh.")
    else:
        EMPTY = False
        print(f"Output CSV {OUTPUT_CSV_2} already exists. Loading existing data.")

if EMPTY:
    print(f"parsing through data and saving to {OUTPUT_CSV_2}...")
    for i in range (25, 1600, 50):
        frame_id = i
        left_frame = get_frame(LEFT_VIDEO, frame_id)
        right_frame = get_frame(RIGHT_VIDEO, frame_id)

        if left_frame.shape[0] != right_frame.shape[0]:
            raise ValueError("Left and right frames have different heights. Resize before displaying.")

        current_points["left"] = None
        current_points["right"] = None

        combined = cv2.hconcat([left_frame, right_frame])
        display = combined.copy()

        window_name = f"Frame {frame_id}: click left tip and right tip"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(
            window_name,
            mouse_callback,
            {"width_left": left_frame.shape[1], "scale": 2}  # 2x scale factor to convert resized coords back
        )

        while True:
            display = combined.copy()

            if current_points["left"] is not None:
                uL, vL = current_points["left"]
                cv2.circle(display, (uL, vL), 6, (0, 0, 255), -1)

            if current_points["right"] is not None:
                uR, vR = current_points["right"]
                cv2.circle(display, (uR + left_frame.shape[1], vR), 6, (0, 255, 0), -1)

            cv2.putText(
                display,
                "Click left tip and right tip. Press n=save next, r=reset, q=quit",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Resize to half size for display
            display_half = cv2.resize(display, (display.shape[1] // 2, display.shape[0] // 2))
            cv2.imshow(window_name, display_half)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("r"):
                current_points["left"] = None
                current_points["right"] = None
                print("Reset clicks.")

            elif key == ord("n"):
                if current_points["left"] is None or current_points["right"] is None:
                    print("Need both left and right clicks before saving.")
                    continue

                uL, vL = current_points["left"]
                uR, vR = current_points["right"]

                clicks.append({
                    "frame_id": frame_id,
                    "uL": uL,
                    "vL": vL,
                    "uR": uR,
                    "vR": vR,
                })

                pd.DataFrame(clicks).to_csv(OUTPUT_CSV, index=False
                )
                print(f"Saved frame {frame_id}")
                break

            elif key == ord("q"):
                cv2.destroyAllWindows()
                pd.DataFrame(clicks).to_csv(OUTPUT_CSV, index=False)
                print(f"Saved labels to {OUTPUT_CSV}")
                raise SystemExit

        cv2.destroyWindow(window_name)
    pd.DataFrame(clicks).to_csv(OUTPUT_CSV_2, index=False)
    print(f"Saved labels to {OUTPUT_CSV}")

cv2.destroyAllWindows()
sys.exit()


R = loadmat(R_right)['R_cam2']
t = loadmat(t_right)['t_cam2'].reshape(3, 1)
K_left = loadmat(left_K)['K']
K_right = loadmat(right_K)['K2']

# build projection matrix, this is in mm
P_L = (K_left @ np.eye(3, 4)).astype(np.float64)
P_R = (K_right @ np.hstack((R, t))).astype(np.float64)

print("Project left:", P_L, "Project right:", P_R)

# obtain 2D points in pixels from the output csv and triangulate
if OUTPUT_CSV.exists():
    df = pd.read_csv(OUTPUT_CSV, dtype={"uL": np.float64, "vL": np.float64, "uR": np.float64, "vR": np.float64})
    p_CL_list = []

    for index, row in df.iterrows():
        uL, vL, uR, vR = row["uL"], row["vL"], row["uR"], row["vR"]
        print(f"Processing frame {row.get('frame_id', index)} with points: Left({uL}, {vL}), Right({uR}, {vR})")

        # triangulate expects 2xN arrays; here N=1
        p_4D = cv2.triangulatePoints(P_L, P_R, np.array([[uL], [vL]], dtype=np.float64), np.array([[uR], [vR]], dtype=np.float64))

        # protect against degenerate solutions that is 0
        w = p_4D[3, 0]
        if abs(w) < 1e-12:
            p_CL_list.append("")
            continue

        p_3D = (p_4D[:3, 0] / w)
        print(f"3D point: {p_3D}")

        # reprojection sanity check so we are not getting BS
        X_h = np.array([p_3D[0], p_3D[1], p_3D[2], 1.0], dtype=np.float64)
        xL_proj = P_L @ X_h
        xL_proj = (xL_proj[:2] / xL_proj[2])
        xR_proj = P_R @ X_h
        xR_proj = (xR_proj[:2] / xR_proj[2])
        print("Original left:", [uL, vL], "Reprojected left:", xL_proj)
        print("Original right:", [uR, vR], "Reprojected right:", xR_proj)

        p_CL_list.append(f"[{p_3D[0]:.6f} {p_3D[1]:.6f} {p_3D[2]:.6f}]")

    # append column and overwrite CSV
    df["p_CL"] = p_CL_list
    df.to_csv(OUTPUT_CSV, index=False)
    # print(f"Appended p_CL column to {OUTPUT_CSV} for {len(df)} rows")
else:
    print(f"Output CSV {OUTPUT_CSV} not found; skipping triangulation")

# select sources and targets 
sources = pd.read_csv(OUTPUT_CSV, usecols=["p_API"])['p_API'].tolist()
targets = pd.read_csv(OUTPUT_CSV, usecols=["p_CL"])['p_CL'].tolist()

def parse_vector_field(s):
    """Parse a string like '[x y z]' into a (1,3) numpy array.
    Returns None if parsing fails or values contain NaN.
    """
    if pd.isna(s):
        return None
    if isinstance(s, (float, int)):
        return None
    # extract numeric tokens
    toks = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s))
    if len(toks) < 3:
        return None
    try:
        vals = np.array([float(toks[0]), float(toks[1]), float(toks[2])], dtype=np.float64)
    except Exception:
        return None
    if np.isnan(vals).any():
        return None
    return vals.reshape(1, 3)

sources_list = []
targets_list = []
skipped_frames = []
bad_reasons = {}

for i in range(len(sources)):
    s_raw = sources[i]
    t_raw = targets[i]
    s_parsed = parse_vector_field(s_raw)
    t_parsed = parse_vector_field(t_raw)
    if s_parsed is None or t_parsed is None:
        skipped_frames.append(i)
        reason = []
        if s_parsed is None:
            reason.append('source')
        if t_parsed is None:
            reason.append('target')
        bad_reasons[i] = ','.join(reason)
        continue
    sources_list.append(s_parsed)
    targets_list.append(t_parsed)

print('Skipped frames:', skipped_frames)
if bad_reasons:
    print('Skip reasons (index: reason):')
    for k, v in list(bad_reasons.items())[:10]:
        print(k, v)

def least_square(X, Y):
    # using umeyama method to find least square
    # input (m,n) numpy array whee m is dimension of points (3) and n is number of points (N)
    # Estimates c, R, and t such that Y = c * R * X + t, c = 1.0 here because 

    # finds the average position of each point in space
    mu_x = X.mean(axis = 1, keepdims=True)
    mu_y = Y.mean(axis = 1, keepdims=True)

    # print(mu_x.shape)
    # print(mu_y.shape)  
    print (X.shape[1])
    var_xy =np.square(X - mu_x).sum(axis=0).mean() # variance of points in source, one number

    # cross 
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1] # cross covariance between source and target, a matrix (mxm) 
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = 1.0
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t

source_N3 = np.vstack(sources_list)
target_N3 = np.vstack(targets_list)
print(source_N3) # (N,3)
print(target_N3) # (N,3)
print(source_N3.T) # (3,N)
print(target_N3.T) #(3,N)

c, R, t = least_square(source_N3.T, target_N3.T)

print(R,t)

np.savez("point_hand_eye_calib_result.npz", R=R, t=t)


