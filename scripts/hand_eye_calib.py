from pathlib import Path
import cv2
import pandas as pd
from scipy.io import loadmat
import numpy as np
import datetime

LEFT_VIDEO = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\xi_api_data 05142026\endo_video\video_stream_1_20260514_201037.mp4")
RIGHT_VIDEO = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\xi_api_data 05142026\endo_video\video_stream_2_20260514_201037.mp4")
OUTPUT_CSV = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\hand_eye_calib_data.csv")

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
if OUTPUT_CSV.exists():
    if OUTPUT_CSV.stat().st_size <=5: # file is empty or only has header
        print(f"Output CSV {OUTPUT_CSV} exists but is empty. Starting fresh.")
    else:
        EMPTY = False
        print(f"Output CSV {OUTPUT_CSV} already exists. Loading existing data.")

if EMPTY:
    print(f"parsing through data and saving to {OUTPUT_CSV}...")
    for i in range (0, 1600, 50):
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
    pd.DataFrame(clicks).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved labels to {OUTPUT_CSV}")

cv2.destroyAllWindows()


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

        # protect against degenerate solutions
        w = p_4D[3, 0]
        if abs(w) < 1e-12:
            print(f"Degenerate triangulation for row {index}; appending empty value")
            p_CL_list.append("")
            continue

        p_3D = (p_4D[:3, 0] / w)
        print(f"3D point: {p_3D}")

        # optional reprojection sanity check
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



