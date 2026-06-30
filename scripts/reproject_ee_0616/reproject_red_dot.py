import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import cv2
from scipy.io import loadmat

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPTS_DIR))

from xi_hand_eye_calibration_0616 import parse_xi_raw_transform

REPO_ROOT = Path(__file__).resolve().parents[2]
 
HAND_EYE_NPZ = REPO_ROOT / "data" / "xi_data_0616" / "xi_hand_eye_0616_translation_first.npz"
input_data = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\20260616_OR_Xi_EE\20260616_185054.606\KinManip2_20260616_185054.633_2026-06-16_18.csv")
VIDEO_PATH = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\20260616_OR_Xi_EE\video_stream_1_20260616_185106.mp4")
video_ts_data = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\20260616_OR_Xi_EE\video_stream_1_20260616_185106.mp4")

OUTPUT_PATH = REPO_ROOT/"data"/"xi_ee_0616"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

EE_file = OUTPUT_PATH/"EE_transforms.csv"
ECM_file = OUTPUT_PATH/"ECM_transforms.csv"
OUTPUT_VIDEO = OUTPUT_PATH / "EE_red_dot_projection.mp4"


## CAMERA PARAMETERS
LEFT_K_PATH = Path(r"C:\Users\Tracy\Documents\MATLAB\20260616_xi_data\K_left.mat")
K = loadmat(LEFT_K_PATH)["K_left"].astype(np.float64)
DIST_COEFFS = np.array(
    [-0.002994058031627, 0.013201245121718, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
solved = np.load(HAND_EYE_NPZ)
ECM_T_camera = solved["X"]




def load_xi_transform_file(csv_path):
    df = pd.read_csv(csv_path)

    parsed_transforms = []
    invalid_frame_ids = []

    for _, row in df.iterrows():
        try:
            T = parse_xi_raw_transform(
                row["Raw_Transform"],
                layout="translation_first",
                validate=True,
            )
            parsed_transforms.append(T)
        except ValueError as exc:
            parsed_transforms.append(None)
            frame_id = int(row["frame_id"]) if "frame_id" in row else None
            invalid_frame_ids.append(frame_id)
            print(f"{csv_path.name}: skipped frame {frame_id}: {exc}")

    df["T_translation_first"] = parsed_transforms
    df = df.dropna(subset=["T_translation_first"]).copy()

    print(f"{csv_path.name}: loaded {len(df)} valid translation-first transforms")
    if invalid_frame_ids:
        print(f"{csv_path.name}: invalid frames: {invalid_frame_ids}")

    return df


def format_array(array):
    return np.array2string(
        np.asarray(array, dtype=np.float64),
        precision=10,
        suppress_small=False,
        separator=" ",
    )


def get_video_size(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Video size: {width} x {height} pixels")
    return width, height


def project_camera_point_to_pixel(point_camera, K, image_width, image_height):
    X, Y, Z = np.asarray(point_camera, dtype=np.float64).reshape(3)

    if not np.isfinite([X, Y, Z]).all() or abs(Z) < 1e-9:
        return np.nan, np.nan, np.nan, np.nan, False

    x_norm = X / Z
    y_norm = Y / Z

    u = K[0, 0] * x_norm + K[0, 2]
    v = K[1, 1] * y_norm + K[1, 2]

    in_image = 0 <= u < image_width and 0 <= v < image_height
    return x_norm, y_norm, u, v, bool(in_image)


def find_EE_in_camera(EE_transforms, ECM_transforms, ECM_T_camera):
    """
    Compute EE pose in the camera frame.

    Assumes:
        Base_T_EE     comes from EE_file
        Base_T_ECM    comes from ECM_file
        ECM_T_camera  comes from hand-eye calibration

    Then:
        Base_T_camera = Base_T_ECM @ ECM_T_camera
        Camera_T_EE   = inv(Base_T_camera) @ Base_T_EE
    """
    ee_by_frame = {
        int(row["frame_id"]): row["T_translation_first"]
        for _, row in EE_transforms.iterrows()
    }
    ecm_by_frame = {
        int(row["frame_id"]): row["T_translation_first"]
        for _, row in ECM_transforms.iterrows()
    }

    # check that same frame are in both ee and ecm
    common_frame_ids = sorted(set(ee_by_frame) & set(ecm_by_frame))
    rows = []

    for frame_id in common_frame_ids:
        Base_T_EE = ee_by_frame[frame_id]
        Base_T_ECM = ecm_by_frame[frame_id]

        try:
            Cam_T_ECM = np.linalg.inv(ECM_T_camera)
            ECM_T_Base = np.linalg.inv(Base_T_ECM)
            Camera_T_EE = Cam_T_ECM @ ECM_T_Base @ Base_T_EE
        except np.linalg.LinAlgError as exc:
            print(f"frame {frame_id}: cannot invert Base_T_camera: {exc}")
            continue

        rows.append(
            {
                "frame_id": frame_id,
                "Base_T_EE": Base_T_EE,
                "Base_T_ECM": Base_T_ECM,
                "Camera_T_EE": Camera_T_EE,
                "EE_position_in_camera": Camera_T_EE[:3, 3],
            }
        )

    print(f"Computed Camera_T_EE for {len(rows)} matched EE/ECM frames")
    missing_ee = sorted(set(ecm_by_frame) - set(ee_by_frame))
    missing_ecm = sorted(set(ee_by_frame) - set(ecm_by_frame))
    if missing_ee:
        print(f"Frames with ECM transform but no EE transform: {missing_ee}")
    if missing_ecm:
        print(f"Frames with EE transform but no ECM transform: {missing_ecm}")

    return pd.DataFrame(rows)


def add_projected_pixels(df, K, image_width, image_height):
    rows = []
    for _, row in df.iterrows():
        x_norm, y_norm, u, v, in_image = project_camera_point_to_pixel(
            row["EE_position_in_camera"],
            K,
            image_width,
            image_height,
        )
        rows.append(
            {
                "x_over_z": x_norm,
                "y_over_z": y_norm,
                "pixel_u": u,
                "pixel_v": v,
                "in_image": in_image,
            }
        )

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def save_red_dot_video(video_path, projections_df, output_video):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output_video}")

    points_by_frame = {}
    for _, row in projections_df.iterrows():
        if not bool(row["in_image"]):
            continue
        if not np.isfinite([row["pixel_u"], row["pixel_v"]]).all():
            continue

        frame_id = int(row["frame_id"])
        points_by_frame[frame_id] = (
            int(round(float(row["pixel_u"]))),
            int(round(float(row["pixel_v"]))),
        )

    frame_id = 0
    drawn = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        point = points_by_frame.get(frame_id)
        if point is not None:
            cv2.circle(frame, point, 6, (0, 0, 255), -1)
            drawn += 1

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()

    print(f"Saved red dot video: {output_video}")
    print(f"Video frames processed: {frame_id} / {total_frames}")
    print(f"Frames with red dot drawn: {drawn}")



image_width, image_height = get_video_size(VIDEO_PATH)
EE_transforms = load_xi_transform_file(EE_file)
ECM_transforms = load_xi_transform_file(ECM_file)
EE_in_camera = find_EE_in_camera(EE_transforms, ECM_transforms, ECM_T_camera)
EE_in_camera = add_projected_pixels(EE_in_camera, K, image_width, image_height)

EE_in_camera_to_save = EE_in_camera.copy()
EE_in_camera_to_save["Camera_T_EE"] = EE_in_camera_to_save["Camera_T_EE"].apply(
    format_array
)
EE_in_camera_to_save["EE_position_in_camera"] = EE_in_camera_to_save[
    "EE_position_in_camera"
].apply(format_array)
EE_in_camera_to_save.to_csv(OUTPUT_PATH / "EE_in_camera.csv", index=False)

save_red_dot_video(VIDEO_PATH, EE_in_camera, OUTPUT_VIDEO)

print(EE_in_camera.head())
