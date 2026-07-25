import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import cv2
from scipy.io import loadmat

"""
Note: When using the FK calculation, everything originates from Base of robot
"""
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPTS_DIR))

from xi_hand_eye_calibration_0616 import parse_xi_raw_transform
from clean_xi_api import plot_positions

REPO_ROOT = Path(__file__).resolve().parents[2]
 
HAND_EYE_NPZ = REPO_ROOT / "data" / "xi_data_0616" / "xi_hand_eye_0616_translation_first.npz"
input_data = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\20260616_OR_Xi_EE\20260616_185054.606\KinManip2_20260616_185054.633_2026-06-16_18_FK.csv")
VIDEO_PATH = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\20260616_OR_Xi_EE\video_stream_1_20260616_185106.mp4")
video_ts_data = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\20260616_OR_Xi_EE\video_stream_1_20260616_185106_timestamps.txt")

OUTPUT_PATH = REPO_ROOT/"data"/"xi_ee_0616_185054_new_measurements"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

BASE_T_EE2_FILE = OUTPUT_PATH/"EE_transforms_FK.csv"
BASE_T_ECM_FILE = OUTPUT_PATH/"ECM_transforms_FK.csv"
OUTPUT_VIDEO = OUTPUT_PATH / "EE_red_dot_projection_FK_new.mp4"


## CAMERA PARAMETERS
LEFT_K_PATH = Path(r"C:\Users\Tracy\Documents\MATLAB\20260616_xi_data\K_left.mat")
K = loadmat(LEFT_K_PATH)["K_left"].astype(np.float64)

DIST_COEFFS = np.array(
    [-0.002994058031627, 0.013201245121718, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
solved = np.load(HAND_EYE_NPZ)
ECM_T_camera = solved["X"]

print(ECM_T_camera)

exit()


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


def format_frame_ranges(frame_ids):
    frame_ids = sorted(set(int(frame_id) for frame_id in frame_ids))
    if not frame_ids:
        return "none"

    ranges = []
    start = previous = frame_ids[0]
    for frame_id in frame_ids[1:]:
        if frame_id == previous + 1:
            previous = frame_id
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = frame_id

    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def get_video_size(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Video size: {width} x {height} pixels")
    return width, height


def find_active_image_roi(
    video_path,
    sample_count=9,
    black_threshold=8,
    min_nonblack_fraction=0.05,
):
    """Find the non-black image rectangle in full-frame pixel coordinates."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    row_votes = np.zeros(height, dtype=np.int32)
    column_votes = np.zeros(width, dtype=np.int32)
    frames_read = 0

    for frame_idx in np.linspace(
        0, max(frame_count - 1, 0), sample_count, dtype=int
    ):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        success, frame = cap.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row_votes += (gray > black_threshold).mean(axis=1) > min_nonblack_fraction
        column_votes += (gray > black_threshold).mean(axis=0) > min_nonblack_fraction
        frames_read += 1

    cap.release()

    if frames_read == 0:
        raise RuntimeError(f"Could not read frames from video: {video_path}")

    required_votes = frames_read // 2 + 1
    active_rows = np.flatnonzero(row_votes >= required_votes)
    active_columns = np.flatnonzero(column_votes >= required_votes)
    if active_rows.size == 0 or active_columns.size == 0:
        print("Could not detect a non-black image region; using the full frame")
        return 0, 0, width, height

    left = int(active_columns[0])
    top = int(active_rows[0])
    right = int(active_columns[-1] + 1)
    bottom = int(active_rows[-1] + 1)
    print(
        f"Active image ROI: x={left}:{right}, y={top}:{bottom} "
        f"({right - left} x {bottom - top} pixels)"
    )
    return left, top, right, bottom


def project_camera_point_to_pixel(point_camera, K, active_roi):
    X, Y, Z = np.asarray(point_camera, dtype=np.float64).reshape(3)

    if not np.isfinite([X, Y, Z]).all() or abs(Z) < 1e-9:
        return np.nan, np.nan, np.nan, np.nan, False
    
    # #### EDIT ALERT, EDIT ALERT
    # Z = Z / 2.0

    x_norm = X / Z
    y_norm = Y / Z

    u = K[0, 0] * x_norm + K[0, 2]
    v = K[1, 1] * y_norm + K[1, 2]

    left, top, right, bottom = active_roi
    in_image = left <= u < right and top <= v < bottom
    return x_norm, y_norm, u, v, bool(in_image)


def find_EE_in_camera(EE2_transforms, ECM_transforms, ECM_T_camera):
    """
    Compute EE pose in the camera frame.

    EE2 and ECM transforms are expressed in the same base frame, so:
    Camera_T_EE = Cam_T_ECM @ ECM_T_Base @ Base_T_EE2.
    """
    ee2_by_frame = {
        int(row["frame_id"]): row["T_translation_first"]
        for _, row in EE2_transforms.iterrows()
    }

    ecm_by_frame = {
        int(row["frame_id"]): row["T_translation_first"]
        for _, row in ECM_transforms.iterrows()
    }

    frame_ids = sorted(set(ee2_by_frame) & set(ecm_by_frame))
    rows = []
    print(ECM_T_camera)
    
    try:
        Cam_T_ECM = np.linalg.inv(ECM_T_camera)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"cannot invert ECM_T_camera: {exc}") from exc

    for frame_id in frame_ids:
        Base_T_EE2 = ee2_by_frame[frame_id]
        Base_T_ECM = ecm_by_frame[frame_id]

        try:
            ECM_T_Base = np.linalg.inv(Base_T_ECM)
        except np.linalg.LinAlgError as exc:
            print(f"frame {frame_id}: cannot invert Base_T_ECM: {exc}")
            continue

        Camera_T_EE = Cam_T_ECM @ ECM_T_Base @ Base_T_EE2

        rows.append(
            {
                "frame_id": frame_id,
                "Base_T_EE2": Base_T_EE2,
                "Base_T_ECM": Base_T_ECM,
                "Camera_T_EE": Camera_T_EE,
                "EE_position_in_camera": Camera_T_EE[:3, 3],
            }
        )

    ee2_frame_ids = set(ee2_by_frame)
    ecm_frame_ids = set(ecm_by_frame)
    all_frame_ids = ee2_frame_ids | ecm_frame_ids
    missing_ee2_frame_ids = all_frame_ids - ee2_frame_ids
    missing_ecm_frame_ids = all_frame_ids - ecm_frame_ids

    print(f"EE2 frames loaded: {len(ee2_frame_ids)}")
    print(f"ECM frames loaded: {len(ecm_frame_ids)}")
    print(f"Common EE2/ECM frames: {len(frame_ids)}")
    print(f"Camera_T_EE transforms computed: {len(rows)}")
    if missing_ee2_frame_ids:
        print(
            f"Missing EE2 transforms ({len(missing_ee2_frame_ids)} frames): "
            f"{format_frame_ranges(missing_ee2_frame_ids)}"
        )
    if missing_ecm_frame_ids:
        print(
            f"Missing ECM transforms ({len(missing_ecm_frame_ids)} frames): "
            f"{format_frame_ranges(missing_ecm_frame_ids)}"
        )

    return pd.DataFrame(rows)


def add_projected_pixels(df, col_name, K, active_roi):
    rows = []
    for _, row in df.iterrows():
        x_norm, y_norm, u, v, in_image = project_camera_point_to_pixel(
            row[col_name],
            K,
            active_roi,
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
    for row_idx, row in projections_df.iterrows():
        if not np.isfinite([row["pixel_u"], row["pixel_v"]]).all():
            continue

        frame_id = (
            int(row["frame_id"])
            if "frame_id" in projections_df.columns
            else int(row_idx)
        )
        point = (
            int(round(float(row["pixel_u"]))),
            int(round(float(row["pixel_v"]))),
        )
        if 0 <= point[0] < width and 0 <= point[1] < height:
            points_by_frame[frame_id] = point

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
active_roi = find_active_image_roi(VIDEO_PATH)
EE2_transforms = load_xi_transform_file(BASE_T_EE2_FILE)
ECM_transforms = load_xi_transform_file(BASE_T_ECM_FILE)


EE_in_camera = find_EE_in_camera(
    EE2_transforms,
    ECM_transforms,
    ECM_T_camera,
)
EE_in_camera = add_projected_pixels(EE_in_camera, "EE_position_in_camera", K, active_roi)

EE_in_camera_to_save = EE_in_camera.copy()
EE_in_camera_to_save["Base_T_EE2"] = EE_in_camera_to_save["Base_T_EE2"].apply(
    format_array
)
EE_in_camera_to_save["Base_T_ECM"] = EE_in_camera_to_save["Base_T_ECM"].apply(
    format_array
)
EE_in_camera_to_save["Camera_T_EE"] = EE_in_camera_to_save["Camera_T_EE"].apply(
    format_array
)
EE_in_camera_to_save["EE_position_in_camera"] = EE_in_camera_to_save[
    "EE_position_in_camera"
].apply(format_array)

EE_in_camera_to_save.to_csv(OUTPUT_PATH / "EE_in_camera.csv", index=False)

# save_red_dot_video(VIDEO_PATH, EE_in_camera, OUTPUT_VIDEO)

print(EE_in_camera.head())

plot_positions(
    EE_in_camera_to_save,
    "EE_position_in_camera",
    "EE position in camera frame",
)

plt.show()
