import numpy as np
from pathlib import Path
from clean_si_api import estimate_checkerboard_pose, checkerboard_inner_corners
from scipy.io import loadmat
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R


# =========================
# Paths
# =========================

left_K_path = Path(
    r"C:\Users\Tracy\Documents\MATLAB\20260605_si_data\K_cam1_Si.mat"
)

robot_T_path = Path(
    r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0605\MATLAB_camera_robot_transform.npz"
)

image_folder = Path(
    r"C:\Users\Tracy\Projects\Xi_EE_Pose\checkerboard_images\left"
)

A_Transform_path = Path(
    r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0605\MATLAB.csv"
)

reprojected_folder = image_folder.parent / "base_prediction_reprojected"
reprojected_folder.mkdir(exist_ok=True)

error_csv_path = reprojected_folder / "base_prediction_reprojection_errors.csv"


# =========================
# Important transform switch
# =========================
# If X means EE_T_Camera, keep this False.
# If X means Camera_T_EE, set this True.
USE_X_INV = False


# =========================
# Load hand-eye transform X
# =========================

calib = np.load(robot_T_path)

X = np.eye(4)
X[:3, :3] = calib["R"]
X[:3, 3] = calib["T"].reshape(3)

X_inv = np.linalg.inv(X)

print("\nLoaded hand-eye transform X:")
print(X)

if USE_X_INV:
    EE_T_Camera = X_inv
    print("\nUsing X_inv as EE_T_Camera")
else:
    EE_T_Camera = X
    print("\nUsing X as EE_T_Camera")


# =========================
# Load camera intrinsics
# =========================

K = loadmat(left_K_path)["K"].astype(np.float64)

dist_coeffs = np.array(
    [-2.008552794758558e-06, -3.925424424248872e-12, 0, 0, 0],
    dtype=np.float64
)

print("\nCamera K:")
print(K)

print("\nDistortion coefficients:")
print(dist_coeffs)


# =========================
# Checkerboard geometry
# =========================

COLUMNS = 7          # inner corners
ROWS = 6             # inner corners
SQUARE_SIZE = 0.015  # meters

pattern_size = (COLUMNS, ROWS)

object_points = checkerboard_inner_corners(
    columns=COLUMNS,
    rows=ROWS,
    square_size=SQUARE_SIZE
).astype(np.float64)


# =========================
# Helper functions
# =========================

def parse_transform_string(T_str):
    """
    Converts a string like:

    [[ 0.98242 -0.14424 -0.11853  0.0929 ]
     [-0.16384 -0.97047 -0.17705  1.0137 ]
     [-0.08949  0.19336 -0.97704  0.42872]
     [ 0.       0.       0.       1.     ]]

    into a 4x4 numpy array.
    """

    cleaned = (
        T_str.replace("[", " ")
             .replace("]", " ")
             .replace(",", " ")
    )

    values = np.fromstring(cleaned, sep=" ")

    if values.size != 16:
        raise ValueError(
            f"Expected 16 values, got {values.size}:\n{T_str}"
        )

    return values.reshape(4, 4)


def load_MATLAB_transform(csv_path):
    """
    Loads robot transforms from MATLAB.csv.

    Expected columns:
    - frame_id
    - Transform
    """

    df = pd.read_csv(csv_path)

    frame_ids = df["frame_id"].to_numpy()

    T_list = np.stack([
        parse_transform_string(T_str)
        for T_str in df["Transform"]
    ])

    return frame_ids, T_list


def project_to_rotation(R_mat):
    """
    Fix small numerical errors so the matrix is a valid rotation matrix.
    """

    U, _, Vt = np.linalg.svd(R_mat)
    R_fixed = U @ Vt

    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1
        R_fixed = U @ Vt

    return R_fixed


def analyze_transform_deviation(T_list):
    """
    Computes mean transform and deviation from mean.

    Input:
        T_list: shape (N, 4, 4)

    Returns:
        T_mean
        translation_errors
        rotation_errors_deg
    """

    T_list = np.asarray(T_list)

    if T_list.ndim != 3 or T_list.shape[1:] != (4, 4):
        raise ValueError(
            f"T_list should have shape (N, 4, 4), got {T_list.shape}"
        )

    R_list = T_list[:, :3, :3]
    t_list = T_list[:, :3, 3]

    R_list = np.array([
        project_to_rotation(Ri)
        for Ri in R_list
    ])

    # Mean translation
    t_mean = np.mean(t_list, axis=0)

    # Mean rotation
    rotations = R.from_matrix(R_list)
    R_mean = rotations.mean().as_matrix()

    # Build mean transform
    T_mean = np.eye(4)
    T_mean[:3, :3] = R_mean
    T_mean[:3, 3] = t_mean

    translation_errors = []
    rotation_errors_deg = []

    T_mean_inv = np.linalg.inv(T_mean)

    for Ti in T_list:
        T_err = T_mean_inv @ Ti

        t_err = T_err[:3, 3]
        t_err_norm = np.linalg.norm(t_err)

        R_err = T_err[:3, :3]
        R_err = project_to_rotation(R_err)

        angle_deg = R.from_matrix(R_err).magnitude() * 180.0 / np.pi

        translation_errors.append(t_err_norm)
        rotation_errors_deg.append(angle_deg)

    translation_errors = np.array(translation_errors)
    rotation_errors_deg = np.array(rotation_errors_deg)

    print("\nMean Base_T_Checkerboard:")
    print(T_mean)

    print("\nMean checkerboard translation in base:")
    print(t_mean)

    print("\nTranslation error for each accepted transform:")
    print(translation_errors)

    print("\nRotation error in degrees for each accepted transform:")
    print(rotation_errors_deg)

    print("\nBase_T_Checkerboard deviation summary:")
    print(f"Translation RMS error: {np.sqrt(np.mean(translation_errors ** 2))}")
    print(f"Translation mean error: {np.mean(translation_errors)}")
    print(f"Translation max error: {np.max(translation_errors)}")
    print(f"Rotation RMS error deg: {np.sqrt(np.mean(rotation_errors_deg ** 2))}")
    print(f"Rotation mean error deg: {np.mean(rotation_errors_deg)}")
    print(f"Rotation max error deg: {np.max(rotation_errors_deg)}")

    return T_mean, translation_errors, rotation_errors_deg


def draw_points(image, points, color, radius=4):
    """
    Draws 2D points on an image.

    Handles points shaped like:
    - (N, 2)
    - (N, 1, 2)
    - (N, 1, 1, 2)
    """

    points = np.asarray(points, dtype=np.float64)
    points = points.reshape(-1, 2)

    h, w = image.shape[:2]

    for p in points:
        u = int(round(float(p[0])))
        v = int(round(float(p[1])))

        # Optional: only draw points inside the image
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(image, (u, v), radius, color, -1)


# =========================
# Load robot/API transforms
# =========================

frame_ids, A_X = load_MATLAB_transform(A_Transform_path)

print("\nLoaded robot/API transforms:")
print("A_X shape:", A_X.shape)


# =========================
# First pass:
# collect accepted frames only
# =========================

accepted_data = []

for idx, frame_id in enumerate(frame_ids):

    frame_id = int(frame_id)

    image_path = image_folder / f"frame_{frame_id}_left.jpg"

    if not image_path.exists():
        print(f"image {frame_id} not found")
        continue

    image_i = cv2.imread(str(image_path))

    if image_i is None:
        print(f"image {frame_id} could not be read")
        continue

    result_i = estimate_checkerboard_pose(
        image=image_i,
        object_points=object_points,
        K=K,
        dist_coeffs=dist_coeffs,
        pattern_size=pattern_size
    )

    if result_i is None:
        print(f"frame {frame_id} pattern is None")
        continue

    # Expected return:
    # B_i = Camera_T_Checkerboard
    # rvec_i, tvec_i = solvePnP outputs
    # detected_corners = actual image checkerboard corners
    B_i, rvec_i, tvec_i, detected_corners = result_i

    A_i = A_X[idx]  # Base_T_EE

    accepted_data.append({
        "idx": idx,
        "frame_id": frame_id,
        "image_path": image_path,
        "A_i": A_i,
        "B_i": B_i,
        "rvec_i": rvec_i,
        "tvec_i": tvec_i,
        "detected_corners": detected_corners
    })

print("\nAccepted frames:", len(accepted_data))

if len(accepted_data) == 0:
    raise RuntimeError("No accepted checkerboard frames found.")


# =========================
# Compute Base_T_Checkerboard
# from accepted frames only
# =========================

BaseToChecker_List = []

for item in accepted_data:

    A_i = item["A_i"]  # Base_T_EE
    B_i = item["B_i"]  # Camera_T_Checkerboard

    Base_T_Camera = A_i @ EE_T_Camera

    Base_T_Checkerboard = Base_T_Camera @ B_i

    BaseToChecker_List.append(Base_T_Checkerboard)

BaseToChecker_Array = np.asarray(BaseToChecker_List)

print("\nBaseToChecker_Array shape:")
print(BaseToChecker_Array.shape)


# =========================
# Analyze variation of checkerboard
# in base coordinates
# =========================

T_mean, t_errors, R_errors_deg = analyze_transform_deviation(
    BaseToChecker_Array
)


# =========================
# Second pass:
# Reproject base-predicted checkerboard
# into each accepted image
# =========================

all_pixel_errors = []
error_rows = []

for accepted_idx, item in enumerate(accepted_data):

    frame_id = item["frame_id"]
    image_path = item["image_path"]
    A_i = item["A_i"]
    detected_corners = item["detected_corners"]

    image_i = cv2.imread(str(image_path))

    if image_i is None:
        print(f"frame {frame_id}: image could not be read during reprojection")
        continue

    image_draw = image_i.copy()

    # Current camera pose in base frame
    Base_T_Camera = A_i @ EE_T_Camera

    # Convert fixed checkerboard pose from base frame into current camera frame
    Camera_T_Base = np.linalg.inv(Base_T_Camera)

    Camera_T_Checkerboard_pred = Camera_T_Base @ T_mean

    # Extract predicted rotation and translation
    R_pred = Camera_T_Checkerboard_pred[:3, :3]
    R_pred = project_to_rotation(R_pred)

    t_pred = Camera_T_Checkerboard_pred[:3, 3].reshape(3, 1)

    rvec_pred, _ = cv2.Rodrigues(R_pred)

    # Project where robot/base thinks checkerboard should be
    projected_pred, _ = cv2.projectPoints(
        object_points,
        rvec_pred,
        t_pred,
        K,
        dist_coeffs
    )

    projected_pred = projected_pred.reshape(-1, 2)

    if detected_corners is None:
        print(f"frame {frame_id}: detected_corners is None")
        continue

    detected_corners = detected_corners.reshape(-1, 2)

    if detected_corners.shape != projected_pred.shape:
        print(
            f"frame {frame_id}: corner shape mismatch, "
            f"detected={detected_corners.shape}, "
            f"predicted={projected_pred.shape}"
        )
        continue

    # Pixel error: actual observed checkerboard vs base-predicted checkerboard
    pixel_errors = np.linalg.norm(
        projected_pred - detected_corners,
        axis=1
    )

    mean_px = np.mean(pixel_errors)
    rms_px = np.sqrt(np.mean(pixel_errors ** 2))
    max_px = np.max(pixel_errors)

    all_pixel_errors.extend(pixel_errors)

    error_rows.append({
        "frame_id": frame_id,
        "mean_pixel_error": mean_px,
        "rms_pixel_error": rms_px,
        "max_pixel_error": max_px,
        "base_translation_error": t_errors[accepted_idx],
        "base_rotation_error_deg": R_errors_deg[accepted_idx]
    })

    print(
        f"frame {frame_id}: "
        f"mean pixel error = {mean_px:.2f} px, "
        f"RMS = {rms_px:.2f} px, "
        f"max = {max_px:.2f} px"
    )

    # Draw actual detected corners in green
    draw_points(
        image_draw,
        detected_corners,
        color=(0, 255, 0),
        radius=4
    )

    # Draw base-predicted checkerboard corners in red
    draw_points(
        image_draw,
        projected_pred,
        color=(0, 0, 255),
        radius=4
    )

    # Draw blue lines from actual corners to predicted corners
    for p_actual, p_pred in zip(detected_corners, projected_pred):
        u1, v1 = int(round(p_actual[0])), int(round(p_actual[1]))
        u2, v2 = int(round(p_pred[0])), int(round(p_pred[1]))

        cv2.line(
            image_draw,
            (u1, v1),
            (u2, v2),
            (255, 0, 0),
            1
        )

    # Add text label
    label = (
        f"frame {frame_id} | "
        f"mean {mean_px:.2f}px | "
        f"rms {rms_px:.2f}px | "
        f"max {max_px:.2f}px"
    )

    cv2.putText(
        image_draw,
        label,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    output_path = (
        reprojected_folder /
        f"frame_{frame_id}_base_prediction.jpg"
    )

    cv2.imwrite(str(output_path), image_draw)


# =========================
# Save and print overall error
# =========================

all_pixel_errors = np.asarray(all_pixel_errors)

if all_pixel_errors.size > 0:

    print("\nOverall base-prediction reprojection error:")
    print(f"Mean pixel error: {np.mean(all_pixel_errors):.2f} px")
    print(f"RMS pixel error: {np.sqrt(np.mean(all_pixel_errors ** 2)):.2f} px")
    print(f"Max pixel error: {np.max(all_pixel_errors):.2f} px")

    error_df = pd.DataFrame(error_rows)
    error_df.to_csv(error_csv_path, index=False)

    print("\nSaved per-frame reprojection errors to:")
    print(error_csv_path)

else:
    print("\nNo pixel errors were computed.")

print("\nSaved reprojected images to:")
print(reprojected_folder)