import numpy as np
from pathlib import Path
from clean_si_api import estimate_checkerboard_pose, checkerboard_inner_corners
from hand_eye_calib import load_transform_list
from scipy.io import loadmat
import pandas as pd
import cv2 
from scipy.spatial.transform import Rotation as R

left_K_path     = Path(r"C:\Users\Tracy\Documents\MATLAB\20260612_si_data\K_left.mat")
robot_T = Path(r"C:\Users\Tracy\Documents\MATLAB\20260612_si_data\cameraTformWRTGripper.mat")
OLD_robot_T = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0605\MATLAB_camera_robot_transform.npz")
image_folder = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\checkerboard_images\si_20260612\calibration")
A_Transform = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0612\matched_transforms.csv")

reprojected_folder = image_folder.parent / "base_prediction_reprojected"
reprojected_folder.mkdir(exist_ok=True)


# calib = np.load(OLD_robot_T)
# X = np.eye(4)
# X[:3, :3] = calib["R"]
# X[:3, 3]  = calib["T"].reshape(3)

# print(X)
# exit()
# # Load hand-eye X
X = loadmat(robot_T)["cameraTformWRTGripper_matrix"].astype(np.float64)


# X = np.eye(4)
# X[:3, :3] = calib["R"]
# X[:3, 3]  = calib["T"].reshape(3)

X_inv = np.linalg.inv(X)

print(X)
print(X_inv)

K = loadmat(left_K_path)["K"].astype(np.float64)
print(K)

DIST_COEFFS = np.array(
    [-0.002994058031627, 0.013201245121718, 0.0, 0.0, 0.0],
    dtype=np.float64,
)

 
# Set checkerboard geometry
COLUMNS      = 7    # inner corners
ROWS         = 6
SQUARE_MM    = 0.005 #m
pattern_size = (COLUMNS, ROWS)

# object list  
object_points = checkerboard_inner_corners(
    columns=COLUMNS,
    rows=ROWS,
    square_size=SQUARE_MM
).astype(np.float64)

print(object_points)

def parse_transform_string(T_str):
    # Remove brackets and convert the remaining numbers to floats
    cleaned = (
        T_str.replace("[", " ")
             .replace("]", " ")
             .replace(",", " ")
    )

    values = np.fromstring(cleaned, sep=" ")

    if values.size != 16:
        raise ValueError(f"Expected 16 values, got {values.size}:\n{T_str}")

    return values.reshape(4, 4)


def load_MATLAB_transform(csv_path):
    df = pd.read_csv(csv_path)

    frame_ids = df["frame_id"].to_numpy()

    T_list = np.stack([
        parse_transform_string(T_str)
        for T_str in df["Transform"]
    ])

    return frame_ids, T_list


frame_ids, A_X = load_MATLAB_transform(A_Transform)

print(A_X)
print(A_X.shape)

A_Transform = []
B_Transform = []


# Set checkerboard
for idx, frame_id in enumerate(frame_ids):

    image_path = image_folder / f"frame_00{frame_id}.png"

    if not image_path.exists():
        print(f"image {frame_id} not found")
        continue

    image_i = cv2.imread(str(image_path))

    # find checkerboard pose
    result_i = estimate_checkerboard_pose(
        image=image_i,
        object_points=object_points,
        K=K,
        dist_coeffs=dist_coeffs,
        pattern_size=pattern_size
    )

    if result_i == None:
        print(f"frame{frame_id} pattern is None")
        continue

    B_i, _, _, _ = result_i

    A_Transform.append(A_X[idx])
    B_Transform.append(B_i)

print(len(A_Transform))
print(len(B_Transform))

if (len(A_Transform) != len(B_Transform)):
    print("not same length")
    exit()

BaseToChecker_List = []

for i in range (0, len(A_Transform)):
    # print(i)
    # print(A_Transform[i])
    # print(B_Transform[i])
    BaseToChecker = A_Transform[i] @ X @ B_Transform[i]
    BaseToChecker_List.append(BaseToChecker)

print(BaseToChecker_List)


def transform_points(T, points_3d):
    points_3d = np.asarray(points_3d, dtype=np.float64)

    ones = np.ones((points_3d.shape[0], 1))
    points_h = np.hstack([points_3d, ones])  # (N, 4)

    transformed_h = (T @ points_h.T).T       # (N, 4)

    return transformed_h[:, :3]

# literally do this on JUST frame 0

Base_T_Camera = A_Transform[0] @ X

Camera_T_Base = np.linalg.inv(Base_T_Camera)

points_base = transform_points(BaseToChecker_List[0], object_points)

points_camera = transform_points(Camera_T_Base, points_base)



def project_to_rotation(R_mat):
    """
    Fix tiny numerical errors so R is a valid rotation matrix.
    """
    U, _, Vt = np.linalg.svd(R_mat)
    R_fixed = U @ Vt

    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1
        R_fixed = U @ Vt

    return R_fixed


def analyze_transform_deviation(T_list):
    """
    T_list shape should be (N, 4, 4)
    """

    T_list = np.asarray(T_list)

    # Separate rotation and translation
    R_list = T_list[:, :3, :3]
    t_list = T_list[:, :3, 3]

    # Clean rotations
    R_list = np.array([project_to_rotation(Ri) for Ri in R_list])

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

    for Ti in T_list:
        # Error transform from mean to this sample
        T_err = np.linalg.inv(T_mean) @ Ti

        # Translation error
        t_err = T_err[:3, 3]
        t_err_norm = np.linalg.norm(t_err)

        # Rotation error
        R_err = T_err[:3, :3]
        R_err = project_to_rotation(R_err)
        angle_deg = R.from_matrix(R_err).magnitude() * 180 / np.pi

        translation_errors.append(t_err_norm)
        rotation_errors_deg.append(angle_deg)

    translation_errors = np.array(translation_errors)
    rotation_errors_deg = np.array(rotation_errors_deg)

    print("Mean transform:")
    print(T_mean)

    print("\nTranslation mean:")
    print(t_mean)

    print("\nTranslation error for each transform:")
    print(translation_errors)

    print("\nRotation error in degrees for each transform:")
    print(rotation_errors_deg)

    print("\nSummary:")
    print(f"Translation RMS error: {np.sqrt(np.mean(translation_errors**2))}")
    print(f"Translation mean error: {np.mean(translation_errors)}")
    print(f"Translation max error: {np.max(translation_errors)}")

    print(f"Rotation RMS error deg: {np.sqrt(np.mean(rotation_errors_deg**2))}")
    print(f"Rotation mean error deg: {np.mean(rotation_errors_deg)}")
    print(f"Rotation max error deg: {np.max(rotation_errors_deg)}")

    return T_mean, translation_errors, rotation_errors_deg

T_mean, t_errors, R_errors_deg = analyze_transform_deviation(BaseToChecker_List)

