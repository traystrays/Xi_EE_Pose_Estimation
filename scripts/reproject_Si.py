import numpy as np
from clean_si_api import estimate_checkerboard_pose, checkerboard_inner_corners
from hand_eye_calib import load_transform_list
from pathlib import Path
from scipy.io import loadmat
import cv2
import pandas as pd

# AX = XB. B we obtain as T_camera_world using Solve.PNP 

# Paths

image_folder    = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\checkerboard_images\left")
output_folder   = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\checkerboard_images\reprojection_verification")
Hand_eye_npz    = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0605\MATLAB_camera_robot_transform.npz")
A_csv           = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0605\A_matrices_METERS.csv")
B_csv = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0605\B_matrices_METERS.csv")
left_K_path     = Path(r"C:\Users\Tracy\Documents\MATLAB\20260605_si_data\K_cam1_Si.mat")

output_folder.mkdir(parents=True, exist_ok=True)


# Load hand-eye X
calib = np.load(Hand_eye_npz)
X = np.eye(4)
X[:3, :3] = calib["R_X"]
X[:3, 3]  = calib["t_X"].reshape(3)
X_inv = np.linalg.inv(X)

print(X)


# print(X@X_inv)

K = loadmat(left_K_path)["K"].astype(np.float64)
print(K)

# Unitless

dist_coeffs = np.array(
    [-2.008552794758558e-06, -3.925424424248872e-12, 0, 0, 0],
    dtype=np.float64
)
 
print("\nCamera K:")
print(K)
 
# Set checkerboard geometry
COLUMNS      = 7    # inner corners
ROWS         = 6
SQUARE_MM    = 0.015 #m
pattern_size = (COLUMNS, ROWS)

# object list  
object_points = checkerboard_inner_corners(
    columns=COLUMNS,
    rows=ROWS,
    square_size=SQUARE_MM
).astype(np.float64)
 
# Load A matrices
A_keys, A_list = load_transform_list(A_csv)
B_keys, B_list = load_transform_list(B_csv)

# exit()

# Helper: reproject object_points given a T_cam_board

def reproject_board(T_cam_board, object_points, K, dist_coeffs):
    """
    Given a 4x4 camera-to-board transform, project all object_points
    back onto the image plane.
    Returns (N, 2) array of pixel coordinates.
    """
    rvec, _ = cv2.Rodrigues(T_cam_board[:3, :3])
    tvec    = T_cam_board[:3, 3].reshape(3, 1)
 
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        dist_coeffs
    )
    return projected.reshape(-1, 2)
 
#  def assemble_A(from_frame, to_frame):



# ------------------------------------------------------------
# Helper: draw verification result on image
# ------------------------------------------------------------
def draw_verification(image, detected_corners, projected_corners, mean_error, pair_label):
    out = image.copy()
    h, w = out.shape[:2]
 
    def to_pt(pt):
        return (int(round(float(pt[0]))), int(round(float(pt[1]))))
 
    def in_bounds(pt):
        x, y = float(pt[0]), float(pt[1])
        return 0 <= x < w and 0 <= y < h
 
    # Green = corners actually detected in this image (ground truth)
    for pt in detected_corners:
        if in_bounds(pt):
            cv2.circle(out, to_pt(pt), 5, (0, 255, 0), -1)
 
    # Red = corners reprojected from the predicted pose via AX=XB
    # Only draw if in bounds — wildly off-screen means the predicted pose is bad
    n_visible = 0
    for pt in projected_corners:
        if in_bounds(pt):
            cv2.circle(out, to_pt(pt), 5, (0, 0, 255), -1)
            n_visible += 1
 
    # Yellow lines — only draw if both endpoints are in bounds
    for det, proj in zip(detected_corners, projected_corners):
        if in_bounds(det) and in_bounds(proj):
            cv2.line(out, to_pt(det), to_pt(proj), (0, 255, 255), 1)
 
    label = f"Pair {pair_label}  mean error: {mean_error:.2f} px  ({n_visible}/{len(projected_corners)} corners visible)"
    cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(out, "Green = detected   Red = predicted via AX=XB",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
 
    if n_visible == 0:
        cv2.putText(out, "WARNING: all predicted corners off-screen — bad A matrix or degenerate pose",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    return out
 
 
# ------------------------------------------------------------
# Main reprojection loop over all A-matrix pairs
# ------------------------------------------------------------
all_errors = []
 
AX = []
XB = []

for idx, (key, A, B) in enumerate(zip(A_keys, A_list, B_list)):
    from_frame, to_frame = key


    print(from_frame, to_frame, "print B:", B)
 
    image_i_path = image_folder / f"frame_{from_frame}_left.jpg"
    image_j_path = image_folder / f"frame_{to_frame}_left.jpg"
 
    if not image_i_path.exists() or not image_j_path.exists():
        print(f"Skipping pair {key}: image(s) not found")
        continue
 
    image_i = cv2.imread(str(image_i_path))
    image_j = cv2.imread(str(image_j_path))
 
    if image_i is None or image_j is None:
        print(f"Skipping pair {key}: could not load image(s)")
        continue
 
    # Find B_i pose based on 
    result_i = estimate_checkerboard_pose(
        image=image_i,
        object_points=object_points,
        K=K,
        dist_coeffs=dist_coeffs,
        pattern_size=pattern_size
    )
 
    if result_i is None:
        print(f"Skipping pair {key}: checkerboard not detected in frame {from_frame}")
        continue
 
    B_i, _, _, _ = result_i

    print("this is Bi", B_i)
 
    # Find B_j (ground truth for comparison) ---
    result_j = estimate_checkerboard_pose(
        image=image_j,
        object_points=object_points,
        K=K,
        dist_coeffs=dist_coeffs,
        pattern_size=pattern_size
    )
 
    if result_j is None:
        print(f"Skipping pair {key}: checkerboard not detected in frame {to_frame}")
        continue
 
    B_j, _, _, detected_corners_j = result_j

    # pixel locations 
    detected_corners_j = detected_corners_j.reshape(-1, 2)

    # print("A:\n", A)
    # print("B:\n", B)
    print("AX:\n", A @ X)
    print("XB:\n", X @ B)

    AX.append(A @ X)
    XB.append(X @ B)
    # print("diff AX - XB:\n", A @ X - X @ B)
    
    # print("Bi:\n", B_i)
    # print("Bj:\n", B_j)

 
    # --- Predict T_cam_board_j using AX = XB ---
    # Rearranged: T_cam_board_j_predicted = X_inv @ Aj_inv @Ai @ X @ T_cam_board_i
   
    inv_A = np.linalg.inv(A)
    T_cam_board_j_predicted = X_inv @ inv_A @ X @ B_i

    # print("B_j_projected", T_cam_board_j_predicted)

    # print("detected_corners:", detected_corners_j)
 
    # # Sanity check: Z of predicted board should be positive (in front of camera)
    # predicted_z = T_cam_board_j_predicted[2, 3]
    # if predicted_z <= 0:
    #     print(f"  WARNING pair {key}: predicted board Z={predicted_z:.3f} (behind camera) — A matrix likely degenerate")
 
    # --- Reproject object_points using the predicted pose ---
    projected_corners_j = reproject_board(
        T_cam_board=T_cam_board_j_predicted, 
        object_points=object_points,
        K=K,
        dist_coeffs=dist_coeffs
    )
 
    # print(projected_corners_j)
    # --- Compute per-corner pixel errors ---
    errors      = np.linalg.norm(projected_corners_j - detected_corners_j, axis=1)
    mean_error  = errors.mean()
    max_error   = errors.max()
 
    all_errors.append(mean_error)
 
    print(f"Pair {from_frame}->{to_frame}:  mean={mean_error:.2f}px  max={max_error:.2f}px")
 
    # --- Save visualisation onto frame j ---
    output_image = draw_verification(
        image=image_j,
        detected_corners=detected_corners_j,               
        projected_corners=projected_corners_j,
        mean_error=mean_error,
        pair_label=f"{from_frame}->{to_frame}"
    )
 
    out_path = output_folder / f"verify_{from_frame}_to_{to_frame}.jpg"
    cv2.imwrite(str(out_path), output_image)


Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\AXXB").mkdir(parents=True, exist_ok=True)
B_csv_path = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\AXXB\.csv")

B_df = pd.DataFrame({"AX": AX, "XB": XB})
B_df.to_csv(B_csv_path, index=False)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
if all_errors:
    print("\n========== SUMMARY ==========")
    print(f"Pairs evaluated :  {len(all_errors)}")
    print(f"Mean error (all):  {np.mean(all_errors):.2f} px")
    print(f"Median error    :  {np.median(all_errors):.2f} px")
    print(f"Max mean error  :  {np.max(all_errors):.2f} px")
    print(f"\nOutput images saved to: {output_folder}")
else:
    print("No pairs could be evaluated — check image paths and checkerboard detection.")

# ------------------------------------------------------------
# Display saved output images one by one
# Press any key to advance, ESC to quit early
# ------------------------------------------------------------
output_images = sorted(output_folder.glob("verify_*.jpg"))
 
print(f"\nShowing {len(output_images)} images — press any key to advance, ESC to quit.")
 
for img_path in output_images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    cv2.imshow("Reprojection Verification", img)
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break
 
cv2.destroyAllWindows()