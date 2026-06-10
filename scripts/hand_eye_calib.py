from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import time

# Using the classic AX = XB method 
# to find the hand-eye calibration between between ECM and camera

# Using data from Si first, then go to OR to collect data on Xi
A_csv = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0605\A_matrices_METERS.csv")
B_csv = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0605\B_matrices_METERS.csv")

# Assemble to lists
def load_transform_list(csv_path):
    df = pd.read_csv(csv_path)

    # Keep frame pair information
    keys = list(zip(df["from_frame"].astype(int), df["to_frame"].astype(int)))

    # Use every column except from_frame and to_frame
    matrix_df = df.drop(columns=["from_frame", "to_frame"])

    # Convert to numpy matrices
    T_list = matrix_df.to_numpy(dtype=float).reshape(-1, 4, 4)

    return keys, T_list

# Match the A and B 
def match_A_B(A_keys, A_list, B_keys, B_list):
    """
    Match A_ij and B_ij by the same from_frame and to_frame.
    """
    B_dict = {key: B for key, B in zip(B_keys, B_list)}

    matched_keys = []
    A_matched = []
    B_matched = []

    for key, A in zip(A_keys, A_list):
        if key in B_dict:
            matched_keys.append(key)
            A_matched.append(A)
            B_matched.append(B_dict[key])

    if len(matched_keys) == 0:
        raise ValueError("No matching frame pairs found between A and B.")

    return matched_keys, np.array(A_matched), np.array(B_matched)

###### HELPER FUNKS for Rotation ######
def project_to_SO3(R):

    # Fix small numerical errors so that R is a valid rotation matrix.
    
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt

    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt

    return R_clean


def rotation_log_vector(R):
    """
    Convert rotation matrix to rotation vector.

    This is equivalent to taking log(R).
    cv2.Rodrigues returns the axis-angle rotation vector.
    """
    R = project_to_SO3(R)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


# get Rotation of A, B, hard-coded 0.5 degrees
def solve_Rx(A_list, B_list, min_rotation_deg=0.5):
    """
    Solve rotation part of AX = XB:

        R_A R_X = R_X R_B

    Using:
        alpha_i = R_X beta_i

    where:
        alpha_i = log(R_Ai)
        beta_i  = log(R_Bi)
    """
    min_rotation_rad = np.deg2rad(min_rotation_deg)

    M = np.zeros((3, 3))
    used_indices = []

    for i, (A, B) in enumerate(zip(A_list, B_list)):
        R_A = A[:3, :3]
        R_B = B[:3, :3]

        alpha = rotation_log_vector(R_A)
        beta = rotation_log_vector(R_B)

        # Skip almost-identity rotations
        if np.linalg.norm(alpha) < min_rotation_rad and np.linalg.norm(beta) < min_rotation_rad:
            continue

        # We want alpha = R_X beta
        # Build M = sum(beta * alpha.T)
        M += np.outer(beta, alpha)
        used_indices.append(i)

    if len(used_indices) < 2:
        raise ValueError(
            "Not enough useful rotational motion to solve R_X. "
            "Your A and B matrices may be too close to identity."
        )

    U, _, Vt = np.linalg.svd(M)

    # R_X = V U.T
    R_X = Vt.T @ U.T

    # Fix reflection if needed
    if np.linalg.det(R_X) < 0:
        Vt[-1, :] *= -1
        R_X = Vt.T @ U.T

    R_X = project_to_SO3(R_X)

    return R_X, used_indices

def solve_tx(A_list, B_list, R_X):
    """
    Solve translation part after R_X is known.

    From:

        A X = X B

    translation part:

        R_A t_X + t_A = R_X t_B + t_X

    rearrange:

        (I - R_A) t_X = t_A - R_X t_B
    """
    C_blocks = []
    d_blocks = []

    I = np.eye(3)

    for A, B in zip(A_list, B_list):
        R_A = A[:3, :3]
        t_A = A[:3, 3]

        t_B = B[:3, 3]

        C_i = I - R_A
        d_i = t_A - R_X @ t_B

        C_blocks.append(C_i)
        d_blocks.append(d_i)

    C = np.vstack(C_blocks)
    d = np.hstack(d_blocks)

    t_X, residuals, rank, singular_values = np.linalg.lstsq(C, d, rcond=None)

    info = {
        "translation_residuals": residuals,
        "translation_rank": rank,
        "translation_singular_values": singular_values
    }

    return t_X, info

def solve_X(A_list, B_list):
    R_X, used_indices = solve_Rx(A_list, B_list)
    t_X, translation_info = solve_tx(A_list, B_list, R_X)

    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = t_X

    info = {
        "num_total_pairs": len(A_list),
        "num_rotation_pairs_used": len(used_indices),
        "used_rotation_indices": used_indices,
        **translation_info
    }

    return X, R_X, t_X, info


# -----------------------------
# Residual checking
# -----------------------------

def inv_T(T):
    """
    Inverse of a 4x4 rigid transform.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def residual_errors(A_list, B_list, X, keys=None):
    """
    Check how close A_i X is to X B_i.
    """
    rows = []

    for i, (A, B) in enumerate(zip(A_list, B_list)):
        left = A @ X
        right = X @ B

        error_T = inv_T(left) @ right

        R_error = error_T[:3, :3]
        t_error = error_T[:3, 3]

        rot_error_vec = rotation_log_vector(R_error)
        rot_error_deg = np.rad2deg(np.linalg.norm(rot_error_vec))
        trans_error = np.linalg.norm(t_error)

        if keys is not None:
            from_frame, to_frame = keys[i]
        else:
            from_frame, to_frame = None, None

        rows.append({
            "index": i,
            "from_frame": from_frame,
            "to_frame": to_frame,
            "rotation_error_deg": rot_error_deg,
            "translation_error": trans_error
        })

    return pd.DataFrame(rows)


def save_X_to_csv(X, output_csv):
    """
    Save final X matrix to one-row CSV.
    """
    row = {}

    for r in range(4):
        for c in range(4):
            row[f"X{r}{c}"] = X[r, c]

    pd.DataFrame([row]).to_csv(output_csv, index=False)


if __name__ == "__main__":

    np.set_printoptions(precision=10, suppress=True)

    A_keys, A_list = load_transform_list(A_csv)
    B_keys, B_list = load_transform_list(B_csv)

    matched_keys, A_list, B_list = match_A_B(A_keys, A_list, B_keys, B_list)

    # print(A_keys[0], A_list[0])
    # print(A_list.shape, B_list.shape)

    X, R_X, t_X, info = solve_X(A_list, B_list)




    print(R_X)
    print(t_X)
    np.savez(r"C:\Users\Tracy\Projects\Xi_EE_Pose\scripts\hand_eye_calib_result.npz", R_X=R_X, t_X=t_X)

    time.sleep(10)
    calib = np.load("hand_eye_calib_result.npz")
    print("HELLO", calib["R_X"])
    print(calib["t_X"])

