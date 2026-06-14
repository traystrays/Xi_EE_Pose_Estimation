"""
Match saved frames with Transform values.

For each frame_number in saved_frames.csv, finds the row in
hand_eye_calibration_data_METERS.csv where frame_id == frame_number,
and outputs the corresponding Transform.

Output: matched_transforms.csv with columns:
    frame_number, image_filename, frame_id, frame_time, Transform
"""

import os
import pandas as pd
import scipy.io as sio
import numpy as np
import pandas as pd
import ast
import re


SAVED_FRAMES_CSV = r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0612\saved_frames.csv"
TRANSFORM_CSV = r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0612\hand_eye_calibration_data_METERS.csv"
OUTPUT_CSV = r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\si_data_0612\matched_transforms.csv"


# def main():
#     saved_frames = pd.read_csv(SAVED_FRAMES_CSV)
#     transforms = pd.read_csv(TRANSFORM_CSV)

#     # Merge on frame_number (left) == frame_id (right)
#     merged = saved_frames.merge(
#         transforms,
#         left_on="frame_number",
#         right_on="frame_id",
#         how="left"
#     )

#     # Report any unmatched frame numbers
#     missing = merged[merged["frame_id"].isna()]
#     if not missing.empty:
#         print("Warning: no Transform found for frame_number(s):")
#         print(missing["frame_number"].tolist())

#     os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
#     merged.to_csv(OUTPUT_CSV, index=False)
#     print(f"Saved matched results to: {OUTPUT_CSV}")
#     print(merged.head())


# if __name__ == "__main__":
#     main()

# df = pd.read_csv(OUTPUT_CSV)

# num_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# def reformat_transform(s):
#     nums = [float(x) for x in num_pattern.findall(s)]
#     if len(nums) != 16:
#         raise ValueError(f"Expected 16 numbers, got {len(nums)}: {s}")

#     rows = []
#     for i in range(4):
#         row_vals = nums[i*4:(i+1)*4]
#         row_str = " ".join(f"{v: .5f}" for v in row_vals)
#         rows.append(f"[{row_str}]")

#     return "[" + "\n ".join(rows) + "]"

# df["Transform"] = df["Transform"].apply(reformat_transform)

# df.to_csv(OUTPUT_CSV, index=False)
# print("Done. Reformatted all Transform rows.")

df = pd.read_csv(OUTPUT_CSV)



transforms = []
for s in df["Transform"]:
    s_clean = s.replace("[", "").replace("]", "")
    # Split into 4 rows by newline
    rows = s_clean.strip().split("\n")
    mat = []
    for row in rows:
        mat.append([float(x) for x in row.split()])
    mat = np.array(mat)
    assert mat.shape == (4, 4), f"Unexpected shape: {mat.shape}"
    transforms.append(mat)

transforms_array = np.array(transforms)  # shape (30, 4, 4)
print(transforms_array.shape)
sio.savemat("gripper_transforms.mat", {"gripperTforms": transforms_array})