import numpy as np
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# stitch together csv data points of pixel location of end effector

# OUTPUT_CSV_1 = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\hand_eye_calib_data_1.csv")
# OUTPUT_CSV_2 = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\hand_eye_calib_data_2.csv")

# synced_data = pd.read_csv(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\synced_frame_api.csv", usecols=["manipulator", "field", "data"])

# PSM2_transform = synced_data.loc[(synced_data["manipulator"] == 1) & (synced_data["field"] == "ISI_TIP_TRANSFORM"), "data"].item()
# print (PSM2_transform)



# data_path = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\KinCsvLog_14-05-2026_20-09-43.699 (1).csv")
# data_path = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\synced_frame_api.csv")

data_path = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\full_API_projection.csv")

# synced_data = pd.read_csv(data_path, usecols=["field", "manipulator", "data"])


synced_data = pd.read_csv(data_path, usecols=["p_API"])

x_coords = []
y_coords = []
z_coords = []

for idx, row in synced_data.iterrows():
    p_API = row["p_API"]

    if isinstance(p_API, str):
        p_API = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", p_API)
        if not p_API:
            print(f"Skipping row {idx}: unable to parse point string")
            continue
        p_API = np.array(p_API, dtype=np.float64)

    else:
        p_API = np.asarray(p_API, dtype=np.float64)

    if p_API.size >= 3:
        x_coords.append(p_API[0])
        y_coords.append(p_API[1])
        z_coords.append(p_API[2])
        print(f"Included point {idx}: [{p_API[0]}, {p_API[1]}, {p_API[2]}]")

# position_rows = synced_data[(synced_data["manipulator"] == 2) & (synced_data["field"] == "ISI_TIP_POSITION")]
# position_rows = position_rows.iloc[:20000]
# print(f"Found {len(position_rows)} position rows for manipulator 2 (limited to first 20000).")

# for idx, row in position_rows.iterrows():
#     PSM2_transform = row["data"]

#     if isinstance(PSM2_transform, str):
#         PSM2_transform = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", PSM2_transform)
#         if not PSM2_transform:
#             print(f"Skipping row {idx}: unable to parse point string")
#             continue
#         PSM2_transform = np.array(PSM2_transform, dtype=np.float64)

#     else:
#         PSM2_transform = np.asarray(PSM2_transform, dtype=np.float64)

#     if PSM2_transform.size >= 3:
#         x_coords.append(PSM2_transform[0])
#         y_coords.append(PSM2_transform[1])
#         z_coords.append(PSM2_transform[2])
#         print(f"Included point {idx}: [{PSM2_transform[0]}, {PSM2_transform[1]}, {PSM2_transform[2]}]")

print(f"Total included points: {len(x_coords)}")

# 3. Create the 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

print((x_coords[:5], y_coords[:5], z_coords[:5]))

# Plot only the points as individual dots
ax.scatter(x_coords, y_coords, z_coords, c=range(len(x_coords)), cmap='viridis', s=10, label='Tip Positions')

# Customize plot labels and title
ax.set_title('3D Tip Positions', fontsize=14)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.legend()

plt.show()