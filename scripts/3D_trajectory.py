import numpy as np
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# OUTPUT_CSV_1 = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\hand_eye_calib_data_1.csv")
# OUTPUT_CSV_2 = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\hand_eye_calib_data_2.csv")

# synced_data = pd.read_csv(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\synced_frame_api.csv", usecols=["manipulator", "field", "data"])

# PSM2_transform = synced_data.loc[(synced_data["manipulator"] == 1) & (synced_data["field"] == "ISI_TIP_TRANSFORM"), "data"].item()
# print (PSM2_transform)



data_path_1 = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\full_API_projection.csv")
data_path_2 = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\xi_data_0514\full_API+3D_projection.csv")
data_path_3 = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\OLDRx_full_API_projection.csv")

NEW_3D = pd.read_csv(data_path_1, usecols=["p_API"])
OLD_3D = pd.read_csv(data_path_3, usecols=["p_API"])
OG_3D = pd.read_csv(data_path_2, usecols=["p_OG"])

full_API_csv = Path(r"C:\Users\Tracy\Projects\Xi_EE_Pose\data\full_API_projection.csv")
Calc_API_Tip = pd.read_csv(full_API_csv)

def parse_3d_column(df, col_name):
    """Parse a 3D point column (string format) and return x, y, z lists."""
    x_coords = []
    y_coords = []
    z_coords = []
    
    for idx, row in df.iterrows():
        point_str = row[col_name]
        
        if isinstance(point_str, str):
            values = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", point_str)
            if not values:
                print(f"Skipping row {idx} from '{col_name}': unable to parse point string")
                continue
            values = np.array(values, dtype=np.float64)
        else:
            values = np.asarray(point_str, dtype=np.float64)
        
        if values.size >= 3:
            x_coords.append(values[0])
            y_coords.append(values[1])
            z_coords.append(values[2])
    
    return x_coords, y_coords, z_coords

# Parse all three datasets
x1, y1, z1 = parse_3d_column(Calc_API_Tip, "p_API")
x2, y2, z2 = parse_3d_column(OLD_3D, "p_API")
x3, y3, z3 = parse_3d_column(OG_3D, "p_OG")

print(f"Calc_API_Tip: {len(x1)} points")
print(f"OLD_3D: {len(x2)} points")
print(f"OG_3D: {len(x3)} points")

# Create color arrays: dark green (1.0) to light green (0.0)
colors1 = np.linspace(1, 0, len(x1)) if len(x1) > 0 else np.array([])
colors2 = np.linspace(1, 0, len(x2)) if len(x2) > 0 else np.array([])
colors3 = np.linspace(1, 0, len(x3)) if len(x3) > 0 else np.array([])

def mark_start_finish(ax, x, y, z, name):
    """
    Mark start point blue and finish point red.
    """
    if len(x) == 0:
        return

    # Start point: blue
    ax.scatter(
        x[0], y[0], z[0],
        c="blue",
        s=80,
        marker="o",
        label=f"{name} start"
    )

    # Finish point: red
    ax.scatter(
        x[-1], y[-1], z[-1],
        c="red",
        s=80,
        marker="X",
        label=f"{name} finish"
    )

# Create 3 subplots in a 1x3 grid
fig = plt.figure(figsize=(18, 5))

# Plot 1: Calc_API_Tip (NEW)
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(x1, y1, z1, c=colors1, cmap='Greens', s=10, label='Calc API Tip')
mark_start_finish(ax1, x1, y1, z1, "NEW")
ax1.set_title('NEW 3D Trajectory (Calc API)', fontsize=12)
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
ax1.set_zlabel('Z Position')
ax1.legend()

# Plot 2: OLD_3D
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(x2, y2, z2, c=colors2, cmap='Greens', s=10, label='Old API')
mark_start_finish(ax2, x2, y2, z2, "OLD")
ax2.set_title('OLD 3D Trajectory (p_API)', fontsize=12)
ax2.set_xlabel('X Position')
ax2.set_ylabel('Y Position')
ax2.set_zlabel('Z Position')
ax2.legend()

# Plot 3: OG_3D
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(x3, y3, z3, c=colors3, cmap='Greens', s=10, label='Original')
mark_start_finish(ax3, x3, y3, z3, "OG")
ax3.set_title('OG 3D Trajectory (p_OG)', fontsize=12)
ax3.set_xlabel('X Position')
ax3.set_ylabel('Y Position')
ax3.set_zlabel('Z Position')
ax3.legend()

plt.tight_layout()
plt.show()

