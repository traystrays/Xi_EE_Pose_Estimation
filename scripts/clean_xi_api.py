import pandas as pd
import numpy as np
from pathlib import Path
from threeD_trajectory import parse_3d_column
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Each input data file only contains 1 manip information
# Matches video frame time with timestamp of api data

# Change path for input and output files

# manip 1 -> ECM in this case, both absolute paths 
input_data = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\20260616_OR_Xi_EE\20260616_185054.606\KinManip2_20260616_185054.633_2026-06-16_18.csv") 
video_data = Path(r"C:\Users\Tracy\Documents\2026 RCL UBC\20260616_OR_Xi_EE\video_stream_1_20260616_185106_timestamps.txt") 
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data" / "xi_ee_0616_185054_new_measurements"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

matched_path = OUTPUT_PATH/"EE_transforms_direct.csv"

t_xyz_path = OUTPUT_PATH/"xyz_position_filtered.csv"
t_a_xyz_path = OUTPUT_PATH/"xyz_position_translation_first.csv"
wrong_t_xyz_path = OUTPUT_PATH/"xyz_position_wrong_row_major.csv"
translation_animation_path = OUTPUT_PATH/"translation_position_vs_time.gif"


def format_frame_ranges(frame_ids):
    frame_ids = sorted(set(int(frame_id) for frame_id in frame_ids))
    if not frame_ids:
        return "none"

    ranges = []
    start = prev = frame_ids[0]

    for frame_id in frame_ids[1:]:
        if frame_id == prev + 1:
            prev = frame_id
            continue

        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = frame_id

    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(ranges)


def parse_xi_transform(transform_str):
    values = np.fromstring(
        str(transform_str)
        .replace("[", " ")
        .replace("]", " ")
        .replace(",", " "),
        sep=" ",
    )

    if values.size != 12:
        raise ValueError(f"Expected 12 values, got {values.size}")



    # INCORRECT
    # Xi ISI_TIP_TRANSFORM format is row-major 3x4:
    # [R00 R01 R02 tx R10 R11 R12 ty R20 R21 R22 tz]
    Wrong_T = np.eye(4, dtype=np.float64)
    T_temp = values.reshape(3, 4)
    Wrong_T[:3, :3] = T_temp[:, :3]
    Wrong_T[:3, 3] = T_temp[:, 3]

    wrong_t = Wrong_T[:3,3]
    wrong_t_string = f"[{wrong_t[0]:.10f} {wrong_t[1]:.10f} {wrong_t[2]:.10f}]"
    

    # CORRECT
    # Xi_ISI_TIP_TRANSFORM format, translation-first:
    # [tx ty tz R00 R01 R02 R10 R11 R12 R20 R21 R22]
    T = np.eye(4, dtype=np.float64)
    t = values[:3]
    R = values[3:].reshape(3,3)
    T[:3,:3] = R
    T[:3,3] = t

    return T, wrong_t_string


def parse_transform_values(transform_str):
    return np.fromstring(
        str(transform_str)
        .replace("[", " ")
        .replace("]", " ")
        .replace(",", " "),
        sep=" ",
    )


def format_array(array):
    return np.array2string(
        np.asarray(array, dtype=np.float64),
        precision=10,
        suppress_small=False,
        separator=" ",
    )


def format_position(position):
    position = np.asarray(position, dtype=np.float64).reshape(3)
    return f"[{position[0]:.10f} {position[1]:.10f} {position[2]:.10f}]"


def remove_zero_positions(df, col_name="t_position", eps=1e-6, max_abs_value=10.0):
    valid_rows = []
    removed_rows = []

    for idx, row in df.iterrows():
        values = parse_transform_values(row[col_name])
        is_valid = (
            values.size == 3
            and np.isfinite(values).all()
            and not np.any(np.abs(values) < eps)
            and not np.any(np.abs(values) > max_abs_value)
        )

        if is_valid:
            valid_rows.append(idx)
        else:
            removed_rows.append(idx)

    cleaned = df.loc[valid_rows].copy()
    print(f"Removed zero/invalid t_xyz rows: {len(removed_rows)}")
    print(f"Remaining t_xyz rows: {len(cleaned)}")
    return cleaned


# read csv to dataframe
api_data = pd.read_csv(input_data, usecols=["system_timestamp_ms","field","data"]) 

# Keep only rows needed for the matched output.
output_fields = [
    "ISI_TIP_TRANSFORM",
    "ISI_SUJ_JOINT_VALUES",
    "ISI_JOINT_VALUES",
]
mask = api_data["field"].isin(output_fields)
api_data = api_data[mask].copy()

api_data["robot_ts"] = (
    pd.to_datetime(
        api_data["system_timestamp_ms"],
        unit="ms",  
        utc=True,
    )
    .dt.tz_convert("America/Vancouver")
)

video_frame = pd.read_csv(video_data, header=None, names=["video_ts"])
video_frame["video_ts"] = (
    pd.to_datetime(video_frame["video_ts"])
    .dt.tz_localize("America/Vancouver")
)

video_frame["frame_id"] = range(len(video_frame))

# Normalize both columns to the same datetime resolution for merge_asof.
timestamp_dtype = "datetime64[ns, America/Vancouver]"
api_data["robot_ts"] = api_data["robot_ts"].astype(timestamp_dtype)
video_frame["video_ts"] = video_frame["video_ts"].astype(timestamp_dtype)

# merge_asof requires both timestamp columns to be sorted.
api_data = api_data.sort_values("robot_ts")
video_frame = video_frame.sort_values("video_ts")

# match transform robot time to each video frame time
transform_data = api_data[api_data["field"] == "ISI_TIP_TRANSFORM"].copy()
matched = pd.merge_asof(
    video_frame,
    transform_data[["robot_ts", "data"]],
    left_on="video_ts",
    right_on="robot_ts",
    direction="nearest",
    tolerance=pd.Timedelta("20ms"), # tolerance per frame is 20 ms
)

for field_name in ["ISI_SUJ_JOINT_VALUES", "ISI_JOINT_VALUES"]:
    field_data = api_data[api_data["field"] == field_name].copy()
    field_data = field_data.rename(columns={"data": field_name})
    matched = pd.merge_asof(
        matched.sort_values("video_ts"),
        field_data[["robot_ts", field_name]],
        left_on="video_ts",
        right_on="robot_ts",
        direction="nearest",
        tolerance=pd.Timedelta("20ms"),
    ).rename(columns={"robot_ts_x": "robot_ts"})

    if "robot_ts_y" in matched.columns:
        matched = matched.drop(columns=["robot_ts_y"])

unmatched_list = (
    matched.loc[matched["robot_ts"].isna(), "frame_id"]
    .astype(int)
    .tolist()
)
unmatched_count = int(matched["robot_ts"].isna().sum())

matched["time_error_ms"] = (
    matched["video_ts"] - matched["robot_ts"]
).dt.total_seconds().abs() * 1000

# Frames without a robot sample within 20 ms are not valid matches.

matched = matched.dropna(subset=["robot_ts", "data"]).copy()
matched = matched.rename(columns={"data": "Raw_Transform"})

parsed_transforms = []
invalid_frame_ids = []
t_rows = []
t_a_rows = []
wrong_t_rows = []

for idx, row in matched.iterrows():
    try:
        # get right Transform matrix and WRONG translation string
        Transform, wrong_t_string = parse_xi_transform(row["Raw_Transform"])

        parsed_transforms.append(Transform)

        t_a_rows.append(format_position(Transform[:3, 3]))
        wrong_t_rows.append(wrong_t_string)

    except ValueError:
        parsed_transforms.append(None)
        invalid_frame_ids.append(int(row["frame_id"]))

t_a_xyz = pd.DataFrame({"t_a": t_a_rows})
wrong_t_xyz = pd.DataFrame({"wrong_t": wrong_t_rows})

matched["Transform_Matrix"] = parsed_transforms
matched = matched.dropna(subset=["Transform_Matrix"]).copy()

matched["Transform"] = matched["Transform_Matrix"].apply(format_array)
matched["Rotation"] = matched["Transform_Matrix"].apply(lambda T: format_array(T[:3, :3]))
matched["Translation"] = matched["Transform_Matrix"].apply(lambda T: format_array(T[:3, 3]))
matched = matched[
    [
        "frame_id",
        "video_ts",
        "robot_ts",
        "time_error_ms",
        "Raw_Transform",
        "ISI_SUJ_JOINT_VALUES",
        "ISI_JOINT_VALUES",
        "Transform",
        "Rotation",
        "Translation",
    ]
]

matched.to_csv(matched_path, index=False)

print(f"Saved {len(matched)} matched frames to: {matched_path}")
print(f"Unmatched frames tolerance: {unmatched_count}")
print(f"Invalid transform strings skipped: {len(invalid_frame_ids)}")



def remove_outliers(csv_path):

    df = pd.read_csv(csv_path)
    valid_rows = []
    removed_frame_ids = []
    Rotation_col = "Rotation"
    Translation_col = "Translation"
    count = 0
    for idx, row in df.iterrows():
        try:
            Rotation = row[Rotation_col]

            # turn everything into a string
            Rotation = parse_transform_values(row[Rotation_col])
            Translation = parse_transform_values(row[Translation_col])

            if Rotation.size != 9 or Translation.size != 3:
                raise ValueError("Rotation or Translation has the wrong number of values")

            R = Rotation.reshape(3, 3)
            det_R = np.linalg.det(R)
            orth_error = np.linalg.norm(R.T @ R - np.eye(3))

            zero_count = np.sum(np.abs(Translation) < 1e-6)

            is_valid = (
                np.isfinite(Rotation).all()
                and np.isfinite(Translation).all()
                and np.all(np.abs(Rotation) <= 1.0)
                and np.all(np.abs(Translation) <= 10)
                and zero_count < 2
                )

            if is_valid:
                valid_rows.append(idx)
                count+=1
            else:
                removed_frame_ids.append(int(row["frame_id"]))

        except (TypeError, ValueError):
            removed_frame_ids.append(int(row["frame_id"]))

    cleaned = df.loc[valid_rows].copy()
    cleaned.to_csv(csv_path, index=False)

    print(f"Removed invalid transforms: {len(removed_frame_ids)}")
    print(f"Remaining valid transforms: {count}")
    # print(f"Removed frame IDs: {format_frame_ranges(removed_frame_ids)}")


    return cleaned

cleaned = remove_outliers(matched_path)


def plot_positions(position_df, col_name, title):
    x, y, z = parse_3d_column(position_df, col_name)

    if len(x) == 0:
        print(f"No points to plot for: {title}")
        return

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = np.linspace(1, 0, len(x))

    ax.scatter(x, y, z, c=colors, cmap="viridis", s=8)
    ax.scatter(x[0], y[0], z[0], c="blue", s=80, marker="o", label="start")
    ax.scatter(x[-1], y[-1], z[-1], c="red", s=80, marker="X", label="finish")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.tight_layout()


def animate_translation_over_time(df, timeline_df, output_path):
    valid_rows = []

    for _, row in df.iterrows():
        translation = parse_transform_values(row["Translation"])
        if translation.size != 3 or not np.isfinite(translation).all():
            continue

        valid_rows.append(
            {
                "frame_id": int(row["frame_id"]),
                "x": translation[0],
                "y": translation[1],
                "z": translation[2],
            }
        )

    if not valid_rows:
        print("No valid Translation rows available for animation.")
        return

    # Keep every video frame in the animation. Frames without a valid transform
    # have NaN coordinates, so they consume time but do not move or draw the tip.
    valid_df = pd.DataFrame(valid_rows)
    anim_df = (
        timeline_df[["frame_id", "video_ts"]]
        .merge(valid_df, on="frame_id", how="left", validate="one_to_one")
        .sort_values("video_ts")
        .reset_index(drop=True)
    )
    elapsed_s = (
        anim_df["video_ts"] - anim_df["video_ts"].iloc[0]
    ).dt.total_seconds().to_numpy()

    x = anim_df["x"].to_numpy()
    y = anim_df["y"].to_numpy()
    z = anim_df["z"].to_numpy()
    is_valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    valid_x = x[is_valid]
    valid_y = y[is_valid]
    valid_z = z[is_valid]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    margin = 0.02
    ax.set_xlim(np.min(valid_x) - margin, np.max(valid_x) + margin)
    ax.set_ylim(np.min(valid_y) - margin, np.max(valid_y) + margin)
    ax.set_zlim(np.min(valid_z) - margin, np.max(valid_z) + margin)
    ax.set_title("Translation Position vs Time")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    line, = ax.plot([], [], [], color="tab:blue", linewidth=2)
    gap_line, = ax.plot(
        [], [], [], color="lightskyblue", linewidth=2, linestyle="--"
    )
    point = ax.scatter([], [], [], color="red", s=60)
    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def update(frame_idx):
        # NaNs break the solid line wherever the data is invalid.
        solid_x = x[: frame_idx + 1]
        solid_y = y[: frame_idx + 1]
        solid_z = z[: frame_idx + 1]
        line.set_data(solid_x, solid_y)
        line.set_3d_properties(solid_z)

        # Connect valid points across invalid frame ranges with dashed segments.
        drawn_valid_indices = np.flatnonzero(is_valid[: frame_idx + 1])
        gap_x = []
        gap_y = []
        gap_z = []
        for previous_idx, current_idx in zip(
            drawn_valid_indices[:-1], drawn_valid_indices[1:]
        ):
            if current_idx > previous_idx + 1:
                gap_x.extend([x[previous_idx], x[current_idx], np.nan])
                gap_y.extend([y[previous_idx], y[current_idx], np.nan])
                gap_z.extend([z[previous_idx], z[current_idx], np.nan])

        gap_line.set_data(gap_x, gap_y)
        gap_line.set_3d_properties(gap_z)

        if is_valid[frame_idx]:
            point._offsets3d = ([x[frame_idx]], [y[frame_idx]], [z[frame_idx]])
        else:
            point._offsets3d = ([], [], [])

        time_text.set_text(
            f"frame {int(anim_df.iloc[frame_idx]['frame_id']) + 1}/{len(anim_df)} "
            f"| t={elapsed_s[frame_idx]:.3f}s"
        )
        return line, gap_line, point, time_text

    if len(elapsed_s) > 1:
        median_dt = np.median(np.diff(elapsed_s))
        fps = int(np.clip(round(1.0 / median_dt), 1, 30)) if median_dt > 0 else 10
    else:
        fps = 10

    animation = FuncAnimation(
        fig,
        update,
        frames=len(anim_df),
        interval=1000 / fps,
        blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps))
    print(f"Saved Translation animation: {output_path}")



# wrong_t_xyz = remove_zero_positions(wrong_t_xyz, col_name="wrong_t")

# t_xyz.to_csv(t_xyz_path, index=False)
t_a_xyz.to_csv(t_a_xyz_path, index=False)
wrong_t_xyz.to_csv(wrong_t_xyz_path, index=False)

plot_positions(cleaned, "Translation", "Matched Xi 185054 Manip3 Positions, translation-first")
# plot_positions(t_a_xyz, "t_a", "Matched Xi Tip Positions, translation-first filtered")
plot_positions(wrong_t_xyz, "wrong_t", "Matched Xi Tip Positions, row-major translation")
# animate_translation_over_time(cleaned, video_frame, translation_animation_path)

plt.show()
