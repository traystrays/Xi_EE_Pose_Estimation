"""Create a live-dashboard animation from EE translation data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


def parse_translation(value: str) -> tuple[float, float, float]:
    values = np.fromstring(str(value).strip().strip("[]"), sep=" ")
    if values.size != 3:
        return np.nan, np.nan, np.nan
    return float(values[0]), float(values[1]), float(values[2])


def padded_limits(values: np.ndarray, padding: float = 0.08) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    low, high = np.quantile(finite, [0.01, 0.99])
    span = high - low
    if span == 0:
        span = max(abs(low), 1.0) * 0.1
    return low - padding * span, high + padding * span


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=5, help="Animation frames per second")
    args = parser.parse_args()

    output = args.output or args.csv_path.parent / "translation_live_dashboard.gif"
    output.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.csv_path)
    timestamps = pd.to_datetime(data["video_ts"], errors="raise", utc=True)
    xyz = np.asarray(data["Translation"].map(parse_translation).tolist(), dtype=float)
    elapsed = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()

    median_dt = np.nanmedian(np.diff(elapsed))
    source_fps = 1.0 / median_dt if median_dt > 0 else args.fps
    stride = max(1, round(source_fps / args.fps))
    frame_indices = np.arange(0, len(data), stride)
    if frame_indices[-1] != len(data) - 1:
        frame_indices = np.append(frame_indices, len(data) - 1)

    x, y, z = xyz.T
    physically_plausible = np.isfinite(xyz).all(axis=1) & (np.abs(xyz) <= 10).all(axis=1)
    plausible_xyz = xyz[physically_plausible]
    if plausible_xyz.size == 0:
        raise ValueError("No finite Translation samples within the +/-10 display range")
    xlim, ylim, zlim = (
        padded_limits(plausible_xyz[:, column]) for column in range(3)
    )
    plausible = (
        physically_plausible
        & (x >= xlim[0]) & (x <= xlim[1])
        & (y >= ylim[0]) & (y <= ylim[1])
        & (z >= zlim[0]) & (z <= zlim[1])
    )
    trail = xyz.copy()
    trail[~plausible] = np.nan

    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
    grid = figure.add_gridspec(3, 2, width_ratios=(1.05, 1.45))
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    trace_axes = [figure.add_subplot(grid[row, 1]) for row in range(3)]

    axis_3d.set(xlim=xlim, ylim=ylim, zlim=zlim, xlabel="X", ylabel="Y", zlabel="Z")
    axis_3d.set_title("EE Translation Live Trajectory")
    trajectory = axis_3d.scatter([], [], [], color="#2563EB", s=10, alpha=0.7)
    current_point = axis_3d.scatter([], [], [], color="#DC2626", s=55)

    colors = ("#D1495B", "#00798C", "#5C4D7D")
    trace_lines = []
    trace_points = []
    for axis, values, label, color in zip(trace_axes, (x, y, z), "XYZ", colors):
        axis.set_xlim(elapsed[0], elapsed[-1])
        axis.set_yscale("symlog", linthresh=1e-4)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        line, = axis.plot([], [], color=color, linewidth=1.2)
        point, = axis.plot([], [], marker="o", color="#111827", markersize=4)
        trace_lines.append(line)
        trace_points.append(point)
    trace_axes[-1].set_xlabel("Elapsed time (seconds)")
    status = figure.suptitle("", fontsize=10)

    def update(animation_index: int):
        index = int(frame_indices[animation_index])
        visible_trail = trail[: index + 1]
        trajectory._offsets3d = (
            visible_trail[:, 0], visible_trail[:, 1], visible_trail[:, 2]
        )
        if plausible[index]:
            current_point._offsets3d = ([x[index]], [y[index]], [z[index]])
        else:
            current_point._offsets3d = ([], [], [])

        for line, point, values in zip(trace_lines, trace_points, (x, y, z)):
            line.set_data(elapsed[: index + 1], values[: index + 1])
            if np.isfinite(values[index]):
                point.set_data([elapsed[index]], [values[index]])
            else:
                point.set_data([], [])
        status.set_text(
            f"frame {index + 1}/{len(data)}  |  t={elapsed[index]:.2f}s  |  "
            f"display rate {args.fps} fps"
        )
        return trajectory, current_point, status, *trace_lines, *trace_points

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000 / args.fps,
        blit=False,
    )
    if output.suffix.lower() == ".mp4":
        try:
            import imageio_ffmpeg

            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        writer = FFMpegWriter(fps=args.fps, codec="h264", extra_args=["-pix_fmt", "yuv420p"])
    else:
        writer = PillowWriter(fps=args.fps)
    animation.save(output, writer=writer)
    plt.close(figure)
    print(f"Saved {len(frame_indices)}-frame dashboard animation to {output.resolve()}")


if __name__ == "__main__":
    main()
