"""Plot x, y, and z translation values from an EE transform CSV over time."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_translation(value: str) -> tuple[float, float, float]:
    values = np.fromstring(str(value).strip().strip("[]"), sep=" ")
    if values.size != 3:
        raise ValueError(f"Expected 3 Translation values, found {values.size}: {value!r}")
    return float(values[0]), float(values[1]), float(values[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="CSV containing video_ts and Translation")
    parser.add_argument("--output-dir", type=Path, help="Directory for the three PNG graphs")
    args = parser.parse_args()

    output_dir = args.output_dir or args.csv_path.parent / "translation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.csv_path)
    required = {"video_ts", "Translation"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")

    timestamps = pd.to_datetime(data["video_ts"], errors="raise", utc=True)
    data[["x", "y", "z"]] = pd.DataFrame(
        data["Translation"].map(parse_translation).tolist(), index=data.index
    )
    data["elapsed_seconds"] = (timestamps - timestamps.iloc[0]).dt.total_seconds()

    colors = {"x": "#D1495B", "y": "#00798C", "z": "#5C4D7D"}
    figure, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, constrained_layout=True
    )
    figure.suptitle("EE Translation X, Y, and Z Through Time", fontsize=16)

    for axes_for_value, axis_name in zip(axes, ("x", "y", "z")):
        axes_for_value.plot(
            data["elapsed_seconds"],
            data[axis_name],
            color=colors[axis_name],
            linewidth=1.0,
        )
        axes_for_value.set_ylabel(f"{axis_name.upper()} translation")
        axes_for_value.set_yscale("symlog", linthresh=1e-4)
        axes_for_value.grid(True, alpha=0.3, linewidth=0.7)
        axes_for_value.margins(x=0)

    axes[-1].set_xlabel("Elapsed time (seconds)")
    figure.savefig(output_dir / "translation_xyz_over_time.png", dpi=180)
    plt.close(figure)

    print(f"Plotted {len(data)} rows to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
