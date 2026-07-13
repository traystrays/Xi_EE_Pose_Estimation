"""Plot direct and FK hand-eye trajectories as unconnected 3D points."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "xi_ee_0623_162612"
NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+\.?\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def parse_translation(value: object) -> np.ndarray:
    values = np.asarray(NUMBER_PATTERN.findall(str(value)), dtype=np.float64)
    if values.size != 3:
        raise ValueError(f"Expected 3 translation values, got {values.size}")
    return values


def set_equal_limits(ax, positions: np.ndarray) -> None:
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct",
        type=Path,
        default=DATA_DIR / "EE_in_camera_hand_eye.csv",
    )
    parser.add_argument(
        "--fk",
        type=Path,
        default=DATA_DIR / "EE_in_camera_FK_hand_eye.csv",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=DATA_DIR / "EE_hand_eye_direct_vs_FK_trajectory.png",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direct = pd.read_csv(args.direct)[["clip_frame_id", "Translation"]].rename(
        columns={"Translation": "direct_translation"}
    )
    fk = pd.read_csv(args.fk)[["clip_frame_id", "Translation"]].rename(
        columns={"Translation": "fk_translation"}
    )
    matched = direct.merge(fk, on="clip_frame_id", how="inner", validate="one_to_one")
    if matched.empty:
        raise SystemExit("No matching clip_frame_id values")

    direct_points = np.stack(matched["direct_translation"].map(parse_translation))
    fk_points = np.stack(matched["fk_translation"].map(parse_translation))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        direct_points[:, 0],
        direct_points[:, 1],
        direct_points[:, 2],
        color="tab:red",
        s=12,
        alpha=0.75,
        label="Direct + hand-eye",
    )
    ax.scatter(
        fk_points[:, 0],
        fk_points[:, 1],
        fk_points[:, 2],
        color="tab:cyan",
        s=12,
        alpha=0.75,
        label="FK + hand-eye",
    )
    set_equal_limits(ax, np.vstack((direct_points, fk_points)))
    ax.set_xlabel("Camera X (m)")
    ax.set_ylabel("Camera Y (m)")
    ax.set_zlabel("Camera Z (m)")
    ax.set_title(f"Direct and FK hand-eye trajectories ({len(matched)} matched frames)")
    ax.legend()
    fig.tight_layout()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=200, bbox_inches="tight")
    print(f"Matched frames: {len(matched)}")
    print(f"Direct mean position: {direct_points.mean(axis=0)}")
    print(f"FK mean position: {fk_points.mean(axis=0)}")
    print(f"Saved: {args.save.resolve()}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
