"""Plot direct and FK hand-eye trajectories as unconnected 3D points."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def best_fit_rigid_transform(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return R, t, transformed source minimizing ||target - (R source + t)||."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center

    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    translation = target_center - rotation @ source_center
    transformed = (rotation @ source.T).T + translation
    return rotation, translation, transformed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct",
        type=Path,
        required=True,
        help="CSV containing the direct hand-eye trajectory",
    )
    parser.add_argument(
        "--fk",
        type=Path,
        required=True,
        help="CSV containing the FK hand-eye trajectory",
    )
    return parser.parse_args()


def load_trajectory(path: Path, translation_name: str) -> pd.DataFrame:
    """Load a trajectory using either frame_id or clip_frame_id as its key."""
    data = pd.read_csv(path)
    frame_column = next(
        (name for name in ("frame_id", "clip_frame_id") if name in data.columns),
        None,
    )
    if frame_column is None:
        raise ValueError(
            f"{path} must contain either a 'frame_id' or 'clip_frame_id' column"
        )
    if "Translation" not in data.columns:
        raise ValueError(f"{path} must contain a 'Translation' column")
    return data[[frame_column, "Translation"]].rename(
        columns={frame_column: "frame_id", "Translation": translation_name}
    )


def main() -> None:
    args = parse_args()
    direct = load_trajectory(args.direct, "direct_translation")
    fk = load_trajectory(args.fk, "fk_translation")
    matched = direct.merge(fk, on="frame_id", how="inner", validate="one_to_one")
    if matched.empty:
        raise SystemExit("No matching frame_id values")

    direct_points = np.stack(matched["direct_translation"].map(parse_translation))
    fk_points = np.stack(matched["fk_translation"].map(parse_translation))
    tre_m = np.linalg.norm(direct_points - fk_points, axis=1)
    tre_mm = tre_m * 1000.0
    rotation, translation, aligned_fk_points = best_fit_rigid_transform(
        fk_points, direct_points
    )
    aligned_tre_mm = (
        np.linalg.norm(direct_points - aligned_fk_points, axis=1) * 1000.0
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        direct_points[:, 0],
        direct_points[:, 1],
        direct_points[:, 2],
        color="tab:red",
        s=12,
        alpha=0.75,
        label="Direct",
    )
    ax.scatter(
        fk_points[:, 0],
        fk_points[:, 1],
        fk_points[:, 2],
        color="tab:cyan",
        s=12,
        alpha=0.75,
        label="FK",
    )
    set_equal_limits(ax, np.vstack((direct_points, fk_points)))
    ax.set_xlabel("Camera X (m)")
    ax.set_ylabel("Camera Y (m)")
    ax.set_zlabel("Camera Z (m)")
    ax.set_title(
        f"Direct and FK hand-eye trajectories ({len(matched)} matched frames)\n"
        f"Mean TRE: {tre_mm.mean():.2f} mm"
    )
    ax.legend()
    fig.tight_layout()

    aligned_fig = plt.figure(figsize=(10, 8))
    aligned_ax = aligned_fig.add_subplot(111, projection="3d")
    aligned_ax.scatter(
        direct_points[:, 0],
        direct_points[:, 1],
        direct_points[:, 2],
        color="tab:red",
        s=12,
        alpha=0.75,
        label="Direct (ground truth)",
    )
    aligned_ax.scatter(
        aligned_fk_points[:, 0],
        aligned_fk_points[:, 1],
        aligned_fk_points[:, 2],
        color="tab:green",
        s=12,
        alpha=0.75,
        label="FK + best-fit transform",
    )
    set_equal_limits(
        aligned_ax, np.vstack((direct_points, aligned_fk_points))
    )
    aligned_ax.set_xlabel("Camera X (m)")
    aligned_ax.set_ylabel("Camera Y (m)")
    aligned_ax.set_zlabel("Camera Z (m)")
    aligned_ax.set_title(
        f"Direct and aligned FK trajectories ({len(matched)} matched frames)\n"
        f"Mean post-alignment TRE: {aligned_tre_mm.mean():.2f} mm"
    )
    aligned_ax.legend()
    aligned_fig.tight_layout()

    print(f"Matched frames: {len(matched)}")
    print(f"Direct mean position: {direct_points.mean(axis=0)}")
    print(f"FK mean position: {fk_points.mean(axis=0)}")
    print("Before alignment TRE (Target Registration Error):")
    print(f"  Mean:   {tre_mm.mean():.3f} mm")
    print(f"  Median: {np.median(tre_mm):.3f} mm")
    print(f"  RMS:    {np.sqrt(np.mean(tre_mm**2)):.3f} mm")
    print(f"  Min:    {tre_mm.min():.3f} mm")
    print(f"  Max:    {tre_mm.max():.3f} mm")
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    print("Best-fit rigid transform (FK to direct):")
    print(transform)
    print("After alignment TRE:")
    print(f"  Mean:   {aligned_tre_mm.mean():.3f} mm")
    print(f"  Median: {np.median(aligned_tre_mm):.3f} mm")
    print(f"  RMS:    {np.sqrt(np.mean(aligned_tre_mm**2)):.3f} mm")
    print(f"  Min:    {aligned_tre_mm.min():.3f} mm")
    print(f"  Max:    {aligned_tre_mm.max():.3f} mm")
    plt.show()


if __name__ == "__main__":
    main()
