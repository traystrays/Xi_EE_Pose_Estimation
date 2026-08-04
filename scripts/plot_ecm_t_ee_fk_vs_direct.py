"""Plot FK-derived and direct ECM-to-end-effector translation trajectories.

The two inputs are matched by ``frame_id`` before plotting:

    ECM_T_EE_from_FK.csv
        Uses ``ECM_T_EE_Translation`` when available, otherwise extracts the
        translation from ``ECM_T_EE``.

    EE_transforms_correct.csv
        Extracts the translation from ``Transform``.

Example:
    python scripts/plot_ecm_t_ee_fk_vs_direct.py
    python scripts/plot_ecm_t_ee_fk_vs_direct.py --save data/trajectory_comparison.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "xi_ee_0616_185054_new_measurements"
NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+\.?\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def parse_numbers(value: object) -> np.ndarray:
    """Return all floating-point values found in a CSV string field."""
    return np.asarray(NUMBER_PATTERN.findall(str(value)), dtype=np.float64)


def parse_translation(value: object) -> np.ndarray:
    """Parse a three-element translation vector."""
    values = parse_numbers(value)
    if values.size != 3:
        raise ValueError(f"Expected 3 translation values, got {values.size}")
    return values


def transform_translation(value: object) -> np.ndarray:
    """Extract the translation column from a row-major 4x4 transform."""
    values = parse_numbers(value)
    if values.size != 16:
        raise ValueError(f"Expected 16 transform values, got {values.size}")
    return values.reshape(4, 4)[:3, 3]


def load_fk_translations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "frame_id" not in df:
        raise ValueError(f"{csv_path} does not contain frame_id")

    if "ECM_T_EE_Translation" in df:
        translations = df["ECM_T_EE_Translation"].map(parse_translation)
    elif "Transform" in df:
        translations = df["Transform"].map(transform_translation)
    elif "ECM_T_EE" in df:
        translations = df["ECM_T_EE"].map(transform_translation)
    elif "ECM_T_EE" in df:
        translations = df["ECM_T_EE"].map(transform_translation)
    else:
        raise ValueError(
            f"{csv_path} contains neither ECM_T_EE_Translation nor ECM_T_EE"
        )

    result = df[["frame_id"]].copy()
    result["fk_translation"] = translations
    return result


def load_direct_translations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"frame_id", "Transform"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    result = df[["frame_id"]].copy()
    result["direct_translation"] = df["Transform"].map(transform_translation)
    return result


def set_equal_3d_limits(ax, positions: np.ndarray) -> None:
    """Use the same physical scale for the X, Y, and Z plot axes."""
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def rigidly_align(source: np.ndarray, target: np.ndarray):
    """Return the rigid transform and aligned points mapping source to target."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center

    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T

    translation = target_center - rotation @ source_center
    aligned = (rotation @ source.T).T + translation
    return rotation, translation, aligned


def tre_mm(measured: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Return pointwise target registration errors in millimetres."""
    return np.linalg.norm(measured - ground_truth, axis=1) * 1000.0


def print_tre_summary(label: str, errors_mm: np.ndarray) -> None:
    """Print common summary statistics for pointwise TRE values."""
    print(label)
    print(f"  Mean:   {errors_mm.mean():.3f} mm")
    print(f"  Median: {np.median(errors_mm):.3f} mm")
    print(f"  RMS:    {np.sqrt(np.mean(errors_mm**2)):.3f} mm")
    print(f"  Min:    {errors_mm.min():.3f} mm")
    print(f"  Max:    {errors_mm.max():.3f} mm")


def plot_trajectories(
    measured: np.ndarray,
    direct: np.ndarray,
    measured_label: str,
    title: str,
):
    """Plot a measured trajectory against the direct ground-truth trajectory."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        measured[:, 0],
        measured[:, 1],
        measured[:, 2],
        label=measured_label,
        color="tab:blue",
        s=10,
        alpha=0.8,
    )
    ax.scatter(
        direct[:, 0],
        direct[:, 1],
        direct[:, 2],
        label="Direct (ground truth)",
        color="tab:orange",
        s=10,
        alpha=0.8,
    )

    ax.scatter(*measured[0], color="tab:blue", marker="o", s=60)
    ax.scatter(*direct[0], color="tab:orange", marker="o", s=60)
    ax.scatter(*measured[-1], color="tab:blue", marker="X", s=70)
    ax.scatter(*direct[-1], color="tab:orange", marker="X", s=70)

    set_equal_3d_limits(ax, np.vstack((measured, direct)))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def build_plots(matched: pd.DataFrame):
    fk = np.stack(matched["fk_translation"].to_numpy())
    direct = np.stack(matched["direct_translation"].to_numpy())
    errors_before = tre_mm(fk, direct)
    rotation, translation, aligned_fk = rigidly_align(fk, direct)
    errors_after = tre_mm(aligned_fk, direct)

    print_tre_summary("Before alignment TRE (FK vs direct):", errors_before)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    print("Best-fit rigid transform (FK to direct):")
    np.set_printoptions(suppress=True, precision=8)
    print(transform)
    print_tre_summary("After best-fit alignment TRE:", errors_after)

    original_fig = plot_trajectories(
        fk,
        direct,
        "FK: inv(Base_T_ECM) @ Base_T_EE",
        f"ECM-to-EE translation comparison ({len(matched)} matched frames)\n"
        f"Mean TRE: {errors_before.mean():.3f} mm",
    )
    aligned_fig = plot_trajectories(
        aligned_fk,
        direct,
        "FK + best-fit rigid transform",
        f"Best-fit aligned ECM-to-EE comparison ({len(matched)} matched frames)\n"
        f"Mean TRE: {errors_after.mean():.3f} mm",
    )
    return original_fig, aligned_fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fk",
        type=Path,
        default=DEFAULT_DATA_DIR / "EE_in_ECM_FK.csv",
        help="FK-derived ECM_T_EE CSV",
    )
    parser.add_argument(
        "--direct",
        type=Path,
        default=DEFAULT_DATA_DIR / "EE_transforms_direct.csv",
        help="Direct end-effector transform CSV",
    )
    parser.add_argument("--save", type=Path, help="Optional output image path")
    parser.add_argument(
        "--no-show", action="store_true", help="Do not open the interactive plot window"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fk = load_fk_translations(args.fk)
    direct = load_direct_translations(args.direct)
    matched = (
        fk.merge(direct, on="frame_id", how="inner", validate="one_to_one")
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    if matched.empty:
        raise SystemExit("The two CSVs have no matching frame_id values.")

    print(f"FK rows: {len(fk)}")
    print(f"Direct rows: {len(direct)}")
    print(f"Matched frames: {len(matched)}")
    print(f"Frame range: {matched.frame_id.iloc[0]} to {matched.frame_id.iloc[-1]}")

    original_fig, aligned_fig = build_plots(matched)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        original_fig.savefig(args.save, dpi=200, bbox_inches="tight")
        aligned_path = args.save.with_name(
            f"{args.save.stem}_best_fit{args.save.suffix}"
        )
        aligned_fig.savefig(aligned_path, dpi=200, bbox_inches="tight")
        print(f"Saved original plot: {args.save.resolve()}")
        print(f"Saved aligned plot: {aligned_path.resolve()}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(original_fig)
        plt.close(aligned_fig)


if __name__ == "__main__":
    main()
