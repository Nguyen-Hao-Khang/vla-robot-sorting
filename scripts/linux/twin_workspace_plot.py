#!/usr/bin/env python3
"""Workspace figures of the Panthera-HT arm for the mid-term report (pure numpy + matplotlib).

Uses only the arm model in twin_kinematics.py (joint chain and limits copied from the robot's
URDF) and the table layout in twin_scene.py; no simulator or GPU is needed, so the figures can
be regenerated on any machine with the repository.

Figures written to --out (default ~/Downloads/twin_report):
  workspace_reach.png   top and side views of every point the tool can reach (random joint
                        samples within the limits), with the table, object row and containers;
  orientation_map.png   which table cells the tool reaches at grasp height and at lift height
                        with the gripper pointing straight down vs tilted 40 deg;
  height_vs_tilt.png    highest tool position over two object slots as the gripper tilts;
  workspace_numbers.csv the numbers behind the last two figures.

Usage: python scripts/linux/twin_workspace_plot.py [--out DIR] [--samples 60000]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import twin_kinematics as tk
import twin_scene as ts

TABLE_Z = 0.74  # table top in the twin world (m)
GRASP_Z = 0.80  # tool height when holding a 5 cm box at mid height
LIFT_Z = GRASP_Z + 0.15  # 15 cm lift
CLEAR_Z = 1.03  # carried object clears the 7 cm deep-bin wall (used in the twin)
TILT_MAP_DEG = 40.0
INK, MUTED, GRID = "#14213D", "#5A6673", "#D5DCE3"
BLUE, ORANGE, TEAL = "#2F6FEB", "#E3A35B", "#1F7A8C"


def orientation(xy, tilt_deg: float, flip: bool = False) -> np.ndarray:
    """Tool rotation: x = approach (down, leaned away from the base by tilt), fingers horizontal."""
    base = np.array(tk.BASE_XYZ[:2])
    lean = np.asarray(xy, float) - base
    lean = lean / np.linalg.norm(lean)
    lean3 = np.array([lean[0], lean[1], 0.0])
    t = math.radians(tilt_deg)
    a = math.cos(t) * np.array([0.0, 0.0, -1.0]) + math.sin(t) * lean3
    y = np.cross([0.0, 0.0, 1.0], lean3)
    y = -y if flip else y
    z = np.cross(a, y)
    return np.column_stack([a, y / np.linalg.norm(y), z / np.linalg.norm(z)])


def reachable(p, tilt_deg: float) -> bool:
    return any(tk.ik(p, orientation(p[:2], tilt_deg, flip)) is not None for flip in (False, True))


def max_height(xy, tilt_deg: float, lo: float = TABLE_Z, hi: float = 1.40) -> float | None:
    if not reachable((xy[0], xy[1], lo + 0.03), tilt_deg):
        return None
    lo += 0.03
    for _ in range(14):  # ~0.05 mm resolution
        mid = 0.5 * (lo + hi)
        if reachable((xy[0], xy[1], mid), tilt_deg):
            lo = mid
        else:
            hi = mid
    return lo


def layout(ax) -> None:
    (x0, x1), (y0, y1) = ts.TABLE_BOUNDS
    ax.add_patch(plt_rect((x0, y0), x1 - x0, y1 - y0, fc="#F2F4F7", ec=GRID, lw=1.5, z=0))
    for c in (ts.WIDE_TRAY, ts.DEEP_BIN):
        (cx, cy), (hx, hy) = c.xy, c.inner_half
        ax.add_patch(plt_rect((cx - hx, cy - hy), 2 * hx, 2 * hy, fc="none", ec=INK, lw=1.5, z=3))
        ax.text(cx, cy - hy - 0.025, c.name, ha="center", va="top", fontsize=8, color=INK)
    ax.scatter(
        ts.OBJECT_SLOTS_X,
        [ts.OBJECT_ROW_Y] * 4,
        s=40,
        marker="s",
        color=INK,
        zorder=4,
        label="object slots",
    )
    ax.scatter(
        [tk.BASE_XYZ[0]],
        [tk.BASE_XYZ[1]],
        s=120,
        marker="^",
        color=ORANGE,
        zorder=5,
        label="robot base",
    )


def plt_rect(xy, w, h, fc, ec, lw, z):
    from matplotlib.patches import Rectangle

    return Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path.home() / "Downloads" / "twin_report")
    ap.add_argument("--samples", type=int, default=60000)
    ap.add_argument("--grid", type=float, default=0.04, help="map cell size (m)")
    args = ap.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK})
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # 1. Reachable cloud
    lo = np.array([a for a, _ in tk.LIMITS])
    hi = np.array([b for _, b in tk.LIMITS])
    qs = rng.uniform(lo, hi, size=(args.samples, 6))
    pts = np.array([tk.fk(q)[0] for q in qs])
    above = pts[:, 2] >= TABLE_Z
    fig, (top, side) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
    top.scatter(pts[above, 0], pts[above, 1], s=1, alpha=0.08, color=BLUE, rasterized=True)
    layout(top)
    top.set(
        xlabel="x (m)",
        ylabel="y (m)",
        title="Top view: tool positions above the table",
        xlim=(-0.95, 0.95),
        ylim=(-1.0, 0.75),
        aspect="equal",
    )
    top.legend(loc="lower left", fontsize=8, frameon=False)
    r = np.hypot(pts[:, 0] - tk.BASE_XYZ[0], pts[:, 1] - tk.BASE_XYZ[1])
    side.scatter(r, pts[:, 2], s=1, alpha=0.06, color=BLUE, rasterized=True)
    side.axhline(TABLE_Z, color=INK, lw=1.5)
    side.text(0.02, TABLE_Z - 0.03, "table top", color=INK, fontsize=8, va="top")
    for z, lab in ((CLEAR_Z, "height needed to carry over the bin wall"),):
        side.axhline(z, color=ORANGE, lw=1.2, ls="--")
        side.text(0.02, z + 0.01, lab, color="#9A5B17", fontsize=8)
    near = math.dist(tk.BASE_XYZ[:2], (0.06, ts.OBJECT_ROW_Y))
    far = math.dist(tk.BASE_XYZ[:2], (ts.OBJECT_SLOTS_X[-1], ts.OBJECT_ROW_Y))
    side.axvspan(min(near, far) - 0.05, max(near, far) + 0.05, color=GRID, alpha=0.5, zorder=0)
    side.text(near, 0.45, "object row", ha="center", fontsize=8, color=MUTED)
    side.set(
        xlabel="horizontal distance from the base axis (m)",
        ylabel="height z (m)",
        title="Side view: reach envelope (any orientation)",
        xlim=(0, 1.0),
        ylim=(0.3, 1.6),
    )
    fig.tight_layout()
    fig.savefig(args.out / "workspace_reach.png", dpi=200)
    plt.close(fig)

    # 2. Orientation maps
    (x0, x1), (y0, y1) = ts.TABLE_BOUNDS
    xs = np.arange(x0 + args.grid / 2, x1, args.grid)
    ys = np.arange(y0 + 0.1 + args.grid / 2, y1, args.grid)
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    for i, z in enumerate((GRASP_Z, LIFT_Z)):
        for j, tilt in enumerate((0.0, TILT_MAP_DEG)):
            ax = axes[i, j]
            ok = np.zeros((len(ys), len(xs)), bool)
            for a, y in enumerate(ys):
                for b, x in enumerate(xs):
                    ok[a, b] = reachable((x, y, z), tilt)
            rows += [("map", z, tilt, float(ok.mean()))]
            ax.imshow(
                ok,
                origin="lower",
                extent=(x0, x1, ys[0] - args.grid / 2, ys[-1] + args.grid / 2),
                cmap=matplotlib.colors.ListedColormap(["#FFFFFF", "#BFD3FB"]),
                vmin=0,
                vmax=1,
                zorder=1,
                alpha=0.95,
            )
            layout(ax)
            name = "pointing straight down" if tilt == 0 else f"tilted {tilt:.0f} deg"
            ax.set_title(
                f"z = {z:.2f} m, gripper {name}: {100 * ok.mean():.0f}% of cells", fontsize=10
            )
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal")
    axes[1, 0].set_xlabel("x (m)")
    axes[1, 1].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)  [grasp height]")
    axes[1, 0].set_ylabel("y (m)  [after 15 cm lift]")
    fig.suptitle("Shaded = the tool can reach this point with this gripper orientation", color=INK)
    fig.tight_layout()
    fig.savefig(args.out / "orientation_map.png", dpi=200)
    plt.close(fig)

    # 3. Highest reachable tool position vs tilt over two object slots
    tilts = list(range(0, 65, 5))
    slots = {
        "near slot (x=0.06)": (ts.OBJECT_SLOTS_X[2], ts.OBJECT_ROW_Y),
        "far slot (x=0.18)": (ts.OBJECT_SLOTS_X[3], ts.OBJECT_ROW_Y),
    }
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (label, xy), color in zip(slots.items(), (BLUE, INK), strict=True):
        hs = [max_height(xy, t) for t in tilts]
        rows += [("height", label, t, h) for t, h in zip(tilts, hs, strict=True)]
        ax.plot(
            tilts,
            [h if h is not None else np.nan for h in hs],
            marker="o",
            color=color,
            label=label,
            lw=2,
        )
    ax.axhline(CLEAR_Z, color=ORANGE, ls="--", lw=1.5)
    ax.text(
        1,
        CLEAR_Z + 0.005,
        f"needed to carry over the bin wall ({CLEAR_Z:.2f} m)",
        color="#9A5B17",
        fontsize=9,
    )
    ax.set(
        xlabel="gripper tilt from vertical (deg)",
        ylabel="highest tool position z (m)",
        title="How high the arm can lift over an object slot",
    )
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "height_vs_tilt.png", dpi=200)
    plt.close(fig)

    with (args.out / "workspace_numbers.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["figure", "slot_or_height", "tilt_deg", "value"])
        w.writerows(rows)
    for row in rows:
        print(*row, sep="\t")
    print(f"written: {args.out}/workspace_reach.png, orientation_map.png, height_vs_tilt.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
