"""Per-cycle champion vs challenger ATE bar chart for cls_out10.

Mirrors visualisation_tool/experiment/plot_generation_ate.py, but reads from
the recovered 04_cycle_ate_data.txt (this session has no eval_scores in zarr).

Columns in the txt file:
  cycle  champion_ate  challenger_ate  outcome
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

SESSION_DIR = Path(__file__).parents[1]
DATA_FILE   = Path(__file__).parent / "04_cycle_ate_data.txt"

CHAMPION_COLOR = "#4C78A8"
ACCEPTED_COLOR = "green"
REJECTED_COLOR = "red"


def _grid(ax):
    for which, a, lw in [("major", 0.25, 0.6), ("minor", 0.10, 0.35)]:
        ax.grid(which=which, alpha=a, linewidth=lw)


def load_data(path):
    cycles, champ, chall, swapped = [], [], [], []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            cycles.append(int(p[0]))
            champ.append(float(p[1]))
            chall.append(float(p[2]))
            swapped.append(p[3].upper() == "SWAPPED")
    return (
        np.array(cycles),
        np.array(champ),
        np.array(chall),
        np.array(swapped),
    )


def save_legend(out_dir):
    fig, ax = plt.subplots(figsize=(3.5, 0.8))
    ax.axis("off")
    handles = [
        Patch(color=CHAMPION_COLOR, label="champion"),
        Patch(color=ACCEPTED_COLOR, label="challenger accepted"),
        Patch(color=REJECTED_COLOR, label="challenger rejected"),
    ]
    ax.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=9)
    fig.savefig(out_dir / "legend.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    cycles, ate_cur, ate_new, swapped = load_data(DATA_FILE)

    x     = np.arange(len(cycles))
    width = 0.34

    fig, ax = plt.subplots(figsize=(max(8.0, len(cycles) * 0.4), 2.5))

    for xi, ac, an, sw in zip(x, ate_cur, ate_new, swapped):
        if np.isfinite(ac):
            ax.bar(xi - width / 2, ac, width, color=CHAMPION_COLOR, zorder=3)
        if np.isfinite(an):
            color = ACCEPTED_COLOR if sw else REJECTED_COLOR
            ax.bar(xi + width / 2, an, width, color=color, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(cycles, fontsize=max(8, min(9, 180 // len(cycles))))
    ax.set_xlabel("cycle", fontsize=12)
    ax.set_ylabel("ATE RMSE (m)", fontsize=12)
    ax.set_yscale("log")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _grid(ax)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_dir = SESSION_DIR / "generation_ate_plots"
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / "generation_ate.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    save_legend(out_dir)
    print(f"Saved → {out_dir / 'legend.png'}")


if __name__ == "__main__":
    main()
