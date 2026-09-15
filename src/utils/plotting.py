"""CloudShield - shared plot styling (spec section 10).

Every figure across the three tracks uses the same palette, sizing and save
path so the final report and dashboard look like one project rather than three.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES, FIG_DPI, PALETTE


def set_style() -> None:
    """Apply the house style. Call once at the top of each notebook."""
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": FIG_DPI,
        "figure.figsize": (10, 6),
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": True,
    })


def save_fig(fig, name: str, subdir: str = "", quiet: bool = False) -> Path:
    """Save a figure into results/figures/ with a tight bounding box."""
    out = FIGURES / subdir if subdir else FIGURES
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    if not quiet:
        print(f"saved -> {path.relative_to(FIGURES.parents[1])}")
    return path
