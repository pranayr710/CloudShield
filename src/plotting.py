"""Shared plotting style helpers for all notebooks."""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIGDIR = Path(__file__).parent.parent / "reports" / "figures"

def set_style():
    """Set the house style for all plots."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.figsize": (10, 6),
    })

def save(fig, name):
    """Save figure to reports/figures/ with tight layout."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {name}.png")

set_style()
