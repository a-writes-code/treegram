"""Figure 4: transfer from pre-training corpora to fine-tuning corpora.

Three 11x11 heatmaps, for an L=4 transformer (TinyStories) and L=2 and L=4 transformers (WikiText-103). Each cell is
the seed-median final validation loss of a model pre-trained on the row corpus and fine-tuned on the column corpus,
divided by the column corpus's loss floor, minus one. A few cells come from fine-tuning re-runs at other peak learning
rates: among the rates run with the most seeds (3), the one with the best seed-median final loss. For the WikiText-103
L=2 n=32, d_tree=64 diagonal this is the best of three lower rates x 3 seeds. Each panel has its own colour scale;
green boxes mark matched source and target.
Reads `corpora`, `tinystories_L4`, `wikitext103_L2` and `wikitext103_L4` from fig4.npz.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
STYLE = {
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
}
PANELS = (
    ("tinystories_L4", "$L=4$ transformer (TinyStories)"),
    ("wikitext103_L2", "$L=2$ transformer (WikiText-103)"),
    ("wikitext103_L4", "$L=4$ transformer (WikiText-103)"),
)


def corpus_label(key, separator):
    if key in ("bigram", "real"):
        return key
    n, d = key[1:].split("_d")
    return f"$n={n}${separator}$d_{{\\mathrm{{tree}}}}={d}$"


def draw_matrix(ax, matrix, title, corpora, show_xlabels):
    vmax = matrix.max()
    image = ax.imshow(matrix, cmap="magma_r", vmin=0, vmax=vmax, aspect="auto")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=4.8,
                color="white" if value > 0.55 * vmax else ".2")
    for edge in (0.5, 5.5, 9.5):
        ax.axhline(edge, color="white", lw=1.35, zorder=6)
        ax.axvline(edge, color="white", lw=1.35, zorder=6)
    for index in range(len(matrix)):
        ax.add_patch(Rectangle((index - 0.5, index - 0.5), 1, 1, fill=False, ec="#1f9e4a", lw=1.0, zorder=8))
    ax.set_xticks(range(len(corpora)))
    ax.set_yticks(range(len(corpora)))
    ax.set_yticklabels([corpus_label(key, ", ") for key in corpora], fontsize=5.5)
    if show_xlabels:
        ax.set_xticklabels([corpus_label(key, "\n") for key in corpora], rotation=90, fontsize=5.5)
        ax.set_xlabel("fine-tuning target", fontsize=8.0, labelpad=4)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(pad=1.2, length=2.2)
    ax.set_ylabel("pre-training source", fontsize=7.4, labelpad=3)
    ax.set_title(title, fontsize=8.6, pad=2.5)
    return image


def main(data_dir=None, out_dir=None) -> Path:
    data_dir = Path(data_dir) if data_dir else ROOT / "data" / "plot_data"
    out_dir = Path(out_dir) if out_dir else ROOT / "plots" / "reproduced"
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_dir / "fig4.npz", allow_pickle=False) as data:
        corpora = [str(key) for key in data["corpora"]]
        matrices = {key: data[key] for key, _ in PANELS}
    path = out_dir / "fig4.pdf"
    with plt.style.context(["default", STYLE]):
        fig = plt.figure(figsize=(6.5, 7.3))
        grid = fig.add_gridspec(3, 2, width_ratios=[1, 0.032], left=0.19, right=0.88, bottom=0.17, top=0.965,
                                hspace=0.32, wspace=0.06)
        axes = [fig.add_subplot(grid[row, 0]) for row in range(3)]
        for row, (ax, (key, title)) in enumerate(zip(axes, PANELS)):
            image = draw_matrix(ax, matrices[key], title, corpora, show_xlabels=row == 2)
            colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[row, 1]))
            ticks = np.linspace(0.0, image.norm.vmax, 5)
            colorbar.set_ticks(ticks, labels=[f"{tick:.4f}" for tick in ticks])
            colorbar.set_label("val loss / target floor $-1$", fontsize=7.2, labelpad=4)
            colorbar.ax.tick_params(labelsize=6.2, pad=2, length=2.5)
            colorbar.ax.grid(True, alpha=0.22, linewidth=0.5)
        fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.01, metadata={"CreationDate": None})
        plt.close(fig)
    return path


if __name__ == "__main__":
    print(main())
