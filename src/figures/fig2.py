"""Figure 2: excess last-token loss over H_n for the tree-gram and for parameter-matched
transformers of depth L, (a) against context length n and (b) against tree-gram rank d.
H_n is the loss of a separate, larger transformer (12 layers, width 1024, context 128) trained
on the same data, at the position where it has seen n - 1 tokens; it is not the reference
envelope of Figure 1.
Bands are +/-1 SD over seeds, non-positive excess sits on a 1e-3 floor, and (b) adds
power-law fits for the tree-gram and the deepest transformer.

Reads fig2.npz: depth, context_length, rank, {n,d}_treegram_{excess,std},
{n,d}_transformer_{excess,std} (one row per depth), d_transformer_params_median and
d_fit_{x,y,beta,r2} (rows: tree-gram, deepest transformer).
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[2]
RC = {
    "axes.labelsize": 7.5, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
    "savefig.dpi": 400, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "mathtext.fontset": "cm", "font.family": "serif",
}
FLOOR = 1e-3
TREE_COLOR, TREE_FIT_COLOR = "#c81e3a", "#8f1028"
Y_MAJOR = 10.0 ** np.arange(-3, 1)
Y_MINOR = (np.arange(2.0, 10.0) * 10.0 ** np.arange(-4, 0)[:, None]).ravel()
TOP_MINOR = [60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900]
PANELS = (("n", "context_length", r"context length $n$", (8e-4, 1.5)),
          ("d", "rank", r"tree-gram rank $d$", (4e-2, 1.2)))


def excess_line(ax, x, excess, std, *, color, alpha, lw, marker, label=None, zorder=2):
    shown = np.where(excess > 0, excess, FLOOR)
    ax.plot(x, shown, color=color, alpha=alpha, lw=lw, marker=marker,
            ms=3.8 if marker == "*" else 2.0, label=label, zorder=zorder)
    ax.fill_between(x, np.maximum(shown - std, FLOOR), np.maximum(shown + std, FLOOR),
                    color=color, alpha=.12 * alpha, lw=0, zorder=zorder - 1)
    below = excess <= 0
    if below.any():
        ax.scatter(x[below], np.full(below.sum(), FLOOR), marker="v",
                   s=18 if marker == "*" else 10, facecolor="white", edgecolor=color,
                   linewidth=.65, alpha=alpha, zorder=zorder + 2, clip_on=False)


def main(data_dir=None, out_dir=None) -> Path:
    data_dir = Path(data_dir or ROOT / "data" / "plot_data")
    out_dir = Path(out_dir or ROOT / "plots" / "reproduced")
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_dir / "fig2.npz", allow_pickle=False) as f:
        z = dict(f)
    depth = z["depth"]
    deepest = int(depth.max())
    viridis = matplotlib.colormaps["viridis"]
    norm = Normalize(float(depth.min()), float(deepest))
    colors = [viridis(norm(L)) for L in depth]
    path = out_dir / "fig2.pdf"

    with plt.style.context(["default", RC]):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.91875))
        for ax, (p, xkey, xlabel, (lo, hi)) in zip(axes, PANELS):
            x = z[xkey]
            for excess, std, color in zip(z[f"{p}_transformer_excess"],
                                          z[f"{p}_transformer_std"], colors):
                excess_line(ax, x, excess, std, color=color, alpha=.28, lw=.65, marker="o")
            excess_line(ax, x, z[f"{p}_treegram_excess"], z[f"{p}_treegram_std"],
                        color=TREE_COLOR, alpha=1.0, lw=1.9, marker="*",
                        label="tree-gram", zorder=8)
            ax.set_xscale("log", base=2)
            ax.set_xticks(x, [str(v) for v in x])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator(Y_MAJOR))
            ax.yaxis.set_minor_locator(FixedLocator(Y_MINOR[(Y_MINOR >= lo) & (Y_MINOR <= hi)]))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(xlabel)
            ax.set_ylim(lo, hi)
            ax.grid(True, which="both", alpha=.18, linewidth=.4)

        deep_fit_color = tuple(.65 * np.asarray(viridis(norm(deepest))[:3]))
        for y, color, zorder in zip(z["d_fit_y"], (TREE_FIT_COLOR, deep_fit_color), (9, 7)):
            axes[1].plot(z["d_fit_x"], y, "--", color=color, lw=1.0, zorder=zorder)
        (tree_beta, deep_beta), (tree_r2, deep_r2) = z["d_fit_beta"], z["d_fit_r2"]
        axes[1].text(.03, .045, "\n".join((
            r"fit $\Delta\mathcal{L}\propto d^{\beta}$",
            rf"tree: $\beta={tree_beta:+.2f}$  ($R^2={tree_r2:.2f}$)",
            rf"$L={deepest}$: $\beta={deep_beta:+.2f}$  ($R^2={deep_r2:.2f}$)")),
            transform=axes[1].transAxes, ha="left", va="bottom", fontsize=3.8,
            bbox=dict(fc="white", ec="none", alpha=.82, pad=.8), zorder=20)

        axes[0].set_ylabel(r"excess loss $\mathcal{L}_{\rm last}-H_n$")
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(Line2D([], [], marker="v", ls="none", ms=3.2, mfc="white",
                              mec="0.35", mew=.6))
        labels.append(r"$\mathcal{L}-H_n\leq 0$ (floor)")
        axes[0].legend(handles, labels, loc="upper left", fontsize=4.2, handlelength=1.2,
                       labelspacing=.15)

        top = axes[1].secondary_xaxis("top")
        top.set_xticks(z["rank"], [f"{p / 1e6:.1f}M" for p in z["d_transformer_params_median"]],
                       fontsize=4.3)
        top.xaxis.set_minor_locator(FixedLocator(TOP_MINOR))
        top.xaxis.set_minor_formatter(NullFormatter())
        top.set_xlabel("matched transformer parameters", fontsize=5.2, labelpad=1.5)
        top.tick_params(length=1.6, pad=1.0)

        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=viridis),
                          cax=fig.add_axes([.925, .29, .012, .47]), ticks=depth)
        cb.set_label(r"transformer depth $L$", fontsize=5.2, labelpad=2)
        cb.ax.tick_params(labelsize=4.4, length=1.5, pad=1)
        cb.outline.set_linewidth(.5)

        fig.subplots_adjust(left=0.105, right=0.90, top=0.77, bottom=0.28, wspace=0.22)
        for ax, letter in zip(axes, "ab"):
            ax.text(-.13, 1.13, f"({letter})", transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=7.5, fontweight="bold", clip_on=False)
        fig.savefig(path, bbox_inches=None, pad_inches=0, metadata={"CreationDate": None})
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(main())
