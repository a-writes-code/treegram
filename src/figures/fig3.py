"""Figure 3: learning-rate transfer from synthetic proxy corpora to real-data targets.

Rows are TinyStories and WikiText-103. For depth L = 2 and 4, a pair of panels plots, against each proxy
corpus's relative complexity, the target's excess loss at the proxy-optimal learning rate (values below
1e-3 nats are drawn hollow at 1e-3) and the log2 proxy-optimal rate, with family medians and the target
optimum as lines. The last panel shows the target's learning-rate sweeps on real data.
Reads fig3.npz: `<corpus>_corpus`, `<corpus>_relative_complexity` and the per-depth `<corpus>_L<L>_*` arrays.
"""
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[2]
logging.getLogger("fontTools").setLevel(logging.ERROR)

STYLE = {
    "font.size": 11.5, "axes.titlesize": 7.5, "axes.labelsize": 11.5,
    "xtick.labelsize": 10.0, "ytick.labelsize": 10.0, "legend.fontsize": 5.5,
    "figure.dpi": 160, "savefig.dpi": 400, "savefig.pad_inches": 0.0, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.22, "grid.linewidth": 0.5, "lines.linewidth": 1.2, "lines.markersize": 3.0,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42,
    "mathtext.fontset": "cm", "font.family": "serif", "font.serif": ["DejaVu Serif"],
}
ROWS = (("tinystories", "TinyStories"), ("wikitext103", "WikiText-103"))
# (x ticks, y ticks) of the five panels in each row
TICKS = {
    "tinystories": (((.5, .75, 1.), (1e-3, 1e-2, 1e-1)), ((.5, .75, 1.), (-3., -1.5)),
                    ((.5, .75, 1.), (1e-3, 1e-2)), ((.5, .75, 1.), (-2., -1.6, -1.2)),
                    ((-4., 0.), (1.5, 3., 4.5))),
    "wikitext103": (((.8, 1.), (1e-2, 1e-1)), ((.75, .9), (-1.2, -.6, 0.)),
                    ((.8, 1.), (1e-3, 1e-2, 1e-1)), ((.75, .9), (-1.2, -.8, -.4)),
                    ((-4., 0.), (4.5, 6.))),
}
MINOR_LOG_TICKS = (np.arange(2, 10) * 10.0 ** np.arange(-4, 1)[:, None]).ravel()
CUTOFF = 1e-3
ORANGE, GREEN, BLUE, RED = "#d97706", "#2ca25f", "#377eb8", "#b00020"
SWEEP_COLORS = {2: "#4c78a8", 4: "#e07b39"}


def groups(names):
    # n32_d1024 belongs to both families and is drawn in both
    return ((names == "bigram", "D", ORANGE),
            (np.char.startswith(names, "n32_d"), "o", GREEN),
            (np.char.startswith(names, "n") & np.char.endswith(names, "_d1024"), "^", BLUE),
            (names == "real", "s", "black"))


def excess_panel(ax, z, c, depth):
    x, y = z[f"{c}_relative_complexity"], z[f"{c}_L{depth}_excess_loss"]
    for (mask, marker, color), size in zip(groups(z[f"{c}_corpus"]), (29, 27, 33, 29)):
        below = mask & (y < CUTOFF)
        above = mask & ~below
        ax.scatter(x[above], y[above], marker=marker, s=size, color=color, edgecolor="white", lw=.4, zorder=5)
        ax.scatter(x[below], np.full(below.sum(), CUTOFF), marker=marker, s=size,
                   facecolors="white", edgecolors=color, lw=1.1, zorder=6)
    medians = [float(z[f"{c}_L{depth}_{family}_median_excess_loss"]) for family in ("fixed_n", "fixed_d_tree")]
    for median, color in zip(medians, (GREEN, BLUE)):
        ax.axhline(max(CUTOFF, median), c=color, ls="--", lw=1, zorder=2)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(FixedLocator(MINOR_LOG_TICKS))
    ax.yaxis.set_minor_formatter(NullFormatter())
    shown = np.maximum(CUTOFF, np.r_[y, medians])
    ax.set_ylim(shown.min() / 1.24, shown.max() * 1.24)
    ax.set_ylabel("Excess loss", labelpad=1.5)
    spread = x.max() - x.min()
    ax.set_xlim(x.min() - .05 * spread, x.max() + .05 * spread)
    ax.set_xlabel("Relative complexity", labelpad=2)


def optimal_lr_panel(ax, z, c, depth):
    x, y = z[f"{c}_relative_complexity"], z[f"{c}_L{depth}_log2_optimal_lr"]
    for (mask, marker, color), size, zorder in zip(groups(z[f"{c}_corpus"]), (22, 20, 25, 23), (5, 5, 6, 6)):
        ax.scatter(x[mask], y[mask], marker=marker, s=size, color=color, edgecolor="white", lw=.35, zorder=zorder)
    ax.axhline(float(z[f"{c}_L{depth}_target_log2_optimal_lr"]), color="black", lw=.9, ls="--", zorder=2)
    for family, color in (("fixed_n", GREEN), ("fixed_d_tree", BLUE)):
        ax.axhline(float(z[f"{c}_L{depth}_{family}_median_log2_optimal_lr"]), color=color, lw=1.0,
                   ls=(0, (3.2, 1.2, .9, 1.2)), zorder=3)
    ax.set_ylabel(r"$\log_2$ (Optimal LR)", labelpad=1.0)
    spread = x.max() - x.min()
    ax.set_xlim(x.min() - .04 * spread, x.max() + .04 * spread)
    ax.set_xlabel("Relative Complexity", labelpad=1.0)


def sweep_panel(ax, z, c):
    lines, bounds = [], []
    for depth, color in SWEEP_COLORS.items():
        key = f"{c}_L{depth}_"
        lr, loss, sd = z[key + "sweep_log2_lr"], z[key + "sweep_loss_mean"], z[key + "sweep_loss_sd"]
        bounds += [loss - sd, loss + sd]
        ax.fill_between(lr, loss - sd, loss + sd, color=color, alpha=.14, lw=0)
        lines += ax.plot(lr, loss, "o-", color=color, ms=2.0, lw=1.0, label=rf"$L={depth}$")
        target = float(z[key + "target_log2_optimal_lr"])
        ax.axvline(target, color=color, ls="--", lw=.9, alpha=.9, zorder=2)
        ax.scatter([target], [float(z[key + "sweep_loss_at_target_lr"])], marker="*", s=34,
                   color=color, edgecolor="white", lw=.35, zorder=5)
    bounds = np.concatenate(bounds)
    low, high = bounds.min(), bounds.max()
    span = high - low
    cross_y = high + .055 * span
    for depth in SWEEP_COLORS:
        diverged = z[f"{c}_L{depth}_sweep_diverged_log2_lr"]
        if diverged.size:
            ax.scatter(diverged, np.full(diverged.size, cross_y), marker="x", s=18, color=RED, lw=.85,
                       clip_on=False, zorder=7)
    ax.set_ylim(low - .03 * span, cross_y + .035 * span)
    ax.set_xlabel(r"$\log_2$ (LR)", labelpad=1.0)
    ax.set_ylabel("Target loss on real data", labelpad=3)
    ax.set_title("Target LR sweeps", fontsize=11.5, pad=7)
    return lines


def main(data_dir=None, out_dir=None) -> Path:
    data_dir = Path(data_dir) if data_dir else ROOT / "data" / "plot_data"
    out_dir = Path(out_dir) if out_dir else ROOT / "plots" / "reproduced"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig3.pdf"
    with np.load(data_dir / "fig3.npz", allow_pickle=False) as npz:
        z = dict(npz)
    with plt.style.context(STYLE, after_reset=True):
        fig, axes = plt.subplots(2, 5, figsize=(10.5, 7.45))
        fig.subplots_adjust(left=.068, right=.991, bottom=.21, top=.875, wspace=.57, hspace=.92)
        for row, (c, name) in enumerate(ROWS):
            for pair, depth in enumerate((2, 4)):
                excess_panel(axes[row, 2 * pair], z, c, depth)
                optimal_lr_panel(axes[row, 2 * pair + 1], z, c, depth)
            sweep_lines = sweep_panel(axes[row, 4], z, c)
            for ax, (xticks, yticks) in zip(axes[row], TICKS[c]):
                ax.xaxis.set_major_locator(FixedLocator(xticks))
                ax.yaxis.set_major_locator(FixedLocator(yticks))
                ax.tick_params(pad=2, length=2.8)
                ax.grid(True, alpha=.18, lw=.4)
            top = axes[row, 0].get_position().y1
            fig.text(.5, top + .065, name, ha="center", va="bottom", fontsize=15)
            for pair, depth in enumerate((2, 4)):
                xmid = (axes[row, 2 * pair].get_position().x0 + axes[row, 2 * pair + 1].get_position().x1) / 2
                title = (rf"$L={depth}$, proxy $d_{{\mathrm{{tfr}}}}=64$, "
                         rf"target $d_{{\mathrm{{tfr}}}}={int(z[f'{c}_L{depth}_target_width'])}$")
                fig.text(xmid, top + .014, title, ha="center", va="bottom", fontsize=11.5)
        markers = [Line2D([], [], marker=marker, ls="", color=color, ms=5.8, label=label)
                   for marker, color, label in (("D", ORANGE, "bigram"), ("o", GREEN, r"fixed $n$"),
                                                ("^", BLUE, r"fixed $d_{\mathrm{tree}}$"), ("s", "black", "real"),
                                                ("x", RED, "diverged target run"))]
        fig.legend(handles=markers + sweep_lines, loc="lower center", bbox_to_anchor=(.5, .082), ncol=7,
                   fontsize=10.5, columnspacing=1.0, handletextpad=.4, frameon=False)
        medians = [tuple(Line2D([], [], c=color, ls=ls, lw=1.2) for color in (GREEN, BLUE)) for ls in ("--", "-.")]
        fig.legend(handles=medians + [Line2D([], [], c="black", ls="--")],
                   labels=["Family-median LR excess", "Family-median log optimal LR", "Target log optimal LR"],
                   loc="lower center", bbox_to_anchor=(.5, .047), ncol=3, fontsize=10.0, handlelength=3.0,
                   frameon=False, handler_map={tuple: HandlerTuple(ndivide=None)})
        hollow = Line2D([], [], marker="o", ls="", mfc="white", mec=".25", ms=6)
        fig.legend(handles=[hollow], labels=[r"Hollow markers: fall below stdev (shown at $10^{-3}$ nats)"],
                   loc="lower center", bbox_to_anchor=(.5, .009), frameon=False, fontsize=10.5)
        fig.savefig(path, pad_inches=0, metadata={"CreationDate": None})
        plt.close(fig)
    return path


if __name__ == "__main__":
    print(main())
