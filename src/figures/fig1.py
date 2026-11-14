"""Figure 1. (a) The tree-gram architecture, a drawing read from fig1_architecture.png. (b) Loss L_n
against context length n for the tree-gram, four reference language models, their element-wise
minimum (the reference envelope) and ten baselines. (c) Per-logit loss L_i against logit index i
for five tree widths d, with the reference envelope dashed. Arrays read from fig1.npz:
reference_model, reference_n, reference_loss, envelope_loss, <baseline>_n and <baseline>_loss,
treegram_n, treegram_loss, treegram_loss_sd, logit_index, width, per_logit_loss, per_logit_loss_sd.
The 'neural LM + unigram' curve coincides with 'neural LM': the unigram weight selected on
validation data is 0.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

FIGSIZE = (5.6, 2.4)
ARCHITECTURE_DPI = 600
STYLE = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.labelsize": 6.2, "xtick.labelsize": 5.6, "ytick.labelsize": 5.6, "axes.labelpad": 5.0,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "xtick.major.pad": 3.1, "ytick.major.pad": 4.4,
    "legend.frameon": False,
}
FAINT = 0.40
REFERENCE_FORMATS = ("--", "-", ":", "-.")
BASELINES = (
    ("infinigram_unigram", "infini-gram +\nunigram", "-", "#e5b700", None),
    ("infinigram_neural", "infini-gram +\nneural LM", ":", "#e5b700", None),
    ("raw_count", "raw count", "-s", "#2f8f46", None),
    ("spectral_wfa", "spectral WFA", "--", "#2a78d6", None),
    ("neural_lm", "neural LM", "-", "#e86f00", None),
    ("neural_lm_unigram", "neural LM +\nunigram", ":.", "#e86f00", "none"),
    ("mlp", "MLP", "--", "#e86f00", None),
    ("umps", "u-MPS", "-", "#8e44ad", None),
    ("tree_tn", "tree TN", "--", "#8e44ad", None),
    ("ttlm", "TTLM", ":", "#8e44ad", None),
)
PANEL_LABELS = (("(a)", 0.0109, 0.9111), ("(b)", 0.5764, 0.9278), ("(c)", 0.5819, 0.4317))


def _format_axes(ax, xticks, xlim, xlabel, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks, [str(t) for t in xticks])
    ax.set_yticks([2, 3, 4], ["2", "3", "4"])
    ax.minorticks_off()
    ax.set_xlim(*xlim)
    ax.set_ylim(1.1, 4.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main(data_dir=None, out_dir=None) -> Path:
    data_dir = Path(data_dir) if data_dir else ROOT / "data" / "plot_data"
    out_dir = Path(out_dir) if out_dir else ROOT / "plots" / "reproduced"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fig1.pdf"
    architecture = plt.imread(data_dir / "fig1_architecture.png")
    with (np.load(data_dir / "fig1.npz", allow_pickle=False) as z,
          plt.style.context(["default", STYLE])):
        fig = plt.figure(figsize=FIGSIZE)
        w, h = np.divide(architecture.shape[1::-1], FIGSIZE) / ARCHITECTURE_DPI
        ax = fig.add_axes([0, 1 - h, w, h])
        ax.imshow(architecture, interpolation="none", aspect="auto")
        ax.set_axis_off()

        ax = fig.add_axes([0.6406, 0.6223, 0.2119, 0.3435])
        n = z["reference_n"]
        for model, loss, fmt in zip(z["reference_model"], z["reference_loss"], REFERENCE_FORMATS):
            ax.plot(n, loss, fmt, color="#7a4a21", lw=0.55, alpha=FAINT, label=model)
        ax.plot(n, z["envelope_loss"], "--x", color="black", lw=1.0, ms=2.4, mew=0.6,
                markevery=np.flatnonzero(np.isin(n, (2, 3, 4, 8, 16, 32))),
                label="reference\nenvelope")
        for key, label, fmt, colour, face in BASELINES:
            ax.plot(z[f"{key}_n"], z[f"{key}_loss"], fmt, color=colour, mfc=face, lw=0.75, ms=2.7,
                    alpha=FAINT, label=label)
        n, loss, sd = z["treegram_n"], z["treegram_loss"], z["treegram_loss_sd"]
        ax.plot(n, loss, marker="*", color="#c81e3a", lw=1.5, ms=2.5, label="tree-gram")
        ax.fill_between(n, loss - sd, loss + sd, color="#c81e3a", alpha=0.14, lw=0)
        _format_axes(ax, [2, 4, 8, 16, 32], (1.85, 36), "$n$", r"$\mathcal{L}_n$")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0331, 1.0144), fontsize=5.6, handlelength=1.2,
                  handletextpad=0.4, labelspacing=0.18)

        ax = fig.add_axes([0.6406, 0.1310, 0.2119, 0.3435])
        i = z["logit_index"]
        colours = matplotlib.colormaps["viridis"](np.linspace(0.06, 0.88, len(z["width"])))
        curves = zip(z["width"], z["per_logit_loss"], z["per_logit_loss_sd"], colours)
        for d, loss, sd, colour in curves:
            ax.plot(i, loss, color=colour, lw=1.05, label=f"$d={d}$")
            ax.fill_between(i, loss - sd, loss + sd, color=colour, alpha=0.10, lw=0)
        ax.plot(i, z["envelope_loss"], "--", color="black", lw=1.0, label="Empirical $n$-gram")
        _format_axes(ax, [1, 2, 4, 8, 16, 31], (0.9, 34), "logit index $i$", r"$\mathcal{L}_i$")
        ax.legend(loc="upper right", bbox_to_anchor=(0.9944, 1.0101), fontsize=3.6, ncols=2,
                  handlelength=1.0, handletextpad=0.3, columnspacing=0.25, labelspacing=0)

        for text, x, y in PANEL_LABELS:
            fig.text(x, y, text, family="DejaVu Sans", weight="bold", size=5.11, color="#180603",
                     alpha=0.8)
        fig.savefig(out, bbox_inches=None, pad_inches=0, metadata={"CreationDate": None})
        plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
