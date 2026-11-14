"""Figure 5: the seven fixed tree topologies and the MLP ablation.

(a) Per-logit validation loss of the seven topologies (seed mean, with a seed-SD band where
there are several seeds); top right, each topology's tree, labelled with its last-logit loss;
(b) tau_0 and tau_-1 with MLP blocks (rows 0 and -1 of panel a) and without them at matched
parameter count, against the empirical n-gram entropy; (c) spread (max - min over the seven
topologies) of the validation loss during training, with and without MLP blocks.

Reads fig5.npz: logit_index, topology_labels, per_logit_loss_mean/std,
no_mlp_per_logit_loss_mean/std (rows tau_0, tau_-1), ngram_entropy, last_logit_loss_spread,
tree_nodes, tree_edges ((child, parent) node indices; the last edge joins the two top nodes,
the final merge), tree_edge_group_size (children merged by the parent; 1 for the final merge,
which is drawn as a single edge), training_tokens, ar_loss_spread_mlp/no_mlp.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

ROOT = Path(__file__).resolve().parents[2]
STYLE = {
    "axes.labelsize": 7.5, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
    "figure.dpi": 160, "savefig.dpi": 400, "savefig.pad_inches": 0.0,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "mathtext.fontset": "cm", "font.family": "serif", "font.serif": ["DejaVu Serif"],
}
MLP, NO_MLP = "#3b3b98", "#2ca25f"
ABLATION_ROWS = ((0, "-"), (-1, (0, (3, 1.5))))
TREE_SLOTS = [(0.002, 0.58), (0.251, 0.58), (0.500, 0.58), (0.749, 0.58),
              (0.126, 0.105), (0.375, 0.105), (0.624, 0.105)]
TOKEN_MINOR_TICKS = [m * 10.0**e for e in range(4, 9) for m in range(2, 10) if m * 10.0**e < 9e8]


def line_and_band(axis, x, mean, std, color, alpha, **line):
    axis.plot(x, mean, color=color, **line)
    if std.any():
        axis.fill_between(x, mean - std, mean + std, color=color, alpha=alpha, lw=0)


def per_logit_axis(axis):
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 2, 4, 8, 16, 31], labels=["1", "2", "4", "8", "16", "31"])
    axis.minorticks_off()
    axis.set_yticks([1.5, 2.0, 2.5, 3.0, 3.5])
    axis.set_xlim(0.9, 34)
    axis.set_ylim(1.3, 3.8)
    axis.set_xlabel("logit index $i$", labelpad=1.0)
    axis.set_ylabel("Per-logit Val loss", fontsize=7.2)


def draw(z):
    layout = dict(left=0.078, right=0.998, bottom=0.13, top=0.945, hspace=0.30, wspace=0.18)
    figure, ((axis_a, topologies), (axis_b, axis_c)) = plt.subplots(
        2, 2, figsize=(5.5, 2.75), gridspec_kw=layout)
    box = topologies.get_position()
    topologies.set_position([box.x0 - 0.055, box.y0, box.width + 0.055, box.height])
    i, labels = z["logit_index"], z["topology_labels"]
    mean, std = z["per_logit_loss_mean"], z["per_logit_loss_std"]

    colors = matplotlib.colormaps["viridis"](np.linspace(0.08, 0.86, len(labels)))
    for row_mean, row_std, color in zip(mean, std, colors):
        line_and_band(axis_a, i, row_mean, row_std, color, 0.08, lw=0.9)
    axis_a.text(0.015, 0.035, f"spread at $i=31$: {float(z['last_logit_loss_spread']):.3f} nats",
                transform=axis_a.transAxes, fontsize=5.0)
    per_logit_axis(axis_a)

    topologies.set_facecolor("none")
    topologies.set_xticks([])
    topologies.set_yticks([])
    topologies.set_title("topologies (last token loss in nats)", fontsize=7.0, pad=1.5)
    for spine in topologies.spines.values():
        spine.set_visible(False)
    nodes = z["tree_nodes"]
    for edges, group_sizes, label, loss, color, slot in zip(
            z["tree_edges"], z["tree_edge_group_size"], labels, mean[:, -1], colors, TREE_SLOTS):
        tree = topologies.inset_axes([*slot, 0.245, 0.34])
        for edge, group_size in zip(edges, group_sizes):
            single = group_size == 1
            tree.plot(*nodes[edge].T, color="#2563eb" if single else "#dc2626",
                      lw=0.34 if single else 0.47, alpha=0.85, zorder=1)
        tree.scatter(*nodes.T, s=0.78, color=".72", edgecolor="none", zorder=2)
        tree.set_xlim(-0.02, 1.02)
        tree.set_ylim(-0.25, 5.35)
        tree.axis("off")
        tree.text(0.5, -0.07, f"{label} ({loss:.3f})", transform=tree.transAxes, ha="center",
                  va="top", fontsize=5.6, color=color)

    for row, ls in ABLATION_ROWS:
        line_and_band(axis_b, i, mean[row], std[row], MLP, 0.07, ls=ls, lw=1.0)
        line_and_band(axis_b, i, z["no_mlp_per_logit_loss_mean"][row],
                      z["no_mlp_per_logit_loss_std"][row], NO_MLP, 0.07, ls=ls, lw=1.0)
    axis_b.plot(i, z["ngram_entropy"], "--", color="0.35", lw=1.0)
    per_logit_axis(axis_b)
    for title, color, anchor in (("MLP", MLP, 0.72), ("no MLP", NO_MLP, 1.0)):
        handles = [Line2D([], [], color=color, ls=ls, lw=1.2, label=labels[row])
                   for row, ls in ABLATION_ROWS]
        axis_b.add_artist(axis_b.legend(
            handles=handles, title=title, loc="upper right", bbox_to_anchor=(anchor, 1.0),
            fontsize=5.8, title_fontsize=5.8, handlelength=1.35, handletextpad=0.3,
            labelspacing=0.15, borderaxespad=0.2, alignment="center"))
    entropy = Line2D([], [], color="0.35", ls="--", lw=1.0, label="Empirical\n$n$-gram\nentropy")
    axis_b.legend(handles=[entropy], loc="lower left", fontsize=5.7, handlelength=1.4)

    for spread, color, ls, name in ((z["ar_loss_spread_mlp"], MLP, "-", "MLP"),
                                    (z["ar_loss_spread_no_mlp"], NO_MLP, "--", "no MLP")):
        axis_c.plot(z["training_tokens"], spread, color=color, ls=ls, lw=1.25, label=name)
    axis_c.set_xscale("log")
    axis_c.xaxis.set_major_locator(FixedLocator(10.0 ** np.arange(5, 9)))
    axis_c.xaxis.set_minor_locator(FixedLocator(TOKEN_MINOR_TICKS))
    axis_c.yaxis.set_major_locator(FixedLocator([0.0, 1.0, 2.0]))
    axis_c.set_xlabel("training tokens", labelpad=1.0)
    axis_c.set_ylabel("Spread of AR loss\namong topologies", fontsize=5.7, labelpad=1.0)
    axis_c.legend(loc="upper right", fontsize=6.0, handlelength=1.45, labelspacing=0.18)

    # Enlarge legend text only: spacing and handles keep the size the legends were built at.
    for legend in figure.findobj(Legend):
        for text in (*legend.get_texts(), legend.get_title()):
            text.set_fontsize(7.7)
    for axis, letter in zip((axis_a, axis_b, axis_c), "abc"):
        axis.text(-0.16, 1.015, f"({letter})", transform=axis.transAxes, ha="left", va="bottom",
                  fontsize=7.5)
    return figure


def main(data_dir=None, out_dir=None) -> Path:
    data_dir = ROOT / "data" / "plot_data" if data_dir is None else Path(data_dir)
    out_dir = ROOT / "plots" / "reproduced" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_dir / "fig5.npz", allow_pickle=False) as data:
        z = dict(data)
    path = out_dir / "fig5.pdf"
    with plt.style.context(STYLE, after_reset=True):
        figure = draw(z)
        figure.savefig(path, pad_inches=0, metadata={"CreationDate": None})
    plt.close(figure)
    return path


if __name__ == "__main__":
    print(main())
