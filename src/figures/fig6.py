"""Figure 6: router paths of the dynamic tree with m = 2, over two training seeds.

Left: joint path probabilities in descending order, averaged rank by rank over the seeds
(path_prob_mean) with the seed range (path_prob_min, path_prob_max), power-law fits to the
head and the tail (head_fit_*, tail_fit_*) and a power law with a stretched-exponential
cutoff normalised over all ranks (cutoff_fit_*). Right: validation loss when every router
level keeps only a fraction of its candidates for each token, seed mean and range
(prune_kept_fraction, prune_loss_mean/min/max), with the fixed tree tau_0 as reference
(prune_ref_loss_tau0). All arrays are read from fig6.npz.
"""
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
logging.getLogger("fontTools").setLevel(logging.ERROR)
STYLE = {
    "font.family": "DejaVu Serif", "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "axes.linewidth": .65, "axes.spines.top": False, "axes.spines.right": False,
    "grid.alpha": .2, "grid.linewidth": .45,
    "xtick.major.size": 2.5, "xtick.major.width": .7, "ytick.major.size": 2.5, "ytick.major.width": .7,
}
LEGEND = dict(fontsize=8.2, edgecolor=".65", framealpha=.95, labelspacing=.35)
FIT_LINE = dict(c=".23", lw=1.05)
LOSS_TICKS = [1.94, 1.95, 1.96, 1.97, 1.98]


def main(data_dir=None, out_dir=None) -> Path:
    data_dir = Path(data_dir) if data_dir else ROOT / "data" / "plot_data"
    out_dir = Path(out_dir) if out_dir else ROOT / "plots" / "reproduced"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fig6.pdf"
    with np.load(data_dir / "fig6.npz", allow_pickle=False) as d, plt.style.context(STYLE, after_reset=True):
        fig = plt.figure(figsize=(6.6, 2.6))
        left = fig.add_axes([.105, .24, .35, .73])
        mean = d["path_prob_mean"]
        rank = np.arange(1, mean.size + 1)
        left.fill_between(rank, d["path_prob_min"], d["path_prob_max"], color="#d9750d", alpha=.23, lw=0)
        handles = left.loglog(rank, mean, color="#d9750d", lw=1.4, label="$m = 2$")
        for part, ls in (("head", ":"), ("tail", "--")):
            first, last = d[f"{part}_fit_ranks"]
            r, slope = np.arange(first, last + 1), d[f"{part}_fit_slope"]
            handles += left.loglog(r, np.exp(d[f"{part}_fit_log_intercept"]) * r ** slope, ls=ls,
                                   label=f"{part} slope : {slope:.2f}", **FIT_LINE)
        alpha, r_c, beta, log_norm = (d[f"cutoff_fit_{k}"] for k in ("alpha", "rank_scale", "beta", "log_norm"))
        handles += left.loglog(rank, np.exp(-alpha * np.log(rank) - (rank / r_c) ** beta - log_norm), ls="-.",
                               label="cutoff fit", **FIT_LINE)
        left.set(xlim=(1, 45000), ylim=(1e-13, .5), xlabel="path-probability rank", ylabel="path probability")
        left.set_xticks([1, 100, 10000])
        left.set_xticks([k * 10 ** e for e in range(5) for k in range(2, 10) if k * 10 ** e < 45000], minor=True)
        left.set_yticks([1e-1, 1e-4, 1e-7, 1e-10, 1e-13])
        left.set_yticks([], minor=True)
        left.tick_params(labelsize=8.5)
        left.xaxis.label.set_fontsize(10)
        left.yaxis.label.set_fontsize(10)
        left.grid(True)
        left.legend(handles=handles, loc="lower left", **LEGEND)

        right = fig.add_axes([.655, .24, .33, .73])
        kept = d["prune_kept_fraction"]
        right.fill_between(kept, d["prune_loss_min"], d["prune_loss_max"], color="#238b60", alpha=.23, lw=0)
        handles = right.plot(kept, d["prune_loss_mean"], color="#238b60", marker="o", ms=2.7, lw=1.4, label="$m = 2$")
        right.set_xscale("log")
        right.set_xlim(1.10, .0125)
        right.grid(True)
        right.set_xticks([1, .1, .02], ["1", "0.1", "0.02"])
        right.minorticks_off()
        right.tick_params(labelsize=8.5)
        handles.append(right.axhline(float(d["prune_ref_loss_tau0"]), c="black", ls=(0, (5, 3)), lw=.9,
                                     label=r"fixed tree $\tau_0$"))
        right.set_ylim(1.94, 1.98)
        right.set_yticks(LOSS_TICKS, [f"{t:.3f}" for t in LOSS_TICKS])
        right.set_xlabel("fraction kept per level", fontsize=10)
        right.set_ylabel("validation loss (nats)", fontsize=10)
        right.legend(handles=handles, loc="upper left", **LEGEND)
        fig.savefig(out, dpi=400, pad_inches=0, metadata={"CreationDate": None})
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
