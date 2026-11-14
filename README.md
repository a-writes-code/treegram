# Tree-grams

Code and data for the paper *What's in a Tree? Hierarchical n-gram Models as Controlled Approximators of Language*.

A tree-gram is an n-gram language model shaped like a tree. It reads the last n − 1 tokens, merges neighbouring tokens into summaries, merges those summaries into larger ones, and combines the result into a prediction for the next token. Two numbers set its size: the context length n and the width d of each summary. A fixed tree-gram always merges in the same pattern, its topology. A dynamic tree-gram has a router that picks among candidate merges at every level of the tree.

This repository has the model and training code, 2M-token slices of TinyStories and WikiText-103 to train on, and the six main-text figures of the paper with the values and code that redraw them.

## Contents

```
src/
  models.py           FixedTree and DynamicTree
  data.py             non-overlapping n-token windows read from a memory-mapped .npy file
  trainer.py          training loop, validation, learning-rate schedules
  train.py            command-line entry point; trains one model
  figures/            fig1.py ... fig6.py, one script per main-text figure
expts/
  fixed_tree.sh       trains a fixed-topology tree-gram
  dynamic_tree.sh     trains a dynamic tree-gram
  run_all.sh          trains both on every corpus in data/train_data/
data/
  train_data/         tinystories/ and wikitext103/, each with train.npy, val.npy and tokenizer.json
  plot_data/          the values behind the six figures
plots/                the six main-text figures as they appear in the paper
requirements.txt
LICENSE               MIT, for the code
```

## Setup

You need Python 3.10 or newer; on Intel Macs, use 3.10 to 3.12, because PyTorch stops at version 2.2 there. The `python3` that comes with macOS is 3.9, so check `python3 --version` first.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs PyTorch, NumPy and Matplotlib. Training runs on a CPU and uses a CUDA GPU when one is visible. Redrawing the figures needs neither a GPU nor PyTorch. Run the commands below from the repository root.

## Training

`fixed_tree.sh` and `dynamic_tree.sh` each train one model on one corpus. The corpus name is the first argument and defaults to `tinystories`. `run_all.sh` trains both models on both corpora and then prints a summary.

```bash
bash expts/fixed_tree.sh                 # fixed topology tau_0 on TinyStories
bash expts/fixed_tree.sh wikitext103     # the same model on WikiText-103
bash expts/dynamic_tree.sh               # dynamic tree-gram, top-1 routing
bash expts/run_all.sh                    # both models on both corpora
```

All of them use one hyper-parameter setting, small enough to train on a laptop CPU:

| setting | value |
|---|---|
| context length n | 16: 15 context tokens, one prediction after every prefix |
| width d | 64 |
| MLP width factor | 4 |
| optimizer | AdamW, weight decay 0.1 |
| learning rate | warmed up from 1e-5 to 3e-3 over the first 1% of steps, then cosine decay to 3e-5 |
| batch | 128 windows of 16 tokens |
| length | one pass over the 2M training tokens, 976 steps |
| validation | every 75 steps, the last step included, on all 250,000 validation tokens |
| parameters | 903,296 (fixed), 906,496 (dynamic) |

The dynamic tree-gram uses top-1 routing. Load-balancing biases on its router are updated at a rate of 1e-3 for the first 5% of training (1% with `M=2`) and then held fixed, and the balancing targets carry noise of scale 0.1.

Every setting can be changed from the environment, and arguments after the corpus name are passed on to `src/train.py`:

```bash
TOPO=-1 bash expts/fixed_tree.sh                      # topology tau_-1 instead of tau_0
D=128 LR=1e-3 bash expts/fixed_tree.sh wikitext103
M=2 bash expts/dynamic_tree.sh                        # mix the top two candidates at each level, as in Figure 6
DEVICE=cpu SEED=1 bash expts/dynamic_tree.sh
```

The variables are `N`, `D`, `DFF`, `LR`, `FRAC_LR`, `WD`, `WARMUP`, `BS`, `VAL_BS`, `EPOCHS`, `UPDATE_FREQ`, `SEED`, `DEVICE` (`auto`, `cpu` or `cuda`), `OUT_DIR`, `PYTHON` and `OMP_NUM_THREADS` (CPU threads, 4 by default), plus `TOPO` for the fixed tree-gram and `M`, `ROUTER_BIAS_LR`, `CONST_FRAC` and `NOISY_TGT` for the dynamic one. `OUT_DIR` is the output directory of one run (`runs/<corpus>` by default); for `run_all.sh` it is the parent directory, with one subdirectory per corpus.

Each run writes a history and a checkpoint. The file name records the model, n, d, the topology or m, the MLP factor, the peak learning rate and the seed. Two runs that differ only in another setting, such as `WD` or `BS`, get the same name and the second overwrites the first, so give them different `OUT_DIR`s.

```
runs/<corpus>/history/<run>.pkl    dict: train_loss, val_loss, steps, tokens, params, flops, config
runs/<corpus>/models/<run>.pt      dict: model (state_dict), optimizer, scheduler, step, tokens, config
```

Losses are in nats per token, averaged over all positions of a window; `train_loss` is the loss of the batch at that step. The validation loss starts near ln 8192 ≈ 9.0. On a laptop CPU (Apple M4 Pro, 4 threads, PyTorch 2.12), each run took 3 to 4 minutes, `run_all.sh` took 14 minutes, and the runs with seed 0 ended at these validation losses:

| corpus | fixed tree-gram | dynamic tree-gram |
|---|---|---|
| TinyStories | 3.33 | 3.50 |
| WikiText-103 | 5.70 | 5.97 |

Other machines and PyTorch versions give slightly different numbers. With 2M tokens and 0.9M parameters these runs show that the code works; their losses are not comparable to the paper's.

### The paper's configuration

Without hyper-parameter options, `src/train.py` uses the paper's main configuration: n = 32, d = 1024, batch 1024, peak learning rate 1e-3 with the same warmup and cosine schedule, weight decay 0.1. The paper used these settings for topologies τ0, τ1 and τ2, with `--init_lr 1e-6 --max_lr 3e-4` for τ−1 and τ−2, and `--wd 0.01` for τa and τb. At n = 32 the seven topologies of Figure 5 are `--topo 0, 1, 2, 24974, 24975, -2, -1` (τ0, τ1, τ2, τa, τb, τ−2, τ−1); the index of a topology depends on n.

The paper trained these models on the full corpora (529M TinyStories tokens, 132M WikiText-103 tokens), splitting each batch of 1024 windows across four A100 GPUs. This repository trains on one device, without gradient accumulation, so at the defaults a single forward pass holds all 1024 windows: about 68 GiB of activations for the fixed tree-gram and 88 GiB for the dynamic one. A smaller `--bs` fits in less memory but changes the optimization, so the tuned learning rates no longer apply. To train on a full corpus, point the script to its token files:

```bash
python src/train.py --model fixed --corpus tinystories --train_path /path/to/train.npy --val_path /path/to/val.npy
```

`python src/train.py --help` lists every option. `--init_from <checkpoint>` starts from saved weights with a fresh optimizer and learning-rate schedule.

## Data

`data/train_data/` holds the start of two tokenized corpora, as 1-D `.npy` arrays of uint16 token ids, with the tokenizer that produced each:

| corpus | tokenizer | train.npy | val.npy |
|---|---|---|---|
| TinyStories | BPE with 8192 tokens, from a public tokenization of TinyStories | first 2,000,000 tokens of the training split | first 250,000 tokens of the validation split |
| WikiText-103 | BPE with 8192 tokens, fitted to WikiText-103 for this paper | first 2,000,000 tokens of the training split | first 250,000 tokens of the validation split |

Token 1 ends each story in TinyStories and each non-blank line in WikiText-103. Token 0 is the unknown token. The six files take 9.8 MB.

The TinyStories ids are the start of the BPE-8192 tokenization published, with its tokenizer, at [huggingface.co/datasets/fracagnetta/tinystories.BPE8192](https://huggingface.co/datasets/fracagnetta/tinystories.BPE8192) under the MIT license; the paper trained on all of it. TinyStories itself (Eldan and Li, 2023, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)) is under [CDLA-Sharing-1.0](https://spdx.org/licenses/CDLA-Sharing-1.0.html), and WikiText-103 (Merity et al., 2016, [arXiv:1609.07843](https://arxiv.org/abs/1609.07843)) is under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The files here are tokenized, truncated copies and keep those licenses.

To read the tokens as text, install the `tokenizers` package (`pip install tokenizers`) and decode them:

```python
import numpy as np
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/train_data/tinystories/tokenizer.json")
print(tokenizer.decode(np.load("data/train_data/tinystories/train.npy")[:40].tolist()))
```

Both tokenizers split text at whitespace and punctuation and have no decoder, so the decoded text has spaces around punctuation and inside some words: "Once upon a time there was a little boy named Ben . Ben loved to explore ...".

The paper also uses FineWeb-Edu, tokenized with the GPT-2 tokenizer (50,257 tokens), which is not included here. To train on it, save its token ids as uint16 arrays in `data/train_data/fineweb_edu/train.npy` and `val.npy` and pass `fineweb_edu` as the corpus name. `src/train.py` then sets the vocabulary size to 50,257.

## Figures

`plots/` has the six main-text figures of the paper as PDFs. They are numbered as in the revised paper; in the submitted version, Figures 3 and 4 were one figure, Figure 5 was Figure 4, and the routing results of Figure 6 were part of Figure 5. To redraw them from `data/plot_data/`:

```bash
python -m src.figures          # all six, written to plots/reproduced/
python -m src.figures 3 5      # Figures 3 and 5 only
```

This takes a few seconds and uses only NumPy and Matplotlib.

| figure | what it shows | data |
|---|---|---|
| 1 | TinyStories. (a) The architecture; (b) loss against context length n for the tree-gram, four reference language models and count-based, spectral, neural and tensor-network baselines; (c) loss at each position for widths d = 64 to 1024 | `fig1.npz`, `fig1_architecture.png` |
| 2 | TinyStories. Loss at the last position minus H_n, the loss of a larger 12-layer transformer with the same n − 1 tokens of context, for the tree-gram and for parameter-matched transformers with 1 to 6, 10 and 12 layers, against n and against d, with power-law fits | `fig2.npz` |
| 3 | Hyper-parameter transfer on TinyStories and WikiText-103. A width-64 transformer (the proxy) is tuned on a synthetic corpus or on real text, and its best learning rate is used to train a wider transformer (the target) on real text, with 2 and 4 layers. Shown against the complexity of the proxy's corpus: the target's excess loss at that rate and the proxy's best rate; last column: the target's learning-rate sweeps | `fig3.npz` |
| 4 | Pre-training and fine-tuning: a transformer pre-trained on each source corpus and fine-tuned on each target corpus (TinyStories with 4 layers, WikiText-103 with 2 and 4 layers). Each cell is the seed-median final validation loss divided by the target's loss floor, minus one | `fig4.npz` |
| 5 | TinyStories. Seven fixed topologies: loss at each position, with and without MLP blocks, and the spread among the topologies during training | `fig5.npz` |
| 6 | TinyStories. The dynamic tree-gram with top-2 routing: ranked probabilities of the routed paths with power-law fits, and the loss when low-probability candidates are pruned | `fig6.npz` |

In Figures 3 and 4 the synthetic corpora are samples from tree-grams trained on the real corpus (n = 3 to 32 at d = 1024, and d = 64 to 512 at n = 32) and from a bigram model. In Figure 4 the loss floor of a synthetic target is the entropy rate of the model that generated it, and that of real text is the loss of a 12-layer reference transformer. A few cells of Figure 4 come from fine-tuning re-runs at other peak learning rates, taking the rate with the best seed-median final loss among those run with three seeds. The figures call the width d the rank (Figure 2) or d_tree (Figures 3 and 4).

Each `src/figures/figN.py` reads `data/plot_data/figN.npz` and writes `plots/reproduced/figN.pdf`. The `.npz` files hold plain arrays, so `np.load(path)` opens them. They contain the final aggregated values (seed means, standard deviations, medians and fitted parameters), not the training runs behind them. The four reference language models in Figure 1 are results published by Cagnetta et al. (2026, [arXiv:2602.07488](https://arxiv.org/abs/2602.07488)). The other baselines in Figure 1, the transformers in Figures 2 to 4, the sampling of the synthetic corpora and the analyses behind Figures 5 and 6 used code that is not part of this repository; only their results are included.

With Matplotlib 3.10 (`pip install "matplotlib==3.10.*"`), Figures 2 to 6 come out pixel-identical to the PDFs in `plots/`; other versions render text slightly differently. Compare the rendered pages, not the files, since the PDFs carry different metadata. Figure 1 is the exception. Its panel (a) is a drawing with no data behind it, so it is included as an image. In the paper, panels (b) and (c) were laid out by hand; the redrawn panels show the same data with slightly narrower text and small differences in spacing.

## License

The code is under the MIT License (`LICENSE`). It does not cover `data/train_data/`, whose files keep the licenses given under Data.
