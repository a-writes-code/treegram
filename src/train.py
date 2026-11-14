import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import TokenWindows
from src.models import DynamicTree, FixedTree
from src.trainer import router_bias_schedule, train, tree_schedule

CORPORA = {"tinystories": 8192, "wikitext103": 8192, "fineweb_edu": 50257}

DEFAULTS = {
    "fixed": dict(n=32, d=1024, max_lr=1e-3, frac_lr=0.01, wd=0.1, bs=1024, val_bs=2048),
    "dynamic": dict(n=32, d=1024, max_lr=1e-3, frac_lr=0.01, wd=0.1, bs=1024, val_bs=512),
}


def parse_args():
    p = argparse.ArgumentParser(description="Train a tree-gram with a fixed or a dynamic (routed) topology.")
    p.add_argument("--model", choices=sorted(DEFAULTS), required=True)
    p.add_argument("--corpus", choices=sorted(CORPORA), default="tinystories",
                   help="reads data/train_data/<corpus>/{train,val}.npy unless --train_path/--val_path are given")
    p.add_argument("--n", type=int, help="window length: n-1 context tokens, one loss per prefix (default 32)")
    p.add_argument("--d", type=int, help="width (default 1024)")
    p.add_argument("--topo", type=int, default=0, help="fixed topology index; -1 is the last one")
    p.add_argument("--dff", type=float, default=4.0, help="MLP width factor; 0 removes the MLPs")
    p.add_argument("--mu", type=float, default=1.0, help="residual stream strength")
    p.add_argument("--m", type=int, default=1, help="top-m routing of the dynamic tree-gram")
    p.add_argument("--router_bias_lr", type=float, default=1e-3)
    p.add_argument("--router_bias_frac_lr", type=float, default=0.0)
    p.add_argument("--const_frac", type=float,
                   help="fraction of training before the router bias rate drops (default 0.05, or 0.01 with --m 2)")
    p.add_argument("--noisy_tgt", type=float, default=0.1, help="noise on the router load-balancing targets")
    p.add_argument("--train_path", help="token file (.npy); default data/train_data/<corpus>/train.npy")
    p.add_argument("--val_path", help="token file (.npy); default data/train_data/<corpus>/val.npy")
    p.add_argument("--vocab_size", type=int, help="default: 8192 for tinystories and wikitext103, 50257 for fineweb_edu")
    p.add_argument("--train_tokens", type=int, default=-1, help="-1 uses the whole file")
    p.add_argument("--val_tokens", type=int, default=1_050_000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, help="windows per optimizer step (default 1024)")
    p.add_argument("--val_bs", type=int, help="windows per validation batch (default 2048 fixed, 512 dynamic)")
    p.add_argument("--init_lr", type=float, default=1e-5)
    p.add_argument("--max_lr", type=float, help="peak learning rate (default 1e-3)")
    p.add_argument("--frac_lr", type=float, help="final learning rate as a fraction of the peak (default 0.01)")
    p.add_argument("--warmup_frac", type=float, default=0.01)
    p.add_argument("--wd", type=float, help="AdamW weight decay (default 0.1)")
    p.add_argument("--update_freq", type=int, help="validate every this many steps (default: 100 times per run)")
    p.add_argument("--init_from", default="", help="checkpoint whose weights initialise the model (fine-tuning)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_dir", help="default runs/<corpus>")
    args = p.parse_args()
    for key, value in DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.const_frac is None:
        args.const_frac = 0.01 if args.m == 2 else 0.05
    corpus_dir = ROOT / "data" / "train_data" / args.corpus
    args.train_path = args.train_path or str(corpus_dir / "train.npy")
    args.val_path = args.val_path or str(corpus_dir / "val.npy")
    args.vocab_size = args.vocab_size or CORPORA[args.corpus]
    args.out_dir = args.out_dir or str(ROOT / "runs" / args.corpus)
    for path in (args.train_path, args.val_path):
        if not Path(path).is_file():
            hint = " (FineWeb-Edu is not bundled; see the README)" if args.corpus == "fineweb_edu" else ""
            p.error(f"token file not found: {path}{hint}")
    return args


def run_name(args):
    extra = f"_t{args.topo}" if args.model == "fixed" else f"_m{args.m}"
    return f"{args.model}_n{args.n}_d{args.d}{extra}_dff{args.dff:g}_lr{args.max_lr:g}_s{args.seed}"


def main():
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device.replace("auto", "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_set = TokenWindows(args.train_path, args.n, args.train_tokens)
    val_set = TokenWindows(args.val_path, args.n, args.val_tokens)
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.num_workers,
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.val_bs, shuffle=False, num_workers=0,
                            pin_memory=False, drop_last=True)
    if len(train_loader) == 0 or len(val_loader) == 0:
        sys.exit("not enough tokens for one batch: lower --bs/--val_bs or use more tokens")
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_frac * total_steps)

    if args.model == "fixed":
        model = FixedTree(args.vocab_size, args.n, args.d, topo=args.topo, dff=args.dff, mu=args.mu)
    else:
        model = DynamicTree(args.vocab_size, args.n, args.d, m=args.m, dff=args.dff, mu=args.mu,
                            noisy_tgt=args.noisy_tgt, router_bias_lr=args.router_bias_lr)
    model = model.to(device)
    if args.init_from:
        topology = model.fixed_topo_idxs.clone() if args.model == "fixed" else None
        model.load_state_dict(torch.load(args.init_from, map_location=device, weights_only=False)["model"])
        if topology is not None and not torch.equal(topology, model.fixed_topo_idxs):
            sys.exit(f"{args.init_from} was trained with a different topology than --topo {args.topo}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.wd)
    scheduler = tree_schedule(optimizer, args.init_lr, args.max_lr, args.frac_lr, warmup_steps, total_steps)

    after_step = None
    if args.model == "dynamic":
        bias_lr = router_bias_schedule(args.router_bias_lr, args.router_bias_frac_lr, args.const_frac, total_steps)

        def after_step(step):
            model.router_bias_lr = bias_lr(step)
            model.update_router_biases()

    name = run_name(args)
    out = Path(args.out_dir)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_params:,} parameters, {total_steps} steps of {args.bs} windows of {args.n} tokens, "
          f"corpus {args.corpus}, on {device}")
    train(model, optimizer, scheduler, train_loader, val_loader, device, args.epochs, args.update_freq,
          out / "history" / f"{name}.pkl", out / "models" / f"{name}.pt", vars(args), after_step)
    print(f"history: {out / 'history' / f'{name}.pkl'}\ncheckpoint: {out / 'models' / f'{name}.pt'}")


if __name__ == "__main__":
    main()
