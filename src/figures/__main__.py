import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIGURES = [1, 2, 3, 4, 5, 6]


def main():
    p = argparse.ArgumentParser(description="Redraw the main-text figures from data/plot_data/.")
    p.add_argument("figures", nargs="*", type=int, metavar="N", help="figure numbers, 1 to 6 (default: all six)")
    p.add_argument("--data_dir", default=str(ROOT / "data" / "plot_data"))
    p.add_argument("--out_dir", default=str(ROOT / "plots" / "reproduced"))
    args = p.parse_args()
    unknown = sorted(set(args.figures) - set(FIGURES))
    if unknown:
        p.error(f"no figure {', '.join(map(str, unknown))}: choose from 1 to 6")
    for k in args.figures or FIGURES:
        module = importlib.import_module(f"src.figures.fig{k}")
        path = module.main(Path(args.data_dir), Path(args.out_dir))
        print(f"Figure {k}: {path}")


if __name__ == "__main__":
    main()
