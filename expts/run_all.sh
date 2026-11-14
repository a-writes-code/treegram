#!/usr/bin/env bash
# Train the fixed-topology and the dynamic tree-gram on every corpus in data/train_data/,
# then print the first and last validation loss of each run.
#
#   bash expts/run_all.sh
#
# Settings come from the environment, as for expts/fixed_tree.sh and expts/dynamic_tree.sh
# (e.g. DEVICE=cpu bash expts/run_all.sh). OUT_DIR, if set, gets one subdirectory per corpus.
set -euo pipefail

if [ $# -gt 0 ]; then
    echo "run_all.sh takes no arguments; set options in the environment (e.g. DEVICE=cpu SEED=1)" >&2
    exit 2
fi

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python3}
BASE=${OUT_DIR:-$ROOT/runs}
export PYTHON

corpora=()
for corpus in tinystories wikitext103 fineweb_edu; do
    dir="$ROOT/data/train_data/$corpus"
    if [ ! -f "$dir/train.npy" ]; then
        echo "== skipping $corpus: data/train_data/$corpus/train.npy not found"
    elif [ ! -f "$dir/val.npy" ]; then
        echo "data/train_data/$corpus/train.npy exists but val.npy does not" >&2
        exit 1
    else
        corpora+=("$corpus")
    fi
done
if [ ${#corpora[@]} -eq 0 ]; then
    echo "no corpus found under data/train_data/" >&2
    exit 1
fi

for corpus in "${corpora[@]}"; do
    OUT_DIR="$BASE/$corpus" bash "$ROOT/expts/fixed_tree.sh" "$corpus"
    OUT_DIR="$BASE/$corpus" bash "$ROOT/expts/dynamic_tree.sh" "$corpus"
done

"$PYTHON" - "$BASE" "${corpora[@]}" <<'PY'
import glob, os, pickle, sys

base, corpora = sys.argv[1], sys.argv[2:]
for corpus in corpora:
    for path in sorted(glob.glob(os.path.join(base, corpus, "history", "*.pkl"))):
        with open(path, "rb") as fp:
            hist = pickle.load(fp)
        print(f"{corpus}/{os.path.basename(path)}: {hist['params'][0]:,} parameters, val loss "
              f"{hist['val_loss'][0]:.3f} (step 0) -> {hist['val_loss'][-1]:.3f} (step {hist['steps'][-1]}) nats/token")
PY
