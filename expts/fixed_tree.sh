#!/usr/bin/env bash
# Train a tree-gram with a fixed merge topology at one fixed hyper-parameter setting.
#
#   bash expts/fixed_tree.sh                  # TinyStories slice, topology tau_0
#   bash expts/fixed_tree.sh wikitext103      # WikiText-103 slice
#   TOPO=-1 bash expts/fixed_tree.sh          # topology tau_-1 (the last one)
#
# Every setting below can be overridden from the environment (e.g. D=128 LR=1e-3).
# Arguments after the corpus name are passed on to src/train.py.
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS=tinystories
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then CORPUS=$1; shift; fi

PYTHON=${PYTHON:-python3}
N=${N:-16}
D=${D:-64}
TOPO=${TOPO:-0}
DFF=${DFF:-4}
LR=${LR:-3e-3}
FRAC_LR=${FRAC_LR:-0.01}
WD=${WD:-0.1}
WARMUP=${WARMUP:-0.01}
BS=${BS:-128}
VAL_BS=${VAL_BS:-625}
EPOCHS=${EPOCHS:-1}
UPDATE_FREQ=${UPDATE_FREQ:-75}
SEED=${SEED:-0}
DEVICE=${DEVICE:-auto}
OUT_DIR=${OUT_DIR:-$ROOT/runs/$CORPUS}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

"$PYTHON" "$ROOT/src/train.py" --model fixed --corpus "$CORPUS" --topo "$TOPO" \
    --n "$N" --d "$D" --dff "$DFF" --max_lr "$LR" --frac_lr "$FRAC_LR" --wd "$WD" --warmup_frac "$WARMUP" \
    --bs "$BS" --val_bs "$VAL_BS" --epochs "$EPOCHS" --update_freq "$UPDATE_FREQ" --seed "$SEED" \
    --device "$DEVICE" --out_dir "$OUT_DIR" "$@"
