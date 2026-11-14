#!/usr/bin/env bash
# Train a dynamic tree-gram: a router picks the merge topology for each input (top-m routing).
#
#   bash expts/dynamic_tree.sh                # TinyStories slice, top-1 routing
#   bash expts/dynamic_tree.sh wikitext103    # WikiText-103 slice
#   M=2 bash expts/dynamic_tree.sh            # mix the top two candidates at each level
#
# Every setting below can be overridden from the environment.
# Arguments after the corpus name are passed on to src/train.py.
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS=tinystories
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then CORPUS=$1; shift; fi

PYTHON=${PYTHON:-python3}
N=${N:-16}
D=${D:-64}
M=${M:-1}
DFF=${DFF:-4}
LR=${LR:-3e-3}
FRAC_LR=${FRAC_LR:-0.01}
WD=${WD:-0.1}
WARMUP=${WARMUP:-0.01}
ROUTER_BIAS_LR=${ROUTER_BIAS_LR:-1e-3}
NOISY_TGT=${NOISY_TGT:-0.1}
BS=${BS:-128}
VAL_BS=${VAL_BS:-625}
EPOCHS=${EPOCHS:-1}
UPDATE_FREQ=${UPDATE_FREQ:-75}
SEED=${SEED:-0}
DEVICE=${DEVICE:-auto}
OUT_DIR=${OUT_DIR:-$ROOT/runs/$CORPUS}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

ROUTING=(--m "$M" --router_bias_lr "$ROUTER_BIAS_LR" --noisy_tgt "$NOISY_TGT")
if [ -n "${CONST_FRAC:-}" ]; then ROUTING+=(--const_frac "$CONST_FRAC"); fi

"$PYTHON" "$ROOT/src/train.py" --model dynamic --corpus "$CORPUS" "${ROUTING[@]}" \
    --n "$N" --d "$D" --dff "$DFF" --max_lr "$LR" --frac_lr "$FRAC_LR" --wd "$WD" --warmup_frac "$WARMUP" \
    --bs "$BS" --val_bs "$VAL_BS" --epochs "$EPOCHS" --update_freq "$UPDATE_FREQ" --seed "$SEED" \
    --device "$DEVICE" --out_dir "$OUT_DIR" "$@"
