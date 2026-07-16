#!/usr/bin/env bash
# Rerun the das_true/das_wrong Phase A configs (fixed wiring) for one task.
# Usage: bash scripts/rerun_baselines.sh <task> [steps] [parallel]
set -euo pipefail

TASK="${1:?task required}"
STEPS="${2:-4000}"
PAR="${3:-3}"
UV=~/.local/bin/uv
OUT_DIR="experiments/results/phase_a"
LOG_DIR="experiments/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

CMDS="$LOG_DIR/rerun_cmds_${TASK}.txt"
: > "$CMDS"
add() { echo "$UV run python experiments/run_phase_a.py --task $TASK --device cuda --steps $STEPS $* " >> "$CMDS"; }

for seed in 0 1 2; do
  for layer in 0 1 2; do
    add --method das_true --site-layer "$layer" --seed "$seed" --out "$OUT_DIR/${TASK}_das_true_l${layer}_s${seed}.json"
  done
  add --method das_wrong --site-layer 1 --seed "$seed" --out "$OUT_DIR/${TASK}_das_wrong_l1_s${seed}.json"
done

echo "$(wc -l < "$CMDS") reruns, parallel=$PAR"
xargs -a "$CMDS" -d '\n' -P "$PAR" -I {} bash -c '{} >> '"$LOG_DIR"'/rerun_runs.log 2>&1 || echo "FAILED: {}" >> '"$LOG_DIR"'/rerun_failures.log'
echo "RERUN_DONE task=$TASK"
[ -f "$LOG_DIR/rerun_failures.log" ] && { echo FAILURES; cat "$LOG_DIR/rerun_failures.log"; } || true
