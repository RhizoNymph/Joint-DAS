#!/usr/bin/env bash
# Run the Phase A grid for one task on this machine's GPU.
# Usage: bash scripts/launch_phase_a.sh <task> [steps] [parallel]
set -euo pipefail

TASK="${1:?task required}"
STEPS="${2:-4000}"
PAR="${3:-3}"
UV=~/.local/bin/uv
OUT_DIR="experiments/results/phase_a"
LOG_DIR="experiments/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Warm the toy-model checkpoint cache once to avoid concurrent training races.
for layer in 0 1 2; do
  $UV run python -c "
import torch
from jdas.tasks import HierarchicalEqualityTask, BooleanCompositionTask
from jdas.models.toy import load_or_train_toy_model
task = {'hierarchical_equality': HierarchicalEqualityTask, 'boolean_comp': BooleanCompositionTask}['$TASK']()
load_or_train_toy_model(task, $layer, torch.device('cuda'))
print('warm ok layer $layer')
"
done

CMDS="$LOG_DIR/cmds_${TASK}.txt"
: > "$CMDS"
add() { echo "$UV run python experiments/run_phase_a.py --task $TASK --device cuda --steps $STEPS $* " >> "$CMDS"; }

for seed in 0 1 2; do
  for layer in 0 1 2; do
    add --method joint    --site-layer "$layer" --seed "$seed" --out "$OUT_DIR/${TASK}_joint_l${layer}_s${seed}.json"
    add --method das_true --site-layer "$layer" --seed "$seed" --out "$OUT_DIR/${TASK}_das_true_l${layer}_s${seed}.json"
  done
  add --method das_wrong       --site-layer 1 --seed "$seed" --out "$OUT_DIR/${TASK}_das_wrong_l1_s${seed}.json"
  add --method random_rotation --site-layer 1 --seed "$seed" --out "$OUT_DIR/${TASK}_random_rotation_l1_s${seed}.json"
done

echo "$(wc -l < "$CMDS") runs, parallel=$PAR"
xargs -a "$CMDS" -d '\n' -P "$PAR" -I {} bash -c '{} >> '"$LOG_DIR"'/phase_a_runs.log 2>&1 || echo "FAILED: {}" >> '"$LOG_DIR"'/phase_a_failures.log'
echo "PHASE_A_GRID_DONE task=$TASK"
[ -f "$LOG_DIR/phase_a_failures.log" ] && { echo "FAILURES:"; cat "$LOG_DIR/phase_a_failures.log"; } || true
