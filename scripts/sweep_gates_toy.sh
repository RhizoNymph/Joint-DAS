#!/usr/bin/env bash
# Night-3 toy gate sweep: does the L0 gate penalty prune k_max=4 -> 2?
# 45 runs: {hier l1, hier l2, bool l1} x lambda_gate {0,0.01,0.03,0.1,0.3} x seeds {0,1,2}
set -u
cd "$(dirname "$0")/.."
OUTDIR=experiments/results/night3/gates_toy
mkdir -p "$OUTDIR" experiments/logs

jobs_file=$(mktemp)
for cfg in "hierarchical_equality 1" "hierarchical_equality 2" "boolean_comp 1"; do
  task=${cfg% *}; layer=${cfg#* }
  for lg in 0 0.01 0.03 0.1 0.3; do
    for seed in 0 1 2; do
      short=$([ "$task" = "hierarchical_equality" ] && echo hier || echo bool)
      out="$OUTDIR/${short}_l${layer}_lg${lg}_s${seed}.json"
      [ -f "$out" ] && continue
      echo "uv run python experiments/run_phase_a.py --task $task --site-layer $layer \
        --method joint --gates --lambda-gate $lg --seed $seed --steps 1500 --k-max 4 \
        --device cpu --out $out > experiments/logs/gates_${short}_l${layer}_lg${lg}_s${seed}.log 2>&1" >> "$jobs_file"
    done
  done
done

echo "$(wc -l < "$jobs_file") jobs"
xargs -P 6 -I {} bash -c '{}' < "$jobs_file"
rm -f "$jobs_file"
echo "sweep done: $(ls "$OUTDIR" | wc -l) results"
