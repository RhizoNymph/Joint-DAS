#!/usr/bin/env bash
cd $HOME/Code/ai/learning-causal-representations
mkdir -p experiments/results/night3/gates_toy experiments/logs experiments/toy_ckpts
# Wait for the LM run on this node to finish before touching the GPU.
while pgrep -f 'run_phase_b[.]py' > /dev/null; do sleep 60; done
run() { # task layer lg seed
  local task=$1 layer=$2 lg=$3 seed=$4
  local short=$([ "$task" = "hierarchical_equality" ] && echo hier || echo bool)
  local out="experiments/results/night3/gates_toy/${short}_l${layer}_lg${lg}_s${seed}.json"
  [ -f "$out" ] && return
  ~/.local/bin/uv run python experiments/run_phase_a.py --task $task --site-layer $layer \
    --method joint --gates --lambda-gate $lg --seed $seed --steps 1500 --k-max 4 \
    --device cuda --out "$out" \
    > "experiments/logs/gates_${short}_l${layer}_lg${lg}_s${seed}.log" 2>&1 \
    || echo "FAILED ${short}_l${layer}_lg${lg}_s${seed}" >> experiments/logs/gates_toy_failures.log
}
for lg in 0.03 0.1 0.3; do for s in 0 1 2; do run boolean_comp 1 $lg $s; done; done
echo TOY_DONE_node2 >> experiments/logs/gates_toy_status.log
