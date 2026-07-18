#!/usr/bin/env bash
# Night-3 LM gate sweep: capped recipe + hard-concrete gates on Qwen2.5-1.5B l17.
# 7 runs over 3 nodes, sequential per node, nohup'd. Usage: bash scripts/launch_gates_lm.sh
set -u
REMOTE_DIR="Code/ai/learning-causal-representations"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
COMMON="--model $MODEL --layer 17 --template-id 3 --device cuda --steps 800 \
  --batch-size 32 --k-max 4 --max-width 128 --init-width 32 \
  --sparse-mode per_dim --lambda-sparse 0.02 --gates --no-refit --local-files-only"

# node -> list of "lambda_gate seed" runs (control lg0 on node0)
declare -A PLAN=(
  [node0]="0.01 0|0 0"
  [node1]="0.05 0|0.05 1"
  [node2]="0.2 0|0.2 1|0.01 1"
)

for node in node0 node1 node2; do
  script="experiments/logs/gates_lm_${node}.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "export HF_HOME=\$HOME/hf-cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1"
    echo "cd \$HOME/$REMOTE_DIR"
    echo "mkdir -p experiments/results/night3/gates_lm experiments/ckpts experiments/logs"
    IFS='|' read -ra runs <<< "${PLAN[$node]}"
    for run in "${runs[@]}"; do
      lg=${run% *}; seed=${run#* }
      tag="pt_gates_l17_lg${lg}_s${seed}"
      echo "~/.local/bin/uv run python experiments/run_phase_b.py --method joint $COMMON \\"
      echo "  --lambda-gate $lg --seed $seed \\"
      echo "  --save-ckpt experiments/ckpts/${tag}.pt \\"
      echo "  --out experiments/results/night3/gates_lm/${tag}.json \\"
      echo "  >> experiments/logs/${tag}.log 2>&1 || echo FAILED_${tag} >> experiments/logs/gates_lm_failures.log"
    done
    echo "echo NODE_DONE_${node} >> experiments/logs/gates_lm_status.log"
  } > "$script"
  ssh -n "$node" "mkdir -p $REMOTE_DIR/experiments/logs $REMOTE_DIR/experiments/results/night3/gates_lm"
  scp -q "$script" "$node:$REMOTE_DIR/$script"
  ssh -n "$node" "(cd $REMOTE_DIR && nohup bash $script > /dev/null 2>&1 &)"
  echo "launched on $node: ${PLAN[$node]}"
done
