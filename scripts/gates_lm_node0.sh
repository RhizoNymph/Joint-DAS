#!/usr/bin/env bash
export HF_HOME=$HOME/hf-cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
cd $HOME/Code/ai/learning-causal-representations
mkdir -p experiments/results/night3/gates_lm experiments/ckpts experiments/logs
COMMON="--method joint --model Qwen/Qwen2.5-1.5B-Instruct --layer 17 --template-id 3 --device cuda --steps 800 --batch-size 32 --k-max 4 --max-width 128 --init-width 32 --sparse-mode per_dim --lambda-sparse 0.02 --gates --no-refit --local-files-only"
for run in "0.01 0" "0 0"; do
  lg=${run% *}; seed=${run#* }
  tag="pt_gates_l17_lg${lg}_s${seed}"
  ~/.local/bin/uv run python experiments/run_phase_b.py $COMMON \
    --lambda-gate $lg --seed $seed \
    --save-ckpt experiments/ckpts/${tag}.pt \
    --out experiments/results/night3/gates_lm/${tag}.json \
    >> experiments/logs/${tag}.log 2>&1 || echo "FAILED_${tag}" >> experiments/logs/gates_lm_failures.log
done
echo NODE_DONE_node0 >> experiments/logs/gates_lm_status.log
