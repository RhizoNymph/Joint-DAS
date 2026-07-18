#!/usr/bin/env bash
# Night-3 v2 (gate-lr fix): LM runs then toy runs, sequential on this GPU.
cd $HOME/Code/ai/learning-causal-representations
mkdir -p experiments/results/night3/gates_lm_v3 experiments/results/night3/gates_toy_v3 \
  experiments/ckpts experiments/logs experiments/toy_ckpts
export HF_HOME=$HOME/hf-cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
LMCOMMON="--method joint --model Qwen/Qwen2.5-1.5B-Instruct --layer 17 --template-id 3 --device cuda --steps 800 --batch-size 32 --k-max 4 --max-width 128 --init-width 32 --sparse-mode per_dim --lambda-sparse 0.02 --gates --gate-lr 0.05 --gate-warmup 200 --gate-lambda-ramp 200 --gate-clamp 3.0 --no-refit --local-files-only"
lm() { # lg seed
  local tag="pt_gates3_l17_lg${1}_s${2}"
  [ -f "experiments/results/night3/gates_lm_v3/${tag}.json" ] && return
  ~/.local/bin/uv run python experiments/run_phase_b.py $LMCOMMON --lambda-gate $1 --seed $2 \
    --save-ckpt experiments/ckpts/${tag}.pt \
    --out experiments/results/night3/gates_lm_v3/${tag}.json \
    > experiments/logs/${tag}.log 2>&1 || echo "FAILED ${tag}" >> experiments/logs/gates_v3_failures.log
}
toy() { # task layer lg seed
  local short=$([ "$1" = "hierarchical_equality" ] && echo hier || echo bool)
  local out="experiments/results/night3/gates_toy_v3/${short}_l${2}_lg${3}_s${4}.json"
  [ -f "$out" ] && return
  ~/.local/bin/uv run python experiments/run_phase_a.py --task $1 --site-layer $2 \
    --method joint --gates --gate-lr 0.05 --gate-warmup 300 --gate-lambda-ramp 300 --gate-clamp 3.0 --lambda-gate $3 --seed $4 --steps 1500 --k-max 4 \
    --device cuda --out "$out" > "experiments/logs/gates3_${short}_l${2}_lg${3}_s${4}.log" 2>&1 \
    || echo "FAILED toy ${short}_l${2}_lg${3}_s${4}" >> experiments/logs/gates_v3_failures.log
}
lm 0.05 0
lm 0.05 1
for lg in 0 0.01 0.03 0.1 0.3; do for s in 0 1 2; do toy hierarchical_equality 2 $lg $s; done; done
echo V2_DONE_node1 >> experiments/logs/gates_v3_status.log
