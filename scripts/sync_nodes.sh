#!/usr/bin/env bash
# Sync this repo to GPU nodes and ensure the venv is up to date.
# Usage: scripts/sync_nodes.sh [node0 node1 node2]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="Code/ai/learning-causal-representations"
NODES=("${@:-node0 node1 node2}")
[ $# -eq 0 ] && NODES=(node0 node1 node2)

for node in "${NODES[@]}"; do
  echo "=== syncing $node ==="
  ssh "$node" "mkdir -p $REMOTE_DIR"
  rsync -az --delete \
    --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
    --exclude 'experiments/results' --exclude 'experiments/logs' \
    --exclude 'experiments/toy_ckpts' --exclude '*.egg-info' \
    "$REPO_DIR/" "$node:$REMOTE_DIR/"
  ssh "$node" "cd $REMOTE_DIR && ~/.local/bin/uv sync --quiet" &
done
wait
echo "all nodes synced"
