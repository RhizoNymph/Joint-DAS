# Learning Causal Representations (Joint-DAS)

An extension of Distributed Alignment Search (DAS) that **learns the high-level
causal model jointly with the alignment**, instead of requiring a hand-proposed
causal model. A small discrete causal model (k binary variables with
straight-through discretization, tiny encoders/decoder) is trained together
with an orthogonal rotation and learned subspace boundaries, supervised by
interchange-intervention agreement with the frozen target network.
Multi-source multi-variable interventions provide the identifiability pressure
that separates real causal factorizations from degenerate ones.

- Method and rationale: [docs/DESIGN.md](docs/DESIGN.md)
- Results and analysis: [RESULTS.md](RESULTS.md)
- Codebase map: [docs/OVERVIEW.md](docs/OVERVIEW.md)

## Layout

- `src/jdas/` — core library: rotation + subspace layout, learned/fixed causal
  models, interchange interventions, joint/DAS trainers, evaluation (IIA,
  ground-truth recovery, effective-k).
- `src/jdas/tasks/` — toy tasks with known causal structure (hierarchical
  equality, boolean composition) and an LM task (price tagging).
- `src/jdas/models/` — toy MLPs and the HuggingFace intervention site.
- `src/jdas/cli/` — the unified `jdas` CLI (run/analyze/sweep/cluster) and
  `EnvConfig` loaded from `jdas.toml`.
- `experiments/` — thin run shims, screening/analysis/introspection modules,
  and declarative sweep specs under `experiments/sweeps/`.

## Quickstart

```bash
uv sync
uv run pytest tests -q

# Toy experiment (CPU works; CUDA faster)
uv run jdas run toy \
  --task hierarchical_equality --method joint --site-layer 1 \
  --seed 0 --device cpu --steps 4000 --out results.json

# LM experiment (needs a GPU + Qwen2.5-1.5B-Instruct in HF cache)
uv run jdas run lm \
  --model Qwen/Qwen2.5-1.5B-Instruct --layer 17 --method joint \
  --template-id 3 --device cuda --steps 2000 --out results.json
```

Methods: `joint` (ours), `das_true` (classic DAS with the ground-truth model),
`das_wrong` (output-copy strawman), `random_rotation` (frozen-Q control).
