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
- `experiments/` — analyzer modules, one-off diagnostics, committed result
  JSONs under `experiments/results/`, and declarative sweep specs under
  `experiments/sweeps/`.
- `jdas.toml` — environment config (cluster hosts, remote dir, paths, model
  ids, HF env). Machine-specific overrides go in a gitignored
  `jdas.local.toml` (or point `JDAS_CONFIG` / `--config` at any TOML).

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
`das_wrong` (output-copy strawman), `das_wrong_and` (wrong composition law,
with analytic agreement ceiling), `random_rotation` (frozen-Q control).

## Sweeps and cluster runs

Sweeps are declarative TOML specs (grid axes + fixed args + output pattern);
the specs under `experiments/sweeps/` reproduce every committed result grid.
Runs whose output JSON already exists are skipped, so re-running a spec
resumes it.

```bash
# Expand a spec and show the run list + host assignment (nothing executes)
uv run jdas sweep run experiments/sweeps/gates_toy_v3.toml --dry-run

# Run locally, or fan out over the hosts in jdas.toml (one GPU per host)
uv run jdas sweep run experiments/sweeps/gates_toy_v3.toml --where local
uv run jdas sweep run experiments/sweeps/gates_lm_v3.toml --where cluster --wait

uv run jdas sweep status  experiments/sweeps/gates_lm_v3.toml
uv run jdas sweep collect experiments/sweeps/gates_lm_v3.toml  # rsync results back

# Cluster utilities
uv run jdas cluster sync      # rsync repo + uv sync on every host
uv run jdas cluster status    # per-host processes + GPU memory
```

## Analysis

```bash
uv run jdas analyze toy --results-dir experiments/results/phase_a
uv run jdas analyze gates                # gate sweeps (toy + LM)
uv run jdas analyze capped-lm            # capped LM method comparison
uv run jdas analyze seed-basis           # basis (non-)identifiability study
uv run jdas analyze search               # brute-force hypothesis search ranking
uv run jdas analyze falsification        # wrong-composition ceilings
```

Tables land next to the result JSONs; figures under `docs/assets/`.
