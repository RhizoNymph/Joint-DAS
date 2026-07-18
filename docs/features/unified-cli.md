# Unified CLI (`jdas`)

## Scope

One entry point for everything the repo can do — single runs, sweeps, cluster
operations, analysis — replacing the per-experiment python scripts and the
per-night bash drivers in `scripts/`. Environment specifics (hosts, paths,
model ids, env vars) live in a checked-in config file with local override,
never in code.

Non-scope: changing any experiment semantics, loss, or result JSON schema.
Existing result files must remain readable by the new analyzers.

## Environment config

`jdas.toml` at the repo root (checked in, non-secret), stdlib `tomllib`:

```toml
[cluster]
hosts = ["node0", "node1", "node2"]   # ssh destinations (aliases or user@host)
remote_dir = "Code/ai/learning-causal-representations"  # relative to remote $HOME
uv_path = "~/.local/bin/uv"

[cluster.env]                          # exported before every remote command
HF_HOME = "$HOME/hf-cache"
HF_HUB_OFFLINE = "1"
HF_HUB_DISABLE_XET = "1"

[paths]                                # all relative to repo root
results = "experiments/results"
logs = "experiments/logs"
ckpts = "experiments/ckpts"
toy_ckpts = "experiments/toy_ckpts"

[models]
lm_default = "Qwen/Qwen2.5-1.5B-Instruct"
```

Precedence: `--config PATH` flag > `JDAS_CONFIG` env var > `./jdas.local.toml`
(gitignored, for machine-specific overrides) > `./jdas.toml`. Loaded into a
frozen typed dataclass (`EnvConfig`); unknown keys are an error (typo guard).
A missing config file yields the defaults above with a warning — the CLI must
work out of the box on this repo.

## Command tree

```
jdas run toy      --task ... --method ... [toy-model alignment run]
jdas run lm       --layer ... --method ... [HF language-model alignment run]
jdas run search   [discrete search baseline]
jdas run seed-study [seed / basis variance study]
jdas analyze gates                    # night-3 gate sweeps (toy + lm)
jdas analyze toy   --results-dir ...  # aggregate toy-model / lm result table + plots
jdas analyze capped-lm|seed-basis|search|falsification   # per-study plots
jdas sweep run    SPEC.toml [--where local|cluster] [--wait] [--dry-run]
jdas sweep status SPEC.toml
jdas sweep collect SPEC.toml          # rsync results back from hosts
jdas cluster sync                     # rsync repo + uv sync on every host
jdas cluster status                   # per host: relevant processes + GPU memory
jdas cluster exec -- CMD...           # run a command on every host
jdas cluster kill PATTERN             # pkill -f on every host (bracket-escaped)
```

`jdas analyze toy` aggregates a directory of toy-model **or** LM result JSONs
(schema-tolerant); the four study subcommands each render one figure from the
committed `experiments/results/night2` study JSONs.

argparse subparsers, no new dependencies. Registered as a console script
(`[project.scripts] jdas = "jdas.cli:main"`) so `uv run jdas ...` works.

## Sweep specs

Declarative TOML in `experiments/sweeps/`, one file per sweep; the night-1/2/3
grids are re-expressed as specs so the committed results stay reproducible:

```toml
[sweep]
name = "gates_toy_v3"                  # tag; also the default log prefix
runner = "toy"                         # toy | lm | search | seed-study
out_dir = "night3/gates_toy_v3"        # under paths.results
out_pattern = "{task_short}_l{site_layer}_lg{lambda_gate}_s{seed}.json"

[grid]                                 # cartesian product, list per axis
"task+site_layer" = [["hierarchical_equality", 1], ["hierarchical_equality", 2], ["boolean_comp", 1]]
lambda_gate = [0, 0.01, 0.03, 0.1, 0.3]
seed = [0, 1, 2]

[fixed]                                # constant args for every run
method = "joint"
gates = true
gate_lr = 0.05
gate_warmup = 300
gate_lambda_ramp = 300
gate_clamp = 3.0
steps = 1500
k_max = 4
device = "cuda"
```

Semantics:
- `"a+b"` axis names zip several parameters that vary together; plain axes are
  independent grid dimensions. `{task_short}` in patterns maps task names to
  short forms (hierarchical_equality→hier, boolean_comp→bool).
- **Skip-existing**: a run whose output JSON already exists is skipped —
  re-running a spec resumes it.
- Execution local: runs sequentially (or `--parallel N`).
- Execution cluster: runs are dealt round-robin to `cluster.hosts`; each host
  executes its share sequentially (one GPU per host). The executor generates a
  per-host driver script, copies it over, and launches it detached.
- `--dry-run` prints the expanded run list and per-host assignment.

## Remote execution rules (hard-won; encode, don't rediscover)

- Launch: `ssh -f -n HOST 'nohup bash DRIVER < /dev/null > /dev/null 2>&1 &'`.
  Never `ssh -n HOST '(nohup ... &)'` — the remote stdin keeps the connection
  open and the local ssh blocks until the job exits.
- Every remote run appends to a per-run log under `paths.logs`; failures append
  a line to `<name>_failures.log`; each host appends `DONE_<host>` to
  `<name>_status.log` when its queue drains.
- `sweep status` = count of expected vs present output files (locally and per
  host) + failure-log contents + whether the driver process is alive.
- `sweep collect` = `rsync --ignore-existing` from every host's out_dir into
  the local one (collect is also run automatically at the end of `--wait`).
- `cluster kill` wraps the pattern's first char in `[]` so pkill can never
  match its own command line.
- `--wait` polls status every N seconds (default 300) until complete/timeout.

## Files

- `src/jdas/cli/__init__.py` — `main()`, argparse tree, dispatch.
- `src/jdas/cli/config.py` — `EnvConfig` dataclass + TOML loader + precedence.
- `src/jdas/cli/runners.py` — toy/lm/search/seed-study run logic (the sole
  home; argument names unchanged from the original per-experiment scripts).
- `src/jdas/cli/sweeps.py` — spec model, grid expansion, out-pattern
  rendering, local executor, driver generation.
- `src/jdas/cli/cluster.py` — sync/status/exec/kill/launch/collect primitives
  used by both `jdas cluster` and the sweep executor.
- `src/jdas/cli/analyze.py` — analyze subcommand dispatch to the analyzer
  modules (`analyze_gates`, `analyze_toy_lm`, `analyze_studies`).
- `experiments/analyze_toy_lm.py` — toy-model / LM aggregate table + plots
  (`jdas analyze toy`). `experiments/analyze_studies.py` — the four study
  plotters (`jdas analyze capped-lm|seed-basis|search|falsification`).
  `experiments/analyze_gates.py` — night-3 gate sweeps (`jdas analyze gates`).
- `experiments/introspect_toy.py`, `screen_lm.py` — one-off diagnostics that
  stay importable modules (`python experiments/...`), not part of the `jdas`
  command tree.
- `experiments/sweeps/*.toml` — night-1/2/3 grids as specs (at minimum:
  gates_toy_v3, gates_lm_v3, plus the v1/v2 variants and `capped_lm.toml`).
- `jdas.toml` — this environment's config. `jdas.local.toml` gitignored.
- DELETED: every file in `scripts/` (sync_nodes.sh, launch_*.sh,
  rerun_baselines.sh, sweep_gates_toy.sh, gates_*_node*.sh, gates_v*_node*.sh).

## Invariants and constraints

- No hostname, absolute path, model id, or HF env var may appear in code —
  only in `jdas.toml` (enforced by review; config is the single source).
- Sweep expansion is deterministic: same spec → same run list and same
  host assignment (host = index % len(hosts) over the sorted run list).
- The executor never launches a run whose output exists (idempotent resume).
- Remote drivers are generated from the spec at launch time, never hand
  edited; they are written under `paths.logs` on the remote side only.
- All subprocess/ssh construction lives in `cluster.py` behind small
  functions that tests can exercise by inspecting generated commands
  (tests never actually ssh).
- RESULTS.md §7 (Reproduction) and docs/OVERVIEW.md must be updated to the
  `jdas` commands.
