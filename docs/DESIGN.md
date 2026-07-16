# Joint-DAS: Learning the Causal Model Jointly with the Alignment

## Background

Distributed Alignment Search (DAS, Geiger et al. 2023) tests whether a neural
network N implements a *hypothesized* high-level causal model H. It learns an
orthogonal rotation Q of a hidden representation h ∈ R^d and assigns each
high-level variable Z_i a subspace (block of rotated coordinates) B_i. An
**interchange intervention** swaps the B_i coordinates of a base input's hidden
state with those from a source input; DAS trains Q so the network's output under
this swap matches H's counterfactual prediction when Z_i is set to its
source-input value. The metric is IIA (interchange intervention accuracy).

**Limitation this project attacks:** H must be proposed by hand. We instead
*learn H jointly* with Q, or search over a family of H, so the causal
representation itself is discovered.

## Method

### Parameterization of the learned causal model H_θ

- k latent variables Z_1..Z_k (k is an upper bound; unused variables can die).
- Each Z_i ∈ {0..v-1} discrete (default v=2, binary). Computed from the *raw
  task input* by a small per-variable MLP encoder g_i with Gumbel-softmax /
  straight-through discretization (temperature annealed).
- A small decoder r(Z_1..Z_k) → label distribution (linear over concatenated
  one-hots, or 1-hidden-layer MLP).
- Alignment: rotated space Q·h partitioned; variable Z_i owns block B_i.
  Block widths learned Boundless-DAS-style: per-variable continuous boundary
  masks (sigmoid, temperature annealed), plus leftover "don't-touch" dims.

### Why this is constrained enough to be meaningful (anti-degeneracy)

The claim being optimized is: "the counterfactual behavior of N at this site
factors through k low-cardinality variables computed from the input and
linearly encoded in disjoint subspaces." Capacity bottleneck = few binary
variables + tiny decoder. Two failure modes and their mitigations:

1. **Output-copy variable** (Z_1 = y): legitimate but uninteresting. Mitigated
   by **multi-source multi-variable interventions** (below) — a single
   output-copy variable cannot reproduce counterfactuals that require composing
   values from different sources; the true factored structure can.
2. **Dead/constant variables**: fine; report effective k. Sparsity penalty on
   total aligned dimensions.

### Interventions (the identifiability workhorse)

Sample base b and sources s_1..s_m. Choose a non-empty subset I ⊆ {1..k};
for each i ∈ I take Z_i's value (and subspace content) from a *different*
source. Distribution over |I|: {1: 0.5, 2: 0.5} (include m=2 distinct sources
for |I|=2). High-level counterfactual: ŷ_cf = r(z(b) with z_i ← z_i(s_i)).
Low-level: run N on b with block B_i of Q·h(b) replaced by block B_i of
Q·h(s_i). Multi-source composition forces factored solutions — this is the key
identifiability mechanism.

### Losses (N is frozen throughout; trainable: Q, boundaries, θ_H)

- **L_cf** = CE(N's output distribution under low-level intervention,
  H_θ's counterfactual label via straight-through). Gradients flow to both Q
  and θ_H.
- **L_task** = CE(r(z(x)), y_true(x)) on clean inputs — H must implement the
  task itself. Without this, H can be vacuous.
- **L_sparse** = λ · (total aligned dims / d), boundary sparsity.
- Anneal: Gumbel temperature 1.0 → 0.1; boundary mask temperature likewise.
- Optionally freeze-and-refit: after joint training, discretize H (hard
  argmax), refit Q alone (standard DAS) to the frozen discovered H, report
  final IIA. This separates "did we find a good H" from soft-training slack.

### Evaluation

- **IIA** on held-out counterfactual pairs (hard discretization), for
  |I|=1 and |I|=2 separately (|I|=2 with distinct sources is the stringent
  test).
- **Recovery**: match learned variables to ground-truth variables (allow
  permutation and value relabeling): for each (Z_i, GT_j) report agreement
  accuracy over held-out inputs; summarize as best-match assignment.
- **Baselines/controls**:
  (a) DAS with the *true* hand-specified H (upper bound / sanity);
  (b) DAS with a deliberately wrong H (e.g., single output-copy variable);
  (c) random frozen rotation + learned H (control: is rotation doing work?);
  (d) joint method (ours);
  (e) if time: discrete search — enumerate small H family, fit Q per
      candidate, select by held-out IIA; compare to joint gradient method.

## Experiments

### Phase A — toy models with known ground truth (node0, node2)

Tasks (inputs are random vectors per symbol, as in the DAS paper):
1. **Hierarchical equality**: input (a,b,c,d) ∈ R^{4·n_emb}; y = ((a==b) == (c==d)).
   GT variables: E1=(a==b), E2=(c==d). Expect k_eff=2 recovered.
2. **Boolean composition**: y = (x1 ∧ x2) ∨ x3 over token inputs. GT var:
   A=(x1∧x2) (plus x3 passthrough). Expect k_eff∈{1,2}.

Toy network: MLP (3 hidden layers, width 256) or tiny transformer trained to
>99% task accuracy. Intervention site: hidden layer 1 or 2 (sweep).
Seeds: ≥3 per config. These run fast (CPU-viable, GPU trivial).

### Phase B — real LM (node1)

Model: small HF instruct model (~0.5B, e.g. Qwen2.5-0.5B-Instruct) — must fit
training + backprop-through-frozen-model on a 24GB 3090.
Task: **price tagging** (Boundless DAS): "Does the following item cost between
X.xx and Y.yy dollars? Item: Z.zz" → Yes/No. GT variables: L=(Z≥X), U=(Z≤Y) —
same two-boolean skeleton as hierarchical equality, so recovery is measurable
on a real LM. Site: residual stream at final token, sweep 3–4 layers.
N frozen; backprop through frozen weights to the intervention site only
(truncate: layers below the site don't need gradients).

## Engineering

```
src/jdas/
  rotation.py       # OrthogonalRotation (nn.utils.parametrizations.orthogonal),
                    # SubspaceMasks (learned boundaries)
  causal_model.py   # LearnedCausalModel, FixedCausalModel (for baselines)
  intervention.py   # interchange machinery + hooks (toy MLP, HF transformer)
  training.py       # JointTrainer, DASTrainer, configs (dataclasses)
  eval.py           # IIA, recovery matrix, controls
  tasks/            # hierarchical_equality.py, boolean_comp.py, price_tagging.py
  models/           # toy model defs + training loops
experiments/        # config-driven entry points, results as JSON
scripts/            # node sync + launch + collect
docs/               # this file, OVERVIEW.md, features/
```

- Python ≥3.12 (nodes have 3.12), uv-managed. torch pinned (cu12x wheels),
  transformers pinned, numpy, matplotlib.
- Everything runs CPU (tests, smoke) and CUDA (real runs). Results = JSON files
  under experiments/results/ (rsynced back from nodes).
- Nodes: node0/node2 Phase A sweeps + seeds; node1 Phase B.
- Tests first where practical: rotation orthogonality, mask behavior,
  interchange correctness on a hand-computable example, causal-model
  counterfactual semantics, recovery metric on synthetic assignments.

## Timeline (8h, started 00:30)

- 00:30–01:00 design, docs, scaffolding (this doc)
- 01:00–02:30 core lib + toy tasks (parallel Opus agents), node setup
- 02:30–04:00 Phase A runs + iterate; Phase B code built in parallel
- 04:00–06:30 Phase B runs, seed sweeps, ablations
- 06:30–08:00 analysis, plots, RESULTS.md, PR
