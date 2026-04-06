# Plan: (k,5)-Cage Search in Axplorer

## Goal

Find a (k,g)-cage — a k-regular graph of girth g with minimum order.
Primary target: **(8,5)-cage**. Current bounds: lower=67, upper=80 (Royle).
A valid (8,5)-graph on any N in [67,79] establishes a new upper bound record.

---

## Problem Formulation

| Element | Definition |
|---|---|
| Object | Undirected graph on N vertices (adjacency matrix) |
| Parameter N | Number of vertices — swept from 79 downward |
| Runtime arg k | Degree regularity target (default 8) |
| Constraint | Every vertex has degree exactly k AND girth ≥ 5 (no 3- or 4-cycles) |
| Score | See below |

### Scoring Function

```
score = [C(N,3) − triangles]
      + [3·C(N,4) − 4-cycles]
      + [N·k    − degree_penalty]
```

where `degree_penalty = Σ_v |deg(v) − k|`.

- **Max score** = `C(N,3) + 3·C(N,4) + N·k` — achieved iff graph is k-regular with girth ≥ 5.
- **Cycle terms**: penalize each triangle or 4-cycle by 1. Baseline is the total possible in K_N so the score stays non-negative for sparse graphs.
- **Degree term**: `N·k` at max (perfect regularity), decreases by 1 per unit of degree deviation.
- **Scale note**: C(N,4) ~ O(N⁴) dominates. For near-valid graphs all three penalties are small integers — comparable signal. The large constants are offsets only.

---

## Implementation Plan

### Step 1 — `src/envs/cage.py`

#### `CageDataPoint(DataPoint)`

**`__init__(N, k=8, init=False)`**
- If `init=True`: greedy construction.
  - Maintain candidate edges: pairs (i,j) that can be added without creating a triangle or 4-cycle and without exceeding degree k.
  - Shuffle vertices, iterate; for each under-degree vertex pick a random valid candidate neighbor.
  - Recompute valid candidates after each edge addition.
- Call `calc_features()` and `calc_score()` at end.

**`calc_score()`**
- Compute degree_penalty = `sum(|deg(v) - k|)`.
- Compute triangles = `trace(A³) // 6` (via matrix multiply, cast to int32 to avoid overflow).
- Compute 4-cycles: `(trace(A⁴) - Σ_i A²[i,i]·(A²[i,i]-1)/2 ... )` — use the identity:
  `4-cycles = (trace(A⁴) - 2·m·(2k-1) + ...) / 8` for regular graphs,
  or more robustly: `Σ_{i<j} max(0, A²[i,j] - A[i,j]) · (A²[i,j] - A[i,j] - 1) / 2`
  i.e. for each pair, count how many extra common neighbors beyond the direct edge they have, then C(that,2) is the number of 4-cycles through that pair.
  Simpler correct formula: `four_cycles = (Σ_{i<j} C(A²[i,j] - A[i,j], 2))` — pending exact derivation during implementation.
- `self.score = C(N,3) - triangles + 3*C(N,4) - four_cycles + N*k - degree_penalty`

**`calc_features()`**
- Upper triangle of adjacency matrix as comma-separated string (same as `SquareDataPoint`).

**`local_search(improve_with_local_search)`**

Two strategies will be implemented and profiled head-to-head (see Phase 2):

- **Strategy A — Remove & Greedily Re-add** (mirrors `SquareEnvironment`):
  1. Remove edges causing degree violations (over-degree vertices lose random edges).
  2. Remove edges involved in short cycles (highest-multiplicity first).
  3. Greedily re-add edges: for each under-degree vertex, try random candidates that don't create 3/4-cycles and don't exceed degree k.

- **Strategy B — Edge Swaps**:
  1. Fix degree violations via swaps: for an over-degree vertex u and an under-degree vertex w, find a neighbor v of u such that (w,v) is not an existing edge and adding (w,v) + removing (u,v) preserves girth ≥ 5.
  2. For girth violations: find a short cycle edge and attempt a swap that eliminates it.
  - More targeted, but each swap requires a girth re-check.

#### `CageEnvironment(BaseEnvironment)`

- `k = 2` (pairs of vertices), `are_coordinates_symmetric = True`
- Tokenizer: `SparseTokenizerSingleInteger` (edge (i,j) encoded as `N·i + j`)
- Register args: `--N` (number of vertices), `--k` (degree, default 8)
- `max_len` must cover up to `N·k/2` edges + special tokens (e.g. for N=79, k=8: 316 edges + ~4 special tokens → `max_len=350` is safe)

#### Register in `src/envs/__init__.py`
```python
from src.envs.cage import CageEnvironment
ENVS["cage"] = CageEnvironment
```

---

### Step 2 — Local Validation (mandatory before any commit)

#### 2a. Unit test `calc_score()` on known graphs

```bash
python3 -c "
from src.envs.cage import CageDataPoint
import numpy as np

# Test 1: Petersen graph — (3,5)-cage on 10 vertices
# Should have: degree_penalty=0, triangles=0, 4-cycles=0 → max score for k=3, N=10
petersen_edges = [
    (0,1),(0,4),(0,5),(1,2),(1,6),(2,3),(2,7),(3,4),(3,8),(4,9),
    (5,7),(5,8),(6,8),(6,9),(7,9)
]
dp = CageDataPoint(N=10, k=3, init=False)
for i,j in petersen_edges:
    dp.data[i,j] = dp.data[j,i] = 1
dp.calc_score()
max_score = ...  # C(10,3) + 3*C(10,4) + 10*3
print(f'Petersen: score={dp.score}, expected={max_score}, valid={dp.score==max_score}')

# Test 2: K_4 — complete graph on 4 vertices, girth=3
# Should have triangles=4, 4-cycles=3, degree=3 each
dp2 = CageDataPoint(N=4, k=3, init=False)
for i in range(4):
    for j in range(i+1, 4):
        dp2.data[i,j] = dp2.data[j,i] = 1
dp2.calc_score()
print(f'K4: score={dp2.score}, expect heavy penalty for triangles')

# Test 3: Empty graph N=10, k=8 — no cycles, but degree_penalty=10*8=80
dp3 = CageDataPoint(N=10, k=8, init=False)
dp3.calc_score()
print(f'Empty: score={dp3.score}')
"
```

#### 2b. Profile local search strategies

```bash
python3 -c "
import time, numpy as np
from src.envs.cage import CageDataPoint

N, k = 79, 8
TRIALS = 100

# Strategy A
t0 = time.time()
scores_a = []
for _ in range(TRIALS):
    dp = CageDataPoint(N=N, k=k, init=True)
    dp.local_search(improve_with_local_search=True)
    scores_a.append(dp.score)
print(f'Strategy A: {time.time()-t0:.2f}s, mean score={sum(scores_a)/len(scores_a):.0f}')

# Strategy B
t0 = time.time()
scores_b = []
for _ in range(TRIALS):
    dp = CageDataPoint(N=N, k=k, init=True)
    dp.local_search_swap(improve_with_local_search=True)
    scores_b.append(dp.score)
print(f'Strategy B: {time.time()-t0:.2f}s, mean score={sum(scores_b)/len(scores_b):.0f}')
"
```

Pick the strategy with better mean score per unit time. If close, prefer A (simpler, fewer bugs).

#### 2c. Smoke test the full pipeline

Target: completes in under 2 minutes locally.

```bash
python3 train.py \
    --env_name cage \
    --exp_name cage_smoke \
    --N 20 \
    --k 3 \
    --max_epochs 2 \
    --max_steps 100 \
    --gensize 200 \
    --pop_size 100 \
    --num_samples_from_model 100 \
    --batch_size 8 \
    --n_layer 2 \
    --n_head 4 \
    --n_embd 64 \
    --encoding_tokens single_integer \
    --max_len 50 \
    --always_search true
```

Check: no crashes, scores logged and non-trivial, local search runs without error, model samples decode cleanly.

**Do not proceed to Phase 3 until 2a, 2b, and 2c all pass.**

---

### Step 3 — Commit & Push

Files to commit:
- `src/envs/cage.py`
- `src/envs/__init__.py`
- `RUNNING.md` (see below)
- `plan.md` (this file)

`RUNNING.md` will contain exactly one thing: the production training command.

---

## Production Training Command (L40s, 45 GiB VRAM)

Target N=79 first. Sweep downward if a valid graph is found.

gensize is set so initial data generation completes within ~10 minutes on CPU before GPU training begins.

```bash
python3 train.py \
    --env_name cage \
    --exp_name cage85_N79 \
    --N 79 \
    --k 8 \
    --encoding_tokens single_integer \
    --max_len 350 \
    --gensize 20000 \
    --pop_size 10000 \
    --num_samples_from_model 5000 \
    --max_epochs 1000 \
    --max_steps 500 \
    --temperature 0.8 \
    --inc_temp 0.1 \
    --always_search true \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 256 \
    --batch_size 128
```

**Notes:**
- `gensize=20000`: TBD — will be calibrated during local profiling in Step 2b to hit ~10 min.
- `max_len=350`: covers 316 edges (N=79, k=8) + special tokens with margin.
- `batch_size=128`: reasonable for 45 GiB VRAM with this model size; adjust if OOM.
- `pop_size=10000`: keeps top-10K near-valid graphs as training data.

---

## Sweep Schedule

Run in order, stop when a valid graph is found:

| Run | N | exp_name |
|-----|---|----------|
| 1 | 79 | cage85_N79 |
| 2 | 78 | cage85_N78 |
| ... | ... | ... |
| 13 | 67 | cage85_N67 |

A valid graph is found when `max score = C(N,3) + 3·C(N,4) + N·k` appears in the logs.
