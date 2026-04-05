"""
Plateau-breaking search: 2-flip local search + Kempe-chain (ejection chain)
perturbation, each with its own dedicated pool of 16 workers.

Rationale: when every graph in the pool is a 1-flip local maximum (the current
situation at gap=25), neither single-flip hill climbing nor small random kicks
will escape. Both strategies here are designed specifically to break 1-flip
basins:

  - 2-flip LS: finds the best *simultaneous pair* of edge flips among the top-K
    most promising singles. This provably escapes 1-flip basins because it can
    select pairs where each flip individually hurts but together improve score.

  - Kempe-chain ejection: starts at the worst node (highest total violation
    delta), flips the locally-best incident edge with a tabu, then "walks" to
    the other endpoint and repeats. Builds a chain of mutually-reinforcing
    moves, landing in a different basin before SA reconstructs.

Usage:
    python plateau_break.py \\
        --pkl checkpoint/r46_n36/r46_n36/<exp_id>/train_data.pkl \\
        --workers_two_flip 16 \\
        --workers_kempe 16
"""

import argparse
import copy
import os
import pickle
import signal
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from src.envs.ramsey import RamseyDataPoint


# ── worker functions (top-level for pickling) ─────────────────────────────────

def _run_two_flip(args_tuple):
    """
    2-flip perturbation: enumerate the top-K single-flip candidates (by delta,
    including non-improving ones), try every pair among them, apply the pair
    that yields the largest combined score, then SA.
    """
    dp, pars, sa_steps, top_k = args_tuple
    dp.__class__._update_class_params(pars)
    dp = copy.deepcopy(dp)
    n = dp.N

    # Single-flip deltas (does not mutate the graph).
    singles = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dp._score_delta_for_flip(i, j)
            singles.append((d, i, j))
    singles.sort(reverse=True)
    cand = singles[: max(2, top_k)]

    orig_score = int(dp.score) if dp.score is not None else 0
    if orig_score == 0:
        dp.calc_score()
        orig_score = int(dp.score)

    best_delta = -(1 << 30)
    best_pair = None

    # Try all pairs (i1,j1),(i2,j2). Apply both, full rescore, revert.
    for a_idx in range(len(cand)):
        _, i1, j1 = cand[a_idx]
        dp._flip_edge(i1, j1)
        for b_idx in range(a_idx + 1, len(cand)):
            _, i2, j2 = cand[b_idx]
            dp._flip_edge(i2, j2)
            dp.calc_score()
            d = int(dp.score) - orig_score
            if d > best_delta:
                best_delta = d
                best_pair = (i1, j1, i2, j2)
            dp._flip_edge(i2, j2)  # revert second
        dp._flip_edge(i1, j1)  # revert first

    # Apply best pair (even if delta ≤ 0 — it's a perturbation)
    if best_pair is not None:
        i1, j1, i2, j2 = best_pair
        dp._flip_edge(i1, j1)
        dp._flip_edge(i2, j2)

    dp.calc_score()
    dp.calc_features()
    dp.local_search_fast_v2(sa_steps=sa_steps)
    return dp


def _run_kempe_chain(args_tuple):
    """
    Ejection chain walking starting from the node with highest cumulative
    violation pressure. At each step: among non-tabu edges incident to the
    current focus node, pick the one with the largest (least-negative) delta,
    flip it, and move focus to the other endpoint. Repeat chain_len times,
    then SA.
    """
    dp, pars, sa_steps, chain_len = args_tuple
    dp.__class__._update_class_params(pars)
    dp = copy.deepcopy(dp)
    n = dp.N

    # Compute per-node badness from single-flip deltas.
    deltas = {}
    node_badness = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            d = dp._score_delta_for_flip(i, j)
            deltas[(i, j)] = d
            if d < 0:
                node_badness[i] += -d
                node_badness[j] += -d

    if node_badness.max() > 0:
        focus = int(np.argmax(node_badness))
    else:
        focus = int(np.random.randint(n))

    tabu = set()
    for _ in range(chain_len):
        best_d = -(1 << 30)
        best_e = None
        for j in range(n):
            if j == focus:
                continue
            a, b = (focus, j) if focus < j else (j, focus)
            if (a, b) in tabu:
                continue
            d = dp._score_delta_for_flip(a, b)
            if d > best_d:
                best_d = d
                best_e = (a, b)
        if best_e is None:
            break
        a, b = best_e
        dp._flip_edge(a, b)
        tabu.add(best_e)
        focus = b if a == focus else a

    dp.calc_score()
    dp.calc_features()
    dp.local_search_fast_v2(sa_steps=sa_steps)
    return dp


# ── helpers ───────────────────────────────────────────────────────────────────

def wl_hash(data_matrix):
    import networkx as nx
    from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
    return weisfeiler_lehman_graph_hash(nx.from_numpy_array(data_matrix))


def _save(pool, out_path):
    tmp = out_path + ".tmp"
    pickle.dump(pool, open(tmp, "wb"))
    os.replace(tmp, out_path)


def print_dist(data, label="Pool", top_n=10):
    scores = [d.score for d in data]
    if not scores:
        return
    counts = Counter(scores)
    top = sorted(counts.items(), reverse=True)[:top_n]
    print(f"\n{label}: {len(scores)} graphs | max={max(scores)} | mean={sum(scores)/len(scores):.1f}")
    for score, count in top:
        print(f"  {score}: x{count}")


def _worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--workers_two_flip", type=int, default=16)
    parser.add_argument("--workers_kempe", type=int, default=16)
    parser.add_argument("--frontier_k", type=int, default=32,
                        help="number of frontier graphs to search per pass")
    parser.add_argument("--sa_mult", type=int, default=50,
                        help="SA depth multiplier; sa_steps = N*N*sa_mult")
    parser.add_argument("--two_flip_top_k", type=int, default=40,
                        help="number of candidate edges to consider for 2-flip pairs (K^2/2 pair tests)")
    parser.add_argument("--kempe_chain_len", type=int, default=12,
                        help="ejection chain length (number of flips in the chain)")
    # (no --save_interval — we only save on new best or on exit)
    args = parser.parse_args()

    out_path = args.out or args.pkl

    print(f"Loading {args.pkl} ...")
    data = pickle.load(open(args.pkl, "rb"))
    data.sort(key=lambda d: d.score, reverse=True)
    print(f"Loaded {len(data)} graphs")

    N = data[0].N
    max_possible = RamseyDataPoint.max_possible_score(N)
    best_score = data[0].score
    sa_steps = N * N * args.sa_mult
    print(f"Max possible: {max_possible} | Current best: {best_score} "
          f"(gap={max_possible - best_score}) | sa_steps={sa_steps}")
    print(f"Workers: 2-flip={args.workers_two_flip}  kempe={args.workers_kempe}  "
          f"(total={args.workers_two_flip + args.workers_kempe})")
    print(f"2-flip top_k={args.two_flip_top_k} (≈{args.two_flip_top_k*(args.two_flip_top_k-1)//2} pair tests/task)")
    print(f"Kempe chain_len={args.kempe_chain_len}")

    RamseyDataPoint._nb_warmup()
    pars = RamseyDataPoint._save_class_params()

    pool = list(data)
    seen_features = {d.features for d in pool}
    print("Building WL seen set...")
    seen_wl = {wl_hash(d.data) for d in tqdm(pool)}

    stop = False

    def _handle_sigint(sig, frame):
        nonlocal stop
        stop = True
        print("\nCtrl+C received — finishing current batch then saving...", flush=True)

    signal.signal(signal.SIGINT, _handle_sigint)

    pass_num = 0
    strategy_hits = Counter()

    # Two dedicated executors running concurrently.
    with ProcessPoolExecutor(max_workers=args.workers_two_flip, initializer=_worker_init) as exec_tf, \
         ProcessPoolExecutor(max_workers=args.workers_kempe, initializer=_worker_init) as exec_kc:

        while not stop:
            pass_num += 1
            max_score_val = max(d.score for d in pool)
            top_tier = [d for d in pool if d.score == max_score_val]
            if len(top_tier) >= args.frontier_k:
                idx = np.random.choice(len(top_tier), args.frontier_k, replace=False)
                frontier = [top_tier[i] for i in idx]
            else:
                frontier = sorted(pool, key=lambda d: d.score, reverse=True)[: args.frontier_k]

            # Submit a full batch to each executor: one task per frontier graph
            # per strategy. Each executor saturates its own 16 workers.
            futs = {}
            for dp in frontier:
                f = exec_tf.submit(_run_two_flip,
                                   (copy.deepcopy(dp), pars, sa_steps, args.two_flip_top_k))
                futs[f] = "2flip"
            for dp in frontier:
                f = exec_kc.submit(_run_kempe_chain,
                                   (copy.deepcopy(dp), pars, sa_steps, args.kempe_chain_len))
                futs[f] = "kempe"

            n_added = 0
            n_improved = 0
            pass_start = time.time()

            print(f"\n=== Pass {pass_num} | frontier: {len(frontier)} at {max_score_val} "
                  f"(gap={max_possible - max_score_val}) | {len(futs)} tasks "
                  f"({len(frontier)} × 2 strategies) ===")

            try:
                for future in tqdm(as_completed(futs), total=len(futs), desc=f"Pass {pass_num}"):
                    if stop:
                        break
                    label = futs[future]
                    try:
                        dp = future.result()
                    except Exception as e:
                        print(f"  [{label}] worker error: {e}")
                        continue

                    # Only keep STRICT improvements over the current best —
                    # the pool is already saturated with same-score graphs and
                    # we don't want to bloat it further.
                    if dp.score <= best_score:
                        continue
                    if dp.features in seen_features:
                        continue
                    h = wl_hash(dp.data)
                    if h in seen_wl:
                        continue

                    seen_features.add(dp.features)
                    seen_wl.add(h)
                    pool.append(dp)
                    n_added += 1
                    best_score = dp.score
                    strategy_hits[f"{label}-NEW-BEST"] += 1
                    n_improved += 1
                    print(f"\n  *** [{label}] NEW BEST: {dp.score} "
                          f"(gap={max_possible - dp.score}) ***", flush=True)
                    # Save ONLY on a new best — we don't care about anything
                    # else, and save_interval churn was slowing the search.
                    pool.sort(key=lambda d: d.score, reverse=True)
                    _save(pool, out_path)
            finally:
                if stop:
                    for f in futs:
                        f.cancel()

            elapsed = time.time() - pass_start
            print(f"Pass {pass_num} done | {len(futs)/max(elapsed,1e-6):.1f} tasks/s "
                  f"| new bests this pass: {n_improved} | best overall: {best_score} "
                  f"(gap={max_possible - best_score})")
            if strategy_hits:
                print(f"Strategy hits: {dict(strategy_hits.most_common())}")

    # Final save on exit (Ctrl+C).
    pool.sort(key=lambda d: d.score, reverse=True)
    _save(pool, out_path)
    print(f"\nDone. Final pool: {len(pool)} | best: {best_score} "
          f"(gap={max_possible - best_score})")
    if strategy_hits:
        print(f"Strategy hits: {dict(strategy_hits.most_common())}")


if __name__ == "__main__":
    main()
