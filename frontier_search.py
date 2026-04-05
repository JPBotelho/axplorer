"""
Run all perturbation strategies at varying depths on the frontier graphs (top-50 by score).
Prints immediately when a new best is found. Loops until Ctrl+C.

Usage:
    python frontier_search.py \
        --pkl checkpoint/r46_n36/r46_n36/<exp_id>/train_data.pkl \
        --num_workers 32
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


# ── worker functions (must be top-level for pickling) ────────────────────────

def _run_double_bridge(args_tuple):
    dp, pars, sa_steps, n_flips = args_tuple
    dp.__class__._update_class_params(pars)
    dp = copy.deepcopy(dp)
    n = dp.N
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    chosen = [all_pairs[k] for k in np.random.choice(len(all_pairs), size=min(n_flips, len(all_pairs)), replace=False)]
    for i, j in chosen:
        dp._flip_edge(i, j)
    dp.calc_score(); dp.calc_features()
    dp.local_search_fast_v2(sa_steps=sa_steps)
    return dp


def _run_targeted(args_tuple):
    dp, pars, sa_steps, n_flip = args_tuple
    dp.__class__._update_class_params(pars)
    dp = copy.deepcopy(dp)
    n = dp.N
    deltas = [(dp._score_delta_for_flip(i, j), i, j) for i in range(n) for j in range(i + 1, n)]
    deltas.sort()
    worst = deltas[:max(n_flip * 2, 10)]
    chosen = [worst[k] for k in np.random.choice(len(worst), size=min(n_flip, len(worst)), replace=False)]
    for _, i, j in chosen:
        dp._flip_edge(i, j)
    dp.calc_score(); dp.calc_features()
    dp.local_search_fast_v2(sa_steps=sa_steps)
    return dp


def _run_violation(args_tuple):
    dp, pars, sa_steps, n_flip = args_tuple
    dp.__class__._update_class_params(pars)
    dp = copy.deepcopy(dp)
    n = dp.N
    deltas = {}
    for i in range(n):
        for j in range(i + 1, n):
            deltas[(i, j)] = dp._score_delta_for_flip(i, j)
    node_badness = np.zeros(n)
    for (i, j), d in deltas.items():
        if d < 0:
            node_badness[i] += abs(d)
            node_badness[j] += abs(d)
    worst_node = int(np.argmax(node_badness))
    incident = sorted(
        [(deltas[(min(worst_node, j), max(worst_node, j))], min(worst_node, j), max(worst_node, j))
         for j in range(n) if j != worst_node],
        key=lambda x: x[0]
    )
    for _, i, j in incident[:n_flip]:
        dp._flip_edge(i, j)
    dp.calc_score(); dp.calc_features()
    dp.local_search_fast_v2(sa_steps=sa_steps)
    return dp


def _run_consensus(args_tuple):
    dp1, dp2, pars, sa_steps = args_tuple
    dp1.__class__._update_class_params(pars)
    dp1 = copy.deepcopy(dp1); dp2 = copy.deepcopy(dp2)
    n = dp1.N
    for i in range(n):
        for j in range(i + 1, n):
            if dp1.data[i, j] != dp2.data[i, j]:
                if np.random.randint(2):
                    dp1._flip_edge(i, j)
    dp1.calc_score(); dp1.calc_features()
    dp1.local_search_fast_v2(sa_steps=sa_steps)
    return dp1


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


def make_tasks(frontier, pars, N, sa_mults):
    """Generate one task per (strategy, depth, graph) combination."""
    tasks = []
    n_flip_base = max(2, N // 10)

    for sa_mult in sa_mults:
        sa_steps = N * N * sa_mult
        for dp in frontier:
            # double-bridge at varying flip counts
            for n_flips in [4, 8, 16]:
                tasks.append((_run_double_bridge, (copy.deepcopy(dp), pars, sa_steps, n_flips),
                               f"dbridge-{n_flips}f-{sa_mult}x"))
            # targeted
            tasks.append((_run_targeted, (copy.deepcopy(dp), pars, sa_steps, n_flip_base),
                           f"targeted-{sa_mult}x"))
            # violation
            tasks.append((_run_violation, (copy.deepcopy(dp), pars, sa_steps, n_flip_base),
                           f"violation-{sa_mult}x"))

        # consensus: pair each frontier graph with a random other
        for dp in frontier:
            dp2 = frontier[np.random.randint(len(frontier))]
            tasks.append((_run_consensus, (copy.deepcopy(dp), copy.deepcopy(dp2), pars, sa_steps),
                           f"consensus-{sa_mult}x"))

    np.random.shuffle(tasks)
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--frontier_k", type=int, default=50, help="number of frontier graphs to search from")
    parser.add_argument("--sa_mults", type=str, default="10,50,200", help="comma-separated SA depth multipliers")
    parser.add_argument("--save_interval", type=int, default=60)
    parser.add_argument("--max_pool_size", type=int, default=10000,
                        help="cap pool at this many graphs (keep top by score); 0 = unbounded")
    args = parser.parse_args()

    out_path = args.out or args.pkl
    sa_mults = [int(x) for x in args.sa_mults.split(",")]

    print(f"Loading {args.pkl} ...")
    data = pickle.load(open(args.pkl, "rb"))
    data.sort(key=lambda d: d.score, reverse=True)
    print(f"Loaded {len(data)} graphs")

    N = data[0].N
    max_possible = RamseyDataPoint.max_possible_score(N)
    best_score = data[0].score
    print(f"Max possible: {max_possible} | Current best: {best_score} (gap={max_possible - best_score})")

    RamseyDataPoint._nb_warmup()
    pars = RamseyDataPoint._save_class_params()

    pool = list(data)
    seen_features = {d.features for d in pool}
    print("Building WL seen set...")
    seen_wl = {wl_hash(d.data) for d in tqdm(pool)}

    stop = False
    pass_num = 0
    strategy_hits = Counter()

    # Workers must ignore SIGINT — only main process handles it
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _worker_init():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _handle_sigint(sig, frame):
        nonlocal stop
        stop = True
        print("\nCtrl+C received — finishing current batch then saving...", flush=True)

    signal.signal(signal.SIGINT, _handle_sigint)

    while not stop:
        pass_num += 1
        max_score_val = max(d.score for d in pool)
        top_tier = [d for d in pool if d.score == max_score_val]
        if len(top_tier) >= args.frontier_k:
            frontier = [top_tier[i] for i in np.random.choice(len(top_tier), args.frontier_k, replace=False)]
        else:
            frontier = sorted(pool, key=lambda d: d.score, reverse=True)[:args.frontier_k]

        tasks = make_tasks(frontier, pars, N, sa_mults)
        n_added = 0
        n_improved = 0
        last_save = time.time()
        pass_start = time.time()

        # Count tasks per depth for visibility
        depth_counts = Counter()
        for _, _, label in tasks:
            # label format: "<strategy>-<sa_mult>x" or "<strategy>-<n>f-<sa_mult>x"
            depth_counts[label.rsplit("-", 1)[1]] += 1
        depth_breakdown = " ".join(f"{k}={v}" for k, v in sorted(depth_counts.items()))

        print(f"\n=== Pass {pass_num} | frontier: {len(frontier)} graphs at {max_score_val} "
              f"(gap={max_possible - max_score_val}) | {len(tasks)} tasks | depths: {depth_breakdown} ===")

        with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_worker_init) as executor:
            futures = {executor.submit(fn, task_args): label for fn, task_args, label in tasks}
            try:
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Pass {pass_num}"):
                    if stop:
                        break
                    label = futures[future]
                    try:
                        dp = future.result()
                    except Exception as e:
                        print(f"  [{label}] Worker error: {e}")
                        continue

                    if dp.features not in seen_features:
                        h = wl_hash(dp.data)
                        if h not in seen_wl:
                            seen_features.add(dp.features)
                            seen_wl.add(h)
                            pool.append(dp)
                            n_added += 1
                            if dp.score > best_score:
                                best_score = dp.score
                                strategy_hits[label] += 1
                                n_improved += 1
                                print(f"\n  *** [{label}] NEW BEST: {dp.score} (gap={max_possible - dp.score}) ***",
                                      flush=True)
                            elif dp.score == max_score_val:
                                strategy_hits[label] += 1

                    if time.time() - last_save >= args.save_interval:
                        pool.sort(key=lambda d: d.score, reverse=True)
                        _save(pool, out_path)
                        last_save = time.time()
            finally:
                # Cancel pending futures and shut down executor cleanly
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)

        elapsed = time.time() - pass_start
        pool.sort(key=lambda d: d.score, reverse=True)
        if args.max_pool_size > 0 and len(pool) > args.max_pool_size:
            dropped = len(pool) - args.max_pool_size
            pool = pool[: args.max_pool_size]
            # Rebuild seen sets so dedup stays consistent with the capped pool.
            # (Otherwise seen sets grow unbounded and block re-discovery of trimmed graphs.)
            seen_features = {d.features for d in pool}
            seen_wl = {wl_hash(d.data) for d in pool}
            print(f"  Capped pool at {args.max_pool_size} (dropped {dropped} lowest-score graphs)")
        print(f"Pass {pass_num} done | {len(tasks)/elapsed:.1f} tasks/s | added: {n_added} | new bests: {n_improved}")
        print_dist(pool[:100], label="Pool top 100")
        if strategy_hits:
            print(f"Strategy hits so far: {dict(strategy_hits.most_common())}")
        _save(pool, out_path)
        print(f"Saved {len(pool)} graphs to {out_path}")

    print(f"\nDone. Final pool: {len(pool)} graphs | best: {best_score} (gap={max_possible - best_score})")
    if strategy_hits:
        print(f"Strategy hits: {dict(strategy_hits.most_common())}")


if __name__ == "__main__":
    main()
