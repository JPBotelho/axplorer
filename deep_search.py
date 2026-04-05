"""
Deduplicate a pool (by WL hash) and run continuous deep local search on the top-k graphs.
Saves results back to the same pkl (or a separate --out path).
Ctrl+C saves and exits cleanly.

Usage:
    python deep_search.py \
        --pkl checkpoint/r46_n36/r46_n36/<exp_id>/train_data.pkl \
        --num_workers 32 \
        --top_k 1000 \
        --sa_mult 100
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


def _run_ls(dp_and_pars):
    dp, pars, sa_steps = dp_and_pars
    RamseyDataPoint._update_class_params(pars)
    dp = copy.deepcopy(dp)
    dp.local_search_fast_v2(sa_steps=sa_steps)
    return dp


def wl_hash(data_matrix):
    import networkx as nx
    from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
    g = nx.from_numpy_array(data_matrix)
    return weisfeiler_lehman_graph_hash(g)


def wl_dedup(data):
    seen = set()
    out = []
    for d in tqdm(data, desc="WL dedup"):
        h = wl_hash(d.data)
        if h not in seen:
            seen.add(h)
            out.append(d)
    return out


def _save(pool, out_path):
    tmp = out_path + ".tmp"
    pickle.dump(pool, open(tmp, "wb"))
    os.replace(tmp, out_path)


def print_dist(data, label="Pool", top_n=15):
    scores = [d.score for d in data]
    if not scores:
        print(f"{label}: empty")
        return
    counts = Counter(scores)
    top = sorted(counts.items(), reverse=True)[:top_n]
    print(f"\n{label}: {len(scores)} graphs | max={max(scores)} | mean={sum(scores)/len(scores):.1f}")
    for score, count in top:
        print(f"  Score {score}: Count {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Path to train_data.pkl")
    parser.add_argument("--out", type=str, default=None, help="Output pkl path (default: overwrites input)")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=1000, help="Search top-k graphs by score")
    parser.add_argument("--sa_mult", type=int, default=100, help="SA steps = N^2 * sa_mult")
    parser.add_argument("--save_interval", type=int, default=60, help="Save every N seconds")
    parser.add_argument("--no_wl_dedup", action="store_true", help="Skip WL deduplication")
    args = parser.parse_args()

    out_path = args.out or args.pkl

    print(f"Loading {args.pkl} ...")
    data = pickle.load(open(args.pkl, "rb"))
    data.sort(key=lambda d: d.score, reverse=True)
    print(f"Loaded {len(data)} graphs")
    print_dist(data[:50], label="Input (top 50)")

    # Step 1: WL dedup
    if not args.no_wl_dedup:
        print(f"\nDeduplicating {len(data)} graphs by WL hash...")
        before = len(data)
        data = wl_dedup(data)
        print(f"Removed {before - len(data)} isomorphic duplicates → {len(data)} unique graphs")
        print_dist(data[:50], label="After dedup (top 50)")
        _save(data, out_path)
        print(f"Saved deduped pool to {out_path}")

    # Step 2: select top-k for search
    seeds = data[:args.top_k]
    N = seeds[0].N
    sa_steps = N * N * args.sa_mult
    print(f"\nRunning deep LS on top {len(seeds)} graphs | N={N} | sa_steps={sa_steps} | workers={args.num_workers}")

    RamseyDataPoint._nb_warmup()
    pars = RamseyDataPoint._save_class_params()
    max_possible = RamseyDataPoint.max_possible_score(N)
    print(f"Max possible score: {max_possible}")

    pool = list(data)  # start with full deduped pool, add improvements
    seen = {d.features for d in pool}
    best_score = max(d.score for d in pool)

    stop = False
    pass_num = 0
    main_pid = os.getpid()

    def _handle_sigint(sig, frame):
        if os.getpid() != main_pid:
            return
        pool.sort(key=lambda d: d.score, reverse=True)
        _save(pool, out_path)
        print(f"\nSaved {len(pool)} graphs to {out_path}. Exiting.")
        os._exit(0)
    signal.signal(signal.SIGINT, _handle_sigint)

    while not stop:
        pass_num += 1
        current_seeds = sorted(pool, key=lambda d: d.score, reverse=True)[:args.top_k]
        tasks = [(copy.deepcopy(dp), pars, sa_steps) for dp in current_seeds]
        n_added = 0
        last_save = time.time()
        pass_start = time.time()

        print(f"\n=== Pass {pass_num} | pool: {len(pool)} | top: {current_seeds[0].score} (gap={max_possible - current_seeds[0].score}) ===")

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(_run_ls, t): None for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Pass {pass_num}"):
                try:
                    dp = future.result()
                except Exception as e:
                    print(f"  Worker error: {e}")
                    continue

                if dp.features not in seen:
                    seen.add(dp.features)
                    pool.append(dp)
                    n_added += 1
                    if dp.score > best_score:
                        best_score = dp.score
                        print(f"  *** NEW BEST: {dp.score} (gap={max_possible - dp.score}) ***", flush=True)

                if time.time() - last_save >= args.save_interval:
                    pool.sort(key=lambda d: d.score, reverse=True)
                    _save(pool, out_path)
                    last_save = time.time()

        elapsed = time.time() - pass_start
        pool.sort(key=lambda d: d.score, reverse=True)
        print(f"Pass {pass_num} done | added: {n_added} | {len(tasks)/elapsed:.1f} g/s")
        print_dist(pool[:50], label="Pool (top 50)")
        _save(pool, out_path)
        print(f"Saved {len(pool)} graphs to {out_path}")


if __name__ == "__main__":
    main()
