"""
Benchmark local_search_fast (full recompute) vs local_search_fast_v2 (delta-based).
Verifies correctness (greedy phase is deterministic → exact score match) and measures speedup.

Usage:
    python benchmark_ls.py --pkl <path/to/train_data.pkl> --n 50
"""
import argparse
import copy
import pickle
import time

import numpy as np

from src.envs.ramsey import RamseyDataPoint, _NUMBA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--n", type=int, default=50, help="Number of graphs to benchmark")
    parser.add_argument("--sa_steps", type=int, default=None)
    args = parser.parse_args()

    print(f"Numba available: {_NUMBA}")
    print(f"Loading {args.pkl} ...")
    data = pickle.load(open(args.pkl, "rb"))
    data.sort(key=lambda d: d.score, reverse=True)
    sample = data[: args.n]
    print(f"Benchmarking on top {len(sample)} graphs (score range: {sample[-1].score}–{sample[0].score})")
    print(f"Max possible score: {RamseyDataPoint.max_possible_score(sample[0].N)}")

    print("\nWarming up numba JIT...")
    RamseyDataPoint._nb_warmup()
    print("Done.\n")

    # ── Correctness check: greedy phase only (deterministic) ──────────────────
    print("=== Correctness check (greedy phase only, sa_steps=0) ===")
    greedy_v1, greedy_v2 = [], []

    for dp in sample[:10]:
        c1 = copy.deepcopy(dp)
        c1.local_search_fast(sa_steps=0)
        greedy_v1.append(c1.score)

        c2 = copy.deepcopy(dp)
        c2.local_search_fast_v2(sa_steps=0)
        greedy_v2.append(c2.score)

    mismatches = [(i, greedy_v1[i], greedy_v2[i]) for i in range(len(greedy_v1)) if greedy_v1[i] != greedy_v2[i]]
    if mismatches:
        print(f"MISMATCH on {len(mismatches)} graphs!")
        for i, s1, s2 in mismatches:
            print(f"  graph {i}: v1={s1}, v2={s2}")
    else:
        print(f"  All {len(greedy_v1)} greedy scores match exactly. ✓")

    # ── Speed benchmark: v1 ───────────────────────────────────────────────────
    print(f"\n=== Speed benchmark (sa_steps={args.sa_steps or 'default'}) ===")
    copies_v1 = [copy.deepcopy(dp) for dp in sample]
    t0 = time.time()
    for dp in copies_v1:
        dp.local_search_fast(sa_steps=args.sa_steps)
    t1 = time.time()
    scores_v1 = [dp.score for dp in copies_v1]
    elapsed_v1 = t1 - t0

    # ── Speed benchmark: v2 ───────────────────────────────────────────────────
    copies_v2 = [copy.deepcopy(dp) for dp in sample]
    t0 = time.time()
    for dp in copies_v2:
        dp.local_search_fast_v2(sa_steps=args.sa_steps)
    t1 = time.time()
    scores_v2 = [dp.score for dp in copies_v2]
    elapsed_v2 = t1 - t0

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n  v1 (full recompute): {elapsed_v1:.2f}s | {len(sample)/elapsed_v1:.1f} graphs/s")
    print(f"  v2 (delta-based):    {elapsed_v2:.2f}s | {len(sample)/elapsed_v2:.1f} graphs/s")
    print(f"  Speedup: {elapsed_v1/elapsed_v2:.1f}x")

    print(f"\n  v1 scores — mean: {np.mean(scores_v1):.1f}, max: {max(scores_v1)}, min: {min(scores_v1)}")
    print(f"  v2 scores — mean: {np.mean(scores_v2):.1f}, max: {max(scores_v2)}, min: {min(scores_v2)}")

    print(f"\n  Per-graph comparison (first 10):")
    print(f"  {'#':>3}  {'start':>10}  {'v1 final':>10}  {'v2 final':>10}  {'diff':>6}")
    for i in range(min(10, len(sample))):
        diff = scores_v2[i] - scores_v1[i]
        sign = f"+{diff}" if diff >= 0 else str(diff)
        print(f"  {i+1:>3}  {sample[i].score:>10}  {scores_v1[i]:>10}  {scores_v2[i]:>10}  {sign:>6}")


if __name__ == "__main__":
    main()
