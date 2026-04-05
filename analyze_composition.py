"""
Score-composition breakdown for a Ramsey pool.

Usage:
    python analyze_composition.py --pkl checkpoint/r46_n36/r46_n36/<exp_id>/train_data.pkl
"""

import argparse
import math
import pickle
from collections import Counter

import numpy as np

from src.envs.ramsey import (
    RamseyDataPoint,
    _NUMBA,
    _nb_count_ks_cliques,
    count_ks_cliques_bitmask,
)


def count_for(dp, N, S, T):
    dp._sync_from_data()
    if _NUMBA:
        adj = np.array(dp.adj, dtype=np.int64)
        cadj = np.array(dp.cadj, dtype=np.int64)
        ks = int(_nb_count_ks_cliques(adj, N, S))
        kt = int(_nb_count_ks_cliques(cadj, N, T))
    else:
        ks = count_ks_cliques_bitmask(dp.adj, N, S)
        kt = count_ks_cliques_bitmask(dp.cadj, N, T)
    return ks, kt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--sample", type=int, default=500,
                        help="how many top-tier graphs to profile for the distribution")
    args = parser.parse_args()

    print(f"Loading {args.pkl} ...")
    data = pickle.load(open(args.pkl, "rb"))
    data.sort(key=lambda d: d.score, reverse=True)
    print(f"Loaded {len(data)} graphs")

    N = data[0].N
    S, T = RamseyDataPoint.S, RamseyDataPoint.T
    max_ks = math.comb(N, S)
    max_kt = math.comb(N, T)
    max_total = max_ks + max_kt

    print(f"\nN={N}  S={S}  T={T}")
    print(f"max K{S} count  = C({N},{S}) = {max_ks}")
    print(f"max K{T} count  = C({N},{T}) = {max_kt}")
    print(f"max total score = {max_total}")

    # Best graph
    best = data[0]
    ks, kt = count_for(best, N, S, T)
    print(f"\n── BEST graph  (score={best.score}, gap={max_total - best.score}) ──")
    print(f"  red  K{S} violations  = {ks:>6}  /  {max_ks} max")
    print(f"  blue K{T} violations  = {kt:>6}  /  {max_kt} max")
    print(f"  score from non-K{S}   = {max_ks - ks:>6}  "
          f"({100*(max_ks - ks)/max_total:5.2f}% of max, "
          f"{100*(max_ks - ks)/best.score:5.2f}% of actual score)")
    print(f"  score from non-K{T}   = {max_kt - kt:>6}  "
          f"({100*(max_kt - kt)/max_total:5.2f}% of max, "
          f"{100*(max_kt - kt)/best.score:5.2f}% of actual score)")
    print(f"  violations ratio       K{S}:K{T} = {ks}:{kt}")

    # Distribution over top tier
    top_tier = [d for d in data if d.score == best.score]
    print(f"\n── Top-tier analysis  (score={best.score}) ──")
    print(f"Top-tier has {len(top_tier)} graphs")

    sample = top_tier[: min(args.sample, len(top_tier))]
    ks_list, kt_list = [], []
    for d in sample:
        a, b = count_for(d, N, S, T)
        ks_list.append(a)
        kt_list.append(b)

    print(f"\nOver {len(sample)} top-tier graphs:")
    print(f"  K{S} violations:  min={min(ks_list)}  max={max(ks_list)}  "
          f"mean={np.mean(ks_list):.2f}")
    print(f"  K{T} violations:  min={min(kt_list)}  max={max(kt_list)}  "
          f"mean={np.mean(kt_list):.2f}")

    pairs = Counter(zip(ks_list, kt_list))
    print(f"\n  (K{S}, K{T}) distribution:")
    for (a, b), c in sorted(pairs.items()):
        pct = 100.0 * c / len(sample)
        print(f"    ({a:>3}, {b:>3}):  x{c:<5}  {pct:5.1f}%")

    # Broader picture across score bands
    print(f"\n── Score bands ──")
    band_counts = Counter(d.score for d in data)
    for score, cnt in sorted(band_counts.items(), reverse=True)[:15]:
        print(f"  {score}  (gap={max_total - score}):  x{cnt}")


if __name__ == "__main__":
    main()
