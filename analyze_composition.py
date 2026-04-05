"""
Score-composition breakdown for a Ramsey pool.

Usage:
    python analyze_composition.py --pkl <train_data.pkl> --s 4 --t 6
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
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=True)
    p.add_argument("--s", type=int, default=4, help="size of red clique (K_s)")
    p.add_argument("--t", type=int, default=6, help="size of blue clique (K_t)")
    p.add_argument("--sample", type=int, default=500,
                   help="top-tier graphs to profile")
    args = p.parse_args()

    print(f"Loading {args.pkl} ...")
    data = pickle.load(open(args.pkl, "rb"))
    data.sort(key=lambda d: d.score, reverse=True)
    N = data[0].N
    S, T = args.s, args.t

    # CRITICAL: set class params so clique counting uses the right S,T.
    RamseyDataPoint.S = S
    RamseyDataPoint.T = T
    RamseyDataPoint.N = N

    max_ks = math.comb(N, S)
    max_kt = math.comb(N, T)
    max_total = max_ks + max_kt
    print(f"Loaded {len(data)} graphs | N={N} S={S} T={T}")
    print(f"max = C({N},{S}) + C({N},{T}) = {max_ks} + {max_kt} = {max_total}")

    # ── Best graph ───────────────────────────────────────────────────────────
    best = data[0]
    ks, kt = count_for(best, N, S, T)
    gap = max_total - best.score
    print(f"\n── BEST ── score={best.score}  gap={gap}")
    print(f"  red  K{S} violations = {ks:>5} / {max_ks}")
    print(f"  blue K{T} violations = {kt:>5} / {max_kt}")
    print(f"  gap split: K{S}={ks} ({100*ks/max(gap,1):.1f}% of gap)  "
          f"K{T}={kt} ({100*kt/max(gap,1):.1f}% of gap)")

    # ── Top tier summary ─────────────────────────────────────────────────────
    top_tier = [d for d in data if d.score == best.score]
    sample = top_tier[: min(args.sample, len(top_tier))]
    ks_list, kt_list = [], []
    for d in sample:
        a, b = count_for(d, N, S, T)
        ks_list.append(a); kt_list.append(b)
    ks_arr = np.array(ks_list); kt_arr = np.array(kt_list)

    print(f"\n── TOP TIER ── score={best.score}, {len(top_tier)} graphs "
          f"(profiled {len(sample)})")
    print(f"  K{S} violations: min={ks_arr.min()} max={ks_arr.max()} "
          f"mean={ks_arr.mean():.2f}")
    print(f"  K{T} violations: min={kt_arr.min()} max={kt_arr.max()} "
          f"mean={kt_arr.mean():.2f}")

    pairs = Counter(zip(ks_list, kt_list))
    print(f"  all (K{S},K{T}) splits ({len(pairs)} distinct):")
    for (a, b), c in sorted(pairs.items()):
        print(f"    ({a:>2},{b:>2}): x{c:<5} ({100*c/len(sample):5.1f}%)")

    # ── Score bands near the best ────────────────────────────────────────────
    print(f"\n── SCORE BANDS (top 10) ──")
    bands = Counter(d.score for d in data)
    for score, cnt in sorted(bands.items(), reverse=True)[:10]:
        print(f"  {score}  gap={max_total - score:>4}  x{cnt}")


if __name__ == "__main__":
    main()
