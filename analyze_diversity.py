"""Analyze structural diversity of a Ramsey graph pool using WL hashing."""
import argparse
import pickle
import time
from collections import Counter

import networkx as nx
import numpy as np
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash


def wl_hash(adj_matrix):
    g = nx.from_numpy_array(adj_matrix)
    return weisfeiler_lehman_graph_hash(g)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl", help="Path to train_data.pkl")
    parser.add_argument("--top_k", type=int, default=0, help="Only analyze top K by score (0 = all)")
    args = parser.parse_args()

    print(f"Loading {args.pkl}...")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} graphs")

    if args.top_k > 0:
        data = sorted(data, key=lambda d: d.score, reverse=True)[:args.top_k]
        print(f"Analyzing top {len(data)} by score")

    scores = [d.score for d in data]
    print(f"Score range: {min(scores)} - {max(scores)} (stdev {np.std(scores):.2f})")

    print(f"\nComputing WL hashes...")
    t0 = time.time()
    hashes = []
    for d in data:
        h = wl_hash(d.data)
        hashes.append(h)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/len(data)*1000:.2f}ms each)")

    hash_counts = Counter(hashes)
    n_unique = len(hash_counts)
    print(f"\n=== Results ===")
    print(f"Total graphs:       {len(data)}")
    print(f"Unique structures:  {n_unique}")
    print(f"Duplicates:         {len(data) - n_unique} ({(len(data) - n_unique) / len(data) * 100:.1f}%)")

    # Group size distribution
    sizes = sorted(hash_counts.values(), reverse=True)
    print(f"\n=== Group sizes (top 20) ===")
    for i, (h, count) in enumerate(hash_counts.most_common(20)):
        # Find score range for this group
        group_scores = [scores[j] for j, hh in enumerate(hashes) if hh == h]
        print(f"  #{i+1}: {count} copies, score={group_scores[0]}")

    print(f"\n=== Group size distribution ===")
    size_dist = Counter(sizes)
    for size in sorted(size_dist.keys()):
        print(f"  {size} copies: {size_dist[size]} groups")

    # Score vs diversity analysis
    print(f"\n=== Score tiers ===")
    score_tiers = {}
    for s, h in zip(scores, hashes):
        tier = int(s)
        if tier not in score_tiers:
            score_tiers[tier] = set()
        score_tiers[tier].add(h)

    for tier in sorted(score_tiers.keys(), reverse=True)[:20]:
        n_in_tier = sum(1 for s in scores if int(s) == tier)
        n_unique_in_tier = len(score_tiers[tier])
        print(f"  Score {tier}: {n_in_tier} graphs, {n_unique_in_tier} unique structures ({n_unique_in_tier/n_in_tier*100:.0f}%)")


if __name__ == "__main__":
    main()
