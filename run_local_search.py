"""
Run fast local search on the top-K graphs from a saved train_data.pkl.
Safe to run alongside an active data generation job — reads pkl read-only,
writes results to a separate file.

Usage:
    python run_local_search.py --pkl checkpoint/ramsey_r55_n43_nols1/<exp_id>/train_data.pkl \
        --top_k 1000 --num_workers 192 --out results_after_ls.pkl
"""

import argparse
import copy
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from src.envs.ramsey import RamseyDataPoint, _NUMBA, _nb_count_ks_cliques


def _run_ls(dp_and_pars):
    dp, pars, sa_steps = dp_and_pars
    RamseyDataPoint._update_class_params(pars)
    dp = copy.deepcopy(dp)
    dp.local_search_fast(sa_steps=sa_steps)
    return dp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Path to train_data.pkl (read-only)")
    parser.add_argument("--top_k", type=int, default=1000, help="Number of top graphs to run local search on")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sa_steps", type=int, default=None, help="SA steps per graph (default: N*N*10)")
    parser.add_argument("--out", type=str, default=None, help="Output pkl path (default: <pkl_dir>/ls_results.pkl)")
    args = parser.parse_args()

    out_path = args.out or os.path.join(os.path.dirname(args.pkl), "ls_results.pkl")

    print(f"Loading {args.pkl} ...")
    data = pickle.load(open(args.pkl, "rb"))
    data.sort(key=lambda d: d.score, reverse=True)
    top = data[:args.top_k]
    print(f"Loaded {len(data)} graphs, running local search on top {len(top)}")
    print(f"Score range before: {top[-1].score} – {top[0].score}")

    max_score = RamseyDataPoint.max_possible_score(top[0].N)
    print(f"Max possible score: {max_score}")

    # Warmup numba in main process so forked workers inherit compiled code
    RamseyDataPoint._nb_warmup()

    pars = RamseyDataPoint._save_class_params()
    tasks = [(dp, pars, args.sa_steps) for dp in top]

    results = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_run_ls, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Local search"):
            results.append(future.result())

    elapsed = time.time() - start
    results.sort(key=lambda d: d.score, reverse=True)

    before_scores = sorted([dp.score for dp in top], reverse=True)
    after_scores = sorted([dp.score for dp in results], reverse=True)

    print(f"\nDone in {elapsed:.1f}s ({len(results)/elapsed:.1f} graphs/s)")
    print(f"\nTop 10 before → after:")
    for i in range(min(10, len(results))):
        change = after_scores[i] - before_scores[i]
        sign = f"+{change}" if change >= 0 else str(change)
        print(f"  [{i+1:3d}] {before_scores[i]} → {after_scores[i]}  ({sign})")

    print(f"\nSaving {len(results)} results to {out_path}")
    pickle.dump(results, open(out_path, "wb"))

    # Write top 10 DOT files
    dot_dir = os.path.join(os.path.dirname(out_path), "ls_top_graphs")
    os.makedirs(dot_dir, exist_ok=True)
    for rank, dp in enumerate(results[:10], 1):
        path = os.path.join(dot_dir, f"rank_{rank:02d}_score_{dp.score}.dot")
        n = dp.N
        lines = [f"graph rank{rank} {{", f'  label="rank {rank} | score {dp.score}";']
        for i in range(n):
            for j in range(i + 1, n):
                if dp.data[i, j]:
                    lines.append(f"  {i} -- {j};")
        lines.append("}")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
    print(f"Top 10 DOT files written to {dot_dir}/")


if __name__ == "__main__":
    main()
