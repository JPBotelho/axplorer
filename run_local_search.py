"""
Run fast local search on the top-K graphs from a saved train_data.pkl.
Safe to run alongside an active data generation job — reads pkl read-only,
writes results to a separate file.

Usage:
    python run_local_search.py \
        --pkl checkpoint/ramsey_r55_n43_nols1/<exp_id>/train_data.pkl \
        --top_k 999999 --num_workers 192 --out ls_results.pkl
"""

import argparse
import copy
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.envs.ramsey import RamseyDataPoint


def _run_ls(dp_and_pars):
    dp, pars, sa_steps = dp_and_pars
    RamseyDataPoint._update_class_params(pars)
    dp = copy.deepcopy(dp)
    dp.local_search_fast(sa_steps=sa_steps)
    return dp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Path to train_data.pkl (read-only)")
    parser.add_argument("--top_k", type=int, default=999999, help="Number of top graphs to run local search on (default: all)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sa_steps", type=int, default=None, help="SA steps per graph (default: N*N*10)")
    parser.add_argument("--report_every", type=int, default=1000, help="Print progress every N completed graphs")
    parser.add_argument("--out", type=str, default=None, help="Output pkl path (default: <pkl_dir>/ls_results.pkl)")
    args = parser.parse_args()

    out_path = args.out or os.path.join(os.path.dirname(args.pkl), "ls_results.pkl")

    print(f"Loading {args.pkl} ...")
    data = pickle.load(open(args.pkl, "rb"))
    data.sort(key=lambda d: d.score, reverse=True)
    top = data[:args.top_k]
    print(f"Loaded {len(data)} graphs, running local search on {len(top)}")
    print(f"Score range before: {top[-1].score} – {top[0].score}")
    print(f"Max possible score: {RamseyDataPoint.max_possible_score(top[0].N)}")

    RamseyDataPoint._nb_warmup()

    pars = RamseyDataPoint._save_class_params()
    tasks = [(dp, pars, args.sa_steps) for dp in top]

    results = []
    seen = set()
    unique_results = []
    best_score = None
    n_done = 0
    start = time.time()

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_run_ls, t): t for t in tasks}
        for future in as_completed(futures):
            dp = future.result()
            results.append(dp)
            if dp.features not in seen:
                seen.add(dp.features)
                unique_results.append(dp)
            if best_score is None or dp.score > best_score:
                best_score = dp.score
            n_done += 1
            if n_done % args.report_every == 0:
                elapsed = time.time() - start
                print(f"  [{n_done}/{len(top)}] "
                      f"{n_done/elapsed:.1f} graphs/s | "
                      f"unique so far: {len(unique_results)} | "
                      f"best score: {best_score}")

    elapsed = time.time() - start
    unique_results.sort(key=lambda d: d.score, reverse=True)
    before_scores = [dp.score for dp in top]

    print(f"\nDone in {elapsed:.1f}s ({len(top)/elapsed:.1f} graphs/s)")
    print(f"Starting graphs: {len(top)} → unique post-LS results: {len(unique_results)}")
    print(f"\nTop 10 before → after:")
    for i in range(min(10, len(unique_results))):
        b = before_scores[i] if i < len(before_scores) else "?"
        after = unique_results[i].score
        change = after - b if isinstance(b, int) else "?"
        sign = f"+{change}" if isinstance(change, int) and change >= 0 else str(change)
        print(f"  [{i+1:3d}] {b} → {after}  ({sign})")

    print(f"\nSaving {len(unique_results)} unique results to {out_path}")
    pickle.dump(unique_results, open(out_path, "wb"))

    # Write top 10 DOT files
    dot_dir = os.path.join(os.path.dirname(out_path), "ls_top_graphs")
    os.makedirs(dot_dir, exist_ok=True)
    for rank, dp in enumerate(unique_results[:10], 1):
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
