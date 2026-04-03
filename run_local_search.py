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

from tqdm import tqdm

from src.envs.ramsey import RamseyDataPoint


def _parse_time_budget(s):
    """Parse time budget string like '1h', '90m', '3600s', or plain seconds."""
    s = s.strip()
    if s.endswith('h'):
        return float(s[:-1]) * 3600
    if s.endswith('m'):
        return float(s[:-1]) * 60
    if s.endswith('s'):
        return float(s[:-1])
    return float(s)


def _run_ls(dp_and_pars):
    dp, pars, sa_steps, time_limit = dp_and_pars
    RamseyDataPoint._update_class_params(pars)
    dp = copy.deepcopy(dp)
    dp.local_search_fast_v2(sa_steps=sa_steps, time_limit=time_limit)
    return dp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Path to train_data.pkl (read-only)")
    parser.add_argument("--top_k", type=int, default=999999, help="Number of top graphs to run local search on (default: all)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sa_steps", type=int, default=None, help="SA steps per graph (default: N*N*10, ignored when --time_budget is set)")
    parser.add_argument("--time_budget", type=str, default=None, help="Total wall-clock budget e.g. '1h', '90m', '3600s'. Overrides --sa_steps.")
    parser.add_argument("--top_tier", type=int, default=100, help="Top N graphs get extra time (default: 100)")
    parser.add_argument("--top_tier_mult", type=float, default=10.0, help="Multiplier for top-tier time budget (default: 10x)")
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

    # Build per-graph time limits when --time_budget is set.
    # Top-tier graphs (by rank) get top_tier_mult x more time than the rest.
    # Total wall-clock time ≈ budget: sum(limits) / num_workers = budget_s.
    if args.time_budget is not None:
        budget_s = _parse_time_budget(args.time_budget)
        n_top = min(args.top_tier, len(top))
        n_rest = len(top) - n_top
        m = args.top_tier_mult
        # base_time * (n_top * m + n_rest) / workers = budget_s
        base_time = budget_s * args.num_workers / max(1, n_top * m + n_rest)
        top_time = base_time * m
        time_limits = [top_time] * n_top + [base_time] * n_rest
        sa_steps = 10 ** 9  # effectively unlimited — time_limit governs
        print(f"Time budget: {args.time_budget} | top {n_top} graphs: {top_time:.1f}s each | rest: {base_time:.2f}s each")
    else:
        time_limits = [None] * len(top)
        sa_steps = args.sa_steps

    pars = RamseyDataPoint._save_class_params()
    tasks = [(dp, pars, sa_steps, tl) for dp, tl in zip(top, time_limits)]

    # build a lookup from original graph features -> score for before/after comparison
    before_by_rank = [dp.score for dp in top]

    results = []
    seen = set()
    unique_results = []
    n_done = 0
    start = time.time()

    def _report(label):
        unique_results.sort(key=lambda d: d.score, reverse=True)
        elapsed = time.time() - start
        print(f"\n{label} | {n_done}/{len(top)} done | "
              f"{n_done/max(elapsed,1e-9):.1f} graphs/s | "
              f"unique: {len(unique_results)}")
        after_scores = [d.score for d in unique_results[:10]]
        for i, after in enumerate(after_scores):
            b = before_by_rank[i] if i < len(before_by_rank) else "?"
            change = after - b if isinstance(b, int) else "?"
            sign = f"+{change}" if isinstance(change, int) and change >= 0 else str(change)
            print(f"  [{i+1:3d}] {b} → {after}  ({sign})")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_run_ls, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Local search"):
            dp = future.result()
            results.append(dp)
            if dp.features not in seen:
                seen.add(dp.features)
                unique_results.append(dp)
            n_done += 1
            if n_done % args.report_every == 0:
                _report(f"Progress")

    _report("Final")

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
