"""
Continuous local search. Phase 1: one pass over all graphs. Phase 2: loops
over top --round_size graphs indefinitely, broadening if stuck. Saves every
--save_interval seconds. Ctrl+C stops cleanly after current round.

Usage:
    python run_local_search.py \
        --pkl checkpoint/ramsey_r55_n43_nols1/<exp_id>/train_data.pkl \
        --num_workers 192 --out ls_results.pkl
"""

import argparse
import copy
import os
import pickle
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from src.envs.ramsey import RamseyDataPoint


def _run_ls(dp_and_pars):
    dp, pars, sa_steps = dp_and_pars
    RamseyDataPoint._update_class_params(pars)
    dp = copy.deepcopy(dp)
    dp.local_search_fast_v2(sa_steps=sa_steps)
    return dp


def _save(pool, out_path):
    tmp = out_path + ".tmp"
    pickle.dump(pool, open(tmp, "wb"))
    os.replace(tmp, out_path)


def _run_batch(seeds, num_workers, pars, sa_steps, pool, seen, out_path,
               save_interval, report_every, before_scores, desc):
    """Submit seeds, collect results into pool, save periodically. Returns n_added."""
    tasks = [(dp, pars, sa_steps) for dp in seeds]
    n_done = 0
    n_added = 0
    last_save = [time.time()]
    start = time.time()

    def _maybe_save():
        if time.time() - last_save[0] >= save_interval:
            pool.sort(key=lambda d: d.score, reverse=True)
            _save(pool, out_path)
            last_save[0] = time.time()
            print(f"  [saved {len(pool)} graphs | top: {pool[0].score}]")

    def _report():
        pool.sort(key=lambda d: d.score, reverse=True)
        elapsed = time.time() - start
        print(f"\n  {desc} | {n_done}/{len(seeds)} | "
              f"{n_done/max(elapsed,1e-9):.1f} g/s | pool: {len(pool)}")
        for i, dp in enumerate(pool[:10]):
            b = before_scores[i] if i < len(before_scores) else "?"
            change = dp.score - b if isinstance(b, int) else "?"
            sign = f"+{change}" if isinstance(change, int) and change >= 0 else str(change)
            print(f"    [{i+1:3d}] {b} → {dp.score}  ({sign})")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_run_ls, t): None for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc=desc):
            dp = future.result()
            if dp.features not in seen:
                seen.add(dp.features)
                pool.append(dp)
                n_added += 1
            n_done += 1
            if n_done % report_every == 0:
                _report()
            _maybe_save()

    pool.sort(key=lambda d: d.score, reverse=True)
    _save(pool, out_path)
    last_save[0] = time.time()
    return n_added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Path to train_data.pkl")
    parser.add_argument("--top_k", type=int, default=999999, help="Graphs for initial pass (default: all)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sa_steps", type=int, default=None, help="SA steps per graph (default: N*N*10)")
    parser.add_argument("--round_size", type=int, default=1000, help="Graphs per round in continuous phase (default: 1000)")
    parser.add_argument("--save_interval", type=int, default=60, help="Save every N seconds (default: 60)")
    parser.add_argument("--report_every", type=int, default=5000, help="Print progress every N graphs (default: 5000)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(os.path.dirname(args.pkl), "ls_results.pkl")

    print(f"Loading {args.pkl} ...")
    data = pickle.load(open(args.pkl, "rb"))
    data.sort(key=lambda d: d.score, reverse=True)
    initial = data[:args.top_k]
    print(f"Loaded {len(data)} graphs, initial pass on {len(initial)}")
    print(f"Score range: {initial[-1].score} – {initial[0].score}")
    print(f"Max possible: {RamseyDataPoint.max_possible_score(initial[0].N)}")

    RamseyDataPoint._nb_warmup()
    pars = RamseyDataPoint._save_class_params()

    pool = []
    seen = set()
    stop = False

    def _handle_sigint(sig, frame):
        nonlocal stop
        print("\nCtrl+C — finishing current round then saving...")
        stop = True
    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Phase 1: initial pass ────────────────────────────────────────────────
    print("\n=== Phase 1: initial pass ===")
    before_scores = [dp.score for dp in initial]
    n_added = _run_batch(initial, args.num_workers, pars, args.sa_steps,
                         pool, seen, out_path, args.save_interval,
                         args.report_every, before_scores, "Phase 1")
    print(f"\nPhase 1 done: {len(pool)} unique | top: {pool[0].score} | added: {n_added}")

    # ── Phase 2: continuous rounds ───────────────────────────────────────────
    round_num = 0
    stale_rounds = 0  # rounds since top score improved

    while not stop:
        round_num += 1
        top_score_before = pool[0].score

        # Broaden search after stale rounds: 1000, 2000, 4000, ... capped at pool size
        effective_size = min(args.round_size * (2 ** stale_rounds), len(pool))
        seeds = pool[:effective_size]

        print(f"\n=== Round {round_num} | top {effective_size} graphs | best: {pool[0].score} "
              f"| stale: {stale_rounds} ===")

        before_scores = [dp.score for dp in pool[:10]]
        _run_batch(seeds, args.num_workers, pars, args.sa_steps,
                   pool, seen, out_path, args.save_interval,
                   args.report_every, before_scores, f"Round {round_num}")

        if pool[0].score > top_score_before:
            print(f"  Top score improved: {top_score_before} → {pool[0].score}")
            stale_rounds = 0
        else:
            stale_rounds += 1
            print(f"  No top-score improvement ({stale_rounds} stale rounds, next: top {min(args.round_size * (2**(stale_rounds)), len(pool))})")

    print(f"\nFinal pool: {len(pool)} unique graphs | top: {pool[0].score} → {out_path}")

    # Write top 10 DOT files
    dot_dir = os.path.join(os.path.dirname(out_path), "ls_top_graphs")
    os.makedirs(dot_dir, exist_ok=True)
    for rank, dp in enumerate(pool[:10], 1):
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
