"""
Continuous local search. Repeatedly runs one pass over all graphs, saving every
--save_interval seconds. Ctrl+C stops cleanly after the current pass.

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--top_k", type=int, default=999999)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sa_steps", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=60)
    parser.add_argument("--report_every", type=int, default=5000)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(os.path.dirname(args.pkl), "ls_results.pkl")

    print(f"Loading {args.pkl} ...")
    data = pickle.load(open(args.pkl, "rb"))
    data.sort(key=lambda d: d.score, reverse=True)
    seeds = data[:args.top_k]
    print(f"Loaded {len(data)} graphs, searching top {len(seeds)}")
    print(f"Score range: {seeds[-1].score} – {seeds[0].score}")
    print(f"Max possible: {RamseyDataPoint.max_possible_score(seeds[0].N)}")

    RamseyDataPoint._nb_warmup()
    pars = RamseyDataPoint._save_class_params()

    pool = []
    seen = set()
    stop = False
    pass_num = 0
    start_total = time.time()

    def _handle_sigint(sig, frame):
        if os.getpid() != main_pid:
            return  # ignore in worker processes
        pool.sort(key=lambda d: d.score, reverse=True)
        if pool:
            _save(pool, out_path)
            print(f"\nSaved {len(pool)} graphs to {out_path}. Exiting.")
        else:
            print("\nNo results yet. Exiting without saving.")
        os._exit(0)
    main_pid = os.getpid()
    signal.signal(signal.SIGINT, _handle_sigint)

    while not stop:
        pass_num += 1
        current_seeds = pool if pool else seeds
        current_seeds.sort(key=lambda d: d.score, reverse=True)
        tasks = [(dp, pars, args.sa_steps) for dp in current_seeds]
        n_done = 0
        n_added = 0
        last_save = time.time()
        pass_start = time.time()

        before_scores = sorted([dp.score for dp in current_seeds], reverse=True)

        def _report(label):
            pool.sort(key=lambda d: d.score, reverse=True)
            elapsed = time.time() - pass_start
            print(f"\n{label} | {n_done}/{len(current_seeds)} | {n_done/max(elapsed,1e-9):.1f} g/s | pool: {len(pool)}")
            for i, dp in enumerate(pool[:10]):
                b = before_scores[i] if i < len(before_scores) else "?"
                change = dp.score - b if isinstance(b, int) else "?"
                sign = f"+{change}" if isinstance(change, int) and change >= 0 else str(change)
                print(f"  [{i+1:3d}] {b} → {dp.score}  ({sign})")

        print(f"\n=== Pass {pass_num} | pool: {len(pool)} | top seed: {current_seeds[0].score} ===")

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
                n_done += 1

                if n_done % args.report_every == 0:
                    _report("Progress")

                if time.time() - last_save >= args.save_interval:
                    pool.sort(key=lambda d: d.score, reverse=True)
                    _save(pool, out_path)
                    last_save = time.time()
                    print(f"  [saved | pool: {len(pool)} | top: {pool[0].score}]")

        _report(f"Pass {pass_num} done | added: {n_added}")
        _save(pool, out_path)

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

    print(f"\nDone. {pass_num} passes | pool: {len(pool)} | top: {pool[0].score} → {out_path}")


if __name__ == "__main__":
    main()
