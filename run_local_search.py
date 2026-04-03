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
        pool.sort(key=lambda d: d.score, reverse=True)
        _save(pool, out_path)
        print(f"\nSaved {len(pool)} graphs to {out_path}. Exiting.")
        os._exit(0)
    signal.signal(signal.SIGINT, _handle_sigint)

    while not stop:
        pass_num += 1
        tasks = [(dp, pars, args.sa_steps) for dp in seeds]
        n_done = 0
        n_added = 0
        last_save = time.time()
        pass_start = time.time()

        print(f"\n=== Pass {pass_num} | pool: {len(pool)} ===")
        for i, dp in enumerate(pool[:10]):
            print(f"  [{i+1:3d}] {dp.score}")

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
                    pool.sort(key=lambda d: d.score, reverse=True)
                    elapsed = time.time() - pass_start
                    print(f"\n  {n_done}/{len(seeds)} | {n_done/elapsed:.1f} g/s | pool: {len(pool)}")
                    for i, dp in enumerate(pool[:10]):
                        print(f"    [{i+1:3d}] {dp.score}")

                if time.time() - last_save >= args.save_interval:
                    pool.sort(key=lambda d: d.score, reverse=True)
                    _save(pool, out_path)
                    last_save = time.time()
                    print(f"  [saved | pool: {len(pool)} | top: {pool[0].score}]")

        pool.sort(key=lambda d: d.score, reverse=True)
        _save(pool, out_path)
        elapsed = time.time() - pass_start
        print(f"\nPass {pass_num} done | added: {n_added} | pool: {len(pool)} | {elapsed:.0f}s")
        for i, dp in enumerate(pool[:10]):
            print(f"  [{i+1:3d}] {dp.score}")

    print(f"\nDone. {pass_num} passes | pool: {len(pool)} | top: {pool[0].score} → {out_path}")


if __name__ == "__main__":
    main()
