import argparse
import copy
import itertools
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger

import numpy as np
import torch
from tqdm import tqdm

from src.datasets import CharDataset, InfiniteDataLoader, load_initial_data, update_datasets
from src.envs import ENVS, build_env
from src.envs.environment import do_stats
from src.evaluator import sample_and_score
from src.models.model import Transformer
from src.trainer import reload_model_optimizer, train
from src.utils import bool_flag, force_release_memory, initialize_exp, log_resources, write_important_metrics

logger = getLogger()


def _run_ls(dp_and_pars):
    """Worker function for background local search."""
    dp, pars, sa_steps = dp_and_pars
    dp.__class__._update_class_params(pars)
    dp = copy.deepcopy(dp)
    dp.local_search_fast_v2(sa_steps=sa_steps)
    return dp


def _kill_executor(executor):
    """Shut down executor quickly without leaving zombies."""
    import signal
    # Prevent deadlock: cancel the internal feeder thread join so
    # shutdown doesn't block on a pipe write to dead workers.
    try:
        executor._call_queue.cancel_join_thread()
    except Exception:
        pass
    # SIGTERM first for clean shutdown
    for p in list(executor._processes.values()):
        if p.is_alive():
            try:
                p.terminate()
            except OSError:
                pass
    # Wait briefly, then SIGKILL stragglers and reap zombies
    deadline = time.monotonic() + 1.0
    for p in list(executor._processes.values()):
        remaining = max(0.01, deadline - time.monotonic())
        p.join(timeout=remaining)
        if p.is_alive():
            try:
                os.kill(p.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            p.join(timeout=1.0)
    executor.shutdown(wait=False, cancel_futures=True)


def run_background_cpu_work(classname, pool, args, stop_event, max_score=None):
    """
    Run random generation and local search on CPU while GPU trains.
    Returns (generated_graphs, ls_improved_graphs).
    """
    generated = []
    ls_results = []
    seen_features = {d.features for d in pool}
    data_lock = threading.Lock()  # protects seen_features, generated, ls_results, top10_scores

    # Compute score thresholds for alerts (mutable so they update as better graphs are found)
    pool_scores = sorted([d.score for d in pool], reverse=True)
    top10_scores = pool_scores[:10] if len(pool_scores) >= 10 else pool_scores[:]
    if not top10_scores:
        top10_scores = [0]

    # Use at most 50% of cores for background work to avoid starving GPU training
    max_bg_workers = max(1, args.num_workers // 2)
    if args.bg_generation and args.bg_local_search:
        n_workers_gen = args.bg_workers_gen or max_bg_workers // 2
        n_workers_ls = args.bg_workers_ls or max_bg_workers - n_workers_gen
    elif args.bg_generation:
        n_workers_gen = args.bg_workers_gen or max_bg_workers
        n_workers_ls = 0
    else:
        n_workers_gen = 0
        n_workers_ls = args.bg_workers_ls or max_bg_workers
    # Reserve 4 cores for elite search if LS is enabled
    if args.bg_local_search and n_workers_ls > 4:
        n_workers_ls -= 4
    logger.info(f"[BG] Using {n_workers_gen} gen + {n_workers_ls} LS + 4 elite workers (of {args.num_workers} total)")

    def _run_generation():
        if not args.bg_generation or n_workers_gen < 1:
            return
        pars = classname._save_class_params()
        batch_size = 1000
        n_gen = 0
        executor = ProcessPoolExecutor(max_workers=n_workers_gen)
        try:
            pending = set()
            for _ in range(n_workers_gen * 2):
                if stop_event.is_set():
                    break
                f = executor.submit(classname._batch_generate_and_score, batch_size, args.N, pars, args.per_batch_top_k)
                pending.add(f)

            while pending and not stop_event.is_set():
                done, pending = _wait_any(pending, timeout=0.5)
                for future in done:
                    try:
                        chunk = future.result()
                    except Exception as e:
                        print(f"[BG-GEN] Worker error: {e}", file=sys.stderr)
                        continue
                    if chunk:
                        with data_lock:
                            for dp in chunk:
                                if dp.features not in seen_features:
                                    seen_features.add(dp.features)
                                    generated.append(dp)
                                    if dp.score > top10_scores[-1]:
                                        top10_scores.append(dp.score)
                                        top10_scores.sort(reverse=True)
                                        del top10_scores[10:]
                                        print(f"[BG-GEN] NEW TOP-10! score={dp.score} (top10 min={top10_scores[-1]})")
                    n_gen += batch_size
                    if not stop_event.is_set():
                        try:
                            f = executor.submit(classname._batch_generate_and_score, batch_size, args.N, pars, args.per_batch_top_k)
                            pending.add(f)
                        except RuntimeError:
                            break
        finally:
            _kill_executor(executor)
        logger.info(f"[BG-GEN] Finished: {n_gen} generated, {len(generated)} unique kept")

    def _run_local_search():
        if not args.bg_local_search or n_workers_ls < 1:
            return
        pars = classname._save_class_params()
        sa_steps = args.N * args.N * args.ls_sa_mult
        n_done = 0
        n_pass = 0

        executor = ProcessPoolExecutor(max_workers=n_workers_ls)
        try:
            while not stop_event.is_set():
                n_pass += 1
                with data_lock:
                    all_candidates = list(pool) + list(ls_results)
                # If all graphs are already at max score, nothing to improve
                if max_score is not None and all(d.score >= max_score for d in all_candidates):
                    logger.info(f"[BG-LS] All graphs at max score, stopping.")
                    break
                top_pool = sorted(all_candidates, key=lambda d: d.score, reverse=True)[:min(len(all_candidates), 5000)]
                tasks = [(copy.deepcopy(dp), pars, sa_steps) for dp in top_pool]

                # Submit incrementally to avoid filling the internal queue
                task_iter = iter(tasks)
                pending = set()
                for t in itertools.islice(task_iter, n_workers_ls * 2):
                    pending.add(executor.submit(_run_ls, t))

                while pending and not stop_event.is_set():
                    done, pending = _wait_any(pending, timeout=0.5)
                    for future in done:
                        try:
                            dp = future.result()
                        except Exception as e:
                            print(f"[BG-LS] Worker error: {e}", file=sys.stderr)
                            continue
                        with data_lock:
                            if dp.features not in seen_features:
                                seen_features.add(dp.features)
                                ls_results.append(dp)
                                if dp.score > top10_scores[-1]:
                                    top10_scores.append(dp.score)
                                    top10_scores.sort(reverse=True)
                                    del top10_scores[10:]
                                    print(f"[BG-LS]  NEW TOP-10! score={dp.score} (top10 min={top10_scores[-1]})")
                                elif dp.score >= top10_scores[0] - 5:
                                    print(f"[BG-LS]  Near-top graph: score={dp.score} (best={top10_scores[0]})")
                        n_done += 1
                        # Submit next task
                        t = next(task_iter, None)
                        if t is not None and not stop_event.is_set():
                            pending.add(executor.submit(_run_ls, t))

                logger.info(f"[BG-LS] Pass {n_pass} done: {n_done} total searched, {len(ls_results)} unique improved")
        finally:
            _kill_executor(executor)

        logger.info(f"[BG-LS] Finished: {n_pass} passes, {n_done} searched, {len(ls_results)} unique improved")

    def _pick_elite_task(pars, sa_steps):
        """Pick a random graph from the current top-5 distinct scores."""
        with data_lock:
            all_candidates = list(pool) + list(ls_results)
        if not all_candidates:
            return None
        elite = sorted(all_candidates, key=lambda d: d.score, reverse=True)[:100]
        dp = copy.deepcopy(elite[np.random.randint(len(elite))])
        return (dp, pars, sa_steps)

    def _run_elite_search():
        """Dedicate 4 cores to continuously searching the current top-5 scores (dynamic)."""
        pars = classname._save_class_params()
        sa_steps = args.N * args.N * args.ls_sa_mult * 10  # 10x more SA effort for elite
        n_done = 0
        n_elite_workers = 4

        executor = ProcessPoolExecutor(max_workers=n_elite_workers)
        try:
            pending = set()
            # Fill initial slots
            for _ in range(n_elite_workers * 2):
                if stop_event.is_set():
                    break
                t = _pick_elite_task(pars, sa_steps)
                if t is not None:
                    pending.add(executor.submit(_run_ls, t))

            while pending and not stop_event.is_set():
                done, pending = _wait_any(pending, timeout=0.5)
                for future in done:
                    try:
                        dp = future.result()
                    except Exception as e:
                        print(f"[BG-ELITE] Worker error: {e}", file=sys.stderr)
                        continue
                    with data_lock:
                        if dp.features not in seen_features:
                            seen_features.add(dp.features)
                            ls_results.append(dp)
                            if dp.score > top10_scores[-1]:
                                top10_scores.append(dp.score)
                                top10_scores.sort(reverse=True)
                                del top10_scores[10:]
                                print(f"[BG-ELITE] NEW TOP-10! score={dp.score} (top10 min={top10_scores[-1]})")
                    n_done += 1
                    # Immediately re-pick from current top-100 (includes own discoveries)
                    if not stop_event.is_set():
                        t = _pick_elite_task(pars, sa_steps)
                        if t is not None:
                            pending.add(executor.submit(_run_ls, t))
        finally:
            _kill_executor(executor)
        logger.info(f"[BG-ELITE] Finished: {n_done} total searched")

    # Run generation and LS in separate threads (each manages its own ProcessPoolExecutor)
    threads = []
    if args.bg_generation:
        t = threading.Thread(target=_run_generation, name="bg-gen")
        threads.append(t)
    if args.bg_local_search:
        # Reserve 4 cores for elite search from the LS allocation
        t = threading.Thread(target=_run_elite_search, name="bg-elite")
        threads.append(t)
        t = threading.Thread(target=_run_local_search, name="bg-ls")
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return generated, ls_results


def _wait_any(futures, timeout=0.5):
    """Wait for at least one future to complete, with timeout for stop_event checking."""
    import concurrent.futures
    done, not_done = concurrent.futures.wait(futures, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED)
    return done, not_done


def get_parser():
    parser = argparse.ArgumentParser("A simple Axplorer loop for different maths problems")

    parser.add_argument("--gensize", type=int, default=100000, help="Number of generate initial values")
    parser.add_argument("--max_epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=50000, help="number of training steps.")
    parser.add_argument("--num_samples_from_model", type=int, default=500000, help="sample the specified number from the model in each loop")
    parser.add_argument("--pop_size", type=int, default=200000, help="Total maximum number of examples at each epoch")
    parser.add_argument("--ntest", type=int, default=1000, help="Size of test set")
    parser.add_argument("--env_name", type=str, default="square", help="Math problem to be addressed")
    ENVS[parser.parse_known_args()[0].env_name].register_args(parser)

    parser.add_argument("--process_pool", type=bool_flag, default="true", help="use process_pool to generate and score initial data")
    parser.add_argument("--always_search", type=bool_flag, default="true", help="if True, use local search for all examples generated")
    parser.add_argument("--redeem_only", type=bool_flag, default="false", help="if True, save invalid examples only")
    parser.add_argument("--new_proportion", type=float, default=0.0, help="proportion of new samples in test set")

    parser.add_argument("--num_workers", type=int, default=8, help="number of data workers for both train/test")
    parser.add_argument("--num_eval_steps", type=int, default=500, help="number of step between each evaluation during training.")
    parser.add_argument("--seed", type=int, default=-1, help="seed")
    # sampling
    parser.add_argument("--top_k", type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument("--n_layer", type=int, default=4, help="number of layers")
    parser.add_argument("--n_head", type=int, default=8, help="number of heads (in a transformer)")
    parser.add_argument("--n_embd", type=int, default=256, help="number of feature channels in the model")
    parser.add_argument("--no_positional", type=bool_flag, default="false", help="no positional embedding")
    parser.add_argument("--max_len", type=int, default=500, help="Block size, maximum length of sequences")

    # optimization
    parser.add_argument("--batch_size", type=int, default=32, help="batch size during optimization")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    # evaluation against known "good sequences"
    parser.add_argument("--gen_batch_size", type=int, default=1000, help="generation batch size")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--temp_span", type=int, default=0, help="temperature span")
    parser.add_argument("--inc_temp", type=float, default=0.0, help="temperature")
    parser.add_argument("--keep_only_unique", type=bool_flag, default="true", help="keep only unique data")
    parser.add_argument("--save_best", type=bool_flag, default="false", help="save best model based on test loss")

    # path and ports
    parser.add_argument("--dump_path", type=str, default="checkpoint", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--cpu", type=bool_flag, default="false", help="run on cpu only")
    parser.add_argument("--data_generation_only", type=bool_flag, default="false", help="only generate data and exit")
    parser.add_argument("--gen_log_interval", type=int, default=1_000_000, help="log best score every N generated examples (0 to disable)")
    parser.add_argument("--gen_save_interval", type=int, default=60, help="save pool to disk every N seconds during generation (0 to disable)")
    parser.add_argument("--per_batch_top_k", type=int, default=10, help="how many top candidates to return per generation batch")

    # background CPU work during training
    parser.add_argument("--bg_generation", type=bool_flag, default="false", help="run random graph generation on CPU during training")
    parser.add_argument("--bg_local_search", type=bool_flag, default="false", help="run local search on pool during training")
    parser.add_argument("--bg_workers_gen", type=int, default=0, help="CPU cores for background generation (0 = num_workers // 2)")
    parser.add_argument("--bg_workers_ls", type=int, default=0, help="CPU cores for background local search (0 = num_workers // 2)")
    parser.add_argument("--ls_sa_mult", type=int, default=10, help="SA steps multiplier: total steps = N^2 * this value")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.exp_id == "" and os.environ.get("MODAL_EXP_ID") is None:
        os.environ["MODAL_EXP_ID"] = time.strftime("%Y_%m_%d_%H_%M_%S")
        args.exp_id = os.environ["MODAL_EXP_ID"]

    args.device = "cpu" if args.cpu else ("mps" if torch.backends.mps.is_available() else "cuda")
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "mps":
        torch.mps.manual_seed(args.seed)

    fused = True if args.device in ["cuda", "mps"] else False

    logger = initialize_exp(args)
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)

    if args.seed < 0:
        args.seed = np.random.randint(1_000_000_000)
    logger.info(f"seed: {args.seed}")

    env = build_env(args)

    classname = env.data_class

    # system inits
    torch.manual_seed(args.seed)

    args.vocab_size = len(env.tokenizer.itos)

    args.block_size = args.max_len + 2
    stoi = env.tokenizer.stoi
    itos = env.tokenizer.itos

    # Initialize transformer
    model = Transformer(args, stoi["PAD"], stoi["EOS"])
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8, fused=fused)
    reload_model_optimizer(args, model, optimizer)

    train_set, test_set = load_initial_data(args, classname)
    if args.data_generation_only:
        logger.info("Data generation only mode. Exiting...")
        exit(0)
    train_data_path = os.path.join(args.dump_path, "train_data.pkl")
    test_data_path = os.path.join(args.dump_path, "test_data.pkl")

    # log initial stats
    metrics = do_stats(-1, data=train_set)
    temperature = args.temperature
    # Loop of Axplorer
    best_loss = None
    epoch_file = os.path.join(args.dump_path, "epoch.txt")
    if os.path.isfile(epoch_file):
        with open(epoch_file, "r") as f:
            n_epoch = int(f.read())
    else:
        n_epoch = 0
    temp_file = os.path.join(args.dump_path, "temperature.txt")
    if os.path.isfile(temp_file):
        with open(temp_file, "r") as f:
            temperature = float(f.read())
    else:
        temperature = args.temperature

    metric_file = os.path.join(args.dump_path, "metrics.txt")
    write_important_metrics(metrics, n_epoch, metric_file, command=args.command)

    for epoch in range(n_epoch, args.max_epochs):
        logger.info(f"==== Starting Epoch {n_epoch} =====")
        log_resources(f"Epoch {epoch} START")

        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()

        # tokenize
        train_words = [env.tokenizer.encode(d) for d in tqdm(train_set, desc="Tokenizing train", unit="ex")]
        test_words = [env.tokenizer.encode(d) for d in tqdm(test_set, desc="Tokenizing test", unit="ex")]
        # data loaders
        train_dataset = CharDataset(train_words, args.max_len, stoi)
        test_dataset = CharDataset(test_words, args.max_len, stoi)
        force_release_memory()

        if args.device == "cuda":
            logger.info(
                f"Memory allocated: {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB"
            )
        elif args.device == "mps":
            logger.info(
                f"Memory allocated: {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB, reserved: {torch.mps.driver_allocated_memory()/(1024*1024):.2f}MB"
            )

        # Start background CPU work (generation + local search) during training
        bg_use = args.bg_generation or args.bg_local_search
        if bg_use:
            bg_stop = threading.Event()
            bg_result = [None]  # mutable container for thread result

            def _bg_wrapper():
                bg_result[0] = run_background_cpu_work(classname, train_set, args, bg_stop, max_score=classname.max_possible_score(args.N))

            bg_thread = threading.Thread(target=_bg_wrapper, name="bg-cpu", daemon=True)
            bg_thread.start()
            logger.info(f"[BG] Started background CPU work (gen={args.bg_generation}, ls={args.bg_local_search})")

        batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=min(31, args.num_workers))
        try:
            best_loss = train(model, args, batch_loader, optimizer, test_dataset, current_best_loss=best_loss)
        except KeyboardInterrupt:
            if bg_use:
                bg_stop.set()
            raise
        log_resources(f"Epoch {epoch} AFTER_TRAIN")
        force_release_memory()

        # Stop background CPU work and collect results
        bg_generated, bg_ls_improved = [], []
        if bg_use:
            bg_stop.set()
            bg_thread.join()
            if bg_result[0] is not None:
                bg_generated, bg_ls_improved = bg_result[0]
            logger.info(f"[BG] Collected: {len(bg_generated)} generated, {len(bg_ls_improved)} from LS")

        logger.info(f"Sample with temperature {temperature} to {temperature+0.1*args.temp_span}")
        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()

        new_data = sample_and_score(model, args, stoi, itos, env, temperature, args.temp_span)
        log_resources(f"Epoch {epoch} AFTER_SAMPLE")

        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()

        # Merge model samples with background CPU results
        all_new_data = new_data + bg_generated + bg_ls_improved
        train_set, test_set, inc_temp = update_datasets(args, all_new_data, train_set, test_set, train_data_path, test_data_path)
        log_resources(f"Epoch {epoch} AFTER_UPDATE_DATASETS")
        force_release_memory()

        if inc_temp and args.inc_temp > 0.0:
            temperature += args.inc_temp

        metrics = do_stats(-1, data=train_set)

        n_epoch += 1
        with open(epoch_file, "w") as f:
            f.write(str(n_epoch))
        with open(temp_file, "w") as f:
            f.write(str(temperature))

        write_important_metrics(metrics, n_epoch, metric_file)
