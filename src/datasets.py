import os
import pickle
import random
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import repeat
from logging import getLogger

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.envs.environment import do_stats

logger = getLogger()


def detokenize(data, args, env, executor=None):
    res = []
    pars = env.tokenizer.dataclass._save_class_params()
    if args.process_pool:
        BATCH = args.gen_batch_size
        data_slices = [data[i : i + BATCH] for i in range(0, len(data), BATCH)]

        if executor is not None:
            for chunk in executor.map(env.tokenizer.decode_batch, data_slices, repeat(pars, len(data_slices))):
                if chunk:
                    res.extend(chunk)
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
                for chunk in ex.map(env.tokenizer.decode_batch, data_slices, repeat(pars, len(data_slices))):
                    if chunk:
                        res.extend(chunk)
    else:
        res = env.tokenizer.decode_batch(data, pars)
    return res


# helper functions for creating the training and test Datasets


def generate_and_score(args, classname, train_data_path=None, test_data_path=None):
    """
    Generation method if no data
    """
    data = []
    BATCH = args.gen_batch_size
    num_batches = args.gensize // BATCH
    rem = args.gensize % BATCH
    # Use itertools.repeat to avoid materializing a potentially huge list
    import itertools
    batch_counts_iter = itertools.chain(itertools.repeat(BATCH, num_batches), [rem] if rem else [])
    num_batches_total = num_batches + (1 if rem else 0)

    max_score = classname.max_possible_score(args.N)
    if max_score is not None:
        logger.info(f"Max possible score: {max_score}")

    if hasattr(classname, '_nb_warmup'):
        classname._nb_warmup()

    # Return only the local top-k per batch to avoid shipping gensize objects over IPC.
    # Use 4x pop_size as budget so select_best still has good candidates to choose from.
    pop_size = getattr(args, 'pop_size', None)
    per_batch_top_k = getattr(args, 'per_batch_top_k', 10)

    gen_log_interval = getattr(args, 'gen_log_interval', 0)
    gen_save_interval = getattr(args, 'gen_save_interval', 600)
    last_logged = 0
    last_saved = time.time()
    n_generated = 0
    seen_features = set()

    def _write_top_dot(top_k=10):
        if not data or train_data_path is None:
            return
        top = sorted(data, key=lambda d: d.score, reverse=True)[:top_k]
        dot_dir = os.path.join(os.path.dirname(train_data_path), "top_graphs")
        os.makedirs(dot_dir, exist_ok=True)
        for rank, d in enumerate(top, 1):
            path = os.path.join(dot_dir, f"rank_{rank:02d}_score_{d.score}.dot")
            n = d.N
            lines = [f"graph rank{rank} {{", f'  label="rank {rank} | score {d.score}";']
            for i in range(n):
                for j in range(i + 1, n):
                    if d.data[i, j]:
                        lines.append(f"  {i} -- {j};")
            lines.append("}")
            with open(path, "w") as f:
                f.write("\n".join(lines) + "\n")
        logger.info(f"Top {len(top)} graphs written to {dot_dir}/")

    def _save():
        nonlocal last_saved
        if not data or train_data_path is None:
            return
        update_datasets(args, data, [], None, train_data_path, test_data_path)
        _write_top_dot()
        logger.info(f"Checkpoint saved ({len(data)} pool items)")
        last_saved = time.time()

    def _log_stats():
        nonlocal last_logged
        if gen_log_interval <= 0 or n_generated - last_logged < gen_log_interval:
            return
        last_logged = n_generated
        if not data:
            return
        scores = np.array([d.score for d in data])
        counts = {}
        for s in scores:
            counts[s] = counts.get(s, 0) + 1
        top10 = sorted(counts.items(), reverse=True)[:10]
        pcts = np.percentile(scores, [50, 75, 90, 99])
        lines = [f"gen_progress: {n_generated}/{args.gensize} | pool: {len(scores)}"]
        for s, c in reversed(top10):
            lines.append(f"  {s} x{c}")
        lines.append(f"  mean: {scores.mean():.1f} | p50: {pcts[0]:.0f} | p75: {pcts[1]:.0f} | p90: {pcts[2]:.0f} | p99: {pcts[3]:.0f}")
        logger.info("\n".join(lines))
        if gen_save_interval > 0 and time.time() - last_saved >= gen_save_interval:
            _save()

    found_max = False
    try:
        if args.process_pool:
            import concurrent.futures
            pars = classname._save_class_params()
            max_pending = args.num_workers * 2
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                pending = {}
                for bc in itertools.islice(batch_counts_iter, max_pending):
                    f = executor.submit(classname._batch_generate_and_score, bc, args.N, pars, per_batch_top_k)
                    pending[f] = bc
                with tqdm(total=args.gensize, desc="Generating data", unit="ex") as pbar:
                    while pending and not found_max:
                        done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in done:
                            bc = pending.pop(future)
                            chunk = future.result()
                            n_generated += bc
                            pbar.update(bc)
                            if chunk:
                                for dp in chunk:
                                    if dp.features not in seen_features:
                                        seen_features.add(dp.features)
                                        data.append(dp)
                                        if max_score is not None and dp.score >= max_score:
                                            logger.info(f"[GEN] Found perfect graph (score={dp.score})! Stopping early.")
                                            found_max = True
                                _log_stats()
                            if found_max:
                                break
                            # submit next batch
                            try:
                                next_bc = next(batch_counts_iter)
                                f = executor.submit(classname._batch_generate_and_score, next_bc, args.N, pars, per_batch_top_k)
                                pending[f] = next_bc
                            except StopIteration:
                                pass
        else:
            with tqdm(total=args.gensize, desc="Generating data", unit="ex") as pbar:
                for t in batch_counts_iter:
                    if found_max:
                        break
                    chunk = classname._batch_generate_and_score(t, args.N, return_top_k=per_batch_top_k)
                    n_generated += t
                    pbar.update(t)
                    if chunk:
                        for dp in chunk:
                            if dp.features not in seen_features:
                                seen_features.add(dp.features)
                                data.append(dp)
                                if max_score is not None and dp.score >= max_score:
                                    logger.info(f"[GEN] Found perfect graph (score={dp.score})! Stopping early.")
                                    found_max = True
                        _log_stats()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # block further Ctrl+C until save completes
        logger.info(f"Interrupted after {n_generated} examples — saving current pool (do not Ctrl+C again)...")
        if data and train_data_path is not None:
            import pickle
            _save()
            logger.info(f"Pool saved to {train_data_path}. Exiting.")
        raise SystemExit(1)

    if data:
        scores = np.array([d.score for d in data])
        logger.info(f"gen_complete: {n_generated} generated | pool: {len(scores)} | max: {scores.max()} | mean: {scores.mean():.1f}")
        _write_top_dot()
    return data, found_max


def select_best(n, data):
    if len(data) <= n:
        random.shuffle(data)
        return data
    sorted_data = sorted(data, key=lambda x: x.score, reverse=True)[:n]
    random.shuffle(sorted_data)
    return sorted_data


def make_train_test(data, ntest):
    """
    Create a train and test dataset from a dataset.
    The top-scoring examples are always kept in train to avoid losing the best graphs.
    """
    if len(data) <= ntest:
        return data, []
    # Sort by score descending, protect the top ntest examples from being split into test
    sorted_data = sorted(data, key=lambda x: x.score, reverse=True)
    protected = sorted_data[:ntest]  # top ntest always in train
    rest = sorted_data[ntest:]
    indices = np.random.permutation(len(rest))
    rest = [rest[i] for i in indices]
    test_set = rest[:ntest]
    train_set = protected + rest[ntest:]
    random.shuffle(train_set)
    return train_set, test_set


def wl_dedup(data):
    """Deduplicate by WL graph hash (catches isomorphic graphs that feature-string dedup misses)."""
    try:
        import networkx as nx
        from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
    except ImportError:
        logger.warning("networkx not available, skipping WL dedup")
        return data
    seen = set()
    out = []
    for d in data:
        g = nx.from_numpy_array(d.data)
        h = weisfeiler_lehman_graph_hash(g)
        if h not in seen:
            seen.add(h)
            out.append(d)
    return out


def compute_unique_data(old_data, new_data=None):
    def add_unique(src, unique_hashes):
        des = []
        for d in src:
            if d.features not in unique_hashes:
                unique_hashes.add(d.features)
                des.append(d)
        return des, unique_hashes

    unique_hashes = set()
    unique_old_data, unique_hashes = add_unique(old_data, unique_hashes)
    if new_data is not None:
        unique_new_data, unique_hashes = add_unique(new_data, unique_hashes)
    else:
        unique_new_data = None
    return unique_old_data, unique_new_data


def update_datasets(args, data, train_set, test_set, train_path, test_path):
    inc_temp = False
    if args.keep_only_unique:
        bef = len(data)
        data, _ = compute_unique_data(data)
        aft = len(data)
        logger.info(f"Unique processing: {aft} examples left, {bef-aft} duplicates")
        if getattr(args, 'wl_dedup', False):
            bef2 = len(data)
            data = wl_dedup(data)
            aft2 = len(data)
            logger.info(f"WL dedup: {aft2} examples left, {bef2-aft2} isomorphic duplicates removed")
        do_stats(-1, data)
        if aft / (bef + 1) < 0.9:
            inc_temp = True
    if args.new_proportion > 0.0:
        new_data = select_best(int(args.new_proportion * args.pop_size), data)
    else:
        new_data = select_best(args.pop_size, data)

    if len(new_data) >= 2 * args.ntest or test_set is None:
        new_train, test_set = make_train_test(new_data, args.ntest)
    else:
        new_train = new_data
    logger.info(f"New train and test generated. Size are train: {len(new_train)}, test {len(test_set)}")
    # Get all examples of previous train and current train and then select best.
    if args.keep_only_unique:
        train_set, new_train = compute_unique_data(train_set, new_train)
        logger.info(f"Unique data computed for original train set: {len(train_set)}, generated train set: {len(new_train)}")
    if args.new_proportion > 0.0:
        train_set = select_best(int((1.0 - args.new_proportion) * args.pop_size), train_set) + new_train
    else:
        train_set = select_best(args.pop_size, train_set + new_train)
    logger.info(f"Final train and test generated. Size are train: {len(train_set)}, test {len(test_set)}")

    pickle.dump(test_set, open(test_path, "wb"))
    pickle.dump(train_set, open(train_path, "wb"))
    return train_set, test_set, inc_temp


def load_initial_data(args, classname):
    train_data_path = os.path.join(args.dump_path, "train_data.pkl")
    test_data_path = os.path.join(args.dump_path, "test_data.pkl")
    if os.path.isfile(train_data_path) and os.path.isfile(test_data_path):
        logger.info("resuming from existing data")
        train_set = pickle.load(open(train_data_path, "rb"))
        test_set = pickle.load(open(test_data_path, "rb"))
    else:
        if os.path.isfile(train_data_path) and not os.path.isfile(test_data_path):
            logger.info("Found interrupted generation checkpoint — running update_datasets to finalize")
            data = pickle.load(open(train_data_path, "rb"))
            found_max = False
        else:
            data, found_max = generate_and_score(args, classname=classname, train_data_path=train_data_path, test_data_path=test_data_path)
        test_set = []
        train_set = []
        train_set, test_set, _ = update_datasets(args, data, train_set, test_set, train_data_path, test_data_path)
        if found_max:
            perfect = [d for d in train_set if d.score >= classname.max_possible_score(args.N)]
            logger.info(f"Perfect graph(s) found during generation ({len(perfect)} total). Saved to {train_data_path}. Exiting.")
            import sys
            sys.exit(0)
    return train_set, test_set


class CharDataset(Dataset):
    def __init__(self, encoded_data, max_len, stoi):
        self.encoded_data = encoded_data
        self.max_len = max_len
        self.pad_token_id = stoi["PAD"]

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def collate_fn(self, batch):
        x = np.full((len(batch), self.max_len + 2), self.pad_token_id, dtype=np.int32)

        for i, el in enumerate(batch):
            x[i, : el.shape[0]] = el
        valid_col = (x != self.pad_token_id).any(axis=0)
        last_col = np.nonzero(valid_col)[0][-1] + 1
        x = x[:, :last_col]
        y = np.concatenate([x[:, 1:], np.full((len(batch), 1), self.pad_token_id, dtype=x.dtype)], axis=1)
        return torch.LongTensor(x), torch.LongTensor(y)


class InfiniteDataLoader:
    """
    Create a infinite datalaoder in PyTorch
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, collate_fn=dataset.collate_fn, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:  # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch
