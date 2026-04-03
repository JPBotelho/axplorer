"""
Prepares a local search results pkl for use as axplorer training data.
Splits into train/test and writes to the target experiment directory.

Usage:
    python prep_training_data.py \
        --pkl checkpoint/.../ls_results_v5_trimmed.pkl \
        --dump_path checkpoint/ramsey_r55_n43_from_ls \
        --ntest 1000
"""

import argparse
import gzip
import os
import pickle
import random

parser = argparse.ArgumentParser()
parser.add_argument("--pkl", required=True)
parser.add_argument("--dump_path", required=True)
parser.add_argument("--ntest", type=int, default=1000)
args = parser.parse_args()

os.makedirs(args.dump_path, exist_ok=True)

print(f"Loading {args.pkl} ...")
opener = gzip.open if args.pkl.endswith(".gz") else open
with opener(args.pkl, "rb") as f:
    data = pickle.load(f)

data.sort(key=lambda d: d.score, reverse=True)
print(f"Loaded {len(data)} graphs | top: {data[0].score} | bottom: {data[-1].score}")

random.shuffle(data)
test = data[:args.ntest]
train = data[args.ntest:]

train_path = os.path.join(args.dump_path, "train_data.pkl")
test_path = os.path.join(args.dump_path, "test_data.pkl")
pickle.dump(train, open(train_path, "wb"))
pickle.dump(test, open(test_path, "wb"))

print(f"Train: {len(train)} graphs → {train_path}")
print(f"Test:  {len(test)} graphs → {test_path}")
print(f"\nReady. Run train.py with --dump_path {args.dump_path}")
