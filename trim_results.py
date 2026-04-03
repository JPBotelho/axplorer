import argparse
import gzip
import os
import pickle

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pkl", required=True)
parser.add_argument("--top_k", type=int, default=100000)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

out = args.out or args.pkl.replace(".pkl", f"_trimmed.pkl")

print(f"Loading {args.pkl} ...")
file_size = os.path.getsize(args.pkl)
with open(args.pkl, "rb") as f:
    with tqdm(total=file_size, unit="B", unit_scale=True, desc="Loading") as pbar:
        class ProgressFile:
            def read(self, n=-1):
                data = f.read(n)
                pbar.update(len(data))
                return data
            def readline(self):
                data = f.readline()
                pbar.update(len(data))
                return data
            def readinto(self, b):
                n = f.readinto(b)
                pbar.update(n)
                return n
        data = pickle.load(ProgressFile())
data.sort(key=lambda d: d.score, reverse=True)
top = data[:args.top_k]

pickle.dump(top, open(out, "wb"))

print(f"Saved {len(top)} graphs to {out}")
print(f"Score range: {top[-1].score} – {top[0].score}")
