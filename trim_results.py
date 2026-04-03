import argparse
import gzip
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--pkl", required=True)
parser.add_argument("--top_k", type=int, default=100000)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

out = args.out or args.pkl.replace(".pkl", f"_trimmed.pkl.gz")

print(f"Loading {args.pkl} ...")
data = pickle.load(open(args.pkl, "rb"))
data.sort(key=lambda d: d.score, reverse=True)
top = data[:args.top_k]

with gzip.open(out, "wb") as f:
    pickle.dump(top, f)

print(f"Saved {len(top)} graphs to {out}")
print(f"Score range: {top[-1].score} – {top[0].score}")
