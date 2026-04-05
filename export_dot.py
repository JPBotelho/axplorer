"""
Export the top-K graphs from a pkl as DOT files.

Edges are 2-colored: red = data[i,j]==1 (the K_s side), blue = data[i,j]==0
(the K_t side). Each graph is written to <out_dir>/<prefix>_rank<k>_score<s>.dot.

Usage:
    python export_dot.py --pkl <train_data.pkl> --out dot_exports/r46_n35 --k 10 --prefix r46_n35
"""

import argparse
import os
import pickle


def to_dot(dp, name="G"):
    n = dp.N
    lines = [f"graph {name} {{"]
    lines.append("  layout=circo;")
    lines.append("  node [shape=circle, style=filled, fillcolor=white];")
    for i in range(n):
        lines.append(f"  {i};")
    for i in range(n):
        for j in range(i + 1, n):
            color = "red" if dp.data[i, j] == 1 else "blue"
            lines.append(f"  {i} -- {j} [color={color}];")
    lines.append("}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=True)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--prefix", type=str, default="graph")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.pkl} ...")
    data = pickle.load(open(args.pkl, "rb"))
    data.sort(key=lambda d: d.score, reverse=True)
    top = data[: args.k]
    print(f"Loaded {len(data)} graphs; exporting top {len(top)}")
    print(f"  best score: {top[0].score}  |  worst of top-{len(top)}: {top[-1].score}")

    for rank, dp in enumerate(top, start=1):
        fname = f"{args.prefix}_rank{rank:02d}_score{dp.score}.dot"
        path = os.path.join(args.out, fname)
        with open(path, "w") as f:
            f.write(to_dot(dp, name=f"{args.prefix}_rank{rank}"))
        print(f"  wrote {path}")

    print(f"\nDone. {len(top)} DOT files in {args.out}")
    print(f"Render one with:  dot -Kcirco -Tsvg {args.out}/{args.prefix}_rank01_score*.dot -o rank01.svg")


if __name__ == "__main__":
    main()
