#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np

from src.envs.cage import _score_violations


def cage_to_dot(cage) -> str:
    """
    Convert a cage (expects .data adjacency matrix, plus .N/.K_REG/.score) to Graphviz DOT.

    The cage is assumed undirected; edges are emitted only for i<j.
    """
    N = int(cage.N)
    k = int(getattr(cage, "K_REG", -1))
    score = getattr(cage, "score", None)
    A = np.asarray(cage.data)

    # Use upper triangle to avoid double-emitting edges.
    rows, cols = np.triu_indices(N, k=1)
    mask = A[rows, cols].astype(bool)

    label_parts = [f"N={N}"]
    if k >= 0:
        label_parts.append(f"k={k}")
    if score is not None:
        label_parts.append(f"score={score}")
    graph_label = ", ".join(label_parts)

    lines = []
    lines.append("graph Cage {")
    lines.append('  graph [fontname="Helvetica"];')
    lines.append('  node [fontname="Helvetica", shape=circle];')
    lines.append(f'  label="{graph_label}";')

    for i in range(N):
        lines.append(f'  {i} [label="{i}"];')

    for u, v in zip(rows[mask], cols[mask]):
        lines.append(f"  {int(u)} -- {int(v)};")

    lines.append("}")
    return "\n".join(lines)

def extract_valid_cages(checkpoint_dir=None, export_dot=True):
    """Extract all cages with max score."""
    if checkpoint_dir is None:
        # Load the most recent training data
        checkpoints = sorted(
            Path("checkpoint").rglob("train_data.pkl"),
            key=lambda p: p.parent.name,
            reverse=True
        )
        if not checkpoints:
            print("No checkpoint found")
            return
        latest = checkpoints[0]
    else:
        latest = Path(checkpoint_dir) / "train_data.pkl"

    print(f"Loading {latest}")
    with open(latest, "rb") as f:
        train_data = pickle.load(f)

    # Find max score
    max_score = max(d.score for d in train_data)
    cages_with_max_score = [(i, d) for i, d in enumerate(train_data) if d.score == max_score]

    print(f"\n=== CAGES WITH MAX SCORE ===")
    print(f"Max score: {max_score}")
    print(f"Number of cages with max score: {len(cages_with_max_score)}\n")

    # Check violations for each
    for idx, (i, cage) in enumerate(cages_with_max_score):
        viol = _score_violations(cage.data, cage.N, cage.K_REG)
        print(f"Cage {idx+1}:")
        print(f"  Index: {i}")
        print(f"  N: {cage.N}")
        print(f"  K: {cage.K_REG}")
        print(f"  Score: {cage.score}")
        print(f"  Violations: {viol}")

        # Save each cage
        output_path = f"cage_score_{max_score}_{idx}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(cage, f)
        print(f"  Saved to {output_path}\n")

        if export_dot:
            dot_path = f"cage_score_{max_score}_{idx}.dot"
            dot_txt = cage_to_dot(cage)
            with open(dot_path, "w", encoding="utf-8") as f:
                f.write(dot_txt)
            print(f"  Saved DOT to {dot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract cages with max score from train_data.pkl")
    parser.add_argument(
        "--pkl",
        default=None,
        help="Optional: convert a single existing cage pickle (*.pkl) into a sibling Graphviz (*.dot) file.",
    )
    parser.add_argument(
        "checkpoint_dir",
        nargs="?",
        default=None,
        help="Directory containing train_data.pkl (default: search latest in ./checkpoint)",
    )
    parser.add_argument(
        "--skip-dot",
        action="store_true",
        help="Do not write Graphviz .dot files (still writes .pkl files).",
    )
    args = parser.parse_args()
    # If an explicit cage pickle is provided, convert it and exit.
    if args.pkl is not None:
        pkl_path = Path(args.pkl)
        with open(pkl_path, "rb") as f:
            cage = pickle.load(f)
        dot_path = pkl_path.with_suffix(".dot")
        dot_txt = cage_to_dot(cage)
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write(dot_txt)
        print(f"Saved DOT to {dot_path}")
        raise SystemExit(0)

    extract_valid_cages(checkpoint_dir=args.checkpoint_dir, export_dot=not args.skip_dot)
