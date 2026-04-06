#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path
from src.envs.cage import _score_violations

def extract_valid_cages(checkpoint_dir=None):
    """Extract all cages with violations=0, sorted by max score."""
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

    # Find all valid cages (violations=0)
    valid_cages = []
    for i, d in enumerate(train_data):
        viol = _score_violations(d.data, d.N, d.K_REG)
        if viol == 0:
            valid_cages.append((i, d, d.score))

    if not valid_cages:
        print(f"No valid cages found (violations=0)")
        print(f"Total datapoints: {len(train_data)}")
        best = sorted(train_data, key=lambda d: d.score, reverse=True)[:5]
        print(f"Top 5 scores: {[d.score for d in best]}")
        return

    # Sort by score descending
    valid_cages.sort(key=lambda x: x[2], reverse=True)
    max_score = valid_cages[0][2]

    print(f"\n=== VALID CAGES (violations=0) ===")
    print(f"Total valid cages: {len(valid_cages)}")
    print(f"Max score: {max_score}\n")

    # Extract all cages with max score
    cages_with_max_score = [c for c in valid_cages if c[2] == max_score]

    for idx, (i, cage, score) in enumerate(cages_with_max_score):
        print(f"Cage {idx+1}:")
        print(f"  Index: {i}")
        print(f"  N: {cage.N}")
        print(f"  K: {cage.K_REG}")
        print(f"  Score: {score}")

        # Save each cage
        output_path = f"cage_score_{score}_{idx}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(cage, f)
        print(f"  Saved to {output_path}\n")

if __name__ == "__main__":
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else None
    extract_valid_cages(checkpoint_dir)
