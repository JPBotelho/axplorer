#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path
from src.envs.cage import _score_violations

def extract_valid_cages(checkpoint_dir=None):
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

if __name__ == "__main__":
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else None
    extract_valid_cages(checkpoint_dir)
