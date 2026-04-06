#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path
from src.envs.cage import _score_violations

checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else None

if checkpoint_dir is None:
    checkpoints = sorted(
        Path("checkpoint").rglob("train_data.pkl"),
        key=lambda p: p.parent.name,
        reverse=True
    )
    if not checkpoints:
        print("No checkpoint found")
        sys.exit(1)
    latest = checkpoints[0]
else:
    latest = Path(checkpoint_dir) / "train_data.pkl"

print(f"Loading {latest}")
with open(latest, "rb") as f:
    train_data = pickle.load(f)

# Get top scores
top_by_score = sorted(train_data, key=lambda d: d.score, reverse=True)[:10]

print(f"\nTop 10 by score:")
print(f"{'Index':<6} {'Score':<10} {'Violations':<12} {'N':<5} {'K':<5}")
print("-" * 45)

for i, cage in enumerate(top_by_score):
    viol = _score_violations(cage.data, cage.N, cage.K_REG)
    print(f"{i:<6} {cage.score:<10} {viol:<12} {cage.N:<5} {cage.K_REG:<5}")
