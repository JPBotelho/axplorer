#!/bin/bash
set -e

# Create base env
micromamba env create -f environment.yml -y 2>/dev/null || conda env create -f environment.yml -y

# Activate
eval "$(conda shell.bash hook 2>/dev/null || micromamba shell hook -s bash)"
conda activate env_axplorer 2>/dev/null || micromamba activate env_axplorer

# Install pytorch with CUDA 12.4 (must be done separately to avoid solver picking CPU build)
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "Done. Verify with: python -c 'import torch; print(torch.cuda.is_available())'"
