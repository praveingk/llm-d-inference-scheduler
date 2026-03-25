#!/usr/bin/env bash
set -euo pipefail

for d in results/production-mix-*/; do
    rm -rf "$d/plots"
    echo "=== Plotting: $(basename "$d") ==="
    python3 analyze.py "$d"
done
