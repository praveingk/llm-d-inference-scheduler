#!/usr/bin/env bash
set -euo pipefail

SCENARIO_DIR="scenarios"

for scenario in "$SCENARIO_DIR"/production-mix-*-120.yaml; do
    name=$(basename "$scenario" .yaml)
    echo "========== Running: $name =========="
    SCENARIO="$scenario" ./run_test.sh
    echo "========== Done: $name =========="
    echo
done

echo "All scenarios complete."