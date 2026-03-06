#!/usr/bin/env bash
#
# Fairness A/B Test Orchestrator
#
# Reads a scenario YAML and runs an A/B test:
#   1. Optionally deploys a kind cluster
#   2. Tunes the vllm-sim deployment
#   3. For each phase: switches EPP config, runs loadgen, snapshots metrics
#   4. Runs analysis across all phases
#
# Prerequisites:
#   - Python 3.9+ with aiohttp, numpy, matplotlib, pyyaml
#   - kubectl, kind (if infra.kind=true)
#
# Usage:
#   ./run_test.sh                                          # default scenario
#   SCENARIO=scenarios/uniform-fairness.yaml ./run_test.sh # custom scenario

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Configuration ---
SCENARIO="${SCENARIO:-scenarios/stress-h100.yaml}"

# --- YAML Parsing Helper ---
# Extracts a value from the scenario YAML via Python.
yaml_get() {
    python3 -c "
import yaml, sys
with open('$SCRIPT_DIR/$SCENARIO') as f:
    cfg = yaml.safe_load(f)
# Navigate dotted path
val = cfg
for key in '$1'.split('.'):
    if val is None:
        val = None
        break
    if isinstance(val, list):
        val = val[int(key)]
    else:
        val = val.get(key)
if val is None:
    print('${2:-}')
elif isinstance(val, (list, dict)):
    import json
    print(json.dumps(val))
else:
    print(val)
"
}

# --- Read scenario ---
MODEL="$(yaml_get model)"
INFRA_KIND="$(yaml_get infra.kind false)"
GATEWAY_URL="$(yaml_get infra.gateway_url http://localhost:30080)"
NAMESPACE="$(yaml_get infra.namespace default)"
METRICS_URL="${METRICS_URL:-http://localhost:9090}"

# Derive names from model.
MODEL_SLUG="$(echo "${MODEL##*/}" | tr '[:upper:]' '[:lower:]' | tr ' /_.' '-')"
EPP_NAME="${EPP_NAME:-${MODEL_SLUG}-endpoint-picker}"
VLLM_DEPLOY="${MODEL_SLUG}-vllm-sim"

# If kind, override gateway URL.
if [ "$INFRA_KIND" = "true" ] || [ "$INFRA_KIND" = "True" ]; then
    GATEWAY_URL="http://localhost:30080"
fi

# Output directories.
RESULTS_DIR="$SCRIPT_DIR/results"

# --- Helpers ---
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

check_gateway() {
    log "Checking gateway at $GATEWAY_URL ..."
    local retries=36
    for i in $(seq 1 $retries); do
        if curl -sf --max-time 10 "$GATEWAY_URL/v1/models" > /dev/null 2>&1; then
            log "Gateway OK."
            return
        fi
        if [ $i -lt $retries ]; then
            log "Gateway not ready yet, retrying in 5s ($i/$retries) ..."
            sleep 30
        fi
    done
    die "Gateway not responding at $GATEWAY_URL after $retries attempts. Ensure port-forward is active."
}

snapshot_metrics() {
    local output="$1"
    log "Snapshotting metrics to $output ..."
    curl -sf --max-time 10 "$METRICS_URL/metrics" > "$output" 2>/dev/null || \
        log "WARNING: Could not fetch metrics from $METRICS_URL (is port-forward active?)"
}

switch_config() {
    local config_file="$1"
    local config_name="$2"
    log "Switching EPP config to $config_name ..."

    # Resolve relative path from SCRIPT_DIR.
    local abs_config="$SCRIPT_DIR/$config_file"
    if [ ! -f "$abs_config" ]; then
        die "EPP config not found: $abs_config"
    fi

    # Update ConfigMap.
    kubectl -n "$NAMESPACE" create configmap epp-config \
        --from-file=epp-config.yaml="$abs_config" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Restart EPP.
    kubectl -n "$NAMESPACE" rollout restart deployment/"$EPP_NAME"
    kubectl -n "$NAMESPACE" rollout status deployment/"$EPP_NAME" --timeout=120s

    # Wait for EPP to be ready.
    log "Waiting 10s for EPP to stabilize ..."
    sleep 10

    # Verify gateway still responds after config switch.
    check_gateway
    log "Config switched to $config_name."
}

deploy_kind() {
    log "Setting up Go modules ..."
    cd "$REPO_DIR"
    export GO111MODULE=on
    export GOPRIVATE=github.com/llm-d/*
    go mod download
    go get sigs.k8s.io/gateway-api-inference-extension@08fc9b098204edf50dca24b0b5a98f3a0c600e41
    go mod tidy

    log "Building EPP image ..."
    EPP_TAG="${EPP_TAG:-program-aware}" make -C "$REPO_DIR" image-build-epp

    log "Building UDS tokenizer image ..."
    make -C "$REPO_DIR" image-build-uds-tokenizer

    log "Deploying kind cluster with MODEL_NAME=$MODEL ..."
    EPP_TAG="${EPP_TAG:-program-aware}" \
    EPP_CONFIG="$SCRIPT_DIR/$(yaml_get "phases.0.epp_config")" \
    MODEL_NAME="$MODEL" \
    make -C "$REPO_DIR" env-dev-kind

    cd "$SCRIPT_DIR"
    log "Kind cluster deployed."
}

tune_simulator() {
    # Base args — same as deploy/components/vllm-sim/deployments.yaml.
    local base_args=(
        "--port=8200"
        "--model=$MODEL"
        "--enable-kvcache=false"
        "--block-size=16"
        "--zmq-endpoint=tcp://${EPP_NAME}.${NAMESPACE}.svc.cluster.local:5557"
        "--event-batch-size=16"
        "--tokenizers-cache-dir=/tokenizer-cache"
        "--data-parallel-size=1"
    )

    # Read extra_args from YAML.
    local extra_json
    extra_json="$(yaml_get infra.vllm.extra_args '[]')"

    # Parse extra_args JSON array into bash array.
    local extra_args=()
    if [ "$extra_json" != "[]" ] && [ -n "$extra_json" ]; then
        while IFS= read -r arg; do
            extra_args+=("$arg")
        done < <(python3 -c "import json; [print(a) for a in json.loads('$extra_json')]")
    fi

    if [ ${#extra_args[@]} -eq 0 ]; then
        log "No vllm extra_args in scenario, skipping simulator tuning."
        return
    fi

    # Combine base + extra.
    local all_args=("${base_args[@]}" "${extra_args[@]}")

    # Build JSON array for kubectl patch.
    local args_json
    args_json="$(printf '%s\n' "${all_args[@]}" | python3 -c "import sys, json; print(json.dumps([line.strip() for line in sys.stdin]))")"

    log "Tuning vllm-sim with ${#all_args[@]} args ..."
    kubectl -n "$NAMESPACE" patch deployment "$VLLM_DEPLOY" --type='json' \
        -p="[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$args_json}]"

    # Wait for rollout.
    kubectl -n "$NAMESPACE" rollout status deployment/"$VLLM_DEPLOY" --timeout=120s

    log "Waiting 10s for simulator to stabilize ..."
    sleep 10
    check_gateway
    log "Simulator tuned."
}

run_loadgen() {
    local output_file="$1"
    local label="$2"

    log "Starting $label load test ..."
    python3 "$SCRIPT_DIR/fairness_loadgen.py" \
        --scenario "$SCRIPT_DIR/$SCENARIO" \
        --gateway-url "$GATEWAY_URL" \
        --model "$MODEL" \
        --output "$output_file"
    log "$label load test complete."
}

# --- Main ---
main() {
    log "=== Fairness A/B Test ==="
    log "Scenario: $SCENARIO"
    log "Model:    $MODEL"
    log "Gateway:  $GATEWAY_URL"
    log "Kind:     $INFRA_KIND"
    log ""

    # Deploy kind cluster if requested.
    if [ "$INFRA_KIND" = "true" ] || [ "$INFRA_KIND" = "True" ]; then
        deploy_kind
    fi

    # Pre-flight check.
    check_gateway

    # Tune simulator.
    tune_simulator

    # Read phases from YAML.
    local phases_json
    phases_json="$(yaml_get phases '[]')"
    local phase_count
    phase_count="$(python3 -c "import json; print(len(json.loads('$phases_json')))")"

    log "Running $phase_count phases ..."

    local phase_dirs=()
    for i in $(seq 0 $((phase_count - 1))); do
        local phase_name
        phase_name="$(yaml_get "phases.$i.name")"
        local epp_config
        epp_config="$(yaml_get "phases.$i.epp_config")"

        local phase_dir="$RESULTS_DIR/$phase_name"
        mkdir -p "$phase_dir"
        phase_dirs+=("$phase_dir")

        log ""
        log "========================================="
        log "  PHASE $((i+1))/$phase_count: $phase_name"
        log "========================================="

        switch_config "$epp_config" "$phase_name"
        run_loadgen "$phase_dir/results.jsonl" "$phase_name"
        snapshot_metrics "$phase_dir/metrics_final.txt"

        # Brief pause between phases.
        if [ $i -lt $((phase_count - 1)) ]; then
            log "Pausing 15s between phases ..."
            sleep 15
        fi
    done

    # --- Analysis ---
    log ""
    log "========================================="
    log "  ANALYSIS"
    log "========================================="
    python3 "$SCRIPT_DIR/analyze_results.py" \
        --results-dir "$RESULTS_DIR"

    log ""
    log "=== Test complete! ==="
    log "Results: $RESULTS_DIR"
    for dir in "${phase_dirs[@]}"; do
        log "  $(basename "$dir"): $dir/results.jsonl"
    done
}

main "$@"