#!/usr/bin/env bash
#
# Load Test Orchestrator
#
# Reads a scenario YAML and runs a load test:
#   1. Optionally deploys a kind cluster and tunes vllm-sim
#   2. For each phase: switches EPP config and runs the load generator
#
# Prerequisites:
#   - python3 with pyyaml available on PATH
#   - kubectl, kind (if infra.kind=true)
#
# Usage:
#   ./run_test.sh scenarios/simple-ab.yaml
#   SCENARIO=scenarios/simple-ab.yaml ./run_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="python3"

SCENARIO="${1:-${SCENARIO:-scenarios/simple-ab.yaml}}"

# --- YAML Parsing Helper ---
yaml_get() {
    "$PYTHON" -c "
import yaml, sys
with open('$SCRIPT_DIR/$SCENARIO') as f:
    cfg = yaml.safe_load(f)
val = cfg
for key in '$1'.split('.'):
    if val is None:
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
NAMESPACE="$(yaml_get infra.namespace default)"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:30080}"

# Derive names from model.
MODEL_SLUG="$(echo "${MODEL##*/}" | tr '[:upper:]' '[:lower:]' | tr ' /_.' '-')"
EPP_NAME="${EPP_NAME:-${MODEL_SLUG}-endpoint-picker}"
VLLM_DEPLOY="${VLLM_DEPLOY:-${MODEL_SLUG}-vllm-sim}"

SCENARIO_NAME="$(basename "$SCENARIO" .yaml)"
RESULTS_DIR="$SCRIPT_DIR/results/$SCENARIO_NAME"

METRICS_URL="${METRICS_URL:-http://localhost:9090}"
METRICS_PF_PID=""

# --- Helpers ---
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# --- Metrics port-forward ---
start_metrics_portforward() {
    if [ -n "$METRICS_PF_PID" ] && kill -0 "$METRICS_PF_PID" 2>/dev/null; then
        kill "$METRICS_PF_PID" 2>/dev/null || true
        wait "$METRICS_PF_PID" 2>/dev/null || true
    fi

    log "Starting metrics port-forward (deployment/$EPP_NAME 9090:9090) ..."
    kubectl -n "$NAMESPACE" port-forward "deployment/$EPP_NAME" 9090:9090 > /dev/null 2>&1 &
    METRICS_PF_PID=$!

    sleep 3
    if kill -0 "$METRICS_PF_PID" 2>/dev/null; then
        log "Metrics port-forward started (PID $METRICS_PF_PID)."
    else
        log "WARNING: Metrics port-forward failed. Metrics scraping will be skipped."
        METRICS_PF_PID=""
    fi
}

cleanup_metrics_portforward() {
    if [ -n "$METRICS_PF_PID" ] && kill -0 "$METRICS_PF_PID" 2>/dev/null; then
        kill "$METRICS_PF_PID" 2>/dev/null || true
        wait "$METRICS_PF_PID" 2>/dev/null || true
    fi
}

# --- Gateway health check ---
check_gateway() {
    log "Checking gateway at $GATEWAY_URL ..."
    local retries=36
    for i in $(seq 1 $retries); do
        if curl -sf --max-time 10 "$GATEWAY_URL/v1/models" > /dev/null 2>&1; then
            log "Gateway OK."
            return
        fi
        if [ $i -lt $retries ]; then
            log "Gateway not ready ($i/$retries), retrying in 5s ..."
            sleep 5
        fi
    done
    die "Gateway not responding at $GATEWAY_URL after $retries attempts."
}

# --- Kind cluster deployment ---
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

    log "Deploying kind cluster (MODEL_NAME=$MODEL) ..."
    EPP_TAG="${EPP_TAG:-program-aware}" \
    EPP_CONFIG="$SCRIPT_DIR/$(yaml_get "phases.0.epp_config")" \
    MODEL_NAME="$MODEL" \
    make -C "$REPO_DIR" env-dev-kind

    cd "$SCRIPT_DIR"
    log "Kind cluster deployed."
}

# --- vllm-sim tuning ---
tune_simulator() {
    local base_args=(
        "--port=8200"
        "--model=$MODEL"
        "--enable-kvcache=false"
        "--block-size=16"
        "--zmq-endpoint=tcp://${EPP_NAME}.${NAMESPACE}.svc.cluster.local:5557"
        "--event-batch-size=16"
        "--tokenizers-cache-dir=/tokenizer-cache"
        "--data-parallel-size=1"
        "--seed=50"
    )

    local extra_json
    extra_json="$(yaml_get infra.vllm.extra_args '[]')"

    local extra_args=()
    if [ "$extra_json" != "[]" ] && [ -n "$extra_json" ]; then
        while IFS= read -r arg; do
            extra_args+=("$arg")
        done < <("$PYTHON" -c "import json; [print(a) for a in json.loads('$extra_json')]")
    fi

    if [ ${#extra_args[@]} -eq 0 ]; then
        log "No vllm extra_args in scenario, skipping simulator tuning."
        return
    fi

    local all_args=("${base_args[@]}" "${extra_args[@]}")
    local args_json
    args_json="$(printf '%s\n' "${all_args[@]}" | "$PYTHON" -c "import sys, json; print(json.dumps([l.strip() for l in sys.stdin]))")"

    log "Tuning vllm-sim with ${#all_args[@]} args ..."
    kubectl -n "$NAMESPACE" patch deployment "$VLLM_DEPLOY" --type='json' \
        -p="[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$args_json}]"
    kubectl -n "$NAMESPACE" rollout status deployment/"$VLLM_DEPLOY" --timeout=120s

    log "Waiting 10s for simulator to stabilize ..."
    sleep 10
    check_gateway
    log "Simulator tuned."
}

# --- EPP config switch ---
switch_config() {
    local config_file="$1"
    local phase_name="$2"
    local abs_config="$SCRIPT_DIR/$config_file"

    [ -f "$abs_config" ] || die "EPP config not found: $abs_config"

    log "Switching EPP config to $phase_name ..."
    kubectl -n "$NAMESPACE" create configmap epp-config \
        --from-file=epp-config.yaml="$abs_config" \
        --dry-run=client -o yaml | kubectl apply -f -

    kubectl -n "$NAMESPACE" rollout restart deployment/"$EPP_NAME"
    kubectl -n "$NAMESPACE" rollout status deployment/"$EPP_NAME" --timeout=120s

    # Disable metrics auth so scraper can reach /metrics without credentials.
    kubectl -n "$NAMESPACE" patch deployment "$EPP_NAME" --type='json' \
        -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--metrics-endpoint-auth=false"}]' \
        2>/dev/null || log "WARNING: Could not disable metrics auth (may already be set)."
    kubectl -n "$NAMESPACE" rollout status deployment/"$EPP_NAME" --timeout=120s 2>/dev/null || true

    log "Waiting 10s for EPP to stabilize ..."
    sleep 10
    check_gateway
    log "EPP config switched to $phase_name."
}

# --- Main ---
main() {
    log "=== Load Test ==="
    log "Scenario: $SCENARIO"
    log "Model:    $MODEL"
    log "Gateway:  $GATEWAY_URL"
    log "Kind:     $INFRA_KIND"
    log ""

    if [ "$INFRA_KIND" = "true" ] || [ "$INFRA_KIND" = "True" ]; then
        deploy_kind
        tune_simulator
    fi

    check_gateway
    trap cleanup_metrics_portforward EXIT

    local phases_json
    phases_json="$(yaml_get phases '[]')"
    local phase_count
    phase_count="$("$PYTHON" -c "import json; print(len(json.loads('$phases_json')))")"

    log "Running $phase_count phases ..."
    mkdir -p "$RESULTS_DIR"

    for i in $(seq 0 $((phase_count - 1))); do
        local phase_name epp_config metrics_subsystem phase_dir
        phase_name="$(yaml_get "phases.$i.name")"
        epp_config="$(yaml_get "phases.$i.epp_config")"
        metrics_subsystem="$(yaml_get "phases.$i.metrics_subsystem" "program_aware")"
        phase_dir="$RESULTS_DIR/$phase_name"

        mkdir -p "$phase_dir"
        echo "$metrics_subsystem" > "$phase_dir/metrics_subsystem.txt"

        log ""
        log "========================================="
        log "  PHASE $((i+1))/$phase_count: $phase_name"
        log "========================================="

        switch_config "$epp_config" "$phase_name"

        # Restart port-forward (EPP pod changed after config switch)
        start_metrics_portforward

        # Start metrics scraper in background
        local scraper_pid=""
        if [ -n "$METRICS_PF_PID" ]; then
            log "Starting metrics scraper (subsystem=$metrics_subsystem) ..."
            "$PYTHON" "$SCRIPT_DIR/scrape_metrics.py" \
                --url "$METRICS_URL/metrics" \
                --subsystem "$metrics_subsystem" \
                --duration 86400 \
                --output "$phase_dir/metrics.jsonl" \
                > "$phase_dir/scraper.log" 2>&1 &
            scraper_pid=$!
            log "Metrics scraper started (PID $scraper_pid)"
        fi

        log "Running load generator ..."
        "$PYTHON" "$SCRIPT_DIR/loadgen.py" \
            --scenario "$SCRIPT_DIR/$SCENARIO" \
            --phase "$phase_name" \
            --gateway "$GATEWAY_URL" \
            --output "$phase_dir"

        # Stop scraper
        if [ -n "$scraper_pid" ] && kill -0 "$scraper_pid" 2>/dev/null; then
            log "Stopping metrics scraper ..."
            kill "$scraper_pid" 2>/dev/null || true
            wait "$scraper_pid" 2>/dev/null || true
        fi

        if [ $i -lt $((phase_count - 1)) ]; then
            log "Pausing 10s between phases ..."
            sleep 10
        fi
    done

    log ""
    log "Running analysis ..."
    "$PYTHON" "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"

    log ""
    log "=== All phases done. Results in $RESULTS_DIR ==="
    log "=== Plots in $RESULTS_DIR/plots/ ==="
}

main "$@"