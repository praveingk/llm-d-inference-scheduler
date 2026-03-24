# Scenario YAML Schema

Each scenario file describes a load test: what programs send traffic, how many requests each sends, and which EPP configurations to compare.

## Top-level fields

```yaml
name: <string>          # Scenario name (used as results directory name)
model: <string>         # HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B-Instruct)
```

## infra

Infrastructure deployment settings.

```yaml
infra:
  kind: true/false      # If true, deploy a local kind cluster before running
  namespace: default    # Kubernetes namespace

  vllm:
    extra_args: []      # Extra args patched onto the vllm-sim deployment
                        # e.g. ["--max-num-seqs=64", "--latency-model-prefill-intercept=0.006"]
```

When `kind: false`, the script assumes a cluster is already running and reachable.

## test

Warmup settings. The measurement window starts after all warmup requests complete.

```yaml
test:
  warmup:
    total_requests: <int>   # Number of warmup requests to send before measurement
    concurrency: <int>      # Max simultaneous in-flight warmup requests
    prompt_tokens: <int>
    max_tokens: <int>
```

Warmup requests are sent to the gateway but excluded from `results.jsonl`. They heat up the system so the measurement window starts in steady state.

## programs

Each key is a program name. Programs run concurrently during the measurement window.

```yaml
programs:
  <name>:
    total_requests: <int>      # Exactly how many requests this program sends
    concurrency: <int>         # Max simultaneous in-flight requests.
                               # If all slots are full, new sends wait until a slot opens.
                               # No requests are dropped.
    prompt_tokens: <int>       # Number of tokens in the prompt (fixed)
    max_tokens: <int>          # Number of tokens to generate (fixed, ignore_eos=true enforced)
    start_time: <int>          # Seconds after measurement start to begin sending (default: 0)
    request_timeout: <int>     # Per-request HTTP timeout in seconds (default: 60)
    no_fairness_header: false  # If true, omit x-gateway-inference-fairness-id header.
                               # The request will be tracked as "default-flow" by the EPP.
```

Each program sends exactly `total_requests` requests as fast as `concurrency` allows, then stops. Programs without `start_time` begin immediately.

The `x-gateway-inference-fairness-id` header is set to the program name by default. If `no_fairness_header: true`, the upstream EPP assigns the flow to `"default-flow"`.

## phases

Each phase runs the full program set against a different EPP configuration. Results are saved per phase under `results/<scenario-name>/<phase-name>/`.

```yaml
phases:
  - name: <string>
    epp_config: configs/<file>.yaml   # Path to EPP EndpointPickerConfig (relative to scenario file)
    metrics_subsystem: <string>       # Prometheus metric prefix for this EPP config.
                                      # program_aware  → program_aware_* metrics
                                      # round_robin    → round_robin_* metrics
```

## Example

See `simple-ab.yaml` for a minimal two-program, two-phase comparison.
