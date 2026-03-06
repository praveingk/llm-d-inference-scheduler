# Fairness A/B Test

End-to-end test harness for comparing EPP scheduling strategies under
multi-program workloads. It measures per-program latency fairness across
different configurations (e.g. program-aware vs round-robin vs baseline).

## Structure

```
test/fairness/
├── run_test.sh            # Orchestrator — deploys infra, runs phases, triggers analysis
├── fairness_loadgen.py    # Async load generator (one sender per program instance)
├── analyze_results.py     # Post-hoc analysis: latency tables, CDF/bar plots, fairness ratios
├── scenarios/             # Scenario YAML files defining workload profiles
│   ├── stress-h100.yaml
│   └── uniform-fairness.yaml
├── configs/               # EPP config files switched between phases
│   ├── program-aware.yaml
│   ├── baseline.yaml
│   └── round-robin.yaml
└── results/               # Generated — per-phase JSONL results + comparison plots
```

## Prerequisites

- Python 3.9+ with `aiohttp`, `numpy`, `matplotlib`, `pyyaml`
- `kubectl`, `kind` (if using `infra.kind: true` in the scenario)

## Quick start

```bash
# Run the default scenario (stress-h100) with a local kind cluster
./run_test.sh

# Run a specific scenario
SCENARIO=scenarios/uniform-fairness.yaml ./run_test.sh
```

### Using an existing cluster (no kind)

If you already have an EPP + vllm deployment running, set `infra.kind: false`
in your scenario YAML and pass the gateway URL via the `infra.gateway_url` field
or the environment variable:

```bash
# Via scenario YAML
infra:
  kind: false
  gateway_url: http://my-gateway.example.com:8080

# Or override with an environment variable
GATEWAY_URL=http://my-gateway.example.com:8080 ./run_test.sh
```

The script still needs `kubectl` access to the target cluster to switch EPP
configs between phases. Set `infra.namespace` in the scenario if the deployment
is not in `default`.

## How it works

1. **Infra setup** — Optionally deploys a kind cluster with vllm-sim and the EPP.
2. **Simulator tuning** — Patches the vllm-sim deployment with scenario-specific args (latency model, max seqs, etc.).
3. **Phase loop** — For each phase defined in the scenario YAML:
   - Switches the EPP ConfigMap to the phase's config and restarts the EPP.
   - Runs the load generator, which spawns async senders per program instance at configured rates.
   - Each request carries an `x-gateway-inference-fairness-id` header identifying its program.
   - Results are written as JSONL to `results/<phase>/results.jsonl`.
4. **Analysis** — `analyze_results.py` auto-discovers phase directories and produces:
   - Per-program latency comparison tables (P50/P95/P99)
   - CDF and bar chart plots
   - Fairness ratio analysis (cross-program and cross-phase)
   - Throughput summary

## Scenario YAML format

```yaml
model: Qwen/Qwen2-0.5B-Instruct

infra:
  kind: true
  vllm:
    extra_args: ["--max-num-seqs=32", ...]

test:
  duration: 60      # measurement window (seconds)
  warmup: 15        # warmup excluded from analysis
  concurrency: 50   # max in-flight requests per instance
  timeout: 60       # per-request timeout

programs:
  prog-heavy:
    count: 5         # number of sender instances
    rate: 5          # requests/sec per instance
    prompt_tokens: 700
    max_tokens: 1024

phases:
  - name: program-aware
    epp_config: configs/program-aware.yaml
  - name: baseline
    epp_config: configs/baseline.yaml
```