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
├── generate_scenario.py   # Scenario generator for large-scale benchmarks
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

# Override the model (e.g. use the TinyLlama model from `make env-dev-kind`)
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 ./run_test.sh
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

## Generating scenarios

`generate_scenario.py` creates scenario YAMLs programmatically for large-scale
benchmarks (50-100 programs, 10-30 min durations). It auto-computes
`max-num-seqs`, per-program rates, concurrency, and warmup from the vllm-sim
per-token latency model. All generated scenarios include the 3 standard strategy
phases (program-aware EWMA, DRR, round-robin).

Three scenario types are supported:

```bash
# Steady-state: N identical background programs, uniform token profile
python3 generate_scenario.py steady \
  --program-count 50 --duration 600 --load-level 0.8 --max-num-seqs 256 \
  -o scenarios/bench-steady-p50.yaml

# Waves: background residents + foreground programs arriving in waves
python3 generate_scenario.py waves \
  --program-count 60 --bg-count 10 --num-waves 4 --vary-tokens --max-num-seqs 256 \
  -o scenarios/bench-waves-p60.yaml

# Production mix: simple (auto mix, backward compatible)
python3 generate_scenario.py production \
  -n 50 --bg-fraction 0.4 --heavy-fraction 0.3 --seed 42 \
  -o scenarios/bench-prod-simple.yaml

# Production mix: explicit phases with ramp-up pattern
python3 generate_scenario.py production \
  -n 60 --bg-fraction 0.3 --seed 42 \
  --bg-mix 'heavy-slow=0.3,medium-med=0.5,light-fast=0.2' \
  --phase '0-200:14:heavy-fast=0.4,medium-med=0.4,light-fast=0.2' \
  --phase '200-400:16:medium-med=0.3,light-fast=0.5,light-med=0.2' \
  --phase '400-600:12:light-fast=0.6,light-med=0.2,medium-slow=0.2' \
  -o scenarios/bench-prod-ramp.yaml

# Production mix: burst in the middle
python3 generate_scenario.py production \
  -n 50 --bg-fraction 0.4 --seed 42 \
  --phase '0-150:5:light-med=0.6,medium-med=0.4' \
  --phase '150-400:15:heavy-fast=0.4,medium-fast=0.3,light-fast=0.3' \
  --phase '400-600:5:light-slow=0.5,medium-slow=0.3,light-med=0.2' \
  -o scenarios/bench-prod-burst.yaml

# Production mix: custom rate tiers
python3 generate_scenario.py production \
  -n 40 --bg-fraction 0.4 --seed 42 \
  --rate-fast 15 --rate-med 8 --rate-slow 2 \
  --phase '0-600:24:heavy-med=0.3,medium-med=0.4,light-fast=0.3' \
  -o scenarios/bench-prod-custom-rates.yaml
```

Production scenarios support 9 compound profiles (`{heavy,medium,light}-{fast,med,slow}`)
combining 3 token tiers with 3 rate tiers. Use `--phase` to control foreground
temporal patterns, `--bg-mix` for background distribution, and
`--rate-fast/--rate-med/--rate-slow` to override rate tier values.

Common options: `--duration`, `--warmup`, `--load-level`, `--prompt-tokens`,
`--max-tokens`, `--max-num-seqs` (default: 256). Run
`python3 generate_scenario.py <type> --help` for all options.

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