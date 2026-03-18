# Fairness A/B Test

End-to-end test harness for comparing EPP scheduling strategies under
multi-program workloads. It measures per-program latency fairness across
different configurations (e.g. program-aware vs round-robin vs baseline).

## Structure

```
test/fairness/
├── run_test.sh                    # Orchestrator — deploys infra, runs phases, triggers analysis
├── fairness_loadgen.py            # Async load generator (one sender per program instance)
├── scrape_metrics_realtime.py     # Real-time Prometheus metrics scraper (NEW)
├── analyze_results.py             # Post-hoc analysis: latency tables, CDF/bar plots, fairness ratios
├── generate_scenario.py           # Scenario generator for large-scale benchmarks
├── scenarios/                     # Scenario YAML files defining workload profiles
│   ├── stress-h100.yaml
│   └── uniform-fairness.yaml
├── configs/                       # EPP config files switched between phases
│   ├── program-aware.yaml
│   ├── baseline.yaml
│   └── round-robin.yaml
└── results/                       # Generated — per-phase JSONL results + comparison plots
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
   - **Starts real-time metrics scraper** to collect Jain's fairness index every second (NEW).
   - Runs the load generator, which spawns async senders per program instance at configured rates.
   - Each request carries an `x-gateway-inference-fairness-id` header identifying its program.
   - Results are written as JSONL to `results/<phase>/results.jsonl`.
   - Metrics timeseries are written to `results/<phase>/metrics_timeseries.jsonl` (NEW).
4. **Analysis** — `analyze_results.py` auto-discovers phase directories and produces:
   - Per-program latency comparison tables (P50/P95/P99)
   - CDF and bar chart plots
   - Fairness ratio analysis (cross-program and cross-phase)
   - Throughput summary
   - **Fairness index timeseries plots** showing how fairness changes over time (NEW)

## New Feature: Real-time Fairness Tracking

The test harness now includes real-time tracking of Jain's fairness index throughout each phase:

### What's New

1. **Real-time Metrics Scraper** (`scrape_metrics_realtime.py`):
   - Scrapes Prometheus metrics every second during the test
   - Captures fairness index, request counts, and wait times
   - Outputs timestamped data to `metrics_timeseries.jsonl`

2. **Fairness Timeseries Plots**:
   - **Per-phase line graphs**: Shows fairness index evolution for each phase separately
   - **Overlay graph**: Compares fairness trends across all phases on a single plot
   - Includes reference line at 1.0 (perfect fairness)

3. **Automatic Integration**:
   - Scraper starts automatically when `run_test.sh` begins each phase
   - Runs in parallel with the load generator
   - Analysis script automatically detects and plots timeseries data

### Output Files

For each phase, you'll now find:
- `results/<phase>/metrics_timeseries.jsonl` - Raw timestamped metrics
- `results/<phase>/scraper.log` - Scraper execution log
- `results/comparison/fairness_index_timeseries.png` - Per-phase line graphs
- `results/comparison/fairness_index_timeseries.txt` - Tabular data
- `results/comparison/fairness_index_overlay.png` - All phases overlaid

### Example Usage

```bash
# Run a test with real-time fairness tracking
SCENARIO=scenarios/bench-steady-p50.yaml ./run_test.sh

# After completion, view the fairness evolution plots
open results/bench-steady-p50/comparison/fairness_index_timeseries.png
open results/bench-steady-p50/comparison/fairness_index_overlay.png
```

### Manual Scraper Usage

You can also run the scraper independently:

```bash
python3 scrape_metrics_realtime.py \
    --metrics-url http://localhost:9090/metrics \
    --output my_metrics.jsonl \
    --duration 300 \
    --subsystem program_aware \
    --interval 1.0
```

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

# Production: two phases with default-flow baseline
python3 generate_scenario.py production --seed 42 \
  --phase '0-300:15:heavy-fast=0.5,light-slow=0.5|df:rate=8,prompt=300,max=256' \
  --phase '300-600:15:medium-med=1.0|df:rate=2,prompt=150,max=64' \
  -o scenarios/bench-prod-phases.yaml

# Production: single phase, no default-flow
python3 generate_scenario.py production --seed 42 \
  --phase '0-600:30:heavy-med=0.33,medium-med=0.34,light-med=0.33' \
  -o scenarios/bench-prod-simple.yaml

# Production: burst in the middle with custom rate tiers
python3 generate_scenario.py production --seed 42 \
  --rate-fast 15 --rate-med 8 --rate-slow 2 \
  --phase '0-150:5:light-med=0.6,medium-med=0.4|df:rate=5,prompt=300,max=256' \
  --phase '150-400:15:heavy-fast=0.4,medium-fast=0.3,light-fast=0.3|df:rate=10,prompt=300,max=256' \
  --phase '400-600:5:light-slow=0.5,medium-slow=0.3,light-med=0.2|df:rate=3,prompt=150,max=64' \
  -o scenarios/bench-prod-burst.yaml
```

Production scenarios use `--phase` flags (required, repeatable) to define foreground
program phases. Each phase specifies a time window, program count, and profile mix.
Append `|df:rate=R,prompt=P,max=M` to add a `default-flow` baseline program for
that phase. 9 compound profiles (`{heavy,medium,light}-{fast,med,slow}`) combine
3 token tiers with 3 rate tiers. Override rate tier values with
`--rate-fast/--rate-med/--rate-slow`.

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
    metrics_subsystem: program_aware
  - name: baseline
    epp_config: configs/baseline.yaml
    metrics_subsystem: program_aware