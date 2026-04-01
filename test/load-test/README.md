# Load Test

End-to-end load testing framework for comparing EPP scheduling strategies. Runs identical workloads against different EPP configs and produces latency, fairness, and throughput plots.

## Running a Load Test

```bash
SCENARIO=scenarios/<name>.yaml ./run_test.sh
```

The script will:
1. Deploy a kind cluster and vllm-sim (if `infra.kind: true` in the scenario)
2. For each phase: swap the EPP config via configmap, run the load generator, and scrape metrics
3. Run analysis and generate comparison plots

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `GATEWAY_URL` | `http://localhost:30080` | Gateway endpoint |
| `METRICS_URL` | `http://localhost:9090` | Prometheus endpoint |
| `EPP_TAG` | — | EPP image tag (triggers a build if set) |

**Results** are saved to `results/<scenario-name>/<phase-name>/` containing:
- `results.jsonl` — per-request latencies, status, tokens
- `metrics.jsonl` — time-series Prometheus metrics
- `pick.jsonl` — EPP scheduling decisions
- plots (latency CDFs, fairness index, queue depth, etc.)

**`configs/`** contains the EPP EndpointPickerConfig YAMLs referenced by scenario phases:
- `program-aware.yaml` — EWMA wait-time strategy
- `program-aware-drr.yaml` — Deficit Round-Robin strategy
- `program-aware-throughput.yaml` — throughput-optimizing strategy
- `program-aware-service.yaml` — service-time tracking
- `round-robin.yaml` — simple round-robin
- `baseline.yaml` — no fairness policy

## Generating Scenarios with gen_scenario.py

```bash
python3 gen_scenario.py -i scenario-configs/<name>.yaml -o scenarios/<name>.yaml
```

Reads a scenario-config template from `scenario-configs/`, distributes programs by ratio, generates Poisson-spaced start times, and outputs a full scenario YAML with all phases.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `-i` | — | Input scenario-config YAML |
| `-o` | stdout | Output file path |
| `--seed` | `42` | Random seed for reproducibility |

Without `-i`, uses built-in default profiles and a 100-program production mix.

**`scenario-configs/`** contains pre-built templates:
- `starvation-test.yaml` — aggressive vs polite (starvation resistance)
- `homogeneous.yaml` — identical programs (JFI baseline)
- `burst-arrivals.yaml` — staggered arrival times
- `asymmetric-load.yaml` — heavy load asymmetry
- `hol.yaml` — head-of-line blocking
- `scale-stress.yaml` — 100+ programs
- `token-diversity.yaml` — varied token sizes
- `default-flow-impact.yaml` — anonymous vs identified programs
- `production-mix-{1,2,3}-{60,120}.yaml` — production-like mixes at different durations

## Scenario YAML Reference

Full scenario files live in `scenarios/` and are consumed by `run_test.sh`.

```yaml
name: simple-ab                              # scenario name (used as results dir)
model: meta-llama/Llama-3.1-8B-Instruct     # HuggingFace model ID
```

### `infra`

```yaml
infra:
  kind: true              # deploy a local kind cluster (false = use existing cluster)
  namespace: default      # Kubernetes namespace
  vllm:
    extra_args:           # extra args patched onto the vllm-sim deployment
      - --max-num-seqs=256
      - --max-model-len=4096
```

### `test.warmup`

Warmup requests run before the measurement window starts. Excluded from results.

```yaml
test:
  warmup:
    total_requests: 0     # number of warmup requests (0 to skip)
    concurrency: 4        # max in-flight warmup requests
    prompt_tokens: 128
    max_tokens: 64
```

### `programs`

Each key is a program name. All programs run concurrently during the measurement window.

```yaml
programs:
  worker-a:
    total_requests: 50          # exactly how many requests to send
    concurrency: 10             # max simultaneous in-flight requests
    prompt_tokens: 512          # fixed prompt token count
    max_tokens: 128             # fixed generation length (ignore_eos=true)
    start_time: 0               # seconds after measurement start to begin sending
    request_timeout: 60         # per-request HTTP timeout in seconds
    no_fairness_header: false   # if true, omit fairness header (tracked as "default-flow")
    initial_request_interval: 0 # seconds between first N requests (stagger initial burst)
```

### `phases`

Each phase runs the full program set against a different EPP config. Results are saved per phase.

```yaml
phases:
  - name: program-aware-ewma                    # phase name (used as results subdir)
    epp_config: configs/program-aware.yaml       # path to EPP config (relative to test/load-test/)
    metrics_subsystem: program_aware             # Prometheus metric prefix to scrape
```

### Example (`simple-ab.yaml`)

```yaml
name: simple-ab
model: meta-llama/Llama-3.1-8B-Instruct

infra:
  kind: true
  namespace: default
  vllm:
    extra_args:
      - --latency-calculator=per-token
      - --prefill-overhead=6ms
      - --prefill-time-per-token=17us
      - --inter-token-latency=6ms
      - --inter-token-latency-std-dev=1ms
      - --max-num-seqs=256
      - --max-model-len=4096

test:
  warmup:
    total_requests: 0
    concurrency: 4
    prompt_tokens: 128
    max_tokens: 64

programs:
  worker-a:
    total_requests: 50
    concurrency: 10
    prompt_tokens: 512
    max_tokens: 128
  worker-b:
    total_requests: 50
    concurrency: 10
    prompt_tokens: 512
    max_tokens: 128

phases:
  - name: program-aware-ewma
    epp_config: configs/program-aware.yaml
    metrics_subsystem: program_aware
  - name: program-aware-drr
    epp_config: configs/program-aware-drr.yaml
    metrics_subsystem: program_aware
  - name: round-robin
    epp_config: configs/round-robin.yaml
    metrics_subsystem: round_robin
```

## Scenario Config YAML Reference

Scenario-config files live in `scenario-configs/` and are consumed by `gen_scenario.py`.

### `profiles`

Named program profiles. Each profile defines the request shape and behavior.

```yaml
profiles:
  aggressive:
    prompt_tokens: 600        # fixed prompt token count
    max_tokens: 512           # fixed generation length
    concurrency: 30           # max simultaneous in-flight requests
    total_requests: 300       # how many requests each program with this profile sends
    request_timeout: 120      # per-request HTTP timeout in seconds
    no_fairness_header: false # if true, programs with this profile omit fairness header
```

### `scenario`

Controls how programs are generated and distributed.

```yaml
scenario:
  name: starvation-test                        # scenario name
  model: meta-llama/Llama-3.1-8B-Instruct     # HuggingFace model ID
  max_num_seqs: 16                             # patched onto vllm-sim --max-num-seqs
  total_programs: 2                            # total number of programs to generate
  window: 60                                   # arrival window in seconds (0 = all start at t=0)
  spread: 1                                    # multiplier for arrival window (effective window = window * spread)
  warmup:                                      # optional (defaults: 0 requests, conc=4, 128/64 tokens)
    total_requests: 0
    concurrency: 4
    prompt_tokens: 128
    max_tokens: 64
  programs:
    - profile: aggressive       # references a key from profiles
      ratio: 0.5               # fraction of total_programs with this profile
    - profile: polite
      ratio: 0.5
      window: [30, 60]         # optional per-entry arrival window override
```

### Example (`starvation-test.yaml`)

```yaml
profiles:
  aggressive:
    prompt_tokens: 600
    max_tokens: 512
    concurrency: 30
    total_requests: 300
    request_timeout: 120
  polite:
    prompt_tokens: 600
    max_tokens: 512
    concurrency: 2
    total_requests: 300
    request_timeout: 120

scenario:
  name: starvation-test
  model: meta-llama/Llama-3.1-8B-Instruct
  max_num_seqs: 16
  window: 0
  spread: 0
  total_programs: 2
  programs:
    - profile: aggressive
      ratio: 0.5
    - profile: polite
      ratio: 0.5
```