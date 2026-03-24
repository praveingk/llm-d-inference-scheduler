#!/usr/bin/env python3
"""
Scenario generator for test/load-test.

Reads an input YAML that defines program profiles and a scenario config,
then generates a full scenario YAML consumable by loadgen.py / run_test.sh.

Usage:
    python3 gen_scenario.py -i scenario-configs/starvation-test.yaml
    python3 gen_scenario.py -i scenario-configs/starvation-test.yaml -o scenarios/starvation-test.yaml
    python3 gen_scenario.py -i scenario-configs/starvation-test.yaml --seed 42 -o scenarios/out.yaml

    # Backward compat (uses built-in defaults):
    python3 gen_scenario.py -o scenarios/production-mix.yaml

Input YAML structure:

    profiles:
      aggressive:
        prompt_tokens: 600
        max_tokens: 512
        concurrency: 30
        total_requests: 300
        request_timeout: 120

    scenario:
      name: starvation-test
      model: meta-llama/Llama-3.1-8B-Instruct
      max_num_seqs: 16
      window: 60
      spread: 1
      total_programs: 2
      warmup:
        total_requests: 0
        concurrency: 4
        prompt_tokens: 128
        max_tokens: 64
      programs:
        - profile: aggressive
          ratio: 0.5
        - profile: polite
          ratio: 0.5
          window: [30, 60]   # optional per-entry arrival window

Programs arrive with Poisson-spaced start times across a configurable window.
Each program sends exactly total_requests as fast as concurrency allows.

EPP phases generated: program-aware, program-aware-drr, round-robin
"""

import argparse
import random
import sys
from typing import Dict, List

import yaml


# ---------------------------------------------------------------------------
# Built-in defaults (used when no -i input YAML is provided)
# ---------------------------------------------------------------------------

_DEFAULT_PROFILES: Dict[str, dict] = {
    "heavy-aggressive": {
        "prompt_tokens":   600,
        "max_tokens":      512,
        "concurrency":     30,
        "total_requests":  500,
        "request_timeout": 120,
    },
    "heavy-polite": {
        "prompt_tokens":   600,
        "max_tokens":      512,
        "concurrency":     2,
        "total_requests":  500,
        "request_timeout": 120,
    },
    "heavy-slow": {
        "prompt_tokens":   600,
        "max_tokens":      512,
        "concurrency":     4,
        "total_requests":  200,
        "request_timeout": 120,
    },
    "medium-normal": {
        "prompt_tokens":   300,
        "max_tokens":      256,
        "concurrency":     8,
        "total_requests":  300,
        "request_timeout": 60,
    },
    "medium-aggressive": {
        "prompt_tokens":   300,
        "max_tokens":      256,
        "concurrency":     20,
        "total_requests":  300,
        "request_timeout": 60,
    },
    "light-fast": {
        "prompt_tokens":   150,
        "max_tokens":      64,
        "concurrency":     8,
        "total_requests":  400,
        "request_timeout": 60,
    },
    "light-slow": {
        "prompt_tokens":   150,
        "max_tokens":      64,
        "concurrency":     2,
        "total_requests":  100,
        "request_timeout": 60,
    },
}

_DEFAULT_SCENARIO = {
    "name":  "production-mix",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "warmup": {
        "total_requests": 0,
        "concurrency":    4,
        "prompt_tokens":  128,
        "max_tokens":     64,
    },
    "window": 60,
    "spread": 1,
    "total_programs": 100,
    "max_num_seqs": 64,
    "programs": [
        {"profile": "heavy-aggressive", "ratio": 0.3},
        {"profile": "medium-normal",    "ratio": 0.5},
        {"profile": "light-slow",       "ratio": 0.2},
    ],
}


# ---------------------------------------------------------------------------
# Poisson arrival times
# ---------------------------------------------------------------------------

def poisson_start_times(n: int, window_start: float, window_end: float,
                        rng: random.Random) -> List[float]:
    """
    n start times in [window_start, window_end) with uniform spacing + ±50% jitter.
    Sorted and clamped to the window.
    """
    if n <= 0:
        return []
    span = window_end - window_start
    if span <= 0:
        return [round(window_start)] * n
    gap  = span / n
    times = []
    for i in range(n):
        base   = window_start + i * gap
        jitter = rng.uniform(-gap * 0.5, gap * 0.5)
        t      = max(window_start, min(window_end - 1, base + jitter))
        times.append(round(t))
    times.sort()
    return times


# ---------------------------------------------------------------------------
# Profile distribution helper
# ---------------------------------------------------------------------------

def distribute_programs(n: int, mix: List[dict],
                        rng: random.Random) -> List[dict]:
    """
    Return a list of n entries, each a dict with 'profile' and 'window' (if present).
    Entries are drawn according to ratio fractions.
    Uses round-then-assign-remainder to guarantee exactly n entries.
    """
    result: List[dict] = []
    remainder = n
    # Sort largest ratio first for stable assignment.
    items = sorted(mix, key=lambda x: -x["ratio"])
    for i, entry in enumerate(items):
        count = remainder if i == len(items) - 1 else round(entry["ratio"] * n)
        count = max(0, min(count, remainder))
        result.extend([entry] * count)
        remainder -= count
    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(cfg: dict, profiles: Dict[str, dict], seed: int) -> dict:
    rng           = random.Random(seed)
    warmup        = cfg["warmup"]
    window        = cfg["window"]
    spread        = cfg["spread"]
    total_progs   = cfg["total_programs"]
    max_num_seqs  = cfg["max_num_seqs"]

    # Validate profiles.
    for entry in cfg["programs"]:
        pname = entry["profile"]
        if pname not in profiles:
            raise ValueError(f"Unknown profile '{pname}'. Define it in the profiles section.")

    # Expand total_programs by ratios.  Each element carries the original
    # program entry (with optional per-entry window).
    assigned = distribute_programs(total_progs, cfg["programs"], rng)

    # Group programs by their window range, generate start times per group,
    # then zip back together.
    global_window = [0, window * spread]

    # Group by window range to call poisson_start_times per group.
    from collections import defaultdict
    groups: Dict[tuple, List[int]] = defaultdict(list)  # window_tuple -> [indices]
    for idx, entry in enumerate(assigned):
        win = entry.get("window", global_window)
        win_key = (win[0], win[1])
        groups[win_key].append(idx)

    start_time_by_idx: Dict[int, float] = {}
    for win_key, indices in groups.items():
        times = poisson_start_times(len(indices), win_key[0], win_key[1], rng)
        for i, t in zip(indices, times):
            start_time_by_idx[i] = t

    programs: Dict[str, dict] = {}
    counter: Dict[str, int] = {}

    for idx, entry in enumerate(assigned):
        profile_name = entry["profile"]
        p = profiles[profile_name]
        t_start = start_time_by_idx[idx]

        pidx = counter.get(profile_name, 0)
        name = f"fg-{profile_name}-{pidx:03d}"
        counter[profile_name] = pidx + 1

        programs[name] = {
            "total_requests":    p["total_requests"],
            "concurrency":       p["concurrency"],
            "prompt_tokens":     p["prompt_tokens"],
            "max_tokens":        p["max_tokens"],
            "start_time":        int(t_start),
            "request_timeout":   p["request_timeout"],
            "no_fairness_header": p.get("no_fairness_header", False),
        }

    return {
        "name":  cfg["name"],
        "model": cfg["model"],

        "infra": {
            "kind":      True,
            "namespace": "default",
            "vllm": {
                "extra_args": [
                    "--latency-calculator=per-token",
                    "--prefill-overhead=6ms",
                    "--prefill-time-per-token=17us",
                    "--inter-token-latency=6ms",
                    "--inter-token-latency-std-dev=1ms",
                    f"--max-num-seqs={max_num_seqs}",
                    "--max-model-len=4096",
                ],
            },
        },

        "test": {
            "warmup": warmup,
        },

        "programs": programs,

        "phases": [
            {"name": "program-aware",     "epp_config": "configs/program-aware.yaml",     "metrics_subsystem": "program_aware"},
            {"name": "program-aware-drr", "epp_config": "configs/program-aware-drr.yaml", "metrics_subsystem": "program_aware"},
            {"name": "round-robin",       "epp_config": "configs/round-robin.yaml",        "metrics_subsystem": "round_robin"},
        ],
    }


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------

def _ordered_representer(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


def dump_scenario(scenario: dict) -> str:
    yaml.add_representer(dict, _ordered_representer)
    # Extract max-num-seqs from extra_args for header.
    max_num_seqs = "?"
    for arg in scenario["infra"]["vllm"]["extra_args"]:
        if arg.startswith("--max-num-seqs="):
            max_num_seqs = arg.split("=")[1]
    total_reqs = sum(p["total_requests"] for p in scenario["programs"].values())
    header = (
        f"# Generated scenario: {scenario['name']}\n"
        f"# Programs: {len(scenario['programs'])}  "
        f"total_requests: {total_reqs}  "
        f"max-num-seqs: {max_num_seqs}\n"
    )
    return header + yaml.dump(scenario, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Input YAML loader
# ---------------------------------------------------------------------------

def load_input_yaml(path: str) -> tuple:
    """Load an input YAML and return (profiles, scenario_config)."""
    with open(path) as f:
        data = yaml.safe_load(f)

    profiles = data.get("profiles", {})
    scenario = data.get("scenario", {})

    if not profiles:
        raise ValueError(f"Input YAML '{path}' missing 'profiles' section.")
    if not scenario:
        raise ValueError(f"Input YAML '{path}' missing 'scenario' section.")

    # Validate required scenario fields.
    required = ["name", "model", "total_programs", "max_num_seqs", "programs"]
    for key in required:
        if key not in scenario:
            raise ValueError(f"Input YAML scenario missing required field '{key}'.")

    # Apply defaults for optional fields.
    scenario.setdefault("window", 60)
    scenario.setdefault("spread", 1)
    scenario.setdefault("warmup", {
        "total_requests": 0,
        "concurrency": 4,
        "prompt_tokens": 128,
        "max_tokens": 64,
    })

    return profiles, scenario


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a load-test scenario YAML")
    parser.add_argument("-i", "--input", default=None,
                        help="Input YAML with profiles and scenario config")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file (default: stdout)")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.input:
        profiles, scenario_cfg = load_input_yaml(args.input)
    else:
        profiles     = _DEFAULT_PROFILES
        scenario_cfg = _DEFAULT_SCENARIO

    result = generate(scenario_cfg, profiles, seed=args.seed)
    text   = dump_scenario(result)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Written to {args.output}  ({len(result['programs'])} programs)", file=sys.stderr)
    else:
        print(text)