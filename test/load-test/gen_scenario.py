#!/usr/bin/env python3
"""
Scenario generator for test/load-test.

Edit PROGRAM_PROFILES and SCENARIO_CONFIG below, then run:

    python3 gen_scenario.py                          # prints to stdout
    python3 gen_scenario.py -o scenarios/my-test.yaml
    python3 gen_scenario.py --seed 42 -o scenarios/my-test.yaml

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
# *** PROGRAM PROFILES — edit to add/change profiles ***
#
# concurrency controls sender pipelining:
#   low (1-2)  → near-sequential, polite sender
#   medium     → moderate pipelining
#   high       → aggressive, keeps EPP queue full
# ---------------------------------------------------------------------------

PROGRAM_PROFILES: Dict[str, dict] = {
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


# ---------------------------------------------------------------------------
# *** SCENARIO CONFIG — edit to define the workload ***
# ---------------------------------------------------------------------------

SCENARIO_CONFIG = {
    "name":  "production-mix",
    "model": "meta-llama/Llama-3.1-8B-Instruct",

    # Warmup before measurement window.
    "warmup": {
        "total_requests": 0,
        "concurrency":    4,
        "prompt_tokens":  128,
        "max_tokens":     64,
    },

    # Window for spreading program start times (seconds).
    "window": 60,

    # Fraction of window over which start times are distributed.
    "spread": 1,

    # Total number of sender programs (scale knob).
    "total_programs": 100,

    # vllm-sim concurrency limit (written into extra_args directly).
    "max_num_seqs": 64,

    # Program mix by profile — ratios must sum to 1.0.
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
                        rng: random.Random) -> List[str]:
    """
    Return a list of n profile names drawn according to ratio fractions.
    Uses round-then-assign-remainder to guarantee exactly n entries.
    """
    profiles: List[str] = []
    remainder = n
    # Sort largest ratio first for stable assignment.
    items = sorted(mix, key=lambda x: -x["ratio"])
    for i, entry in enumerate(items):
        count = remainder if i == len(items) - 1 else round(entry["ratio"] * n)
        count = max(0, min(count, remainder))
        profiles.extend([entry["profile"]] * count)
        remainder -= count
    rng.shuffle(profiles)
    return profiles


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
            raise ValueError(f"Unknown profile '{pname}'. Add it to PROGRAM_PROFILES.")

    # Expand total_programs by ratios.
    profile_list = distribute_programs(total_progs, cfg["programs"], rng)

    # Generate Poisson start times across [0, window * spread].
    spread_end   = window * spread
    start_times  = poisson_start_times(total_progs, 0, spread_end, rng)

    programs: Dict[str, dict] = {}
    counter: Dict[str, int] = {}

    for profile_name, t_start in zip(profile_list, start_times):
        p = profiles[profile_name]

        idx  = counter.get(profile_name, 0)
        name = f"fg-{profile_name}-{idx:03d}"
        counter[profile_name] = idx + 1

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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a load-test scenario YAML")
    parser.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")
    parser.add_argument("-s", "--seed", type=int, default=42,  help="Random seed (default: 42)")
    args = parser.parse_args()

    result = generate(SCENARIO_CONFIG, PROGRAM_PROFILES, seed=args.seed)
    text   = dump_scenario(result)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Written to {args.output}  ({len(result['programs'])} programs)", file=sys.stderr)
    else:
        print(text)
