#!/usr/bin/env python3
"""
Scenario generator for test/load-test.

Edit PROGRAM_PROFILES and SCENARIO_CONFIG below, then run:

    python3 gen_scenario.py                          # prints to stdout
    python3 gen_scenario.py -o scenarios/my-test.yaml
    python3 gen_scenario.py --seed 42 -o scenarios/my-test.yaml

Each workload phase defines programs that arrive during a time window with a
mix of named profiles. Start times within each window follow a Poisson process
(uniformly spaced with random jitter) for realistic staggering.

EPP phases generated: program-aware, program-aware-drr, round-robin
"""

import argparse
import math
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
        "prompt_tokens": 600,
        "max_tokens":    512,
        "rate":          10.0,
        "concurrency":   30,
    },
    "heavy-polite": {
        "prompt_tokens": 600,
        "max_tokens":    512,
        "rate":          10.0,
        "concurrency":   2,
    },
    "heavy-slow": {
        "prompt_tokens": 600,
        "max_tokens":    512,
        "rate":          1.0,
        "concurrency":   4,
    },
    "medium-normal": {
        "prompt_tokens": 300,
        "max_tokens":    256,
        "rate":          5.0,
        "concurrency":   8,
    },
    "medium-aggressive": {
        "prompt_tokens": 300,
        "max_tokens":    256,
        "rate":          5.0,
        "concurrency":   20,
    },
    "light-fast": {
        "prompt_tokens": 150,
        "max_tokens":    64,
        "rate":          10.0,
        "concurrency":   8,
    },
    "light-slow": {
        "prompt_tokens": 150,
        "max_tokens":    64,
        "rate":          1.0,
        "concurrency":   2,
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
        "seconds":       30,
        "rate":          2.0,
        "prompt_tokens": 128,
        "max_tokens":    64,
    },

    # Total measurement duration (seconds). Must be >= end of last phase.
    "duration": 600,

    # Per-request timeout (seconds).
    "timeout": 60,

    # Load level: fraction of simulator capacity to target (0.0–1.0).
    # Controls max_num_seqs in vllm-sim extra_args.
    #   0.75 → moderate pressure, 25% spare capacity
    #   0.95 → near-saturated, heavy queuing
    "load_level": 0.75,

    # Workload phases — programs arrive during [start, end) seconds.
    # mix: {profile_name: fraction}  fractions must sum to 1.0
    # programs: total number of sender programs in this phase
    "phases": [
        {
            "start":    0,
            "end":      300,
            "programs": 10,
            "mix": {
                "heavy-aggressive": 0.3,
                "medium-normal":    0.5,
                "light-slow":       0.2,
            },
        },
        {
            "start":    300,
            "end":      600,
            "programs": 15,
            "mix": {
                "light-fast":       0.5,
                "heavy-polite":     0.3,
                "medium-normal":    0.2,
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Capacity model (matches vllm-sim per-token latency defaults)
# ---------------------------------------------------------------------------

def estimate_request_time(prompt_tokens: int, max_tokens: int) -> float:
    """Estimated seconds per request under the default vllm-sim latency model."""
    prefill_ms    = 6.0 + prompt_tokens * 0.017
    generation_ms = max_tokens * 6.0
    return (prefill_ms + generation_ms) / 1000.0


def auto_max_num_seqs(total_rate: float, load_level: float,
                      prompt_tokens: int, max_tokens: int) -> int:
    """Compute max_num_seqs so capacity ≈ total_rate / load_level, +25% headroom."""
    req_time        = estimate_request_time(prompt_tokens, max_tokens)
    target_capacity = total_rate / max(load_level, 0.01)
    return max(4, int(math.ceil(target_capacity * req_time)))


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

def distribute_programs(n: int, mix: Dict[str, float],
                        rng: random.Random) -> List[str]:
    """
    Return a list of n profile names drawn according to mix fractions.
    Uses round-then-assign-remainder to guarantee exactly n entries.
    """
    profiles: List[str] = []
    remainder = n
    items     = sorted(mix.items(), key=lambda x: -x[1])  # largest fraction first
    for i, (profile, frac) in enumerate(items):
        count = remainder if i == len(items) - 1 else round(frac * n)
        count = max(0, min(count, remainder))
        profiles.extend([profile] * count)
        remainder -= count
    rng.shuffle(profiles)
    return profiles


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(cfg: dict, profiles: Dict[str, dict], seed: int) -> dict:
    rng      = random.Random(seed)
    duration = cfg["duration"]
    timeout  = cfg["timeout"]
    warmup   = cfg["warmup"]
    load_lvl = cfg["load_level"]

    programs: Dict[str, dict] = {}
    all_rates: List[float]    = []

    # Blended token profile for capacity estimation.
    total_prompt_w = 0.0
    total_max_w    = 0.0
    total_weight   = 0.0

    phase_counter: Dict[str, int] = {}

    for ph_idx, phase in enumerate(cfg["phases"]):
        p_start      = float(phase["start"])
        p_end        = float(phase["end"])
        n            = int(phase["programs"])
        mix          = phase["mix"]
        prog_duration = int(p_end - p_start)

        for pname in mix:
            if pname not in profiles:
                raise ValueError(f"Unknown profile '{pname}'. Add it to PROGRAM_PROFILES.")

        profile_list = distribute_programs(n, mix, rng)
        start_times  = poisson_start_times(n, p_start, p_end, rng)

        for profile_name, t_start in zip(profile_list, start_times):
            p = profiles[profile_name]

            key  = f"p{ph_idx}-{profile_name}"
            idx  = phase_counter.get(key, 0)
            name = f"fg-{key}-{idx:03d}"
            phase_counter[key] = idx + 1

            programs[name] = {
                "rate":          p["rate"],
                "concurrency":   p["concurrency"],
                "prompt_tokens": p["prompt_tokens"],
                "max_tokens":    p["max_tokens"],
                "start_time":    int(t_start),
                "duration":      prog_duration,
            }

            all_rates.append(p["rate"])
            w = p["rate"] * estimate_request_time(p["prompt_tokens"], p["max_tokens"])
            total_prompt_w += p["prompt_tokens"] * w
            total_max_w    += p["max_tokens"]    * w
            total_weight   += w

    # Blended token profile for max_num_seqs.
    if total_weight > 0:
        blended_prompt = int(total_prompt_w / total_weight)
        blended_max    = int(total_max_w    / total_weight)
    else:
        blended_prompt, blended_max = 300, 256

    total_rate   = sum(all_rates)
    max_num_seqs = auto_max_num_seqs(total_rate, load_lvl, blended_prompt, blended_max)

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
            "duration": duration,
            "timeout":  timeout,
            "warmup":   warmup,
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
    header = (
        f"# Generated scenario: {scenario['name']}\n"
        f"# Programs: {len(scenario['programs'])}  "
        f"Duration: {scenario['test']['duration']}s  "
        f"max-num-seqs: {scenario['infra']['vllm']['extra_args'][-2].split('=')[1]}\n"
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