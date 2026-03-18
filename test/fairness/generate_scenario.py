#!/usr/bin/env python3
"""Generate scenario YAML files for large-scale fairness benchmarks.

Supports three scenario types:
  steady      — all background, uniform programs
  waves       — background residents + foreground waves arriving over time
  production  — mixed bg/fg with varied token costs and randomized timing

Generated YAMLs are consumed by the existing run_test.sh + fairness_loadgen.py
+ analyze_results.py pipeline with no modifications.

Examples:
  python3 generate_scenario.py steady -n 50 --duration 600 -o scenarios/bench-steady-p50.yaml
  python3 generate_scenario.py waves -n 60 --bg-count 10 --num-waves 4 -o scenarios/bench-waves.yaml
  python3 generate_scenario.py production -n 50 --seed 42 -o scenarios/bench-prod.yaml
"""

import argparse
import math
import os
import random
import sys
from collections import OrderedDict

import yaml


# ---------------------------------------------------------------------------
# YAML helpers — preserve insertion order, avoid anchors/aliases
# ---------------------------------------------------------------------------

class _OrderedDumper(yaml.SafeDumper):
    pass

def _dict_representer(dumper, data):
    return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

_OrderedDumper.add_representer(OrderedDict, _dict_representer)

def _dump_yaml(data):
    return yaml.dump(data, Dumper=_OrderedDumper, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Capacity model (matches vllm-sim per-token latency model)
# ---------------------------------------------------------------------------

def estimate_request_time(prompt_tokens, max_tokens):
    """Estimated wall-clock seconds per request."""
    prefill_ms = 6.0 + prompt_tokens * 0.017
    generation_ms = max_tokens * 6.0
    return (prefill_ms + generation_ms) / 1000.0


def estimate_capacity(max_num_seqs, prompt_tokens, max_tokens):
    """Max sustained req/s for given concurrency and token profile."""
    return max_num_seqs / estimate_request_time(prompt_tokens, max_tokens)


def auto_max_num_seqs(total_rate, load_level, prompt_tokens, max_tokens):
    """Compute max_num_seqs so that capacity = total_rate / load_level, +25% headroom."""
    target_capacity = total_rate / load_level
    req_time = estimate_request_time(prompt_tokens, max_tokens)
    raw = target_capacity * req_time * 1.25
    return max(20, min(512, int(math.ceil(raw))))


def auto_concurrency(program_count):
    return min(50, 2000 // program_count)


def auto_warmup(program_count, rate_per_program):
    if rate_per_program <= 0:
        return 120
    return min(120, max(60, int(math.ceil(15.0 / rate_per_program)) + 30))


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------

STANDARD_PHASES = [
    OrderedDict([
        ("name", "program-aware"),
        ("epp_config", "configs/program-aware.yaml"),
        ("metrics_subsystem", "program_aware"),
    ]),
    OrderedDict([
        ("name", "program-aware-drr"),
        ("epp_config", "configs/program-aware-drr.yaml"),
        ("metrics_subsystem", "program_aware"),
    ]),
    OrderedDict([
        ("name", "round-robin"),
        ("epp_config", "configs/round-robin.yaml"),
        ("metrics_subsystem", "round_robin"),
    ]),
]


def _infra_block(max_num_seqs):
    return OrderedDict([
        ("kind", False),
        ("namespace", "default"),
        ("vllm", OrderedDict([
            ("extra_args", [
                "--latency-calculator=per-token",
                "--prefill-overhead=6ms",
                "--prefill-time-per-token=17us",
                "--inter-token-latency=6ms",
                "--inter-token-latency-std-dev=1ms",
                f"--max-num-seqs={max_num_seqs}",
                "--max-model-len=4096",
            ]),
        ])),
    ])


def _prog_name(index):
    return f"prog-{index:03d}"


def _round2(x):
    """Round to 2 decimal places."""
    return round(x, 2)


# ---------------------------------------------------------------------------
# Scenario: steady
# ---------------------------------------------------------------------------

def generate_steady(args):
    n = args.program_count
    prompt_tokens = args.prompt_tokens
    max_tokens = args.max_tokens

    # Compute rates
    if args.max_num_seqs:
        mns = args.max_num_seqs
    else:
        # first estimate total rate, then derive max_num_seqs
        # For steady: total_rate = n * rate_per_prog
        # We pick rate_per_prog so total_rate = capacity * load_level
        # Bootstrapping: assume default tokens to get initial capacity
        req_time = estimate_request_time(prompt_tokens, max_tokens)
        # target: capacity * load_level = total_rate = n * rate
        # capacity = mns / req_time
        # So mns = total_rate * req_time / load_level * 1.25
        # But total_rate = capacity * load_level = (mns / req_time) * load_level
        # We want a reasonable mns. Use heuristic: start from n programs at ~1 req/s
        initial_total_rate = max(n * 1.0, 10.0)
        mns = auto_max_num_seqs(initial_total_rate, args.load_level, prompt_tokens, max_tokens)

    capacity = estimate_capacity(mns, prompt_tokens, max_tokens)
    total_rate = capacity * args.load_level
    rate_per_prog = _round2(total_rate / n)
    # Recompute actual total and load
    actual_total = rate_per_prog * n
    actual_load = actual_total / capacity if capacity > 0 else 0

    warmup = args.warmup if args.warmup else auto_warmup(n, rate_per_prog)
    concurrency = auto_concurrency(n)

    programs = OrderedDict()
    for i in range(n):
        programs[_prog_name(i)] = OrderedDict([
            ("background", True),
            ("count", 1),
            ("rate", rate_per_prog),
            ("prompt_tokens", prompt_tokens),
            ("max_tokens", max_tokens),
        ])

    scenario = OrderedDict([
        ("name", args.name or f"bench-steady-p{n}"),
        ("model", "meta-llama/Llama-3.1-8B-Instruct"),
        ("infra", _infra_block(mns)),
        ("test", OrderedDict([
            ("duration", args.duration),
            ("warmup", warmup),
            ("concurrency", concurrency),
            ("timeout", 600),
        ])),
        ("programs", programs),
        ("phases", STANDARD_PHASES),
    ])

    comment = (
        f"# Generated scenario: steady\n"
        f"#\n"
        f"# {n} identical background programs, uniform token profile.\n"
        f"# Rate per program: {rate_per_prog} req/s | Total: {_round2(actual_total)} req/s\n"
        f"# Capacity: {_round2(capacity)} req/s (max-num-seqs={mns}) | Load: {_round2(actual_load * 100)}%\n"
        f"# Tokens: {prompt_tokens} prompt / {max_tokens} max | Request time: {_round2(estimate_request_time(prompt_tokens, max_tokens))}s\n"
        f"#\n"
    )

    return scenario, comment


# ---------------------------------------------------------------------------
# Scenario: waves
# ---------------------------------------------------------------------------

def generate_waves(args):
    n = args.program_count
    bg_count = args.bg_count
    fg_count = n - bg_count
    num_waves = args.num_waves
    prompt_tokens = args.prompt_tokens
    max_tokens = args.max_tokens
    duration = args.duration

    if fg_count <= 0:
        print(f"Error: program-count ({n}) must be > bg-count ({bg_count})", file=sys.stderr)
        sys.exit(1)
    if num_waves <= 0:
        print("Error: num-waves must be > 0", file=sys.stderr)
        sys.exit(1)

    # Distribute foreground programs across waves
    base_per_wave = fg_count // num_waves
    remainder = fg_count % num_waves
    wave_sizes = []
    for w in range(num_waves):
        wave_sizes.append(base_per_wave + (1 if w >= num_waves - remainder else 0))

    # Compute max_num_seqs
    if args.max_num_seqs:
        mns = args.max_num_seqs
    else:
        # Peak load: all programs active at once (conservative)
        peak_rate_estimate = n * 1.5
        mns = auto_max_num_seqs(peak_rate_estimate, args.load_level, prompt_tokens, max_tokens)

    capacity = estimate_capacity(mns, prompt_tokens, max_tokens)

    # Background rate: bg programs share ~40% of total load
    bg_share = 0.4
    bg_rate = _round2((capacity * args.load_level * bg_share) / bg_count) if bg_count > 0 else 0

    # Foreground rate: remaining load spread across avg active fg programs
    # Estimate avg active fg at any time ~ fg_count * 0.6
    fg_load = capacity * args.load_level * (1.0 - bg_share)
    avg_active_fg = max(1, fg_count * 0.6)
    base_fg_rate = _round2(fg_load / avg_active_fg)

    # Wave timing
    measurement_window = duration
    wave_spacing = measurement_window / (num_waves + 1)

    # Token profiles for --vary-tokens
    token_profiles = [
        (prompt_tokens, max_tokens),
        (prompt_tokens, max_tokens),
        (int(prompt_tokens * 1.33), int(max_tokens * 1.17)),
        (int(prompt_tokens * 0.67), int(max_tokens * 0.5)),
    ]

    warmup = args.warmup if args.warmup else auto_warmup(n, bg_rate)
    concurrency = auto_concurrency(n)

    programs = OrderedDict()

    # Background programs
    for i in range(bg_count):
        programs[_prog_name(i)] = OrderedDict([
            ("background", True),
            ("count", 1),
            ("rate", bg_rate),
            ("prompt_tokens", prompt_tokens),
            ("max_tokens", max_tokens),
        ])

    # Foreground waves
    prog_idx = bg_count
    wave_details = []
    for w in range(num_waves):
        wave_start = int(wave_spacing * (w + 1))
        # Later waves have shorter active windows
        wave_duration = max(60, int(measurement_window - wave_start - 30))
        wave_duration = min(wave_duration, measurement_window - wave_start)

        # Rate scales up slightly for later waves
        rate_scale = 1.0 + 0.2 * w
        wave_rate = _round2(base_fg_rate * rate_scale)

        if args.vary_tokens and w < len(token_profiles):
            w_prompt, w_max = token_profiles[w]
        else:
            w_prompt, w_max = prompt_tokens, max_tokens

        wave_details.append((w, wave_start, wave_duration, wave_sizes[w], wave_rate, w_prompt, w_max))

        for j in range(wave_sizes[w]):
            programs[_prog_name(prog_idx)] = OrderedDict([
                ("start_time", wave_start),
                ("duration", wave_duration),
                ("count", 1),
                ("rate", wave_rate),
                ("prompt_tokens", w_prompt),
                ("max_tokens", w_max),
            ])
            prog_idx += 1

    # Summary stats
    total_bg_rate = bg_rate * bg_count
    peak_fg_rate = sum(ws[3] * ws[4] for ws in wave_details)
    peak_total = total_bg_rate + peak_fg_rate
    actual_load = peak_total / capacity if capacity > 0 else 0

    scenario = OrderedDict([
        ("name", args.name or f"bench-waves-p{n}"),
        ("model", "meta-llama/Llama-3.1-8B-Instruct"),
        ("infra", _infra_block(mns)),
        ("test", OrderedDict([
            ("duration", duration),
            ("warmup", warmup),
            ("concurrency", concurrency),
            ("timeout", 600),
        ])),
        ("programs", programs),
        ("phases", STANDARD_PHASES),
    ])

    wave_lines = []
    for w, ws, wd, wn, wr, wp, wm in wave_details:
        wave_lines.append(f"#   Wave {w+1} (t={ws}s, {wn} progs, {wd}s): rate={wr}, {wp}/{wm} tokens")

    comment = (
        f"# Generated scenario: waves\n"
        f"#\n"
        f"# {bg_count} background + {fg_count} foreground programs in {num_waves} waves.\n"
        f"# Background rate: {bg_rate} req/s each ({_round2(total_bg_rate)} total)\n"
        + "\n".join(wave_lines) + "\n"
        f"# Peak total rate: {_round2(peak_total)} req/s | Capacity: {_round2(capacity)} req/s (max-num-seqs={mns})\n"
        f"# Peak load: {_round2(actual_load * 100)}%\n"
        f"#\n"
    )

    return scenario, comment


# ---------------------------------------------------------------------------
# Scenario: production
# ---------------------------------------------------------------------------

# Token profiles
HEAVY_PROFILE = (600, 512)
LIGHT_PROFILE = (150, 64)


def generate_production(args):
    n = args.program_count
    duration = args.duration
    bg_fraction = args.bg_fraction
    heavy_fraction = args.heavy_fraction
    seed = args.seed

    rng = random.Random(seed)

    bg_count = max(1, int(n * bg_fraction))
    fg_count = n - bg_count
    bg_heavy_count = max(0, int(bg_count * heavy_fraction))
    bg_light_count = bg_count - bg_heavy_count

    # Compute max_num_seqs from a blended average
    avg_prompt = int(HEAVY_PROFILE[0] * heavy_fraction + LIGHT_PROFILE[0] * (1 - heavy_fraction))
    avg_max = int(HEAVY_PROFILE[1] * heavy_fraction + LIGHT_PROFILE[1] * (1 - heavy_fraction))

    if args.max_num_seqs:
        mns = args.max_num_seqs
    else:
        peak_estimate = n * 1.2
        mns = auto_max_num_seqs(peak_estimate, args.load_level, avg_prompt, avg_max)

    # Capacity based on blended profile
    capacity = estimate_capacity(mns, avg_prompt, avg_max)
    target_total = capacity * args.load_level

    # Rate budget: heavy programs get lower rate, light programs get higher rate
    heavy_req_time = estimate_request_time(*HEAVY_PROFILE)
    light_req_time = estimate_request_time(*LIGHT_PROFILE)
    # Rate ratio: inversely proportional to request time
    rate_ratio = heavy_req_time / light_req_time  # heavy/light > 1 means heavy is slower
    # If we have H heavy and L light programs, total_rate = H * r_heavy + L * r_light
    # r_heavy = r_light / rate_ratio
    # target_total = H * (r_light / rate_ratio) + L * r_light
    # target_total = r_light * (H / rate_ratio + L)

    # Background rates
    if bg_count > 0:
        bg_budget = target_total * 0.5  # bg gets ~50% of budget
        if bg_light_count + bg_heavy_count / rate_ratio > 0:
            r_light_bg = bg_budget / (bg_light_count + bg_heavy_count / rate_ratio)
        else:
            r_light_bg = bg_budget / max(1, bg_count)
        r_heavy_bg = r_light_bg / rate_ratio
    else:
        r_light_bg = 0
        r_heavy_bg = 0

    # Foreground rates
    fg_budget = target_total * 0.5
    fg_heavy_count = max(0, int(fg_count * heavy_fraction))
    fg_light_count = fg_count - fg_heavy_count
    if fg_count > 0:
        # Not all fg programs active at once; assume ~60% overlap on average
        overlap_factor = 0.6
        effective_fg_light = fg_light_count * overlap_factor
        effective_fg_heavy = fg_heavy_count * overlap_factor
        denom = effective_fg_light + effective_fg_heavy / rate_ratio
        if denom > 0:
            r_light_fg = fg_budget / denom
        else:
            r_light_fg = fg_budget / max(1, fg_count)
        r_heavy_fg = r_light_fg / rate_ratio
    else:
        r_light_fg = 0
        r_heavy_fg = 0

    warmup = args.warmup if args.warmup else auto_warmup(n, min(r_light_bg, r_light_fg) if r_light_bg > 0 else r_light_fg)
    concurrency = auto_concurrency(n)

    programs = OrderedDict()
    prog_idx = 0

    # Background heavy
    for i in range(bg_heavy_count):
        programs[_prog_name(prog_idx)] = OrderedDict([
            ("background", True),
            ("count", 1),
            ("rate", _round2(r_heavy_bg)),
            ("prompt_tokens", HEAVY_PROFILE[0]),
            ("max_tokens", HEAVY_PROFILE[1]),
        ])
        prog_idx += 1

    # Background light
    for i in range(bg_light_count):
        programs[_prog_name(prog_idx)] = OrderedDict([
            ("background", True),
            ("count", 1),
            ("rate", _round2(r_light_bg)),
            ("prompt_tokens", LIGHT_PROFILE[0]),
            ("max_tokens", LIGHT_PROFILE[1]),
        ])
        prog_idx += 1

    # Foreground programs
    total_fg_rate = 0
    for i in range(fg_count):
        is_heavy = rng.random() < heavy_fraction
        profile = HEAVY_PROFILE if is_heavy else LIGHT_PROFILE
        rate = r_heavy_fg if is_heavy else r_light_fg

        start_time = rng.randint(0, max(1, int(duration * 0.7)))
        prog_duration = rng.randint(120, 300)
        # Clamp so program doesn't exceed measurement window
        prog_duration = min(prog_duration, duration - start_time)
        prog_duration = max(30, prog_duration)  # minimum 30s

        programs[_prog_name(prog_idx)] = OrderedDict([
            ("start_time", start_time),
            ("duration", prog_duration),
            ("count", 1),
            ("rate", _round2(rate)),
            ("prompt_tokens", profile[0]),
            ("max_tokens", profile[1]),
        ])
        total_fg_rate += rate
        prog_idx += 1

    # Summary stats
    total_bg_rate = r_heavy_bg * bg_heavy_count + r_light_bg * bg_light_count
    peak_total = total_bg_rate + total_fg_rate  # conservative (not all fg active simultaneously)
    actual_load = peak_total / capacity if capacity > 0 else 0

    scenario = OrderedDict([
        ("name", args.name or f"bench-prod-p{n}"),
        ("model", "meta-llama/Llama-3.1-8B-Instruct"),
        ("infra", _infra_block(mns)),
        ("test", OrderedDict([
            ("duration", duration),
            ("warmup", warmup),
            ("concurrency", concurrency),
            ("timeout", 600),
        ])),
        ("programs", programs),
        ("phases", STANDARD_PHASES),
    ])

    comment = (
        f"# Generated scenario: production\n"
        f"#\n"
        f"# {bg_count} background ({bg_heavy_count} heavy, {bg_light_count} light) + "
        f"{fg_count} foreground ({fg_heavy_count} heavy, {fg_light_count} light).\n"
        f"# BG rates: heavy={_round2(r_heavy_bg)} light={_round2(r_light_bg)} req/s | "
        f"FG rates: heavy={_round2(r_heavy_fg)} light={_round2(r_light_fg)} req/s\n"
        f"# Total BG: {_round2(total_bg_rate)} req/s | Capacity: {_round2(capacity)} req/s (max-num-seqs={mns})\n"
        f"# Seed: {seed}\n"
        f"#\n"
    )

    return scenario, comment


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate scenario YAML files for fairness benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-n", "--program-count", type=int, required=True,
                        help="Total number of programs")
    common.add_argument("--duration", type=int, default=600,
                        help="Measurement duration in seconds (default: 600)")
    common.add_argument("--warmup", type=int, default=None,
                        help="Warmup duration in seconds (auto-computed if omitted)")
    common.add_argument("--load-level", type=float, default=0.8,
                        help="Target load as fraction of capacity (default: 0.8)")
    common.add_argument("--prompt-tokens", type=int, default=300,
                        help="Prompt tokens per request (default: 300)")
    common.add_argument("--max-tokens", type=int, default=256,
                        help="Max output tokens per request (default: 256)")
    common.add_argument("--max-num-seqs", type=int, default=256,
                        help="max-num-seqs for vllm-sim (default: 256)")
    common.add_argument("--name", type=str, default=None,
                        help="Scenario name (auto-generated if omitted)")
    common.add_argument("-o", "--output", type=str, default=None,
                        help="Output YAML path (default: scenarios/bench-<type>-p<N>.yaml)")

    sub = parser.add_subparsers(dest="scenario_type", required=True)

    # steady
    sub.add_parser("steady", parents=[common],
                   help="All background, uniform programs")

    # waves
    waves_p = sub.add_parser("waves", parents=[common],
                             help="Background residents + foreground waves")
    waves_p.add_argument("--bg-count", type=int, default=10,
                         help="Number of always-on background programs (default: 10)")
    waves_p.add_argument("--num-waves", type=int, default=4,
                         help="Number of foreground waves (default: 4)")
    waves_p.add_argument("--vary-tokens", action="store_true",
                         help="Vary token profiles across waves")

    # production
    prod_p = sub.add_parser("production", parents=[common],
                            help="Mixed bg+fg with varied token costs")
    prod_p.add_argument("--bg-fraction", type=float, default=0.4,
                        help="Fraction of programs that are background (default: 0.4)")
    prod_p.add_argument("--heavy-fraction", type=float, default=0.3,
                        help="Fraction of programs with heavy token profile (default: 0.3)")
    prod_p.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Generate scenario
    generators = {
        "steady": generate_steady,
        "waves": generate_waves,
        "production": generate_production,
    }

    scenario, comment = generators[args.scenario_type](args)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"scenarios/bench-{args.scenario_type}-p{args.program_count}.yaml"

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write YAML with comment header
    yaml_text = _dump_yaml(scenario)
    with open(output_path, "w") as f:
        f.write(comment)
        f.write("\n")
        f.write(yaml_text)

    # Print summary
    test = scenario["test"]
    programs = scenario["programs"]
    infra_args = scenario["infra"]["vllm"]["extra_args"]
    mns = [a for a in infra_args if a.startswith("--max-num-seqs=")][0].split("=")[1]

    bg_progs = sum(1 for p in programs.values() if p.get("background", False))
    fg_progs = len(programs) - bg_progs
    total_rate = sum(p["rate"] * p.get("count", 1) for p in programs.values())

    print(f"Scenario: {scenario['name']}")
    print(f"  Type: {args.scenario_type}")
    print(f"  Programs: {len(programs)} ({bg_progs} bg, {fg_progs} fg)")
    print(f"  Total rate: {_round2(total_rate)} req/s")
    print(f"  max-num-seqs: {mns}")
    print(f"  Duration: {test['duration']}s | Warmup: {test['warmup']}s | Concurrency: {test['concurrency']}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
