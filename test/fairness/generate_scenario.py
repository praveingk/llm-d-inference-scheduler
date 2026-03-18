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
  python3 generate_scenario.py production -n 50 --seed 42 --phase '0-300:15:heavy-fast=0.3,medium-med=0.4,light-fast=0.3' --phase '300-600:15:light-fast=0.5,medium-med=0.3,heavy-slow=0.2' -o scenarios/bench-prod.yaml
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
# Scenario: production — compound profiles & load phases
# ---------------------------------------------------------------------------

# Token tiers: (prompt_tokens, max_tokens)
TOKEN_TIERS = {
    "heavy":  (600, 512),
    "medium": (300, 256),
    "light":  (150, 64),
}

# Rate tiers: req/s (defaults, overridable via CLI)
RATE_TIERS = {
    "fast": 10,
    "med":  5,
    "slow": 1,
}

# All 9 valid compound profile names
VALID_PROFILES = {
    f"{t}-{r}" for t in TOKEN_TIERS for r in RATE_TIERS
}

# Short abbreviations for program names
_TOKEN_SHORT = {"heavy": "th", "medium": "tm", "light": "tl"}
_RATE_SHORT  = {"fast": "rf", "med": "rm", "slow": "rs"}


def _parse_mix(mix_str):
    """Parse 'profile=frac,profile=frac,...' into dict.

    Validates profile names and that fractions sum to ~1.0.
    """
    parts = [p.strip() for p in mix_str.split(",") if p.strip()]
    mix = {}
    for part in parts:
        if "=" not in part:
            print(f"Error: invalid mix entry '{part}', expected 'profile=fraction'", file=sys.stderr)
            sys.exit(1)
        name, frac_str = part.split("=", 1)
        name = name.strip()
        if name not in VALID_PROFILES:
            print(f"Error: unknown profile '{name}'. Valid profiles: {sorted(VALID_PROFILES)}", file=sys.stderr)
            sys.exit(1)
        mix[name] = float(frac_str)
    total = sum(mix.values())
    if abs(total - 1.0) > 0.01:
        print(f"Error: mix fractions sum to {total}, expected ~1.0", file=sys.stderr)
        sys.exit(1)
    return mix


def _parse_phase(phase_str, duration):
    """Parse 'START-END:COUNT:profile=frac,...' into a phase dict.

    Returns {"start": int, "end": int, "count": int, "mix": {...}}.
    """
    parts = phase_str.split(":", 2)
    if len(parts) != 3:
        print(f"Error: invalid phase spec '{phase_str}', expected 'START-END:COUNT:mix'", file=sys.stderr)
        sys.exit(1)
    time_part, count_str, mix_str = parts
    if "-" not in time_part:
        print(f"Error: invalid time range '{time_part}', expected 'START-END'", file=sys.stderr)
        sys.exit(1)
    start_str, end_str = time_part.split("-", 1)
    start, end = int(start_str), int(end_str)
    if end <= start:
        print(f"Error: phase end ({end}) must be > start ({start})", file=sys.stderr)
        sys.exit(1)
    if end > duration:
        print(f"Error: phase end ({end}) exceeds --duration ({duration})", file=sys.stderr)
        sys.exit(1)
    count = int(count_str)
    mix = _parse_mix(mix_str)
    return {"start": start, "end": end, "count": count, "mix": mix}


def _mix_from_heavy_fraction(heavy_fraction):
    """Derive a default profile mix from the legacy --heavy-fraction flag."""
    remainder = 1.0 - heavy_fraction
    return {
        "heavy-med": heavy_fraction,
        "medium-med": round(remainder * 0.5, 4),
        "light-med": round(remainder - remainder * 0.5, 4),
    }


def generate_production(args):
    n = args.program_count
    duration = args.duration
    bg_fraction = args.bg_fraction
    heavy_fraction = args.heavy_fraction
    seed = args.seed

    rng = random.Random(seed)

    # Build rate lookup from defaults + CLI overrides
    rates = {
        "fast": args.rate_fast,
        "med":  args.rate_med,
        "slow": args.rate_slow,
    }

    bg_count = max(1, int(n * bg_fraction))
    fg_count = n - bg_count

    # --- Parse background mix ---
    if args.bg_mix:
        bg_mix = _parse_mix(args.bg_mix)
    else:
        bg_mix = _mix_from_heavy_fraction(heavy_fraction)

    # --- Parse foreground phases ---
    if args.phase:
        phases = [_parse_phase(p, duration) for p in args.phase]
        total_phase_fg = sum(p["count"] for p in phases)
        if total_phase_fg != fg_count:
            print(
                f"Error: sum of phase counts ({total_phase_fg}) != "
                f"fg program count ({fg_count} = {n} - {bg_count})",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # Single default phase spanning full duration
        fg_mix = _mix_from_heavy_fraction(heavy_fraction)
        phases = [{"start": 0, "end": duration, "count": fg_count, "mix": fg_mix}]

    # --- Compute blended avg token profile for capacity estimation ---
    all_mixes = [bg_mix] + [p["mix"] for p in phases]
    all_counts = [bg_count] + [p["count"] for p in phases]
    weighted_prompt, weighted_max, total_weight = 0.0, 0.0, 0.0
    for mix, count in zip(all_mixes, all_counts):
        for profile_name, frac in mix.items():
            token_tier = profile_name.split("-")[0]
            pt, mt = TOKEN_TIERS[token_tier]
            w = frac * count
            weighted_prompt += pt * w
            weighted_max += mt * w
            total_weight += w
    if total_weight > 0:
        avg_prompt = int(weighted_prompt / total_weight)
        avg_max = int(weighted_max / total_weight)
    else:
        avg_prompt, avg_max = 300, 256

    if args.max_num_seqs:
        mns = args.max_num_seqs
    else:
        peak_estimate = n * 1.2
        mns = auto_max_num_seqs(peak_estimate, args.load_level, avg_prompt, avg_max)

    capacity = estimate_capacity(mns, avg_prompt, avg_max)

    # --- Helper to distribute count across mix fractions ---
    def _distribute(count, mix):
        """Return list of (profile_name, per_profile_count) tuples."""
        items = list(mix.items())
        counts = []
        assigned = 0
        for i, (name, frac) in enumerate(items):
            if i == len(items) - 1:
                c = count - assigned  # remainder goes to last
            else:
                c = round(count * frac)
            counts.append((name, c))
            assigned += c
        return counts

    warmup = args.warmup if args.warmup else auto_warmup(n, 1.0)
    concurrency = auto_concurrency(n)

    programs = OrderedDict()
    bg_counter = 0

    # --- Background programs ---
    for profile_name, pcount in _distribute(bg_count, bg_mix):
        token_tier, rate_tier = profile_name.split("-")
        pt, mt = TOKEN_TIERS[token_tier]
        rate = rates[rate_tier]
        for _ in range(pcount):
            name = f"bg-{_TOKEN_SHORT[token_tier]}{_RATE_SHORT[rate_tier]}-{bg_counter:03d}"
            programs[name] = OrderedDict([
                ("background", True),
                ("count", 1),
                ("rate", _round2(rate)),
                ("prompt_tokens", pt),
                ("max_tokens", mt),
            ])
            bg_counter += 1

    # --- Foreground programs per phase ---
    fg_counter = 0
    phase_details = []
    for phase_idx, phase in enumerate(phases, start=1):
        p_start, p_end = phase["start"], phase["end"]
        p_length = p_end - p_start
        phase_fg_names = []

        for profile_name, pcount in _distribute(phase["count"], phase["mix"]):
            token_tier, rate_tier = profile_name.split("-")
            pt, mt = TOKEN_TIERS[token_tier]
            rate = rates[rate_tier]

            for _ in range(pcount):
                # Duration: random within [min(60, p_length), min(300, p_length)]
                min_dur = min(60, p_length)
                max_dur = min(300, p_length)
                prog_duration = rng.randint(min_dur, max_dur)

                # Start time: random within [p_start, p_end - prog_duration]
                latest_start = max(p_start, p_end - prog_duration)
                start_time = rng.randint(p_start, latest_start)

                # Clamp duration to not exceed phase window
                prog_duration = min(prog_duration, p_end - start_time)
                prog_duration = max(30, prog_duration)

                name = f"fg-p{phase_idx}-{_TOKEN_SHORT[token_tier]}{_RATE_SHORT[rate_tier]}-{fg_counter:03d}"
                programs[name] = OrderedDict([
                    ("start_time", start_time),
                    ("duration", prog_duration),
                    ("count", 1),
                    ("rate", _round2(rate)),
                    ("prompt_tokens", pt),
                    ("max_tokens", mt),
                ])
                phase_fg_names.append(name)
                fg_counter += 1

        phase_details.append((phase_idx, p_start, p_end, phase["count"], phase["mix"], phase_fg_names))

    # --- Summary stats ---
    total_bg_rate = sum(
        p["rate"] * p.get("count", 1)
        for p in programs.values()
        if p.get("background", False)
    )
    total_fg_rate = sum(
        p["rate"] * p.get("count", 1)
        for p in programs.values()
        if not p.get("background", False)
    )
    peak_total = total_bg_rate + total_fg_rate

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

    # --- Comment header ---
    bg_breakdown = ", ".join(
        f"{pcount} {pname}" for pname, pcount in _distribute(bg_count, bg_mix) if pcount > 0
    )
    comment_lines = [
        f"# Generated scenario: production",
        f"#",
        f"# {bg_count} background ({bg_breakdown}) + {fg_count} foreground.",
        f"# BG total: {_round2(total_bg_rate)} req/s | Capacity: {_round2(capacity)} req/s (max-num-seqs={mns})",
    ]
    for pi, ps, pe, pc, pmix, _ in phase_details:
        mix_str = ", ".join(f"{k}={v}" for k, v in pmix.items())
        comment_lines.append(f"#   Phase {pi} (t={ps}-{pe}s, {pc} fg): {mix_str}")
    comment_lines += [
        f"# Seed: {seed} | Rates: fast={rates['fast']} med={rates['med']} slow={rates['slow']}",
        f"#",
    ]
    comment = "\n".join(comment_lines) + "\n"

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
                        help="Fraction used for default mix when --phase/--bg-mix omitted (default: 0.3)")
    prod_p.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    prod_p.add_argument("--phase", action="append", default=None,
                        help="Foreground phase spec: 'START-END:COUNT:profile=frac,...' (repeatable)")
    prod_p.add_argument("--bg-mix", type=str, default=None,
                        help="Background profile distribution: 'profile=frac,profile=frac,...'")
    prod_p.add_argument("--rate-fast", type=float, default=RATE_TIERS["fast"],
                        help=f"req/s for 'fast' rate tier (default: {RATE_TIERS['fast']})")
    prod_p.add_argument("--rate-med", type=float, default=RATE_TIERS["med"],
                        help=f"req/s for 'med' rate tier (default: {RATE_TIERS['med']})")
    prod_p.add_argument("--rate-slow", type=float, default=RATE_TIERS["slow"],
                        help=f"req/s for 'slow' rate tier (default: {RATE_TIERS['slow']})")

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
