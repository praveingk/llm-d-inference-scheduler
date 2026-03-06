#!/usr/bin/env python3
"""
Analyze fairness A/B test results.

Auto-discovers phase subdirectories under --results-dir, each containing
results.jsonl. The first phase (alphabetically) is treated as baseline.

Produces:
- Per-program latency table (P50, P95, P99)
- CDF plot: per-program latency, per phase
- Bar chart: P50/P95/P99 grouped by program and phase
- Fairness ratio analysis
- Throughput check

Usage:
    python3 analyze_results.py --results-dir results/
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Plots will be skipped. Install with: pip install matplotlib")


def load_results(path: str) -> list[dict]:
    """Load JSONL results file."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def group_by_program(results: list[dict]) -> dict[str, list[float]]:
    """Group latencies by program_id, filtering to successful requests only."""
    groups = defaultdict(list)
    for r in results:
        if r["status"] == "ok":
            groups[r["program_id"]].append(r["total_ms"])
    return dict(groups)


def percentile(data: list[float], p: float) -> float:
    """Compute percentile."""
    if not data:
        return 0.0
    return float(np.percentile(sorted(data), p))


def compute_stats(latencies: list[float]) -> dict:
    """Compute latency statistics."""
    if not latencies:
        return {"count": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
    s = sorted(latencies)
    return {
        "count": len(s),
        "mean": float(np.mean(s)),
        "p50": percentile(s, 50),
        "p95": percentile(s, 95),
        "p99": percentile(s, 99),
        "min": s[0],
        "max": s[-1],
    }


import re


def program_base_name(program_id: str) -> str:
    """Strip the instance suffix (-0, -1, ...) to get the program group name."""
    return re.sub(r'-\d+$', '', program_id)


def group_by_program_grouped(results: list[dict]) -> dict[str, list[float]]:
    """Group latencies by program base name (instances merged), filtering to successful requests."""
    groups = defaultdict(list)
    for r in results:
        if r["status"] == "ok":
            groups[program_base_name(r["program_id"])].append(r["total_ms"])
    return dict(groups)


def discover_phases(results_dir: str) -> list[tuple[str, str]]:
    """Discover phase subdirectories containing results.jsonl.

    Returns list of (phase_name, jsonl_path) sorted alphabetically.
    """
    phases = []
    for entry in sorted(os.listdir(results_dir)):
        jsonl = os.path.join(results_dir, entry, "results.jsonl")
        if os.path.isdir(os.path.join(results_dir, entry)) and os.path.isfile(jsonl):
            phases.append((entry, jsonl))
    return phases


def print_comparison_table(phase_groups: dict[str, dict[str, list[float]]]):
    """Print side-by-side latency comparison table."""
    all_programs = sorted(set(
        prog for groups in phase_groups.values() for prog in groups
    ))
    phase_names = list(phase_groups.keys())

    print("\n" + "=" * 90)
    print("LATENCY COMPARISON (milliseconds)")
    print("=" * 90)
    header = f"{'Program':<20} {'Phase':<18} {'Count':>7} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}"
    print(header)
    print("-" * 90)

    for prog in all_programs:
        for phase in phase_names:
            latencies = phase_groups[phase].get(prog, [])
            stats = compute_stats(latencies)
            print(
                f"{prog:<20} {phase:<18} {stats['count']:>7} "
                f"{stats['mean']:>8.1f} {stats['p50']:>8.1f} "
                f"{stats['p95']:>8.1f} {stats['p99']:>8.1f}"
            )
        print()


def print_fairness_analysis(phase_groups: dict[str, dict[str, list[float]]]):
    """Print fairness ratio analysis."""
    all_programs = sorted(set(
        prog for groups in phase_groups.values() for prog in groups
    ))
    phase_names = list(phase_groups.keys())

    print("=" * 70)
    print("FAIRNESS ANALYSIS")
    print("=" * 70)

    for phase in phase_names:
        groups = phase_groups[phase]
        stats_by_prog = {}
        for prog in all_programs:
            latencies = groups.get(prog, [])
            stats_by_prog[prog] = compute_stats(latencies)

        print(f"\n{phase}:")

        # Find the program with highest request count (likely "heavy").
        heavy_prog = max(stats_by_prog, key=lambda p: stats_by_prog[p]["count"]) if stats_by_prog else None
        if heavy_prog and stats_by_prog[heavy_prog]["p95"] > 0:
            for prog in all_programs:
                if prog == heavy_prog:
                    continue
                ratio_p95 = stats_by_prog[prog]["p95"] / stats_by_prog[heavy_prog]["p95"]
                ratio_p99 = stats_by_prog[prog]["p99"] / stats_by_prog[heavy_prog]["p99"] if stats_by_prog[heavy_prog]["p99"] > 0 else float("inf")
                print(f"  P95 ratio {prog}/{heavy_prog}: {ratio_p95:.2f}x")
                print(f"  P99 ratio {prog}/{heavy_prog}: {ratio_p99:.2f}x")

    # Cross-phase improvement (first phase = baseline).
    if len(phase_names) >= 2:
        baseline = phase_names[0]
        print(f"\nCross-phase improvement vs {baseline}:")
        for phase in phase_names[1:]:
            print(f"\n  {phase} vs {baseline}:")
            for prog in all_programs:
                bl = compute_stats(phase_groups[baseline].get(prog, []))
                ph = compute_stats(phase_groups[phase].get(prog, []))
                if bl["p95"] > 0 and ph["p95"] > 0:
                    improvement = (bl["p95"] - ph["p95"]) / bl["p95"] * 100
                    print(f"    {prog} P95: {bl['p95']:.1f}ms -> {ph['p95']:.1f}ms ({improvement:+.1f}%)")
                if bl["p99"] > 0 and ph["p99"] > 0:
                    improvement = (bl["p99"] - ph["p99"]) / bl["p99"] * 100
                    print(f"    {prog} P99: {bl['p99']:.1f}ms -> {ph['p99']:.1f}ms ({improvement:+.1f}%)")


def print_throughput_check(phase_results: dict[str, list[dict]]):
    """Print throughput comparison."""
    print("\n" + "=" * 70)
    print("THROUGHPUT CHECK")
    print("=" * 70)

    for phase, results in phase_results.items():
        ok = sum(1 for r in results if r["status"] == "ok")
        err = sum(1 for r in results if r["status"] != "ok")
        total = len(results)
        if total > 0:
            timestamps = [r["sent_at"] for r in results]
            duration = max(timestamps) - min(timestamps)
            rps = ok / duration if duration > 0 else 0
            print(f"  {phase}: {ok} ok / {err} errors / {total} total ({rps:.1f} req/s effective)")


def plot_cdf(phase_groups: dict[str, dict[str, list[float]]], output_dir: str):
    """Plot per-program latency CDF, one subplot per phase."""
    if not HAS_MATPLOTLIB:
        return

    phase_names = list(phase_groups.keys())
    all_programs = sorted(set(
        prog for groups in phase_groups.values() for prog in groups
    ))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_programs), 1)))

    fig, axes = plt.subplots(1, len(phase_names), figsize=(7 * len(phase_names), 6), sharey=True, squeeze=False)

    for ax_idx, phase in enumerate(phase_names):
        ax = axes[0][ax_idx]
        for i, prog in enumerate(all_programs):
            latencies = sorted(phase_groups[phase].get(prog, []))
            if latencies:
                cdf = np.arange(1, len(latencies) + 1) / len(latencies)
                ax.plot(latencies, cdf, label=prog, color=colors[i], linewidth=1.5)

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_title(phase)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "cdf_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved CDF plot: {path}")


def plot_cdf_overlay(phase_groups: dict[str, dict[str, list[float]]], output_dir: str):
    """Plot overlaid CDF — all programs, all phases on one chart."""
    if not HAS_MATPLOTLIB:
        return

    phase_names = list(phase_groups.keys())
    all_programs = sorted(set(
        prog for groups in phase_groups.values() for prog in groups
    ))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_programs), 1)))
    linestyles = ["-", "--", ":", "-."]

    fig, ax = plt.subplots(figsize=(10, 6))

    for phase_idx, phase in enumerate(phase_names):
        ls = linestyles[phase_idx % len(linestyles)]
        for i, prog in enumerate(all_programs):
            latencies = sorted(phase_groups[phase].get(prog, []))
            if latencies:
                cdf = np.arange(1, len(latencies) + 1) / len(latencies)
                ax.plot(latencies, cdf, label=f"{prog} ({phase})", color=colors[i], linestyle=ls, linewidth=1.5)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF — All Phases")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "cdf_overlay.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved overlay CDF plot: {path}")


def plot_bar_chart(phase_groups: dict[str, dict[str, list[float]]], output_dir: str):
    """Plot bar chart of P50/P95/P99 grouped by program and phase."""
    if not HAS_MATPLOTLIB:
        return

    phase_names = list(phase_groups.keys())
    all_programs = sorted(set(
        prog for groups in phase_groups.values() for prog in groups
    ))
    metrics_names = ["P50", "P95", "P99"]

    phase_vals = {phase: {m: [] for m in metrics_names} for phase in phase_names}
    for prog in all_programs:
        for phase in phase_names:
            stats = compute_stats(phase_groups[phase].get(prog, []))
            for m, key in [("P50", "p50"), ("P95", "p95"), ("P99", "p99")]:
                phase_vals[phase][m].append(stats[key])

    bar_colors = plt.cm.tab10(np.linspace(0, 1, max(len(phase_names), 1)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(all_programs))
    width = 0.8 / len(phase_names)

    for ax, metric in zip(axes, metrics_names):
        for p_idx, phase in enumerate(phase_names):
            offset = (p_idx - len(phase_names) / 2 + 0.5) * width
            ax.bar(x + offset, phase_vals[phase][metric], width, label=phase, color=bar_colors[p_idx], alpha=0.8)
        ax.set_xlabel("Program")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{metric} Latency")
        ax.set_xticks(x)
        ax.set_xticklabels(all_programs, rotation=25, ha="right", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "bar_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved bar chart: {path}")


def plot_timeseries(phase_results: dict[str, list[dict]], output_dir: str):
    """Plot latency over time for each program, one subplot per phase."""
    if not HAS_MATPLOTLIB:
        return

    phase_names = list(phase_results.keys())
    fig, axes = plt.subplots(len(phase_names), 1, figsize=(14, 4 * len(phase_names)), sharex=True, squeeze=False)

    for ax_idx, phase in enumerate(phase_names):
        ax = axes[ax_idx][0]
        results = phase_results[phase]
        by_prog = defaultdict(lambda: ([], []))
        t0 = min(r["sent_at"] for r in results) if results else 0
        for r in results:
            if r["status"] == "ok":
                by_prog[r["program_id"]][0].append(r["sent_at"] - t0)
                by_prog[r["program_id"]][1].append(r["total_ms"])

        for prog in sorted(by_prog.keys()):
            times, latencies = by_prog[prog]
            ax.scatter(times, latencies, label=prog, alpha=0.3, s=8)

        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{phase} — Latency Over Time")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Time (s)")
    plt.tight_layout()
    path = os.path.join(output_dir, "timeseries.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved timeseries plot: {path}")


def plot_cdf_grouped(phase_grouped: dict[str, dict[str, list[float]]], output_dir: str):
    """Plot CDF with instances merged into program groups, one subplot per phase."""
    if not HAS_MATPLOTLIB:
        return

    phase_names = list(phase_grouped.keys())
    all_groups = sorted(set(
        grp for groups in phase_grouped.values() for grp in groups
    ))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_groups), 1)))

    fig, axes = plt.subplots(1, len(phase_names), figsize=(7 * len(phase_names), 6), sharey=True, squeeze=False)

    for ax_idx, phase in enumerate(phase_names):
        ax = axes[0][ax_idx]
        for i, grp in enumerate(all_groups):
            latencies = sorted(phase_grouped[phase].get(grp, []))
            if latencies:
                cdf = np.arange(1, len(latencies) + 1) / len(latencies)
                ax.plot(latencies, cdf, label=grp, color=colors[i], linewidth=1.5)

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_title(f"{phase} (grouped)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "cdf_grouped.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved grouped CDF plot: {path}")


def plot_bar_mean_individual(phase_groups: dict[str, dict[str, list[float]]], output_dir: str):
    """Plot bar chart of mean latency per individual instance, per phase."""
    if not HAS_MATPLOTLIB:
        return

    phase_names = list(phase_groups.keys())
    all_programs = sorted(set(
        prog for groups in phase_groups.values() for prog in groups
    ))

    phase_means = {phase: [] for phase in phase_names}
    for prog in all_programs:
        for phase in phase_names:
            stats = compute_stats(phase_groups[phase].get(prog, []))
            phase_means[phase].append(stats["mean"])

    bar_colors = plt.cm.tab10(np.linspace(0, 1, max(len(phase_names), 1)))
    fig, ax = plt.subplots(figsize=(max(10, len(all_programs) * 1.2), 6))
    x = np.arange(len(all_programs))
    width = 0.8 / len(phase_names)

    for p_idx, phase in enumerate(phase_names):
        offset = (p_idx - len(phase_names) / 2 + 0.5) * width
        ax.bar(x + offset, phase_means[phase], width, label=phase, color=bar_colors[p_idx], alpha=0.8)

    ax.set_xlabel("Program Instance")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Mean Latency — Individual Instances")
    ax.set_xticks(x)
    ax.set_xticklabels(all_programs, rotation=25, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "bar_mean_individual.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved individual mean bar chart: {path}")


def plot_bar_mean_grouped(phase_grouped: dict[str, dict[str, list[float]]], output_dir: str):
    """Plot bar chart of mean latency per program group (instances merged), per phase."""
    if not HAS_MATPLOTLIB:
        return

    phase_names = list(phase_grouped.keys())
    all_groups = sorted(set(
        grp for groups in phase_grouped.values() for grp in groups
    ))

    phase_means = {phase: [] for phase in phase_names}
    for grp in all_groups:
        for phase in phase_names:
            stats = compute_stats(phase_grouped[phase].get(grp, []))
            phase_means[phase].append(stats["mean"])

    bar_colors = plt.cm.tab10(np.linspace(0, 1, max(len(phase_names), 1)))
    fig, ax = plt.subplots(figsize=(max(8, len(all_groups) * 2), 6))
    x = np.arange(len(all_groups))
    width = 0.8 / len(phase_names)

    for p_idx, phase in enumerate(phase_names):
        offset = (p_idx - len(phase_names) / 2 + 0.5) * width
        ax.bar(x + offset, phase_means[phase], width, label=phase, color=bar_colors[p_idx], alpha=0.8)

    ax.set_xlabel("Program Group")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Mean Latency — Grouped Programs")
    ax.set_xticks(x)
    ax.set_xticklabels(all_groups, rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "bar_mean_grouped.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved grouped mean bar chart: {path}")


def plot_bar_chart_grouped(phase_grouped: dict[str, dict[str, list[float]]], output_dir: str):
    """Plot bar chart of P50/P95/P99 for program groups (instances merged), per phase."""
    if not HAS_MATPLOTLIB:
        return

    phase_names = list(phase_grouped.keys())
    all_groups = sorted(set(
        grp for groups in phase_grouped.values() for grp in groups
    ))
    metrics_names = ["P50", "P95", "P99"]

    phase_vals = {phase: {m: [] for m in metrics_names} for phase in phase_names}
    for grp in all_groups:
        for phase in phase_names:
            stats = compute_stats(phase_grouped[phase].get(grp, []))
            for m, key in [("P50", "p50"), ("P95", "p95"), ("P99", "p99")]:
                phase_vals[phase][m].append(stats[key])

    bar_colors = plt.cm.tab10(np.linspace(0, 1, max(len(phase_names), 1)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(all_groups))
    width = 0.8 / len(phase_names)

    for ax, metric in zip(axes, metrics_names):
        for p_idx, phase in enumerate(phase_names):
            offset = (p_idx - len(phase_names) / 2 + 0.5) * width
            ax.bar(x + offset, phase_vals[phase][metric], width, label=phase, color=bar_colors[p_idx], alpha=0.8)
        ax.set_xlabel("Program Group")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{metric} Latency (Grouped)")
        ax.set_xticks(x)
        ax.set_xticklabels(all_groups, rotation=15, ha="right", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "bar_comparison_grouped.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved grouped P50/P95/P99 bar chart: {path}")


def count_dropped(results: list[dict], program_id: str) -> int:
    """Count non-ok requests for a given program_id."""
    return sum(1 for r in results if r["program_id"] == program_id and r["status"] != "ok")


def count_dropped_grouped(results: list[dict], group_name: str) -> int:
    """Count non-ok requests for all instances of a program group."""
    return sum(1 for r in results if program_base_name(r["program_id"]) == group_name and r["status"] != "ok")


def plot_bar_dropped_individual(phase_results: dict[str, list[dict]], phase_groups: dict[str, dict[str, list[float]]], output_dir: str):
    """Plot bar chart of dropped (non-ok) request counts per individual instance, per phase."""
    if not HAS_MATPLOTLIB:
        return

    phase_names = list(phase_results.keys())
    all_programs = sorted(set(
        prog for groups in phase_groups.values() for prog in groups
    ))

    phase_dropped = {phase: [] for phase in phase_names}
    for prog in all_programs:
        for phase in phase_names:
            phase_dropped[phase].append(count_dropped(phase_results[phase], prog))

    bar_colors = plt.cm.tab10(np.linspace(0, 1, max(len(phase_names), 1)))
    fig, ax = plt.subplots(figsize=(max(10, len(all_programs) * 1.2), 6))
    x = np.arange(len(all_programs))
    width = 0.8 / len(phase_names)

    for p_idx, phase in enumerate(phase_names):
        offset = (p_idx - len(phase_names) / 2 + 0.5) * width
        ax.bar(x + offset, phase_dropped[phase], width, label=phase, color=bar_colors[p_idx], alpha=0.8)

    ax.set_xlabel("Program Instance")
    ax.set_ylabel("Dropped Requests")
    ax.set_title("Dropped Requests — Individual Instances")
    ax.set_xticks(x)
    ax.set_xticklabels(all_programs, rotation=25, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "bar_dropped_individual.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved individual dropped bar chart: {path}")


def plot_bar_dropped_grouped(phase_results: dict[str, list[dict]], phase_grouped: dict[str, dict[str, list[float]]], output_dir: str):
    """Plot bar chart of dropped (non-ok) request counts per program group, per phase."""
    if not HAS_MATPLOTLIB:
        return

    phase_names = list(phase_results.keys())
    all_groups = sorted(set(
        grp for groups in phase_grouped.values() for grp in groups
    ))

    phase_dropped = {phase: [] for phase in phase_names}
    for grp in all_groups:
        for phase in phase_names:
            phase_dropped[phase].append(count_dropped_grouped(phase_results[phase], grp))

    bar_colors = plt.cm.tab10(np.linspace(0, 1, max(len(phase_names), 1)))
    fig, ax = plt.subplots(figsize=(max(8, len(all_groups) * 2), 6))
    x = np.arange(len(all_groups))
    width = 0.8 / len(phase_names)

    for p_idx, phase in enumerate(phase_names):
        offset = (p_idx - len(phase_names) / 2 + 0.5) * width
        ax.bar(x + offset, phase_dropped[phase], width, label=phase, color=bar_colors[p_idx], alpha=0.8)

    ax.set_xlabel("Program Group")
    ax.set_ylabel("Dropped Requests")
    ax.set_title("Dropped Requests — Grouped Programs")
    ax.set_xticks(x)
    ax.set_xticklabels(all_groups, rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "bar_dropped_grouped.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved grouped dropped bar chart: {path}")


def save_summary_json(phase_groups: dict[str, dict[str, list[float]]], output_dir: str):
    """Save machine-readable summary as JSON."""
    all_programs = sorted(set(
        prog for groups in phase_groups.values() for prog in groups
    ))
    summary = {}
    for phase, groups in phase_groups.items():
        summary[phase] = {}
        for prog in all_programs:
            summary[phase][prog] = compute_stats(groups.get(prog, []))

    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary JSON: {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze fairness A/B test results")
    parser.add_argument("--results-dir", required=True, help="Results directory containing phase subdirectories")
    args = parser.parse_args()

    phases = discover_phases(args.results_dir)
    if not phases:
        print(f"No phase subdirectories with results.jsonl found in {args.results_dir}")
        sys.exit(1)

    print(f"Discovered {len(phases)} phases: {', '.join(name for name, _ in phases)}")

    # Load all results.
    phase_results: dict[str, list[dict]] = {}
    phase_groups: dict[str, dict[str, list[float]]] = {}
    for name, path in phases:
        results = load_results(path)
        phase_results[name] = results
        phase_groups[name] = group_by_program(results)
        print(f"  {name}: {len(results)} records")

    print_comparison_table(phase_groups)
    print_fairness_analysis(phase_groups)
    print_throughput_check(phase_results)

    comparison_dir = os.path.join(args.results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Build grouped views (instances merged by program base name).
    phase_grouped: dict[str, dict[str, list[float]]] = {}
    for name, path in phases:
        phase_grouped[name] = group_by_program_grouped(phase_results[name])

    print("\nGenerating plots...")
    plot_cdf(phase_groups, comparison_dir)
    plot_cdf_overlay(phase_groups, comparison_dir)
    plot_cdf_grouped(phase_grouped, comparison_dir)
    plot_bar_chart(phase_groups, comparison_dir)
    plot_bar_chart_grouped(phase_grouped, comparison_dir)
    plot_bar_mean_individual(phase_groups, comparison_dir)
    plot_bar_mean_grouped(phase_grouped, comparison_dir)
    plot_bar_dropped_individual(phase_results, phase_groups, comparison_dir)
    plot_bar_dropped_grouped(phase_results, phase_grouped, comparison_dir)
    plot_timeseries(phase_results, comparison_dir)
    save_summary_json(phase_groups, comparison_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()