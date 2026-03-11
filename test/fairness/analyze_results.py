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

    txt_path = os.path.join(output_dir, "cdf_comparison.txt")
    with open(txt_path, "w") as f:
        for phase in phase_names:
            f.write(f"# Phase: {phase}\n")
            f.write("program\tlatency_ms\tcdf\n")
            for prog in all_programs:
                latencies = sorted(phase_groups[phase].get(prog, []))
                if latencies:
                    cdf = np.arange(1, len(latencies) + 1) / len(latencies)
                    for lat, c in zip(latencies, cdf):
                        f.write(f"{prog}\t{lat:.2f}\t{c:.4f}\n")
            f.write("\n")


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

    txt_path = os.path.join(output_dir, "cdf_overlay.txt")
    with open(txt_path, "w") as f:
        f.write("phase\tprogram\tlatency_ms\tcdf\n")
        for phase in phase_names:
            for prog in all_programs:
                latencies = sorted(phase_groups[phase].get(prog, []))
                if latencies:
                    cdf = np.arange(1, len(latencies) + 1) / len(latencies)
                    for lat, c in zip(latencies, cdf):
                        f.write(f"{phase}\t{prog}\t{lat:.2f}\t{c:.4f}\n")


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

    txt_path = os.path.join(output_dir, "bar_comparison.txt")
    with open(txt_path, "w") as f:
        f.write("program\tphase\tP50\tP95\tP99\n")
        for prog in all_programs:
            for phase in phase_names:
                stats = compute_stats(phase_groups[phase].get(prog, []))
                f.write(f"{prog}\t{phase}\t{stats['p50']:.2f}\t{stats['p95']:.2f}\t{stats['p99']:.2f}\n")


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

    txt_path = os.path.join(output_dir, "timeseries.txt")
    with open(txt_path, "w") as f:
        f.write("phase\tprogram\ttime_s\tlatency_ms\n")
        for phase in phase_names:
            results = phase_results[phase]
            t0 = min(r["sent_at"] for r in results) if results else 0
            for r in results:
                if r["status"] == "ok":
                    f.write(f"{phase}\t{r['program_id']}\t{r['sent_at'] - t0:.3f}\t{r['total_ms']:.2f}\n")


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

    txt_path = os.path.join(output_dir, "cdf_grouped.txt")
    with open(txt_path, "w") as f:
        for phase in phase_names:
            f.write(f"# Phase: {phase}\n")
            f.write("group\tlatency_ms\tcdf\n")
            for grp in all_groups:
                latencies = sorted(phase_grouped[phase].get(grp, []))
                if latencies:
                    cdf = np.arange(1, len(latencies) + 1) / len(latencies)
                    for lat, c in zip(latencies, cdf):
                        f.write(f"{grp}\t{lat:.2f}\t{c:.4f}\n")
            f.write("\n")


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

    txt_path = os.path.join(output_dir, "bar_mean_individual.txt")
    with open(txt_path, "w") as f:
        f.write("program\tphase\tmean_ms\n")
        for prog in all_programs:
            for phase in phase_names:
                stats = compute_stats(phase_groups[phase].get(prog, []))
                f.write(f"{prog}\t{phase}\t{stats['mean']:.2f}\n")


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

    txt_path = os.path.join(output_dir, "bar_mean_grouped.txt")
    with open(txt_path, "w") as f:
        f.write("group\tphase\tmean_ms\n")
        for grp in all_groups:
            for phase in phase_names:
                stats = compute_stats(phase_grouped[phase].get(grp, []))
                f.write(f"{grp}\t{phase}\t{stats['mean']:.2f}\n")


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

    txt_path = os.path.join(output_dir, "bar_comparison_grouped.txt")
    with open(txt_path, "w") as f:
        f.write("group\tphase\tP50\tP95\tP99\n")
        for grp in all_groups:
            for phase in phase_names:
                stats = compute_stats(phase_grouped[phase].get(grp, []))
                f.write(f"{grp}\t{phase}\t{stats['p50']:.2f}\t{stats['p95']:.2f}\t{stats['p99']:.2f}\n")


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

    txt_path = os.path.join(output_dir, "bar_dropped_individual.txt")
    with open(txt_path, "w") as f:
        f.write("program\tphase\tdropped\n")
        for prog in all_programs:
            for phase in phase_names:
                f.write(f"{prog}\t{phase}\t{count_dropped(phase_results[phase], prog)}\n")


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

    txt_path = os.path.join(output_dir, "bar_dropped_grouped.txt")
    with open(txt_path, "w") as f:
        f.write("group\tphase\tdropped\n")
        for grp in all_groups:
            for phase in phase_names:
                f.write(f"{grp}\t{phase}\t{count_dropped_grouped(phase_results[phase], grp)}\n")


def parse_prometheus_snapshot(path: str) -> dict[str, float]:
    """Parse a Prometheus text exposition file into a flat dict of metric_name{labels} -> value."""
    metrics = {}
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return metrics
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Lines look like: metric_name{label="val",...} 123.0
            # or: metric_name 123.0
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                try:
                    metrics[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    return metrics


def extract_program_counter(metrics: dict[str, float], metric_name: str) -> dict[str, float]:
    """Extract per-program values for a counter/gauge metric.

    Returns {program_id: value}.
    """
    result = {}
    prefix = metric_name + "{"
    for key, val in metrics.items():
        if key.startswith(prefix):
            # Extract program_id from labels like {program_id="prog-heavy"}
            match = re.search(r'program_id="([^"]+)"', key)
            if match:
                result[match.group(1)] = val
    return result


def extract_histogram_avg(metrics: dict[str, float], metric_name: str) -> dict[str, float]:
    """Extract per-program average from a histogram (_sum / _count).

    Returns {program_id: average}.
    """
    sums = extract_program_counter(metrics, metric_name + "_sum")
    counts = extract_program_counter(metrics, metric_name + "_count")
    result = {}
    for prog in sums:
        if prog in counts and counts[prog] > 0:
            result[prog] = sums[prog] / counts[prog]
    return result


def extract_global_histogram(metrics: dict[str, float], metric_name: str) -> tuple[float, float]:
    """Extract global (non-labeled) histogram sum and count.

    Returns (sum, count).
    """
    sum_key = metric_name + "_sum"
    count_key = metric_name + "_count"
    return metrics.get(sum_key, 0.0), metrics.get(count_key, 0.0)


def load_phase_subsystems(results_dir: str, phases: list[tuple[str, str]]) -> dict[str, str]:
    """Load metrics subsystem per phase from metrics_subsystem.txt.

    Returns {phase_name: subsystem}. Defaults to 'program_aware' if file is missing.
    """
    subsystems = {}
    for name, jsonl_path in phases:
        sub_path = os.path.join(os.path.dirname(jsonl_path), "metrics_subsystem.txt")
        if os.path.isfile(sub_path):
            with open(sub_path) as f:
                subsystems[name] = f.read().strip() or "program_aware"
        else:
            subsystems[name] = "program_aware"
    return subsystems


def load_phase_prometheus(results_dir: str, phases: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
    """Load Prometheus snapshots for all phases.

    Returns {phase_name: flat_metrics_dict}. Phases with empty/missing snapshots get empty dicts.
    """
    phase_metrics = {}
    for name, jsonl_path in phases:
        metrics_path = os.path.join(os.path.dirname(jsonl_path), "metrics_final.txt")
        phase_metrics[name] = parse_prometheus_snapshot(metrics_path)
    return phase_metrics


def plot_prometheus_requests_dispatched(phase_metrics: dict[str, dict[str, float]], output_dir: str, phase_subsystems: dict[str, str] | None = None):
    """Plot requests_total vs dispatched_total per program, per phase."""
    if not HAS_MATPLOTLIB:
        return

    if phase_subsystems is None:
        phase_subsystems = {}

    phase_names = list(phase_metrics.keys())
    all_programs = sorted(set(
        prog
        for phase, m in phase_metrics.items()
        for prog in extract_program_counter(m, f"{phase_subsystems.get(phase, 'program_aware')}_requests_total")
    ))
    if not all_programs:
        return

    fig, axes = plt.subplots(1, len(phase_names), figsize=(7 * len(phase_names), 6), squeeze=False)

    for ax_idx, phase in enumerate(phase_names):
        ax = axes[0][ax_idx]
        m = phase_metrics[phase]
        sub = phase_subsystems.get(phase, "program_aware")
        requests = extract_program_counter(m, f"{sub}_requests_total")
        dispatched = extract_program_counter(m, f"{sub}_dispatched_total")

        x = np.arange(len(all_programs))
        width = 0.35
        req_vals = [requests.get(p, 0) for p in all_programs]
        disp_vals = [dispatched.get(p, 0) for p in all_programs]

        ax.bar(x - width / 2, req_vals, width, label="requests", color="steelblue", alpha=0.8)
        ax.bar(x + width / 2, disp_vals, width, label="dispatched", color="seagreen", alpha=0.8)
        ax.set_xlabel("Program")
        ax.set_ylabel("Count")
        ax.set_title(f"{phase}")
        ax.set_xticks(x)
        ax.set_xticklabels(all_programs, rotation=25, ha="right", fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Prometheus: Requests vs Dispatched", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "prometheus_requests_dispatched.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved Prometheus requests/dispatched plot: {path}")

    txt_path = os.path.join(output_dir, "prometheus_requests_dispatched.txt")
    with open(txt_path, "w") as f:
        f.write("phase\tprogram\trequests\tdispatched\n")
        for phase in phase_names:
            m = phase_metrics[phase]
            sub = phase_subsystems.get(phase, "program_aware")
            requests = extract_program_counter(m, f"{sub}_requests_total")
            dispatched = extract_program_counter(m, f"{sub}_dispatched_total")
            for p in all_programs:
                f.write(f"{phase}\t{p}\t{requests.get(p, 0):.0f}\t{dispatched.get(p, 0):.0f}\n")


def plot_prometheus_tokens(phase_metrics: dict[str, dict[str, float]], output_dir: str, phase_subsystems: dict[str, str] | None = None):
    """Plot input_tokens_total vs output_tokens_total per program, per phase."""
    if not HAS_MATPLOTLIB:
        return

    if phase_subsystems is None:
        phase_subsystems = {}

    phase_names = list(phase_metrics.keys())
    all_programs = sorted(set(
        prog
        for phase, m in phase_metrics.items()
        for prog in extract_program_counter(m, f"{phase_subsystems.get(phase, 'program_aware')}_input_tokens_total")
    ))
    if not all_programs:
        return

    fig, axes = plt.subplots(1, len(phase_names), figsize=(7 * len(phase_names), 6), squeeze=False)

    for ax_idx, phase in enumerate(phase_names):
        ax = axes[0][ax_idx]
        m = phase_metrics[phase]
        sub = phase_subsystems.get(phase, "program_aware")
        input_tokens = extract_program_counter(m, f"{sub}_input_tokens_total")
        output_tokens = extract_program_counter(m, f"{sub}_output_tokens_total")

        x = np.arange(len(all_programs))
        width = 0.35
        in_vals = [input_tokens.get(p, 0) for p in all_programs]
        out_vals = [output_tokens.get(p, 0) for p in all_programs]

        ax.bar(x - width / 2, in_vals, width, label="input tokens", color="coral", alpha=0.8)
        ax.bar(x + width / 2, out_vals, width, label="output tokens", color="mediumpurple", alpha=0.8)
        ax.set_xlabel("Program")
        ax.set_ylabel("Tokens")
        ax.set_title(f"{phase}")
        ax.set_xticks(x)
        ax.set_xticklabels(all_programs, rotation=25, ha="right", fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Prometheus: Token Counts", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "prometheus_tokens.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved Prometheus tokens plot: {path}")

    txt_path = os.path.join(output_dir, "prometheus_tokens.txt")
    with open(txt_path, "w") as f:
        f.write("phase\tprogram\tinput_tokens\toutput_tokens\n")
        for phase in phase_names:
            m = phase_metrics[phase]
            sub = phase_subsystems.get(phase, "program_aware")
            input_tokens = extract_program_counter(m, f"{sub}_input_tokens_total")
            output_tokens = extract_program_counter(m, f"{sub}_output_tokens_total")
            for p in all_programs:
                f.write(f"{phase}\t{p}\t{input_tokens.get(p, 0):.0f}\t{output_tokens.get(p, 0):.0f}\n")


def plot_prometheus_wait_time(phase_metrics: dict[str, dict[str, float]], output_dir: str, phase_subsystems: dict[str, str] | None = None):
    """Plot average wait time per program, per phase."""
    if not HAS_MATPLOTLIB:
        return

    if phase_subsystems is None:
        phase_subsystems = {}

    phase_names = list(phase_metrics.keys())
    all_programs = sorted(set(
        prog
        for phase, m in phase_metrics.items()
        for prog in extract_histogram_avg(m, f"{phase_subsystems.get(phase, 'program_aware')}_wait_time_milliseconds")
    ))
    if not all_programs:
        return

    bar_colors = plt.cm.tab10(np.linspace(0, 1, max(len(phase_names), 1)))
    fig, ax = plt.subplots(figsize=(max(8, len(all_programs) * 2), 6))
    x = np.arange(len(all_programs))
    width = 0.8 / len(phase_names)

    for p_idx, phase in enumerate(phase_names):
        sub = phase_subsystems.get(phase, "program_aware")
        avgs = extract_histogram_avg(phase_metrics[phase], f"{sub}_wait_time_milliseconds")
        vals = [avgs.get(p, 0) for p in all_programs]
        offset = (p_idx - len(phase_names) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=phase, color=bar_colors[p_idx], alpha=0.8)

    ax.set_xlabel("Program")
    ax.set_ylabel("Avg Wait Time (ms)")
    ax.set_title("Prometheus: Average Queue Wait Time")
    ax.set_xticks(x)
    ax.set_xticklabels(all_programs, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "prometheus_wait_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved Prometheus wait time plot: {path}")

    txt_path = os.path.join(output_dir, "prometheus_wait_time.txt")
    with open(txt_path, "w") as f:
        f.write("phase\tprogram\tavg_wait_ms\n")
        for phase in phase_names:
            sub = phase_subsystems.get(phase, "program_aware")
            avgs = extract_histogram_avg(phase_metrics[phase], f"{sub}_wait_time_milliseconds")
            for p in all_programs:
                f.write(f"{phase}\t{p}\t{avgs.get(p, 0):.2f}\n")


def plot_prometheus_pick_latency_sum(phase_metrics: dict[str, dict[str, float]], output_dir: str, phase_subsystems: dict[str, str] | None = None):
    """Plot total (sum) pick latency per phase."""
    if not HAS_MATPLOTLIB:
        return

    if phase_subsystems is None:
        phase_subsystems = {}

    phase_names = list(phase_metrics.keys())
    sums = []
    for phase in phase_names:
        sub = phase_subsystems.get(phase, "program_aware")
        s, _ = extract_global_histogram(phase_metrics[phase], f"{sub}_pick_latency_microseconds")
        sums.append(s)

    if all(v == 0 for v in sums):
        return

    fig, ax = plt.subplots(figsize=(max(6, len(phase_names) * 2), 5))
    bar_colors = plt.cm.tab10(np.linspace(0, 1, max(len(phase_names), 1)))
    x = np.arange(len(phase_names))
    ax.bar(x, sums, color=bar_colors[:len(phase_names)], alpha=0.8)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Total Pick Latency (µs)")
    ax.set_title("Prometheus: Total Pick Latency (Sum)")
    ax.set_xticks(x)
    ax.set_xticklabels(phase_names, rotation=15, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "prometheus_pick_latency_sum.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved Prometheus pick latency sum plot: {path}")

    txt_path = os.path.join(output_dir, "prometheus_pick_latency_sum.txt")
    with open(txt_path, "w") as f:
        f.write("phase\ttotal_pick_latency_us\n")
        for phase, s in zip(phase_names, sums):
            f.write(f"{phase}\t{s:.2f}\n")


def plot_prometheus_pick_latency_mean(phase_metrics: dict[str, dict[str, float]], output_dir: str, phase_subsystems: dict[str, str] | None = None):
    """Plot mean pick latency per phase."""
    if not HAS_MATPLOTLIB:
        return

    if phase_subsystems is None:
        phase_subsystems = {}

    phase_names = list(phase_metrics.keys())
    means = []
    for phase in phase_names:
        sub = phase_subsystems.get(phase, "program_aware")
        s, c = extract_global_histogram(phase_metrics[phase], f"{sub}_pick_latency_microseconds")
        means.append(s / c if c > 0 else 0)

    if all(v == 0 for v in means):
        return

    fig, ax = plt.subplots(figsize=(max(6, len(phase_names) * 2), 5))
    bar_colors = plt.cm.tab10(np.linspace(0, 1, max(len(phase_names), 1)))
    x = np.arange(len(phase_names))
    ax.bar(x, means, color=bar_colors[:len(phase_names)], alpha=0.8)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Mean Pick Latency (µs)")
    ax.set_title("Prometheus: Mean Pick Latency")
    ax.set_xticks(x)
    ax.set_xticklabels(phase_names, rotation=15, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "prometheus_pick_latency_mean.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved Prometheus pick latency mean plot: {path}")

    txt_path = os.path.join(output_dir, "prometheus_pick_latency_mean.txt")
    with open(txt_path, "w") as f:
        f.write("phase\tmean_pick_latency_us\n")
        for phase, m in zip(phase_names, means):
            f.write(f"{phase}\t{m:.2f}\n")


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

    # Prometheus metrics graphs (best-effort — skip if snapshots are empty).
    phase_prom = load_phase_prometheus(args.results_dir, phases)
    phase_subs = load_phase_subsystems(args.results_dir, phases)
    if any(m for m in phase_prom.values()):
        print("\nGenerating Prometheus plots...")
        print(f"  Subsystems: {phase_subs}")
        plot_prometheus_requests_dispatched(phase_prom, comparison_dir, phase_subs)
        plot_prometheus_tokens(phase_prom, comparison_dir, phase_subs)
        plot_prometheus_wait_time(phase_prom, comparison_dir, phase_subs)
        plot_prometheus_pick_latency_sum(phase_prom, comparison_dir, phase_subs)
        plot_prometheus_pick_latency_mean(phase_prom, comparison_dir, phase_subs)

    print("\nDone!")


if __name__ == "__main__":
    main()