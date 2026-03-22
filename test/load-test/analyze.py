#!/usr/bin/env python3
"""
Post-run analysis for test/load-test.

Reads per-phase results.jsonl and metrics.jsonl, produces 5 comparison plots:
  1. latency.png            — P50/P95/P99 end-to-end latency bar chart per program x phase
  2. fairness_index.png     — Jain's fairness index over time, all phases overlaid
  3. wait_time_phases.png   — Per-program EWMA wait time over time, one subplot per phase
  4. wait_time_overlay.png  — All phases+programs on one chart for direct comparison
  5. error_cumulative.png   — Cumulative errors per program over time, one subplot per phase

Usage:
    python3 analyze.py results/simple-ab/
"""

import argparse
import json
import os
import statistics
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(phase_dir: str) -> List[dict]:
    path = os.path.join(phase_dir, "results.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        records.append(r)
                except json.JSONDecodeError:
                    continue
    return records


def load_all_results(phase_dir: str) -> List[dict]:
    """Load all results (including errors) from results.jsonl."""
    path = os.path.join(phase_dir, "results.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_metrics(phase_dir: str) -> List[dict]:
    path = os.path.join(phase_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if "error" not in r:
                        records.append(r)
                except json.JSONDecodeError:
                    continue
    return records


def discover_phases(results_dir: str) -> List[str]:
    """Return phase names (subdirs containing results.jsonl or metrics.jsonl), sorted."""
    phases = []
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if entry.is_dir():
            has_results = os.path.exists(os.path.join(entry.path, "results.jsonl"))
            has_metrics = os.path.exists(os.path.join(entry.path, "metrics.jsonl"))
            if has_results or has_metrics:
                phases.append(entry.name)
    return phases


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def group_latencies_by_program(records: List[dict]) -> Dict[str, List[float]]:
    groups: Dict[str, List[float]] = {}
    for r in records:
        pid = r.get("program_id", "unknown")
        groups.setdefault(pid, []).append(r["latency_ms"])
    return groups


# ---------------------------------------------------------------------------
# Profile-based color mapping
# ---------------------------------------------------------------------------

def _extract_profile(pid: str) -> str:
    """Extract profile name from program ID: fg-heavy-aggressive-003 -> heavy-aggressive."""
    if not pid.startswith("fg-"):
        return pid
    rest = pid[3:]
    parts = rest.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return rest


def profile_color_map(program_ids) -> Dict[str, tuple]:
    """Assign one color per unique profile. Programs sharing a profile get the same color."""
    profiles = sorted(set(_extract_profile(pid) for pid in program_ids))
    colors = plt.cm.tab10.colors
    return {p: colors[i % len(colors)] for i, p in enumerate(profiles)}


# ---------------------------------------------------------------------------
# Plot 1: P50/P99 latency bar chart
# ---------------------------------------------------------------------------

def plot_latency(phases: List[str], results_dir: str, out_path: str):
    # Collect p50/p95/p99 per (phase, program).
    phase_data = {}
    all_programs = []
    for phase in phases:
        records = load_results(os.path.join(results_dir, phase))
        groups  = group_latencies_by_program(records)
        phase_data[phase] = {
            pid: {
                "p50": percentile(lats, 50),
                "p95": percentile(lats, 95),
                "p99": percentile(lats, 99),
            }
            for pid, lats in groups.items()
        }
        for pid in groups:
            if pid not in all_programs:
                all_programs.append(pid)

    if not all_programs:
        print("[analyze] No latency data found, skipping latency.png")
        return

    all_programs = sorted(all_programs)
    n_programs   = len(all_programs)
    n_phases     = len(phases)

    # Three rows (p50, p95, p99) stacked vertically.
    fig, axes = plt.subplots(3, 1, figsize=(min(120, max(10, n_programs * 0.5 + 2)), 4 * 3), sharey=False)
    colors = plt.cm.tab10.colors

    for ax_idx, pct_label in enumerate(["p50", "p95", "p99"]):
        ax = axes[ax_idx]
        x  = range(n_programs)
        bar_w = 0.8 / max(n_phases, 1)

        for i, phase in enumerate(phases):
            vals = [
                phase_data[phase].get(pid, {}).get(pct_label) or 0
                for pid in all_programs
            ]
            offsets = [xi - 0.4 + (i + 0.5) * bar_w for xi in x]
            ax.bar(offsets, vals, width=bar_w * 0.9,
                   label=phase, color=colors[i % len(colors)])

        ax.set_title(f"{pct_label.upper()} Latency (ms)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(all_programs, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Latency (ms)")
        ax.legend(fontsize=7)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("End-to-End Request Latency by Program and Phase", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analyze] Wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Jain's fairness index over time (all phases overlaid)
# ---------------------------------------------------------------------------

def plot_fairness_index(phases: List[str], results_dir: str, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors  = plt.cm.tab10.colors
    any_data = False

    for i, phase in enumerate(phases):
        records = load_metrics(os.path.join(results_dir, phase))
        if not records:
            continue
        t0 = records[0]["ts"]
        xs = [r["ts"] - t0 for r in records]
        ys = [r.get("fairness_index") for r in records]
        # Filter out None values.
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, label=phase, color=colors[i % len(colors)], linewidth=1.5)
        any_data = True

    if not any_data:
        print("[analyze] No fairness index data found, skipping fairness_index.png")
        plt.close(fig)
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle("Jain's Fairness Index Over Time — All Phases", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analyze] Wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Per-program EWMA wait time — one subplot per phase
# ---------------------------------------------------------------------------

def plot_wait_time_phases(phases: List[str], results_dir: str, out_path: str):
    n = len(phases)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(12, 5 * n), squeeze=False)
    any_data = False

    # Collect all program IDs across phases for consistent coloring.
    all_pids: set = set()
    for phase in phases:
        records = load_metrics(os.path.join(results_dir, phase))
        for r in records:
            all_pids.update(r.get("per_program", {}).keys())
    cmap = profile_color_map(all_pids)

    for i, phase in enumerate(phases):
        ax = axes[i][0]
        records = load_metrics(os.path.join(results_dir, phase))
        if not records:
            ax.set_title(phase, fontsize=9)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue

        t0 = records[0]["ts"]
        program_series: Dict[str, list] = {}
        for r in records:
            t = r["ts"] - t0
            for pid, pdata in r.get("per_program", {}).items():
                w = pdata.get("ewma_wait_ms")
                if w is not None:
                    program_series.setdefault(pid, []).append((t, w))

        seen_profiles: set = set()
        for pid, series in sorted(program_series.items()):
            xs, ys = zip(*series)
            profile = _extract_profile(pid)
            label = profile if profile not in seen_profiles else "_nolegend_"
            seen_profiles.add(profile)
            ax.plot(xs, ys, label=label, color=cmap[profile], linewidth=1.2)
            any_data = True

        ax.set_title(phase, fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EWMA Wait Time (ms)")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1)
        ax.grid(alpha=0.3)

    if not any_data:
        print("[analyze] No EWMA wait data found, skipping wait_time_phases.png")
        plt.close(fig)
        return

    fig.suptitle("Per-Program EWMA Wait Time Over Time", fontsize=11)
    fig.subplots_adjust(right=0.75)
    fig.tight_layout(rect=[0, 0, 0.75, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: EWMA wait time overlay — all phases+programs on one chart
# ---------------------------------------------------------------------------

def plot_wait_time_overlay(phases: List[str], results_dir: str, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    any_data = False

    # Collect all program IDs across phases for consistent coloring.
    all_pids: set = set()
    for phase in phases:
        records = load_metrics(os.path.join(results_dir, phase))
        for r in records:
            all_pids.update(r.get("per_program", {}).keys())
    cmap = profile_color_map(all_pids)

    # Use different line styles per phase to distinguish them.
    line_styles = ["-", "--", "-.", ":"]

    for pi, phase in enumerate(phases):
        records = load_metrics(os.path.join(results_dir, phase))
        if not records:
            continue
        t0 = records[0]["ts"]
        program_series: Dict[str, list] = {}
        for r in records:
            t = r["ts"] - t0
            for pid, pdata in r.get("per_program", {}).items():
                w = pdata.get("ewma_wait_ms")
                if w is not None:
                    program_series.setdefault(pid, []).append((t, w))

        seen_profiles: set = set()
        for pid, series in sorted(program_series.items()):
            xs, ys = zip(*series)
            profile = _extract_profile(pid)
            key = f"{phase}:{profile}"
            label = key if key not in seen_profiles else "_nolegend_"
            seen_profiles.add(key)
            ax.plot(xs, ys, label=label,
                    color=cmap[profile],
                    linestyle=line_styles[pi % len(line_styles)],
                    linewidth=1.2)
            any_data = True

    if not any_data:
        print("[analyze] No EWMA wait data found, skipping wait_time_overlay.png")
        plt.close(fig)
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EWMA Wait Time (ms)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    ax.grid(alpha=0.3)
    fig.suptitle("Per-Program EWMA Wait Time — All Phases Overlaid", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: Cumulative errors per program — one subplot per phase
# ---------------------------------------------------------------------------

def plot_error_cumulative(phases: List[str], results_dir: str, out_path: str):
    n = len(phases)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(12, 5 * n), squeeze=False)
    any_data = False

    # Collect all program IDs across phases for consistent coloring.
    all_pids: set = set()
    for phase in phases:
        records = load_all_results(os.path.join(results_dir, phase))
        for r in records:
            pid = r.get("program_id", "unknown")
            all_pids.add(pid)
    cmap = profile_color_map(all_pids)

    for i, phase in enumerate(phases):
        ax = axes[i][0]
        records = load_all_results(os.path.join(results_dir, phase))
        if not records:
            ax.set_title(phase, fontsize=9)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Find phase start time (earliest completed_at).
        t0 = min(r["completed_at"] for r in records if "completed_at" in r)

        # Group all records by program, sorted by time.
        by_program: Dict[str, List[dict]] = {}
        for r in records:
            pid = r.get("program_id", "unknown")
            by_program.setdefault(pid, []).append(r)

        seen_profiles: set = set()
        for pid, prog_records in sorted(by_program.items()):
            prog_records.sort(key=lambda r: r.get("completed_at", 0))
            cum_errors = 0
            xs, ys = [], []
            for r in prog_records:
                if r.get("status") != "ok":
                    cum_errors += 1
                t = r.get("completed_at", 0) - t0
                xs.append(t)
                ys.append(cum_errors)
            if cum_errors > 0:
                profile = _extract_profile(pid)
                label = profile if profile not in seen_profiles else "_nolegend_"
                seen_profiles.add(profile)
                ax.plot(xs, ys, label=label, color=cmap[profile], linewidth=1.2)
                any_data = True

        ax.set_title(phase, fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative Errors")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1)
        ax.grid(alpha=0.3)

    if not any_data:
        print("[analyze] No error data found, skipping error_cumulative.png")
        plt.close(fig)
        return

    fig.suptitle("Cumulative Errors Per Program Over Time", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.75, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze load-test results and produce plots")
    parser.add_argument("results_dir", help="Path to scenario results directory (e.g. results/simple-ab/)")
    args = parser.parse_args()

    results_dir = args.results_dir.rstrip("/")
    phases = discover_phases(results_dir)
    if not phases:
        print(f"[analyze] No phase directories found in {results_dir}")
        return

    print(f"[analyze] Found phases: {phases}")

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_latency(
        phases, results_dir,
        os.path.join(plots_dir, "latency.png"),
    )
    plot_fairness_index(
        phases, results_dir,
        os.path.join(plots_dir, "fairness_index.png"),
    )
    plot_wait_time_phases(
        phases, results_dir,
        os.path.join(plots_dir, "wait_time_phases.png"),
    )
    plot_wait_time_overlay(
        phases, results_dir,
        os.path.join(plots_dir, "wait_time_overlay.png"),
    )
    plot_error_cumulative(
        phases, results_dir,
        os.path.join(plots_dir, "error_cumulative.png"),
    )

    print(f"[analyze] All plots written to {plots_dir}/")


if __name__ == "__main__":
    main()