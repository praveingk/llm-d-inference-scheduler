#!/usr/bin/env python3
"""
Prometheus metrics scraper for test/load-test.

Polls the EPP metrics endpoint every second during a phase run and writes
timestamped JSONL records for later analysis.

Metrics captured (per subsystem prefix):
  {subsystem}_jains_fairness_index
  {subsystem}_ewma_wait_time_milliseconds       per program_id
  {subsystem}_throughput_tokens_per_second       per program_id
  {subsystem}_service_rate_tokens_per_second     per program_id
  {subsystem}_attained_service_tokens            per program_id  (program-aware only)
  {subsystem}_queue_score                        per program_id  (program-aware only)
  {subsystem}_requests_total                     per program_id
  {subsystem}_dispatched_total                   per program_id
  inference_extension_flow_control_queue_size     per fairness_id

Usage:
    python3 scrape_metrics.py \
        --url http://localhost:9090/metrics \
        --subsystem program_aware \
        --duration 150 \
        --output results/simple-ab/program-aware/metrics.jsonl
"""

import argparse
import json
import re
import time
import urllib.request
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Prometheus text format parser
# ---------------------------------------------------------------------------

def parse_prometheus(text: str) -> Dict[str, float]:
    """Parse Prometheus text exposition into a flat {metric_line: value} dict."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            try:
                result[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return result


def extract_scalar(metrics: Dict[str, float], name: str) -> Optional[float]:
    """Return a scalar metric value (no labels)."""
    return metrics.get(name)


def extract_by_label(metrics: Dict[str, float], metric_name: str, label: str) -> Dict[str, float]:
    """Return {label_value: metric_value} for all series matching metric_name{label=...}."""
    result = {}
    prefix = metric_name + "{"
    pattern = re.compile(rf'{label}="([^"]+)"')
    for key, val in metrics.items():
        if key.startswith(prefix):
            m = pattern.search(key)
            if m:
                result[m.group(1)] = val
    return result


# ---------------------------------------------------------------------------
# Single scrape
# ---------------------------------------------------------------------------

def scrape_once(url: str, subsystem: str) -> dict:
    ts = time.time()
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        return {"ts": ts, "error": str(e)}

    metrics = parse_prometheus(text)

    fairness_index = extract_scalar(metrics, f"{subsystem}_jains_fairness_index")
    ewma_wait      = extract_by_label(metrics, f"{subsystem}_ewma_wait_time_milliseconds", "program_id")
    requests       = extract_by_label(metrics, f"{subsystem}_requests_total",  "program_id")
    dispatched     = extract_by_label(metrics, f"{subsystem}_dispatched_total", "program_id")
    queue_size     = extract_by_label(metrics, "inference_extension_flow_control_queue_size", "fairness_id")
    throughput     = extract_by_label(metrics, f"{subsystem}_throughput_tokens_per_second", "program_id")
    queue_score    = extract_by_label(metrics, f"{subsystem}_queue_score", "program_id")
    service_rate   = extract_by_label(metrics, f"{subsystem}_service_rate_tokens_per_second", "program_id")
    attained_svc   = extract_by_label(metrics, f"{subsystem}_attained_service_tokens", "program_id")

    # Build per-program dict keyed by all seen program IDs.
    all_ids = set(ewma_wait) | set(requests) | set(dispatched) | set(queue_score) | set(throughput) | set(service_rate) | set(attained_svc)
    per_program = {
        pid: {
            "ewma_wait_ms":      ewma_wait.get(pid),
            "throughput_tps":    throughput.get(pid),
            "service_rate_tps":  service_rate.get(pid),
            "attained_service":  attained_svc.get(pid),
            "requests":          requests.get(pid),
            "dispatched":        dispatched.get(pid),
            "queue_size":        queue_size.get(pid),  # matches when fairness_id == program name
            "queue_score":       queue_score.get(pid),
        }
        for pid in sorted(all_ids)
    }

    return {
        "ts":             ts,
        "fairness_index": fairness_index,
        "per_program":    per_program,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prometheus metrics scraper for test/load-test")
    parser.add_argument("--url",       required=True,                    help="Metrics endpoint URL")
    parser.add_argument("--subsystem", default="program_aware",          help="Metric prefix (program_aware / round_robin)")
    parser.add_argument("--duration",  type=int, required=True,          help="How long to scrape (seconds)")
    parser.add_argument("--output",    required=True,                    help="Output JSONL file path")
    parser.add_argument("--interval",  type=float, default=1.0,          help="Scrape interval (seconds)")
    args = parser.parse_args()

    print(f"[scraper] url={args.url}  subsystem={args.subsystem}  duration={args.duration}s  output={args.output}")

    t0       = time.time()
    end_time = t0 + args.duration
    count    = 0
    next_t   = t0

    with open(args.output, "w") as f:
        while time.time() < end_time:
            now = time.time()
            if now >= next_t:
                record = scrape_once(args.url, args.subsystem)
                f.write(json.dumps(record) + "\n")
                f.flush()
                count += 1
                next_t += args.interval

                # Print summary every 10 scrapes.
                if count % 10 == 0:
                    elapsed = now - t0
                    fi = record.get("fairness_index")
                    fi_str = f"{fi:.4f}" if fi is not None else "N/A"
                    waits = " | ".join(
                        f"{pid}: wait={v['ewma_wait_ms']:.0f}ms" if v.get("ewma_wait_ms") is not None else f"{pid}: wait=N/A"
                        for pid, v in record.get("per_program", {}).items()
                    )
                    print(f"[T+{elapsed:5.0f}s] fairness_index={fi_str}  {waits}")

            sleep_for = next_t - time.time()
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.1))

    print(f"[scraper] Done. {count} samples written to {args.output}")


if __name__ == "__main__":
    main()