#!/usr/bin/env python3
"""
Real-time Prometheus metrics scraper for fairness testing.

Scrapes Jain's fairness index and other metrics every second during a test phase,
writing timestamped data to a JSONL file for later analysis.

Usage:
    python3 scrape_metrics_realtime.py \
        --metrics-url http://localhost:9090/metrics \
        --output results/phase-name/metrics_timeseries.jsonl \
        --duration 600 \
        --subsystem program_aware
"""

import argparse
import json
import re
import sys
import time
from typing import Dict, Optional


def parse_prometheus_text(text: str) -> Dict[str, float]:
    """Parse Prometheus text exposition format into a flat dict."""
    metrics = {}
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.rsplit(' ', 1)
        if len(parts) == 2:
            try:
                metrics[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return metrics


def extract_fairness_index(metrics: Dict[str, float], subsystem: str) -> Optional[float]:
    """Extract Jain's fairness index for the given subsystem."""
    key = f"{subsystem}_jains_fairness_index"
    return metrics.get(key)


def extract_program_counter(metrics: Dict[str, float], metric_name: str) -> Dict[str, float]:
    """Extract per-program values for a counter/gauge metric."""
    result = {}
    prefix = metric_name + "{"
    for key, val in metrics.items():
        if key.startswith(prefix):
            match = re.search(r'program_id="([^"]+)"', key)
            if match:
                result[match.group(1)] = val
    return result


def scrape_once(metrics_url: str, subsystem: str) -> Dict:
    """Scrape metrics once and return structured data."""
    import urllib.request
    
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as response:
            text = response.read().decode('utf-8')
        
        metrics = parse_prometheus_text(text)
        
        # Extract key metrics
        fairness_index = extract_fairness_index(metrics, subsystem)
        requests_total = extract_program_counter(metrics, f"{subsystem}_requests_total")
        dispatched_total = extract_program_counter(metrics, f"{subsystem}_dispatched_total")
        
        # Extract histogram data for wait times
        wait_time_sums = extract_program_counter(metrics, f"{subsystem}_wait_time_milliseconds_sum")
        wait_time_counts = extract_program_counter(metrics, f"{subsystem}_wait_time_milliseconds_count")
        
        # Compute average wait times
        avg_wait_times = {}
        for prog in wait_time_sums:
            if prog in wait_time_counts and wait_time_counts[prog] > 0:
                avg_wait_times[prog] = wait_time_sums[prog] / wait_time_counts[prog]
        
        return {
            "timestamp": time.time(),
            "fairness_index": fairness_index,
            "requests_total": requests_total,
            "dispatched_total": dispatched_total,
            "avg_wait_time_ms": avg_wait_times,
        }
    except Exception as e:
        return {
            "timestamp": time.time(),
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Real-time Prometheus metrics scraper")
    parser.add_argument("--metrics-url", required=True, help="Prometheus metrics endpoint URL")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--duration", type=int, required=True, help="Scraping duration in seconds")
    parser.add_argument("--subsystem", default="program_aware", help="Metrics subsystem (program_aware, round_robin)")
    parser.add_argument("--interval", type=float, default=1.0, help="Scraping interval in seconds")
    args = parser.parse_args()
    
    print(f"Starting real-time metrics scraper:")
    print(f"  URL:       {args.metrics_url}")
    print(f"  Duration:  {args.duration}s")
    print(f"  Interval:  {args.interval}s")
    print(f"  Subsystem: {args.subsystem}")
    print(f"  Output:    {args.output}")
    print()
    
    start_time = time.time()
    end_time = start_time + args.duration
    
    with open(args.output, 'w') as f:
        scrape_count = 0
        next_scrape = start_time
        
        while time.time() < end_time:
            now = time.time()
            
            if now >= next_scrape:
                data = scrape_once(args.metrics_url, args.subsystem)
                f.write(json.dumps(data) + '\n')
                f.flush()
                
                scrape_count += 1
                elapsed = now - start_time
                
                # Print progress every 10 scrapes
                if scrape_count % 10 == 0:
                    fairness = data.get("fairness_index")
                    if fairness is not None:
                        print(f"[{elapsed:6.1f}s] Scraped {scrape_count:4d} samples | Fairness: {fairness:.4f}")
                    else:
                        print(f"[{elapsed:6.1f}s] Scraped {scrape_count:4d} samples | Fairness: N/A")
                
                next_scrape += args.interval
            
            # Sleep until next scrape time
            sleep_time = next_scrape - time.time()
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.1))
    
    total_elapsed = time.time() - start_time
    print()
    print(f"Scraping complete. Collected {scrape_count} samples in {total_elapsed:.1f}s")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

# Made with Bob
