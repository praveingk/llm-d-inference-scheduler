#!/usr/bin/env python3
"""
Load generator for test/load-test.

Sends requests to the gateway for each program defined in the scenario YAML,
records per-request results to JSONL, and prints live stats every 10s.

Usage:
    python3 loadgen.py --scenario scenarios/simple-ab.yaml \
                       --phase program-aware \
                       --gateway http://localhost:30080 \
                       --output results/simple-ab/program-aware/
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp
import yaml

FAIRNESS_HEADER = "x-gateway-inference-fairness-id"
STATS_INTERVAL = 10.0  # seconds between live stats prints


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProgramConfig:
    name: str
    total_requests: int     # exactly how many requests to send
    concurrency: int        # max simultaneous in-flight
    prompt_tokens: int
    max_tokens: int
    start_time: float       # seconds after measurement start
    request_timeout: float = 60.0  # per-request HTTP timeout
    no_fairness_header: bool = False
    initial_request_interval: float = 0.0  # seconds between launches for first `concurrency` requests


@dataclass
class ProgramStats:
    name: str
    sent: int = 0
    ok: int = 0
    err: int = 0
    in_flight: int = 0
    latencies: List[float] = field(default_factory=list)

    def p50(self) -> Optional[float]:
        if not self.latencies:
            return None
        return statistics.median(self.latencies)

    def p99(self) -> Optional[float]:
        if len(self.latencies) < 2:
            return None
        s = sorted(self.latencies)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(prompt_tokens: int) -> str:
    """Repeat a word to approximate the desired token count (~1.3 chars/token)."""
    word = "hello "
    target_chars = int(prompt_tokens * 1.3)
    return (word * (target_chars // len(word) + 1))[:target_chars]


# ---------------------------------------------------------------------------
# Single-program sender
# ---------------------------------------------------------------------------

async def run_program(
    program: ProgramConfig,
    model: str,
    gateway: str,
    measurement_start: float,
    stats: ProgramStats,
    results: List[dict],
    record_results: bool,
    session: aiohttp.ClientSession,
):
    """Send exactly total_requests for one program, gated by concurrency."""
    sem = asyncio.Semaphore(program.concurrency)
    prompt = build_prompt(program.prompt_tokens)
    url = f"{gateway}/v1/completions"
    headers = {"Content-Type": "application/json"}
    if not program.no_fairness_header:
        headers[FAIRNESS_HEADER] = program.name

    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": program.max_tokens,
        "temperature": 0,
        "ignore_eos": True,
    }

    # Wait until this program's start_time (relative to measurement_start).
    wall_start = measurement_start + program.start_time
    wait = wall_start - time.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)

    async def send_one():
        sent_at = time.time()
        stats.in_flight += 1
        error_response = None
        try:
            req_timeout = aiohttp.ClientTimeout(total=program.request_timeout)
            async with session.post(url, json=body, headers=headers, timeout=req_timeout) as resp:
                resp_text = await resp.text()
                completed_at = time.time()
                latency_ms = (completed_at - sent_at) * 1000
                try:
                    resp_data = json.loads(resp_text)
                except (json.JSONDecodeError, ValueError):
                    resp_data = None
                if resp.status == 200 and resp_data is not None:
                    stats.ok += 1
                    stats.latencies.append(latency_ms)
                    status = "ok"
                    actual_output_tokens = (resp_data.get("usage") or {}).get("completion_tokens", program.max_tokens)
                else:
                    stats.err += 1
                    status = f"http_{resp.status}"
                    if resp_data is None:
                        status += ":json_decode_error"
                    actual_output_tokens = 0
                    error_response = resp_text
        except Exception as e:
            completed_at = time.time()
            latency_ms = (completed_at - sent_at) * 1000
            stats.err += 1
            status = f"error:{type(e).__name__}"
            actual_output_tokens = 0
            error_response = str(e)
        finally:
            stats.in_flight -= 1
            sem.release()

        if record_results:
            rec = {
                "program_id": program.name,
                "sent_at": sent_at,
                "completed_at": completed_at,
                "latency_ms": round(latency_ms, 2),
                "status": status,
                "prompt_tokens": program.prompt_tokens,
                "output_tokens": actual_output_tokens,
            }
            if error_response is not None:
                rec["error_response"] = error_response
            results.append(rec)

    pending: List[asyncio.Task] = []
    stagger = program.initial_request_interval
    rng = random.Random(program.name)  # deterministic per-program jitter
    for i in range(program.total_requests):
        await sem.acquire()
        if stagger > 0 and 0 < i < program.concurrency:
            jitter = stagger * rng.uniform(0.9, 1.1)
            await asyncio.sleep(jitter)
        stats.sent += 1
        pending.append(asyncio.create_task(send_one()))

    # Drain all in-flight requests.
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


# ---------------------------------------------------------------------------
# Live stats printer
# ---------------------------------------------------------------------------

def format_table(all_stats: List[ProgramStats], header: bool = False) -> str:
    col_w = [22, 7, 7, 7, 10, 10, 10]
    cols  = ["program", "sent", "ok", "err", "in_flight", "p50_ms", "p99_ms"]
    sep   = "  "
    lines = []
    if header:
        lines.append(sep.join(c.ljust(w) for c, w in zip(cols, col_w)))
        lines.append(sep.join("-" * w for w in col_w))
    for s in all_stats:
        p50 = f"{s.p50():.0f}" if s.p50() is not None else "-"
        p99 = f"{s.p99():.0f}" if s.p99() is not None else "-"
        row = [s.name[:22], str(s.sent), str(s.ok), str(s.err),
               str(s.in_flight), p50, p99]
        lines.append(sep.join(v.ljust(w) for v, w in zip(row, col_w)))
    return "\n".join(lines)


async def stats_printer(all_stats: List[ProgramStats], stop: asyncio.Event, start_time: float):
    first = True
    while not stop.is_set():
        await asyncio.sleep(STATS_INTERVAL)
        elapsed = int(time.monotonic() - start_time)
        print(f"\n[T+{elapsed}s]")
        print(format_table(all_stats, header=first))
        first = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(scenario_path: str, phase_name: str, gateway: str, output_dir: str):
    with open(scenario_path) as f:
        cfg = yaml.safe_load(f)

    model    = cfg["model"]
    test_cfg = cfg.get("test", {})

    programs: List[ProgramConfig] = []
    for name, pc in cfg.get("programs", {}).items():
        programs.append(ProgramConfig(
            name=name,
            total_requests=int(pc.get("total_requests", 100)),
            concurrency=int(pc.get("concurrency", 4)),
            prompt_tokens=int(pc.get("prompt_tokens", 512)),
            max_tokens=int(pc.get("max_tokens", 128)),
            start_time=float(pc.get("start_time", 0)),
            request_timeout=float(pc.get("request_timeout", 60)),
            no_fairness_header=bool(pc.get("no_fairness_header", False)),
            initial_request_interval=float(pc.get("initial_request_interval", 0)),
        ))

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.jsonl")

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Run warmup as a regular program (total_requests model, no results recorded).
        warmup_cfg = test_cfg.get("warmup", {})
        warmup_total = int(warmup_cfg.get("total_requests", 0))
        if warmup_total > 0:
            warmup_prog = ProgramConfig(
                name="warmup",
                total_requests=warmup_total,
                concurrency=int(warmup_cfg.get("concurrency", 4)),
                prompt_tokens=int(warmup_cfg.get("prompt_tokens", 128)),
                max_tokens=int(warmup_cfg.get("max_tokens", 64)),
                start_time=0,
                no_fairness_header=True,
            )
            warmup_stats = ProgramStats(name="warmup")
            print(f"[warmup] Sending {warmup_total} requests (concurrency={warmup_prog.concurrency}) ...")
            await run_program(
                program=warmup_prog, model=model, gateway=gateway,
                measurement_start=time.monotonic(),
                stats=warmup_stats, results=[], record_results=False,
                session=session,
            )
            print(f"[warmup] Done. sent={warmup_stats.sent} ok={warmup_stats.ok} err={warmup_stats.err}\n")

        total_reqs = sum(p.total_requests for p in programs)
        print(f"[{phase_name}] Starting: {len(programs)} programs, {total_reqs} total requests")
        print(f"[{phase_name}] Output:   {output_file}\n")

        all_stats   = [ProgramStats(name=p.name) for p in programs]
        all_results: List[dict] = []
        t0          = time.monotonic()
        stop        = asyncio.Event()

        printer = asyncio.create_task(stats_printer(all_stats, stop, t0))
        senders = [
            asyncio.create_task(run_program(
                program=p, model=model, gateway=gateway,
                measurement_start=t0,
                stats=all_stats[i], results=all_results,
                record_results=True, session=session,
            ))
            for i, p in enumerate(programs)
        ]

        await asyncio.gather(*senders)
        stop.set()
        await printer

    # Write JSONL sorted by sent_at.
    all_results.sort(key=lambda r: r["sent_at"])
    with open(output_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Final summary.
    print(f"\n{'='*65}")
    print(f"  Final summary — phase: {phase_name}")
    print(f"{'='*65}")
    print(format_table(all_stats, header=True))
    print(f"{'='*65}")
    total_sent = sum(s.sent for s in all_stats)
    total_ok   = sum(s.ok   for s in all_stats)
    total_err  = sum(s.err  for s in all_stats)
    print(f"  Total  sent={total_sent}  ok={total_ok}  err={total_err}")
    print(f"  Results: {output_file}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load generator for test/load-test")
    parser.add_argument("--scenario", required=True, help="Path to scenario YAML")
    parser.add_argument("--phase",    required=True, help="Phase name (for output labelling)")
    parser.add_argument("--gateway",  default="http://localhost:30080", help="Gateway URL")
    parser.add_argument("--output",   required=True, help="Output directory for results.jsonl")
    args = parser.parse_args()

    asyncio.run(main(args.scenario, args.phase, args.gateway, args.output))
