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
    rate: float            # requests per second
    concurrency: int       # max simultaneous in-flight
    prompt_tokens: int
    max_tokens: int
    start_time: float      # seconds after measurement start
    duration: float        # how long to send (seconds)
    no_fairness_header: bool = False


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
    timeout: float,
):
    """Send requests for one program over its active window."""
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

    interval = 1.0 / program.rate

    # Wait until this program's start_time (relative to measurement_start).
    wall_start = measurement_start + program.start_time
    wait = wall_start - time.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)

    wall_end = wall_start + program.duration
    next_send = time.monotonic()
    pending: List[asyncio.Task] = []

    async def send_one():
        sent_at = time.time()
        stats.in_flight += 1
        try:
            req_timeout = aiohttp.ClientTimeout(total=timeout)
            async with session.post(url, json=body, headers=headers, timeout=req_timeout) as resp:
                resp_data = await resp.json(content_type=None)
                completed_at = time.time()
                latency_ms = (completed_at - sent_at) * 1000
                if resp.status == 200:
                    stats.ok += 1
                    stats.latencies.append(latency_ms)
                    status = "ok"
                    actual_output_tokens = (resp_data.get("usage") or {}).get("completion_tokens", program.max_tokens)
                else:
                    stats.err += 1
                    status = f"http_{resp.status}"
                    actual_output_tokens = 0
        except Exception as e:
            completed_at = time.time()
            latency_ms = (completed_at - sent_at) * 1000
            stats.err += 1
            status = f"error:{type(e).__name__}"
            actual_output_tokens = 0
        finally:
            stats.in_flight -= 1
            sem.release()

        if record_results:
            results.append({
                "program_id": program.name,
                "sent_at": sent_at,
                "completed_at": completed_at,
                "latency_ms": round(latency_ms, 2),
                "status": status,
                "prompt_tokens": program.prompt_tokens,
                "output_tokens": actual_output_tokens,
            })

    while time.monotonic() < wall_end:
        sleep_for = next_send - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

        # Acquire semaphore — waits if all slots are in use (no drops).
        await sem.acquire()
        stats.sent += 1
        pending.append(asyncio.create_task(send_one()))
        next_send += interval

    # Drain all in-flight / queued requests.
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
# Warmup
# ---------------------------------------------------------------------------

async def run_warmup(cfg: dict, model: str, gateway: str, session: aiohttp.ClientSession, timeout: float):
    warmup = cfg.get("test", {}).get("warmup", {})
    seconds = float(warmup.get("seconds", 0))
    if seconds <= 0:
        return

    rate         = float(warmup.get("rate", 1.0))
    prompt_tokens = int(warmup.get("prompt_tokens", 128))
    max_tokens    = int(warmup.get("max_tokens", 64))

    print(f"[warmup] {seconds}s at {rate} req/s (prompt={prompt_tokens} max={max_tokens}) ...")
    prog = ProgramConfig(
        name="warmup",
        rate=rate,
        concurrency=max(4, int(rate * 2)),
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        start_time=0,
        duration=seconds,
        no_fairness_header=True,
    )
    stats = ProgramStats(name="warmup")
    await run_program(
        program=prog, model=model, gateway=gateway,
        measurement_start=time.monotonic(),
        stats=stats, results=[], record_results=False,
        session=session, timeout=timeout,
    )
    print(f"[warmup] Done. sent={stats.sent} ok={stats.ok} err={stats.err}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(scenario_path: str, phase_name: str, gateway: str, output_dir: str):
    with open(scenario_path) as f:
        cfg = yaml.safe_load(f)

    model    = cfg["model"]
    test_cfg = cfg.get("test", {})
    duration = float(test_cfg.get("duration", 120))
    timeout  = float(test_cfg.get("timeout", 60))

    programs: List[ProgramConfig] = []
    for name, pc in cfg.get("programs", {}).items():
        programs.append(ProgramConfig(
            name=name,
            rate=float(pc.get("rate", 1.0)),
            concurrency=int(pc.get("concurrency", 4)),
            prompt_tokens=int(pc.get("prompt_tokens", 512)),
            max_tokens=int(pc.get("max_tokens", 128)),
            start_time=float(pc.get("start_time", 0)),
            duration=float(pc.get("duration", duration)),
            no_fairness_header=bool(pc.get("no_fairness_header", False)),
        ))

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.jsonl")

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        await run_warmup(cfg, model, gateway, session, timeout)

        print(f"[{phase_name}] Starting: {len(programs)} programs, {duration}s")
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
                record_results=True, session=session, timeout=timeout,
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