#!/usr/bin/env python3
"""
Async load generator for fairness A/B testing.

Reads a scenario YAML to determine per-program workload profiles, then spawns
one async sender per program instance. Each instance sends requests at its
configured rate with a unique x-gateway-inference-fairness-id header.
Results are logged as JSONL for post-hoc analysis.

Usage:
    python3 fairness_loadgen.py \
        --scenario scenarios/stress-h100.yaml \
        --gateway-url http://localhost:30080 \
        --output results/baseline/results.jsonl

CLI args override YAML values when provided.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field

import aiohttp
import yaml

# Base sentence (~12 tokens) repeated to approximate target prompt length.
_BASE_SENTENCE = (
    "The quick brown fox jumps over the lazy dog near the riverbank. "
)


def generate_prompt(target_tokens: int) -> str:
    """Generate a prompt of approximately target_tokens length."""
    repeats = max(1, target_tokens // 12)
    return _BASE_SENTENCE * repeats


@dataclass
class ProgramInstance:
    """A single sender instance with a unique fairness ID."""
    fairness_id: str
    rate: float
    prompt: str
    max_tokens: int
    background: bool = True
    start_offset: float = 0.0        # seconds after warmup ends (foreground only)
    active_duration: float | None = None  # None = full test duration (background)


@dataclass
class RequestResult:
    program_id: str
    request_id: str
    sent_at: float
    completed_at: float
    total_ms: float
    status: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


@dataclass
class Stats:
    """Live stats tracker per program instance."""
    sent: int = 0
    completed: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    latencies: list = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.completed if self.completed > 0 else 0.0

    @property
    def p50_ms(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[len(s) // 2]


def expand_programs(programs_cfg: dict) -> list[ProgramInstance]:
    """Expand program configs with count into individual instances.

    Each program can be:
      - background: true  → runs for full test duration (warmup + measurement)
      - background: false (or absent with start_time/duration present)
          → foreground program, must have start_time and duration
      - No background/start_time/duration fields → defaults to background (backward compat)
    """
    instances = []
    for name, cfg in programs_cfg.items():
        count = cfg.get("count", 1)
        rate = cfg["rate"]
        prompt = generate_prompt(cfg["prompt_tokens"])
        max_tokens = cfg["max_tokens"]

        # Determine background vs foreground.
        has_background = "background" in cfg
        has_timing = "start_time" in cfg or "duration" in cfg

        if has_background and cfg["background"]:
            background = True
        elif has_timing:
            background = False
        elif has_background and not cfg["background"]:
            background = False
        else:
            # No fields specified — backward compat: treat as background.
            background = True

        start_offset = 0.0
        active_duration = None
        if not background:
            if "start_time" not in cfg or "duration" not in cfg:
                raise ValueError(
                    f"Program '{name}': foreground programs (background=false) "
                    f"require both 'start_time' and 'duration'"
                )
            start_offset = float(cfg["start_time"])
            active_duration = float(cfg["duration"])

        for i in range(count):
            fairness_id = name if count == 1 else f"{name}-{i}"
            instances.append(ProgramInstance(
                fairness_id=fairness_id,
                rate=rate,
                prompt=prompt,
                max_tokens=max_tokens,
                background=background,
                start_offset=start_offset,
                active_duration=active_duration,
            ))
    return instances


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    instance: ProgramInstance,
    timeout_sec: float,
) -> RequestResult:
    """Send a single completion request and return timing result."""
    request_id = str(uuid.uuid4())[:8]
    payload = {
        "model": model,
        "prompt": instance.prompt,
        "max_tokens": instance.max_tokens,
        "temperature": 0,
        "ignore_eos": True,
    }
    headers = {
        "Content-Type": "application/json",
        "x-gateway-inference-fairness-id": instance.fairness_id,
    }

    sent_at = time.time()
    try:
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout_sec),
        ) as resp:
            completed_at = time.time()
            total_ms = (completed_at - sent_at) * 1000

            if resp.status == 200:
                body = await resp.json()
                usage = body.get("usage", {})
                return RequestResult(
                    program_id=instance.fairness_id,
                    request_id=request_id,
                    sent_at=sent_at,
                    completed_at=completed_at,
                    total_ms=total_ms,
                    status="ok",
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
            else:
                text = await resp.text()
                return RequestResult(
                    program_id=instance.fairness_id,
                    request_id=request_id,
                    sent_at=sent_at,
                    completed_at=completed_at,
                    total_ms=total_ms,
                    status=f"http_{resp.status}",
                    error=text[:200],
                )
    except asyncio.TimeoutError:
        completed_at = time.time()
        return RequestResult(
            program_id=instance.fairness_id,
            request_id=request_id,
            sent_at=sent_at,
            completed_at=completed_at,
            total_ms=(completed_at - sent_at) * 1000,
            status="timeout",
            error=f"Request timed out after {timeout_sec}s",
        )
    except Exception as e:
        completed_at = time.time()
        return RequestResult(
            program_id=instance.fairness_id,
            request_id=request_id,
            sent_at=sent_at,
            completed_at=completed_at,
            total_ms=(completed_at - sent_at) * 1000,
            status="error",
            error=str(e)[:200],
        )


async def instance_sender(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    instance: ProgramInstance,
    duration: float,
    warmup: float,
    timeout_sec: float,
    results: list[RequestResult],
    stats: Stats,
    start_time: float,
    concurrency_limit: int,
):
    """Send requests for one program instance at the configured rate.

    Background programs run from start_time through start_time + warmup + duration.
    Foreground programs run from measurement_start + start_offset for active_duration seconds.
    """
    interval = 1.0 / instance.rate
    sem = asyncio.Semaphore(concurrency_limit)

    measurement_start = start_time + warmup

    if instance.background:
        send_start = start_time
        send_end = start_time + warmup + duration
    else:
        send_start = measurement_start + instance.start_offset
        send_end = send_start + instance.active_duration

    async def do_request():
        async with sem:
            result = await send_request(session, url, model, instance, timeout_sec)
            stats.sent += 1
            # Only record results after warmup for background programs.
            # Foreground programs always start after warmup, so always record.
            if result.sent_at >= measurement_start:
                results.append(result)
                if result.status == "ok":
                    stats.completed += 1
                    stats.total_latency_ms += result.total_ms
                    stats.latencies.append(result.total_ms)
                else:
                    stats.errors += 1

    # Wait until our send window opens.
    now = time.time()
    if now < send_start:
        await asyncio.sleep(send_start - now)

    tasks = []
    next_send = send_start
    while True:
        now = time.time()
        if now >= send_end:
            break

        if now < next_send:
            await asyncio.sleep(next_send - now)

        tasks.append(asyncio.create_task(do_request()))
        next_send += interval

    # Drain in-flight requests with progress reporting.
    pending = [t for t in tasks if not t.done()]
    if pending:
        total = len(tasks)
        print(f"\n  {instance.fairness_id}: window ended, {len(pending)}/{total} in-flight, draining...")
        while pending:
            done, pending = await asyncio.wait(pending, timeout=5)
            if pending:
                print(f"  {instance.fairness_id}: {len(pending)}/{total} still in-flight...")
    elif tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def print_live_status(
    instances: list[ProgramInstance],
    stats_map: dict[str, Stats],
    elapsed: float,
    total_wall: float,
    warmup: float,
):
    """Print a single-line status update."""
    parts = [f"\r[{elapsed:.0f}/{total_wall:.0f}s]"]
    measurement_elapsed = elapsed - warmup
    for inst in instances:
        s = stats_map[inst.fairness_id]
        # Determine program state for foreground programs.
        if not inst.background:
            if measurement_elapsed < inst.start_offset:
                parts.append(f" {inst.fairness_id}:waiting")
                continue
            elif measurement_elapsed >= inst.start_offset + inst.active_duration:
                avg = f"{s.avg_latency_ms:.0f}" if s.completed > 0 else "-"
                parts.append(f" {inst.fairness_id}:done({s.completed}ok/avg={avg}ms)")
                continue
        avg = f"{s.avg_latency_ms:.0f}" if s.completed > 0 else "-"
        parts.append(f" {inst.fairness_id}:{s.completed}ok/{s.errors}err/avg={avg}ms")
    sys.stdout.write("".join(parts))
    sys.stdout.flush()


async def status_printer(
    instances: list[ProgramInstance],
    stats_map: dict[str, Stats],
    start_time: float,
    total_wall: float,
    warmup: float,
):
    """Periodically print live stats."""
    while True:
        await asyncio.sleep(5)
        elapsed = time.time() - start_time
        if elapsed >= total_wall:
            break
        print_live_status(instances, stats_map, elapsed, total_wall, warmup)
    sys.stdout.write("\n")


async def run_loadgen(args):
    # Load scenario YAML.
    with open(args.scenario) as f:
        scenario = yaml.safe_load(f)

    # Extract settings (CLI args override YAML).
    model = args.model or scenario["model"]
    test_cfg = scenario.get("test", {})
    duration = args.duration if args.duration is not None else test_cfg.get("duration", 180)
    warmup = args.warmup if args.warmup is not None else test_cfg.get("warmup", 30)
    timeout = args.timeout if args.timeout is not None else test_cfg.get("timeout", 60)
    concurrency = args.concurrency if args.concurrency is not None else test_cfg.get("concurrency", 50)
    gateway_url = args.gateway_url or scenario.get("infra", {}).get("gateway_url", "http://localhost:30080")

    # Expand programs into instances.
    instances = expand_programs(scenario["programs"])
    total_rate = sum(inst.rate for inst in instances)

    # Auto-extend duration if any foreground program ends later.
    effective_duration = duration
    for inst in instances:
        if not inst.background and inst.active_duration is not None:
            end = inst.start_offset + inst.active_duration
            effective_duration = max(effective_duration, end)
    if effective_duration > duration:
        print(f"NOTE: Auto-extending measurement duration from {duration}s to {effective_duration}s "
              f"to cover all foreground programs.")
        duration = effective_duration

    print("Load generator configuration:")
    print(f"  Gateway:     {gateway_url}")
    print(f"  Model:       {model}")
    print(f"  Scenario:    {args.scenario}")
    print(f"  Duration:    {duration}s (+ {warmup}s warmup)")
    print(f"  Concurrency: {concurrency} per instance")
    print(f"  Instances:   {len(instances)}")
    for inst in instances:
        if inst.background:
            timing = "background (full duration)"
        else:
            timing = f"start={inst.start_offset}s, duration={inst.active_duration}s"
        print(f"    {inst.fairness_id}: {inst.rate} req/s, ~{len(inst.prompt)//4} prompt tokens, "
              f"{inst.max_tokens} max output tokens, {timing}")
    print(f"  Total rate:  {total_rate} req/s (peak, when all active)")
    print(f"  Output:      {args.output}")
    print()

    # Verify gateway is reachable.
    url = f"{gateway_url.rstrip('/')}/v1/completions"
    async with aiohttp.ClientSession() as session:
        try:
            test_payload = {
                "model": model,
                "prompt": "hello",
                "max_tokens": 1,
                "temperature": 0,
                "ignore_eos": True,
            }
            async with session.post(
                url,
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print(f"WARNING: Gateway returned {resp.status}: {text[:200]}")
                else:
                    print("Gateway health check passed.")
        except Exception as e:
            print(f"ERROR: Cannot reach gateway: {e}")
            sys.exit(1)

    # Run the test.
    results_map: dict[str, list[RequestResult]] = {inst.fairness_id: [] for inst in instances}
    stats_map: dict[str, Stats] = {inst.fairness_id: Stats() for inst in instances}
    start_time = time.time()

    print(f"\nStarting load generation (warmup={warmup}s, measurement={duration}s)...")

    async with aiohttp.ClientSession() as session:
        sender_tasks = []
        for inst in instances:
            task = asyncio.create_task(
                instance_sender(
                    session=session,
                    url=url,
                    model=model,
                    instance=inst,
                    duration=duration,
                    warmup=warmup,
                    timeout_sec=timeout,
                    results=results_map[inst.fairness_id],
                    stats=stats_map[inst.fairness_id],
                    start_time=start_time,
                    concurrency_limit=concurrency,
                )
            )
            sender_tasks.append(task)

        total_wall = warmup + duration
        printer_task = asyncio.create_task(
            status_printer(instances, stats_map, start_time, total_wall, warmup)
        )

        await asyncio.gather(*sender_tasks, return_exceptions=True)
        printer_task.cancel()

    # Combine and write results.
    all_results = []
    for inst in instances:
        all_results.extend(results_map[inst.fairness_id])
    all_results.sort(key=lambda r: r.sent_at)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for r in all_results:
            f.write(json.dumps(r.__dict__) + "\n")

    # Print summary.
    total_elapsed = time.time() - start_time
    print(f"\nDone in {total_elapsed:.1f}s. Wrote {len(all_results)} results to {args.output}")
    print()
    print(f"{'Instance':<20} {'Sent':>6} {'OK':>6} {'Err':>5} {'Avg(ms)':>10} {'P50(ms)':>10}")
    print("-" * 65)
    for inst in instances:
        s = stats_map[inst.fairness_id]
        latencies = sorted(s.latencies) if s.latencies else []
        avg = f"{s.avg_latency_ms:.1f}" if s.completed > 0 else "-"
        p50 = f"{latencies[len(latencies)//2]:.1f}" if latencies else "-"
        print(f"{inst.fairness_id:<20} {s.sent:>6} {s.completed:>6} {s.errors:>5} {avg:>10} {p50:>10}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Fairness A/B load generator")
    parser.add_argument("--scenario", required=True, help="Path to scenario YAML file")
    parser.add_argument("--gateway-url", default=None, help="Gateway URL (overrides YAML)")
    parser.add_argument("--model", default=None, help="Model name (overrides YAML)")
    parser.add_argument("--duration", type=int, default=None, help="Measurement duration in seconds (overrides YAML)")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup duration in seconds (overrides YAML)")
    parser.add_argument("--timeout", type=float, default=None, help="Per-request timeout in seconds (overrides YAML)")
    parser.add_argument("--concurrency", type=int, default=None, help="Max concurrent requests per instance (overrides YAML)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()
    asyncio.run(run_loadgen(args))


if __name__ == "__main__":
    main()