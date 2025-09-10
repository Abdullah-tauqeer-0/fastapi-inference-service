from __future__ import annotations

import argparse
import asyncio
from time import perf_counter

import httpx


async def _run_one(client: httpx.AsyncClient, url: str, payload: dict) -> float:
    start = perf_counter()
    response = await client.post(url, json=payload)
    response.raise_for_status()
    return (perf_counter() - start) * 1000


async def run_load_test(base_url: str, total_requests: int, concurrency: int) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    payload = {"features": [1.0, 0.5, -0.2]}

    async with httpx.AsyncClient(timeout=10.0) as client:
        async def task() -> None:
            async with semaphore:
                latency = await _run_one(client, f"{base_url}/predict", payload)
                latencies.append(latency)

        await asyncio.gather(*(task() for _ in range(total_requests)))

    sorted_latencies = sorted(latencies)

    def percentile(p: float) -> float:
        idx = int(round((p / 100) * (len(sorted_latencies) - 1)))
        idx = max(0, min(len(sorted_latencies) - 1, idx))
        return sorted_latencies[idx]

    p50 = percentile(50)
    p95 = percentile(95)
    print(f"Requests: {total_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"p50 latency: {p50:.2f} ms")
    print(f"p95 latency: {p95:.2f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple async load test for /predict")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Service base URL")
    parser.add_argument("--requests", type=int, default=200, help="Total request count")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent request count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_load_test(args.base_url, args.requests, args.concurrency))


if __name__ == "__main__":
    main()
