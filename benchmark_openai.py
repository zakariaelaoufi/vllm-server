# benchmark_openai.py
#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from typing import Tuple, List, Optional, Dict, Any

import httpx
import numpy as np
from datasets import load_dataset

try:
    from tqdm.asyncio import tqdm_asyncio
except ImportError:
    tqdm_asyncio = None


METRICS_FILE = "metrics_diff.json"


# =========================
# DATASET
# =========================

def load_ultrachat_prompts(
    split: str,
    num_samples: int,
    seed: int,
) -> List[str]:
    """
    Load prompts from HuggingFaceH4/ultrachat_200k.
    """

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=split,
    )

    rng = random.Random(seed)

    indices = rng.sample(range(len(ds)), num_samples)

    prompts = []

    for idx in indices:
        row = ds[idx]

        if row.get("prompt"):
            prompts.append(row["prompt"])
        else:
            msgs = row.get("messages", [])
            user_msg = next(
                (m["content"] for m in msgs if m["role"] == "user"),
                None
            )
            if user_msg:
                prompts.append(user_msg)

    return prompts


# =========================
# REQUEST
# =========================

async def _chat_completion(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: dict,
    capture_responses: bool,
) -> Tuple[Optional[float], Optional[int], Optional[Dict[str, Any]]]:

    t0 = time.perf_counter()

    try:
        r = await client.post(
            url,
            headers=headers,
            json=payload,
            timeout=120,
        )

        latency = time.perf_counter() - t0

        r.raise_for_status()

        data = r.json()

        usage = data.get("usage", {})

        tokens = usage.get(
            "total_tokens",
            usage.get("prompt_tokens", 0)
            + usage.get("completion_tokens", 0),
        )

        return (
            latency,
            tokens,
            data if capture_responses else None,
        )

    except Exception as e:
        print(f"Request failed: {e}")
        return None, None, None


# =========================
# REPORT
# =========================

def save_metrics(metrics: dict, path: str = METRICS_FILE):

    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(metrics)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_responses_to_file(
    responses: List[Dict[str, Any]],
    filename: str
):

    directory = os.path.dirname(filename)

    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "w") as f:
        json.dump(responses, f, indent=2)

    print(f"\nResponses written to {filename}")


def _report(
    latencies: List[float],
    tokens: List[int],
    total_req: int,
    wall: float,
    requests_count: int,
    model_name: str,
    url: str,
    concurrency: int,
    backend: str,
    dataset_split: str,
):

    ok = len(latencies)

    print(f"\n✔ {ok}/{total_req} requests succeeded in {wall:.2f}s")

    if ok == 0:
        return

    rps = ok / wall

    print(f"Requests/s:     {rps:10.2f}")

    tps = 0.0

    if any(tokens):
        tps = sum(tokens) / wall
        print(f"Tokens/s:       {tps:10.2f}")

    avg_latency = statistics.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    print(f"Avg latency:    {avg_latency:10.3f}s")
    print(f"p50 latency:    {p50:10.3f}s")
    print(f"p95 latency:    {p95:10.3f}s\n")

    metrics = {
        "timestamp": time.time(),
        "backend": backend,
        "model": model_name,
        "url": url,
        "dataset_split": dataset_split,
        "requests": requests_count,
        "concurrency": concurrency,
        "requests_per_sec": rps,
        "tokens_per_sec": tps,
        "avg_latency_s": avg_latency,
        "p50_latency_s": float(p50),
        "p95_latency_s": float(p95),
        "wall_time_s": wall,
    }

    save_metrics(metrics)


# =========================
# MAIN BENCH
# =========================

async def _run_once(args):

    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        **(
            {"Authorization": f"Bearer {args.api_key}"}
            if args.api_key
            else {}
        ),
    }

    prompts = load_ultrachat_prompts(
        split=args.dataset_split,
        num_samples=args.requests,
        seed=args.seed,
    )

    sem = asyncio.Semaphore(args.concurrency)

    latencies: List[float] = []
    tokens: List[int] = []
    responses: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(
        http2=True,
        timeout=None,
    ) as client:

        # Warmup
        warm_payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False,
        }

        await _chat_completion(
            client,
            url,
            headers,
            warm_payload,
            False,
        )

        async def worker(prompt: str):

            async with sem:

                payload = {
                    "model": args.model,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "stream": False,
                }

                l, t, resp = await _chat_completion(
                    client,
                    url,
                    headers,
                    payload,
                    args.capture_responses,
                )

                if l is not None:
                    latencies.append(l)
                    tokens.append(t)

                    if resp and args.capture_responses:
                        responses.append(resp)

        tic = time.perf_counter()

        tasks = [
            asyncio.create_task(worker(prompt))
            for prompt in prompts
        ]

        if tqdm_asyncio and not args.quiet:
            await tqdm_asyncio.gather(*tasks)
        else:
            await asyncio.gather(*tasks)

        toc = time.perf_counter()

    _report(
        latencies=latencies,
        tokens=tokens,
        total_req=args.requests,
        wall=toc - tic,
        requests_count=args.requests,
        model_name=args.model,
        url=url,
        concurrency=args.concurrency,
        backend=args.backend,
        dataset_split=args.dataset_split,
    )

    if args.capture_responses and responses:
        _write_responses_to_file(
            responses,
            args.output_file,
        )


# =========================
# CLI
# =========================

def _parse():

    p = argparse.ArgumentParser()

    p.add_argument("--backend", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)

    p.add_argument("--api-key", default="")

    p.add_argument("--requests", type=int, default=128)
    p.add_argument("--concurrency", type=int, default=8)

    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument(
        "--dataset-split",
        default="test_sft",
        choices=[
            "train_sft",
            "test_sft",
            "train_gen",
            "test_gen",
        ],
    )

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--quiet", action="store_true")

    p.add_argument(
        "--capture-responses",
        action="store_true",
    )

    p.add_argument(
        "--output-file",
        default="responses.json",
    )

    return p.parse_args()


if __name__ == "__main__":

    args = _parse()

    asyncio.run(_run_once(args))