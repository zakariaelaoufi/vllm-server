# auto_benchmark.py
#!/usr/bin/env python3

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).parent.resolve()

MAX_CONCURRENCY = 1024

VLLM_COMPOSE = ROOT / "vvlm" / "docker-compose.yml"
OLLAMA_COMPOSE = ROOT / "ollama" / "docker-compose.yml"

BENCH_SCRIPT = ROOT / "benchmark_openai.py"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =========================
# HELPERS
# =========================

def powers_of_two(max_value):
    v = 1
    vals = []

    while v <= max_value:
        vals.append(v)
        v *= 2

    return vals


def run(cmd, cwd=None):

    print(f"\n>>> {' '.join(cmd)}")

    subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
    )


def wait_health(url, timeout=900):

    import requests

    start = time.time()

    while time.time() - start < timeout:

        try:
            r = requests.get(url)

            if r.status_code == 200:
                return

        except Exception:
            pass

        time.sleep(5)

    raise RuntimeError(f"Timeout waiting for {url}")


def docker_compose_up(compose_path):

    run(
        [
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "up",
            "-d",
        ]
    )


def docker_compose_down(compose_path):

    run(
        [
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "down",
        ]
    )


def replace_in_file(path, old, new):

    txt = Path(path).read_text()

    txt = txt.replace(old, new)

    Path(path).write_text(txt)


# =========================
# CONFIGURE VLLM
# =========================

def configure_vllm(num_parallel):

    original = Path(VLLM_COMPOSE).read_text()

    import re

    updated = re.sub(
        r"--max-num-seqs\s+\d+",
        f"--max-num-seqs {num_parallel}",
        original,
    )

    Path(VLLM_COMPOSE).write_text(updated)


# =========================
# CONFIGURE OLLAMA
# =========================

def configure_ollama(num_parallel):

    original = Path(OLLAMA_COMPOSE).read_text()

    import re

    updated = re.sub(
        r"OLLAMA_NUM_PARALLEL=\d+",
        f"OLLAMA_NUM_PARALLEL={num_parallel}",
        original,
    )

    Path(OLLAMA_COMPOSE).write_text(updated)


# =========================
# BENCH
# =========================

def benchmark_backend(
    backend,
    compose_path,
    base_url,
    model,
    health_url,
    max_parallel,
):

    for concurrency in powers_of_two(MAX_CONCURRENCY):

        if concurrency > max_parallel:
            break

        print("\n====================================")
        print(f"{backend} | concurrency={concurrency}")
        print("====================================")

        docker_compose_down(compose_path)

        if backend == "vllm":
            configure_vllm(concurrency)

        elif backend == "ollama":
            configure_ollama(concurrency)

        docker_compose_up(compose_path)

        wait_health(health_url)

        output_file = (
            RESULTS_DIR
            / f"{backend}_c{concurrency}.json"
        )

        cmd = [
            sys.executable,
            str(BENCH_SCRIPT),

            "--backend",
            backend,

            "--base-url",
            base_url,

            "--model",
            model,

            "--requests",
            "1024",

            "--concurrency",
            str(concurrency),

            "--dataset-split",
            "test_sft",

            "--max-tokens",
            "200",

            "--seed",
            "42",

            "--capture-responses",

            "--output-file",
            str(output_file),
        ]

        run(cmd)

        docker_compose_down(compose_path)


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    benchmark_backend(
        backend="vllm",
        compose_path=VLLM_COMPOSE,
        base_url="http://localhost:8000",
        model="nvidia/Llama-3.1-8B-Instruct-NVFP4",
        health_url="http://localhost:8000/health",
        max_parallel=1024,
    )

    benchmark_backend(
        backend="ollama",
        compose_path=OLLAMA_COMPOSE,
        base_url="http://localhost:11438",
        model="llama3.1:latest",
        health_url="http://localhost:11438/api/tags",
        max_parallel=8,
    )