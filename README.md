# LLM Benchmarking: vLLM vs Ollama

This repository is designed to benchmark and compare the performance of [vLLM](https://github.com/vllm-project/vllm) and [Ollama](https://github.com/ollama/ollama) at scale. 

**Why Ultrachat500k?**
Instead of sending the same question repeatedly which would heavily trigger vLLM's caching mechanisms and artificially bias the results, we use diverse questions from the Ultrachat dataset. Testing against hundreds of thousands of unique prompts provides a realistic proxy for production workloads and properly assesses the true power of vLLM's PagedAttention mechanism without cache-induced bias. The benchmark progressively increases the number of concurrent requests in powers of two to measure scaling behavior.

## Project Structure

- `run_benchmark.py`: The main orchestration script. It starts/stops the respective Docker Compose services (for vLLM and Ollama) and runs the benchmarking script with exponentially increasing concurrency (1, 2, 4, 8, ... up to 1024).
- `benchmark_openai.py`: The benchmarking worker. It sends concurrent requests to the target API endpoint using `httpx` and `asyncio`, capturing latency and throughput metrics.
- `benchmark_analysis.ipynb`: A Jupyter Notebook to visualize and compare the benchmarking results.
- `vLLM/`: Contains the `docker-compose.yml` for deploying the vLLM server.
- `ollama/`: Contains the `docker-compose.yml` for deploying the Ollama server.
- `requirements.txt`: Python dependencies required for benchmarking and analysis.

## Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Benchmark:**
   Execute the `run_benchmark.py` script. Make sure you have Docker and Docker Compose installed, as the script will automatically manage the containers.
   ```bash
   python run_benchmark.py
   ```
   *Note: You may need to edit `run_benchmark.py` to match the specific model names you have pulled/configured for vLLM and Ollama.*

3. **Analyze Results:**
   Open `benchmark_analysis.ipynb` to compare the performance (e.g., throughput, latency) between vLLM and Ollama across different concurrency levels.
