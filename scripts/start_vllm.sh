#!/usr/bin/env bash
#
# Start vLLM with the Phase 1 configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Workload: 1.5-3K-token prompts (schema + question), short structured SQL
# outputs, ~2-3 dependent calls per agent run, target P95 < 5s @ 10+ RPS.
# MoE 30B-total/3B-active => weights dominate VRAM => quantize to free KV cache.

set -euo pipefail

# FP8 checkpoint: ~30GB weights instead of ~60GB (BF16) => ~45GB free for KV
# cache on the 80GB H100, plus Hopper FP8 tensor cores. If the FP8 repo is
# unavailable, serve the BF16 model and add `--quantization fp8` for online
# quantization instead.
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128
