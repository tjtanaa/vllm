#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Qwen3.8-27B W4A16 + DFlash2 width-7 on AMD gfx1151 (Strix Halo / Radeon 8060S)
# BENCHMARK configuration: exactly what produced the published numbers.
#
#   concurrency 1: 42.69 tok/s (median ITL 125.39 ms, acceptance 6.03)
#   concurrency 2: 49.07 tok/s (median ITL 219.79 ms, acceptance 6.18)
#   concurrency 4: 68.18 tok/s (median ITL 253.74 ms, acceptance 6.04)
#   concurrency 8: 65.79 tok/s (median ITL 521.60 ms, acceptance 6.35)
#
# Protocol: vllm bench serve, 32 prompts x 1024 in / 1024 out, temperature 0,
# seed 42, each concurrency run as the FIRST workload on a freshly started
# server (sustained load throttles this package ~22% after ~4 minutes, so a
# warm server measures low). Prefix caching is off and the KV cache is pinned
# to 16 GiB for run-to-run comparability - both are wrong choices for real use;
# see recipes/serve_gfx1151_qwen38_agent.sh.
#
# Usage:  ./serve_gfx1151_qwen38_bench.sh <logfile> [extra vllm serve args...]
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=${REPO:-/app/qwen38opt/strixhalo}
LOG=$(readlink -f "${1:?log file required}")
shift || true
cd "$REPO"

export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export HF_HOME=${HF_HOME:-/app/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=1
export SPT_NOENV=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/vllm_cache_c1}

export VLLM_GFX1151_W4_LDS_TILE=1
export VLLM_GFX1151_W4_LOGITS=1
export VLLM_GFX1151_W4_LOGITS_TOPK=256
export VLLM_GFX1151_QWEN_ATTENTION=1
export VLLM_GFX1151_QWEN_GDN=1
export VLLM_GFX1151_W4_GROUP_MAJOR=1
export VLLM_GFX1151_QWEN_VERIFY_TILE=16,128,8
export VLLM_GFX1151_QWEN_VERIFY_SPLITKV=1

exec /opt/venv/bin/vllm serve dbirks/Qwen3.8-27B-W4A16-AutoRound \
  --revision 1f05c441c4e64ae0549de44fa9ea5a6d43610314 \
  --tokenizer-revision 1f05c441c4e64ae0549de44fa9ea5a6d43610314 \
  --served-model-name dbirks/Qwen3.8-27B-W4A16-AutoRound \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --seed 42 \
  --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 17179869184 --gpu-memory-utilization 0.8 \
  --cpu-offload-gb 0 \
  --language-model-only --generation-config vllm --no-enable-prefix-caching \
  --compilation-config '{"inductor_compile_config":{"deterministic":true,"combo_kernels":false,"benchmark_combo_kernel":false}}' \
  --speculative-config '{"model":"syvai/Qwen3.8-27B-DFlash2-W4A16","revision":"4d30ec736ffc6b8688dc2ae2b502d9b48bdec279","method":"dflash","num_speculative_tokens":7}' \
  --host 127.0.0.1 --port 8001 "$@" >>"$LOG" 2>&1
