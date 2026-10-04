#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Qwen3.8-27B W4A16 + DFlash2 width-7 on AMD gfx1151 (Strix Halo / Radeon 8060S)
# AGENT configuration: long context, prefix caching on, latency-tuned.
#
# This is the recipe in recipes/gfx1151_qwen38_27b_w4a16.md as a runnable file.
# For the exact command that produced the published benchmark numbers, use
# recipes/serve_gfx1151_qwen38_bench.sh instead.
#
# Usage:  ./serve_gfx1151_qwen38_agent.sh [logfile] [extra vllm serve args...]
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=${REPO:-/app/qwen38opt/strixhalo}
LOG=$(readlink -f "${1:-/tmp/vllm_agent.log}")
shift || true
cd "$REPO"

# --- model + draft (pin the revisions: the fast paths are tuned for them) ----
MODEL=${MODEL:-dbirks/Qwen3.8-27B-W4A16-AutoRound}
MODEL_REVISION=${MODEL_REVISION:-1f05c441c4e64ae0549de44fa9ea5a6d43610314}
DRAFT_MODEL=${DRAFT_MODEL:-syvai/Qwen3.8-27B-DFlash2-W4A16}
DRAFT_REVISION=${DRAFT_REVISION:-4d30ec736ffc6b8688dc2ae2b502d9b48bdec279}
SPEC_TOKENS=${SPEC_TOKENS:-7}

# --- agent sizing ------------------------------------------------------------
# MAX_SEQS is the single most important knob here: with a 7-token draft, the
# verified token batch is MAX_SEQS * 8, and only batch <= 8 reaches the fast
# K-tiled LDS W4A16 decode kernel. 1-2 = lowest per-token latency, 4+ = higher
# aggregate throughput but every request gets slower.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_SEQS=${MAX_SEQS:-2}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
# Prefix caching: ON. Measured identical decode performance with it on and off
# (ITL 100.73 vs 100.31 ms, acceptance 3.91 both) and a 6.1x TTFT win on a
# repeated 9,111-token prompt (39.94 s -> 6.54 s), which is exactly the agent
# pattern of resending a system prompt and tool schema every turn.
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
# Tool calling: without these vLLM rejects any request carrying `tools` with
# HTTP 400 ('"auto" tool choice requires --enable-auto-tool-choice and
# --tool-call-parser to be set').
#
# The parser must match the checkpoint's chat template. `qwen3_xml` (vLLM's
# Qwen3EngineToolParser) is the one recipes.vllm.ai documents for Qwen3.8-27B.
# `hermes` does NOT work here: the request returns 200 but tool_calls comes back
# null and the raw markup leaks into `content`. Set TOOL_PARSER="" to disable
# tool serving entirely.
TOOL_PARSER=${TOOL_PARSER-qwen3_xml}
# The rust API frontend is what recipes.vllm.ai pairs with this model, but it
# needs the `vllm-rs` binary: with VLLM_RUST_FRONTEND_PATH=auto, envs.py raises
# FileNotFoundError when vllm/vllm-rs is absent, and this tree was not built with
# setuptools-rust. Off by default; set RUST_FRONTEND=1 once you build it.
RUST_FRONTEND=${RUST_FRONTEND:-0}
PORT=${PORT:-8001}

# --- environment -------------------------------------------------------------
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-/app/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export SPT_NOENV=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
if [ "$RUST_FRONTEND" = "1" ]; then
  export VLLM_USE_RUST_FRONTEND=1
fi
# Persistent compile cache: without it every restart pays the full ~3.7 min
# startup again. The benchmark script uses /tmp, which is wiped.
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-${HOME:-/root}/.cache/vllm_gfx1151_qwen38}

# gfx1151 optimization gates. The first two are OFF by default in vllm/envs.py
# and carry most of the measured speedup, so they must be set explicitly.
export VLLM_GFX1151_W4_LDS_TILE=${VLLM_GFX1151_W4_LDS_TILE:-1}   # +30% at C1
export VLLM_GFX1151_W4_LOGITS=${VLLM_GFX1151_W4_LOGITS:-1}       # +8% at C1
# Prefill-sized M (>=1024): stream-dequantize int4 weights to a bf16 workspace
# and run rocBLAS mm (26.4 TF vs ~16 TF fused). -29% GEMM time per big chunk.
export VLLM_GFX1151_W4_DEQUANT_MM=${VLLM_GFX1151_W4_DEQUANT_MM:-1}
export VLLM_GFX1151_W4_LOGITS_TOPK=${VLLM_GFX1151_W4_LOGITS_TOPK:-256}
# These three default to on in this tree; pinned here so the recipe is explicit.
export VLLM_GFX1151_QWEN_ATTENTION=${VLLM_GFX1151_QWEN_ATTENTION:-1}
export VLLM_GFX1151_QWEN_GDN=${VLLM_GFX1151_QWEN_GDN:-1}
export VLLM_GFX1151_W4_GROUP_MAJOR=${VLLM_GFX1151_W4_GROUP_MAJOR:-1}
export VLLM_GFX1151_QWEN_VERIFY_TILE=${VLLM_GFX1151_QWEN_VERIFY_TILE:-16,128,8}
export VLLM_GFX1151_QWEN_VERIFY_SPLITKV=${VLLM_GFX1151_QWEN_VERIFY_SPLITKV:-1}
# Must be >= --max-model-len or every tuned attention path silently switches off
# at graph-capture time. See the recipe's "Context length" section.
export VLLM_GFX1151_QWEN_MAX_CONTEXT=${VLLM_GFX1151_QWEN_MAX_CONTEXT:-$MAX_MODEL_LEN}

# Reasoning parser: makes vLLM split the model's thinking out of `content` into
# OpenAI's `reasoning_content`, which pi renders as a separate thinking block,
# and it is what unlocks the `thinking_token_budget` request field. Empty means
# thinking stays inline in the text (works, just less tidy).
REASONING_PARSER=${REASONING_PARSER-qwen3}
if [ -n "$REASONING_PARSER" ]; then
  REASONING_FLAGS="--reasoning-parser $REASONING_PARSER"
else
  REASONING_FLAGS=""
fi

if [ -n "$TOOL_PARSER" ]; then
  TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser $TOOL_PARSER"
else
  TOOL_FLAGS=""
fi

if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
  PREFIX_CACHING_FLAG="--enable-prefix-caching"
else
  PREFIX_CACHING_FLAG="--no-enable-prefix-caching"
fi

echo "log: $LOG"
echo "cache: $VLLM_CACHE_ROOT"
echo "context: $MAX_MODEL_LEN, max-num-seqs: $MAX_SEQS (verify batch $((MAX_SEQS * 8)))"

exec /opt/venv/bin/vllm serve "$MODEL" \
  --revision "$MODEL_REVISION" \
  --tokenizer-revision "$MODEL_REVISION" \
  --served-model-name "$MODEL" \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --cpu-offload-gb 0 \
  --language-model-only --generation-config vllm \
  ${PREFIX_CACHING_FLAG} \
  ${TOOL_FLAGS} \
  ${REASONING_FLAGS} \
  --compilation-config '{"inductor_compile_config":{"deterministic":true,"combo_kernels":false,"benchmark_combo_kernel":false}}' \
  --speculative-config "{\"model\":\"${DRAFT_MODEL}\",\"revision\":\"${DRAFT_REVISION}\",\"method\":\"dflash\",\"num_speculative_tokens\":${SPEC_TOKENS}}" \
  --host 127.0.0.1 --port "$PORT" "$@" >>"$LOG" 2>&1
