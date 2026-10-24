# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_FA through the engine: scheduler, model runners, graphs.

Each run builds a two-layer model from a flagship architecture with dummy
weights (attention dims unchanged, everything else shrunk) and compares its
greedy logprobs with the same model on TRITON_ATTN in eager mode. The batch
mixes long prompts (chunked prefill -> extends), a shared prefix (prefix-cache
hits -> extends) and decode steps, so every phase runs under each:

- CUDA graph mode: NONE, PIECEWISE, FULL_DECODE_ONLY, FULL_AND_PIECEWISE
- async scheduling on/off
- model runner v1 / v2
- KV cache dtype auto / fp8
- n-gram speculative decoding (multi-token decode rows)

Dummy weights make greedy tokens fragile, so the comparison is
``check_logprobs_close``: tokens may diverge only where the other run ranks
the token in its top-k.
"""

import pytest

from tests.models.utils import check_logprobs_close
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCM_AITER_FA is ROCm only"
)

# Qwen3-235B-A22B attention (64 q heads, 4 KV heads, head 128); MoE shrunk.
QWEN3 = (
    "Qwen/Qwen3-235B-A22B",
    {
        "num_hidden_layers": 2,
        "hidden_size": 1024,
        "intermediate_size": 1024,
        "moe_intermediate_size": 256,
        "num_experts": 8,
        "num_experts_per_tok": 2,
    },
)

MAX_TOKENS = 24
NUM_LOGPROBS = 5
SHARED_PREFIX = "You are a careful assistant. Answer briefly and cite sources. " * 8


def _prompts() -> list[str]:
    long = " ".join(f"item{i}" for i in range(200))  # > max_num_batched_tokens
    return [
        "Hello",
        SHARED_PREFIX + "What is the capital of France?",
        SHARED_PREFIX + "What is the capital of Japan?",
        long,
        "one two three one two three one two three one two three",
        "The quick brown fox jumps over the lazy dog. " * 20,
    ]


def _run(
    vllm_runner,
    monkeypatch,
    *,
    backend,
    model=QWEN3,
    kv_cache_dtype="auto",
    cudagraph_mode="NONE",
    async_scheduling=False,
    model_runner_v2=False,
    speculative=False,
):
    name, overrides = model
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if model_runner_v2 else "0")
    kwargs = {}
    if speculative:
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 2,
        }
    eager = cudagraph_mode == "NONE"
    with vllm_runner(
        name,
        load_format="dummy",
        hf_overrides=overrides,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=512,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        kv_cache_dtype=kv_cache_dtype,
        attention_config={"backend": backend},
        enforce_eager=eager,
        compilation_config=None
        if eager
        else {
            "cudagraph_mode": cudagraph_mode,
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
        },
        async_scheduling=async_scheduling,
        gpu_memory_utilization=0.3,
        **kwargs,
    ) as llm:
        return llm.generate_greedy_logprobs(_prompts(), MAX_TOKENS, NUM_LOGPROBS)


_REFERENCE: dict[str, list] = {}


def _reference(vllm_runner, monkeypatch, kv_cache_dtype):
    """TRITON_ATTN, eager, runner v1: computed once per KV dtype."""
    if kv_cache_dtype not in _REFERENCE:
        _REFERENCE[kv_cache_dtype] = _run(
            vllm_runner,
            monkeypatch,
            backend="TRITON_ATTN",
            kv_cache_dtype=kv_cache_dtype,
        )
    return _REFERENCE[kv_cache_dtype]


def _compare(vllm_runner, monkeypatch, kv_cache_dtype="auto", **kwargs):
    ref = _reference(vllm_runner, monkeypatch, kv_cache_dtype)
    out = _run(
        vllm_runner,
        monkeypatch,
        backend="ROCM_AITER_FA",
        kv_cache_dtype=kv_cache_dtype,
        **kwargs,
    )
    check_logprobs_close(
        outputs_0_lst=ref,
        outputs_1_lst=out,
        name_0="TRITON_ATTN",
        name_1="ROCM_AITER_FA",
    )


CORE = [
    # (cudagraph_mode, async_scheduling, model_runner_v2)
    ("NONE", False, False),
    ("PIECEWISE", False, True),
    ("FULL_DECODE_ONLY", True, True),
    ("FULL_AND_PIECEWISE", True, False),
]
FULL = [
    (mode, async_, v2)
    for mode in ("NONE", "PIECEWISE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")
    for async_ in (False, True)
    for v2 in (False, True)
    if (mode, async_, v2) not in CORE
]


@pytest.mark.parametrize("cudagraph_mode, async_scheduling, model_runner_v2", CORE)
def test_engine_modes(
    vllm_runner, monkeypatch, cudagraph_mode, async_scheduling, model_runner_v2
):
    _compare(
        vllm_runner,
        monkeypatch,
        cudagraph_mode=cudagraph_mode,
        async_scheduling=async_scheduling,
        model_runner_v2=model_runner_v2,
    )


@pytest.mark.parametrize("cudagraph_mode", ["NONE", "FULL_AND_PIECEWISE"])
def test_fp8_kv_cache(vllm_runner, monkeypatch, cudagraph_mode):
    _compare(
        vllm_runner, monkeypatch, kv_cache_dtype="fp8", cudagraph_mode=cudagraph_mode
    )


@pytest.mark.parametrize(
    "cudagraph_mode, model_runner_v2",
    [("NONE", False), ("FULL_AND_PIECEWISE", False), ("FULL_AND_PIECEWISE", True)],
)
def test_ngram_spec_decode(vllm_runner, monkeypatch, cudagraph_mode, model_runner_v2):
    # Runner v1 has no async scheduling with n-gram drafts.
    _compare(
        vllm_runner,
        monkeypatch,
        speculative=True,
        cudagraph_mode=cudagraph_mode,
        model_runner_v2=model_runner_v2,
        async_scheduling=model_runner_v2,
    )


@pytest.mark.slow_test
@pytest.mark.parametrize("cudagraph_mode, async_scheduling, model_runner_v2", FULL)
def test_engine_modes_full(
    vllm_runner, monkeypatch, cudagraph_mode, async_scheduling, model_runner_v2
):
    _compare(
        vllm_runner,
        monkeypatch,
        cudagraph_mode=cudagraph_mode,
        async_scheduling=async_scheduling,
        model_runner_v2=model_runner_v2,
    )


@pytest.mark.skipif(current_platform.device_count() < 2, reason="needs 2 GPUs")
def test_context_parallel_is_rejected(vllm_runner, monkeypatch, capfd):
    """The impl declares neither PCP nor DCP; asking for DCP with this backend
    must fail backend selection at startup, not produce wrong attention."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    name, overrides = QWEN3
    # One KV head so DCP passes the TP > KV-heads check and reaches selection.
    with (
        pytest.raises(Exception, match="initialization failed"),
        vllm_runner(
            name,
            load_format="dummy",
            hf_overrides={**overrides, "num_key_value_heads": 1},
            tensor_parallel_size=2,
            decode_context_parallel_size=2,
            attention_config={"backend": "ROCM_AITER_FA"},
            enforce_eager=True,
            gpu_memory_utilization=0.3,
        ),
    ):
        pass
    # The reason is raised in the worker processes; read it from their output.
    out, err = capfd.readouterr()
    assert "DCP not supported" in out + err
