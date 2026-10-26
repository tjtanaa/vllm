# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_MLA through the engine: scheduler, model runners, graphs.

Each run builds a few-layer model from a real MLA architecture with dummy
weights (MLA dims unchanged, everything else shrunk) and compares its
logprobs with the same model on TRITON_MLA under the same engine settings.

- DeepSeek-V3: one dense + one MoE layer, at the per-rank head counts that
  pick each decode kernel: 128 (TP1 / DP attention), 16 (TP8, the asm
  minimum), 12 (Kimi-K3 at TP8: padded asm) and 8 (TP16: Gluon).
- Kimi-Linear: its smallest repeating block, 3 KDA layers + 1 MLA layer, so
  the MLA cache shares the KV pool and block size with the linear layers.
- GLM-4.7-Flash: nope 192 / v 256, the dims the fp8 asm prefill has no
  kernel for (it must take the varlen prefill).

The batch mixes long prompts (chunked prefill -> prefills with context), a
shared prefix (prefix-cache hits) and decode steps, so every phase runs under:

- CUDA graph mode: NONE, PIECEWISE, FULL_DECODE_ONLY, FULL_AND_PIECEWISE
- async scheduling on/off
- model runner v1 / v2
- KV cache dtype auto / fp8
- speculative decoding: n-gram and DeepSeek MTP (multi-token decode rows)
- decode context parallelism (TP2 x DCP2, 2 GPUs)

Dummy weights make greedy tokens fragile, so the comparison is
``check_logprobs_close``: tokens may diverge only where the other run ranks
the token in its top-k.
"""

from dataclasses import dataclass, field, replace
from functools import partial

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCM_AITER_MLA is ROCm only"
)


@dataclass(frozen=True)
class DummyModel:
    name: str
    hf_overrides: dict
    max_model_len: int = 2048
    # Prompt words; above max_num_batched_tokens so prefills are chunked.
    long_words: int = 700
    # Extra LM-head gain that makes the reference logits clearly peaked.
    logit_gain: float = 8.0
    # Mean prompt-logprob difference allowed vs TRITON_MLA, per KV dtype.
    tolerance: dict = field(default_factory=lambda: {"auto": 0.15, "fp8": 0.6})
    check_samples: bool = True
    engine_kwargs: dict = field(default_factory=dict)


def _deepseek_v3(num_heads: int, **extra) -> dict:
    # Layer 0 dense, layer 1 MoE with every expert active (no routing ties
    # for a backend difference to flip).
    return {
        "num_hidden_layers": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 2,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "moe_intermediate_size": 512,
        "intermediate_size": 2048,
        "num_nextn_predict_layers": 0,
        # Unquantized, so the dummy-weight rescale below reaches every matrix.
        "quantization_config": None,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        **extra,
    }


DSV3_H128 = DummyModel("deepseek-ai/DeepSeek-V3", _deepseek_v3(128))
DSV3_H16 = DummyModel("deepseek-ai/DeepSeek-V3", _deepseek_v3(16))
DSV3_H12 = DummyModel("deepseek-ai/DeepSeek-V3", _deepseek_v3(12))
DSV3_H8 = DummyModel("deepseek-ai/DeepSeek-V3", _deepseek_v3(8))
# The MTP layer drafts with the same MLA attention.
DSV3_MTP = DummyModel(
    "deepseek-ai/DeepSeek-V3", _deepseek_v3(16, num_nextn_predict_layers=1)
)
KIMI_LINEAR = DummyModel(
    "moonshotai/Kimi-Linear-48B-A3B-Instruct",
    {
        "num_hidden_layers": 4,
        "linear_attn_config": {
            "full_attn_layers": [4],
            "kda_layers": [1, 2, 3],
            "head_dim": 128,
            "num_heads": 32,
            "short_conv_kernel_size": 4,
        },
        "num_experts": 2,
        "num_experts_per_token": 2,
        "moe_intermediate_size": 512,
        "intermediate_size": 2048,
    },
    engine_kwargs={"trust_remote_code": True, "mamba_ssm_cache_dtype": "float32"},
)
GLM47_FLASH = DummyModel(
    "zai-org/GLM-4.7-Flash",
    {
        "num_hidden_layers": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 2,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "moe_intermediate_size": 512,
        "intermediate_size": 2048,
        "num_nextn_predict_layers": 0,
    },
)
MODELS = [
    pytest.param(DSV3_H128, id="dsv3_h128"),
    pytest.param(DSV3_H16, id="dsv3_h16"),
    pytest.param(DSV3_H12, id="dsv3_h12"),
    pytest.param(DSV3_H8, id="dsv3_h8"),
    pytest.param(KIMI_LINEAR, id="kimi_linear"),
    pytest.param(GLM47_FLASH, id="glm4_7_flash"),
]

MAX_TOKENS = 24
NUM_LOGPROBS = 5
SHARED_PREFIX = "You are a careful assistant. Answer briefly and cite sources. " * 8


def _prompts(model: DummyModel) -> list[str]:
    # Longer than max_num_batched_tokens, so it prefills in chunks.
    long = " ".join(f"item{i}" for i in range(model.long_words))
    return [
        "Hello",
        SHARED_PREFIX + "What is the capital of France?",
        SHARED_PREFIX + "What is the capital of Japan?",
        long,
        "one two three one two three one two three one two three",
        "The quick brown fox jumps over the lazy dog. " * 20,
    ]


# Dummy weights are uniform in +-1e-3, which leaves logits almost flat (every
# logprob near -log(vocab)), so a broken attention would still "match".
DUMMY_HALF_RANGE = 1e-3


def _scale_dummy_weights(model: torch.nn.Module, logit_gain: float) -> None:
    """Give norms unit gain, rescale every matrix to std 1/sqrt(fan_in) and the
    LM head further by ``logit_gain``, in place and identically on every
    backend, so the logits are peaked and greedy tokens depend on what
    attention returns. MLA's absorbed W_UK_T/W_UV are views refreshed from
    kv_b_proj below."""
    uniform_std = DUMMY_HALF_RANGE / 3**0.5
    for name, param in model.named_parameters():
        if name.endswith(("W_UK_T", "W_UV")):
            continue
        if param.dim() == 1 and "norm" in name:
            param.data.fill_(1.0)
        elif param.dim() >= 2 and param.dtype in (torch.bfloat16, torch.float16):
            gain = logit_gain if "lm_head" in name else 1.0
            param.data.mul_(gain * param.shape[-1] ** -0.5 / uniform_std)
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    for module in model.modules():
        if isinstance(module, MLAAttention):
            # Recompute the absorbed weights from the rescaled kv_b_proj.
            module.process_weights_after_loading(torch.bfloat16)


def _run(
    vllm_runner,
    monkeypatch,
    *,
    backend,
    model=DSV3_H16,
    kv_cache_dtype="auto",
    cudagraph_mode="NONE",
    async_scheduling=False,
    model_runner_v2=False,
    speculative=None,
    tensor_parallel_size=1,
    decode_context_parallel_size=1,
):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if model_runner_v2 else "0")
    kwargs = dict(model.engine_kwargs)
    if speculative == "ngram":
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 2,
        }
    elif speculative is not None:
        method, k = speculative
        kwargs["speculative_config"] = {"method": method, "num_speculative_tokens": k}
    if decode_context_parallel_size > 1:
        kwargs["decode_context_parallel_size"] = decode_context_parallel_size
    eager = cudagraph_mode == "NONE"
    with vllm_runner(
        model.name,
        load_format="dummy",
        hf_overrides=model.hf_overrides,
        dtype="bfloat16",
        max_model_len=model.max_model_len,
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
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.3,
        **kwargs,
    ) as llm:
        llm.apply_model(partial(_scale_dummy_weights, logit_gain=model.logit_gain))
        return llm.generate_greedy_logprobs(
            _prompts(model), MAX_TOKENS, NUM_LOGPROBS, num_prompt_logprobs=0
        )


_REFERENCE: dict[tuple, list] = {}


def _mean_prompt_logprob_diff(ref: list, out: list) -> float:
    diffs = []
    for (*_, ref_prompt), (*_, out_prompt) in zip(ref, out):
        for ref_lp, out_lp in zip(ref_prompt[1:], out_prompt[1:]):
            token, logprob = next(iter(ref_lp.items()))
            diffs.append(abs(logprob.logprob - out_lp[token].logprob))
    return sum(diffs) / len(diffs)


def _reference(vllm_runner, monkeypatch, model, kv_cache_dtype, **kwargs):
    """TRITON_MLA under the same engine settings (graph mode, runner, async,
    spec decode, parallelism), so only the attention backend differs."""
    key = (repr(model), kv_cache_dtype, tuple(sorted(kwargs.items())))
    if key not in _REFERENCE:
        _REFERENCE[key] = _run(
            vllm_runner,
            monkeypatch,
            backend="TRITON_MLA",
            model=model,
            kv_cache_dtype=kv_cache_dtype,
            **kwargs,
        )
    return _REFERENCE[key]


def _compare(vllm_runner, monkeypatch, model=DSV3_H16, kv_cache_dtype="auto", **kwargs):
    ref_kwargs = dict(kwargs)
    # TRITON_MLA has no DCP; the reference shards the same heads over TP only.
    ref_kwargs.pop("decode_context_parallel_size", None)
    ref = _reference(vllm_runner, monkeypatch, model, kv_cache_dtype, **ref_kwargs)
    out = _run(
        vllm_runner,
        monkeypatch,
        backend="ROCM_AITER_MLA",
        model=model,
        kv_cache_dtype=kv_cache_dtype,
        **kwargs,
    )
    # The comparison only has power if the reference is far from uniform
    # (a uniform distribution gives about -12 for these vocabularies).
    top = [max(v.logprob for v in lp.values()) for _, _, lps, _ in ref for lp in lps]
    assert sum(top) / len(top) > -6.0, "reference logits are nearly flat"
    diff = _mean_prompt_logprob_diff(ref, out)
    tol = model.tolerance[kv_cache_dtype]
    assert diff <= tol, f"mean prompt logprob diff {diff:.3f} > {tol}"
    # With an fp8 cache the fp8 asm prefill quantizes the decompressed q/k/v
    # at unit scale while TRITON_MLA prefills in bf16, so greedy tokens can
    # flip between near-tied candidates; the prompt-logprob bound still holds.
    if model.check_samples and kv_cache_dtype == "auto":
        check_logprobs_close(
            outputs_0_lst=ref,
            outputs_1_lst=out,
            name_0="TRITON_MLA",
            name_1="ROCM_AITER_MLA",
        )
    return diff


CORE = [
    # (cudagraph_mode, async_scheduling, model_runner_v2)
    ("NONE", False, False),
    ("PIECEWISE", False, False),
    ("FULL_DECODE_ONLY", True, False),
    ("FULL_AND_PIECEWISE", True, False),
    ("PIECEWISE", False, True),
    ("FULL_DECODE_ONLY", True, True),
]
FULL = [
    (mode, async_, v2)
    for mode in ("NONE", "PIECEWISE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")
    for async_ in (False, True)
    for v2 in (False, True)
    if (mode, async_, v2) not in CORE
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("cudagraph_mode, async_scheduling, model_runner_v2", CORE)
def test_engine_modes(
    vllm_runner,
    monkeypatch,
    model,
    cudagraph_mode,
    async_scheduling,
    model_runner_v2,
):
    _compare(
        vllm_runner,
        monkeypatch,
        model=model,
        cudagraph_mode=cudagraph_mode,
        async_scheduling=async_scheduling,
        model_runner_v2=model_runner_v2,
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("cudagraph_mode", ["NONE", "FULL_AND_PIECEWISE"])
def test_fp8_kv_cache(vllm_runner, monkeypatch, model, cudagraph_mode):
    _compare(
        vllm_runner,
        monkeypatch,
        model=model,
        kv_cache_dtype="fp8",
        cudagraph_mode=cudagraph_mode,
    )


@pytest.mark.parametrize(
    "cudagraph_mode, model_runner_v2",
    [("NONE", False), ("FULL_AND_PIECEWISE", False), ("FULL_AND_PIECEWISE", True)],
)
@pytest.mark.parametrize("model", [DSV3_H16, DSV3_H8], ids=["dsv3_h16", "dsv3_h8"])
def test_ngram_spec_decode(
    vllm_runner, monkeypatch, model, cudagraph_mode, model_runner_v2
):
    # Runner v1 has no async scheduling with n-gram drafts.
    _compare(
        vllm_runner,
        monkeypatch,
        model=model,
        speculative="ngram",
        cudagraph_mode=cudagraph_mode,
        model_runner_v2=model_runner_v2,
        async_scheduling=model_runner_v2,
    )


@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_mtp_spec_decode(
    vllm_runner, monkeypatch, num_speculative_tokens, kv_cache_dtype
):
    """The MTP draft layer and the verify block both run on the backend."""
    _compare(
        vllm_runner,
        monkeypatch,
        model=DSV3_MTP,
        kv_cache_dtype=kv_cache_dtype,
        speculative=("mtp", num_speculative_tokens),
        cudagraph_mode="FULL_AND_PIECEWISE",
        async_scheduling=True,
    )


def test_check_catches_a_wrong_softmax_scale(vllm_runner, monkeypatch):
    """Negative control: the comparison must fail when attention scores are
    scaled wrongly. DeepSeek-V3 multiplies the softmax scale by the YaRN
    mscale(factor, mscale_all_dim)^2 (1.87 here); mscale_all_dim 0 drops it."""
    rope = {
        "rope_type": "yarn",
        "rope_theta": 10000.0,
        "factor": 40.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 0.0,
        "original_max_position_embeddings": 4096,
    }
    wrong = replace(
        DSV3_H16, hf_overrides={**DSV3_H16.hf_overrides, "rope_parameters": rope}
    )
    ref = _reference(vllm_runner, monkeypatch, DSV3_H16, "auto")
    out = _run(vllm_runner, monkeypatch, backend="ROCM_AITER_MLA", model=wrong)
    assert _mean_prompt_logprob_diff(ref, out) > DSV3_H16.tolerance["auto"]


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
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("speculative", [None, "ngram"], ids=["decode", "ngram"])
def test_decode_context_parallel(vllm_runner, monkeypatch, kv_cache_dtype, speculative):
    """TP2 x DCP2: each rank holds half of every request's latent cache and
    the decode LSE merge completes attention (DCP verify routes with
    n-gram drafts). The reference is TRITON_MLA at TP2 without DCP."""
    _compare(
        vllm_runner,
        monkeypatch,
        model=DSV3_H16,
        kv_cache_dtype=kv_cache_dtype,
        tensor_parallel_size=2,
        decode_context_parallel_size=2,
        speculative=speculative,
        cudagraph_mode="FULL_AND_PIECEWISE",
    )
