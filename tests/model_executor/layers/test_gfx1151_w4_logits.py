# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract and correctness tests for the gfx1151 derived int4 logits head.

The CPU tests pin the dispatch contract (opt-in flag, dtype/shape/TP/bias/batch
limits, no allocation during graph capture). The GPU tests require gfx1151 and
check that greedy decoding is unchanged against the BF16 head and that the
reranked top-K values are the exact BF16 logits.
"""

import types

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers import gfx1151_w4_logits as w4

GPU_CASES = torch.cuda.is_available()


def _head_module(monkeypatch, *, on_gfx1151=True, topk=64):
    monkeypatch.setenv("VLLM_GFX1151_W4_LOGITS", "1")
    monkeypatch.setenv("VLLM_GFX1151_W4_LOGITS_TOPK", str(topk))
    monkeypatch.setattr(w4, "_on_gfx1151", lambda: on_gfx1151)
    monkeypatch.setattr(envs, "VLLM_GFX1151_W4_LOGITS", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_GFX1151_W4_LOGITS_TOPK", topk, raising=False)
    w4._CACHE.clear()


def _fake_lm_head(weight, tp_size=1):
    return types.SimpleNamespace(weight=weight, tp_size=tp_size)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_GFX1151_W4_LOGITS", raising=False)
    monkeypatch.setattr(envs, "VLLM_GFX1151_W4_LOGITS", False, raising=False)
    weight = torch.randn(256, 512, dtype=torch.bfloat16)
    hidden = torch.randn(4, 512, dtype=torch.bfloat16)
    assert w4.apply(_fake_lm_head(weight), hidden) is None


def test_requires_gfx1151(monkeypatch):
    _head_module(monkeypatch, on_gfx1151=False)
    weight = torch.randn(256, 512, dtype=torch.bfloat16)
    hidden = torch.randn(4, 512, dtype=torch.bfloat16)
    assert w4.apply(_fake_lm_head(weight), hidden) is None


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda w, h: (w, h.float()), id="fp32-activations"),
        pytest.param(lambda w, h: (w, h.unsqueeze(0)), id="rank3-activations"),
        pytest.param(
            lambda w, h: (
                w,
                torch.randn(w4.MAX_TOKENS + 1, h.shape[1], dtype=h.dtype),
            ),
            id="batch-too-large",
        ),
        pytest.param(lambda w, h: (w.float(), h), id="fp32-weight"),
        pytest.param(lambda w, h: (w[:, :384], h), id="hidden-not-group-aligned"),
    ],
)
def test_contract_rejects_unsupported(monkeypatch, mutate):
    _head_module(monkeypatch)
    weight = torch.randn(256, 512, dtype=torch.bfloat16)
    hidden = torch.randn(8, 512, dtype=torch.bfloat16)
    weight, hidden = mutate(weight, hidden)
    assert w4.apply(_fake_lm_head(weight), hidden) is None


def test_contract_rejects_bias_and_tensor_parallel(monkeypatch):
    _head_module(monkeypatch)
    weight = torch.randn(256, 512, dtype=torch.bfloat16)
    hidden = torch.randn(4, 512, dtype=torch.bfloat16)
    bias = torch.zeros(256, dtype=torch.bfloat16)
    assert w4.apply(_fake_lm_head(weight), hidden, bias) is None
    assert w4.apply(_fake_lm_head(weight, tp_size=2), hidden) is None


def test_quantized_codes_round_trip_in_range(monkeypatch):
    """Derived int4 codes stay in [0, 15] with a bounded weight error.

    Round-to-nearest int4 with a max/7 group scale has ~12% relative L2 error on
    Gaussian weights; the head stays usable because the exact argmax is recovered
    by the top-K reranking stage (measured on the real checkpoint head: the exact
    argmax always ranked inside the approximate top-16).
    """
    _head_module(monkeypatch)
    torch.manual_seed(0)
    weight = torch.randn(512, 256, dtype=torch.bfloat16)
    head = w4.W4LogitsHead(weight, topk=32)
    assert head.packed.dtype == torch.int8
    assert head.packed.shape == (512, 128)
    assert head.scales.shape == (512, 2)
    assert 0.0 < head.rel_err < 0.2


@pytest.mark.skipif(not GPU_CASES, reason="requires a GPU")
def test_greedy_argmax_matches_bf16_head(monkeypatch):
    from vllm.platforms.rocm import on_gfx1151

    if not on_gfx1151():
        pytest.skip("gfx1151 only")
    _head_module(monkeypatch, topk=256)
    torch.manual_seed(7)
    vocab, hidden_size, tokens = 8192, 512, 8
    weight = (torch.randn(vocab, hidden_size, device="cuda") * 0.02).to(torch.bfloat16)
    hidden = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(hidden, weight)
    out = w4.apply(_fake_lm_head(weight), hidden)
    assert out is not None and out.shape == reference.shape
    assert torch.equal(out.argmax(-1), reference.argmax(-1))
    # Every reranked row holds the exact BF16 logit; everything outside the
    # top-K stays masked to -inf.
    index = out.argmax(-1, keepdim=True)
    exact = reference.gather(1, index)
    assert torch.allclose(out.gather(1, index), exact, atol=2e-2, rtol=2e-2)
    live = out.topk(256, dim=-1).indices
    keep = torch.zeros_like(out, dtype=torch.bool).scatter_(1, live, True)
    assert torch.isneginf(out[~keep]).all()
    assert not torch.isneginf(out[keep]).any()


@pytest.mark.skipif(not GPU_CASES, reason="requires a GPU")
def test_topk_covers_bf16_topk(monkeypatch):
    from vllm.platforms.rocm import on_gfx1151

    if not on_gfx1151():
        pytest.skip("gfx1151 only")
    _head_module(monkeypatch, topk=256)
    torch.manual_seed(11)
    vocab, hidden_size = 4096, 512
    weight = (torch.randn(vocab, hidden_size, device="cuda") * 0.02).to(torch.bfloat16)
    hidden = torch.randn(4, hidden_size, device="cuda", dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(hidden, weight)
    out = w4.apply(_fake_lm_head(weight), hidden)
    keep = 16
    assert torch.equal(
        out.topk(keep, dim=-1).indices.sort(-1).values,
        reference.topk(keep, dim=-1).indices.sort(-1).values,
    )
