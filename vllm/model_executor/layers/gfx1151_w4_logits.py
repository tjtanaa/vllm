# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1151 W4A16 logits head with exact top-K reranking.

Motivation
----------
Qwen3.8-27B has a 248,320 x 5,120 BF16 ``lm_head``: 2.54 GiB of weights. With
DFlash speculative decoding the head runs twice per decode step (target
verification and the draft's unary logits), so 5.08 GiB - about 22 ms of the
measured 140 ms step on gfx1151 - is spent reading one matrix, and that cost is
independent of the token batch.

This module keeps a derived int4 group-128 copy of the head (0.64 GiB) and
computes logits in two stages:

1. a W4A16 GEMM over the whole vocabulary (about 2.8 ms instead of 11.1 ms),
2. an exact BF16 recomputation of only the top-K rows, gathered from the
   original weight.

Rows outside the top-K are set to ``-inf``, so greedy decoding returns the same
token as the BF16 head whenever the true argmax is inside the approximate
top-K (K defaults to 256; the observed int4 logit error is ~1e-3 relative, so
the true argmax is not merely inside the top-K but far above its boundary).
Sampling and logprob consumers see a top-K truncated distribution, which is why
the path is opt-in (``VLLM_GFX1151_W4_LOGITS``) and limited to small token
batches where the weight read dominates.

The original BF16 weight is never modified; the derived copy is cached per
weight storage and built on first (eager) use.
"""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
    _on_gfx1151,
    pack_int4_exllama_shuffle,
)

GROUP_SIZE = 128
# Token-batch ceiling for the derived head. Measured on gfx1151 against the
# BF16 rocBLAS path (2 head calls per decode step, 248320x5120):
#   M=8  11.32 -> 3.85 ms (2.94x), M=16 11.48 -> 7.87 ms (1.46x),
#   M=32 14.24 -> 8.96 ms (1.59x), M=64 15.31 -> 26.74 ms (0.57x, rejected).
# The int4 GEMM wins while the weight read dominates and loses once the
# batch-64 Triton tile re-reads activations more than the BF16 GEMM does.
MAX_TOKENS = 32


class W4LogitsHead:
    """Derived int4 copy of one BF16 lm_head plus exact top-K reranking."""

    def __init__(self, weight: torch.Tensor, topk: int) -> None:
        vocab_size, hidden_size = weight.shape
        self.weight = weight
        self.topk = min(topk, vocab_size)
        groups = hidden_size // GROUP_SIZE
        flat = weight.detach().view(vocab_size, groups, GROUP_SIZE).float()
        amax = flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = amax / 7.0  # symmetric codes land in [-8, 7]
        codes = torch.clamp(torch.round(flat / scale) + 8.0, 0.0, 15.0)
        codes = codes.view(vocab_size, hidden_size).to(torch.uint8)
        self.scales = scale.squeeze(-1).to(weight.dtype).contiguous()
        self.packed = (
            pack_int4_exllama_shuffle(codes).contiguous().view(torch.int8)
        )
        # Quantization error of the derived head, recorded for evidence.
        deq = (
            (codes.view(vocab_size, groups, GROUP_SIZE).float() - 8.0)
            * scale
        ).view(vocab_size, hidden_size)
        self.rel_err = (
            (deq - weight.detach().float()).norm() / weight.detach().float().norm()
        ).item()

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        from vllm.utils.platform_utils import num_compute_units

        approx = torch.ops.vllm.rdna_hybrid_w4a16_apply(
            hidden,
            self.packed,
            self.scales,
            None,
            None,
            num_compute_units(),
            GROUP_SIZE,
        )
        tokens = hidden.shape[0]
        _, index = approx.topk(self.topk, dim=-1)
        rows = self.weight[index.reshape(-1)].view(tokens, self.topk, -1)
        exact = torch.bmm(
            hidden.view(tokens, 1, -1), rows.transpose(1, 2)
        ).view(tokens, self.topk)
        out = torch.full_like(approx, float("-inf"))
        out.scatter_(1, index, exact.to(out.dtype))
        return out


# Built lazily per weight storage; keyed by (data_ptr, vocab, hidden, topk).
_CACHE: dict[tuple[int, int, int, int], W4LogitsHead] = {}


def eligible(
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    embedding_bias: torch.Tensor | None,
) -> bool:
    """Contract check for the derived-head fast path."""
    if not envs.VLLM_GFX1151_W4_LOGITS or not _on_gfx1151():
        return False
    if embedding_bias is not None or hidden_states.dim() != 2:
        return False
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_cuda:
        return False
    if hidden_states.shape[0] > MAX_TOKENS:
        return False
    weight = getattr(lm_head, "weight", None)
    if (
        weight is None
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or getattr(lm_head, "tp_size", 1) != 1
        or not weight.is_contiguous()
        or weight.requires_grad
    ):
        return False
    vocab_size, hidden_size = weight.shape
    return (
        hidden_states.shape[1] == hidden_size
        and hidden_size % GROUP_SIZE == 0
        and hidden_size % 16 == 0
        and vocab_size % 8 == 0
        and hidden_states.is_contiguous()
        and not hidden_states.requires_grad
    )


def apply(
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    embedding_bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return reranked logits, or None when the caller must use the BF16 head."""
    if not eligible(lm_head, hidden_states, embedding_bias):
        return None
    weight = lm_head.weight
    topk = int(envs.VLLM_GFX1151_W4_LOGITS_TOPK)
    key = (weight.data_ptr(), weight.shape[0], weight.shape[1], topk)
    head = _CACHE.get(key)
    if head is None:
        if torch.cuda.is_current_stream_capturing():
            # Build the derived copy eagerly; never allocate it during capture.
            return None
        head = W4LogitsHead(weight, topk)
        _CACHE[key] = head
    return head.logits(hidden_states)
