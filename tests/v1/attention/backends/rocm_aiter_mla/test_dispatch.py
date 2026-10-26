# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501  (the docstring tables keep one row per line)
"""One test per ROCM_AITER_MLA dispatch path: the kernel that ran, and the
output against a float32 reference.

``MLAAttention.forward_impl`` sends the leading ``num_decode_tokens`` to
``forward_mqa`` (decode, latent space) and the rest to ``forward_mha``
(prefill, decompressed through ``kv_b_proj``). Rows below are
condition -> kernel, on gfx950 (MI350X/MI355X) with this image's AITER.

Prefill (``AiterMLAImpl.forward_mha``)

| # | Condition | Kernel(s) |
|---|---|---|
| P1 | fp8 KV, bf16 model, gfx950, qk/v head dims 192/128, no prefill in the batch has context | builder ``get_ps_metadata_v1``; ``fused_mla_prefill_qkv_fp8`` + ``mla_prefill_ps_asm_fwd`` + ``mla_reduce_v1`` |
| P2 | as P1 with other head dims (GLM-4.7-Flash: qk 256, v 256) | no asm kernel: parent path (P6) |
| P3 | as P1, heads not a multiple of 16 (12) | q/k/v replicate-padded to 16, asm prefill, output sliced |
| P4 | fp8 KV, some prefill in the batch has context | parent path: ``flash_attn_varlen_func`` (new tokens, LSE) + per chunk ``gather_and_maybe_dequant_cache`` -> ``kv_b_proj`` -> ``flash_attn_varlen_func`` + ``merge_attn_states`` |
| P5 | fp8 KV, fp16 model | asm prefill off: parent path |
| P6 | bf16 KV, no context | ``flash_attn_varlen_func`` only |
| P7 | bf16 KV, context longer than the chunked-prefill workspace | several context chunks, each gathered, then merged |
| P8 | DCP > 1 with context (fused fp8 context expansion or the parent CP path) | covered by ``tests/v1/attention/test_rocm_mla_dcp_context.py`` |

Decode (``AiterMLAImpl.forward_mqa``)

| # | Condition | Kernel(s) |
|---|---|---|
| Q1 | heads < 16 dividing 16, qlen 1, bf16 KV, ``VLLM_ROCM_AITER_MLA_ASM_PADDING=auto``, cache <= 2 GiB | ``mla_gluon`` (2-D), no persistent schedule |
| Q2 | heads < 16 not dividing 16 (12, 10, 5), qlen 1 | asm ``mla_decode_fwd`` on q tiled to 16 heads + persistent schedule |
| Q3 | heads < 16, ``..._ASM_PADDING=asm`` | asm on repeat-interleaved q |
| Q4 | heads < 16 not dividing 16, ``..._ASM_PADDING=gluon`` | ``mla_gluon`` |
| Q5 | heads < 16, fp8 KV | asm (Gluon's fp8 regime asserts batch 1) |
| Q6 | heads < 16, KV cache > 2 GiB per layer | asm (Gluon drops its bounds mask past 2 GiB) |
| Q7 | heads >= 16, qlen 1 | asm + persistent schedule (``get_mla_metadata_v1``) |
| Q8 | heads >= 16, causal verify qlen 2..k+1 | asm + persistent schedule with ``uni_seqlen_qo`` |
| Q9 | heads < 16, causal verify, bf16 KV | ``mla_gluon`` 4-D MTP entry |
| Q10 | heads < 16, causal verify, fp8 KV | padded asm, persistent |
| Q11 | heads < 16 padded, causal verify qlen > 4, bf16, ``..._ASM_PADDING=asm`` | asm without persistent schedule, zero-filled output |
| Q12 | non-causal uniform block (DSpark draft) | asm with ``causal=False`` + persistent schedule |
| Q13 | fp8 qlen-2 block on the 16-head kernel family (16, 48/80/112, padded < 16), causal or not | padded to a 4-query block (two leading copies of the first query), asm qseqlen-4 kernel, padded rows dropped: AITER's qseqlen-2 fp8 kernel returns wrong rows at short KV lengths, and has no non-causal entry |
| Q14 | 24 heads | asm at 24 heads with a native-H24 AITER build, else padded to 32 |
| Q15 | adaptive verification (varlen verify rows) | rows gathered to a uniform block, asm, rows scattered back |
| Q16 | DCP decode routes (plain, cprr, segmented, gfx942 Triton) | the plain route's LSE decode on the persistent schedule, against stale memory, here; the routes are covered by ``test_rocm_aiter_mla_dcp_cprr_numerics.py``, ``test_rocm_aiter_mla_dcp_verify_gfx942.py``, ``test_rocm_aiter_mla_mtp_split.py`` and the 2-GPU runtime case |

Other

| # | Condition | Behaviour |
|---|---|---|
| X1 | profiling run (``attn_metadata is None``) | zero output, no kernel, cache untouched |
| X2 | static fp8 output quant requested by the fusion pass | bf16 attention, quantized into the fp8 output |
"""

import pytest
import torch

from tests.v1.attention.backends.harness import make_paged_batch
from tests.v1.attention.backends.rocm_aiter_mla import mla_shapes as shapes
from tests.v1.attention.backends.rocm_aiter_mla.mla_harness import (
    ASM_DECODE,
    CP_GATHER,
    DEVICE,
    FA_PREFILL,
    FP8_PREFILL,
    FP8_Q_SCALE,
    FP8_REDUCE,
    FUSED_QKV,
    GATHER_DEQUANT,
    GLUON,
    MERGE,
    MLA_METADATA,
    PS_METADATA,
    MlaCase,
    blocks_for,
    watch_case_kernels,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.v1.attention.backends.mla import rocm_aiter_mla

pytestmark = pytest.mark.skipif(
    not on_gfx950(), reason="dispatch table is for gfx950 (Gluon, fp8 prefill)"
)


def _setup(spy, shape, seq_lens, **kw) -> MlaCase:
    case = MlaCase(shape, **kw).setup(blocks_for(seq_lens, kw.get("block_size", 16)))
    watch_case_kernels(spy, case)
    spy.reset()
    return case


def _run(case: MlaCase, query_lens, seq_lens, *, causal=True, check=True):
    batch = case.batch(query_lens, seq_lens, causal=causal)
    query = case.query(batch.num_tokens)
    metadata = case.build(batch)
    out = case.run(batch, query, metadata)
    if check:
        case.check(batch, query, out, metadata)
    return batch, query, metadata, out


def _asm_q(spy) -> torch.Tensor:
    (args,) = spy.args_of(ASM_DECODE)
    return args[0]


def _asm_kwargs(spy) -> dict:
    (kwargs,) = spy.kwargs_of(ASM_DECODE)
    return kwargs


def _set_padding_mode(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_ASM_PADDING", mode)


# --------------------------------------------------------------------------
# Prefill
# --------------------------------------------------------------------------

PREFILL_LENS = [300, 1031]


@pytest.mark.parametrize(
    "shape",
    [shapes.DSV3_TP8, shapes.DSV3_TP1, shapes.KIMI_K3_TP8],  # P1, P1, P3
    ids=lambda s: s.id,
)
def test_fp8_asm_prefill(spy, fp8_asm_prefill, shape):
    case = _setup(spy, shape, PREFILL_LENS, kv_cache_dtype="fp8")
    _run(case, PREFILL_LENS, PREFILL_LENS)
    names = spy.names()
    assert PS_METADATA in names
    assert names.count(FP8_PREFILL) == 1 and names.count(FP8_REDUCE) == 1
    assert FUSED_QKV in names
    assert FA_PREFILL not in names
    q = spy.args_of(FP8_PREFILL)[0][0]
    assert q.dtype == current_platform.fp8_dtype()
    assert q.shape[1] == -(-shape.num_heads // 16) * 16


def test_fp8_prefill_without_an_asm_kernel_takes_the_parent_path(spy):
    """P2: AITER only builds the qk 192 / v 128 asm prefill. GLM-4.7-Flash
    (256 / 256) used to load it anyway and abort in mla_reduce_v1."""
    shape = shapes.GLM47_FLASH_TP1
    case = _setup(spy, shape, PREFILL_LENS, kv_cache_dtype="fp8")
    assert not case.builder._fp8_prefill_enabled
    assert not case.layer.impl._fp8_prefill_enabled
    _run(case, PREFILL_LENS, PREFILL_LENS)
    names = spy.names()
    assert FP8_PREFILL not in names and PS_METADATA not in names
    assert names.count(FA_PREFILL) == 1


def test_fp8_prefill_with_context_takes_the_parent_path(spy):
    """P4: one extend in the batch sends every prefill to the varlen path."""
    case = _setup(spy, shapes.DSV3_TP8, [300, 700], kv_cache_dtype="fp8")
    _run(case, [300, 64], [300, 700])
    names = spy.names()
    assert FP8_PREFILL not in names and PS_METADATA not in names
    # New tokens once, the single context chunk once.
    assert names.count(FA_PREFILL) == 2
    assert GATHER_DEQUANT in names and CP_GATHER not in names
    assert MERGE in names


def test_fp8_cache_with_fp16_model_takes_the_parent_path(spy):
    """P5: the asm prefill writes bf16 through a raw pointer."""
    case = _setup(
        spy, shapes.DSV3_TP8, PREFILL_LENS, kv_cache_dtype="fp8", dtype=torch.float16
    )
    _run(case, PREFILL_LENS, PREFILL_LENS)
    names = spy.names()
    assert FP8_PREFILL not in names and PS_METADATA not in names
    assert names.count(FA_PREFILL) == 1


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "bfloat16"])
def test_unquantized_prefill_without_context(spy, kv_cache_dtype):
    """P6"""
    case = _setup(spy, shapes.DSV3_TP8, PREFILL_LENS, kv_cache_dtype=kv_cache_dtype)
    _run(case, PREFILL_LENS, PREFILL_LENS)
    names = spy.names()
    assert names.count(FA_PREFILL) == 1
    assert MERGE not in names and GATHER_DEQUANT not in names


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_context_longer_than_the_workspace_is_chunked(spy, kv_cache_dtype):
    """P7: max_model_len 1024 and 16 seqs give a 32k-token workspace floor of
    8 x max_model_len = 8192 tokens, so 3 x 4000 context tokens need two
    chunks."""
    seq_lens = [4000 + 32] * 3
    case = _setup(
        spy,
        shapes.DSV3_TP8,
        seq_lens,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=1024,
        max_num_seqs=16,
    )
    assert case.builder.chunked_prefill_workspace_size < 3 * 4000
    batch, _, metadata, _ = _run(case, [32] * 3, seq_lens)
    chunks = metadata.prefill.chunked_context.chunks
    assert len(chunks) >= 2
    names = spy.names()
    assert names.count(GATHER_DEQUANT) == len(chunks)
    assert names.count(FA_PREFILL) == 1 + len(chunks)


# --------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------

DECODE_LENS = [129, 1500, 4097]


@pytest.mark.parametrize(
    "shape", [shapes.DSV3_TP16, shapes.KIMI_LINEAR_TP8], ids=lambda s: s.id
)
def test_small_divisor_heads_use_gluon(spy, monkeypatch, shape):
    """Q1"""
    _set_padding_mode(monkeypatch, "auto")
    case = _setup(spy, shape, DECODE_LENS)
    _, _, metadata, _ = _run(case, [1] * 3, DECODE_LENS)
    assert metadata.decode.use_gluon_decode
    assert not metadata.decode.has_persistent_metadata
    names = spy.names()
    assert GLUON in names and ASM_DECODE not in names
    assert MLA_METADATA not in names


@pytest.mark.parametrize(
    "shape",
    [shapes.KIMI_K3_TP8, shapes.GLM47_FLASH_TP2, shapes.GLM47_FLASH_TP4],
    ids=lambda s: s.id,
)
def test_small_non_divisor_heads_pad_onto_asm(spy, monkeypatch, shape):
    """Q2"""
    _set_padding_mode(monkeypatch, "auto")
    case = _setup(spy, shape, DECODE_LENS)
    _, _, metadata, _ = _run(case, [1] * 3, DECODE_LENS)
    assert metadata.decode.has_persistent_metadata
    assert GLUON not in spy.names()
    q = _asm_q(spy)
    assert q.shape[1] == 16 and q.is_contiguous()
    # Tiled, so the real heads come first.
    assert torch.equal(q[:, shape.num_heads :], q[:, : 16 - shape.num_heads])
    assert "work_meta_data" in _asm_kwargs(spy)


def test_asm_padding_mode_forces_asm(spy, monkeypatch):
    """Q3: divisor heads are repeat-interleaved onto the 16-head kernel."""
    _set_padding_mode(monkeypatch, "asm")
    case = _setup(spy, shapes.DSV3_TP16, DECODE_LENS)
    _run(case, [1] * 3, DECODE_LENS)
    assert GLUON not in spy.names()
    q = _asm_q(spy)
    assert q.shape[1] == 16
    assert torch.equal(q[:, 0::2], q[:, 1::2])


def test_gluon_padding_mode_takes_non_divisor_heads(spy, monkeypatch):
    """Q4"""
    _set_padding_mode(monkeypatch, "gluon")
    case = _setup(spy, shapes.KIMI_K3_TP8, DECODE_LENS)
    _run(case, [1] * 3, DECODE_LENS)
    names = spy.names()
    assert GLUON in names and ASM_DECODE not in names


@pytest.mark.parametrize("mode", ["auto", "gluon"])
def test_small_heads_with_fp8_cache_use_asm(spy, monkeypatch, mode):
    """Q5"""
    _set_padding_mode(monkeypatch, mode)
    case = _setup(spy, shapes.DSV3_TP16, DECODE_LENS, kv_cache_dtype="fp8")
    _run(case, [1] * 3, DECODE_LENS)
    assert GLUON not in spy.names()
    assert _asm_q(spy).dtype == current_platform.fp8_dtype()


def test_kv_cache_past_2gib_refuses_gluon(spy, monkeypatch):
    """Q6: 18 KiB pages x 120k blocks is a 2.1 GiB layer cache."""
    _set_padding_mode(monkeypatch, "auto")
    case = MlaCase(shapes.DSV3_TP16).setup(120_000)
    assert case.builder._kv_cache_bytes > 2**31
    watch_case_kernels(spy, case)
    spy.reset()
    _run(case, [1] * 3, DECODE_LENS)
    names = spy.names()
    assert GLUON not in names and ASM_DECODE in names


@pytest.mark.parametrize(
    "shape",
    [shapes.DSV3_TP8, shapes.DSV3_TP4, shapes.KIMI_K3_TP2, shapes.DSV3_TP1],
    ids=lambda s: s.id,
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_aligned_heads_use_persistent_asm(spy, shape, kv_cache_dtype):
    """Q7"""
    case = _setup(spy, shape, DECODE_LENS, kv_cache_dtype=kv_cache_dtype)
    _, _, metadata, _ = _run(case, [1] * 3, DECODE_LENS)
    assert metadata.decode.has_persistent_metadata
    assert spy.names().count(MLA_METADATA) == 1
    q = _asm_q(spy)
    assert q.shape[1] == shape.num_heads
    kwargs = _asm_kwargs(spy)
    assert kwargs["work_meta_data"] is metadata.work_meta_data
    assert kwargs["causal"] is True
    if kv_cache_dtype == "fp8":
        assert q.dtype == current_platform.fp8_dtype()
        assert kwargs["q_scale"] is case.layer._q_scale
        assert kwargs["kv_scale"] is case.layer._k_scale


@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_causal_verify_on_aligned_heads(spy, num_speculative_tokens, kv_cache_dtype):
    """Q8"""
    q = num_speculative_tokens + 1
    case = _setup(
        spy,
        shapes.DSV3_TP8,
        DECODE_LENS,
        kv_cache_dtype=kv_cache_dtype,
        num_speculative_tokens=num_speculative_tokens,
    )
    _, _, metadata, _ = _run(case, [q] * 3, DECODE_LENS)
    # 16 heads at qlen 2 with fp8 run as a padded 4-query block (Q13).
    kernel_q = 4 if (q == 2 and kv_cache_dtype == "fp8") else q
    assert metadata.num_prefills == 0 and metadata.decode.max_qo_len == kernel_q
    (md_kwargs,) = spy.kwargs_of(MLA_METADATA)
    assert md_kwargs["uni_seqlen_qo"] == kernel_q
    assert md_kwargs["max_seqlen_qo"] == kernel_q


def test_small_heads_verify_uses_gluon_mtp(spy, monkeypatch):
    """Q9"""
    _set_padding_mode(monkeypatch, "auto")
    case = _setup(spy, shapes.DSV3_TP16, DECODE_LENS, num_speculative_tokens=3)
    _, _, metadata, _ = _run(case, [4] * 3, DECODE_LENS)
    assert metadata.decode.use_gluon_verify
    (call,) = spy.kwargs_of(GLUON)
    assert call["q_nope"].dim() == 4 and call["q_nope"].shape[:2] == (3, 4)
    assert ASM_DECODE not in spy.names()


def test_small_heads_verify_with_fp8_cache_uses_padded_asm(spy, monkeypatch):
    """Q10"""
    _set_padding_mode(monkeypatch, "auto")
    case = _setup(
        spy,
        shapes.KIMI_K3_TP8,
        DECODE_LENS,
        kv_cache_dtype="fp8",
        num_speculative_tokens=3,
    )
    _, _, metadata, _ = _run(case, [4] * 3, DECODE_LENS)
    assert metadata.decode.has_persistent_metadata
    assert GLUON not in spy.names()
    assert _asm_q(spy).shape[1] == 16


@pytest.mark.parametrize("num_speculative_tokens", [5, 7])
def test_padded_verify_past_qlen4_drops_the_schedule(
    spy, monkeypatch, num_speculative_tokens
):
    """Q11: the padded gqa=16 asm decode has no bf16 persistent kernel past
    qlen 4, so the builder skips the schedule and forward zero-fills."""
    _set_padding_mode(monkeypatch, "asm")
    q = num_speculative_tokens + 1
    case = _setup(
        spy,
        shapes.KIMI_K3_TP8,
        DECODE_LENS,
        num_speculative_tokens=num_speculative_tokens,
    )
    _, _, metadata, _ = _run(case, [q] * 3, DECODE_LENS)
    assert not metadata.decode.has_persistent_metadata
    assert MLA_METADATA not in spy.names()
    assert "work_meta_data" not in _asm_kwargs(spy)


@pytest.mark.parametrize(
    "shape", [shapes.DSV3_TP8, shapes.KIMI_K3_TP8, shapes.DSV3_TP16], ids=lambda s: s.id
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_non_causal_block_uses_asm(spy, monkeypatch, shape, kv_cache_dtype):
    """Q12: Gluon masks causally, so a non-causal block is always asm."""
    _set_padding_mode(monkeypatch, "auto")
    case = _setup(
        spy,
        shape,
        DECODE_LENS,
        kv_cache_dtype=kv_cache_dtype,
        num_speculative_tokens=3,
        non_causal=True,
    )
    _, _, metadata, _ = _run(case, [4] * 3, DECODE_LENS, causal=False)
    assert metadata.num_prefills == 0
    assert GLUON not in spy.names()
    assert _asm_kwargs(spy)["causal"] is False
    assert metadata.decode.has_persistent_metadata
    (md_kwargs,) = spy.kwargs_of(MLA_METADATA)
    assert md_kwargs is not None


# KV lengths just past a tile edge, where AITER's fp8 qseqlen-2 kernel
# returned wrong rows (head 0 of the second query, up to 0.9 abs).
SHORT_LENS = [[9, 10, 11, 12], [33, 900, 129, 5000], [17, 65, 257, 1025]]


@pytest.mark.parametrize(
    "shape",
    [shapes.DSV3_TP8, shapes.DSV3_TP16, shapes.KIMI_K3_TP8, shapes.KIMI_K3_TP2],
    ids=lambda s: s.id,
)
@pytest.mark.parametrize("seq_lens", SHORT_LENS, ids=lambda s: f"s{s[0]}")
def test_fp8_two_token_verify_is_padded(spy, monkeypatch, shape, seq_lens):
    """Q13, causal: DeepSeek-V3 TP8 / Kimi-K3 with MTP-1 and an fp8 cache."""
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_ASM_PADDING", "asm")
    case = _setup(
        spy, shape, [5000] * 4, kv_cache_dtype="fp8", num_speculative_tokens=1
    )
    for seed in range(4):
        spy.reset()
        batch = make_paged_batch(
            [2] * 4, seq_lens, 16, DEVICE, num_blocks=case.num_blocks, seed=seed
        )
        query = case.query(batch.num_tokens)
        metadata = case.build(batch)
        assert metadata.decode.fp8_qlen2_padded
        assert metadata.decode.max_qo_len == 4
        case.check(batch, query, case.run(batch, query, metadata), metadata)
        assert _asm_q(spy).shape[0] == 4 * 4
        (md_kwargs,) = spy.kwargs_of(MLA_METADATA)
        assert md_kwargs["max_seqlen_qo"] == 4


@pytest.mark.parametrize(
    "shape", [shapes.DSV3_TP4, shapes.DSV3_TP1], ids=lambda s: s.id
)
def test_fp8_two_token_verify_native_heads_are_not_padded(spy, shape):
    case = _setup(
        spy, shape, DECODE_LENS, kv_cache_dtype="fp8", num_speculative_tokens=1
    )
    _, _, metadata, _ = _run(case, [2] * 3, DECODE_LENS)
    assert not metadata.decode.fp8_qlen2_padded
    assert _asm_q(spy).shape[0] == 2 * 3


def test_non_causal_fp8_two_token_block_is_padded(spy):
    """Q13, non-causal: AITER has no non-causal fp8 qseqlen-2 entry at 16
    heads (it used to abort, so the builder raised); the padded 4-query block
    takes the non-causal qseqlen-4 kernel."""
    case = _setup(
        spy,
        shapes.DSV3_TP8,
        DECODE_LENS,
        kv_cache_dtype="fp8",
        num_speculative_tokens=1,
        non_causal=True,
    )
    _, _, metadata, _ = _run(case, [2] * 3, DECODE_LENS, causal=False)
    assert metadata.decode.fp8_qlen2_padded
    assert _asm_kwargs(spy)["causal"] is False


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_24_heads(spy, kv_cache_dtype):
    """Q14"""
    case = _setup(spy, shapes.KIMI_K3_TP4, DECODE_LENS, kv_cache_dtype=kv_cache_dtype)
    _run(case, [1] * 3, DECODE_LENS)
    native = rocm_aiter_mla._aiter_mla_native_h24_supported()
    assert _asm_q(spy).shape[1] == (24 if native else 32)


def test_adaptive_verification_varlen_rows(spy):
    """Q15: device-trimmed verify blocks of 1..k+1 tokens."""
    seq_lens = [200, 300, 400, 500]
    case = _setup(
        spy,
        shapes.DSV3_TP8,
        seq_lens,
        num_speculative_tokens=3,
        adaptive_verification=True,
    )
    _, _, metadata, _ = _run(case, [4, 2, 1, 3], seq_lens)
    assert metadata.decode.varlen_q_rows is not None
    assert metadata.num_prefills == 0
    # The kernel sees the uniform 4-token block of every request.
    assert _asm_q(spy).shape[0] == 4 * 4


# --------------------------------------------------------------------------
# Other
# --------------------------------------------------------------------------


def test_profiling_run_touches_nothing(spy):
    """X1"""
    case = _setup(spy, shapes.DSV3_TP8, [64])
    before = case.kv_cache.clone()
    n = 16
    q = case.query(n)
    kv_c, k_pe = case.split_latent(case.latent_all[:n])
    out = torch.full((n, 16 * 128), float("nan"), dtype=case.dtype, device=q.device)
    case.layer.forward_impl(q, kv_c, k_pe, case.kv_cache, None, out)
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.equal(case.kv_cache, before)
    assert not set(spy.names()) & {ASM_DECODE, GLUON, FA_PREFILL, FP8_PREFILL}


@pytest.mark.parametrize("phase", ["decode", "prefill", "mixed"])
def test_static_fp8_output_quant(spy, phase):
    """X2: the fusion pass hands forward_impl an fp8 output and a scale."""
    seq_lens = [300, 400, 500]
    query_lens = {
        "decode": [1, 1, 1],
        "prefill": seq_lens,
        "mixed": [1, 64, 500],
    }[phase]
    if phase == "mixed":
        seq_lens = [300, 400, 500]
    case = _setup(spy, shapes.DSV3_TP8, seq_lens)
    batch = case.batch(query_lens, seq_lens)
    query = case.query(batch.num_tokens)
    metadata = case.build(batch)
    expected = case.reference(batch, query, metadata.num_decode_tokens)
    n = batch.num_tokens
    kv_c, k_pe = case.split_latent(case.latent_all[batch.query_slots])
    fp8 = current_platform.fp8_dtype()
    out = torch.empty(n, 16 * 128, dtype=fp8, device=query.device)
    scale = torch.tensor(0.05, dtype=torch.float32, device=query.device)
    case.layer.forward_impl(
        query, kv_c, k_pe, case.kv_cache, metadata, out, output_scale=scale
    )
    actual = out.float().view(n, 16, 128) * scale
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.07)


@pytest.mark.parametrize(
    "shape", [shapes.DSV3_TP8, shapes.DSV3_TP4, shapes.DSV3_TP1], ids=lambda s: s.id
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_lse_decode_ignores_stale_memory(spy, shape, kv_cache_dtype):
    """Q16 plain route: DCP decode calls aiter.mla.mla_decode_fwd with
    return_lse=True on the builder's persistent schedule. Without the
    schedule, AITER's fp8 16-head decode reads uninitialized scratch for KV
    lengths 65-128; with it, output and natural-log LSE must match the
    reference even when the allocator hands back NaN-filled memory."""
    case = _setup(spy, shape, [300] * 4, kv_cache_dtype=kv_cache_dtype)
    fp8 = current_platform.fp8_dtype()
    lora = shape.kv_lora_rank
    for seq_lens in ([65, 80, 100, 128], [64, 129, 300, 17]):
        batch = make_paged_batch(
            [1] * 4, seq_lens, 16, DEVICE, num_blocks=case.num_blocks, seed=1
        )
        md = case.build(batch)
        assert md.decode.has_persistent_metadata
        q = torch.randn(4, shape.num_heads, lora + 64, device=DEVICE)
        kv = case.kv_cache
        if case.quantized:
            q_in = (q / FP8_Q_SCALE).to(fp8)
            q_ref = q_in.float() * FP8_Q_SCALE
            kv = kv.view(fp8)
        else:
            q_in = q.to(case.dtype)
            q_ref = q_in.float()
        # Poison the caching allocator so stale scratch is NaN, not zero.
        del_me = torch.full((1 << 28,), float("nan"), device=DEVICE)
        del del_me
        out = torch.empty(4, shape.num_heads, lora, dtype=case.dtype, device=DEVICE)
        _, lse = rocm_aiter_mla._get_aiter_mla_decode()(
            q_in,
            kv.unsqueeze(2).view(-1, 1, 1, lora + 64),
            out,
            md.decode.qo_indptr,
            md.decode.paged_kv_indptr,
            md.decode.paged_kv_indices,
            md.decode.paged_kv_last_page_len,
            1,
            sm_scale=case.scale,
            return_lse=True,
            causal=True,
            q_scale=case.layer._q_scale,
            kv_scale=case.layer._k_scale,
            work_meta_data=md.work_meta_data,
            work_indptr=md.work_indptr,
            work_info_set=md.work_info_set,
            reduce_indptr=md.reduce_indptr,
            reduce_final_map=md.reduce_final_map,
            reduce_partial_map=md.reduce_partial_map,
        )
        latent = case.latent_all.float()
        for r in range(4):
            k = latent[batch.slots[r]]
            logits = torch.einsum("hd,kd->hk", q_ref[r], k) * case.scale
            torch.testing.assert_close(
                lse[r].float().flatten()[: shape.num_heads],
                torch.logsumexp(logits, -1),
                atol=1e-2,
                rtol=1e-3,
            )
            torch.testing.assert_close(
                out[r].float(),
                torch.softmax(logits, -1) @ k[:, :lora],
                atol=6e-2,
                rtol=5e-2,
            )
