# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive ROCM_AITER_MLA the way the model runner does, one layer at a time.

``MlaCase`` owns a VllmConfig shaped like one model's MLA layer at one TP
rank, a real ``MLAAttention`` layer bound to ``AiterMLABackend`` (with its
``kv_b_proj``, absorbed ``W_UK_T``/``W_UV`` and prefill backend), the metadata
builder, and a latent KV cache allocated through the runner's view helpers.
``run`` writes the batch's latent through ``update_kv_cache`` and calls
``forward_impl``, so prefill tokens take ``forward_mha`` and decode tokens
take the layer's query absorption, fp8 query quantization, ``forward_mqa``
and ``W_UV`` up-projection. ``reference`` recomputes every token in float32
from the logical (dequantized) latent.
"""

import importlib
from dataclasses import dataclass, field

import torch

from tests.v1.attention.backends.harness import (
    KernelSpy,
    PagedBatch,
    make_paged_batch,
    ref_attention,
)
from tests.v1.attention.model_shapes import MLAShape
from tests.v1.attention.utils import dense_kv_cache_views
from vllm import _custom_ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention import mla_attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import rocm_aiter_mla
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    AiterMLABackend,
    AiterMLAMetadata,
    AiterMLAMetadataBuilder,
)
from vllm.v1.kv_cache_interface import is_quantized_kv_cache
from vllm.v1.kv_cache_layout import KVCacheLayout

DEVICE = torch.device("cuda")
# The config only supplies the architecture; every MLA dimension is overridden.
MODEL = "deepseek-ai/DeepSeek-R1"
LAYER = "model.layers.0.self_attn.attn"

# Non-unit so a missing scale shows up as a numerical error.
FP8_K_SCALE = 0.5
FP8_Q_SCALE = 0.25

FP8_ATOL = 6e-2
FP8_RTOL = 1e-1
FP8_MAX_BAD_FRACTION = 5e-3
FP8_MAX_ABS = 0.3

# Kernel labels used by the dispatch tests.
ASM_DECODE = "rocm_aiter_ops.mla_decode_fwd"
DCP_DECODE = "aiter.mla.mla_decode_fwd"
GLUON = "mla_gluon"
SEGMENTED = "segmented_mla_decode"
TRITON_DECODE = "decode_attention_fwd"
SEGMENT_MERGE = "merge_mla_segments_triton"
MLA_METADATA = "get_mla_metadata_v1"
PS_METADATA = "get_ps_metadata_v1"
FP8_PREFILL = "mla_prefill_ps_asm_fwd"
FP8_REDUCE = "mla_reduce_v1"
FUSED_QKV = "fused_mla_prefill_qkv_fp8"
FA_PREFILL = "prefill.flash_attn_varlen_func"
GATHER_DEQUANT = "gather_and_maybe_dequant_cache"
CP_GATHER = "cp_gather_cache"
MERGE = "merge_attn_states"
WRITE = "concat_and_cache_mla"


def tolerance(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    return {"atol": 2e-2, "rtol": 2e-2}


def fp8_round(x: torch.Tensor, scale: float) -> torch.Tensor:
    """What a tensor quantized to fp8 with ``scale`` dequantizes back to."""
    fp8 = current_platform.fp8_dtype()
    return (x.float() / scale).to(fp8).float() * scale


def make_config(
    shape: MLAShape,
    *,
    dtype: torch.dtype = torch.bfloat16,
    kv_cache_dtype: str = "auto",
    block_size: int = 16,
    num_blocks: int = 1024,
    max_model_len: int = 16384,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 16384,
    num_speculative_tokens: int = 0,
    adaptive_verification: bool = False,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
) -> VllmConfig:
    """A VllmConfig whose MLA dims are one model's at one TP rank."""
    model_config = ModelConfig(
        model=MODEL,
        dtype=dtype,
        seed=0,
        max_model_len=max_model_len,
        hf_overrides={
            "num_attention_heads": shape.num_heads,
            "num_key_value_heads": shape.num_heads,
            "kv_lora_rank": shape.kv_lora_rank,
            "qk_nope_head_dim": shape.qk_nope_head_dim,
            "qk_rope_head_dim": shape.qk_rope_head_dim,
            "v_head_dim": shape.v_head_dim,
            "max_position_embeddings": max_model_len,
        },
    )
    cache_config = CacheConfig(block_size=block_size, cache_dtype=kv_cache_dtype)
    cache_config.num_gpu_blocks = num_blocks
    cache_config.num_cpu_blocks = 0
    cache_config.kv_cache_layout = "LBHNC"
    config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(cudagraph_mode=cudagraph_mode),
    )
    # VllmConfig post-init may rewrite the mode (e.g. no compilation); the
    # builder reads it to decide on persistent buffers, so pin it back.
    config.compilation_config.cudagraph_mode = cudagraph_mode
    if num_speculative_tokens:
        config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=num_speculative_tokens
        )
        config.speculative_config.enable_adaptive_verification = adaptive_verification
    assert model_config.get_num_attention_heads(config.parallel_config) == (
        shape.num_heads
    )
    return config


def _cached_entry_spy(spy: KernelSpy, getter_name: str, label: str) -> None:
    """Spy on a kernel the backend fetches through a cached getter."""
    getter = getattr(rocm_aiter_mla, getter_name)
    try:
        kernel = getter()
    except Exception:  # noqa: BLE001  # not in this AITER build
        return

    def wrapper(*args, **kwargs):
        spy.calls.append((label, args, kwargs))
        return kernel(*args, **kwargs)

    spy.monkeypatch.setattr(rocm_aiter_mla, getter_name, lambda: wrapper)


def watch_all_kernels(spy: KernelSpy) -> None:
    """Spy on every kernel the layer can dispatch to, outside the impl."""
    import aiter

    spy.watch(rocm_aiter_ops, "mla_decode_fwd", ASM_DECODE)
    _cached_entry_spy(spy, "_get_aiter_mla_decode", DCP_DECODE)
    _cached_entry_spy(spy, "_get_mla_gluon", GLUON)
    _cached_entry_spy(spy, "_get_segmented_mla_decode", SEGMENTED)
    spy.watch(rocm_aiter_mla, "decode_attention_fwd", TRITON_DECODE)
    spy.watch(rocm_aiter_mla, "merge_mla_segments_triton", SEGMENT_MERGE)
    # The builder imports these from the aiter package at call time.
    spy.watch(aiter, "get_mla_metadata_v1", MLA_METADATA)
    if hasattr(aiter, "get_ps_metadata_v1"):
        spy.watch(aiter, "get_ps_metadata_v1", PS_METADATA)
    try:
        fused = importlib.import_module(
            "aiter.ops.triton.fusions.fused_mla_prefill_qkv_fp8"
        )
        spy.watch(fused, "fused_mla_prefill_qkv_fp8", FUSED_QKV)
    except ImportError:
        pass
    spy.watch(_custom_ops, "gather_and_maybe_dequant_cache", GATHER_DEQUANT)
    spy.watch(_custom_ops, "cp_gather_cache", CP_GATHER)
    spy.watch(_custom_ops, "concat_and_cache_mla", WRITE)
    spy.watch(mla_attention, "merge_attn_states", MERGE)


def watch_case_kernels(spy: KernelSpy, case: "MlaCase") -> None:
    """Spy on the kernels the impl and prefill backend hold as attributes."""
    impl = case.layer.impl
    if getattr(impl, "_fp8_prefill_enabled", False):
        spy.watch(impl, "_mla_prefill_ps_asm_fwd", FP8_PREFILL)
        spy.watch(impl, "_mla_reduce_v1", FP8_REDUCE)
    # Prefill metadata carries the builder's clone of the layer's backend.
    spy.watch(case.builder._prefill_backend, "flash_attn_varlen_func", FA_PREFILL)


@dataclass
class MlaCase:
    shape: MLAShape
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: str = "auto"
    block_size: int = 16
    max_model_len: int = 16384
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 16384
    num_speculative_tokens: int = 0
    adaptive_verification: bool = False
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE
    non_causal: bool = False
    num_blocks: int = 0
    seed: int = 0

    config: VllmConfig = field(init=False)
    layer: MLAAttention = field(init=False)
    builder: AiterMLAMetadataBuilder = field(init=False)
    kv_cache: torch.Tensor = field(init=False)
    # Logical latent per cache slot: [num_slots, kv_lora_rank + rope].
    latent_all: torch.Tensor = field(init=False)

    @property
    def scale(self) -> float:
        return (self.shape.qk_nope_head_dim + self.shape.qk_rope_head_dim) ** -0.5

    @property
    def quantized(self) -> bool:
        return is_quantized_kv_cache(self.kv_cache_dtype)

    @property
    def impl(self):
        return self.layer.impl

    def setup(self, num_blocks: int) -> "MlaCase":
        torch.manual_seed(self.seed)
        self.num_blocks = num_blocks
        s = self.shape
        self.config = make_config(
            s,
            dtype=self.dtype,
            kv_cache_dtype=self.kv_cache_dtype,
            block_size=self.block_size,
            num_blocks=num_blocks,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            num_speculative_tokens=self.num_speculative_tokens,
            adaptive_verification=self.adaptive_verification,
            cudagraph_mode=self.cudagraph_mode,
        )
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            # The model loader builds layers under the target device.
            with set_current_vllm_config(self.config), torch.device(DEVICE):
                # The model's per-rank kv_b_proj; TP is folded into the shape.
                kv_b_proj = ColumnParallelLinear(
                    s.kv_lora_rank,
                    s.num_heads * (s.qk_nope_head_dim + s.v_head_dim),
                    bias=False,
                    params_dtype=self.dtype,
                    prefix="model.layers.0.self_attn.kv_b_proj",
                    disable_tp=True,
                )
                # Unit-variance K_nope and V per head for unit-variance latent.
                kv_b_proj.weight.data.normal_(std=s.kv_lora_rank**-0.5)
                self.layer = MLAAttention(
                    num_heads=s.num_heads,
                    scale=self.scale,
                    qk_nope_head_dim=s.qk_nope_head_dim,
                    qk_rope_head_dim=s.qk_rope_head_dim,
                    v_head_dim=s.v_head_dim,
                    q_lora_rank=s.q_lora_rank,
                    kv_lora_rank=s.kv_lora_rank,
                    kv_b_proj=kv_b_proj,
                    cache_config=self.config.cache_config,
                    prefix=LAYER,
                    attn_backend=AiterMLABackend,
                    non_causal_multi_token_decode=self.non_causal,
                )
                self.layer.process_weights_after_loading(self.dtype)
                spec = AiterMLABackend.customize_spec(
                    self.layer.get_kv_cache_spec(self.config)
                )
                self.builder = AiterMLAMetadataBuilder(
                    spec, [LAYER], self.config, DEVICE
                )
        finally:
            torch.set_default_dtype(prev_dtype)
        self.spec = spec
        raw = torch.zeros(
            num_blocks * spec.page_size_bytes, dtype=torch.int8, device=DEVICE
        )
        (view,) = dense_kv_cache_views(
            raw, spec, num_blocks, num_layers=1, layout=KVCacheLayout.LBHNC
        )
        self.layer.bind_kv_cache(view)
        self.kv_cache = self.layer.kv_cache
        if self.quantized:
            self.layer._k_scale.fill_(FP8_K_SCALE)
            self.layer._q_scale.fill_(FP8_Q_SCALE)
            self.layer._k_scale_float = FP8_K_SCALE
            self.layer._q_scale_float = FP8_Q_SCALE
        self._fill_cache()
        return self

    @property
    def num_slots(self) -> int:
        return self.num_blocks * self.block_size

    def _fill_cache(self) -> None:
        s = self.shape
        latent = torch.randn(
            self.num_slots,
            s.kv_lora_rank + s.qk_rope_head_dim,
            dtype=self.dtype,
            device=DEVICE,
        )
        if self.quantized:
            # Values the cache stores exactly, so the reference and the kernel
            # see the same latent and only accumulation error remains.
            latent = fp8_round(latent, FP8_K_SCALE).to(self.dtype)
        self.latent_all = latent
        kv_c, k_pe = self.split_latent(latent)
        self.layer.update_kv_cache(
            kv_c,
            k_pe,
            self.kv_cache,
            torch.arange(self.num_slots, dtype=torch.int64, device=DEVICE),
            None,
            self.kv_cache_dtype,
            self.layer._k_scale,
        )

    def split_latent(self, latent: torch.Tensor):
        kv_c, k_pe = latent.split(
            [self.shape.kv_lora_rank, self.shape.qk_rope_head_dim], dim=-1
        )
        return kv_c.contiguous(), k_pe.unsqueeze(1).contiguous()

    def batch(
        self,
        query_lens: list[int],
        seq_lens: list[int],
        *,
        num_padded_tokens: int | None = None,
        causal: bool = True,
    ) -> PagedBatch:
        return make_paged_batch(
            query_lens,
            seq_lens,
            self.block_size,
            DEVICE,
            num_blocks=self.num_blocks,
            num_padded_tokens=num_padded_tokens,
            causal=causal,
            seed=self.seed,
        )

    def build(self, batch: PagedBatch) -> AiterMLAMetadata:
        return self.builder.build(0, batch.common)

    def query(self, num_tokens: int) -> torch.Tensor:
        s = self.shape
        return torch.randn(
            num_tokens,
            s.num_heads,
            s.qk_nope_head_dim + s.qk_rope_head_dim,
            dtype=self.dtype,
            device=DEVICE,
        )

    def run(
        self,
        batch: PagedBatch,
        query: torch.Tensor,
        metadata: AiterMLAMetadata | None = None,
    ) -> torch.Tensor:
        """Forward the layer; returns [num_tokens, H, v_head_dim]."""
        metadata = metadata if metadata is not None else self.build(batch)
        n = query.shape[0]
        s = self.shape
        latent = torch.zeros(
            n, s.kv_lora_rank + s.qk_rope_head_dim, dtype=self.dtype, device=DEVICE
        )
        latent[: batch.num_tokens] = self.latent_all[batch.query_slots]
        kv_c, k_pe = self.split_latent(latent)
        self.layer.update_kv_cache(
            kv_c,
            k_pe,
            self.kv_cache,
            batch.common.slot_mapping,
            metadata,
            self.kv_cache_dtype,
            self.layer._k_scale,
        )
        output = torch.full(
            (n, s.num_heads * s.v_head_dim),
            float("nan"),
            dtype=self.dtype,
            device=DEVICE,
        )
        self.layer.forward_impl(query, kv_c, k_pe, self.kv_cache, metadata, output)
        return output.view(n, s.num_heads, s.v_head_dim)

    def weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """W_UK [H, P, L] and W_UV [H, V, L] in float32."""
        s = self.shape
        w = self.layer.kv_b_proj.weight.float().view(
            s.num_heads, s.qk_nope_head_dim + s.v_head_dim, s.kv_lora_rank
        )
        return w[:, : s.qk_nope_head_dim], w[:, s.qk_nope_head_dim :]

    def reference(
        self,
        batch: PagedBatch,
        query: torch.Tensor,
        num_decode_tokens: int = 0,
    ) -> torch.Tensor:
        """Float32 MLA over the logical latent, in absorbed form.

        Absorbing W_UK into the query and W_UV into the output is exact in
        real arithmetic, so one reference covers the prefill (decompressed)
        and decode (latent) kernels. Decode queries are fp8-rounded when the
        layer quantizes them, as the kernel sees them.
        """
        s = self.shape
        w_uk, w_uv = self.weights()
        q = query[: batch.num_tokens].float()
        q_nope, q_pe = q.split([s.qk_nope_head_dim, s.qk_rope_head_dim], dim=-1)
        q_lat = torch.cat([torch.einsum("thp,hpl->thl", q_nope, w_uk), q_pe], -1)
        if self.quantized and num_decode_tokens:
            q_lat[:num_decode_tokens] = fp8_round(
                q_lat[:num_decode_tokens], FP8_Q_SCALE
            )
        latent = self.latent_all.float().unsqueeze(1)
        # V is the latent without its rope part; ref_attention returns the
        # query's width, so zero those columns and drop them afterwards.
        value = latent.clone()
        value[..., s.kv_lora_rank :] = 0
        out_lat = ref_attention(
            q_lat, latent, value, batch, self.scale, causal=bool(batch.common.causal)
        )[..., : s.kv_lora_rank]
        return torch.einsum("thl,hvl->thv", out_lat, w_uv)

    def check(
        self,
        batch: PagedBatch,
        query: torch.Tensor,
        output: torch.Tensor,
        metadata: AiterMLAMetadata | None = None,
    ) -> None:
        num_decode_tokens = metadata.num_decode_tokens if metadata is not None else 0
        expected = self.reference(batch, query, num_decode_tokens)
        actual = output[: batch.num_tokens].float()
        assert torch.isfinite(actual).all(), "non-finite attention output"
        if not self.quantized:
            torch.testing.assert_close(actual, expected, **tolerance(self.dtype))
            return
        # fp8 prefill casts the decompressed q/k/v to fp8 at unit scale, and
        # fp8 decode accumulates over an fp8 query; neither rounding is in
        # the reference. Bound the bulk tightly and the worst element loosely,
        # so a dropped scale (a 2-4x error) still fails.
        err = (actual - expected).abs()
        bad = (err > FP8_ATOL + FP8_RTOL * expected.abs()).float().mean().item()
        assert bad <= FP8_MAX_BAD_FRACTION, f"{bad:.2%} elements out of tolerance"
        assert err.max().item() <= FP8_MAX_ABS, f"max abs error {err.max():.3f}"


def blocks_for(seq_lens: list[int], block_size: int, slack: int = 8) -> int:
    return sum(-(-s // block_size) for s in seq_lens) + slack
