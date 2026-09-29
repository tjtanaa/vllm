# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in Qwen attention on gfx1151, retaining the ROCm cache contract."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionBackend,
    RocmAttentionImpl,
    RocmAttentionMetadata,
    RocmAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.gfx1151_qwen_prefill import gfx1151_qwen_prefill
from vllm.v1.attention.ops.gfx1151_verify_splitkv import (
    splitkv_supports,
    splitkv_verify_attention,
)
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)
# Delivered longest context for the tuned paths; VLLM_GFX1151_QWEN_MAX_CONTEXT
# raises it per process. See the comment in vllm/envs.py: a value below
# --max-model-len silently disables every tuned gfx1151 attention path, because
# CUDA-graph capture uses max_seq_len = max_model_len.
MAX_CONTEXT = 4096
MAX_BATCH = 8
MAX_QUERY = 32
PARTITION_SIZE = 256
# Tile shipped before the in-situ measurement below; kept for A/B runs.
LEGACY_VERIFY_TILE = (16, 64, 4)
_VERIFY_TILE: tuple[int, int, int] | None = None


def _verification_tile() -> tuple[int, int, int]:
    """(BLOCK_M, BLOCK_N, num_warps) for cached-prefix verification attention.

    Measured with benchmarks/kernels/bench_gfx1151_attention_in_situ.py, which
    reproduces serving conditions (16 GiB scattered page pool, L2 flushed by the
    surrounding weight streaming). At a 1.4K context the legacy 16/64/4 tile
    costs ~870 us per layer versus ~410 us for 16/128/8: the verification grid
    is only one workgroup per query head, so eight warps are needed to hide
    paged-KV latency once the L2 is cold.
    """
    global _VERIFY_TILE
    if _VERIFY_TILE is None:
        raw = str(envs.VLLM_GFX1151_QWEN_VERIFY_TILE).strip()
        if raw.lower() in ("legacy", "0", ""):
            _VERIFY_TILE = LEGACY_VERIFY_TILE
        else:
            try:
                parts = tuple(int(v) for v in raw.split(","))
                if len(parts) != 3 or min(parts) <= 0:
                    raise ValueError(raw)
                _VERIFY_TILE = parts  # type: ignore[assignment]
            except ValueError:
                logger.warning_once(
                    "Ignoring invalid VLLM_GFX1151_QWEN_VERIFY_TILE=%r; using %s",
                    raw,
                    LEGACY_VERIFY_TILE,
                )
                _VERIFY_TILE = LEGACY_VERIFY_TILE
    return _VERIFY_TILE


def supports_gfx1151_qwen_config(vllm_config: VllmConfig, config) -> bool:
    """Keep model-driven selection independent of the generic ROCm selector."""
    if not envs.VLLM_GFX1151_QWEN_ATTENTION or not current_platform.is_rocm():
        return False
    if not hasattr(torch.ops._rocm_C, "gfx1151_qwen_paged_attention"):
        return False
    device = torch.cuda.current_device()
    if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx1151":
        return False
    parallel = vllm_config.parallel_config
    if any(
        size != 1
        for size in (
            parallel.tensor_parallel_size,
            parallel.pipeline_parallel_size,
            parallel.prefill_context_parallel_size,
            parallel.decode_context_parallel_size,
        )
    ):
        return False
    if vllm_config.model_config.dtype != torch.bfloat16:
        return False
    if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        return False
    attention = vllm_config.attention_config
    if attention.backend not in (None, AttentionBackendEnum.GFX1151_QWEN_ATTN):
        return False
    if attention.backend_per_kind or vllm_config.kv_transfer_config is not None:
        return False
    if getattr(config, "dual_chunk_attention_config", None):
        return False
    geometry = (
        getattr(config, "num_attention_heads", None),
        getattr(config, "num_key_value_heads", None),
        getattr(config, "head_dim", None),
    )
    target = (
        getattr(config, "model_type", None) == "qwen3_5_text"
        and geometry == (24, 4, 256)
        and getattr(config, "is_causal", True)
    )
    draft = "DFlash2DraftModel" in (
        getattr(config, "architectures", None) or []
    ) and geometry == (32, 8, 128)
    return bool(target or draft)


@dataclass
class Gfx1151QwenAttentionMetadata(RocmAttentionMetadata):
    num_partitions: int = 0


class Gfx1151QwenAttentionMetadataBuilder(RocmAttentionMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        return Gfx1151QwenAttentionMetadata(
            **vars(metadata),
            num_partitions=(metadata.max_seq_len + PARTITION_SIZE - 1)
            // PARTITION_SIZE,
        )


class Gfx1151QwenAttentionBackend(RocmAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "GFX1151_QWEN_ATTN"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128, 256]

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        if not current_platform.is_rocm():
            return False
        device = torch.cuda.current_device()
        return (
            torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            == "gfx1151"
        )

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @staticmethod
    def get_builder_cls():
        return Gfx1151QwenAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return Gfx1151QwenAttentionImpl


class Gfx1151QwenAttentionImpl(RocmAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config()
        self._enabled = supports_gfx1151_qwen_config(
            config, config.model_config.hf_text_config
        )
        # A draft model may be constructed under the target's config context.
        if not self._enabled and config.speculative_config is not None:
            self._enabled = supports_gfx1151_qwen_config(
                config, config.speculative_config.draft_model_config.hf_config
            )
        self._max_context = max(MAX_CONTEXT, envs.VLLM_GFX1151_QWEN_MAX_CONTEXT)
        self._workspace_shape = (
            min(config.scheduler_config.max_num_seqs, MAX_BATCH) * MAX_QUERY,
            self.num_heads,
            (
                min(config.model_config.max_model_len, self._max_context)
                + PARTITION_SIZE
                - 1
            )
            // PARTITION_SIZE,
            self.head_size + 2,
        )

    def _can_use_paged_layout(
        self, query, kv_cache, metadata, output, output_scale, output_block_scale
    ):
        if not self._enabled or not isinstance(metadata, Gfx1151QwenAttentionMetadata):
            return False
        if (self.num_heads, self.num_kv_heads, self.head_size) not in (
            (24, 4, 256),
            (32, 8, 128),
        ):
            return False
        if (
            not 1024 <= metadata.max_seq_len <= self._max_context
            or not 1 <= metadata.seq_lens.numel() <= MAX_BATCH
            or metadata.num_actual_tokens <= 0
            or metadata.num_partitions
            != (metadata.max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE
            or metadata.use_cascade
        ):
            return False
        if (
            self.attn_type != AttentionType.DECODER
            or self.alibi_slopes is not None
            or self.logits_soft_cap
            or self.kv_cache_dtype not in ("auto", "bfloat16")
            or output_scale is not None
            or output_block_scale is not None
        ):
            return False
        if (
            query.dtype != torch.bfloat16
            or kv_cache.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
            or not query.is_cuda
            or kv_cache.device != query.device
            or output.device != query.device
        ):
            return False
        if (
            query.ndim != 3
            or output.shape != query.shape
            or query.shape[1:] != (self.num_heads, self.head_size)
            or query.stride(2) != 1
            or query.stride(1) < self.head_size
            or query.stride(0) < self.num_heads * query.stride(1)
            or not output.is_contiguous()
            or metadata.num_actual_tokens > query.shape[0]
        ):
            return False
        # split_kv_cache can represent these layouts without materializing a copy.
        if (
            kv_cache.ndim != 4
            or kv_cache.shape[0] == 0
            or kv_cache.shape[1] != 2
            or kv_cache.shape[2] <= 0
            or kv_cache.shape[3] != self.num_kv_heads * self.head_size
            or kv_cache.stride(3) != 1
            or kv_cache.stride(2) != kv_cache.shape[3]
            or min(kv_cache.stride(0), kv_cache.stride(1))
            < kv_cache.shape[2] * kv_cache.shape[3]
        ):
            return False
        nseq = metadata.seq_lens.numel()
        if (
            metadata.seq_lens.ndim != 1
            or metadata.query_start_loc.shape != (nseq + 1,)
            or metadata.block_table.ndim != 2
            or metadata.block_table.shape[0] != nseq
            or metadata.block_table.shape[1]
            < (metadata.max_seq_len + kv_cache.shape[2] - 1) // kv_cache.shape[2]
            or metadata.block_table.stride(1) != 1
            or metadata.block_table.stride(0) < metadata.block_table.shape[1]
        ):
            return False
        for tensor in (
            metadata.seq_lens,
            metadata.query_start_loc,
            metadata.block_table,
        ):
            if tensor.device != query.device or tensor.dtype != torch.int32:
                return False
        if (
            not metadata.seq_lens.is_contiguous()
            or not metadata.query_start_loc.is_contiguous()
        ):
            return False
        return self.sinks is None or not (
            self.sinks.dtype != torch.float32
            or self.sinks.device != query.device
            or self.sinks.shape != (self.num_heads,)
            or not self.sinks.is_contiguous()
        )

    def _can_use_hip(
        self, query, kv_cache, metadata, output, output_scale, output_block_scale
    ):
        if (
            not self._can_use_paged_layout(
                query, kv_cache, metadata, output, output_scale, output_block_scale
            )
            or not is_workspace_manager_initialized()
        ):
            return False
        # Multi-token target and draft paths need reference softmax rounding.
        return (
            self.head_size == 256
            and metadata.max_query_len == 1
            and metadata.num_actual_tokens <= self._workspace_shape[0]
            and metadata.num_partitions <= self._workspace_shape[2]
        )

    def _can_use_triton_prefill(
        self,
        query,
        key,
        value,
        kv_cache,
        metadata,
        output,
        output_scale,
        output_block_scale,
    ):
        if not self._can_use_paged_layout(
            query, kv_cache, metadata, output, output_scale, output_block_scale
        ):
            return False
        if self.head_size == 128:
            return (
                not metadata.causal
                and self.sinks is None
                and self.sliding_window == (2047, 0)
                and metadata.max_query_len in (4, 8, 16, 24, 32)
                and metadata.max_query_len <= metadata.max_seq_len
            )
        if (
            (self.num_heads, self.num_kv_heads, self.head_size) != (24, 4, 256)
            or not metadata.causal
            or self.sinks is not None
            or self.sliding_window[0] > 0
            or not (
                metadata.max_query_len in (4, 8, 16, 24, 32)
                or 1024 <= metadata.max_query_len <= self._max_context
            )
            or metadata.max_query_len > metadata.max_seq_len
        ):
            return False
        if metadata.max_query_len in (4, 8, 16, 32):
            # ROCm's separate cache update has already stored the new K/V.
            # Verification reads that cache, including mixed-M1 graph rows.
            return True
        for tensor in (key, value):
            if (
                tensor is None
                or tensor.ndim != 3
                or tensor.shape[0] < metadata.num_actual_tokens
                or tensor.shape[1:] != (self.num_kv_heads, self.head_size)
                or tensor.dtype != torch.bfloat16
                or tensor.device != query.device
                or tensor.stride(2) != 1
                or tensor.stride(1) < self.head_size
                or tensor.stride(0) < self.num_kv_heads * tensor.stride(1)
            ):
                return False
        return True

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if (
            attn_metadata is None
            and self._enabled
            and is_workspace_manager_initialized()
        ):
            # Reserve once during the profile run, before workspace locking and
            # graph capture. All attention layers share the worker's scratch pool.
            current_workspace_manager().get_simultaneous(
                (self._workspace_shape, torch.float32)
            )
        use_hip = self._can_use_hip(
            query, kv_cache, attn_metadata, output, output_scale, output_block_scale
        )
        use_triton = not use_hip and self._can_use_triton_prefill(
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
        if not (use_hip or use_triton):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        n = attn_metadata.num_actual_tokens
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1), self.num_kv_heads, self.head_size
        )
        if use_triton:
            draft = self.head_size == 128
            width = attn_metadata.max_query_len
            cached_verification = draft or width in (4, 8, 16, 32)
            if (
                cached_verification
                and not draft
                and envs.VLLM_GFX1151_QWEN_VERIFY_SPLITKV
                # Measured: 128.8 -> 125.4 ms/step at concurrency 1, but
                # neutral-to-slightly-worse at concurrency 4 (253.7 -> 257.2 ms),
                # where the prefix path already has 4x24 workgroups and the
                # extra partition combine is not repaid. Gate on the padded
                # batch, which is a static per-graph value.
                and attn_metadata.seq_lens.numel() <= 2
                and splitkv_supports(
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_size,
                    width,
                    attn_metadata.max_seq_len,
                    attn_metadata.causal,
                    max(0, self.sliding_window[0]),
                    self.sinks,
                    query.dtype,
                )
            ):
                # One workgroup covers a whole GQA group and the context is
                # split across workgroups, so each K/V byte is read once and the
                # combine is deterministic. Ragged query lengths (including
                # mixed single-token graph rows) are masked per sequence, so the
                # separate decode launch the prefix path needs is not required.
                splitkv_verify_attention(
                    query[:n],
                    key_cache,
                    value_cache,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    attn_metadata.query_start_loc,
                    output[:n],
                    max_query_len=width,
                    max_seq_len=attn_metadata.max_seq_len,
                    sm_scale=self.scale,
                )
                logger.info_once(
                    "gfx1151 Qwen split-KV verification engaged: D=%d, M=%d",
                    self.head_size,
                    width,
                )
                return output
            if draft:
                config = (16 if width <= 16 else 32, 64, 8)
            elif width in (4, 8, 16):
                config = _verification_tile()
            elif width in (24, 32):
                block_m = 16 if attn_metadata.seq_lens.numel() == 1 else 32
                config = (block_m, 64, 4)
            else:
                # This choice wins for full prompts and remains repeatably
                # faster for cached-prefix chunks, without a host tensor read.
                config = (64, 64, 8)
            gfx1151_qwen_prefill(
                query[:n],
                None if cached_verification else key[:n],
                None if cached_verification else value[:n],
                output[:n],
                key_cache,
                value_cache,
                attn_metadata.block_table,
                attn_metadata.query_start_loc,
                attn_metadata.seq_lens,
                attn_metadata.max_seq_len,
                attn_metadata.max_query_len,
                layer._k_scale,
                layer._v_scale,
                self.scale,
                config,
                causal=attn_metadata.causal,
                window=max(0, self.sliding_window[0]),
            )
            logger.info_once(
                "gfx1151 Qwen Triton prefix engaged: D=%d, M=%d, tile=%s",
                self.head_size,
                attn_metadata.max_query_len,
                config,
            )
            return output
        (workspace,) = current_workspace_manager().get_simultaneous(
            (
                (n, self.num_heads, attn_metadata.num_partitions, self.head_size + 2),
                torch.float32,
            )
        )
        ops.gfx1151_qwen_paged_attention(
            query[:n],
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.query_start_loc,
            self.sinks,
            output[:n],
            workspace,
            attn_metadata.max_seq_len,
            attn_metadata.max_query_len,
            self.scale,
            max(0, self.sliding_window[0]),
            attn_metadata.causal,
        )
        logger.info_once(
            "gfx1151 Qwen HIP attention engaged: D=%d, M=%d",
            self.head_size,
            attn_metadata.max_query_len,
        )
        return output
