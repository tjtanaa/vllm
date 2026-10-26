# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan MoRIIO KV transfers at real model shapes and deployment units, on CPU.

Every prefill and decode rank of a P/D deployment unit is built with its real
per-rank KV cache specs, in the layout each ROCm attention backend resolves to.
The connector's own planner then plans one request:

- WRITE: ``MoRIIOWriter._prepare_transfer_plan`` (and the Mamba variant) for
  every target of the prefill rank's TP mapping;
- READ: ``_compute_block_transfer_offsets`` (and the Mamba variant) on every
  decode rank. On main, READ plans only between equal TP sizes.

Caches are meta tensors, so a 128k-token request at full model size costs no
memory. Per case it reports transfer segments, bytes moved against the bytes
the decode ranks must receive, bytes per segment, and planning time per rank.

``--check`` (on by default) also replays each plan on small real buffers, one
layer per layer kind with a few blocks, and checks that every decode rank holds
exactly its shard afterwards.

``--save`` writes the results as JSON. ``--baseline`` compares with such a file
and exits non-zero when segments or bytes change, planning gets slower than
``--max-slowdown`` times the baseline, or a case that passed now fails.

    python benchmarks/kv_connector/bench_moriio_transfer.py --list
    python benchmarks/kv_connector/bench_moriio_transfer.py \\
        --deployment llama-3.1-70b --mode write --tp 8:4 --save main.json
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest import mock

# Splitting GDN conv state across TP sizes needs the (dim, state_len) layout.
os.environ.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "DS")

import msgspec
import torch
from tabulate import tabulate

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOAgentMetadata,
    RemoteAllocInfo,
    WriteTask,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWriter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout import (
    kda_conv_ssm,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_tp_mapping import (
    _rank_heads,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    derive_mamba_conv_split,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    SparseCacheRole,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

MODEL_DTYPE = torch.bfloat16
# Transfer groups, in the order the hybrid KV cache manager lists them.
GROUPS = ("full", "swa", "mamba")
# Fragmented block tables draw from a pool this many times the request.
POOL_FACTOR = 4

# --------------------------------------------------------------------------
# Deployment units. Shapes come from each model's config.json.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GDN:
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int


@dataclass(frozen=True)
class Deployment:
    model: str
    attn_set: int  # 1: attention, 2: MLA, 3: sparse MLA (see BACKENDS)
    layers: tuple[tuple[str, int], ...]  # (kind, count)
    tp: tuple[tuple[int, int], ...]  # default (prefill, decode) TP sizes
    num_kv_heads: int = 1
    head_dim: int = 0
    sliding_window: int | None = None
    gdn: GDN | None = None


DEPLOYMENTS = {
    "llama-3.1-8b": Deployment(
        "meta-llama/Llama-3.1-8B-Instruct",
        1,
        (("full", 32),),
        ((1, 1), (1, 2), (2, 1), (2, 2)),
        num_kv_heads=8,
        head_dim=128,
    ),
    "llama-3.1-70b": Deployment(
        "meta-llama/Llama-3.1-70B-Instruct",
        1,
        (("full", 80),),
        ((8, 8), (4, 8), (8, 4)),
        num_kv_heads=8,
        head_dim=128,
    ),
    "gpt-oss-120b": Deployment(
        "openai/gpt-oss-120b",
        1,
        (("full", 18), ("swa", 18)),
        ((4, 4), (4, 8), (8, 4)),
        num_kv_heads=8,
        head_dim=64,
        sliding_window=128,
    ),
    "qwen3-next-80b": Deployment(
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        1,
        (("full", 12), ("gdn", 36)),
        ((4, 4), (2, 4), (4, 2)),
        num_kv_heads=2,
        head_dim=256,
        gdn=GDN(16, 32, 128, 128, 4),
    ),
    "deepseek-v3": Deployment(
        "deepseek-ai/DeepSeek-V3",
        2,
        (("mla", 61),),
        ((8, 8), (8, 1), (1, 8)),
        # kv_lora_rank 512 + qk_rope_head_dim 64.
        head_dim=576,
    ),
    "deepseek-v3.2": Deployment(
        "deepseek-ai/DeepSeek-V3.2-Exp",
        3,
        (("mla", 61), ("indexer", 61)),
        ((8, 8),),
        head_dim=576,
    ),
}

# --------------------------------------------------------------------------
# Attention backends: the KV cache layout each one resolves to on ROCm with a
# KV connector, and how it customizes the cache spec.
# --------------------------------------------------------------------------


def _aiter_fa_spec(shuffle: bool) -> Callable[[KVCacheSpec], KVCacheSpec]:
    def customize(spec: KVCacheSpec) -> KVCacheSpec:
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.v1.attention.backends.rocm_aiter_fa import (
            AiterFlashAttentionBackend,
        )

        with mock.patch.object(
            rocm_aiter_ops, "is_shuffle_kv_cache_enabled", lambda: shuffle
        ):
            return AiterFlashAttentionBackend.customize_spec(spec)

    return customize


def _sparse_mla_spec(spec: KVCacheSpec) -> KVCacheSpec:
    from vllm.v1.attention.backends.mla.sparse_utils import align_blocks_to_rows

    if getattr(spec, "cache_role", None) == SparseCacheRole.INDEXER:
        return spec
    return align_blocks_to_rows(spec)


@dataclass(frozen=True)
class Backend:
    attn_set: int
    layout: KVCacheLayout
    block_size: int
    customize: Callable[[KVCacheSpec], KVCacheSpec] = lambda spec: spec
    # K and V are published as two head slots, each holding every KV head.
    kv_planes: bool = False


BACKENDS = {
    # No declared layouts: the default preference, LBNHC, applies.
    "TRITON_ATTN": Backend(1, KVCacheLayout.LBNHC, 16),
    # LBHNC only when a KV connector is configured; preferred block size 64.
    "ROCM_AITER_UNIFIED_ATTN": Backend(1, KVCacheLayout.LBHNC, 64),
    "ROCM_AITER_FA": Backend(1, KVCacheLayout.LBHNC, 16, _aiter_fa_spec(False)),
    # VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1: K and V planes, H outermost.
    "ROCM_AITER_FA_SHUFFLE": Backend(
        1, KVCacheLayout.LHBNC, 16, _aiter_fa_spec(True), kv_planes=True
    ),
    "TRITON_MLA": Backend(2, KVCacheLayout.LBNHC, 16),
    "ROCM_AITER_MLA": Backend(2, KVCacheLayout.LBNHC, 16),
    "ROCM_AITER_MLA_SPARSE": Backend(3, KVCacheLayout.LBNHC, 16, _sparse_mla_spec),
}

# --------------------------------------------------------------------------
# Ranks
# --------------------------------------------------------------------------


@dataclass
class Layer:
    name: str
    kind: str
    group: str


def _layers(dep: Deployment, per_kind: int | None = None) -> list[Layer]:
    group = {"full": "full", "mla": "full", "indexer": "full", "swa": "swa"}
    return [
        Layer(f"{kind}{i}", kind, group.get(kind, "mamba"))
        for kind, count in dep.layers
        for i in range(count if per_kind is None else min(count, per_kind))
    ]


def _groups(dep: Deployment) -> list[str]:
    used = {layer.group for layer in _layers(dep, per_kind=1)}
    return [g for g in GROUPS if g in used]


def _gdn_spec(dep: Deployment, tp_size: int, page_size_padded: int | None):
    gdn = dep.gdn
    assert gdn is not None
    shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp_size,
        gdn.num_k_heads,
        gdn.num_v_heads,
        gdn.head_k_dim,
        gdn.head_v_dim,
        gdn.conv_kernel,
    )
    dtypes = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        MODEL_DTYPE, "auto", "auto"
    )
    return MambaSpec(
        block_size=1,
        shapes=shapes,
        dtypes=dtypes,
        page_size_padded=page_size_padded,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )


def _attn_spec(
    dep: Deployment, backend: Backend, kind: str, heads: int, block_size: int
) -> KVCacheSpec:
    spec: KVCacheSpec
    if kind == "full":
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=heads,
            head_size=dep.head_dim,
            dtype=MODEL_DTYPE,
        )
    elif kind == "swa":
        assert dep.sliding_window is not None
        spec = SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=heads,
            head_size=dep.head_dim,
            dtype=MODEL_DTYPE,
            sliding_window=dep.sliding_window,
        )
    elif kind == "mla":
        spec = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=dep.head_dim,
            dtype=MODEL_DTYPE,
        )
    else:
        # DeepSeek-V3.2 indexer keys: 128 fp8 values and a 4-byte scale per
        # 128, one vector per token.
        spec = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=128 + 128 // 128 * 4,
            dtype=torch.uint8,
            cache_role=SparseCacheRole.INDEXER,
        )
    return backend.customize(spec)


def hybrid_block_size(dep: Deployment, block_size: int, tp_sizes: list[int]) -> int:
    """Attention block size whose page holds a Mamba state on every rank.

    Prefill and decode must use the same block size, so the largest the TP
    sizes need is used on both sides.
    """
    if dep.gdn is None:
        return block_size
    needed = block_size
    for tp in tp_sizes:
        heads = max(1, dep.num_kv_heads // tp)
        per_token = 2 * heads * dep.head_dim * MODEL_DTYPE.itemsize
        state = _gdn_spec(dep, tp, None).page_size_bytes
        tokens = math.ceil(state / per_token)
        needed = max(needed, math.ceil(tokens / block_size) * block_size)
    return needed


@dataclass
class Rank:
    tp_rank: int
    tp_size: int
    layers: list[Layer]
    specs: dict[str, KVCacheSpec]
    caches: dict[str, torch.Tensor]
    regions: dict[str, list[torch.Tensor]]  # per layer, in session order
    worker: MoRIIOConnectorWorker


def build_rank(
    dep: Deployment,
    backend: Backend,
    layers: list[Layer],
    tp_rank: int,
    tp_size: int,
    num_blocks: int,
    block_size: int,
    device: str,
) -> Rank:
    heads = len(_rank_heads(tp_rank, tp_size, dep.num_kv_heads))
    specs: dict[str, KVCacheSpec] = {}
    caches: dict[str, torch.Tensor] = {}
    regions: dict[str, list[torch.Tensor]] = {}
    attn_page = None
    for layer in layers:
        if layer.kind == "gdn":
            continue
        spec = _attn_spec(dep, backend, layer.kind, heads, block_size)
        specs[layer.name] = spec
        strides = compute_layout_strides(spec, num_blocks, 1, backend.layout)[1:]
        shape = (
            num_blocks,
            spec.num_heads,
            spec.num_states,
            spec.state_content_size_bytes,
        )
        size = 1 + sum((n - 1) * s for n, s in zip(shape, strides))
        buffer = torch.zeros(size, dtype=torch.uint8, device=device)
        caches[layer.name] = torch.as_strided(buffer, shape, strides)
        regions[layer.name] = [buffer]
        if layer.kind == "full":
            attn_page = spec.page_size_bytes
    for layer in layers:
        if layer.kind != "gdn":
            continue
        spec = _gdn_spec(dep, tp_size, attn_page)
        specs[layer.name] = spec
        buffer = torch.zeros(
            spec.page_size_bytes * num_blocks, dtype=torch.uint8, device=device
        )
        cache = buffer.view(num_blocks, 1, 1, spec.page_size_bytes)
        caches[layer.name] = cache
        if device != "meta":
            conv, ssm = kda_conv_ssm(cache, spec)
            regions[layer.name] = [
                MoRIIOConnectorWorker._contiguous_byte_alias(conv),
                MoRIIOConnectorWorker._contiguous_byte_alias(ssm),
            ]

    groups = _groups(dep)
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = tp_rank
    worker.world_size = tp_size
    worker.use_mla = dep.attn_set > 1
    worker._has_mamba = dep.gdn is not None
    worker.model_config = SimpleNamespace(
        get_total_num_kv_heads=lambda: dep.num_kv_heads
    )
    worker.kv_caches = caches
    worker.layer_to_spec = specs
    worker.layer_to_group = {layer.name: groups.index(layer.group) for layer in layers}
    worker._mamba_payload_index_by_layer = {
        layer.name: groups.index("mamba") for layer in layers if layer.kind == "gdn"
    }
    worker._conv_decomp = next(
        (
            derive_mamba_conv_split(spec, tp_size)
            for spec in specs.values()
            if isinstance(spec, MambaSpec)
        ),
        None,
    )
    worker.kv_layer_mr_offset = {}
    worker._mamba_offset_templates = {}
    worker._region_session_index = None
    worker.layer_name_to_local_kv_cache_metadata = {
        layer.name: [b""] * (2 if layer.kind == "gdn" else 1) for layer in layers
    }
    worker._record_slice_layouts(caches)
    return Rank(tp_rank, tp_size, layers, specs, caches, regions, worker)


def agent_metadata(rank: Rank, num_blocks: int) -> MoRIIOAgentMetadata:
    """What the rank advertises in the handshake, after its msgpack round trip."""
    worker = rank.worker
    metadata = MoRIIOAgentMetadata(
        engine_id=f"tp{rank.tp_rank}",
        agent_metadata=b"",
        kv_caches_base_addr=[],
        num_blocks=num_blocks,
        block_len=0,
        attn_backend_name="bench",
        kv_head_layouts={
            name: layout.to_wire() for name, layout in worker.kv_head_layouts.items()
        },
        mamba_state_layouts={
            name: layout.to_wire()
            for name, layout in worker.mamba_state_layouts.items()
        },
        attention_block_sizes=worker.attention_block_sizes,
    )
    return msgspec.msgpack.decode(
        msgspec.msgpack.encode(metadata), type=MoRIIOAgentMetadata
    )


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


def request_blocks(
    dep: Deployment,
    tokens: int,
    block_size: int,
    pattern: str,
    seed: int,
) -> tuple[list[list[int]], int]:
    """One side's block ids per transfer group, and the cache size in blocks."""
    counts = {
        "full": math.ceil(tokens / block_size),
        "swa": math.ceil((dep.sliding_window or 0) / block_size) + 1,
        "mamba": 1,
    }
    groups = _groups(dep)
    total = sum(counts[g] for g in groups)
    num_blocks = POOL_FACTOR * total + 1
    if pattern == "contiguous":
        # Block 0 is the null block.
        ids = list(range(1, total + 1))
    else:
        ids = random.Random(seed).sample(range(1, num_blocks), total)
    out, start = [], 0
    for group in groups:
        out.append(ids[start : start + counts[group]])
        start += counts[group]
    return out, num_blocks


@dataclass
class Plan:
    """One transfer: byte ranges in a source region and a destination region."""

    layer: str
    session: int
    src_rank: int
    dst_rank: int
    local: list[int]
    remote: list[int]
    sizes: list[int]


def plan_write(
    producer: Rank,
    consumer_tp: int,
    metadata: list[MoRIIOAgentMetadata],
    local_blocks: list[list[int]],
    remote_blocks: list[list[int]],
) -> list[Plan]:
    """Every transfer one prefill rank posts for one request."""
    worker = producer.worker
    writer = MoRIIOWriter.__new__(MoRIIOWriter)
    writer._worker_ref = lambda: worker
    mapping = worker._write_tp_mapping(consumer_tp)
    info = RemoteAllocInfo(block_ids=remote_blocks)
    plans = []
    for layer in producer.layers:
        task = WriteTask(
            request_id="req",
            transfer_id="xfer",
            dst_engine_id="decode",
            local_block_ids=local_blocks,
            remote_block_ids_hint=None,
            layer_name=layer.name,
            event=None,
            remote_notify_port=0,
            remote_ip="decode",
            remote_tp_size=consumer_tp,
            tp_mapping=mapping,
        )
        is_mamba = layer.kind == "gdn"
        for target in mapping.targets:
            if not is_mamba and not target.num_heads:
                # Another prefill rank writes this attention copy.
                continue
            remote = metadata[target.tp_rank]
            if is_mamba:
                layer_plans = writer._prepare_mamba_transfer_plans(
                    task, info, target, remote
                )
            else:
                layer_plans = [
                    writer._prepare_transfer_plan(
                        task, info, remote, remote_engine_id="decode", target=target
                    )
                ]
            for p in layer_plans:
                plans.append(
                    Plan(
                        layer.name,
                        p.sess_idx - worker._region_session_indices(layer.name)[0],
                        producer.tp_rank,
                        target.tp_rank,
                        p.transfer_local_offsets,
                        p.transfer_remote_offsets,
                        p.transfer_sizes,
                    )
                )
    return plans


def plan_read(
    consumer: Rank,
    producer_tp: int,
    metadata: list[MoRIIOAgentMetadata],
    local_blocks: list[list[int]],
    remote_blocks: list[list[int]],
) -> list[Plan]:
    """Every read one decode rank posts for one request (equal TP only)."""
    worker = consumer.worker
    if producer_tp != worker.world_size:
        raise NotImplementedError("READ across TP sizes is not planned on main")
    source = consumer.tp_rank
    plans = []
    for layer in consumer.layers:
        if layer.kind == "gdn":
            index = worker._mamba_payload_index_by_layer[layer.name]
            local, remote, sizes, n_conv = worker._compute_mamba_transfer_offsets(
                layer.name, local_blocks[index], remote_blocks[index], producer_tp
            )
            for session, part in ((0, slice(0, n_conv)), (1, slice(n_conv, None))):
                plans.append(
                    Plan(
                        layer.name,
                        session,
                        source,
                        consumer.tp_rank,
                        remote[part],
                        local[part],
                        sizes[part],
                    )
                )
            continue
        group = worker.layer_to_group[layer.name]
        local, remote, sizes = worker._compute_block_transfer_offsets(
            layer.name,
            local_blocks[group],
            remote_blocks[group],
            metadata[source],
            remote_tp_size=producer_tp,
        )
        # Stored source-first, like a WRITE plan.
        plans.append(
            Plan(layer.name, 0, source, consumer.tp_rank, remote, local, sizes)
        )
    return plans


# --------------------------------------------------------------------------
# Expected bytes and the replay check
# --------------------------------------------------------------------------


def shard_bytes(rank: Rank, layer: Layer, blocks: list[list[int]], groups) -> int:
    """Bytes of ``layer`` a decode rank must receive for the request."""
    spec = rank.specs[layer.name]
    if layer.kind == "gdn":
        return spec.state_content_size_bytes * len(blocks[groups.index("mamba")])
    return spec.real_page_size_bytes * len(blocks[groups.index(layer.group)])


def _head_chunks(
    cache: torch.Tensor, backend: Backend, block: int, head_dim: int
) -> torch.Tensor:
    """A block's bytes as [plane, KV head, bytes], heads in rank order.

    With K/V planes the head axis of the cache holds the two planes, and every
    KV head is one contiguous run of block_size * head_dim values in a plane.
    """
    page = cache[block]
    if backend.kv_planes:
        run = page.shape[1] * head_dim * MODEL_DTYPE.itemsize
        return torch.stack([page[s].reshape(-1) for s in range(page.shape[0])]).view(
            page.shape[0], -1, run
        )
    return page.reshape(1, page.shape[0], -1)


def _gdn_parts(rank: Rank, layer: Layer, slot: int) -> list[torch.Tensor]:
    """Byte views of a GDN state slot: Q, K and V conv rows, then the SSM heads."""
    conv, ssm = kda_conv_ssm(rank.caches[layer.name], rank.specs[layer.name])
    decomp = rank.worker._conv_decomp
    conv_bytes = conv[slot].view(torch.uint8)
    parts, start = [], 0
    for dim in decomp.local_proj_dims:
        parts.append(conv_bytes[start : start + dim].reshape(-1))
        start += dim
    parts.append(ssm[slot].view(torch.uint8).reshape(-1))
    return parts


def replay_check(
    dep: Deployment,
    backend: Backend,
    mode: str,
    prefill_tp: int,
    decode_tp: int,
    block_size: int,
    check_blocks: int,
) -> str:
    """Replay real plans on small buffers and compare every decode shard."""
    if dep.gdn is not None and not is_conv_state_dim_first():
        return "skip"
    layers = _layers(dep, per_kind=1)
    groups = _groups(dep)
    counts = {"full": check_blocks, "swa": min(check_blocks, 2), "mamba": 1}
    num_blocks = 3 * sum(counts[g] for g in groups) + 1
    rng = random.Random(0)
    src_ids = rng.sample(range(1, num_blocks), sum(counts[g] for g in groups))
    dst_ids = rng.sample(range(1, num_blocks), len(src_ids))

    def split(ids):
        out, start = [], 0
        for g in groups:
            out.append(ids[start : start + counts[g]])
            start += counts[g]
        return out

    src_blocks, dst_blocks = split(src_ids), split(dst_ids)
    prefill = [
        build_rank(dep, backend, layers, p, prefill_tp, num_blocks, block_size, "cpu")
        for p in range(prefill_tp)
    ]
    decode = [
        build_rank(dep, backend, layers, d, decode_tp, num_blocks, block_size, "cpu")
        for d in range(decode_tp)
    ]
    gen = torch.Generator().manual_seed(0)
    # Global content: per attention block [slot, KV head, bytes], per GDN slot
    # the unsharded parts. Each rank holds its heads / its slice of each part.
    content: dict[tuple[str, int], list[torch.Tensor]] = {}

    def global_parts(layer: Layer, rank: Rank, block: int) -> list[torch.Tensor]:
        key = (layer.name, block)
        if key not in content:
            if layer.kind == "gdn":
                parts = [
                    torch.randint(1, 255, (p.numel() * rank.tp_size,), generator=gen)
                    for p in _gdn_parts(rank, layer, block)
                ]
            else:
                chunks = _head_chunks(
                    rank.caches[layer.name], backend, block, dep.head_dim
                )
                heads = 1 if layer.kind in ("mla", "indexer") else dep.num_kv_heads
                shape = (chunks.shape[0], heads, chunks.shape[2])
                parts = [torch.randint(1, 255, shape, generator=gen)]
            content[key] = [p.to(torch.uint8) for p in parts]
        return content[key]

    def shard(layer: Layer, rank: Rank, block: int) -> list[torch.Tensor]:
        parts = global_parts(layer, rank, block)
        if layer.kind == "gdn":
            out = []
            for part in parts:
                size = part.numel() // rank.tp_size
                out.append(part[rank.tp_rank * size : (rank.tp_rank + 1) * size])
            return out
        if layer.kind in ("mla", "indexer"):
            return [parts[0]]
        held = _rank_heads(rank.tp_rank, rank.tp_size, dep.num_kv_heads)
        return [parts[0][:, held]]

    def fill(rank: Rank, layer: Layer, block: int, parts: list[torch.Tensor]):
        if layer.kind == "gdn":
            for view, part in zip(_gdn_parts(rank, layer, block), parts):
                view.copy_(part)
            return
        cache = rank.caches[layer.name]
        if backend.kv_planes:
            for s in range(cache.shape[1]):
                cache[block, s].view(-1).copy_(parts[0][s].reshape(-1))
        else:
            cache[block].copy_(parts[0][0].reshape(cache[block].shape))

    def read(rank: Rank, layer: Layer, block: int) -> list[torch.Tensor]:
        if layer.kind == "gdn":
            return [v.clone() for v in _gdn_parts(rank, layer, block)]
        return [
            _head_chunks(rank.caches[layer.name], backend, block, dep.head_dim).clone()
        ]

    for rank in prefill:
        for layer in layers:
            group = groups.index(layer.group)
            for block in src_blocks[group]:
                fill(rank, layer, block, shard(layer, rank, block))

    if mode == "write":
        metadata = [agent_metadata(rank, num_blocks) for rank in decode]
        plans = [
            p
            for rank in prefill
            for p in plan_write(rank, decode_tp, metadata, src_blocks, dst_blocks)
        ]
    else:
        metadata = [agent_metadata(rank, num_blocks) for rank in prefill]
        plans = [
            p
            for rank in decode
            for p in plan_read(rank, prefill_tp, metadata, dst_blocks, src_blocks)
        ]
    for p in plans:
        src = prefill[p.src_rank].regions[p.layer][p.session]
        dst = decode[p.dst_rank].regions[p.layer][p.session]
        for s, d, n in zip(p.local, p.remote, p.sizes):
            dst[d : d + n] = src[s : s + n]

    for rank in decode:
        for layer in layers:
            group = groups.index(layer.group)
            for src, dst in zip(src_blocks[group], dst_blocks[group]):
                expected = shard(layer, rank, src)
                got = read(rank, layer, dst)
                if layer.kind != "gdn":
                    expected = [expected[0].reshape(got[0].shape)]
                if not all(torch.equal(e, g) for e, g in zip(expected, got)):
                    return f"fail {layer.kind} rank {rank.tp_rank}"
    return "pass"


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------


def run_case(
    dep_name: str,
    backend_name: str,
    mode: str,
    prefill_tp: int,
    decode_tp: int,
    tokens: int,
    pattern: str,
    args,
    check: bool,
) -> dict[str, Any]:
    dep = DEPLOYMENTS[dep_name]
    backend = BACKENDS[backend_name]
    block_size = hybrid_block_size(
        dep, args.block_size or backend.block_size, [prefill_tp, decode_tp]
    )
    row: dict[str, Any] = {
        "deployment": dep_name,
        "backend": backend_name,
        "mode": mode,
        "tp": f"{prefill_tp}:{decode_tp}",
        "tokens": tokens,
        "pattern": pattern,
        "block_size": block_size,
    }
    try:
        src_blocks, num_blocks = request_blocks(dep, tokens, block_size, pattern, 1)
        dst_blocks, _ = request_blocks(dep, tokens, block_size, pattern, 2)
        layers = _layers(dep)
        prefill = [
            build_rank(
                dep, backend, layers, p, prefill_tp, num_blocks, block_size, "meta"
            )
            for p in range(prefill_tp)
        ]
        decode = [
            build_rank(
                dep, backend, layers, d, decode_tp, num_blocks, block_size, "meta"
            )
            for d in range(decode_tp)
        ]
        planners = prefill if mode == "write" else decode
        metadata = [
            agent_metadata(rank, num_blocks)
            for rank in (decode if mode == "write" else prefill)
        ]

        def plan(rank: Rank) -> list[Plan]:
            if mode == "write":
                return plan_write(rank, decode_tp, metadata, src_blocks, dst_blocks)
            return plan_read(rank, prefill_tp, metadata, dst_blocks, src_blocks)

        moved = segments = 0
        rank_ms = []
        for rank in planners:
            times: list[float] = []
            # Up to --iters plans, but stop repeating slow ones after a second.
            while len(times) < args.iters and sum(times) < 1e3:
                start = time.perf_counter()
                rank_plans = plan(rank)
                times.append((time.perf_counter() - start) * 1e3)
            rank_ms.append(statistics.median(times))
            moved += sum(sum(p.sizes) for p in rank_plans)
            segments += sum(len(p.sizes) for p in rank_plans)
            del rank_plans
        groups = _groups(dep)
        expected = sum(
            shard_bytes(rank, layer, dst_blocks, groups)
            for rank in decode
            for layer in layers
        )
        row.update(
            status="ok" if moved == expected else "bytes",
            segments=segments,
            bytes=moved,
            expected_bytes=expected,
            bytes_per_segment=moved // max(segments, 1),
            plan_ms=max(rank_ms),
        )
    except NotImplementedError as e:
        row.update(status="unsupported", error=str(e))
        return row
    except Exception as e:
        row.update(status="error", error=f"{type(e).__name__}: {e}")
    if check:
        try:
            row["check"] = replay_check(
                dep,
                backend,
                mode,
                prefill_tp,
                decode_tp,
                block_size,
                args.check_blocks,
            )
        except Exception as e:
            row["check"] = f"error {type(e).__name__}"
            row.setdefault("error", f"{type(e).__name__}: {e}")
    return row


def _key(row: dict[str, Any]) -> tuple:
    return tuple(
        row[k] for k in ("deployment", "backend", "mode", "tp", "tokens", "pattern")
    )


def compare(rows, baseline_rows, max_slowdown: float) -> list[str]:
    base = {_key(row): row for row in baseline_rows}
    problems = []
    for row in rows:
        old = base.get(_key(row))
        if old is None:
            continue
        name = " ".join(map(str, _key(row)))
        if old.get("status") == "ok" and row.get("status") != "ok":
            problems.append(f"{name}: status {old['status']} -> {row['status']}")
        if old.get("check") == "pass" and row.get("check") not in (None, "pass"):
            problems.append(f"{name}: check {old['check']} -> {row['check']}")
        for field in ("segments", "bytes"):
            if field in old and field in row and old[field] != row[field]:
                problems.append(f"{name}: {field} {old[field]} -> {row[field]}")
        slower = "plan_ms" in old and "plan_ms" in row
        if slower and row["plan_ms"] > max_slowdown * old["plan_ms"]:
            problems.append(
                f"{name}: plan_ms {old['plan_ms']:.2f} -> {row['plan_ms']:.2f}"
            )
    return problems


def main(args) -> int:
    if args.list:
        for name, dep in DEPLOYMENTS.items():
            backends = [n for n, b in BACKENDS.items() if b.attn_set == dep.attn_set]
            tps = " ".join(f"{p}:{d}" for p, d in dep.tp)
            print(f"{name:16} {dep.model:36} tp {tps:16} {' '.join(backends)}")
        return 0
    rows: list[dict[str, Any]] = []
    for dep_name in args.deployment or list(DEPLOYMENTS):
        dep = DEPLOYMENTS[dep_name]
        backends = args.backend or [
            n for n, b in BACKENDS.items() if b.attn_set == dep.attn_set
        ]
        tps = [tuple(map(int, tp.split(":"))) for tp in args.tp] if args.tp else dep.tp
        for backend_name in backends:
            if BACKENDS[backend_name].attn_set != dep.attn_set:
                continue
            for mode in args.mode:
                for prefill_tp, decode_tp in tps:
                    # The replay does not depend on the request size.
                    check = args.check
                    for tokens in args.tokens:
                        for pattern in args.pattern:
                            row = run_case(
                                dep_name,
                                backend_name,
                                mode,
                                prefill_tp,
                                decode_tp,
                                tokens,
                                pattern,
                                args,
                                check,
                            )
                            check = False
                            rows.append(row)
                            print(json.dumps(row), file=sys.stderr, flush=True)
    columns = [
        "deployment",
        "backend",
        "mode",
        "tp",
        "tokens",
        "pattern",
        "status",
        "check",
        "segments",
        "bytes",
        "bytes_per_segment",
        "plan_ms",
    ]
    print(
        tabulate(
            [[row.get(c, "") for c in columns] for row in rows],
            headers=columns,
            floatfmt=".2f",
        )
    )
    for row in rows:
        if "error" in row:
            print(f"{' '.join(map(str, _key(row)))}: {row['error']}")
    if args.save:
        with open(args.save, "w") as f:
            json.dump(rows, f, indent=1)
    if args.baseline:
        with open(args.baseline) as f:
            problems = compare(rows, json.load(f), args.max_slowdown)
        for problem in problems:
            print(f"REGRESSION {problem}")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="List deployments.")
    parser.add_argument("--deployment", nargs="+", choices=list(DEPLOYMENTS))
    parser.add_argument("--backend", nargs="+", choices=list(BACKENDS))
    parser.add_argument(
        "--mode", nargs="+", default=["write", "read"], choices=["write", "read"]
    )
    parser.add_argument(
        "--tp",
        nargs="+",
        help="Prefill:decode TP sizes, e.g. 8:4. Default: per deployment.",
    )
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[1024, 8192, 32768, 131072]
    )
    parser.add_argument(
        "--pattern",
        nargs="+",
        default=["contiguous", "fragmented"],
        choices=["contiguous", "fragmented"],
        help="Block tables: one run (fresh engine) or scattered (busy engine).",
    )
    parser.add_argument(
        "--block-size", type=int, help="Default: the backend's block size."
    )
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay the plans on small real buffers and compare every shard.",
    )
    parser.add_argument("--check-blocks", type=int, default=3)
    parser.add_argument("--save", help="Write the results as JSON.")
    parser.add_argument("--baseline", help="JSON from --save to compare against.")
    parser.add_argument("--max-slowdown", type=float, default=1.5)
    sys.exit(main(parser.parse_args()))
