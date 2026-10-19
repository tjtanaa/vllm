# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mock end-to-end MoRIIO READ between prefill and decode of any TP sizes.

Every decode rank runs the real read path (source selection, offsets, release
bookkeeping) on real cache views; a fake MoRI session copies the planned bytes
out of the chosen prefill rank's cache, and the releases reach the prefill
ranks through the real wrapper and ack counting. Checks that every decode rank
pulls exactly its shard of the requested blocks, nothing else is touched,
every prefill rank is released exactly once per transfer, and TP pairs that
would need split KV heads or recurrent state are refused.
"""

import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import ssm_conv_transfer_utils
from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    MoRIIOAgentMetadata,
    MoRIIOMode,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
    get_moriio_remote_tp_rank,
    resolve_moriio_transfer_ack,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout import (
    kda_conv_ssm,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    derive_mamba_conv_split,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

pytestmark = pytest.mark.cpu_test

NOTIFY_PORT = 20000
NUM_BLOCKS = 6
BLOCK_SIZE = 4
# Prefill blocks and the decode blocks they land in, per transfer group.
PREFILL_BLOCKS = {"full": [1, 4], "swa": [3], "mamba": [2]}
DECODE_BLOCKS = {"full": [5, 2], "swa": [0], "mamba": [3]}
# GDN state: Q and K are KEY_DIM channels, V is V_HEADS * V_DIM channels.
KEY_DIM, V_HEADS, V_DIM, K_DIM, CONV_ROWS = 4, 4, 2, 2, 2


@dataclass(frozen=True)
class Layer:
    name: str
    kind: str  # "full", "swa", "mla" or "mamba"
    group: str  # transfer group the layer's blocks come from


MODELS = {
    "gqa": (4, [Layer("attn0", "full", "full"), Layer("attn1", "full", "full")]),
    "gqa_replicated": (2, [Layer("attn0", "full", "full")]),
    "mla": (1, [Layer("mla0", "mla", "full"), Layer("mla1", "mla", "full")]),
    "sliding": (4, [Layer("attn0", "full", "full"), Layer("swa0", "swa", "swa")]),
    "gdn_hybrid": (
        4,
        [Layer("attn0", "full", "full"), Layer("gdn0", "mamba", "mamba")],
    ),
    "mla_hybrid": (1, [Layer("mla0", "mla", "full"), Layer("kda0", "mamba", "mamba")]),
}


def _groups(model: str, hma: bool) -> list[str]:
    _, layers = MODELS[model]
    groups = sorted({layer.group for layer in layers}, key=list(DECODE_BLOCKS).index)
    return groups if hma else ["full"]


def _spec(layer: Layer, num_heads: int, tp_size: int):
    if layer.kind == "full":
        return FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=num_heads,
            head_size=2,
            dtype=torch.bfloat16,
        )
    if layer.kind == "swa":
        return SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=num_heads,
            head_size=2,
            dtype=torch.bfloat16,
            sliding_window=8,
        )
    if layer.kind == "mla":
        return MLAAttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
        )
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(
            ((2 * KEY_DIM + V_HEADS * V_DIM) // tp_size, CONV_ROWS),
            (V_HEADS // tp_size, V_DIM, K_DIM),
        ),
        dtypes=(torch.uint8, torch.uint8),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )


def _global_state(layers: list[Layer], heads: int) -> dict[str, torch.Tensor]:
    """The unsharded content of every layer, per block / state slot."""
    gen = torch.Generator().manual_seed(0)
    state = {}
    for layer in layers:
        if layer.kind == "mamba":
            n = (2 * KEY_DIM + V_HEADS * V_DIM) * CONV_ROWS + V_HEADS * V_DIM * K_DIM
            shape: tuple[int, ...] = (NUM_BLOCKS, n)
        else:
            spec = _spec(layer, 1 if layer.kind == "mla" else heads, 1)
            shape = (
                NUM_BLOCKS,
                spec.num_heads,
                spec.num_states,
                spec.state_content_size_bytes,
            )
        state[layer.name] = torch.randint(
            1, 255, shape, dtype=torch.uint8, generator=gen
        )
    return state


def _mamba_shard(slot: torch.Tensor, tp_rank: int, tp_size: int) -> tuple:
    """(conv, ssm) bytes of ``tp_rank`` for one unsharded GDN state slot."""
    parts, offset = [], 0
    for channels in (KEY_DIM, KEY_DIM, V_HEADS * V_DIM):
        size = channels * CONV_ROWS
        shard = size // tp_size
        parts.append(slot[offset + tp_rank * shard : offset + (tp_rank + 1) * shard])
        offset += size
    ssm = slot[offset:].view(V_HEADS, -1)
    heads = V_HEADS // tp_size
    return torch.cat(parts), ssm[tp_rank * heads : (tp_rank + 1) * heads].reshape(-1)


@dataclass
class Rank:
    tp_rank: int
    tp_size: int
    caches: dict[str, torch.Tensor]  # registered tensor per layer
    regions: list[torch.Tensor]  # registered byte regions, in session order
    worker: MoRIIOConnectorWorker
    wrapper: Any = None
    sent: list = field(default_factory=list)


def _build_rank(model, tp_rank, tp_size, layout, hma, state, is_producer) -> Rank:
    heads, layers = MODELS[model]
    groups = _groups(model, hma)
    caches, regions, specs, meta_counts = {}, [], {}, {}
    for layer in layers:
        layer_heads = len(_rank_heads(tp_rank, tp_size, heads))
        spec = _spec(layer, layer_heads, tp_size)
        specs[layer.name] = spec
        buffer = torch.zeros(spec.page_size_bytes * NUM_BLOCKS, dtype=torch.uint8)
        if layer.kind == "mamba":
            cache = buffer.view(NUM_BLOCKS, 1, 1, spec.page_size_bytes)
            conv, ssm = kda_conv_ssm(cache, spec)
            if is_producer:
                for slot in range(NUM_BLOCKS):
                    c, s = _mamba_shard(state[layer.name][slot], tp_rank, tp_size)
                    conv[slot].view(-1).copy_(c)
                    ssm[slot].view(-1).copy_(s)
            regions += [
                MoRIIOConnectorWorker._contiguous_byte_alias(conv),
                MoRIIOConnectorWorker._contiguous_byte_alias(ssm),
            ]
            meta_counts[layer.name] = 2
        else:
            strides = compute_layout_strides(spec, NUM_BLOCKS, 1, layout)[1:]
            shape = (
                NUM_BLOCKS,
                spec.num_heads,
                spec.num_states,
                spec.state_content_size_bytes,
            )
            cache = torch.as_strided(buffer, shape, strides)
            if is_producer:
                held = (
                    slice(None)
                    if layer.kind == "mla"
                    else _rank_heads(tp_rank, tp_size, heads)
                )
                cache.copy_(state[layer.name][:, held])
            regions.append(buffer)
            meta_counts[layer.name] = 1
        caches[layer.name] = cache

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = tp_rank
    worker.world_size = tp_size
    worker.use_mla = layers[0].kind == "mla"
    worker._has_mamba = any(layer.kind == "mamba" for layer in layers)
    worker.model_config = SimpleNamespace(get_total_num_kv_heads=lambda: heads)
    worker.kv_caches = caches
    worker.layer_to_spec = specs
    worker.layer_to_group = {
        layer.name: groups.index(layer.group if hma else "full") for layer in layers
    }
    worker._mamba_payload_index_by_layer = {
        layer.name: groups.index("mamba") for layer in layers if layer.kind == "mamba"
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
        layer.name: [b""] * meta_counts[layer.name] for layer in layers
    }
    # Built without __init__: there is no engine to shut down.
    worker.shutdown = lambda: None
    return Rank(tp_rank, tp_size, caches, regions, worker)


def _rank_heads(tp_rank: int, tp_size: int, total_num_kv_heads: int) -> range:
    """KV heads a rank holds: a split when there are enough, else one replica."""
    if tp_size <= total_num_kv_heads:
        per_rank = total_num_kv_heads // tp_size
        return range(tp_rank * per_rank, (tp_rank + 1) * per_rank)
    head = tp_rank * total_num_kv_heads // tp_size
    return range(head, head + 1)


def _expected(
    layer: Layer, group: str, state, heads, tp_rank, tp_size, cache
) -> torch.Tensor:
    """What the decode cache must hold after the transfer."""
    expected = torch.zeros_like(cache)
    pairs = zip(PREFILL_BLOCKS[group], DECODE_BLOCKS[group])
    if layer.kind == "mamba":
        expected = torch.zeros(NUM_BLOCKS, cache.shape[-1], dtype=torch.uint8)
        for src, dst in pairs:
            conv, ssm = _mamba_shard(state[layer.name][src], tp_rank, tp_size)
            expected[dst, : conv.numel() + ssm.numel()] = torch.cat([conv, ssm])
        return expected
    held = slice(None) if layer.kind == "mla" else _rank_heads(tp_rank, tp_size, heads)
    for src, dst in pairs:
        expected[dst] = state[layer.name][src, held]
    return expected


def _supported(model: str, producer_tp: int, consumer_tp: int) -> bool:
    """READ copies whole blocks, so across TP sizes both ranks must hold the
    same heads (MLA latents or replicated KV heads) and no recurrent state."""
    heads, layers = MODELS[model]
    if producer_tp == consumer_tp:
        return True
    if any(layer.kind == "mamba" for layer in layers):
        return False
    return layers[0].kind == "mla" or min(producer_tp, consumer_tp) >= heads


SCENARIOS = [
    (model, producer_tp, consumer_tp, hma)
    for model in MODELS
    for producer_tp, consumer_tp in [
        (1, 2),
        (2, 4),
        (2, 1),
        (4, 2),
        (1, 1),
        (2, 2),
        (4, 1),
    ]
    for hma in (True, False)
    # Recurrent state needs the hybrid KV cache manager.
    if hma or "hybrid" not in model
]


class _Status:
    """A read that completed when it was posted."""

    def Failed(self) -> bool:
        return False

    def Succeeded(self) -> bool:
        return True


class _Session:
    def __init__(self, tp_rank: int, src: torch.Tensor, dst: torch.Tensor):
        self.tp_rank, self.src, self.dst = tp_rank, src, dst


def _producer_wrapper() -> MoRIIOWrapper:
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_req_ids = []
    wrapper.done_remote_allocate_req_dict = {}
    wrapper._terminal_transfer_ids = OrderedDict()
    return wrapper


def _connect(consumer: Rank, producers: list[Rank]) -> list[int]:
    """Point the decode rank's sessions and releases at the prefill ranks.

    Returns the list the prefill rank of every posted read is appended to.
    """
    worker = consumer.worker
    worker.mode = MoRIIOMode.READ
    worker.moriio_config = SimpleNamespace(
        transfer_timeout=60.0, recv_abort_timeout=60.0
    )
    worker._num_mamba_transfer_groups = int(worker._has_mamba)
    worker._recving_transfers = defaultdict(dict)
    worker._recving_transfers_callback_addr = {}
    worker._recving_transfers_start = {}
    worker._recving_local_blocks = {}
    worker._reads_issued_this_step = []
    worker._mamba_reads_this_step = []
    metadata = MoRIIOAgentMetadata(
        engine_id="prefill",
        agent_metadata=b"",
        kv_caches_base_addr=[],
        num_blocks=NUM_BLOCKS,
        block_len=0,
        attn_backend_name="test",
    )
    sources: list[int] = []

    def get_built_session(engine_id: str):
        if "_tp" in engine_id:
            tp_rank = int(engine_id.rsplit("_tp", 1)[1])
        else:
            # The handshake dials this prefill rank for the DP-only key.
            tp_rank = get_moriio_remote_tp_rank(
                consumer.tp_rank, consumer.tp_size, len(producers)
            )
        sessions = [
            _Session(tp_rank, src, dst)
            for src, dst in zip(
                producers[tp_rank].regions, consumer.regions, strict=True
            )
        ]
        return sessions, metadata

    def read_remote_data(sizes, local_offsets, remote_offsets, session):
        sources.append(session.tp_rank)
        for size, local, remote in zip(sizes, local_offsets, remote_offsets):
            session.dst[local : local + size] = session.src[remote : remote + size]
        return _Status()

    def send_notify(transfer_id, ip, port, message_type, message_fields=None):
        consumer.sent.append((int(port) - NOTIFY_PORT, message_type))
        producers[int(port) - NOTIFY_PORT].wrapper._handle_structured_message(
            {"type": message_type, "transfer_id": transfer_id, **(message_fields or {})}
        )

    consumer.wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    consumer.wrapper.lock = threading.Lock()
    consumer.wrapper.read_remote_data = read_remote_data
    consumer.wrapper.send_notify = send_notify
    worker.moriio_wrapper = consumer.wrapper
    worker._get_built_session = get_built_session
    return sources


def _read(consumer: Rank, transfer_id: str, groups: list[str], producer_tp: int):
    consumer.worker._read_blocks_for_req(
        f"req-{transfer_id}",
        SimpleNamespace(
            remote_engine_id="prefill",
            transfer_id=transfer_id,
            local_block_ids=[DECODE_BLOCKS[g] for g in groups],
            remote_block_ids=[PREFILL_BLOCKS[g] for g in groups],
            remote_host="prefill",
            remote_notify_port=NOTIFY_PORT,
            tp_size=producer_tp,
            remote_dp_size=1,
            remote_dp_rank=0,
        ),
    )


def _setup(monkeypatch, model, producer_tp, layout, hma):
    monkeypatch.setattr(
        ssm_conv_transfer_utils, "is_conv_state_dim_first", lambda: True
    )
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.PRODUCER)
    heads, layers = MODELS[model]
    state = _global_state(layers, heads)
    producers = [
        _build_rank(model, p, producer_tp, layout, hma, state, True)
        for p in range(producer_tp)
    ]
    for producer in producers:
        producer.wrapper = _producer_wrapper()
    return state, producers


@pytest.mark.parametrize(
    "layout",
    [KVCacheLayout.LBNHC, KVCacheLayout.LBHNC],
    ids=["token-major", "head-major"],
)
@pytest.mark.parametrize(
    ("model", "producer_tp", "consumer_tp", "hma"),
    [s for s in SCENARIOS if _supported(*s[:3])],
)
def test_read_pulls_every_shard_byte_exact(
    monkeypatch, model, producer_tp, consumer_tp, hma, layout
):
    state, producers = _setup(monkeypatch, model, producer_tp, layout, hma)
    heads, layers = MODELS[model]
    groups = _groups(model, hma)
    # The flexible mirror round-robins its source; read once per prefill rank.
    transfer_ids = [f"t{i}" for i in range(producer_tp)]
    for d in range(consumer_tp):
        consumer = _build_rank(model, d, consumer_tp, layout, hma, state, False)
        sources = _connect(consumer, producers)
        for transfer_id in transfer_ids:
            _read(consumer, transfer_id, groups, producer_tp)
        assert consumer.worker._pop_done_transfers() == set(transfer_ids)

        flexible = consumer_tp == 1 and consumer.worker.use_mla and producer_tp > 1
        assert set(sources) == (
            set(range(producer_tp))
            if flexible
            else {get_moriio_remote_tp_rank(d, consumer_tp, producer_tp)}
        )
        for layer in layers:
            cache = consumer.caches[layer.name]
            received = cache.view(NUM_BLOCKS, -1) if layer.kind == "mamba" else cache
            expected = _expected(
                layer,
                layer.group if hma else "full",
                state,
                heads,
                d,
                consumer_tp,
                received,
            )
            assert torch.equal(received, expected), (layer.name, d)

    # Each prefill rank hears once from every decode rank that stands in for
    # it, and resolves every transfer exactly once.
    for producer in producers:
        acks = producer.wrapper.pop_finished_req_ids()
        assert sorted(ack.transfer_id for ack in acks) == sorted(
            transfer_ids * max(consumer_tp // producer_tp, 1)
        ), producer.tp_rank
        counts: dict[str, int] = {}
        completed: set[str] = set()
        resolved = [
            resolve_moriio_transfer_ack(
                ack, producer_tp, transfer_ids, counts, completed
            )
            for ack in acks
        ]
        assert sorted(filter(None, resolved)) == transfer_ids, producer.tp_rank


@pytest.mark.parametrize(
    ("model", "producer_tp", "consumer_tp", "hma"),
    [s for s in SCENARIOS if not _supported(*s[:3])],
)
def test_read_refuses_split_heads_and_state(
    monkeypatch, model, producer_tp, consumer_tp, hma
):
    layout = KVCacheLayout.LBNHC
    state, producers = _setup(monkeypatch, model, producer_tp, layout, hma)
    consumer = _build_rank(model, 0, consumer_tp, layout, hma, state, False)
    sources = _connect(consumer, producers)
    with pytest.raises(
        NotImplementedError, match="replicated KV heads|heterogeneous-TP mamba"
    ):
        _read(consumer, "t", _groups(model, hma), producer_tp)
    assert not sources
    assert not any(region.any() for region in consumer.regions)
