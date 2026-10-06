# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for the direct-NVMe offloading tier.

Chunks are content-addressed files (block-hash + group index), so the index
rebuilt lazily by `lookup()` via os.path.exists() makes the tier survive
full server restarts with no journal: the first request after a restart
resolves hits synchronously (one stat per chunk, page-cache warm) instead of
missing while an async verdict is in flight.

Capacity is modeled in virtual chunk slots; eviction unlinks the victim's
file (LRU, pinned chunks exempt).
"""

import os
from collections import OrderedDict
from collections.abc import Collection, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.nvme_direct.spec_common import NvmeLoadStoreSpec

logger = init_logger(__name__)


class NvmeDirectManager(OffloadingManager):
    def __init__(self, num_chunks: int, root_dir: str) -> None:
        self._root = root_dir
        self._capacity = num_chunks
        os.makedirs(root_dir, exist_ok=True)
        # stored: key -> virtual slot (LRU order: oldest first)
        self._index: OrderedDict[OffloadKey, int] = OrderedDict()
        self._pending_store: dict[OffloadKey, int] = {}
        self._pins: dict[OffloadKey, int] = {}
        self._free_slots: list[int] = list(range(num_chunks))
        self._adopted = 0
        self._evictions = 0
        self._dbg_lookups = 0
        self._dbg_index_hits = 0
        self._dbg_disk_probes = 0
        self._dbg_sample_miss: str | None = None

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _path(self, key: OffloadKey) -> str:
        h = get_offload_block_hash(key).hex()
        g = get_offload_group_idx(key)
        return os.path.join(self._root, h[:2], f"{h[2:]}_g{g}.bin")

    @staticmethod
    def _tmp_path(path: str) -> str:
        return f"{path}.{os.getpid()}.tmp"

    # ------------------------------------------------------------------
    # Slot accounting
    # ------------------------------------------------------------------
    def _alloc_slot(self) -> tuple[int, list[OffloadKey]]:
        evicted: list[OffloadKey] = []
        while not self._free_slots:
            victim = None
            for k in self._index:  # oldest first
                if self._pins.get(k) or k in self._pending_store:
                    continue
                victim = k
                break
            if victim is None:
                raise RuntimeError(
                    "NvmeDirectManager: no evictable slots (all pinned)"
                )
            slot = self._index.pop(victim)
            try:
                os.unlink(self._path(victim))
            except FileNotFoundError:
                pass
            self._free_slots.append(slot)
            evicted.append(victim)
            self._evictions += 1
        return self._free_slots.pop(), evicted

    # ------------------------------------------------------------------
    # OffloadingManager API
    # ------------------------------------------------------------------
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        self._dbg_lookups += 1
        if self._dbg_lookups % 500 == 1:
            logger.info(
                "NvmeDirect lookup stats: lookups=%d index_hits=%d "
                "disk_probes=%d adopted=%d stored=%d",
                self._dbg_lookups,
                self._dbg_index_hits,
                self._dbg_disk_probes,
                self._adopted,
                len(self._index),
            )
        if key in self._pending_store:
            return LookupResult.MISS  # being written; not readable yet
        if key in self._index:
            self._index.move_to_end(key)
            self._dbg_index_hits += 1
            return LookupResult.HIT
        # Cold index (fresh process): adopt from disk synchronously.
        self._dbg_disk_probes += 1
        path = self._path(key)
        if self._dbg_disk_probes <= 3:
            logger.info(
                "NvmeDirect lookup probe: key=%s g=%d path=%s exists=%s",
                get_offload_block_hash(key).hex()[:16],
                get_offload_group_idx(key),
                path,
                os.path.exists(path),
            )
        if os.path.exists(path):
            try:
                slot, _ = self._alloc_slot()
            except RuntimeError:
                return LookupResult.MISS
            self._index[key] = slot
            self._adopted += 1
            return LookupResult.HIT
        if self._dbg_sample_miss is None:
            self._dbg_sample_miss = path
        return LookupResult.MISS

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> NvmeLoadStoreSpec:
        slots: list[int] = []
        paths: list[str] = []
        groups: list[int] = []
        for k in keys:
            slot = self._index.get(k)
            if slot is None:
                # Adopted at lookup; if it vanished meanwhile, fail the job.
                raise KeyError(f"NvmeDirectManager: key not stored: {k!r}")
            self._pins[k] = self._pins.get(k, 0) + 1
            slots.append(slot)
            paths.append(self._path(k))
            groups.append(get_offload_group_idx(k))
        return NvmeLoadStoreSpec(slots, paths, groups)

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        for k in keys:
            p = self._pins.get(k)
            if p is not None:
                if p <= 1:
                    del self._pins[k]
                else:
                    self._pins[k] = p - 1
            if k in self._index:
                self._index.move_to_end(k)

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        to_store = [
            k
            for k in keys
            if k not in self._index and k not in self._pending_store
        ]
        self._dbg_store_calls = getattr(self, "_dbg_store_calls", 0) + 1
        if self._dbg_store_calls <= 40:
            from collections import Counter

            gc = Counter(get_offload_group_idx(k) for k in keys)
            logger.info(
                "NvmeDirect prepare_store #%d: offered=%d new=%d groups=%s",
                self._dbg_store_calls,
                len(keys),
                len(to_store),
                dict(sorted(gc.items())),
            )
        if not to_store:
            return None
        # Group-major order so the medium-side spec aligns with the
        # GPULoadStoreSpec segmentation (blocks ordered by group index).
        to_store.sort(key=get_offload_group_idx)
        slots: list[int] = []
        paths: list[str] = []
        groups: list[int] = []
        evicted: list[OffloadKey] = []
        for k in to_store:
            try:
                slot, ev = self._alloc_slot()
            except RuntimeError:
                break
            evicted.extend(ev)
            self._pending_store[k] = slot
            slots.append(slot)
            paths.append(self._path(k))
            groups.append(get_offload_group_idx(k))
        if not slots:
            return None
        return PrepareStoreOutput(
            keys_to_store=to_store[: len(slots)],
            store_spec=NvmeLoadStoreSpec(slots, paths, groups),
            evicted_keys=evicted,
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ):
        for k in keys:
            slot = self._pending_store.pop(k, None)
            if slot is None:
                continue
            if success:
                self._index[k] = slot
                self._index.move_to_end(k)
            else:
                try:
                    os.unlink(self._tmp_path(self._path(k)))
                except FileNotFoundError:
                    pass
                self._free_slots.append(slot)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        for k in keys:
            if k in self._index:
                self._index.move_to_end(k)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def on_request_finished(self, req_context: ReqContext) -> None:
        return None

    def take_events(self) -> Iterable[OffloadingEvent]:
        return ()

    def has_pending_work(self) -> bool:
        return bool(self._pending_store)

    def reset_cache(self) -> None:
        self._index.clear()
        self._pending_store.clear()
        self._pins.clear()
        self._free_slots = list(range(self._capacity))

    def get_stats(self):
        return None

    def shutdown(self) -> None:
        logger.info(
            "NvmeDirectManager shutdown: adopted=%d evictions=%d stored=%d "
            "lookups=%d index_hits=%d disk_probes=%d sample_miss=%s",
            self._adopted,
            self._evictions,
            len(self._index),
            self._dbg_lookups,
            self._dbg_index_hits,
            self._dbg_disk_probes,
            self._dbg_sample_miss,
        )
