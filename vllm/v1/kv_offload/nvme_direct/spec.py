# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NvmeDirectOffloadingSpec: KV offload straight between the GPU pool and
NVMe files — no CPU staging region. For unified-memory platforms (gfx1151
Strix Halo, GB10) where the KV pool is host-visible.

kv_connector_extra_config:
    spec_name: "NvmeDirectOffloadingSpec"
    root_dir:  (required) directory for chunk files
    n_threads: (optional, default 8) I/O thread-pool size
    fsync:     (optional, default false) fsync each chunk file after write
               (crash/power-loss durability; process-restart persistence
               does not need it — page cache outlives the process)
    max_disk_bytes: (optional, default 1 TiB) capacity for LRU eviction
    blocks_per_chunk: must be 1 in this version
"""

from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.nvme_direct.manager import NvmeDirectManager
from vllm.v1.kv_offload.nvme_direct.worker import NvmeDirectWorker

logger = init_logger(__name__)


class NvmeDirectOffloadingSpec(OffloadingSpec):
    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        ec = self.extra_config
        self.root_dir = ec.get("root_dir")
        if not self.root_dir:
            raise ValueError(
                "NvmeDirectOffloadingSpec requires 'root_dir' in "
                "kv_connector_extra_config"
            )
        self.n_threads = int(ec.get("n_threads", 8))
        self.fsync = bool(ec.get("fsync", False))
        max_disk_bytes = int(ec.get("max_disk_bytes", 1 << 40))
        if self.blocks_per_chunk != 1:
            raise ValueError(
                "NvmeDirectOffloadingSpec currently requires "
                f"blocks_per_chunk=1 (got {self.blocks_per_chunk})"
            )
        if config.worker_kv_bytes_per_block <= 0:
            raise ValueError(
                "NvmeDirectOffloadingSpec needs worker_kv_bytes_per_block "
                "from the KV cache config"
            )
        self.kv_bytes_per_chunk = (
            config.worker_kv_bytes_per_block * self.blocks_per_chunk
        )
        self.num_chunks = max(1, max_disk_bytes // self.kv_bytes_per_chunk)
        self._manager: NvmeDirectManager | None = None
        self._worker: NvmeDirectWorker | None = None
        logger.info(
            "NvmeDirectOffloadingSpec: root=%s chunk=%d B num_chunks=%d "
            "threads=%d fsync=%s",
            self.root_dir,
            self.kv_bytes_per_chunk,
            self.num_chunks,
            self.n_threads,
            self.fsync,
        )

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, "Any"]:
        return {}

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            self._manager = NvmeDirectManager(
                num_chunks=self.num_chunks, root_dir=self.root_dir
            )
        return self._manager

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            self._worker = NvmeDirectWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                root_dir=self.root_dir,
                n_threads=self.n_threads,
                fsync=self.fsync,
            )
        return self._worker
