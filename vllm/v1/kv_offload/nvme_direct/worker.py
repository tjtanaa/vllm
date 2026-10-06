# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side direct GPU-pool <-> NVMe transfers (unified-memory platforms).

Bypasses the CPU offload region entirely: chunk files are written with a
single vectored os.writev() whose iovecs are ctypes views over the KV pool
pages (host-visible on unified-memory APUs like gfx1151 Strix Halo and GB10),
and loaded with os.preadv() directly into freshly allocated pool blocks.
One file per chunk = per (block, canonical KV group); within a file, pages
are laid out per layer in group_data_refs order, so the granularity is
per-layer and layers could later be filtered or streamed individually.

Latency hiding: stores are enqueued to a thread pool at submit time and the
calling (model) thread never blocks; each job first waits on a CUDA event
recorded at submission, so CPU reads of pool pages only happen after the
producing kernels retired. Loads run on the same pool; the connector gates
request scheduling on job completion (HIT_PENDING -> complete_load).
"""

import ctypes
import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.nvme_direct.spec_common import NvmeLoadStoreSpec

logger = init_logger(__name__)


class NvmeDirectWorker(OffloadingWorker):
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        root_dir: str,
        n_threads: int = 8,
        fsync: bool = False,
    ) -> None:
        if blocks_per_chunk != 1:
            raise ValueError(
                "NvmeDirectWorker currently requires blocks_per_chunk=1 "
                f"(got {blocks_per_chunk})"
            )
        self._root = root_dir
        self._fsync = fsync
        os.makedirs(root_dir, exist_ok=True)

        # Per-tensor (base_ptr, row_stride) for the int8 [blocks, page] views.
        self._tensors: list[tuple[int, int, torch.Tensor]] = []
        for t in kv_caches.tensors:
            view = t.tensor.view(torch.int8).view(-1, t.page_size_bytes)
            # Keep a reference to the tensor so its storage stays alive.
            self._tensors.append((view.data_ptr(), view.stride(0), view))

        # Per-group layer refs: (tensor_idx, page_size_bytes) in file order.
        self._groups: list[list[tuple[int, int]]] = [
            [(r.tensor_idx, r.page_size_bytes) for r in refs]
            for refs in kv_caches.group_data_refs
        ]
        self._chunk_group_bytes = [
            sum(psz for _, psz in g) for g in self._groups
        ]

        # Host-visibility probe: unified memory is a hard requirement.
        ptr, _, _ = self._tensors[0]
        try:
            _ = bytes((ctypes.c_char * 8).from_address(ptr))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "NvmeDirectWorker requires host-visible KV pool memory "
                "(unified-memory APU). Probe failed: "
                f"{e!r}. Use the CPU-tier offloading spec instead."
            ) from e

        self._pool = ThreadPoolExecutor(
            max_workers=n_threads, thread_name_prefix="nvme_direct"
        )
        self._futures: dict[int, list[Future]] = {}
        self._lock = threading.Lock()
        logger.info(
            "NvmeDirectWorker ready: root=%s threads=%d groups=%s "
            "bytes/group=%s fsync=%s",
            root_dir,
            n_threads,
            len(self._groups),
            self._chunk_group_bytes,
            fsync,
        )

    # ------------------------------------------------------------------
    # Buffer construction (zero-copy views over pool pages)
    # ------------------------------------------------------------------
    def _bufs(self, block_id: int, g_idx: int) -> list[memoryview]:
        bufs = []
        for t_idx, page_bytes in self._groups[g_idx]:
            base, stride, _ = self._tensors[t_idx]
            arr = (ctypes.c_char * page_bytes).from_address(base + block_id * stride)
            bufs.append(memoryview(arr))
        return bufs

    def _segments(self, gpu_spec: GPULoadStoreSpec) -> list[list[int]]:
        """Split group-major block_ids into per-group segments."""
        block_ids = [int(b) for b in gpu_spec.block_ids]
        group_sizes = gpu_spec.group_sizes
        if group_sizes is None:
            return [block_ids]
        segs = []
        off = 0
        for gs in group_sizes:
            segs.append(block_ids[off : off + gs])
            off += gs
        return segs

    def _pair_jobs(
        self, gpu_spec: GPULoadStoreSpec, med_spec: NvmeLoadStoreSpec
    ) -> list[tuple[str, int, int]]:
        """Pair each chunk path with its GPU block id, per group segment."""
        segs = self._segments(gpu_spec)
        jobs: list[tuple[str, int, int]] = []
        # Walk groups in order; within a group, k-th path pairs with k-th block.
        for g_idx, seg in enumerate(segs):
            paths_g = [
                i
                for i, gg in enumerate(med_spec.groups)
                if gg == g_idx
            ]
            if len(paths_g) != len(seg):
                raise ValueError(
                    f"group {g_idx}: {len(paths_g)} paths vs {len(seg)} blocks"
                )
            for pos, i in enumerate(paths_g):
                jobs.append((med_spec.paths[i], seg[pos], g_idx))
        return jobs

    # ------------------------------------------------------------------
    # I/O primitives
    # ------------------------------------------------------------------
    def _store_one(self, path: str, block_id: int, g_idx: int, ev) -> int:
        if ev is not None:
            ev.synchronize()
        bufs = self._bufs(block_id, g_idx)
        total = sum(b.nbytes for b in bufs)
        d = os.path.dirname(path)
        os.makedirs(d, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            written = 0
            while written < total:
                n = os.writev(fd, bufs)
                if n <= 0:
                    raise IOError(f"writev returned {n}")
                written += n
                if written < total:
                    # Partial write: rebuild remaining views.
                    rem = written
                    nb = []
                    for b in bufs:
                        if rem >= b.nbytes:
                            rem -= b.nbytes
                            continue
                        nb.append(b[rem:])
                        rem = 0
                    bufs = nb
            if self._fsync:
                os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        return total

    def _load_one(self, path: str, block_id: int, g_idx: int) -> int:
        bufs = self._bufs(block_id, g_idx)
        total = sum(b.nbytes for b in bufs)
        fd = os.open(path, os.O_RDONLY)
        try:
            got = 0
            while got < total:
                n = os.preadv(fd, bufs, got)
                if n <= 0:
                    raise IOError(
                        f"preadv {path}: got {got + max(n,0)} of {total}"
                    )
                got += n
                if got < total:
                    rem = got
                    nb = []
                    for b in bufs:
                        if rem >= b.nbytes:
                            rem -= b.nbytes
                            continue
                        nb.append(b[rem:])
                        rem = 0
                    bufs = nb
        finally:
            os.close(fd)
        return got

    # ------------------------------------------------------------------
    # OffloadingWorker API
    # ------------------------------------------------------------------
    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        try:
            jobs = self._pair_jobs(src_spec, dst_spec)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            logger.error("nvme_direct store pairing failed: %s", e)
            return False
        # Stream-order snapshot: pool pages must not be read before the
        # producing kernels retire. Recorded once per job on the current
        # stream; pool threads wait on it (model thread never blocks).
        ev = None
        try:
            if torch.cuda.is_available():
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream())
        except Exception:  # noqa: BLE001
            ev = None
        futs = [
            self._pool.submit(self._store_one, p, b, g, ev) for p, b, g in jobs
        ]
        with self._lock:
            self._futures[job_id] = futs
        return True

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        try:
            jobs = self._pair_jobs(dst_spec, src_spec)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            logger.error("nvme_direct load pairing failed: %s", e)
            return False
        futs = [self._pool.submit(self._load_one, p, b, g) for p, b, g in jobs]
        with self._lock:
            self._futures[job_id] = futs
        return True

    def get_finished(self) -> list[TransferResult]:
        results = []
        t0 = time.monotonic()
        with self._lock:
            for job_id, futs in list(self._futures.items()):
                if not all(f.done() for f in futs):
                    continue
                ok = True
                size = 0
                for f in futs:
                    exc = f.exception()
                    if exc is not None:
                        ok = False
                        logger.error("nvme_direct chunk job failed: %r", exc)
                    else:
                        size += f.result() or 0
                results.append(
                    TransferResult(
                        job_id=job_id,
                        success=ok,
                        transfer_size=size,
                        transfer_time=time.monotonic() - t0,
                    )
                )
                del self._futures[job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        with self._lock:
            futs = [f for j in job_ids for f in self._futures.get(j, [])]
        for f in futs:
            f.result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
