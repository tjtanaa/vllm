# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Medium-side load/store spec for the direct-NVMe tier."""

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class NvmeLoadStoreSpec(BlockIDsLoadStoreSpec):
    """File-backed chunk descriptor.

    ``block_ids`` carries the manager's virtual chunk slots (kept for
    interface compatibility and capacity accounting); ``paths`` the
    content-addressed file per chunk; ``groups`` the KV-group index per
    chunk so the worker can pair paths with the group-major segments of
    the GPULoadStoreSpec regardless of key interleaving.
    """

    def __init__(
        self, slots: list[int], paths: list[str], groups: list[int]
    ) -> None:
        super().__init__(slots)
        self.paths = paths
        self.groups = groups

    @property
    def chunk_ids(self):
        return self.block_ids

    def __repr__(self) -> str:
        return f"NvmeLoadStoreSpec(chunks={len(self.paths)})"
