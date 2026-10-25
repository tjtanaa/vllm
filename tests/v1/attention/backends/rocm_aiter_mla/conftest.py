# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import tempfile

import pytest
import torch

from tests.v1.attention.backends.harness import KernelSpy
from vllm.platforms import current_platform


def _aiter_mla_runnable() -> str | None:
    if not current_platform.is_rocm():
        return "ROCm only"
    from vllm._aiter_ops import is_aiter_found_and_supported
    from vllm.platforms.rocm import get_cdna_version

    if not is_aiter_found_and_supported():
        return "AITER not installed or unsupported on this GPU"
    if get_cdna_version() <= 2:
        return "ROCM_AITER_MLA needs CDNA3 or newer"
    return None


SKIP_REASON = _aiter_mla_runnable()


def pytest_collection_modifyitems(config, items):
    if SKIP_REASON is None:
        return
    marker = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        if item.path.parent.name == "rocm_aiter_mla":
            item.add_marker(marker)


@pytest.fixture
def spy(monkeypatch):
    from tests.v1.attention.backends.rocm_aiter_mla.mla_harness import (
        watch_all_kernels,
    )

    kernel_spy = KernelSpy(monkeypatch)
    watch_all_kernels(kernel_spy)
    return kernel_spy


@pytest.fixture
def fp8_asm_prefill():
    """Skip unless this GPU and AITER build have the fp8 asm MLA prefill.
    A fixture, not a collection-time skipif: probing imports AITER."""
    from vllm.v1.attention.backends.mla import rocm_aiter_mla

    if not rocm_aiter_mla._fp8_mla_prefill_supported():
        pytest.skip("no mla_prefill_ps_asm_fwd on this GPU / AITER build")


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    # The session-wide TP=1 group below must outlive each test.
    return False


@pytest.fixture(scope="session", autouse=True)
def _tp1_group():
    """A single-rank TP group, so kv_b_proj is built as a model builds it."""
    if SKIP_REASON is not None:
        yield
        return
    from tests.utils import ensure_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        with ensure_current_vllm_config():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{path}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
        yield
        cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


@pytest.fixture(autouse=True)
def _workspace():
    """The builder reserves fp8-prefill and DCP scratch from the workspace
    manager, as it does before the runner locks it after warmup."""
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))
    yield
    reset_workspace_manager()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
