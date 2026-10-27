# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attention benchmark runner builds metadata the backends can split."""

import sys
from pathlib import Path

import pytest
import torch

from vllm.v1.attention.backends.utils import split_decodes_prefills_and_extends

RUNNER_DIR = Path(__file__).parents[2] / "benchmarks" / "attention_benchmarks"


@pytest.fixture
def runner(monkeypatch):
    if not (RUNNER_DIR / "runner.py").exists():
        pytest.skip("benchmarks/attention_benchmarks is not in this checkout")
    # runner.py imports its siblings (batch_spec, common) as top-level modules.
    monkeypatch.syspath_prepend(str(RUNNER_DIR))
    for name in ("runner", "common", "batch_spec"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import runner

    return runner


def test_runner_metadata_splits_into_decode_extend_prefill(runner):
    """Backends that split decodes, extends and prefills (ROCM_AITER_FA, MLA)
    read seq_lens_cpu_upper_bound, so prefill and extend specs need it."""
    q_lens, kv_lens = [1, 8, 32], [100, 64, 32]
    common = runner._build_common_attn_metadata(
        q_lens, kv_lens, block_size=16, device=torch.device("cpu")
    )
    assert common.seq_lens_cpu_upper_bound.tolist() == kv_lens
    num_decodes, num_extends, num_prefills, *_ = split_decodes_prefills_and_extends(
        common, decode_threshold=1
    )
    assert (num_decodes, num_extends, num_prefills) == (1, 1, 1)
