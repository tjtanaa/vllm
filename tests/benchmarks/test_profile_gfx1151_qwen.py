# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode trace/benchmark contracts and isolated kernel correctness checks."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from benchmarks.profile_gfx1151_qwen import (
    ablate_attention_paths,
    profile_batch,
    record_attention_layouts,
)
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind


@pytest.mark.parametrize(
    "overrides,valid",
    [
        ({}, True),
        ({"m": 3}, False),
        ({"split_k": 3}, False),
        ({"split_k": 8}, True),
        ({"hip_row_chunk": 1}, True),
        ({"hip_row_chunk": 2}, True),
        ({"hip_row_chunk": 3}, False),
        ({"hip_row_chunk": 2, "split_k": 8}, False),
        ({"hip_row_chunk": 2, "m": 1}, False),
        ({"hip_row_chunk": 2, "inventory_k": 20480}, False),
        ({"registered_row_chunk": True, "m": 4}, True),
        ({"registered_row_chunk": True}, False),
        ({"registered_row_chunk": True, "m": 4, "split_k": 2}, False),
        ({"registered_row_chunk": True, "m": 4, "hip_row_chunk": 2}, False),
        ({"registered_row_chunk": True, "m": 4, "flydsl_wmma": True}, False),
        ({"registered_row_chunk": True, "m": 4, "inventory_k": 20480}, False),
        ({"native_variant": "unvalidated"}, False),
        ({"triton_group_major": True, "bm": 32}, True),
        ({"triton_group_major": True, "triton_exact_code_cast": True, "bm": 32}, True),
        ({"triton_exact_code_cast": True}, False),
        ({"flydsl_wmma": True, "triton_exact_code_cast": True}, False),
        ({"triton_group_major": True}, False),
        ({"triton_group_major": True, "bm": 32, "split_k": 2}, False),
        ({"triton_group_major": True, "bm": 32, "hip_row_chunk": 2}, False),
        ({"triton_group_major": True, "bm": 32, "flydsl_wmma": True}, False),
        ({"triton_group_major": True, "bm": 32, "stages": 1}, False),
        ({"triton_group_major": True, "bm": 32, "m": 1}, False),
        ({"triton_group_major": True, "bm": 32, "m": 2}, True),
        ({"triton_group_major": True, "bm": 32, "m": 4}, True),
        ({"triton_group_major": True, "bm": 32, "m": 16}, True),
        ({"triton_group_major": True, "bm": 32, "m": 24}, True),
        ({"triton_group_major": True, "bm": 32, "m": 32}, True),
        ({"triton_group_major": True, "bm": 32, "m": 64}, True),
        ({"triton_group_major": True, "bm": 32, "inventory_k": 6144}, True),
        ({"triton_group_major": True, "bm": 32, "inventory_k": 6144, "m": 4}, False),
        (
            {
                "triton_group_major": True,
                "bm": 32,
                "inventory_k": 5120,
                "inventory_n": 34816,
            },
            True,
        ),
        ({"triton_group_major": True, "bm": 32, "inventory_n": 5121}, False),
        ({"triton_group_major": True, "bm": 32, "inventory_k": 20480}, False),
        (
            {
                "triton_group_major": True,
                "bm": 32,
                "hip_wmma_library": Path("candidate.so"),
                "hip_wmma_sha256": "f" * 64,
            },
            False,
        ),
        ({"flydsl_wmma": True}, True),
        ({"flydsl_wmma": True, "flydsl_exact_code_cast": True}, True),
        ({"flydsl_exact_code_cast": True}, False),
        ({"flydsl_wmma": True, "flydsl_exact_code_cast": True, "split_k": 2}, False),
        ({"flydsl_wmma": True, "split_k": 8}, True),
        ({"flydsl_wmma": True, "flydsl_lds": True}, True),
        ({"flydsl_lds": True}, False),
        ({"flydsl_wmma": True, "flydsl_prepack": True}, True),
        ({"flydsl_prepack": True}, False),
        ({"flydsl_wmma": True, "flydsl_prepack": True, "flydsl_lds": True}, False),
        ({"hip_wmma_library": Path("candidate.so"), "hip_wmma_sha256": "f" * 64}, True),
        ({"hip_wmma_library": Path("candidate.so")}, False),
        ({"hip_wmma_sha256": "f" * 64}, False),
        (
            {"hip_wmma_library": Path("candidate.so"), "hip_wmma_sha256": "e" * 64},
            False,
        ),
        ({"hip_wmma_library": Path("candidate.so"), "flydsl_wmma": True}, False),
        ({"hip_wmma_library": Path("candidate.so"), "hip_row_chunk": 2}, False),
        ({"hip_wmma_library": Path("candidate.so"), "m": 1}, False),
        ({"flydsl_wmma": True, "m": 1}, False),
        ({"flydsl_wmma": True, "hip_row_chunk": 2}, False),
        ({"flydsl_wmma": True, "inventory_k": 129}, False),
        ({"flydsl_wmma": True, "inventory_k": 128, "split_k": 2}, False),
        ({"bk": 256}, False),
        ({"bk": 96}, False),
        ({"bm": 256, "bn": 256}, False),
        ({"bm": 16, "bn": 16, "warps": 16}, False),
        ({"rounds": 1}, False),
        ({"flush_mb": 0}, False),
    ],
)
def test_w4a16_plan_rejects_invalid_quantization_tiles_without_gpu(
    monkeypatch, tmp_path, overrides, valid
):
    """Planning must retain BF16 group boundaries and a cold-repeat protocol."""
    from benchmarks.kernels import benchmark_gfx1151_qwen_w4a16 as bench

    overrides = dict(overrides)
    inventory_k = overrides.pop("inventory_k", 17408)
    inventory_n = overrides.pop("inventory_n", 5120)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            dict(
                sources={"kernel.py": "f" * 64},
                snapshot="pinned-checkpoint",
                shapes=[
                    dict(
                        projection="down_proj",
                        n=inventory_n,
                        k=inventory_k,
                        modules=["layers.0.mlp.down_proj"],
                    )
                ],
            )
        )
    )
    monkeypatch.setattr(bench, "INVENTORY", inventory)
    monkeypatch.setattr(bench, "digest", lambda _: "f" * 64)
    settings = dict(
        projection="down_proj",
        m=8,
        split_k=1,
        hip_row_chunk=0,
        flydsl_wmma=False,
        flydsl_lds=False,
        flydsl_prepack=False,
        flydsl_exact_code_cast=False,
        triton_exact_code_cast=False,
        hip_wmma_library=None,
        hip_wmma_sha256=None,
        bm=16,
        bn=32,
        bk=128,
        warps=4,
        stages=2,
        rounds=7,
        repeats=15,
        flush_mb=128,
        seed=42,
    )
    settings.update(overrides)
    if not valid:
        with pytest.raises(AssertionError):
            bench.plan(SimpleNamespace(**settings))
    else:
        result = bench.plan(SimpleNamespace(**settings))
        assert result["executed"] is False
        assert result["baseline_dispatch"] == "Triton"
        assert result["activation_scale_dtype"] == "bfloat16"
        assert (result["m"], result["n"], result["k"]) == (
            settings["m"],
            inventory_n,
            inventory_k,
        )
        chunk = settings["hip_row_chunk"]
        registered = settings.get("registered_row_chunk", False)
        hip_wmma = settings["hip_wmma_library"] is not None
        assert result["candidate_native_calls"] == (
            2
            if registered
            else (1 if settings["split_k"] == 1 else 2)
            if hip_wmma
            else 8 // chunk
            if chunk
            else 0
        )
        assert result["candidate_dispatch"] == (
            "Triton group-major"
            if settings.get("triton_group_major", False)
            else "Registered HIP row chunks"
            if registered
            else "HIP WMMA"
            if hip_wmma
            else "FlyDSL WMMA"
            if settings["flydsl_wmma"]
            else "HIP row chunks"
            if chunk
            else "Triton"
        )
        assert bool(result["flydsl_sources"]) == settings["flydsl_wmma"]
        if settings["flydsl_wmma"] or hip_wmma:
            assert result["tile"] == dict(bm=16, bn=64, bk=128, warps=4, stages=1)
        assert result["flydsl_lds"] == settings["flydsl_lds"]
        assert result["flydsl_prepack"] == settings["flydsl_prepack"]
        assert result["flydsl_exact_code_cast"] == settings["flydsl_exact_code_cast"]
        assert result["triton_exact_code_cast"] == settings["triton_exact_code_cast"]


@pytest.mark.parametrize("m", [1, 8, 16, 33])
def test_triton_exact_int4_cast_preserves_codes_and_dirty_graph(m):
    """All16 codes, row tails, and fractional scales must preserve BF16 bits."""
    if not torch.cuda.is_available():
        pytest.skip("GPU correctness test")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1151":
        pytest.skip("gfx1151-only experiment")
    import triton

    from benchmarks.kernels.benchmark_gfx1151_qwen_w4a16 import group_major_operands
    from benchmarks.kernels.gfx1151_qwen_w4a16_tuning import split_k_w4a16
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
        pack_int4_exllama_shuffle,
    )

    torch.manual_seed(42)
    codes = (
        torch.arange(64, device="cuda", dtype=torch.int32)[:, None]
        + torch.arange(128, device="cuda", dtype=torch.int32)[None, :]
    ) % 16
    packed, scales = group_major_operands(
        pack_int4_exllama_shuffle(codes).contiguous(),
        torch.ones((64, 1), dtype=torch.bfloat16, device="cuda"),
    )
    activation = torch.eye(m, 128, dtype=torch.bfloat16, device="cuda")
    outputs = [
        torch.empty((m, 64), dtype=torch.bfloat16, device="cuda") for _ in range(2)
    ]
    expected = (codes[:, :m].t() - 8).to(torch.bfloat16)
    graphs = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for exact, output in zip((False, True), outputs):

            def run(output=output, exact=exact):
                split_k_w4a16[(triton.cdiv(m, 32), 2, 1)](
                    activation,
                    packed,
                    scales,
                    output,
                    m,
                    64,
                    128,
                    1,
                    32,
                    32,
                    128,
                    GROUP_MAJOR=True,
                    EXACT_CODE_CAST=exact,
                    num_warps=4,
                    num_stages=2,
                )

            run()
            stream.synchronize()
            assert torch.equal(output, expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run()
            output.fill_(float("nan"))
            graph.replay()
            stream.synchronize()
            assert torch.equal(output, expected)
            graphs.append(graph)
    torch.cuda.current_stream().wait_stream(stream)
    activation.copy_(torch.randn_like(activation))
    scales.uniform_(0.001, 0.02)
    for graph, output in zip(graphs, outputs):
        output.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(outputs[1]).all()
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize("lds,prepack", [(False, False), (True, False), (False, True)])
def test_flydsl_exact_int4_cast_preserves_all_codes_and_dirty_graph(lds, prepack):
    """Check the bounded conversion itself, then nonintegral activation math."""
    if not torch.cuda.is_available():
        pytest.skip("GPU correctness test")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1151":
        pytest.skip("gfx1151-only prototype")
    pytest.importorskip("flydsl")
    from benchmarks.kernels.gfx1151_qwen_w4a16_flydsl import make_runner
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
        pack_int4_exllama_shuffle,
    )

    torch.manual_seed(42)
    codes = (torch.arange(128, device="cuda", dtype=torch.int32) % 16).repeat(64, 1)
    packed = pack_int4_exllama_shuffle(codes).contiguous()
    scales = torch.ones((64, 1), dtype=torch.bfloat16, device="cuda")
    activation = torch.eye(16, 128, dtype=torch.bfloat16, device="cuda")
    outputs = [
        torch.empty((16, 64), dtype=torch.bfloat16, device="cuda") for _ in range(2)
    ]
    runners = [
        make_runner(
            activation,
            packed,
            scales,
            output,
            1,
            lds=lds,
            prepack=prepack,
            exact_code_cast=exact,
        )
        for exact, output in zip((False, True), outputs)
    ]
    expected = (
        (torch.arange(16, device="cuda") - 8).to(torch.bfloat16)[:, None].expand(16, 64)
    )
    graphs = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for runner, output in zip(runners, outputs):
            runner()
            stream.synchronize()
            assert torch.equal(output, expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                runner()
            output.fill_(float("nan"))
            graph.replay()
            stream.synchronize()
            assert torch.equal(output, expected)
            graphs.append(graph)
    torch.cuda.current_stream().wait_stream(stream)
    # Mutate activation storage, not packed weights/scales (prepack owns copies).
    activation.copy_(torch.randn_like(activation))
    for output, graph in zip(outputs, graphs):
        output.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(outputs[1]).all()
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize(
    "change,accepted",
    [
        ({}, True),
        ({"m": 1}, False),
        ({"m": 0}, False),
        ({"m": 3}, True),
        ({"m": 5}, True),
        ({"m": 6}, True),
        ({"m": 7}, True),
        ({"m": 33}, True),
        ({"m": 64}, True),
        ({"m": 65}, False),
        ({"x_dtype": torch.float16}, False),
        ({"group_size": 64}, False),
        ({"zp": True}, False),
        ({"bias": True}, False),
        ({"arch": False}, False),
        ({"x_device": "cpu"}, False),
        ({"group_device": "cuda:1"}, False),
        ({"x_contiguous": False}, False),
        ({"weight_contiguous": False}, False),
        ({"x_pointer": 4098}, False),
        ({"x_grad": True}, False),
        ({"group_shape": (136, 5120, 15)}, False),
        ({"scale_dtype": torch.float16}, False),
        ({"group_dtype": torch.int8}, False),
        ({"missing_cache": True}, False),
        ({"k": 6144, "m": 4}, False),
        ({"k": 6144, "m": 6}, True),
        ({"k": 5120}, False),
    ],
)
@pytest.mark.parametrize("production", [False, True])
def test_group_major_model_probe_contract_and_fallback(
    monkeypatch, change, accepted, production
):
    """Bad operands and native-skinny shapes must reach only the old dispatcher."""
    from benchmarks.kernels import gfx1151_qwen_w4a16_group_major as probe

    if production:
        from vllm.model_executor.kernels.linear.mixed_precision import (
            rdna_w4a16_group_major as probe,
        )

        monkeypatch.setenv("VLLM_GFX1151_W4_GROUP_MAJOR", "1")

    def tensor(
        shape, dtype, *, device="cuda:0", contiguous=True, pointer=4096, grad=False
    ):
        return SimpleNamespace(
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            device=torch.device(device),
            is_cuda=device.startswith("cuda"),
            requires_grad=grad,
            is_contiguous=lambda: contiguous,
            data_ptr=lambda: pointer,
        )

    monkeypatch.setattr(probe.original, "_on_gfx1151", lambda: change.get("arch", True))
    m, n, k = change.get("m", 8), 5120, change.get("k", 17408)
    x = tensor(
        (m, k),
        change.get("x_dtype", torch.bfloat16),
        device=change.get("x_device", "cuda:0"),
        contiguous=change.get("x_contiguous", True),
        pointer=change.get("x_pointer", 4096),
        grad=change.get("x_grad", False),
    )
    w = tensor(
        (n, k // 2), torch.int8, contiguous=change.get("weight_contiguous", True)
    )
    s = tensor((n, k // 128), change.get("scale_dtype", torch.bfloat16))
    gq = tensor(
        change.get("group_shape", (k // 128, n, 16)),
        change.get("group_dtype", torch.int32),
        device=change.get("group_device", "cuda:0"),
    )
    gs = tensor((k // 128, n), torch.bfloat16)
    if change.get("missing_cache"):
        gq = None
    zp = object() if change.get("zp") else None
    bias = object() if change.get("bias") else None
    group_size = change.get("group_size", 128)
    args = (x, w, s, zp, bias, 20, group_size, gq, gs)
    assert probe.can_use(x, w, s, zp, bias, group_size, gq, gs) is accepted
    if not accepted:
        expected = object()
        fallback = Mock(return_value=expected)
        monkeypatch.setattr(probe.original, "_rdna_hybrid_w4a16_apply_impl", fallback)
        assert probe.apply(*args) is expected
        fallback.assert_called_once_with(*args[:7])


def test_group_major_probe_install_restores_methods_and_preserves_loader_order(
    monkeypatch,
):
    from benchmarks.kernels import gfx1151_qwen_w4a16_group_major as probe

    events = []

    class Kernel:
        config = SimpleNamespace(group_size=128)

        def process_weights_after_loading(self, layer):
            events.append("original_load")

        def _get_weight_params(self, layer):
            return ("weights", "scales", None)

        def apply_weights(self, layer, x, bias=None):
            return x

    old_load, old_apply = Kernel.process_weights_after_loading, Kernel.apply_weights
    monkeypatch.setattr(probe.original, "RDNAHybridW4A16LinearKernel", Kernel)
    prepare = Mock(side_effect=lambda *args: events.append("prepare"))
    monkeypatch.setattr(probe, "prepare_layer", prepare)
    layer, x = SimpleNamespace(), object()
    probe.install()
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            probe.install()
        Kernel().process_weights_after_loading(layer)
        assert events == ["original_load", "prepare"]
        prepare.assert_called_once_with(layer, "weights", "scales", None, 128)
        assert Kernel().apply_weights(layer, x) is x
    finally:
        probe.uninstall()
    assert Kernel.process_weights_after_loading is old_load
    assert Kernel.apply_weights is old_apply


@pytest.mark.parametrize("suite", ["initial", "extended"])
def test_w4_model_probe_jobs_keep_matched_configurations(monkeypatch, tmp_path, suite):
    """Only the process-local hook and artifact locations may differ within a pair."""
    import ast
    import importlib

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    runner = importlib.import_module("run_w4_group_major_model_parity")
    jobs = list(runner.jobs(tmp_path, suite))
    modes = (
        [("dflash7_b1", 1, 128), ("target_b4", 4, 128)]
        if suite == "initial"
        else [
            ("dflash7_b2", 2, 128),
            ("dflash7_b4", 4, 128),
            ("dflash7_b8", 8, 128),
            ("dflash7_b1_long", 1, 1024),
        ]
    )
    assert [j["label"] for j in jobs] == [
        f"{mode}_{leg}" for mode, _, _ in modes for leg in ("baseline", "candidate")
    ]
    for (mode, batch, tokens), baseline, candidate in zip(
        modes, jobs[::2], jobs[1::2], strict=True
    ):
        assert baseline["mode"] == candidate["mode"] == mode
        assert baseline["batch"] == candidate["batch"] == batch
        assert baseline["output_tokens"] == candidate["output_tokens"] == tokens
        assert baseline["controls"] == candidate["controls"]
        assert baseline["controls"]["GFX1151_SERVING_NATIVE_VARIANT"] == "static-m1"
        assert baseline["controls"]["VLLM_DISABLE_COMPILE_CACHE"] == "1"
        assert baseline["controls"]["VLLM_USE_AOT_COMPILE"] == "1"
        assert baseline["controls"]["VLLM_CACHE_ROOT"] == str(
            tmp_path / "compile_cache"
        )
        assert "VLLM_GFX1151_W4_ROW_CHUNK" not in baseline["controls"]
        argv = []
        for job in (baseline, candidate):
            tree = ast.parse(job["command"][-1])
            compile(tree, "<model-probe-test>", "exec")
            assignment = next(
                n
                for n in tree.body
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Attribute)
                and n.targets[0].attr == "argv"
            )
            argv.append(ast.literal_eval(assignment.value))
        assert argv[0] == argv[1]
        assert argv[0][argv[0].index("--batch") + 1] == str(batch)
        assert argv[0][argv[0].index("--output-tokens") + 1] == str(tokens)
        assert "--enforce-eager" not in argv[0]
        assert "--disable-prefix-caching" in argv[0]
        assert "--require-live-batch" in argv[0]
        if mode.startswith("dflash"):
            assert argv[0][argv[0].index("--spec-tokens") + 1] == "7"


def test_group_major_production_jobs_share_cache_but_keep_flag_and_argv_explicit(
    monkeypatch, tmp_path
):
    """Cold/warm parity must exercise real production dispatch, not monkeypatch it."""
    import ast
    import importlib

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    runner = importlib.import_module("run_w4_group_major_production_parity")
    jobs = list(runner.jobs(tmp_path))
    assert [(j["enabled"], j["cache_state"]) for j in jobs] == [
        (0, "cold"),
        (1, "cold"),
        (0, "warm"),
        (1, "warm"),
    ]
    argvs = []
    for job in jobs:
        controls = job["controls"]
        assert controls["VLLM_GFX1151_W4_GROUP_MAJOR"] == str(job["enabled"])
        assert controls["VLLM_DISABLE_COMPILE_CACHE"] == "0"
        assert controls["VLLM_USE_AOT_COMPILE"] == "1"
        assert controls["VLLM_CACHE_ROOT"] == str(tmp_path / "compile_cache")
        assert controls["GFX1151_SERVING_NATIVE_VARIANT"] == "static-m1"
        command = job["command"][-1]
        assert "probe.install" not in command and "apply_weights" not in command
        tree = ast.parse(command)
        compile(tree, "<production-model-plan-test>", "exec")
        assignment = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Attribute)
            and n.targets[0].attr == "argv"
        )
        argvs.append(ast.literal_eval(assignment.value))
    assert all(argv == argvs[0] for argv in argvs)
    argv = argvs[0]
    assert argv[argv.index("--spec-tokens") + 1] == "7"
    assert argv[argv.index("--output-tokens") + 1] == "128"
    assert "--enforce-eager" not in argv
    assert "--disable-prefix-caching" in argv and "--require-live-batch" in argv


@pytest.mark.parametrize("production", [False, True])
def test_group_major_probe_fake_tensor_contract(production):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from benchmarks.kernels import gfx1151_qwen_w4a16_group_major  # noqa: F401

    if production:
        from vllm.model_executor.kernels.linear.mixed_precision import (
            rdna_w4a16_group_major,  # noqa: F401
        )

    with FakeTensorMode():
        x = torch.empty((8, 6144), device="cuda", dtype=torch.bfloat16)
        w = torch.empty((5120, 3072), device="cuda", dtype=torch.int8)
        s = torch.empty((5120, 48), device="cuda", dtype=torch.bfloat16)
        gq = torch.empty((48, 5120, 16), device="cuda", dtype=torch.int32)
        gs = torch.empty((48, 5120), device="cuda", dtype=torch.bfloat16)
        op = getattr(
            torch.ops.vllm,
            "gfx1151_w4_group_major" if production else "gfx1151_w4_group_major_probe",
        )
        result = op(x, w, s, None, None, 20, 128, gq, gs)
        assert result.shape == (8, 5120)
        assert result.dtype == x.dtype and result.device == x.device


def test_group_major_disabled_op_uses_original_even_with_cached_buffers(monkeypatch):
    from vllm.model_executor.kernels.linear.mixed_precision import (
        rdna_w4a16_group_major as production,
    )

    monkeypatch.setenv("VLLM_GFX1151_W4_GROUP_MAJOR", "0")
    args = tuple(object() for _ in range(9))
    expected = object()
    fallback = Mock(return_value=expected)
    monkeypatch.setattr(production.original, "_rdna_hybrid_w4a16_apply_impl", fallback)
    assert production.apply(*args) is expected
    fallback.assert_called_once_with(*args[:7])


@pytest.mark.parametrize("enabled,cache", [(False, True), (True, False), (True, True)])
def test_group_major_layer_dispatch_respects_flag_and_cache(
    monkeypatch, enabled, cache
):
    from vllm.model_executor.kernels.linear.mixed_precision import (
        rdna_w4a16_group_major as production,
    )
    from vllm.utils import platform_utils

    monkeypatch.setenv("VLLM_GFX1151_W4_GROUP_MAJOR", str(int(enabled)))
    monkeypatch.setattr(platform_utils, "num_compute_units", lambda: 20)
    x = torch.zeros((2, 3, 4))
    w, s = torch.zeros((5, 2)), torch.ones((5, 1))
    kernel = SimpleNamespace(
        config=SimpleNamespace(group_size=128),
        _get_weight_params=lambda layer: (w, s, None),
    )
    layer = SimpleNamespace()
    if cache:
        setattr(layer, production.Q_BUFFER, object())
        setattr(layer, production.S_BUFFER, object())
    original = Mock(return_value=torch.full((6, 5), 1.0))
    candidate = Mock(return_value=torch.full((6, 5), 2.0))
    monkeypatch.setattr(torch.ops.vllm, "rdna_hybrid_w4a16_apply", original)
    monkeypatch.setattr(torch.ops.vllm, "gfx1151_w4_group_major", candidate)
    result = production.original.RDNAHybridW4A16LinearKernel.apply_weights(
        kernel, layer, x
    )
    assert result.shape == (2, 3, 5)
    assert torch.all(result == (2.0 if enabled and cache else 1.0))
    assert candidate.call_count == int(enabled and cache)
    assert original.call_count == int(not (enabled and cache))


@pytest.mark.parametrize("offload", ["disabled", "uva", "prefetch"])
def test_group_major_prepare_skips_disabled_and_offloaded_models(monkeypatch, offload):
    from vllm.model_executor.kernels.linear.mixed_precision import (
        rdna_w4a16_group_major as production,
    )

    monkeypatch.setenv("VLLM_GFX1151_W4_GROUP_MAJOR", str(int(offload != "disabled")))
    config = SimpleNamespace(
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=int(offload == "uva")),
            prefetch=SimpleNamespace(offload_group_size=int(offload == "prefetch")),
        )
    )
    monkeypatch.setattr(production, "get_current_vllm_config_or_none", lambda: config)
    layer = torch.nn.Module()
    assert not production.prepare_layer(layer, None, None, None, 128)
    assert dict(layer.named_buffers()) == {}
    # An existing captured cache cannot be silently abandoned on reload.
    layer.register_buffer(production.Q_BUFFER, torch.zeros(1), persistent=False)
    with pytest.raises(RuntimeError, match="Cannot invalidate"):
        production.prepare_layer(layer, None, None, None, 128)


@pytest.mark.parametrize("malformed", ["missing", "shape", "dtype"])
def test_group_major_reload_rejects_bad_cache_before_mutating_either_buffer(
    monkeypatch, malformed
):
    from vllm.model_executor.kernels.linear.mixed_precision import (
        rdna_w4a16_group_major as production,
    )

    monkeypatch.setenv("VLLM_GFX1151_W4_GROUP_MAJOR", "1")
    monkeypatch.setattr(production, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(production, "eligible_weights", lambda *args: True)
    words = torch.zeros((2, 32), dtype=torch.int32)
    scales = torch.ones((2, 2), dtype=torch.bfloat16)
    layer = torch.nn.Module()
    assert production.prepare_layer(layer, words.view(torch.int8), scales, None, 128)
    q = getattr(layer, production.Q_BUFFER)
    if malformed == "missing":
        delattr(layer, production.S_BUFFER)
    elif malformed == "shape":
        setattr(layer, production.S_BUFFER, torch.empty(1, dtype=torch.bfloat16))
    else:
        setattr(layer, production.S_BUFFER, torch.empty((2, 2), dtype=torch.float16))
    with pytest.raises(RuntimeError, match="Incompatible captured"):
        production.prepare_layer(layer, (words + 1).view(torch.int8), scales, None, 128)
    assert torch.count_nonzero(q) == 0


@pytest.mark.parametrize("production", [False, True])
def test_w4a16_group_major_repacking_preserves_each_word_and_scale(production):
    """A layout-only optimization must not change nibble bits or group mapping."""
    from benchmarks.kernels import benchmark_gfx1151_qwen_w4a16 as bench

    if production:
        from vllm.model_executor.kernels.linear.mixed_precision import (
            rdna_w4a16_group_major as bench,
        )

    weights = torch.arange(-48, 48, dtype=torch.int32).reshape(2, 48)
    scales = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    packed, grouped_scales = bench.group_major_operands(weights, scales)
    assert packed.shape == (3, 2, 16)
    assert grouped_scales.shape == (3, 2)
    assert packed.is_contiguous() and grouped_scales.is_contiguous()
    for group in range(3):
        for row in range(2):
            assert torch.equal(
                packed[group, row], weights[row, group * 16 : (group + 1) * 16]
            )
            assert grouped_scales[group, row] == scales[row, group]
    assert torch.equal(packed.permute(1, 0, 2).reshape_as(weights), weights)
    assert torch.equal(grouped_scales.t(), scales)


@pytest.mark.parametrize("invalid", ["rank", "tail", "groups", "strides", "empty"])
@pytest.mark.parametrize("production", [False, True])
def test_w4a16_group_major_repacking_rejects_ambiguous_layouts(invalid, production):
    from benchmarks.kernels import benchmark_gfx1151_qwen_w4a16 as bench

    if production:
        from vllm.model_executor.kernels.linear.mixed_precision import (
            rdna_w4a16_group_major as bench,
        )

    weights = torch.zeros((2, 32), dtype=torch.int32)
    scales = torch.ones((2, 2), dtype=torch.bfloat16)
    if invalid == "rank":
        weights = weights.flatten()
    elif invalid == "tail":
        weights = torch.zeros((2, 17), dtype=torch.int32)
    elif invalid == "groups":
        scales = torch.ones((2, 3), dtype=torch.bfloat16)
    elif invalid == "strides":
        weights = torch.zeros((32, 2), dtype=torch.int32).t()
    else:
        weights = torch.empty((0, 32), dtype=torch.int32)
        scales = torch.empty((0, 2), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        bench.group_major_operands(weights, scales)


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_w4a16_capture_flag_restores_lazy_environment(present, fails):
    """A candidate capture must not leak its flag into baseline capture."""
    from benchmarks.kernels import benchmark_gfx1151_qwen_w4a16 as bench

    class Environment:
        def __getattr__(self, name):
            assert name == "VLLM_GFX1151_W4_ROW_CHUNK"
            return False

    env = Environment()
    if present:
        env.VLLM_GFX1151_W4_ROW_CHUNK = False

    def operation(value):
        assert env.VLLM_GFX1151_W4_ROW_CHUNK is True
        if fails:
            raise RuntimeError("capture failure")
        return value

    if fails:
        with pytest.raises(RuntimeError, match="capture failure"):
            bench.call_with_row_chunk_flag(env, True, operation, 42)
    else:
        assert bench.call_with_row_chunk_flag(env, True, operation, 42) == 42
    assert env.VLLM_GFX1151_W4_ROW_CHUNK is False
    assert ("VLLM_GFX1151_W4_ROW_CHUNK" in vars(env)) == present


@pytest.mark.parametrize("extra", [[], ["--execute"]])
def test_w4a16_cli_rejects_retired_production_variant(monkeypatch, capsys, extra):
    """Failed model parity must block the retired GPU workflow before planning."""
    from benchmarks.kernels import benchmark_gfx1151_qwen_w4a16 as bench

    monkeypatch.setattr(sys, "argv", ["benchmark", "--registered-row-chunk", *extra])
    planner = Mock(side_effect=AssertionError("retired variant reached planning"))
    monkeypatch.setattr(bench, "plan", planner)
    with pytest.raises(SystemExit) as exc:
        bench.main()
    assert exc.value.code == 2
    assert "rejected after graph-model parity" in capsys.readouterr().err
    planner.assert_not_called()


@pytest.mark.parametrize("chunk", [1, 2])
def test_w4a16_hip_row_chunks_assemble_every_output_row(chunk):
    """Keep each HIP call and output assembly inside the timed candidate."""
    from benchmarks.kernels import benchmark_gfx1151_qwen_w4a16 as bench

    class Rows:
        shape = (4, 17408)

        def __getitem__(self, key):
            return (key.start, key.stop)

    destinations = {}

    class Output:
        def __getitem__(self, key):
            destination = Mock()
            destinations[key.start, key.stop] = destination
            return destination

    output = Output()
    weights, scales = object(), object()
    native = Mock(side_effect=lambda w, rows, s, cu, gs: ("computed", rows))
    buffers = ["stale warmup buffer"]
    result = bench.hip_row_chunks(
        Rows(), weights, scales, output, 20, chunk, native, buffers
    )
    assert result is output
    assert native.call_count == 4 // chunk
    assert len(buffers) == 4 // chunk
    for index, start in enumerate(range(0, 4, chunk)):
        rows = (start, start + chunk)
        assert native.call_args_list[index].args == (weights, rows, scales, 20, 128)
        destinations[rows].copy_.assert_called_once_with(("computed", rows))
        assert buffers[index] == ("computed", rows)


@pytest.mark.parametrize("fail_replay", [False, True])
def test_w4a16_profile_scope_excludes_warmup_and_always_pauses(fail_replay):
    """Selected-region profiling must close even when a graph launch fails."""
    from benchmarks.kernels import benchmark_gfx1151_qwen_w4a16 as bench

    events = []
    roctx = SimpleNamespace(
        roctxProfilerResume=lambda _: events.append("resume") or 0,
        roctxProfilerPause=lambda _: events.append("pause") or 0,
    )

    def replay():
        events.append("replay")
        if fail_replay:
            raise RuntimeError("graph launch failed")

    graph = SimpleNamespace(replay=replay)
    synchronize = lambda: events.append("sync")
    if fail_replay:
        with pytest.raises(RuntimeError, match="graph launch failed"):
            bench.profile_graph(graph, roctx, synchronize, 3)
        assert events == ["sync", "resume", "replay", "pause"]
    else:
        bench.profile_graph(graph, roctx, synchronize, 3)
        assert events == [
            "sync",
            "resume",
            "replay",
            "replay",
            "replay",
            "sync",
            "pause",
        ]


def test_w4a16_execution_stops_at_gpu_ownership_guard(monkeypatch):
    """A busy GPU must fail before dependency loading or tensor construction."""
    from benchmarks.kernels import benchmark_gfx1151_qwen_w4a16 as bench

    guard = Mock(side_effect=RuntimeError("Existing ROCm client"))
    monkeypatch.setattr(
        bench.runpy,
        "run_path",
        lambda _: {
            "require_no_rocm_clients": guard,
        },
    )
    dependencies = Mock(side_effect=AssertionError("Must not load dependencies"))
    monkeypatch.setattr(bench.site, "addsitedir", dependencies)
    with pytest.raises(RuntimeError, match="Existing ROCm client"):
        bench.execute(SimpleNamespace(), {})
    guard.assert_called_once()
    dependencies.assert_not_called()


@pytest.mark.parametrize(
    "options,accepted",
    [
        ([], True),
        (["--draft"], False),
        (["--non-causal"], True),
        (["--window", "1"], True),
        (["--query-len", "1"], False),
        (["--prefix-wrapper-config", "16", "64", "4"], False),
    ],
)
def test_prefix_wrapper_probe_cli_preserves_math_and_rejects_bad_layouts(
    monkeypatch, tmp_path, options, accepted
):
    """The generalized wrapper forwards causal/window settings without coercion."""
    from benchmarks.kernels import benchmark_gfx1151_qwen_attention as bench

    kernel = Mock(return_value={"scope": "mocked_cpu"})
    monkeypatch.setattr(bench, "benchmark", kernel)
    monkeypatch.setattr(torch, "manual_seed", Mock())
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(gcnArchName="gfx1151"),
    )
    result = tmp_path / "probe.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--batch",
            "2",
            "--length",
            "1024",
            "--query-len",
            "8",
            "--prefix-wrapper-config",
            "16",
            "64",
            "4",
            "--output",
            str(result),
            *options,
        ],
    )
    if not accepted:
        with pytest.raises(SystemExit) as error:
            bench.main()
        assert error.value.code == 2
        kernel.assert_not_called()
        assert not result.exists()
    else:
        bench.main()
        kernel.assert_called_once()
        assert kernel.call_args.kwargs["prefix_wrapper_configs"] == [[16, 64, 4]]
        assert kernel.call_args.args[9] is ("--non-causal" not in options)
        assert kernel.call_args.args[10] == (1 if "--window" in options else 0)
        assert json.loads(result.read_text()) == {"scope": "mocked_cpu"}


@pytest.mark.parametrize(
    "kind,shape,stride,dtype",
    [
        ("state", (2, 48, 128, 128), (802816, 16384, 128, 1), torch.float32),
        ("conv", (2, 10240, 3), (1605632, 1, 10240), torch.bfloat16),
    ],
)
def test_gdn_benchmark_retains_live_cache_pitch_and_independent_values(
    kind, shape, stride, dtype
):
    from benchmarks.kernels.benchmark_gfx1151_qwen_gdn import allocate_cache

    source = torch.ones(shape, dtype=dtype)
    source[1].fill_(2)
    cache = allocate_cache(source, kind, live_layout=True)
    assert cache.stride() == stride
    assert torch.equal(cache, source)
    cache.zero_()
    assert source.count_nonzero() == source.numel()


def test_gdn_benchmark_rejects_unmeasured_cache_geometry():
    from benchmarks.kernels.benchmark_gfx1151_qwen_gdn import allocate_cache

    with pytest.raises(ValueError, match="captured M1 geometry"):
        allocate_cache(torch.empty(2, 10240, 4), "conv", live_layout=True)


def test_live_gdn_capture_preserves_strides_offset_and_independent_values():
    """A replay must retain projection layout, not silently benchmark a copy."""
    from benchmarks.kernels.diagnose_gfx1151_qwen_gdn import (
        pack_tensor,
        unpack_tensor,
    )

    source = torch.arange(48, dtype=torch.bfloat16).view(4, 12)[:, 3:9]
    expected = source.clone()
    record = pack_tensor(source)
    source.zero_()
    restored = unpack_tensor(record, "cpu")
    assert torch.equal(restored, expected)
    assert restored.stride() == source.stride()
    assert restored.storage_offset() == source.storage_offset()
    assert restored.data_ptr() % 16 == source.data_ptr() % 16
    restored.zero_()
    assert torch.equal(record["value"], expected)
    assert unpack_tensor(pack_tensor(None), "cpu") is None


def test_live_gdn_compaction_retains_null_slot_and_physical_conv_layout():
    """Compact active requests without changing DS/SD channel/token strides."""
    from benchmarks.kernels.diagnose_gfx1151_qwen_gdn import (
        active_slots,
        compact_tensor,
    )

    source = torch.arange(7 * 3 * 5).view(7, 3, 5).transpose(1, 2)
    indices = torch.tensor([6, 2], dtype=torch.int32)
    slots = active_slots(indices)
    compact = compact_tensor(source, slots)
    assert slots.tolist() == [0, 6, 2]
    assert compact.stride() == source.stride()
    assert torch.equal(compact, source[slots])
    compact.zero_()
    assert source[6].count_nonzero() > 0


@pytest.mark.parametrize("indices", [[], [0], [-1], [1, 1]])
def test_live_gdn_compaction_rejects_aliased_or_invalid_requests(indices):
    from benchmarks.kernels.diagnose_gfx1151_qwen_gdn import active_slots

    with pytest.raises(ValueError, match="unique, positive"):
        active_slots(torch.tensor(indices, dtype=torch.int32))


@pytest.mark.parametrize(
    "disabled",
    ["VLLM_GFX1151_QWEN_ATTENTION", "VLLM_GFX1151_QWEN_GDN"],
)
def test_live_gdn_capture_rejects_disabled_class_before_loading_model(
    monkeypatch, disabled
):
    from benchmarks.kernels.diagnose_gfx1151_qwen_gdn import (
        require_capture_environment,
    )

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    for key in ("VLLM_GFX1151_QWEN_ATTENTION", "VLLM_GFX1151_QWEN_GDN"):
        monkeypatch.setenv(key, "1")
    require_capture_environment()
    monkeypatch.setenv(disabled, "0")
    with pytest.raises(RuntimeError, match="explicit controls"):
        require_capture_environment()


@pytest.mark.parametrize("disable_hip", [False, True])
@pytest.mark.parametrize("disable_prefill", [False, True])
def test_ablation_disables_only_requested_paths_and_restores_on_failure(
    monkeypatch, disable_hip, disable_prefill
):
    """Controlled ablations must not enable kernels or leak into the next run."""
    from vllm.v1.attention.backends.gfx1151_qwen_attn import Gfx1151QwenAttentionImpl

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    cls = Gfx1151QwenAttentionImpl
    originals = {}
    for name in ("_can_use_hip", "_can_use_triton_prefill"):
        originals[name] = Mock(return_value=True)
        monkeypatch.setattr(cls, name, originals[name])
    with (
        pytest.raises(RuntimeError, match="body failed"),
        ablate_attention_paths(
            disable_hip=disable_hip, disable_prefill=disable_prefill
        ) as counts,
    ):
        for name, disabled in zip(originals, (disable_hip, disable_prefill)):
            assert getattr(cls, name)("sentinel", metadata=None) is not disabled
            assert originals[name].call_count == int(not disabled)
            if not disabled:
                originals[name].assert_called_once_with("sentinel", metadata=None)
        raise RuntimeError("body failed")
    assert counts == {
        name: 1
        for name, disabled in zip(originals, (disable_hip, disable_prefill))
        if disabled
    }
    for name, original in originals.items():
        assert getattr(cls, name) is original


@pytest.mark.parametrize("multiprocessing", [None, "1"])
def test_ablation_rejects_workers_that_would_not_inherit_hooks(
    monkeypatch, multiprocessing
):
    if multiprocessing is None:
        monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    else:
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", multiprocessing)
    with (
        pytest.raises(RuntimeError, match="single-process"),
        ablate_attention_paths(disable_hip=True),
    ):
        pytest.fail("Must fail before launching a model")


@pytest.fixture
def generation_pair():
    settings = dict(
        batch=1, output_tokens=4, model="target", revision="pinned", seed=42
    )
    rows = [
        dict(
            settings=settings.copy(),
            prompt_start=index,
            prompt_token_ids=[[10 + index]],
            warm_token_ids=[[1, 99, 2, 3]],
            profile_token_ids=[[1, 99, 2, 3]],
        )
        for index in range(2)
    ]
    return rows, deepcopy(rows)


@pytest.mark.parametrize("sizes", [None, [], [0], [2], [True], [1, 1], [1]])
def test_generation_audit_requires_actual_live_batch_evidence(generation_pair, sizes):
    from benchmarks.compare_gfx1151_qwen_generations import compare_generations

    baseline, candidate = generation_pair
    for row in baseline + candidate:
        row["settings"]["require_live_batch"] = True
        row["engine_steps"] = 1
        row["live_generated_batch_sizes"] = sizes
    if sizes == [1] and type(sizes[0]) is int:
        assert compare_generations(baseline, candidate)["live_batch_verified"]
    else:
        with pytest.raises(ValueError, match="live generation batch"):
            compare_generations(baseline, candidate)


@pytest.mark.parametrize("difference", [0, 2])
def test_generation_audit_never_waives_a_post_eos_difference(
    generation_pair, difference
):
    from benchmarks.compare_gfx1151_qwen_generations import compare_generations

    baseline, candidate = generation_pair
    candidate[1]["profile_token_ids"][0][difference] = 8
    candidate[1]["warm_token_ids"] = deepcopy(candidate[1]["profile_token_ids"])
    report = compare_generations(baseline, candidate, eos_token_ids=[99])
    assert report["baseline_self_exact"] == report["candidate_self_exact"] == 2
    assert report["forced_output_exact"] == 1
    assert report["through_eos_or_limit_exact"] == (2 if difference == 2 else 1)
    assert not report["strict_parity_passed"]
    assert report["prompts"][1]["first_difference"] == difference
    assert report["prompts"][1]["first_different_ids"] == [
        1 if difference == 0 else 2,
        8,
    ]


def test_generation_audit_requires_self_repeat_even_when_cross_outputs_match(
    generation_pair,
):
    from benchmarks.compare_gfx1151_qwen_generations import compare_generations

    baseline, candidate = generation_pair
    report = compare_generations(baseline, candidate)
    assert report["strict_parity_passed"]
    assert report["through_eos_or_limit_exact"] is None
    candidate[0]["warm_token_ids"][0][0] = 8
    report = compare_generations(baseline, candidate, eos_token_ids=[99])
    assert report["forced_output_exact"] == report["through_eos_or_limit_exact"] == 2
    assert report["candidate_self_exact"] == 1
    assert not report["strict_parity_passed"]


@pytest.mark.parametrize("batch", [2, 4, 8])
def test_generation_audit_maps_batched_outputs_to_corpus_indices(
    generation_pair, batch
):
    from benchmarks.compare_gfx1151_qwen_generations import compare_generations

    template = generation_pair[0][0]
    baseline = []
    for start in range(0, 8, batch):
        row = deepcopy(template)
        row["settings"]["batch"] = batch
        row["prompt_start"] = start
        row["prompt_token_ids"] = [[10 + i] for i in range(start, start + batch)]
        for name in ("warm_token_ids", "profile_token_ids"):
            row[name] = [[1, 99, 2, 3] for _ in range(batch)]
        baseline.append(row)
    candidate = deepcopy(baseline)
    assert compare_generations(baseline, candidate)["strict_parity_passed"]
    for name in ("warm_token_ids", "profile_token_ids"):
        candidate[-1][name][-1][2] = 8
    report = compare_generations(baseline, candidate, eos_token_ids=[99])
    assert report["count"] == report["through_eos_or_limit_exact"] == 8
    assert report["forced_output_exact"] == 7
    assert report["prompts"][-1]["prompt_index"] == 7
    assert report["prompts"][-1]["first_difference"] == 2
    assert not report["strict_parity_passed"]


@pytest.mark.parametrize(
    "defect, message",
    [
        ("empty", "No generation"),
        ("missing_prompt", "prompt counts"),
        ("incomplete_output", "Incomplete forced"),
        ("incomplete_batch", "Incomplete batch"),
        ("duplicate_prompt", "duplicate prompt"),
        ("different_prompt", "Prompt tokens differ"),
        ("mixed_settings", "Mixed settings"),
        ("different_settings", "settings differ"),
        ("negative_token", "Invalid token"),
        ("bool_token", "Invalid token"),
    ],
)
def test_generation_audit_rejects_unfair_or_incomplete_records(
    generation_pair, defect, message
):
    from benchmarks.compare_gfx1151_qwen_generations import compare_generations

    baseline, candidate = generation_pair
    if defect == "empty":
        candidate.clear()
    elif defect == "missing_prompt":
        candidate.pop()
    elif defect == "incomplete_output":
        candidate[0]["profile_token_ids"][0].pop()
    elif defect == "incomplete_batch":
        candidate[0]["profile_token_ids"].clear()
    elif defect == "duplicate_prompt":
        candidate[1]["prompt_start"] = 0
    elif defect == "different_prompt":
        candidate[1]["prompt_token_ids"][0][0] = 88
    elif defect == "mixed_settings":
        candidate[1]["settings"]["seed"] = 43
    elif defect == "different_settings":
        for row in candidate:
            row["settings"]["seed"] = 43
    else:
        candidate[1]["profile_token_ids"][0][0] = (
            -1 if defect == "negative_token" else True
        )
    with pytest.raises(ValueError, match=message):
        compare_generations(baseline, candidate, eos_token_ids=[99])


def test_generation_cli_rejects_jointly_truncated_corpus(
    generation_pair, tmp_path, monkeypatch
):
    import json

    from benchmarks.compare_gfx1151_qwen_generations import main

    paths = [tmp_path / "baseline.log", tmp_path / "candidate.log"]
    for path, rows in zip(paths, generation_pair):
        path.write_text(
            "\n".join("GENERATION_RESULTS " + json.dumps(row) for row in rows)
        )
    monkeypatch.setattr(
        "sys.argv", ["compare", *map(str, paths), "--expected-prompts", "4"]
    )
    with pytest.raises(ValueError, match="omit part of the expected corpus"):
        main()


def output(index, tokens, finished=False):
    return SimpleNamespace(
        request_id=f"profile-{index}",
        outputs=[SimpleNamespace(token_ids=tokens)],
        finished=finished,
    )


@pytest.fixture
def profile_fixture(monkeypatch):
    import benchmarks.profile_gfx1151_qwen as profiler

    events, queued = [], []

    def step():
        events.append("step")
        item = queued.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    engine = SimpleNamespace(
        step=step,
        has_unfinished_requests=lambda: bool(queued),
        add_request=Mock(),
    )

    def generate(prompts, params, **kwargs):
        # Match the real offline API's mutation that once emptied our trace.
        params.output_kind = RequestOutputKind.FINAL_ONLY
        return [output(i, [i + 4, i + 5], True) for i in range(len(prompts))]

    llm = SimpleNamespace(llm_engine=engine, generate=generate)
    roctx = SimpleNamespace(
        roctxProfilerResume=lambda _: events.append("resume") or 0,
        roctxProfilerPause=lambda _: events.append("pause") or 0,
    )
    monkeypatch.setattr(profiler.torch.cuda, "synchronize", lambda: None)
    return llm, roctx, queued, events


def test_decode_trace_waits_for_every_prefill_and_preserves_cumulative_params(
    profile_fixture,
):
    llm, roctx, queued, events = profile_fixture
    queued.extend(
        [
            [],
            [output(0, [4])],
            [output(1, [5])],
            [output(1, [5, 6], True)],
            [output(0, [4, 5], True)],
        ]
    )
    params = SamplingParams(max_tokens=2)
    result = profile_batch(
        llm, roctx, [{"prompt_token_ids": [1]}, {"prompt_token_ids": [2]}], params
    )
    assert params.output_kind == RequestOutputKind.CUMULATIVE
    assert events == ["step", "step", "step", "resume", "step", "step", "pause"]
    assert result["engine_steps"] == 2
    assert result["profiled_phase"] == "decode"
    assert result["warm_token_ids"] == result["profile_token_ids"] == [[4, 5], [5, 6]]


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_profile_proves_full_batch_generated_in_one_live_step(profile_fixture, batch):
    llm, roctx, queued, _ = profile_fixture
    queued.extend(
        [
            [output(i, [i + 4]) for i in range(batch)],
            [output(i, [i + 4, i + 5], True) for i in range(batch)],
        ]
    )
    result = profile_batch(
        llm,
        roctx,
        [{"prompt_token_ids": [i]} for i in range(batch)],
        SamplingParams(max_tokens=2),
        require_live_batch=True,
    )
    assert result["live_generated_batch_sizes"] == [batch]


def test_profile_rejects_sequential_generation_mislabeled_as_full_batch(
    profile_fixture,
):
    llm, roctx, queued, events = profile_fixture
    queued.extend(
        [
            [output(0, [4]), output(1, [5])],
            [output(0, [4, 5], True)],
            [output(1, [5, 6], True)],
        ]
    )
    with pytest.raises(RuntimeError, match="entire batch"):
        profile_batch(
            llm,
            roctx,
            [{"prompt_token_ids": [1]}, {"prompt_token_ids": [2]}],
            SamplingParams(max_tokens=2),
            require_live_batch=True,
        )
    assert events[-1] == "pause"


@pytest.mark.parametrize(
    "steps, message, traced",
    [
        ([[]], "Not every request completed prefill", False),
        ([[output(0, [4, 5], True)]], "No decode remains", False),
        ([[output(0, [4])], RuntimeError("decode failed")], "decode failed", True),
    ],
)
def test_decode_trace_rejects_empty_collection_and_pauses_on_failure(
    profile_fixture, steps, message, traced
):
    llm, roctx, queued, events = profile_fixture
    queued.extend(steps)
    with pytest.raises(RuntimeError, match=message):
        profile_batch(
            llm, roctx, [{"prompt_token_ids": [1]}], SamplingParams(max_tokens=2)
        )
    assert ("resume" in events) == traced
    assert ("pause" in events) == traced


def test_full_inference_trace_includes_prefill_steps(profile_fixture):
    llm, roctx, queued, events = profile_fixture
    queued.extend([[], [output(0, [4])], [output(0, [4, 5], True)]])
    result = profile_batch(
        llm,
        roctx,
        [{"prompt_token_ids": [1]}],
        SamplingParams(max_tokens=2),
        profile_prefill=True,
    )
    assert events == ["resume", "step", "step", "step", "pause"]
    assert result["engine_steps"] == 3
    assert result["profiled_phase"] == "prefill_and_decode"
    assert result["warm_token_ids"] == result["profile_token_ids"] == [[4, 5]]


def test_full_inference_trace_pauses_on_prefill_failure(profile_fixture):
    llm, roctx, queued, events = profile_fixture
    queued.append(RuntimeError("prefill failed"))
    with pytest.raises(RuntimeError, match="prefill failed"):
        profile_batch(
            llm,
            roctx,
            [{"prompt_token_ids": [1]}],
            SamplingParams(max_tokens=2),
            profile_prefill=True,
        )
    assert events == ["resume", "step", "pause"]


@pytest.mark.parametrize("enabled", [False, True])
def test_layout_recorder_preserves_views_deduplicates_and_restores(
    monkeypatch, enabled
):
    """The diagnostic observes strides without cloning tensors or persisting hooks."""
    from vllm.v1.attention.backends.gfx1151_qwen_attn import Gfx1151QwenAttentionImpl
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl

    calls, records = [], []

    def original(self, layer, query, key, value, cache, metadata, output):
        calls.append((query, key, value, cache, metadata, output))
        return output

    for cls in (RocmAttentionImpl, Gfx1151QwenAttentionImpl):
        monkeypatch.setattr(cls, "forward", original)
    impl = object.__new__(RocmAttentionImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 24, 4, 256
    impl.scale, impl.sliding_window = 0.0625, (-1, -1)
    q = torch.empty(2, 24, 512, dtype=torch.bfloat16)[..., :256]
    k = torch.empty(2, 4, 256, dtype=q.dtype)
    v = torch.empty(2, 8, 256, dtype=q.dtype)[:, 4:]
    out = torch.empty_like(q)
    cache = torch.empty(2, 2, 912, 1024, dtype=q.dtype)
    metadata = SimpleNamespace(
        causal=True,
        max_query_len=2,
        max_seq_len=1024,
        num_actual_tokens=2,
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([1024], dtype=torch.int32),
    )
    with (
        pytest.raises(RuntimeError, match="body failed"),
        record_attention_layouts(enabled, records.append),
    ):
        for _ in range(2):
            assert impl.forward(None, q, k, v, cache, metadata, out) is out
        raise RuntimeError("body failed")
    assert len(calls) == 2
    assert all(a is b for a, b in zip(calls[0], (q, k, v, cache, metadata, out)))
    assert RocmAttentionImpl.forward is original
    assert Gfx1151QwenAttentionImpl.forward is original
    assert len(records) == int(enabled)
    if enabled:
        record = records[0]
        assert record["query"]["stride"] == list(q.stride())
        assert record["value"]["stride"] == list(v.stride())
        assert record["value"]["storage_offset"] == v.storage_offset()
        assert record["key_cache"]["stride"] == [1867776, 233472, 7296, 8, 1]
        assert record["max_seq_len_bound"] == 1024
