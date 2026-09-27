# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import subprocess
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import urllib3

from ..utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.parametrize("enabled", [False, True])
def test_group_major_accuracy_plan_keeps_fixed_smoke_protocol(monkeypatch, enabled):
    """The new kernel must not silently alter the evaluation or reuse old outputs."""
    import importlib

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    runner = importlib.import_module("run_optimized_gsm8k")
    planned = runner.plan(configuration="reference-order", group_major=enabled)
    assert planned["count"] == 100 and planned["evaluation_scope"] == "smoke100"
    assert planned["controls"]["VLLM_GFX1151_W4_GROUP_MAJOR"] == str(int(enabled))
    assert planned["controls"]["VLLM_GFX1151_QWEN_ATTENTION"] == "1"
    assert planned["controls"]["VLLM_GFX1151_QWEN_GDN"] == "1"
    assert ("_w4_group_major" in planned["label"]) is enabled
    assert planned["label"] in planned["responses"]
    assert "--enforce-eager" in planned["server"]
    evaluator = planned["evaluator"]
    for option, expected in (
        ("--count", "100"),
        ("--max-tokens", "8192"),
        ("--concurrency", "4"),
    ):
        assert evaluator[evaluator.index(option) + 1] == expected
    assert len(planned["comparisons"]) == 2
    assert all("--smoke100" in c["command"] for c in planned["comparisons"])
    if enabled:
        with pytest.raises(AssertionError, match="reference-order"):
            runner.plan(configuration="legacy", group_major=True)


@pytest.mark.parametrize("defect", [None, "archive", "current", "history", "unknown"])
def test_group_major_history_resolver_rejects_unapproved_changes(monkeypatch, defect):
    import importlib

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    evidence = importlib.import_module("gfx1151_group_major_evidence")
    path = "vllm/envs.py"
    before, after = evidence.TRANSITIONS[path]
    expected = before
    archive = evidence.ARCHIVE_ROOT / path
    hashes = {path: after, str(archive): before}
    if defect == "archive":
        hashes[str(archive)] = "tampered"
    elif defect == "current":
        hashes[path] = "unexpected new source"
    elif defect == "history":
        expected = "unknown old source"
    elif defect == "unknown":
        path = "vllm/unapproved.py"
        hashes[path] = "changed"
    monkeypatch.setattr(evidence, "digest", lambda p: hashes[str(p)])
    if defect:
        with pytest.raises(AssertionError):
            evidence.historical_source(path, expected)
    else:
        assert evidence.historical_source(path, expected) == archive


@pytest.mark.parametrize("flag", [b"1", b"0", None])
def test_group_major_owned_worker_flag_must_match_without_exposing_env(
    monkeypatch, flag
):
    import importlib

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    evidence = importlib.import_module("gfx1151_group_major_evidence")
    identities = {pid: dict(pid=pid, pgid=111, start_ticks=42) for pid in (111, 222)}
    proof = dict(server=dict(process=identities[111]), gpu_clients=[identities[222]])
    raw = b"UNRELATED_SECRET=do-not-expose\0"
    if flag is not None:
        raw += b"VLLM_GFX1151_W4_GROUP_MAJOR=" + flag + b"\0"
    monkeypatch.setattr(Path, "read_bytes", lambda _: raw)
    if flag == b"1":
        assert evidence.verify_group_major_process_env(
            proof, identities.__getitem__
        ) == {"111": "1", "222": "1"}
    else:
        with pytest.raises(AssertionError, match="flag differs"):
            evidence.verify_group_major_process_env(proof, identities.__getitem__)


@pytest.fixture
def block_screen_modules(monkeypatch):
    """Import only the stdlib planning/ownership entry, without loading HIP."""
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "build"))
    return (
        importlib.import_module("run_dflash_block_screen"),
        importlib.import_module("gfx1151_block_screen_entry"),
    )


@pytest.mark.parametrize("width", [3, 7, 15, 23, 31])
def test_block_screen_width_comes_from_owned_pinned_server(block_screen_modules, width):
    _, entry = block_screen_modules
    config = dict(
        model=entry.DRAFT,
        revision=entry.DRAFT_REVISION,
        method="dflash",
        num_speculative_tokens=width,
    )
    server = {"command": ["serve", "--speculative-config", json.dumps(config)]}
    assert entry.speculative_tokens(server) == width


@pytest.mark.parametrize(
    "fault",
    ["model", "revision", "method", "bool", "width", "duplicate", "missing", "list"],
)
def test_block_screen_rejects_unpinned_or_ambiguous_width(block_screen_modules, fault):
    _, entry = block_screen_modules
    config = dict(
        model=entry.DRAFT,
        revision=entry.DRAFT_REVISION,
        method="dflash",
        num_speculative_tokens=7,
    )
    if fault in ("model", "revision", "method"):
        config[fault] = "wrong"
    elif fault in ("bool", "width"):
        config["num_speculative_tokens"] = True if fault == "bool" else 8
    command = [
        "serve",
        "--speculative-config",
        json.dumps([] if fault == "list" else config),
    ]
    if fault == "duplicate":
        command.extend(command[1:])
    elif fault == "missing":
        command.pop()
    with pytest.raises(ValueError):
        entry.speculative_tokens({"command": command})


def test_block_screen_plans_fresh_bracketed_controls(block_screen_modules):
    import shlex

    screen, entry = block_screen_modules
    root, jobs = screen.plans("unit_test")
    assert len(jobs) == len({j["identity"] for j in jobs}) == 18
    for round_id, alternatives in ((1, [3, 15, 23, 31]), (2, [31, 23, 15, 3])):
        order = [j["spec_tokens"] for j in jobs if j["round"] == round_id]
        assert order[::2] == [7] * 5 and order[1::2] == alternatives
    for job in jobs:
        command = job["command"]
        serve = shlex.split(entry.argument(command, "--serve-cmd"))
        assert entry.argument(serve, "--max-num-seqs") == "1"
        assert entry.argument(serve, "--kv-cache-memory-bytes") == str(16 * 1024**3)
        assert entry.speculative_tokens({"command": serve}) == job["spec_tokens"]
        assert job["controls"]["VLLM_GFX1151_QWEN_ATTENTION"] == "1"
        assert job["controls"]["VLLM_GFX1151_QWEN_GDN"] == "1"
        assert Path(job["directory"]).parent == root
    assert screen.plans("unit_test") == (root, jobs)
    with pytest.raises(ValueError):
        screen.plans("../escape")


def test_block_screen_capacity_plans_equal_cache_and_alternating_restarts(
    block_screen_modules,
):
    import shlex

    import run_dflash_capacity_screen as capacity

    _, entry = block_screen_modules
    root, jobs = capacity.plans("capacity_unit_test")
    assert [j["spec_tokens"] for j in jobs] == [7, 23, 23, 7, 7, 23]
    assert [j["round"] for j in jobs] == [1, 1, 2, 2, 3, 3]
    assert len({j["identity"] for j in jobs}) == 6
    normalized = []
    for job in jobs:
        serve = shlex.split(entry.argument(job["command"], "--serve-cmd"))
        assert entry.argument(serve, "--max-num-seqs") == "8"
        assert entry.argument(serve, "--kv-cache-memory-bytes") == str(48 * 1024**3)
        assert entry.speculative_tokens({"command": serve}) == job["spec_tokens"]
        assert job["controls"]["VLLM_GFX1151_QWEN_ATTENTION"] == "1"
        assert job["controls"]["VLLM_GFX1151_QWEN_GDN"] == "1"
        assert Path(job["directory"]).parent == root
        position = serve.index("--speculative-config") + 1
        config = json.loads(serve[position])
        config["num_speculative_tokens"] = 7
        serve[position] = json.dumps(config, sort_keys=True)
        normalized.append(serve)
    assert all(command == normalized[0] for command in normalized)
    assert capacity.plans("capacity_unit_test") == (root, jobs)
    with pytest.raises(ValueError):
        capacity.plans("../escape")


def test_block_screen_capacity_summary_keeps_restart_drift_visible(
    block_screen_modules,
):
    import run_dflash_capacity_screen as capacity

    _, jobs = capacity.plans("capacity_unit_test")
    results = {
        job["label"]: dict(
            output_throughput=20.0 if job["spec_tokens"] == 7 else 22.0,
            generated_texts=["same"] * 8,
            spec_decode_acceptance_length=6.0,
            spec_decode_acceptance_rate=70.0,
        )
        for job in jobs
    }
    results[jobs[3]["label"]]["output_throughput"] = 21.0
    results[jobs[2]["label"]]["generated_texts"][1] = "changed"
    report = capacity.summarize(jobs, results)
    assert [p["order"] for p in report["pairs"]] == [[7, 23], [23, 7], [7, 23]]
    assert report["pairs"][0]["throughput_gain_percent"] == pytest.approx(10)
    assert report["pairs"][1]["throughput_gain_percent"] == pytest.approx(100 / 21)
    assert report["pairs"][1]["text_differences"] == [1]
    assert report["restart_text_comparisons"]["23"][0]["text_differences"] == [1]
    assert report["throughput_ranges"]["7"]["restarts"] == 3
    assert report["throughput_ranges"]["7"]["range_over_median_percent"] == 5.0
    assert report["candidate_range_above_baseline"]
    assert not report["promoted"] and not report["accuracy_evaluated"]
    results[jobs[1]["label"]]["output_throughput"] = 21.0
    assert not capacity.summarize(jobs, results)["candidate_range_above_baseline"]


@pytest.mark.parametrize("fault", ["order", "kv", "seqs", "missing", "rate", "count"])
def test_block_screen_capacity_rejects_incomplete_or_mixed_contracts(
    block_screen_modules, fault
):
    import run_dflash_capacity_screen as capacity

    _, jobs = capacity.plans("capacity_unit_test")
    results = {
        job["label"]: dict(
            output_throughput=20.0,
            generated_texts=["same"] * 8,
            spec_decode_acceptance_length=6.0,
            spec_decode_acceptance_rate=70.0,
        )
        for job in jobs
    }
    if fault == "order":
        jobs[0]["spec_tokens"] = 23
    elif fault == "kv":
        jobs[0]["kv_cache_memory_bytes"] = 16 * 1024**3
    elif fault == "seqs":
        jobs[0]["max_num_seqs"] = 1
    elif fault == "missing":
        results.pop(jobs[0]["label"])
    elif fault == "rate":
        results[jobs[0]["label"]]["output_throughput"] = float("nan")
    else:
        results[jobs[0]["label"]]["generated_texts"].pop()
    with pytest.raises(AssertionError):
        capacity.summarize(jobs, results)


def test_block_screen_summary_exposes_control_drift_and_output_changes(
    block_screen_modules,
):
    screen, _ = block_screen_modules
    _, jobs = screen.plans("unit_test")
    results = {
        job["label"]: dict(
            output_throughput=20.0 if job["spec_tokens"] == 7 else 30.0,
            generated_texts=["same"] * 8,
            spec_decode_acceptance_length=6.0,
            spec_decode_acceptance_rate=70.0,
        )
        for job in jobs
    }
    results[jobs[2]["label"]]["output_throughput"] = 25.0
    results[jobs[2]["label"]]["generated_texts"][0] = "control changed"
    results[jobs[1]["label"]]["generated_texts"][1] = "candidate changed"
    report = screen.summarize(jobs, results)
    bracket = report["brackets"][0]
    assert bracket["gain_vs_faster_control_percent"] == pytest.approx(20)
    assert bracket["gain_vs_slower_control_percent"] == pytest.approx(50)
    assert bracket["control_range_over_median_percent"] == pytest.approx(100 * 5 / 22.5)
    assert bracket["control_text_differences"] == [0]
    assert bracket["candidate_vs_before_text_differences"] == [1]
    assert bracket["candidate_vs_after_text_differences"] == [0, 1]
    assert len(report["brackets"]) == 8
    assert report["throughput_ranges"]["7"] == dict(
        restarts=10,
        minimum=20.0,
        median=20.0,
        maximum=25.0,
    )
    assert report["throughput_ranges"]["3"]["restarts"] == 2
    assert not report["promoted"] and not report["accuracy_evaluated"]
    assert report["brackets"][4]["spec_tokens"] == 31
    assert report["brackets"][4]["round"] == 2
    with pytest.raises(AssertionError):
        screen.summarize(jobs[:-1], results)
    for invalid in (0.0, float("nan"), float("inf")):
        broken = deepcopy(results)
        broken[jobs[0]["label"]]["output_throughput"] = invalid
        with pytest.raises(AssertionError):
            screen.summarize(jobs, broken)
    broken = deepcopy(results)
    broken.pop(jobs[0]["label"])
    with pytest.raises(AssertionError):
        screen.summarize(jobs, broken)


def test_block_screen_refuses_unfinished_matrix_without_writes(
    block_screen_modules, tmp_path
):
    screen, _ = block_screen_modules
    with pytest.raises(RuntimeError, match="matrix is incomplete"):
        screen.require_finished_matrix(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_block_screen_execute_checks_matrix_before_launch_or_writes(
    block_screen_modules, tmp_path, monkeypatch
):
    import sys

    screen, _ = block_screen_modules
    monkeypatch.setattr(screen.serving, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["screen", "--execute"])

    def unexpected_launch(*args, **kwargs):
        pytest.fail("Incomplete matrix must prevent GPU checks and launch")

    monkeypatch.setattr(
        screen.serving.guards, "require_no_rocm_clients", unexpected_launch
    )
    monkeypatch.setattr(screen.subprocess, "Popen", unexpected_launch)
    with pytest.raises(RuntimeError, match="matrix is incomplete"):
        screen.main()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("width", [3, 7, 31])
@pytest.mark.parametrize("server_changed", [False, True])
def test_block_screen_native_audit_uses_actual_width(
    block_screen_modules,
    native_serving_result,
    tmp_path,
    monkeypatch,
    width,
    server_changed,
):
    """Exercise serialization/auditing without a server, torch, or GPU work."""
    import shlex
    import sys

    screen, entry = block_screen_modules
    job = next(j for j in screen.plans("unit_test")[1] if j["spec_tokens"] == width)
    serve = shlex.split(entry.argument(job["command"], "--serve-cmd"))
    argv = shlex.split(entry.argument(job["command"], "--bench-cmd"))
    argv = argv[argv.index(screen.ENTRY) :]
    output = tmp_path / "benchmarks/results/screen"
    output.mkdir(parents=True)
    argv.extend(
        [
            "--max-concurrency",
            "1",
            "--num-prompts",
            "8",
            "--result-dir",
            str(output),
            "--result-filename",
            "run=0.json",
        ]
    )
    proof = dict(
        server=dict(
            command=serve, capacity_requirement=dict(model=screen.serving.MODEL)
        )
    )
    monkeypatch.setattr(entry.entry, "ROOT", tmp_path)
    after = deepcopy(proof)
    if server_changed:
        after["server"]["replacement"] = True
    snapshots = iter((proof, after))
    monkeypatch.setattr(entry.entry, "verify_server", lambda: next(snapshots))
    monkeypatch.setattr(
        entry.entry, "verify_capacity", lambda _: dict(running=0, waiting=0)
    )
    monkeypatch.setattr(sys, "argv", argv)
    result = deepcopy(native_serving_result)
    result.update(
        num_prompts=8,
        completed=8,
        input_lens=[1024] * 8,
        output_lens=[1024] * 8,
        reported_output_lens=[1024] * 8,
        total_input_tokens=8192,
        total_output_tokens=8192,
        generated_texts=["answer"] * 8,
        errors=[""] * 8,
        ttfts=[0.1] * 8,
        latencies=[1.0] * 8,
        itls=[[] for _ in range(8)],
        duration=8.0,
        output_throughput=1024.0,
        request_throughput=1.0,
        total_token_throughput=2048.0,
    )
    drafts = 8192 // (width + 1)
    before = dict(
        num_drafts=0,
        num_draft_tokens=0,
        num_accepted_tokens=0,
        accepted_per_pos={str(i): 0 for i in range(width)},
    )
    after = dict(
        num_drafts=drafts,
        num_draft_tokens=drafts * width,
        num_accepted_tokens=drafts * width,
        accepted_per_pos={str(i): drafts for i in range(width)},
    )
    result.update(
        spec_decode_metrics_before=before,
        spec_decode_metrics_after=after,
        spec_decode_num_drafts=drafts,
        spec_decode_draft_tokens=drafts * width,
        spec_decode_accepted_tokens=drafts * width,
        spec_decode_acceptance_rate=100.0,
        spec_decode_acceptance_length=width + 1,
        spec_decode_per_position_acceptance_rates=[1.0] * width,
    )
    native_path = output / "run=0.json"
    monkeypatch.setattr(
        entry.runpy,
        "run_module",
        lambda *a, **kw: native_path.write_text(json.dumps(result)),
    )
    if server_changed:
        with pytest.raises(AssertionError, match="Server changed during timing"):
            entry.benchmark()
        assert not (output / "run=0.native.json").exists()
        return
    entry.benchmark()
    audit = json.loads((output / "run=0.native-audit.json").read_text())
    assert audit["passed"] and audit["expected_spec_tokens"] == width
    assert (output / "run=0.native.json").read_bytes() == native_path.read_bytes()


@pytest.fixture
def native_serving_result():
    """Small native-result schema with independently consistent counter deltas."""
    before = dict(
        num_drafts=10,
        num_draft_tokens=70,
        num_accepted_tokens=30,
        accepted_per_pos=dict(enumerate([8, 7, 5, 4, 3, 2, 1])),
    )
    delta = [2, 1, 1, 1, 1, 0, 0]
    before["accepted_per_pos"] = {
        str(k): v for k, v in before["accepted_per_pos"].items()
    }
    after = dict(
        num_drafts=12,
        num_draft_tokens=84,
        num_accepted_tokens=36,
        accepted_per_pos={
            str(i): before["accepted_per_pos"][str(i)] + n for i, n in enumerate(delta)
        },
    )
    return dict(
        num_prompts=2,
        completed=2,
        failed=0,
        max_concurrency=1,
        input_lens=[6, 6],
        total_input_tokens=12,
        output_lens=[4, 4],
        reported_output_lens=[4, 4],
        total_output_tokens=8,
        errors=["", ""],
        generated_texts=["#### 1", "#### 2"],
        ttfts=[0.01, 0.01],
        latencies=[0.04, 0.04],
        itls=[[0.01] * 3] * 2,
        duration=0.1,
        output_throughput=80.0,
        request_throughput=20.0,
        total_token_throughput=200.0,
        spec_decode_metrics_before=before,
        spec_decode_metrics_after=after,
        spec_decode_num_drafts=2,
        spec_decode_draft_tokens=14,
        spec_decode_accepted_tokens=6,
        spec_decode_acceptance_rate=100 * 6 / 14,
        spec_decode_acceptance_length=4.0,
        spec_decode_per_position_acceptance_rates=[n / 2 for n in delta],
    )


@pytest.mark.parametrize("spec_tokens", [0, 7])
def test_native_serving_audit_preserves_valid_result(
    native_serving_result, spec_tokens
):
    from benchmarks.audit_gfx1151_serving import audit_result

    original = deepcopy(native_serving_result)
    report = audit_result(native_serving_result, [6, 6], 4, 1, True, spec_tokens)
    assert report["passed"] and not report["errors"]
    assert (report["acceptance"] is None) == (spec_tokens == 0)
    assert native_serving_result == original


@pytest.fixture
def native_serving_restarts(native_serving_result):
    """Different durations with consistent native throughput accounting."""
    runs = []
    for duration in (0.10, 0.11, 0.12, 0.05, 0.055, 0.06):
        run = deepcopy(native_serving_result)
        run.update(
            duration=duration,
            output_throughput=8 / duration,
            request_throughput=2 / duration,
            total_token_throughput=20 / duration,
        )
        for metric in ("ttft", "tpot", "itl", "e2el"):
            for q in ("median", "p90", "p99"):
                run[f"{q}_{metric}_ms"] = duration * 1000
        runs.append(run)
    return runs[:3], runs[3:]


def test_native_serving_restarts_report_spread_without_pooling(native_serving_restarts):
    from benchmarks.compare_gfx1151_serving import compare_restarts

    baseline, candidate = native_serving_restarts
    original = deepcopy((baseline, candidate))
    report = compare_restarts(baseline, candidate, [6, 6], 4, 1, True)
    throughput = report["metrics"]["output_throughput"]
    assert throughput["median_change_percent"] == pytest.approx(100)
    assert throughput["observed_ranges_disjoint"]
    assert throughput["baseline"]["minimum"] == pytest.approx(8 / 0.12)
    assert throughput["baseline"]["maximum"] == pytest.approx(80)
    assert report["metrics"]["p99_ttft_ms"]["candidate"]["median"] == 55
    assert (baseline, candidate) == original


@pytest.mark.parametrize("fault", ["partial", "bad_usage", "missing_p99", "nan"])
def test_native_serving_restarts_reject_incomplete_evidence(
    native_serving_restarts, fault
):
    from benchmarks.compare_gfx1151_serving import compare_restarts

    baseline, candidate = native_serving_restarts
    if fault == "partial":
        candidate.pop()
    elif fault == "bad_usage":
        candidate[0]["reported_output_lens"] = [0, 0]
    elif fault == "missing_p99":
        del candidate[0]["p99_ttft_ms"]
    else:
        candidate[0]["p99_ttft_ms"] = float("nan")
    with pytest.raises(ValueError):
        compare_restarts(baseline, candidate, [6, 6], 4, 1, True)


def test_native_serving_restarts_expose_overlapping_ranges(native_serving_restarts):
    from benchmarks.compare_gfx1151_serving import compare_restarts

    baseline, candidate = native_serving_restarts
    candidate[0] = deepcopy(baseline[0])
    candidate[0]["generated_texts"][0] = "different reasoning"
    report = compare_restarts(baseline, candidate, [6, 6], 4, 1, True)
    assert not report["metrics"]["output_throughput"]["observed_ranges_disjoint"]
    assert report["paired_outputs"][0]["exact_texts"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "reused_server",
        "changed_native",
        "sampling",
        "foreign_group",
        "undersized",
        "preemption",
        "changed_server_settings",
    ],
)
def test_native_serving_restart_cli_checks_saved_provenance(
    native_serving_restarts, tmp_path, monkeypatch, capsys, fault
):
    import sys

    from benchmarks.compare_gfx1151_serving import main

    paths = []
    baseline, candidate = native_serving_restarts
    for i, result in enumerate(baseline + candidate):
        path = tmp_path / f"run{i}.json"
        paths.append(str(path))
        result.update(
            model_id="target",
            tokenizer_id="pinned-tokenizer",
            backend="openai",
            request_rate="inf",
            burstiness=1,
        )
        path.with_suffix(".native.json").write_text(json.dumps(result))
        if fault == "changed_native" and i == 5:
            result["output_throughput"] += 1
        path.write_text(json.dumps(result))
        process = dict(pid=100 + i, pgid=100 + i, start_ticks=1000 + i)
        if fault == "reused_server" and i == 5:
            process = dict(pid=100, pgid=100, start_ticks=1000)
        worker = process | {"pid": 1000 + i}
        if fault == "foreign_group" and i == 5:
            worker["pgid"] = 999
        identity = dict(
            server=dict(
                process=process,
                binary_sha256="a" * 64,
                controls={
                    "VLLM_GFX1151_QWEN_ATTENTION": "0" if i < 3 else "1",
                    "VLLM_GFX1151_QWEN_GDN": "0" if i < 3 else "1",
                },
                command=["entry", "serve", "target"],
                capacity_requirement=dict(
                    model="target",
                    max_num_seqs=8,
                    max_model_len=4096,
                    kv_cache_memory_bytes=16 * 1024**3,
                ),
            ),
            capacity=dict(
                required_sequences=8,
                max_context_capacity=9.5,
                kv_cache_memory_bytes=16 * 1024**3,
                num_gpu_blocks=1008,
                block_size=832,
                mamba_block_size=4096,
                preemptions=0,
            ),
            gpu_clients=[worker],
            listener_owners=[process],
            command=[
                "entry",
                "bench",
                "serve",
                "--temperature",
                "1" if fault == "sampling" and i == 5 else "0",
                "--result-dir",
                str(tmp_path),
                "--result-filename",
                path.name,
            ],
        )
        if i == 5:
            if fault == "undersized":
                identity["capacity"]["max_context_capacity"] = 4.75
            if fault == "preemption":
                identity["capacity"]["preemptions"] = 1
            if fault == "changed_server_settings":
                identity["server"]["command"].append("--enforce-eager")
        path.with_suffix(".identity.json").write_text(json.dumps(identity))
        path.with_suffix(".capacity-after.json").write_text(json.dumps(identity))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(prompt_token_lengths=[6, 6], output_budget=4)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--require-capacity",
            "--baseline",
            *paths[:3],
            "--candidate",
            *paths[3:],
            "--concurrency",
            "1",
            "--realistic-manifest",
            str(manifest),
        ],
    )
    if fault is not None:
        with pytest.raises(ValueError):
            main()
    else:
        main()
        report = json.loads(capsys.readouterr().out)
        assert len(report["server_identities"]) == 6
        assert len(report["artifact_sha256"]) == 25


@pytest.mark.parametrize(
    "fault",
    [
        "missing_usage",
        "zero_usage",
        "wrong_usage",
        "incomplete",
        "failed",
        "errors",
        "input_lengths",
        "output_total",
        "throughput",
        "nan_duration",
        "negative_latency",
        "missing_before",
        "missing_after",
        "missing_position",
        "reset",
        "zero_drafts",
        "position_reset",
        "position_sum",
        "wrong_rate",
        "wrong_position_rate",
        "budget",
        "concurrency",
        "short_array",
    ],
)
def test_native_serving_audit_rejects_bad_evidence(native_serving_result, fault):
    from benchmarks.audit_gfx1151_serving import audit_result

    r = native_serving_result
    after = r["spec_decode_metrics_after"]
    if fault in ("missing_usage", "zero_usage", "wrong_usage"):
        r["reported_output_lens"][0] = {
            "missing_usage": None,
            "zero_usage": 0,
            "wrong_usage": 3,
        }[fault]
    elif fault == "incomplete":
        r["completed"] = 1
    elif fault == "failed":
        r["failed"] = 1
    elif fault == "errors":
        r["errors"][0] = "HTTP 500"
    elif fault == "input_lengths":
        r["input_lens"][0] = 5
    elif fault == "output_total":
        r["total_output_tokens"] = 7
    elif fault == "throughput":
        r["output_throughput"] = 100.0
    elif fault == "nan_duration":
        r["duration"] = float("nan")
    elif fault == "negative_latency":
        r["latencies"][0] = -1
    elif fault == "missing_before":
        r["spec_decode_metrics_before"] = None
    elif fault == "missing_after":
        r["spec_decode_metrics_after"] = None
    elif fault == "missing_position":
        del after["accepted_per_pos"]["6"]
    elif fault == "reset":
        after["num_accepted_tokens"] = 0
    elif fault == "zero_drafts":
        after["num_drafts"] = 10
    elif fault == "position_reset":
        after["accepted_per_pos"]["6"] = 0
    elif fault == "position_sum":
        after["accepted_per_pos"]["6"] += 1
    elif fault == "wrong_rate":
        r["spec_decode_acceptance_rate"] = 0.5
    elif fault == "wrong_position_rate":
        r["spec_decode_per_position_acceptance_rates"][0] = 0
    elif fault == "budget":
        r["output_lens"][0] = 3
    elif fault == "concurrency":
        r["max_concurrency"] = 2
    elif fault == "short_array":
        r["ttfts"].pop()
    report = audit_result(r, [6, 6], 4, 1, True, 7)
    assert not report["passed"] and report["errors"]


def test_native_serving_audit_realistic_lengths_are_not_forced(native_serving_result):
    from benchmarks.audit_gfx1151_serving import audit_result

    report = audit_result(native_serving_result, [6, 6], 3072, 1, False, 7)
    assert report["passed"] and report["output_budget_hits"] == 0
    assert "finish reasons" in report["limitation"]


def test_native_serving_audit_keeps_measured_zero_acceptance(native_serving_result):
    from benchmarks.audit_gfx1151_serving import audit_result

    result = native_serving_result
    before = result["spec_decode_metrics_before"]
    after = result["spec_decode_metrics_after"]
    after["num_accepted_tokens"] = before["num_accepted_tokens"]
    after["accepted_per_pos"] = before["accepted_per_pos"].copy()
    result.update(
        spec_decode_accepted_tokens=0,
        spec_decode_acceptance_rate=0.0,
        spec_decode_acceptance_length=1.0,
        spec_decode_per_position_acceptance_rates=[0.0] * 7,
    )
    report = audit_result(result, [6, 6], 4, 1, True, 7)
    assert report["passed"]
    assert report["acceptance"]["spec_decode_acceptance_rate"] == 0


@pytest.mark.parametrize("missing_usage", [0, None])
@pytest.mark.parametrize(
    "snapshots",
    ["absent", "missing_before", "missing_after", "zero", "increasing", "reset"],
)
def test_bench_serve_preserves_measurement_evidence(
    monkeypatch, missing_usage, snapshots
):
    """Audit fields preserve endpoint counts and pre/post-warmup boundaries."""
    from vllm.benchmarks import serve
    from vllm.benchmarks.datasets import SampleRequest
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput

    request_count = 0
    snapshot_request_counts = []
    before = serve.SpecDecodeMetrics(10, 70, 30, {0: 10, 1: 8})
    after = serve.SpecDecodeMetrics(12, 84, 36, {0: 12, 1: 9})
    if snapshots == "absent":
        before = after = None
    elif snapshots == "missing_before":
        before = None
    elif snapshots == "missing_after":
        after = None
    elif snapshots == "zero":
        before = after = serve.SpecDecodeMetrics(0, 0, 0, {})
    elif snapshots == "reset":
        before, after = after, before
    metric_snapshots = iter((before, after))

    async def request(request_func_input, **kwargs):
        nonlocal request_count
        request_count += 1
        return RequestFuncOutput(
            generated_text="one two three",
            success=True,
            prompt_len=request_func_input.prompt_len,
            output_tokens=missing_usage if request_count == 3 else 4,
            latency=0.03,
            ttft=0.01,
            itl=[0.01, 0.01],
            start_time=time.perf_counter(),
        )

    async def fetch_spec(*args):
        snapshot_request_counts.append(request_count)
        return next(metric_snapshots)

    async def no_diffusion(*args):
        return None

    async def forbid_http(*args, **kwargs):
        pytest.fail("CPU evidence test must not make network requests")

    monkeypatch.setitem(serve.ASYNC_REQUEST_FUNCS, "openai", request)
    monkeypatch.setattr(serve, "fetch_spec_decode_metrics", fetch_spec)
    monkeypatch.setattr(serve, "fetch_diffusion_metrics", no_diffusion)
    monkeypatch.setattr(serve.aiohttp.ClientSession, "_request", forbid_http)
    result = asyncio.run(
        serve.benchmark(
            task_type=serve.TaskType.GENERATION,
            endpoint_type="openai",
            api_url="http://unused/v1/completions",
            base_url="http://unused",
            model_id="test-model",
            model_name="test-model",
            tokenizer=lambda *args, **kwargs: SimpleNamespace(input_ids=[1, 2, 3]),
            input_requests=[SampleRequest("prompt", 6, 4) for _ in range(2)],
            logprobs=None,
            request_rate=float("inf"),
            burstiness=1.0,
            disable_tqdm=True,
            num_warmups=1,
            profile=False,
            selected_percentile_metrics=[],
            selected_percentiles=[],
            ignore_eos=False,
            goodput_config_dict={},
            max_concurrency=1,
            lora_modules=None,
            extra_headers=None,
            extra_body=None,
            ready_check_timeout_sec=0,
        )
    )
    assert request_count == 3  # One warmup, then two measured requests.
    assert snapshot_request_counts == [1, 3]
    assert result["completed"] == 2
    assert result["failed"] == 0
    assert result["output_lens"] == [4, 3]  # Existing tokenizer fallback retained.
    assert result["reported_output_lens"] == [4, missing_usage]
    assert result["total_output_tokens"] == 7
    for name, snapshot in (("before", before), ("after", after)):
        expected = None if snapshot is None else serve.asdict(snapshot)
        assert result[f"spec_decode_metrics_{name}"] == expected
    if snapshots == "increasing":
        assert result["spec_decode_num_drafts"] == 2
        assert result["spec_decode_draft_tokens"] == 14
        assert result["spec_decode_accepted_tokens"] == 6
    else:
        # Raw zero/missing/reset snapshots survive even when no delta statistics
        # are emitted. Absence must never be presented as zero acceptance.
        assert "spec_decode_acceptance_rate" not in result


def generate_self_signed_cert(cert_dir: Path) -> tuple[Path, Path]:
    """Generate a self-signed certificate for testing."""
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    # Generate self-signed certificate using openssl
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_file),
            "-out",
            str(cert_file),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    return cert_file, key_file


class RemoteOpenAIServerSSL(RemoteOpenAIServer):
    """RemoteOpenAIServer subclass that supports SSL with self-signed certs."""

    @property
    def url_root(self) -> str:
        return f"https://{self.host}:{self.port}"

    def _wait_for_server(self, *, url: str, timeout: float):
        """Override to use HTTPS with SSL verification disabled."""
        # Suppress InsecureRequestWarning for self-signed certs
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        start = time.time()
        while True:
            try:
                if requests.get(url, verify=False).status_code == 200:
                    break
            except Exception:
                result = self._poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError("Server failed to start in time.") from None


@pytest.fixture(scope="function")
def server():
    args = ["--max-model-len", "1024", "--enforce-eager", "--load-format", "dummy"]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="function")
def ssl_server():
    """Start a vLLM server with SSL enabled using a self-signed certificate."""
    with tempfile.TemporaryDirectory() as cert_dir:
        cert_file, key_file = generate_self_signed_cert(Path(cert_dir))
        args = [
            "--max-model-len",
            "1024",
            "--enforce-eager",
            "--load-format",
            "dummy",
            "--ssl-certfile",
            str(cert_file),
            "--ssl-keyfile",
            str(key_file),
        ]

        with RemoteOpenAIServerSSL(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest.mark.benchmark
def test_bench_serve(server):
    # Test default model detection and input/output len
    command = [
        "vllm",
        "bench",
        "serve",
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "5",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.mark.benchmark
def test_bench_serve_insecure(ssl_server):
    """Test --insecure flag with an HTTPS server using a self-signed certificate."""
    base_url = f"https://{ssl_server.host}:{ssl_server.port}"
    command = [
        "vllm",
        "bench",
        "serve",
        "--base-url",
        base_url,
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "5",
        "--insecure",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.mark.benchmark
def test_bench_serve_chat(server):
    command = [
        "vllm",
        "bench",
        "serve",
        "--model",
        MODEL_NAME,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--dataset-name",
        "random",
        "--random-input-len",
        "32",
        "--random-output-len",
        "4",
        "--num-prompts",
        "5",
        "--endpoint",
        "/v1/chat/completions",
        "--backend",
        "openai-chat",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
