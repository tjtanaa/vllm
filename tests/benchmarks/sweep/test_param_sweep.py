# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import tempfile
from pathlib import Path

import pytest

from vllm.benchmarks.sweep.param_sweep import ParameterSweep, ParameterSweepItem


class TestParameterSweepItem:
    """Test ParameterSweepItem functionality."""

    @pytest.mark.parametrize(
        "input_dict,expected",
        [
            (
                {"compilation_config.use_inductor_graph_partition": False},
                "--compilation-config.use_inductor_graph_partition=false",
            ),
            (
                {"compilation_config.use_inductor_graph_partition": True},
                "--compilation-config.use_inductor_graph_partition=true",
            ),
        ],
    )
    def test_nested_boolean_params(self, input_dict, expected):
        """Test that nested boolean params use =true/false syntax."""
        item = ParameterSweepItem.from_record(input_dict)
        cmd = item.apply_to_cmd(["vllm", "serve", "model"])
        assert expected in cmd

    @pytest.mark.parametrize(
        "input_dict,expected",
        [
            ({"enable_prefix_caching": False}, "--no-enable-prefix-caching"),
            ({"enable_prefix_caching": True}, "--enable-prefix-caching"),
            ({"disable_log_stats": False}, "--no-disable-log-stats"),
            ({"disable_log_stats": True}, "--disable-log-stats"),
        ],
    )
    def test_non_nested_boolean_params(self, input_dict, expected):
        """Test that non-nested boolean params use --no- prefix."""
        item = ParameterSweepItem.from_record(input_dict)
        cmd = item.apply_to_cmd(["vllm", "serve", "model"])
        assert expected in cmd

    @pytest.mark.parametrize(
        "compilation_config",
        [
            {"cudagraph_mode": "full", "mode": 2, "use_inductor_graph_partition": True},
            {
                "cudagraph_mode": "piecewise",
                "mode": 3,
                "use_inductor_graph_partition": False,
            },
        ],
    )
    def test_nested_dict_value(self, compilation_config):
        """Test that nested dict values are serialized as JSON."""
        item = ParameterSweepItem.from_record(
            {"compilation_config": compilation_config}
        )
        cmd = item.apply_to_cmd(["vllm", "serve", "model"])
        assert "--compilation-config" in cmd
        # The dict should be JSON serialized
        idx = cmd.index("--compilation-config")
        assert json.loads(cmd[idx + 1]) == compilation_config

    @pytest.mark.parametrize(
        "input_dict,expected_key,expected_value",
        [
            ({"model": "test-model"}, "--model", "test-model"),
            ({"max_tokens": 100}, "--max-tokens", "100"),
            ({"temperature": 0.7}, "--temperature", "0.7"),
        ],
    )
    def test_string_and_numeric_values(self, input_dict, expected_key, expected_value):
        """Test that string and numeric values are handled correctly."""
        item = ParameterSweepItem.from_record(input_dict)
        cmd = item.apply_to_cmd(["vllm", "serve"])
        assert expected_key in cmd
        assert expected_value in cmd

    @pytest.mark.parametrize(
        "input_dict,expected_key,key_idx_offset",
        [
            ({"max_tokens": 200}, "--max-tokens", 1),
            ({"enable_prefix_caching": False}, "--no-enable-prefix-caching", 0),
        ],
    )
    def test_replace_existing_parameter(self, input_dict, expected_key, key_idx_offset):
        """Test that existing parameters in cmd are replaced."""
        item = ParameterSweepItem.from_record(input_dict)

        if key_idx_offset == 1:
            # Key-value pair
            cmd = item.apply_to_cmd(["vllm", "serve", "--max-tokens", "100", "model"])
            assert expected_key in cmd
            idx = cmd.index(expected_key)
            assert cmd[idx + 1] == "200"
            assert "100" not in cmd
        else:
            # Boolean flag
            cmd = item.apply_to_cmd(
                ["vllm", "serve", "--enable-prefix-caching", "model"]
            )
            assert expected_key in cmd
            assert "--enable-prefix-caching" not in cmd


class TestParameterSweep:
    """Test ParameterSweep functionality."""

    def test_from_records_list(self):
        """Test creating ParameterSweep from a list of records."""
        records = [
            {"max_tokens": 100, "temperature": 0.7},
            {"max_tokens": 200, "temperature": 0.9},
        ]
        sweep = ParameterSweep.from_records(records)
        assert len(sweep) == 2
        assert sweep[0]["max_tokens"] == 100
        assert sweep[1]["max_tokens"] == 200

    def test_read_from_dict(self):
        """Test creating ParameterSweep from a dict format."""
        data = {
            "experiment1": {"max_tokens": 100, "temperature": 0.7},
            "experiment2": {"max_tokens": 200, "temperature": 0.9},
        }
        sweep = ParameterSweep.read_from_dict(data)
        assert len(sweep) == 2

        # Check that items have the _benchmark_name field
        names = {item["_benchmark_name"] for item in sweep}
        assert names == {"experiment1", "experiment2"}

        # Check that parameters are preserved
        for item in sweep:
            if item["_benchmark_name"] == "experiment1":
                assert item["max_tokens"] == 100
                assert item["temperature"] == 0.7
            elif item["_benchmark_name"] == "experiment2":
                assert item["max_tokens"] == 200
                assert item["temperature"] == 0.9

    def test_read_json_list_format(self):
        """Test reading JSON file with list format."""
        records = [
            {"max_tokens": 100, "temperature": 0.7},
            {"max_tokens": 200, "temperature": 0.9},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(records, f)
            temp_path = Path(f.name)

        try:
            sweep = ParameterSweep.read_json(temp_path)
            assert len(sweep) == 2
            assert sweep[0]["max_tokens"] == 100
            assert sweep[1]["max_tokens"] == 200
        finally:
            temp_path.unlink()

    def test_read_json_dict_format(self):
        """Test reading JSON file with dict format."""
        data = {
            "experiment1": {"max_tokens": 100, "temperature": 0.7},
            "experiment2": {"max_tokens": 200, "temperature": 0.9},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            sweep = ParameterSweep.read_json(temp_path)
            assert len(sweep) == 2

            # Check that items have the _benchmark_name field
            names = {item["_benchmark_name"] for item in sweep}
            assert names == {"experiment1", "experiment2"}
        finally:
            temp_path.unlink()

    def test_unique_benchmark_names_validation(self):
        """Test that duplicate _benchmark_name values raise an error."""
        # Test with duplicate names in list format
        records = [
            {"_benchmark_name": "exp1", "max_tokens": 100},
            {"_benchmark_name": "exp1", "max_tokens": 200},
        ]

        with pytest.raises(ValueError, match="Duplicate _benchmark_name values"):
            ParameterSweep.from_records(records)

    def test_unique_benchmark_names_multiple_duplicates(self):
        """Test validation with multiple duplicate names."""
        records = [
            {"_benchmark_name": "exp1", "max_tokens": 100},
            {"_benchmark_name": "exp1", "max_tokens": 200},
            {"_benchmark_name": "exp2", "max_tokens": 300},
            {"_benchmark_name": "exp2", "max_tokens": 400},
        ]

        with pytest.raises(ValueError, match="Duplicate _benchmark_name values"):
            ParameterSweep.from_records(records)

    def test_no_benchmark_names_allowed(self):
        """Test that records without _benchmark_name are allowed."""
        records = [
            {"max_tokens": 100, "temperature": 0.7},
            {"max_tokens": 200, "temperature": 0.9},
        ]
        sweep = ParameterSweep.from_records(records)
        assert len(sweep) == 2

    def test_mixed_benchmark_names_allowed(self):
        """Test that mixing records with and without _benchmark_name is allowed."""
        records = [
            {"_benchmark_name": "exp1", "max_tokens": 100},
            {"max_tokens": 200, "temperature": 0.9},
        ]
        sweep = ParameterSweep.from_records(records)
        assert len(sweep) == 2


class TestParameterSweepItemKeyNormalization:
    """Test key normalization in ParameterSweepItem."""

    def test_underscore_to_hyphen_conversion(self):
        """Test that underscores are converted to hyphens in CLI."""
        item = ParameterSweepItem.from_record({"max_tokens": 100})
        cmd = item.apply_to_cmd(["vllm", "serve"])
        assert "--max-tokens" in cmd

    def test_nested_key_preserves_suffix(self):
        """Test that nested keys preserve the suffix format."""
        # The suffix after the dot should preserve underscores
        item = ParameterSweepItem.from_record(
            {"compilation_config.some_nested_param": "value"}
        )
        cmd = item.apply_to_cmd(["vllm", "serve"])
        # The prefix (compilation_config) gets converted to hyphens,
        # but the suffix (some_nested_param) is preserved
        assert any("compilation-config.some_nested_param" in arg for arg in cmd)


@pytest.mark.parametrize("workload", ["synthetic", "realistic"])
def test_group_major_sweep_isolates_kernel_dispatch(monkeypatch, workload):
    """The sweep must not change attention, draft, capacity, or benchmark work."""
    import importlib
    import shlex

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    runner = importlib.import_module("run_group_major_serving")
    a, b = (runner.plan(workload, s) for s in ("1a", "1b"))
    for flag in ("--serve-cmd", "--bench-cmd", "--after-bench-cmd", "--bench-params"):
        assert (
            a["command"][a["command"].index(flag) + 1]
            == b["command"][b["command"].index(flag) + 1]
        )
    for leg, enabled in ((a, "0"), (b, "1")):
        controls = leg["controls"]
        assert controls["VLLM_GFX1151_W4_GROUP_MAJOR"] == enabled
        assert controls["VLLM_GFX1151_QWEN_ATTENTION"] == "1"
        assert controls["VLLM_GFX1151_QWEN_GDN"] == "1"
        assert controls["VLLM_USE_AOT_COMPILE"] == "1"
        assert controls["VLLM_DISABLE_COMPILE_CACHE"] == "0"
        serve = shlex.split(leg["command"][leg["command"].index("--serve-cmd") + 1])
        assert "--jit-monitor-verbose" in serve
        assert "--enforce-eager" not in serve
        assert (
            json.loads(serve[serve.index("--speculative-config") + 1])[
                "num_speculative_tokens"
            ]
            == 7
        )
    differing = {k for k in a["controls"] if a["controls"][k] != b["controls"][k]}
    assert differing == {
        "VLLM_GFX1151_W4_GROUP_MAJOR",
        "GFX1151_SERVING_IDENTITY",
        "GFX1151_JIT_EVIDENCE",
    }
    assert a["directory"] != b["directory"]


@pytest.mark.parametrize("workload", ["synthetic", "realistic"])
def test_group_major_final_root_changes_only_artifact_paths(
    monkeypatch, tmp_path, workload
):
    """A fresh final matrix must keep the pilot's exact serving protocol."""
    import importlib

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    runner = importlib.import_module("run_group_major_serving")
    original_root = runner.ROOT
    output_root = tmp_path / "final"
    assert list(runner.previous.STARTS) == ["1a", "1b", "2b", "2a", "3a", "3b"]
    for start in runner.previous.STARTS:
        original = runner.plan(workload, start)
        actual = runner.plan(workload, start, root=output_root)
        expected = json.loads(
            json.dumps(original).replace(str(original_root), str(output_root))
        )
        assert actual == expected
        for key in ("directory", "identity", "log", "jit"):
            assert Path(actual[key]).is_relative_to(output_root)
    assert original_root == runner.ROOT


@pytest.mark.parametrize("workload", ["synthetic", "realistic"])
def test_group_major_final_root_dry_run_starts_no_server(
    monkeypatch, tmp_path, capsys, workload
):
    """Planning into a new root is read-only and keeps all six fresh legs."""
    import importlib
    import sys

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    runner = importlib.import_module("run_group_major_serving")
    output_root = tmp_path / "final"

    def forbidden():
        pytest.fail("Read-only planning must not enter execution guards")

    monkeypatch.setattr(runner.previous.guards, "require_no_rocm_clients", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [runner.__file__, "--workload", workload, "--output-root", str(output_root)],
    )
    runner.main()
    report = json.loads(capsys.readouterr().out)
    assert not report["executed"]
    assert report["jobs"] == [
        runner.plan(workload, start, root=output_root)
        for start in runner.previous.STARTS
    ]
    assert not output_root.exists()


@pytest.mark.parametrize(
    "event_time,passed",
    [(9.0, True), (10.0, False), (11.0, False), (13.0, False), (14.0, True)],
)
def test_group_major_jit_audit_excludes_warmup_not_measured_compile(
    monkeypatch, event_time, passed
):
    import importlib

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    runner = importlib.import_module("run_group_major_serving")
    worker = dict(pid=123, pgid=120, start_ticks=5)
    events = [
        dict(
            kind="activated",
            time=1.0,
            process=worker,
            settings=dict(mode="warn", verbose=True),
        ),
        dict(kind="jit", time=event_time, process=worker),
    ]
    result = dict(start_times=[10.0, 12.0], latencies=[1.0, 1.0], completed=2)
    proof = dict(
        server=dict(
            process=dict(pid=120, pgid=120, start_ticks=1),
            controls=dict(VLLM_WORKER_MULTIPROC_METHOD="spawn"),
        ),
        gpu_clients=[worker],
    )
    report = runner.audit_jit_data(result, proof, events)
    assert report["passed"] is passed
    assert report["interval"] == [10.0, 13.0]


@pytest.mark.parametrize("defect", ["missing", "identity", "late", "nonverbose"])
def test_group_major_jit_audit_requires_live_worker_monitor(monkeypatch, defect):
    import importlib

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    runner = importlib.import_module("run_group_major_serving")
    worker = dict(pid=123, pgid=120, start_ticks=5)
    event = dict(
        kind="activated",
        time=1.0,
        process=worker.copy(),
        settings=dict(mode="warn", verbose=True),
    )
    if defect == "identity":
        event["process"]["start_ticks"] += 1
    elif defect == "late":
        event["time"] = 11.0
    elif defect == "nonverbose":
        event["settings"]["verbose"] = False
    events = [] if defect == "missing" else [event]
    proof = dict(
        server=dict(
            process=dict(pid=120, pgid=120, start_ticks=1),
            controls=dict(VLLM_WORKER_MULTIPROC_METHOD="spawn"),
        ),
        gpu_clients=[worker],
    )
    with pytest.raises(AssertionError):
        runner.audit_jit_data(
            dict(start_times=[10.0], latencies=[1.0], completed=1),
            proof,
            events,
        )


@pytest.mark.parametrize(
    "case", ["api_handle", "unknown_worker", "api_only", "wrong_group", "not_spawn"]
)
def test_group_major_jit_gate_distinguishes_api_handle_from_model_workers(
    monkeypatch, case
):
    """A KFD handle is not proof that the MP API executes the model."""
    import importlib

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    runner = importlib.import_module("run_group_major_serving")
    api = dict(pid=120, pgid=120, start_ticks=1)
    worker = dict(pid=123, pgid=120, start_ticks=5)
    proof = dict(
        server=dict(process=api, controls=dict(VLLM_WORKER_MULTIPROC_METHOD="spawn")),
        gpu_clients=[api, worker],
    )
    events = [
        dict(
            kind="activated",
            time=1.0,
            process=worker.copy(),
            settings=dict(mode="warn", verbose=True),
        )
    ]
    if case == "unknown_worker":
        proof["gpu_clients"].append(dict(pid=125, pgid=120, start_ticks=7))
    elif case == "api_only":
        proof["gpu_clients"] = [api]
    elif case == "wrong_group":
        worker["pgid"] = 119
    elif case == "not_spawn":
        proof["server"]["controls"]["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
    if case == "api_handle":
        assert runner.monitored_workers(proof, events, 10.0) == [worker]
        report = runner.audit_jit_data(
            dict(start_times=[10.0], latencies=[1.0], completed=1), proof, events
        )
        assert report["passed"] and report["workers"] == [worker]
        assert report["api_process_exempt_from_model_monitor"] == api
    else:
        with pytest.raises(AssertionError):
            runner.monitored_workers(proof, events, 10.0)


@pytest.mark.parametrize("clean_at", [0, 1, 2, None, "changed-server"])
@pytest.mark.parametrize("explicit_checker", [False, True])
def test_group_major_conditioning_keeps_first_clean_not_fastest(
    monkeypatch, tmp_path, clean_at, explicit_checker
):
    """JIT-contaminated native attempts remain intact; retries are bounded."""
    import importlib
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    module = importlib.import_module("gfx1151_jit_conditioning")
    monkeypatch.setenv("GFX1151_JIT_EVIDENCE", str(tmp_path / "jit"))
    monkeypatch.setenv("VLLM_GFX1151_W4_GROUP_MAJOR", "0")
    calls = []

    def execute(command):
        index = len(calls)
        calls.append(command)
        assert command[-1] == module.MARKER
        path = tmp_path / command[command.index("--result-filename") + 1]
        for suffix in module.SUFFIXES:
            path.with_suffix(suffix).write_text(json.dumps({"attempt": index}))
        return SimpleNamespace(returncode=0 if index == clean_at else 1)

    def check(path, code, directory, expected):
        return dict(
            result=str(path),
            clean=code == 0,
            returncode=code,
            server=len(calls) if clean_at == "changed-server" else 123,
            benchmark_command=["same-benchmark"],
        )

    monkeypatch.setattr(module.subprocess, "run", execute)

    def unexpected(*args):
        raise AssertionError("Explicit checker must replace the default")

    monkeypatch.setattr(
        module, "check_attempt", unexpected if explicit_checker else check
    )
    options = dict(checker=check) if explicit_checker else {}
    argv = [
        "entry.py",
        "bench",
        "serve",
        "--result-dir",
        str(tmp_path),
        "--result-filename",
        "run=0.json",
    ]
    if clean_at == "changed-server":
        with pytest.raises(AssertionError):
            module.run(argv, **options)
        assert len(calls) == 2 and not (tmp_path / "run=0.json").exists()
        return
    elif clean_at is None:
        with pytest.raises(RuntimeError, match="three attempts"):
            module.run(argv, **options)
        assert len(calls) == 3
        assert not (tmp_path / "run=0.json").exists()
    else:
        module.run(argv, **options)
        assert len(calls) == clean_at + 1
        assert json.loads((tmp_path / "run=0.json").read_text()) == {
            "attempt": clean_at
        }
        history = json.loads((tmp_path / "run=0.conditioning.json").read_text())
        assert [a["clean"] for a in history["attempts"]] == [False] * clean_at + [True]
    for index in range(len(calls)):
        assert (tmp_path / f"run=0.jit-attempt-{index}.json").exists()
        assert (tmp_path / f"run=0.jit-attempt-{index}.conditioning.json").exists()


def test_group_width_plans_hold_capacity_and_kernel_fixed(monkeypatch):
    """Only speculative width changes across the six fresh-server controls."""
    import importlib
    import shlex

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    runner = importlib.import_module("run_group_width_screen")
    root, jobs = runner.plans()
    assert [j["spec_tokens"] for j in jobs] == [7, 23, 23, 7, 7, 23]
    assert len({j["directory"] for j in jobs}) == 6
    controls = []
    for job in jobs:
        assert job["controls"]["VLLM_GFX1151_W4_GROUP_MAJOR"] == "1"
        assert job["controls"]["VLLM_USE_AOT_COMPILE"] == "1"
        assert Path(job["jit"]).is_relative_to(root)
        command = shlex.split(job["command"][job["command"].index("--serve-cmd") + 1])
        assert runner.ENTRY in command and "--jit-monitor-verbose" in command
        for flag, value in (
            ("--max-num-seqs", "8"),
            ("--max-model-len", "4096"),
            ("--kv-cache-memory-bytes", str(48 * 1024**3)),
        ):
            assert command[command.index(flag) + 1] == value
        index = command.index("--speculative-config") + 1
        config = json.loads(command[index])
        assert config["num_speculative_tokens"] == job["spec_tokens"]
        config["num_speculative_tokens"] = 7
        command[index] = json.dumps(config, sort_keys=True)
        controls.append(command)
    assert all(c == controls[0] for c in controls)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "width23",
        "width15",
        "draft",
        "count",
        "concurrency",
        "warmups",
        "length",
        "kv",
        "maxseq",
        "context",
        "flag",
        "eos",
    ],
)
def test_group_width_protocol_requires_owned_pinned_screen(monkeypatch, fault):
    """An eight-request override must not weaken the normal matrix protocol."""
    import importlib
    import shlex

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    runner = importlib.import_module("run_group_width_screen")
    entry = importlib.import_module("gfx1151_group_width_entry")
    job = runner.plans()[1][0]
    outer = job["command"]
    serve = shlex.split(outer[outer.index("--serve-cmd") + 1])
    command = shlex.split(outer[outer.index("--bench-cmd") + 1])
    command += ["--num-prompts", "8", "--max-concurrency", "1", "--num-warmups", "1"]
    capacity = dict(
        max_num_seqs=8, max_model_len=4096, kv_cache_memory_bytes=48 * 1024**3
    )
    proof = dict(
        command=command,
        server=dict(command=serve, capacity_requirement=capacity),
        group_major_process_env={"123": "1", "124": "1"},
    )
    if fault in ("width23", "width15", "draft"):
        index = serve.index("--speculative-config") + 1
        config = json.loads(serve[index])
        if fault == "draft":
            config["model"] = "unvalidated-draft"
        else:
            config["num_speculative_tokens"] = 23 if fault == "width23" else 15
        serve[index] = json.dumps(config)
    for name, flag, value in (
        ("count", "--num-prompts", "32"),
        ("concurrency", "--max-concurrency", "2"),
        ("warmups", "--num-warmups", "2"),
        ("length", "--random-output-len", "512"),
    ):
        if fault == name:
            command[command.index(flag) + 1] = value
    if fault == "kv":
        capacity["kv_cache_memory_bytes"] = 16 * 1024**3
    elif fault == "maxseq":
        capacity["max_num_seqs"] = 1
    elif fault == "context":
        capacity["max_model_len"] = 12288
    elif fault == "flag":
        proof["group_major_process_env"]["124"] = "0"
    elif fault == "eos":
        command.remove("--ignore-eos")
    if fault in (None, "width23"):
        assert entry.protocol(proof) == ([1024] * 8, 1024, 23 if fault else 7)
    else:
        with pytest.raises((AssertionError, ValueError)):
            entry.protocol(proof)


def test_group_major_conditioning_does_not_retry_invalid_evidence(
    monkeypatch, tmp_path
):
    """A crashed client or invalid accounting is not a warmup opportunity."""
    import importlib
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    module = importlib.import_module("gfx1151_jit_conditioning")
    monkeypatch.setenv("GFX1151_JIT_EVIDENCE", str(tmp_path))
    monkeypatch.setenv("VLLM_GFX1151_W4_GROUP_MAJOR", "0")
    calls = []

    def execute(command):
        calls.append(command)
        return SimpleNamespace(returncode=2)

    def check(*args):
        raise ValueError("Invalid accounting or missing evidence")

    monkeypatch.setattr(module.subprocess, "run", execute)
    monkeypatch.setattr(module, "check_attempt", check)
    with pytest.raises(ValueError, match="Invalid accounting"):
        module.run(
            [
                "entry.py",
                "bench",
                "serve",
                "--result-dir",
                str(tmp_path),
                "--result-filename",
                "run=0.json",
            ]
        )
    assert len(calls) == 1 and not (tmp_path / "run=0.json").exists()


def test_group_major_conditioning_preserves_existing_result(monkeypatch, tmp_path):
    import importlib

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    module = importlib.import_module("gfx1151_jit_conditioning")
    path = tmp_path / "run=0.json"
    path.write_text("preserve me")
    with pytest.raises(AssertionError):
        module.run(
            [
                "entry.py",
                "bench",
                "serve",
                "--result-dir",
                str(tmp_path),
                "--result-filename",
                path.name,
            ]
        )
    assert path.read_text() == "preserve me"


@pytest.mark.parametrize("kind", ["historical-test", "changed-runtime"])
def test_group_major_history_resolves_only_archived_test(monkeypatch, tmp_path, kind):
    """Extended test coverage must not permit changed runtime sources."""
    import hashlib
    import importlib

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    module = importlib.import_module("gfx1151_profile_test_history")
    current, archive = tmp_path / "test.py", tmp_path / "archive.py"
    current.write_text("extended tests")
    archive.write_text("old tests")
    old_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "PROFILE", current)
    monkeypatch.setattr(module, "ARCHIVE", archive)
    monkeypatch.setattr(module, "OLD_SHA", old_sha)
    hashes = {}
    if kind == "historical-test":
        module.remember_historical_test(
            hashes, current, old_sha, module.evidence.remember
        )
        assert hashes == {str(archive): old_sha}
    else:
        runtime = tmp_path / "runtime.py"
        runtime.write_text("changed runtime")
        with pytest.raises(AssertionError, match="Evidence changed"):
            module.remember_historical_test(
                hashes, runtime, old_sha, module.evidence.remember
            )


def test_group_major_history_restores_strict_reader_on_failure(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(Path("build").resolve()))
    module = importlib.import_module("gfx1151_profile_test_history")
    monkeypatch.setattr(module, "require_current_tests", lambda hashes: None)
    original = module.evidence.remember
    with pytest.raises(RuntimeError, match="stop"), module.extended_profile_tests({}):
        assert module.evidence.remember is not original
        raise RuntimeError("stop")
    assert module.evidence.remember is original
