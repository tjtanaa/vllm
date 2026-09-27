# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GSM8K evaluation using vLLM server and isolated GSM8K script.
Replacement for lm-eval-harness with better performance and control.

Usage:
pytest -s -v tests/evals/gsm8k/test_gsm8k_correctness.py \
    --config-list-file=configs/models-small.txt
"""

import shlex

import pytest
import requests
import yaml

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

from .gsm8k_eval import evaluate_gsm8k

DEFAULT_STARTUP_MAX_WAIT_SECONDS = 1200


@pytest.mark.parametrize(
    "response, expected",
    [
        ("We have 17 apples, then calculate 23", None),
        ("The intermediate result is 17.\n#### 23", "23"),
        ("#### -1,234.50", "-1234.50"),
        ("#### 17\nCorrection: #### 23", "23"),
        ("#### ", None),
    ],
)
def test_saved_gsm8k_audit_requires_final_answer(response, expected):
    from benchmarks.gsm8k_truncation_diagnostic import final_answer

    value = final_answer(response)
    assert (None if value is None else str(value)) == expected


@pytest.fixture
def paired_gsm8k_records():
    from decimal import Decimal

    prompts = [f"Question {i}:" for i in range(4)]
    labels = [Decimal(-10), Decimal(2), Decimal(3), Decimal(4)]

    def records(correct_indices, model):
        protocol = dict(
            format_version=2,
            dataset_sha256="fixture-digest",
            model=model,
            max_tokens=8192,
            temperature=0,
            seed=42,
            num_shots=5,
            count=4,
            concurrency=4,
            score_prefix=False,
            revision=model + "-revision",
            run_label="nospec_eager",
        )
        return [
            dict(
                index=i,
                protocol=protocol.copy(),
                prompt=prompts[i],
                label=str(labels[i]),
                text=f"#### {labels[i] if i in correct_indices or i == 3 else 0}",
                final_answer=str(labels[i] if i in correct_indices or i == 3 else 0),
                final_correct=i in correct_indices,
                finish_reason="length" if i == 3 else "stop",
                token_ids=[42],
                usage={"completion_tokens": 1},
            )
            for i in range(4)
        ]

    return records({0, 1}, "baseline"), records({1, 2}, "candidate"), prompts, labels


def test_saved_gsm8k_comparison_pairs_questions_and_counts_truncation(
    paired_gsm8k_records,
):
    from benchmarks.gsm8k_compare import compare_records

    baseline, candidate, prompts, labels = paired_gsm8k_records
    report = compare_records(
        baseline, candidate[::-1], prompts, labels, "fixture-digest"
    )
    assert report["baseline"]["correct"] == report["candidate"]["correct"] == 2
    assert report["baseline"]["finish_reasons"] == {"stop": 3, "length": 1}
    assert report["paired"] == dict(
        accuracy_delta=0.0,
        both_correct=1,
        both_wrong=1,
        regression_indices=[0],
        improvement_indices=[2],
        mcnemar_exact_two_sided_p=1.0,
    )
    low, high = report["baseline"]["wilson_95_interval"]
    assert low == pytest.approx(0.150038989, abs=1e-8)
    assert high == pytest.approx(0.849961011, abs=1e-8)


@pytest.mark.parametrize("self_comparison, expected_p", [(True, 1.0), (False, 0.125)])
def test_saved_gsm8k_paired_test_handles_no_discordance_and_one_sided_losses(
    paired_gsm8k_records, self_comparison, expected_p
):
    from benchmarks.gsm8k_compare import compare_records

    baseline, candidate, prompts, labels = paired_gsm8k_records
    for row in baseline:
        row.update(
            text=f"#### {row['label']}",
            final_answer=row["label"],
            final_correct=True,
            finish_reason="stop",
        )
    for row in candidate:
        row.update(
            text="#### 0", final_answer="0", final_correct=False, finish_reason="stop"
        )
    report = compare_records(
        baseline,
        baseline if self_comparison else candidate,
        prompts,
        labels,
        "fixture-digest",
    )
    assert report["paired"]["mcnemar_exact_two_sided_p"] == expected_p
    assert report["paired"]["accuracy_delta"] == (0.0 if self_comparison else -1.0)


@pytest.mark.parametrize(
    "defect, match",
    [
        ("missing", "Incomplete"),
        ("duplicate", "Duplicate"),
        ("prompt", "Canonical"),
        ("label", "Canonical"),
        ("score", "Saved score"),
        ("prediction", "Saved score"),
        ("token_count", "token accounting"),
        ("finish", "finish reason"),
        ("mixed_protocol", "Mixed protocols"),
        ("settings", "settings differ"),
        ("dataset", "dataset digest"),
        ("revision", "revision"),
    ],
)
def test_saved_gsm8k_comparison_rejects_unfair_or_corrupt_runs(
    paired_gsm8k_records,
    defect,
    match,
):
    from benchmarks.gsm8k_compare import compare_records

    baseline, candidate, prompts, labels = paired_gsm8k_records
    if defect == "missing":
        candidate.pop()
    elif defect == "duplicate":
        candidate[-1] = candidate[0]
    elif defect == "prompt":
        candidate[0]["prompt"] += "different template"
    elif defect == "label":
        candidate[0]["label"] = "10"  # A lost minus sign must not pass.
    elif defect == "score":
        candidate[0]["final_correct"] = True
    elif defect == "prediction":
        candidate[0]["final_answer"] = "-10"
    elif defect == "token_count":
        candidate[0]["usage"]["completion_tokens"] = 2
    elif defect == "finish":
        candidate[0]["finish_reason"] = "error"
    elif defect == "mixed_protocol":
        candidate[1]["protocol"]["seed"] = 43
    else:
        key, value = {
            "settings": ("max_tokens", 256),
            "dataset": ("dataset_sha256", "different"),
            "revision": ("revision", None),
        }[defect]
        for row in candidate:
            row["protocol"][key] = value
    with pytest.raises(ValueError, match=match):
        compare_records(baseline, candidate, prompts, labels, "fixture-digest")


@pytest.fixture
def smoke_gsm8k_records(paired_gsm8k_records):
    from copy import deepcopy
    from decimal import Decimal

    source = paired_gsm8k_records[0][0]
    prompts = [f"Question {i}:" for i in range(100)]
    labels = [Decimal(-10)] * 100
    records = []
    for i, prompt in enumerate(prompts):
        row = deepcopy(source)
        row.update(index=i, prompt=prompt)
        row["protocol"]["count"] = 1319
        records.append(row)
    return records, deepcopy(records), prompts, labels


def test_saved_gsm8k_smoke_uses_fixed_ids_and_preserves_source_protocol(
    smoke_gsm8k_records,
):
    from copy import deepcopy

    from benchmarks.gsm8k_compare import compare_records, compare_smoke_records

    baseline, candidate, prompts, labels = smoke_gsm8k_records
    extra = deepcopy(candidate[0])
    extra["index"] = 100
    candidate.append(extra)
    original = deepcopy(candidate)
    report = compare_smoke_records(
        baseline, candidate[::-1], prompts, labels, "fixture-digest"
    )
    assert report["evaluation_scope"] == "smoke100"
    assert report["question_indices"] == list(range(100))
    assert report["candidate"]["count"] == report["candidate"]["correct"] == 100
    assert report["candidate"]["protocol"]["count"] == 1319
    assert report["candidate"]["source_record_count"] == 101
    assert candidate == original
    with pytest.raises(ValueError, match="count or dataset"):
        compare_records(baseline, candidate[:100], prompts, labels, "fixture-digest")


@pytest.mark.parametrize(
    "defect, match",
    [
        ("missing", "Incomplete smoke"),
        ("duplicate", "Duplicate"),
        ("mixed", "Mixed protocols"),
        ("wrong_count", "declare 100 or 1319"),
        ("wrong_prompt", "Canonical"),
        ("wrong_score", "Saved score"),
        ("token_count", "token accounting"),
    ],
)
def test_saved_gsm8k_smoke_rejects_missing_or_corrupt_evidence(
    smoke_gsm8k_records, defect, match
):
    from benchmarks.gsm8k_compare import compare_smoke_records

    baseline, candidate, prompts, labels = smoke_gsm8k_records
    if defect == "missing":
        candidate.pop()
    elif defect == "duplicate":
        candidate.append(candidate[0])
    elif defect == "mixed":
        candidate[0]["protocol"]["seed"] = 43
    elif defect == "wrong_count":
        for row in candidate:
            row["protocol"]["count"] = 99
    elif defect == "wrong_prompt":
        candidate[0]["prompt"] += "changed"
    elif defect == "wrong_score":
        candidate[0]["final_correct"] = False
    else:
        candidate[0]["usage"]["completion_tokens"] = 2
    with pytest.raises(ValueError, match=match):
        compare_smoke_records(baseline, candidate, prompts, labels, "fixture-digest")


def test_saved_gsm8k_smoke_accepts_new_100_question_run(smoke_gsm8k_records):
    from benchmarks.gsm8k_compare import compare_smoke_records

    baseline, candidate, prompts, labels = smoke_gsm8k_records
    for row in candidate:
        row["protocol"]["count"] = 100
    report = compare_smoke_records(
        baseline, candidate, prompts, labels, "fixture-digest"
    )
    assert report["candidate"]["correct"] == 100
    assert report["candidate"]["protocol"]["count"] == 100


@pytest.fixture
def budget_records():
    from decimal import Decimal
    from unittest.mock import Mock

    rows = [
        dict(
            index=0,
            text="Reason 17. #### -10",
            token_ids=[10, 11, 12],
            finish_reason="stop",
        ),
        dict(index=1, text="#### 0", token_ids=[20], finish_reason="stop"),
        dict(
            index=2,
            text="Reason 3. #### 0",
            token_ids=[30, 31, 32],
            finish_reason="stop",
        ),
        dict(index=3, text="#### 4", token_ids=[40, 41], finish_reason="length"),
    ]
    decoded = {
        (10, 11, 12): rows[0]["text"] + "Question:",
        (10, 11): "Reason 17.",
        (30, 31, 32): rows[2]["text"],
        (30, 31): "Reason 3.",
    }
    decoder = Mock()
    decoder.decode.side_effect = lambda ids, **kwargs: decoded[tuple(ids)]
    return rows, list(map(Decimal, [-10, 2, 3, 4])), [10, 2, 3, 4], decoder


def test_saved_budget_audit_separates_tail_effects_from_final_answer_scoring(
    budget_records,
):
    from benchmarks.gsm8k_budget_audit import budget_diagnostic

    report = budget_diagnostic(*budget_records, prefix_tokens=2)
    assert report["count"] == 4 and report["clipped_count"] == 2
    assert report["strict_completed_correct"] == 1
    assert report["legacy_completed_correct"] == report["legacy_prefix_correct"] == 2
    assert report["rescued_legacy_indices"] == [0]
    assert report["reverse_legacy_indices"] == [2]
    assert report["correct_final_marker_in_prefix"] == 1
    assert report["legacy_completed_correct_but_strict_wrong"] == [3]
    assert report["questions"][3]["prefix_text"] is None
    assert budget_records[3].decode.call_count == 4


@pytest.mark.parametrize("mismatch", ["full", "prefix"])
def test_saved_budget_audit_rejects_incompatible_tokenizer(budget_records, mismatch):
    from benchmarks.gsm8k_budget_audit import budget_diagnostic

    rows, labels, legacy, decoder = budget_records
    original = decoder.decode.side_effect

    def decode(ids, **kwargs):
        if (mismatch == "full" and len(ids) == 3) or (
            mismatch == "prefix" and len(ids) == 2
        ):
            return "Wrong tokenizer"
        return original(ids, **kwargs)

    decoder.decode.side_effect = decode
    with pytest.raises(ValueError, match="mismatch|not aligned"):
        budget_diagnostic(rows, labels, legacy, decoder, prefix_tokens=2)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "budget"])
def test_saved_budget_audit_rejects_incomplete_records(budget_records, defect):
    from benchmarks.gsm8k_budget_audit import budget_diagnostic

    rows, labels, legacy, decoder = budget_records
    if defect == "missing":
        rows.pop()
    if defect == "duplicate":
        rows[-1] = rows[0]
    with pytest.raises(ValueError, match="Incomplete records"):
        budget_diagnostic(
            rows, labels, legacy, decoder, prefix_tokens=0 if defect == "budget" else 2
        )


def run_gsm8k_eval(eval_config: dict, server_url: str) -> dict:
    """Run GSM8K evaluation using our isolated script."""
    # Extract host and port from server URL
    if "://" in server_url:
        server_url = server_url.split("://")[1]

    host_port = server_url.split("/")[0]  # Remove path if present
    if ":" in host_port:
        host, p = host_port.split(":")
        port = int(p)
    else:
        host = host_port
        port = 8000

    # Add http:// prefix if not present
    if not host.startswith("http"):
        host = f"http://{host}"

    # Run GSM8K evaluation
    request_timeout_seconds = eval_config.get("request_timeout_seconds", 600)
    if current_platform.is_rocm():
        request_timeout_seconds = eval_config.get(
            "rocm_request_timeout_seconds", request_timeout_seconds
        )

    results = evaluate_gsm8k(
        num_questions=eval_config["num_questions"],
        num_shots=eval_config["num_fewshot"],
        max_tokens=eval_config.get("max_tokens", 256),
        model=eval_config["model_name"],
        use_chat_completions=eval_config.get("use_chat_completions", False),
        host=host,
        port=port,
        temperature=eval_config.get("temperature", 0.0),
        seed=eval_config.get("seed", 42),
        request_timeout_seconds=request_timeout_seconds,
        gen_prefix=eval_config.get("gen_prefix", ""),
        max_concurrency=eval_config.get("max_concurrency"),
    )

    return results


def get_acceptance_length(server_url: str) -> float:
    """Mean tokens emitted per verification step, from the server's counters.

    1.0 means every draft was rejected (speculation bought nothing); the
    theoretical maximum is 1 + num_speculative_tokens.
    """
    response = requests.get(f"{server_url.rstrip('/').removesuffix('/v1')}/metrics")
    response.raise_for_status()
    counters: dict[str, float] = {}
    for line in response.text.splitlines():
        if line.startswith("vllm:spec_decode_num_"):
            name, _, value = line.partition(" ")
            counters[name.split("{")[0]] = float(value)

    num_drafts = counters.get("vllm:spec_decode_num_drafts_total", 0.0)
    num_accepted = counters.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    assert num_drafts > 0, (
        "no drafts recorded; speculative decoding did not run for this config"
    )
    return 1.0 + num_accepted / num_drafts


def test_gsm8k_correctness(config_filename):
    """Test GSM8K correctness for a given model configuration."""
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    if (
        not current_platform.is_cuda()
        and "Qwen3-30B-A3B-MXFP4A16" in eval_config["model_name"]
    ):
        pytest.skip(
            "Skipping Qwen3-30B-A3B-MXFP4A16 on non-CUDA platforms. "
            "Marlin kernels are not supported."
        )

    if (
        not current_platform.is_cuda()
        and "gemma-4-E4B-it-qat-mobile-ct" in eval_config["model_name"]
    ):
        pytest.skip(
            "Skipping gemma-4-E4B-it-qat-mobile-ct on non-CUDA platforms. "
            "Its W2A16 (uint2b2) scheme has no kernel outside CUDA."
        )

    # TODO(akaratza): Enable DeepSeek-V3.2 and DeepSeek-R1 on ROCm platforms
    if current_platform.is_rocm() and (
        "deepseek-ai/DeepSeek-V3.2" in eval_config["model_name"]
        or "deepseek-ai/DeepSeek-R1" in eval_config["model_name"]
    ):
        pytest.skip(
            "Skipping DeepSeek-V3.2 and DeepSeek-R1 on ROCm platforms "
            "due to agent pool disk space issues and pod evictions."
        )
    if current_platform.is_rocm() and (
        "Qwen3.5-35B-A3B-MXFP4-AITER-TP2" in config_filename.name
    ):
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            pytest.skip(
                "Skipping Qwen3.5-35B-A3B-MXFP4-AITER-TP2 on non-GFX950 platforms. "
                "The quantization scheme is not supported on non-GFX950 platforms."
            )
    # Parse server arguments from config (use shlex to handle quoted strings)
    server_args_str = eval_config.get("server_args", "")
    server_args = shlex.split(server_args_str) if server_args_str else []

    # Add standard server arguments
    server_args.extend(
        [
            "--trust-remote-code",
            "--disable-uvicorn-access-log",
        ]
    )

    startup_max_wait_seconds = eval_config.get(
        "startup_max_wait_seconds", DEFAULT_STARTUP_MAX_WAIT_SECONDS
    )
    env_dict = dict(eval_config.get("env") or {})
    env_dict["VLLM_ENGINE_READY_TIMEOUT_S"] = str(int(startup_max_wait_seconds))

    print(f"Starting GSM8K evaluation for model: {eval_config['model_name']}")
    print(f"Expected metric threshold: {eval_config['accuracy_threshold']}")
    print(f"Number of questions: {eval_config['num_questions']}")
    print(f"Number of few-shot examples: {eval_config['num_fewshot']}")
    request_timeout_seconds = eval_config.get("request_timeout_seconds", 600)
    if current_platform.is_rocm():
        request_timeout_seconds = eval_config.get(
            "rocm_request_timeout_seconds", request_timeout_seconds
        )
    print(f"Request timeout: {request_timeout_seconds}s")
    print(f"Startup max wait: {startup_max_wait_seconds}s")
    print(f"Server args: {' '.join(server_args)}")
    print(f"Environment variables: {env_dict}")

    # Launch server and run evaluation
    with RemoteOpenAIServer(
        eval_config["model_name"],
        server_args,
        env_dict=env_dict,
        max_wait_seconds=startup_max_wait_seconds,
    ) as remote_server:
        server_url = remote_server.url_for("v1")
        print(f"Server started at: {server_url}")

        results = run_gsm8k_eval(eval_config, server_url)

        measured_metric = results["accuracy"]
        expected_metric = eval_config["accuracy_threshold"]
        tol = eval_config.get("tolerance", 0.08)

        print(f"GSM8K Results for {eval_config['model_name']}:")
        print(f"  Measured metric: {measured_metric:.4f}")
        print(f"  Expected metric: {expected_metric:.4f}")
        print(f"  Tolerance: {tol:.4f}")
        print(f"  Questions: {results['num_questions']}")
        print(f"  Invalid rate: {results['invalid_rate']:.3f}")
        print(f"  Latency: {results['latency']:.1f}s")
        print(f"  QPS: {results['questions_per_second']:.1f}")

        assert measured_metric >= expected_metric - tol, (
            f"GSM8K metric too low: {measured_metric:.4f} < "
            f"{expected_metric:.4f} - {tol:.4f} = {expected_metric - tol:.4f}"
        )

        # Speculative configs additionally assert that drafts are actually
        # landing: accuracy alone passes even when every draft is rejected.
        min_acceptance_length = eval_config.get("min_acceptance_length")
        if min_acceptance_length is not None:
            acceptance_length = get_acceptance_length(server_url)
            print(f"  Mean acceptance length: {acceptance_length:.3f}")
            print(f"  Minimum acceptance length: {min_acceptance_length:.3f}")
            assert acceptance_length >= min_acceptance_length, (
                f"Acceptance length too low: {acceptance_length:.3f} < "
                f"{min_acceptance_length:.3f}"
            )

        print(f"✅ GSM8K test passed for {eval_config['model_name']}")
