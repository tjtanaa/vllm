# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-node MoRIIO WRITE P/D checks on dummy weights.

Each case serves a real model architecture with ``--load-format dummy``: the
config keeps the real hidden size, head geometry, MLA ranks and recurrent-state
shapes, so the connector registers and transfers exactly the production cache
layout, but the layer stack is cut to a few layers that still contain every
cache-group kind and the MoE is cut to a handful of experts. Dummy weights are
seeded, so the prefill, decode and reference servers hold identical weights.

Prefill runs on one GPU, decode on another, and MoRI moves the KV over xGMI.
A case passes when the decode server's first-token log-probabilities, computed
from the transferred cache, match a direct (non-disaggregated) serve of the
same prompt, and the decode server reports external prefix-cache hits (so the
match is not a full local recompute). Two guards keep the comparison honest:

* the reference must move by more than ``SENSITIVITY_FACTOR * TOLERANCE`` when
  the early context changes, so wrong KV would be visible at all;
* ``test_corrupted_prefill_state_is_detected`` makes prefill compute its KV with
  a different norm epsilon (same shapes, different values) and requires the
  comparison to fail.
"""

import contextlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

_ROOT = Path(__file__).resolve().parents[4]
_PROXY = (
    _ROOT / "examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py"
)
# The toy proxy binds its service-discovery socket to this fixed port, so cases
# on one host must run one at a time.
_PROXY_PING_PORT = 36367

# Max allowed difference between centered first-token log-probabilities of the
# P/D and reference outputs, over the reference's top-k tokens.
TOLERANCE = float(os.getenv("MORIIO_DUMMY_E2E_TOL", "0.02"))
SENSITIVITY_FACTOR = 5.0
TOP_K = 20
MAX_TOKENS = 4

# Shorter than one block, a few blocks, and longer than the prefill token
# budget so the request is split into chunks and the final-chunk save path runs.
PROMPT_LENGTHS = (7, 300, 1500, 3000)
MAX_NUM_BATCHED_TOKENS = 1024


def _mori_available() -> bool:
    try:
        import mori.io  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_rocm() or not _mori_available(),
        reason="MoRIIO P/D needs ROCm and MoRI",
    ),
    pytest.mark.skipif(
        current_platform.is_rocm() and current_platform.device_count() < 2,
        reason="needs one GPU for prefill and one for decode",
    ),
]


# ---------------------------------------------------------------------------
# Model architectures


def _kimi_linear_overrides(cfg) -> dict[str, Any]:
    # 3 KDA layers then 1 MLA layer; layer 0 stays dense (first_k_dense_replace).
    linear = dict(cfg.linear_attn_config)
    linear["kda_layers"] = [1, 2, 3]
    linear["full_attn_layers"] = [4]
    groups = max(1, cfg.num_expert_group)
    return {
        "num_hidden_layers": 4,
        "linear_attn_config": linear,
        "num_experts": 2 * groups,
        "num_experts_per_token": 2,
        "topk_group": min(cfg.topk_group, groups),
    }


def _qwen3_5_moe_overrides(cfg) -> dict[str, Any]:
    # 3 Gated DeltaNet layers then 1 full-attention layer.
    return {
        "text_config": {
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "num_experts": 4,
            "num_experts_per_tok": 2,
        }
    }


@dataclass(frozen=True)
class ArchCase:
    model: str
    overrides: Callable[[Any], dict[str, Any]]
    # Norm-epsilon override that perturbs every computed KV/state value
    # without changing a single shape. Used by the corruption control.
    corrupt_overrides: dict[str, Any]
    env: dict[str, str] = field(default_factory=dict)
    args: tuple[str, ...] = ()


ARCHS: dict[str, ArchCase] = {
    "kimi_linear": ArchCase(
        model="moonshotai/Kimi-Linear-48B-A3B-Instruct",
        overrides=_kimi_linear_overrides,
        corrupt_overrides={"rms_norm_eps": 1.0},
        env={"VLLM_ROCM_USE_AITER": "1"},
        args=("--trust-remote-code", "--mamba-ssm-cache-dtype", "float32"),
    ),
    "qwen3_5_moe": ArchCase(
        model="Qwen/Qwen3.5-35B-A3B",
        overrides=_qwen3_5_moe_overrides,
        corrupt_overrides={"text_config": {"rms_norm_eps": 1.0}},
        # AITER's GDN conv-update kernel rejects the DS conv-state layout that
        # MoRIIO's recurrent-state transfer requires.
        env={"VLLM_ROCM_USE_AITER": "0"},
        args=("--language-model-only",),
    ),
}


def _arch_overrides(arch: ArchCase) -> dict[str, Any]:
    from vllm.transformers_utils.config import get_config

    return arch.overrides(get_config(arch.model, trust_remote_code=True))


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Serving


@dataclass(frozen=True)
class Case:
    arch: str
    attention_backend: str
    prefill_graph: str
    decode_graph: str

    @property
    def id(self) -> str:
        return (
            f"{self.arch}-{self.attention_backend}-"
            f"P_{self.prefill_graph}-D_{self.decode_graph}"
        )


_DEFAULT = "FULL_AND_PIECEWISE"
CASES = [
    # Kimi-Linear: KDA + MLA groups.
    Case("kimi_linear", "ROCM_AITER_MLA", _DEFAULT, _DEFAULT),
    Case("kimi_linear", "ROCM_AITER_MLA", "NONE", "NONE"),
    Case("kimi_linear", "ROCM_AITER_MLA", "PIECEWISE", "PIECEWISE"),
    Case("kimi_linear", "ROCM_AITER_MLA", _DEFAULT, "FULL_DECODE_ONLY"),
    Case("kimi_linear", "TRITON_MLA", _DEFAULT, _DEFAULT),
    # Qwen3.5: Gated DeltaNet + full-attention groups.
    Case("qwen3_5_moe", "TRITON_ATTN", _DEFAULT, _DEFAULT),
    Case("qwen3_5_moe", "TRITON_ATTN", "NONE", "NONE"),
    Case("qwen3_5_moe", "TRITON_ATTN", "PIECEWISE", "PIECEWISE"),
    Case("qwen3_5_moe", "TRITON_ATTN", _DEFAULT, "FULL_DECODE_ONLY"),
    Case("qwen3_5_moe", "ROCM_ATTN", _DEFAULT, _DEFAULT),
]


def _graph_args(mode: str) -> list[str]:
    if mode == "NONE":
        return ["--enforce-eager"]
    return ["--compilation-config", json.dumps({"cudagraph_mode": mode})]


def _visible_devices() -> tuple[str, list[str]]:
    selectors = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    selector = next((k for k in selectors if k in os.environ), selectors[0])
    visible = os.getenv(selector)
    devices = (
        visible.split(",")
        if visible
        else [str(i) for i in range(current_platform.device_count())]
    )
    return selector, devices


def _server(
    arch: ArchCase,
    overrides: dict[str, Any],
    backend: str,
    graph: str,
    device: str,
    selector: str,
    *,
    port: int | None = None,
    kv_transfer_config: dict[str, Any] | None = None,
) -> RemoteOpenAIServer:
    port = port or get_open_port()
    args = [
        *arch.args,
        "--port",
        str(port),
        "--load-format",
        "dummy",
        "--hf-overrides",
        json.dumps(overrides),
        "--dtype",
        "bfloat16",
        "--attention-backend",
        backend,
        "--max-model-len",
        "4096",
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs",
        "16",
        "--kv-cache-memory-bytes",
        str(2 * 1024**3),
        "--gpu-memory-utilization",
        "0.5",
        "--enable-prefix-caching",
        *_graph_args(graph),
    ]
    if kv_transfer_config is not None:
        args += ["--kv-transfer-config", json.dumps(kv_transfer_config)]
    env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        selector: device,
        "VLLM_HOST_IP": "127.0.0.1",
        "VLLM_PORT": str(get_open_port()),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        **arch.env,
    }
    return RemoteOpenAIServer(
        arch.model, args, env_dict=env, auto_port=False, max_wait_seconds=900
    )


def _moriio_config(role: str, http_port: int) -> dict[str, Any]:
    return {
        "kv_connector": "MoRIIOConnector",
        "kv_role": role,
        "kv_connector_extra_config": {
            "backend": "xgmi",
            "read_mode": False,
            "host_ip": "127.0.0.1",
            "proxy_ip": "127.0.0.1",
            "proxy_ping_port": str(_PROXY_PING_PORT),
            "http_port": str(http_port),
            "handshake_port": str(get_open_port()),
            "notify_port": str(get_open_port()),
        },
    }


@contextlib.contextmanager
def _proxy() -> Iterator[str]:
    port = get_open_port()
    process = subprocess.Popen(
        [sys.executable, str(_PROXY), "--port", str(port)],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            assert process.poll() is None, "MoRIIO proxy exited during startup"
            try:
                requests.get(f"http://127.0.0.1:{port}/", timeout=1)
                break
            except requests.ConnectionError:
                time.sleep(0.2)
        else:
            raise TimeoutError("MoRIIO proxy did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


@contextlib.contextmanager
def _pd_pair(
    arch: ArchCase,
    prefill_overrides: dict[str, Any],
    decode_overrides: dict[str, Any],
    case: Case,
) -> Iterator[tuple[str, RemoteOpenAIServer]]:
    selector, devices = _visible_devices()
    assert len(devices) >= 2, "needs one GPU for prefill and one for decode"
    prefill_port, decode_port = get_open_port(), get_open_port()
    with _proxy() as proxy_url, contextlib.ExitStack() as stack:
        servers: list[RemoteOpenAIServer] = []
        stack.callback(RemoteOpenAIServer.shutdown_many, servers)
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(
                    _server,
                    arch,
                    prefill_overrides,
                    case.attention_backend,
                    case.prefill_graph,
                    devices[0],
                    selector,
                    port=prefill_port,
                    kv_transfer_config=_moriio_config("kv_producer", prefill_port),
                ),
                pool.submit(
                    _server,
                    arch,
                    decode_overrides,
                    case.attention_backend,
                    case.decode_graph,
                    devices[1],
                    selector,
                    port=decode_port,
                    kv_transfer_config=_moriio_config("kv_consumer", decode_port),
                ),
            ]
            for future in futures:
                servers.append(future.result())
        _wait_for_registration(proxy_url, arch.model)
        yield proxy_url, servers[1]


def _wait_for_registration(proxy_url: str, model: str) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        response = requests.post(
            f"{proxy_url}/v1/completions",
            json={"model": model, "prompt": [1000, 1001], "max_tokens": 1},
            timeout=300,
        )
        if response.status_code == 200:
            return
        time.sleep(1)
    raise TimeoutError("prefill/decode never registered with the MoRIIO proxy")


# ---------------------------------------------------------------------------
# Measurement


def _prompts(seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1000, 30000) for _ in range(n)] for n in PROMPT_LENGTHS]


def _first_step(url: str, model: str, prompt: list[int]) -> dict[str, float]:
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "logprobs": TOP_K,
            "return_tokens_as_token_ids": True,
        },
        timeout=600,
    )
    response.raise_for_status()
    top = response.json()["choices"][0]["logprobs"]["top_logprobs"]
    assert top, "server returned no logprobs"
    return top[0]


def _distance(ref: dict[str, float], got: dict[str, float]) -> float:
    """Max difference of centered log-probs over the shared top-k tokens.

    Centering removes the log-sum-exp offset, which differs slightly between
    servers, leaving the logit differences the KV actually determines.
    """
    common = sorted(ref.keys() & got.keys())
    if len(common) < len(ref) // 2:
        return float("inf")
    ref_mean = sum(ref[t] for t in common) / len(common)
    got_mean = sum(got[t] for t in common) / len(common)
    return max(abs((ref[t] - ref_mean) - (got[t] - got_mean)) for t in common)


def _external_hits(server: RemoteOpenAIServer) -> float:
    from prometheus_client.parser import text_string_to_metric_families

    response = requests.get(server.url_for("metrics"), timeout=10)
    response.raise_for_status()
    return sum(
        sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name == "vllm:external_prefix_cache_hits_total"
    )


def _reference(
    arch: ArchCase, overrides: dict[str, Any], case: Case
) -> list[dict[str, float]]:
    """Direct serve on the decode settings; also checks the metric is useful."""
    selector, devices = _visible_devices()
    prompts = _prompts()
    with _server(
        arch, overrides, case.attention_backend, case.decode_graph, devices[0], selector
    ) as server:
        url = server.url_root
        refs = [_first_step(url, arch.model, p) for p in prompts]
        # Same tail, different early context: the comparison has to see it.
        for prompt, ref in zip(prompts, refs):
            if len(prompt) < 2:
                continue
            altered = [t + 1 for t in prompt[: len(prompt) // 2]]
            altered += prompt[len(prompt) // 2 :]
            moved = _distance(ref, _first_step(url, arch.model, altered))
            print(f"[sensitivity] len={len(prompt)} context-change distance={moved}")
            assert moved > SENSITIVITY_FACTOR * TOLERANCE, (
                f"first-token log-probs barely depend on the context "
                f"(moved {moved:.4g} for len {len(prompt)}); dummy weights are "
                f"too weak to detect wrong KV at tolerance {TOLERANCE}"
            )
        # Determinism of the reference itself.
        again = _first_step(url, arch.model, prompts[-1])
        print(f"[noise] repeat distance={_distance(refs[-1], again)}")
    return refs


def _pd_distances(
    arch: ArchCase,
    prefill_overrides: dict[str, Any],
    decode_overrides: dict[str, Any],
    case: Case,
    refs: list[dict[str, float]],
) -> tuple[list[float], float]:
    prompts = _prompts()
    with _pd_pair(arch, prefill_overrides, decode_overrides, case) as (
        proxy_url,
        decode,
    ):
        hits_before = _external_hits(decode)
        distances = [
            _distance(ref, _first_step(proxy_url, arch.model, prompt))
            for prompt, ref in zip(prompts, refs)
        ]
        hits = _external_hits(decode) - hits_before
    print(f"[pd] {case.id} distances={distances} external_hits={hits}")
    return distances, hits


# ---------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_write_matches_direct_serve(case: Case):
    arch = ARCHS[case.arch]
    overrides = _arch_overrides(arch)
    refs = _reference(arch, overrides, case)
    distances, hits = _pd_distances(arch, overrides, overrides, case, refs)

    # Every token but the last of every prompt should come from prefill.
    assert hits >= 0.5 * sum(n - 1 for n in PROMPT_LENGTHS), (
        f"decode reported only {hits} external prefix-cache hits; "
        "it recomputed the prompts instead of using the transferred KV"
    )
    bad = {n: d for n, d in zip(PROMPT_LENGTHS, distances) if not d <= TOLERANCE}
    assert not bad, f"P/D first-token log-probs differ from direct serve: {bad}"


@pytest.mark.parametrize("arch_name", sorted(ARCHS))
def test_corrupted_prefill_state_is_detected(arch_name: str):
    """Negative control: KV computed with different values must fail."""
    arch = ARCHS[arch_name]
    default = next(c for c in CASES if c.arch == arch_name)
    case = Case(arch_name, default.attention_backend, _DEFAULT, _DEFAULT)
    overrides = _arch_overrides(arch)
    refs = _reference(arch, overrides, case)
    corrupted = _merge(overrides, arch.corrupt_overrides)
    distances, hits = _pd_distances(arch, corrupted, overrides, case, refs)

    assert hits > 0, "decode never used transferred KV; the control proves nothing"
    assert max(distances) > TOLERANCE, (
        f"P/D outputs with wrongly computed prefill KV still matched the "
        f"reference (distances {distances}); the comparison cannot detect "
        "transfer corruption"
    )
