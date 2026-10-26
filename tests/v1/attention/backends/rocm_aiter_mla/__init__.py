# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_MLA backend suite (attention-backend-testing standard).

The dispatch table (condition -> kernel) is the docstring of
``test_dispatch.py``. Coverage matrix, per
``.agents/skills/attention-backend-testing/references/coverage-matrix.md``:

A. Capability contract (test_capabilities.py)
  A1  dtypes fp16/bf16 run, fp32 rejected: covered
  A2  KV dtypes auto/fp16/bf16/fp8/fp8_e4m3 run, others rejected: covered
      (fp8_e5m2 was declared but ROCm cannot write it; now rejected)
  A3  block sizes: MultipleOf(1); 1/16/64/128 accepted and run: covered
  A4  head sizes: any latent width (576, NoPE 512): covered; per-rank head
      counts without a kernel (>128 and not 16-aligned) assert at impl init
  A5  decoder only, encoder kinds rejected: covered
  A6  sinks / sliding window / mm prefix / batch invariance rejected, soft
      cap / ALiBi refused by the impl, non-causal declared on gfx950 only and
      refused for fp16 without an fp8 cache: covered
  A7  PCP / DCP / KV connector accepted, impl returns natural-log LSE: covered
  A8  customize_spec: inherited per-token-head packing, N/A for these dtypes
  A9  UNIFORM_BATCH declared and captured: test_cudagraph.py
  A10 VLLM_ROCM_USE_AITER / VLLM_ROCM_USE_AITER_MLA priority: covered
B. Metadata builder (test_metadata_builder.py)
  B1  decode/prefill split incl. spec-as-decode threshold: covered
  B2  flat page view, indptrs, qo_indptr, prefill rebasing: covered
  B3  chunked context: none / small / several / uneven: covered
  B4  sliding-window tables: N/A (no sliding window)
  B5  capture build: covered; drafting build: N/A (inherited, no override)
  B6  no GPU sync in decode-only builds: covered, including Gluon verify
  B7  full-graph buffers reused across builds, tails flattened: covered
  B8  padded tokens, padded uniform verify requests: covered
  B9  non-causal group / uniform block validation: covered
C. Forward dispatch (test_dispatch.py)
  C1  every table row: covered; DCP routes by the simulated-rank suites
      (test_rocm_aiter_mla_dcp_*.py, test_rocm_aiter_mla_mtp_split.py) and the
      2-GPU runtime case
  C2  profiling run: covered (X1)
  C3  fused static fp8 output quant: covered (X2)
  C4  KV sharing: N/A (the impl refuses it)
  C5  causal and non-causal blocks: covered (Q12, Q13)
  C6  feature x phase: no optional features; phases x KV dtype in
      test_correctness.py
  C7  branch coverage: run ``--cov=vllm/v1/attention/backends/mla/rocm_aiter_mla``
D. KV cache (test_kv_cache.py)
  D1  write round trip per dtype and block size: covered
  D2  non-unit k scale, and a wrong scale is caught: covered
  D3  -1 slots skipped: covered
  D4  fused rope + cache: tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py
  D5  layouts: N/A (one latent head, single layout)
E. Numerics (test_correctness.py)
  E1/E2 shape x phase x KV dtype x block size, tokens 1-8192, batch 1-128:
        covering set pre-merge, full product under ``-m slow_test``
  E3  block / page-chunk / fp8-prefill tile edges: covered
  E4  long context: 8192-token decode and prefill rows
  E5/E6 sliding window, sinks, soft cap, ALiBi: N/A
  E7  outputs finite; padded rows not read: covered
F. CUDA graphs (test_cudagraph.py, runtime test)
  F1  NONE / PIECEWISE / FULL_DECODE_ONLY / FULL_AND_PIECEWISE: runtime
  F2  replay with new lengths and block tables: covered
  F3  spec-decode verify blocks under full graphs: covered
  F4  FULL requested on a UNIFORM_BATCH backend: runner downgrade, runtime
G. Workspace (test_workspace.py)
  G1  buffers per config: covered; G2 hot-path allocations: listed there;
  G3  builder memory budget: covered
H. Runtime (tests/v1/e2e/attention/test_rocm_aiter_mla_runtime.py)
  H1-H3 scheduler, MRv1/v2, async: covered
  H4  n-gram and DeepSeek MTP spec decode: covered
  H5  DCP (TP2 x DCP2): covered; PCP: not run (gap, needs a PCP-capable
      prefill on ROCm)
  H6  hybrid KDA + MLA (Kimi-Linear block): covered
  H7  mixed head sizes: N/A
  H8  logprobs vs TRITON_MLA: covered, with a negative control
I. Task accuracy and perf
  I1  GSM8K configs: tests/evals/gsm8k/configs/DeepSeek-R1-*_MI355.yaml,
      Kimi-K3-pruned75-DSpark-AITER-TP4.yaml
  I2  benchmarks/attention_benchmarks/configs/rocm_aiter_mla_*.yaml
"""
