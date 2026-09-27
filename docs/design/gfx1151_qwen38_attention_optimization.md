# gfx1151 Qwen3.8-27B Attention Optimization Plan

## Scope

Add an isolated attention implementation for AMD gfx1151, initially limited to:

- `Qwen/Qwen3.8-27B` and W4A16 variants using
  `Qwen3_5ForConditionalGeneration`;
- `incoai/Qwen3.8-27B-DFlash2` and compatible DFlash2 checkpoints;
- tensor parallel size 1 on ROCm;
- BF16 activations and BF16 KV cache initially.

Do not change the generic ROCm attention behavior. Unsupported shapes, dtypes,
cache layouts, or devices must fall back to the existing backend.

## Current execution paths

The target model is hybrid:

- full-attention layers instantiate `Qwen3NextAttention` from
  `Qwen3_5DecoderLayer`;
- linear-attention layers instantiate `QwenGatedDeltaNetAttention`;
- full attention selects `RocmAttentionBackend` (`ROCM_ATTN`);
- the current paged decode path can fall back to Triton on gfx1151;
- Qwen GDN prefill uses Triton/FLA;
- Qwen GDN decode uses Triton because
  `fused_gdn_decode_post_conv_mtp` is not built.

DFlash2 uses `DFlashQwen3Attention`. It pre-populates context K/V from target
hidden states, then runs query attention through the generic `Attention`
wrapper. Its workload differs from normal decoding because target verification
uses blocks of speculative query tokens and individual draft layers may be
full-attention or sliding-window attention.

## Proposed isolated classes

### Model-facing classes

Add the following without modifying the behavior of the existing classes:

- `Gfx1151Qwen3_5Attention`
- `Gfx1151QwenGatedDeltaNetAttention`
- `Gfx1151DFlash2Attention`

Place model-facing selection helpers in a gfx1151-specific module, for example:

```text
vllm/model_executor/models/gfx1151_qwen3_5_attention.py
```

`Qwen3_5DecoderLayer` and `DFlashQwen3DecoderLayer` should instantiate these
classes only when all capability checks pass. Keep the original classes as the
fallback and avoid scattering architecture checks through model code.

### Backend class

Add an attention backend isolated from `RocmAttentionBackend`, for example:

```text
vllm/v1/attention/backends/gfx1151_qwen_attn.py
```

with:

- `Gfx1151QwenAttentionBackend`
- `Gfx1151QwenAttentionMetadata`
- `Gfx1151QwenAttentionImpl`

Register it as a distinct backend rather than adding gfx1151 branches inside
the generic ROCm implementation. Selection must require the gfx1151 device
target and the supported Qwen/DFlash configuration. An explicit environment or
CLI override should allow disabling it for A/B testing.

The backend should preserve vLLM's cache and scheduler contracts. It must not
introduce a model-private KV-cache representation unless profiling proves the
layout conversion is amortized.

## Kernel work

Use HIP as the primary production implementation for gfx1151 decode, DFlash2
verification, GDN state updates, and KV-cache operations. These workloads are
small or irregular and require explicit control of wave32 decomposition, LDS,
vector loads, occupancy, and cache behavior.

Use Triton to establish readable reference kernels and correctness baselines,
for generic fallback, and for regular prefill workloads when measurement shows
it is competitive. A Triton prototype does not need to be replaced when it
meets the performance target and matches or beats HIP after dispatch overhead.

Treat FlyDSL from `docker/Dockerfile.rocm_base` as optional. Use it only when it
compiles and links reliably for gfx1151, expresses the operation cleanly, and
beats or matches the HIP/Triton alternatives. FlyDSL availability must not
block the backend, become a mandatory runtime dependency, or determine the
fallback hierarchy.

The implementation preference is therefore workload-specific:

| Workload | Primary | Secondary/fallback |
| --- | --- | --- |
| Paged single-token decode | HIP | existing ROCm/Triton backend |
| DFlash2 block verification | HIP | Triton reference/fallback |
| Fused GDN decode/state update | HIP | current Triton GDN path |
| KV-cache update and decode fusion | HIP | existing vLLM operations |
| RoPE/QK-normalization fusion | HIP | existing native operations |
| Long, regular attention prefill | measured HIP or Triton winner | existing backend |
| Unsupported shapes or configurations | existing generic backend | none |

Prototype in Triton when that shortens validation, then write a specialized HIP
kernel only after profiling identifies a material opportunity. Select the final
path by measured end-to-end performance, not implementation language alone.

### Full-attention prefill

Implement or specialize a variable-length causal attention kernel for:

- BF16 Q/K/V;
- full-attention head dimension 256 (24 query heads, 4 KV heads);
- grouped-query attention used by Qwen3.8-27B;
- sequence lengths centered on 1K to 4K initially;
- M-RoPE-compatible Q/K input (RoPE remains a separate operation initially);
- contiguous and paged KV inputs required by vLLM prefill.

These dimensions come from the target checkpoint's `text_config`. The GDN
key/value head dimension is 128; do not confuse it with softmax attention's
256-dimensional heads. Read the draft checkpoint's configuration separately.

Start with online softmax and tiled QK/PV computation. Tune block sizes,
wavefront count, LDS use, and vector widths specifically for 40 gfx1151 CUs.
Do not assume CUDA warp size or NVIDIA shared-memory behavior.

### Full-attention decode

Add a gfx1151 paged-attention decode kernel supporting:

- query batch sizes 1, 2, 4, and 8;
- Qwen GQA head mapping;
- the active LBHNC/paged KV-cache layout;
- non-partitioned short-context decode and partitioned long-context decode;
- attention sinks when configured;
- causal and sliding-window limits;
- graph capture with stable addresses and no host synchronization.

Fuse cache reads, QK reduction, online softmax, and PV reduction. Provide tuned
configurations by context-length bucket instead of one universal launch shape.

### DFlash2 verification attention

DFlash2 requires a separate fast path rather than treating verification as
ordinary prefill. Optimize query block sizes:

```text
M = 4, 8, 16, 24, 32
```

The implementation must support both non-causal full attention and the draft's
configured causal/sliding-window modes. Context K/V is already present in the
cache, while the query block contributes its own K/V. Avoid materializing a
concatenated context-plus-query tensor.

Add a dedicated `Gfx1151DFlash2Attention` wrapper so draft-specific metadata
and verification kernels remain separate from normal target attention.

### Qwen GDN

Build a gfx1151 implementation for the missing fused decode operation equivalent
to `fused_gdn_decode_post_conv_mtp`. The first version should fuse:

- short convolution state update;
- projection unpacking;
- recurrent GDN state update;
- normalization and output packing where numerically safe.

Retain the current Triton/FLA prefill path until profiling shows it is a
bottleneck. GDN is not softmax attention and should remain a separate backend
component even though model-facing selection is grouped under the gfx1151
Qwen attention class.

## Selection and fallback rules

Enable the new path only when all of the following hold:

1. ROCm reports architecture `gfx1151`.
2. The model architecture is the supported Qwen3.5/Qwen3.8 target or DFlash2.
3. Head dimension, dtype, KV-cache dtype, cache layout, and attention type are
   supported by the selected kernel.
4. Tensor-parallel and pipeline-parallel constraints are satisfied.

Every kernel entry point must validate its contract before launch. Unsupported
cases should use the existing `Attention`, `RocmAttentionBackend`, or Triton GDN
path. Do not silently reinterpret strides or cache layouts.

## Correctness requirements

Compare each kernel against the existing backend across:

- batch sizes 1, 2, 4, and 8;
- prompt lengths around page boundaries and 1K/2K/4K lengths;
- decode positions at the start, middle, and end of a cache page;
- full, causal, non-causal, and sliding-window DFlash attention;
- target verification blocks of 4 through 32 tokens;
- graph-captured and eager execution;
- dense BF16 and W4A16 model projections.

Use relative and absolute error gates appropriate for BF16 and require no
NaNs/Infs. End-to-end greedy token IDs must match the current backend on a
fixed prompt corpus. During optimization, use a 100-question, 5-shot GSM8K
smoke test on fixed question IDs 0..99 with the matched generation/scoring
protocol. Per the user's updated workflow, run full 1,319-question GSM8K only
after optimization is finished, comparing original BF16 and optimized W4A16.
Smoke results do not satisfy the final full-dataset accuracy requirement.

## Performance validation

Benchmark kernels independently before end-to-end serving tests. Record median
and high-percentile latency after warmup, without profiling overhead in final
numbers.

Required serving matrix:

```text
vllm bench serve
input length:  1024
output length: 1024
temperature:   0
concurrency:   1, 2, 4, 8
```

Also measure realistic ShareGPT/GSM8K prompts because speculative acceptance
changes the optimal verification block size.

Initial targets:

- eliminate the Triton paged-attention fallback on supported gfx1151 shapes;
- reduce full-attention decode latency by at least 20%;
- reduce DFlash verification-attention latency by at least 25% for M=8..24;
- reduce GDN decode launch and memory overhead by at least 20%;
- preserve the measured concurrency-1 result above 40 output tokens/s;
- aim for 70--90 decode tokens/s on high-acceptance concurrency-1 workloads.

Reject a specialized kernel if it does not beat the existing path by a
repeatable margin after including dispatch overhead.

## Profiling order

1. Profile target-only W4A16 decode and identify time in full attention versus
   GDN and linear layers.
2. Profile DFlash2 at 7, 15, and 23 speculative tokens, separating draft,
   target verification, rejection sampling, and scheduler overhead.
3. Implement the paged full-attention decode kernel.
4. Implement the DFlash verification kernel.
5. Implement fused gfx1151 GDN decode.
6. Re-profile and tune only the shapes engaged by Qwen3.8-27B.
7. Run unprofiled end-to-end performance, using 100-question GSM8K smoke tests
   during optimization; run full GSM8K comparisons only after optimization.

## Build fallback

If HIP source changes require reinstalling vLLM, use the project/workstation
workflow requested for gfx1151:

```bash
python3 -m pip uninstall -y vllm
rm -rf build/ .deps vllm.egg-info vllm/*.so
PYTORCH_ROCM_ARCH=gfx1151 python3 setup.py develop
```

Before upstream submission, reconcile this local workflow with `AGENTS.md`,
which requires the repository's `uv` and incremental-build process. Preserve
unrelated local modifications and run the duplicate-PR checks required there.

## Deliverables for the main thread

1. Isolated gfx1151 model-facing attention classes.
2. A separately registered gfx1151 attention backend with strict fallback.
3. Production HIP kernels for paged decode and DFlash verification, with
   Triton reference/fallback implementations where useful.
4. A fused gfx1151 Qwen GDN decode implementation.
5. Correctness tests extending existing attention and DFlash suites.
6. Kernel benchmarks under `benchmarks/kernels/`.
7. The 1K/1K concurrency 1/2/4/8 serving matrix.
8. Full GSM8K results for original BF16 and optimized W4A16 configurations.
