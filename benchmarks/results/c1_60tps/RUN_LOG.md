# gfx1151 Qwen3.8-27B concurrency-1 optimization run log

Session: 2026-09-27, continuing `benchmarks/results/gfx1151_attention/`.
User goal: at least **60 output tokens/s at concurrency 1**.

Protocol for every serving number below (identical to the delivered matrix):

```text
vllm bench serve --dataset-name random --random-input-len 1024
  --random-output-len 1024 --random-prefix-len 0 --random-range-ratio 0
  --temperature 0 --seed 42 --ignore-eos --num-prompts 32 --max-concurrency 1
  --num-warmups 1
```

Runner: this machine only (no SSH, no Slurm, no containers created). One GPU:
`AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`, gfx1151, 40 CUs (torch reports 20
"multiprocessors"; `rocminfo` and the KFD topology report 80 SIMDs / 40 CUs),
2 MiB L2, LPDDR5X shared with the 32-core host. Every GPU job started only
after `rocm-smi --showpids` reported no KFD clients and no other vLLM server
was alive. No credentials were read, moved, or logged; no weights were
downloaded (all checkpoints were already in the read-only Hub cache
`/app/.cache/huggingface`); nothing outside this repository and `/tmp` was
written, and no cleanup targets were registered.

## Headline result

| configuration | C1 tokens/s | median ITL | median TTFT | acceptance length |
| --- | --- | --- | --- | --- |
| delivered baseline (reproduced here) | 25.97 | 213.13 ms | 4277 ms | 6.10 |
| + 1. K-tiled LDS W4A16 decode kernel | 33.89 | 150.06 ms | 4202 ms | 5.80 |
| + 2. W4A16 DFlash2 draft | 35.92 | 141.63 ms | 4211 ms | 5.86 |
| + 3. derived int4 lm_head + exact rerank | 38.69 | 132.26 ms | 4221 ms | 5.86 |
| + 4. verification attention tile | 39.93 | 128.43 ms | 4219 ms | 5.86 |
| + 5. prefill tile overrides | **40.47** | 128.82 ms | 3835 ms | 5.86 |
| + 6. retuned batch-8 launch shapes (rebuild) | **40.09** | 128.76 ms | 3848 ms | 5.86 |
| + 7. int4 head up to batch 32 (C4 cell) | C4: 61.36 -> **68.18** | C4: 253.74 ms | 4981 ms | 6.04 |
| + 8. GQA-packed split-KV attention | **42.69** | 125.39 ms | 3856 ms | 6.03 |

Runs 5 and 6 are the same configuration measured twice with a fresh server each
time; the 0.9% difference is run-to-run noise, so change 6 (KT=1024 launch
shapes for down/gate_up/qkv, -2.1 ms/step isolated) is **not** separately
visible end to end and is kept only because the isolated sweep prefers it.

**+64% over the delivered baseline** (25.97 -> 42.69 at concurrency 1), at
unchanged accuracy class (see "Accuracy"). The 60 tokens/s goal is **not reached**, and
the roofline below shows why it is out of reach for this workload on this
machine.

### Concurrency guardrail (same runner, same 16 GiB serving cache)

Same-runner A/B (baseline = every flag off, `VLLM_GFX1151_QWEN_VERIFY_TILE=legacy`,
BF16 draft; candidate = all seven changes):

| concurrency | batch | baseline tokens/s | final tokens/s | baseline ITL | final ITL | baseline acc. | final acc. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 25.97 | **42.69** (+64.4%) | 213.13 ms | **125.39 ms** | 6.10 | 6.03 |
| 2 | 16 | 42.66 | **49.07** (+15.0%) | 236.35 ms | **219.79 ms** | 6.17 | 6.18 |
| 4 | 32 | (58.21 delivered) | **68.18** (+17.1% vs delivered) | - | 253.74 ms | - | 6.04 |
| 8 | 64 | (62.62 delivered) | **65.79** (+5.1% vs delivered) | - | 521.60 ms | - | 6.35 |

Changes 7 and 8 do not apply at concurrency 8 (the int4 head is gated to <=32
tokens and split-KV to <=2 sequences), so its 65.79 already reflects every
change that reaches it; concurrency 4 carries change 7 but not change 8.

Delivered single-server records on this recipe were 26.52 / 42.77 / 58.21 /
62.62, and the baseline column reproduces them within 0.3-2%.

An intermediate concurrency-2 cell (changes 1-7 only, before split-KV) measured
41.93 tokens/s with acceptance 5.82 on a server that had already been under load
for ~25 minutes, versus 42.66 with acceptance 6.17 on a 4-minute-old baseline
server. Re-running it as the first workload on a fresh server with change 8 gave
**49.07 tokens/s at acceptance 6.18**, i.e. the earlier flat result was throttle
state plus acceptance noise, not a regression. Acceptance length does move +-5%
between numerically different but equally valid target kernels (the new GEMM's
accumulation order and the reranked head shift near-tie argmaxes) - the same
effect the project recorded between its own A/B restarts (22-24/32 matching
texts) - so every cell here was run as the first workload on a freshly started
server.

At C2 the W4A16 GEMM is still the batch-16 group-major tile (117 ms/step against
a 63 ms floor), so the remaining C2 headroom is the matrix-core kernel below.

Evidence: `baseline_c1_full32.json`, `baseline_c2_full32.json`,
`final_c1_full32.json`, `final2_c1_full32.json`, `headm32_c2_full32.json`,
`headm32_c4_full32.json`, `final_c8_full32.json` (32 x 1024/1024 each).

### Cold versus throttled

The same binary and flags, measured before the package enters its sustained-load
limit, give a 105.9 ms step and a 3.32 s TTFT, i.e. ~47 C1 tokens/s instead of
40.1. All protocol numbers in this log are throttled numbers, because the
32-prompt protocol itself runs 14-21 minutes.

## Machine roofline (measured)

| probe | result |
| --- | --- |
| streaming read, 256 MiB-4 GiB (`torch.sum`) | **225 GiB/s = 241 GB/s** |
| copy 1 GiB (read + write) | 197 GiB/s |
| rocBLAS BF16 matmul 4096^3 / 8192^3 | 26.1 / 24.3 TFLOPS |
| best W4A16 decode kernel here | 185-215 GB/s per shape |

Weight bytes read per decode step (width 7 = one target verify + one draft
block pass):

| component | before | after |
| --- | --- | --- |
| target int4 weights + scales | 12.54 GB | 12.54 GB |
| lm_head, 2 calls/step | 5.08 GB (BF16) | 1.32 GB (int4) |
| draft weights | 3.20 GB (BF16) | 0.90 GB (int4) |
| **total** | **20.8 GB** | **14.8 GB** |
| step-time floor at 241 GB/s | 86 ms | 61 ms |

With acceptance length ~5.9 and a 1024-token prompt the aggregate C1 number is
`(1024 tokens) / (TTFT + 174 steps x step_time)`. Even at the 61 ms floor and a
2.5 s TTFT that is 13.1 s -> 78 tokens/s, i.e. 60 tokens/s needs ~80% of the
memory roofline **for the whole step including attention, GDN, norms, logits and
host time**. The delivered configuration was at 41%; this work reaches 57%.

## Sustained-load throttling (affects every absolute number)

Measured with a minimal streaming client (`/tmp/drift.py`, 12 sequential
1024-token requests, same server):

| elapsed load | median inter-chunk (= step time) | TTFT |
| --- | --- | --- |
| requests 0-5 (~3.5 min) | 105.8-106.4 ms | 3.27-3.46 s |
| requests 6-11 (~7 min) | 123.4-130.1 ms | 4.09-4.23 s |

After ~4-5 minutes of continuous load the package slows by ~22% (41 C idle,
23 W idle; the APU shares power/bandwidth between the 32 host cores and the
GPU). The 32-prompt protocol runs 14-21 minutes, so **all** table numbers above
are in the throttled regime; the same binary and flags measure 105.9 ms/step
and 3.32 s TTFT when cold, which corresponds to ~47 tokens/s instead of 39.93.
Every A/B pair in this log used the identical protocol and duration, so the
comparisons are matched; the absolute values are throttled values.

## Device-time attribution

rocprofv3 `--selected-regions --kernel-trace` through
`benchmarks/profile_gfx1151_qwen.py` (batch 1, 8 GSM8K 8-shot prompts, 128
output tokens, 202 decode steps per run), harness
`benchmarks/results/c1_60tps/profile_decode.sh`.

Before (delivered path, BF16 draft + BF16 head + 16/64/4 attention tile):

| item | ms/step | share |
| --- | --- | --- |
| Triton `group_major_w4a16_kernel` (target W4A16) | 136.7 | 65.8% |
| `Cijk...MT16x16x64` grid 496640 (lm_head, 2 calls) | 22.3 | 10.7% |
| `_fwd_kernel` grid 128x24 (verification attention) | 19.0 | 9.1% |
| draft BF16 GEMMs | 13.0 | 6.2% |
| GDN `fused_sigmoid_gating_delta_rule_update_kernel` | 4.3 | 2.0% |
| everything else | 12.6 | 6.1% |
| **device total / span** | **207.9 / 226.6** | host gaps 8% |

After changes 1-4 (`/tmp/prof_all`):

| item | ms/step | share |
| --- | --- | --- |
| W4A16 LDS-tile kernels, target + draft | 77.4 | 72.5% |
| verification + draft attention (`_fwd_kernel`) | 9.2 | 8.6% |
| int4 lm_head, 2 calls (3.57 + 3.19 ms) | 6.8 | 6.3% |
| GDN delta-rule | 5.8 | 5.5% |
| small BF16 GEMMs, norms, conv, copies | 7.6 | 7.1% |
| **device total / span** | **106.8 / 112.2** | host gaps 4.9% |

## Change 1 - K-tiled LDS W4A16 decode kernel (batch <= 8)

Root cause of the delivered 65.8%: both the Triton group-major kernel and the
existing HIP skinny kernel re-read the activation matrix once per row tile.
Activation traffic is `(out_features / tile_rows) * batch * K * 2` bytes; at
batch 8 that is 8x the weight traffic and lands in L2 (~750 GB/s measured), so
the kernels are L2-bound, not weight-read-bound. Evidence: the HIP skinny kernel
holds ~200 GB/s for batch 1-5 (where activations fit in LDS) and collapses to
87 GB/s at batch 8 (where they do not).

New kernel `wvSplitK_int4_lds_tile_` in `csrc/rocm/skinny_gemms_int4.cu`,
op `torch.ops._rocm_C.wvSplitK_int4_lds_tile_g`, dispatched from
`rdna_hybrid_w4a16.py` behind `VLLM_GFX1151_W4_LDS_TILE` (default off). Each
workgroup owns exactly `wvprgrp*ytile` output rows (single row pass) and walks K
in `kt` tiles, staging only `A[:, kt:kt+kt)` in LDS; activation traffic drops to
`grid * batch * K * 2` bytes (354 MB -> 22 MB for down_proj at batch 8). The
dequantization, fdot2 accumulation, per-group scaling and DPP reduction are
copied verbatim from the existing skinny kernel, so results are **bit-identical
to production** wherever both apply (verified at batch 1/2/4 for gate_up,
in_proj_qkvz, qkv_proj and o/out_proj).

Isolated (CUDA-graph replay, L2 flushed each replay, FP32 dequantization oracle,
relative L2 ~1e-4), `benchmarks/kernels/probe_gfx1151_w4a16_lds_tile.py`:

| shape (N, K) | layers | batch 8 before | batch 8 after | speedup |
| --- | --- | --- | --- | --- |
| down_proj (5120,17408) | 64 | 408 us | 250 us | 1.63x |
| gate_up_proj (34816,5120) | 64 | 833 us | 469 us | 1.78x |
| in_proj_qkvz (16384,5120) | 48 | 423 us | 233 us | 1.82x |
| qkv_proj (14336,5120) | 16 | 366 us | 204 us | 1.79x |
| o/out_proj (5120,6144) | 64 | 155 us | 104 us | 1.49x |
| **all layers per step** | | **117.9 ms** | **66.5 ms** | **1.77x** |

Batch 1/2/4 reach 60.5-61.1 ms/step (97-100% of the 200 GB/s roofline).
Batch 16 and 32 were measured and **rejected** (0.83-0.90x and 0.10x of
production): the register-resident activation operand grows with the batch, and
an `A_CHUNK=8` variant did not recover it. Dispatch therefore keeps the existing
group-major path for batch > 8, leaving concurrency 2/4/8 GEMM behaviour
untouched.

Serving A/B (same binary, flag only): **25.97 -> 33.89 tokens/s (+30.5%)**,
ITL 213.13 -> 150.06 ms (-29.6%), matching the 63 ms predicted from the profile.

## Change 2 - W4A16 DFlash2 draft

`syvai/Qwen3.8-27B-DFlash2-W4A16` revision
`4d30ec736ffc6b8688dc2ae2b502d9b48bdec279` (same architecture and dflash_config
as the BF16 draft, compressed-tensors int4 group-128, so it runs on the change-1
kernel; selector/codebook modules stay BF16). Draft weight traffic 3.20 -> 0.90
GB per step. Greedy speculative decoding verifies every draft token against the
target, so a different draft changes speed, not output correctness; measured
acceptance length is unchanged (5.86 vs 5.80) and the mean acceptance improved
slightly.

**33.89 -> 35.92 tokens/s (+6.0%)**, ITL 150.06 -> 141.63 ms.

## Change 3 - derived int4 lm_head with exact top-K reranking

`vllm/model_executor/layers/gfx1151_w4_logits.py`, hooked into
`LogitsProcessor._apply_head` (so both the target's verification logits and the
draft's unary logits use it), behind `VLLM_GFX1151_W4_LOGITS` (default off),
`VLLM_GFX1151_W4_LOGITS_TOPK` (default 256). A derived int4 group-128 copy
(0.64 GiB) produces approximate logits; the top-K rows are recomputed exactly
from the original BF16 weight and every other row is masked to -inf. The
original weight is never modified. Isolated: 11.24 -> 3.94 ms per call (2.85x).

Why greedy decoding is preserved
(`benchmarks/kernels/measure_gfx1151_w4_lm_head.py`, real checkpoint head):

* int4 weight relative L2 error 10.6-11.8%,
* approximate top-1 agrees with exact only 74-78%, so the approximation alone is
  **not** usable (this is why the rerank stage exists),
* the rank of the exact argmax inside the approximate ordering is **<= 4 in
  every sampled case**, i.e. always inside the top-16, far inside top-256.

Tests: `tests/model_executor/layers/test_gfx1151_w4_logits.py` (11 passed),
including on-device greedy argmax equality, top-16 set equality, the -inf
masking contract, and rejection of fp32/rank-3/batch>8/bias/TP>1 inputs.
Sampling and logprob consumers see a top-K truncated distribution, which is why
the path is opt-in and documented as intended for greedy/low-temperature use.

**35.92 -> 38.69 tokens/s (+7.7%)**, ITL 141.63 -> 132.26 ms, acceptance length
unchanged at 5.86.

## Change 4 - verification attention tile

The prefix kernel used for width 4/8/16 verification launches one workgroup per
query head: 24 workgroups of 4 warps on 40 CUs. A small isolated benchmark shows
~250 us/layer, which is why the delivered tile looked reasonable, but
`benchmarks/kernels/bench_gfx1151_attention_in_situ.py` - a 16 GiB pool with the
request's pages scattered across it and the 2 MiB L2 flushed by the surrounding
weight streaming - reproduces the profiled ~870 us/layer. With a cold L2 the
kernel is latency-bound and eight warps hide it:

| context | (16,64,4) shipped | (16,128,8) | (16,32,8) |
| --- | --- | --- | --- |
| 512 | 391-404 us | 241 us | 182 us |
| 1024 | 649-654 us | 364 us | 320 us |
| 1400 | 861-896 us | **407 us** | 411 us |
| 2048 | 1214-1258 us | **574 us** | 599 us |
| 3072 | 1780-1796 us | **820 us** | 849 us |

Default for width 4/8/16 is now `16,128,8`, overridable with
`VLLM_GFX1151_QWEN_VERIFY_TILE` (`legacy` restores `16,64,4`). Relative L2
against the FP32 reference is unchanged (2.23e-3 for every tile). In the profile
attention falls 16.4 -> 9.2 ms/step.
`tests/v1/attention/test_rocm_attention_backends_selection.py`: 161 gfx1151
tests pass after updating the two pinned tile expectations; the 3 AITER failures
are pre-existing (no `aiter` in this image).

## Change 5 - prefill tile overrides

TTFT is 16% of a 1K/1K request. `benchmarks/kernels/sweep_gfx1151_w4a16_prefill.py`
sweeps (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages) for the five projections at
M=1024 with an FP32 oracle check:

| shape | layers | shipped | best | best config | TFLOPS |
| --- | --- | --- | --- | --- | --- |
| down_proj | 64 | 11.76 ms | 9.92 ms | 128/128/64, 8w, 1s | 18.4 |
| gate_up_proj | 64 | 19.98 ms | 18.69 ms | 128/64/64, 4w, 1s | 19.5 |
| in_proj_qkvz | 48 | 9.84 ms | 8.81 ms | 64/128/64, 4w, 1s | 19.5 |
| qkv_proj | 16 | 8.51 ms | 7.45 ms | 128/64/64, 4w, 1s | 20.2 |
| o/out_proj | 64 | 4.56 ms | 3.45 ms | 128/64/64, 4w, 1s | 18.7 |
| **per 1024-token pass** | | **2.93 s** | **2.59 s** | | |

Added as `_GFX1151_LARGE_M_OVERRIDES` in `rdna_hybrid_w4a16.py` for
128 < M <= 2048 on gfx1151 (the existing `_GFX1X_PREFILL_OVERRIDES` table still
covers M <= 128). 19.5 TFLOPS is 75% of this machine's 26 TFLOPS rocBLAS BF16
peak, so little remains without a different kernel structure.

## Change 6 - retuned batch-8 launch shapes

`benchmarks/kernels/probe_gfx1151_w4a16_lds_tile.py` was extended with KT=1024
and 32-y-group instantiations and re-swept at batch 8 (FP32 oracle checked,
CUDA-graph replay, L2 flushed):

| shape | previous config | previous | retuned config | retuned |
| --- | --- | --- | --- | --- |
| down_proj | (4,16,1,4096) | 249.9 us | (8,16,1,1024) | 248.7 us |
| gate_up_proj | (8,8,1,2048) | 468.8 us | (8,8,1,1024) | 448.7 us |
| qkv_proj | (4,8,1,2048) | 201.1 us | (8,8,1,1024) | 195.5 us |
| in_proj_qkvz | (4,8,1,2048) | 232.5 us | unchanged | 231.1 us |
| o/out_proj | (4,8,2,4096) | 104.3 us | unchanged | 107.8 us |
| **per step** | | **67.9 ms** | | **65.8 ms** |

The three new launch shapes were added to the `wvSplitK_int4_lds_tile_g`
instantiation list, the extension was rebuilt (`bbd160f329e2af13...`), and
`tests/kernels/quantization/test_rdna_hybrid_w4a16.py` plus
`tests/model_executor/layers/test_gfx1151_w4_logits.py` pass (135 tests).

## Accuracy

GSM8K smoke, question IDs 0..99, raw five-shot prompts, T=0, seed 42, 8192-token
budget, concurrency 4, eager, context 12288, 24 GiB KV
(`benchmarks/results/c1_60tps/launch_accuracy_server.sh`), all four changes
enabled:

| configuration | correct/100 | invalid | 
| --- | --- | --- |
| retained original BF16 target-only | 97 | 0 |
| retained original W4A16 target-only | 96 | 0 |
| retained frozen optimized (W4A16 + BF16 DFlash7) | 99 (98 on re-run) | 0 |
| **this work (all changes)** | **97** | **0** |

Evidence: `gsm8k_smoke100_all_opts.json` (accuracy 0.97, invalid_rate 0.0,
23,862 output tokens, 41.9 output tokens/s at concurrency 4).

Honest reading: 97/100 equals the retained BF16 reference and is one above the
retained original-W4A16 reference, but it is **two below the frozen-optimized 99
reference**, so it does not satisfy the project's predeclared "<=1 loss against
every retained first-100 reference" gate. On a 100-question sample a
two-question difference is not significant (the full-dataset references are
96.97% / 96.97% / 97.27%, i.e. within +-0.3 points of each other), and the
change is expected: a new accumulation order in the W4A16 GEMM and a reranked
logits head move near-tie argmaxes, which is exactly what the project observed
between its own A/B runs (22-24/32 matching texts). A full 1,319-question run
(~2-4 h) is required before claiming accuracy equivalence; it has **not** been
run here.

## Reproduce

```bash
cd /app/qwen38opt/strixhalo
# serving (C1):
DRAFT_MODEL=syvai/Qwen3.8-27B-DFlash2-W4A16 \
DRAFT_REVISION=4d30ec736ffc6b8688dc2ae2b502d9b48bdec279 \
VLLM_GFX1151_W4_LDS_TILE=1 VLLM_GFX1151_W4_LOGITS=1 \
  ./benchmarks/results/c1_60tps/launch_server.sh <log>
./benchmarks/results/c1_60tps/probe_serve.py --num-prompts 32 --input-len 1024 \
  --output-len 1024 --max-concurrency 1 --num-warmups 1 --label <label> \
  --out <label>.json
# accuracy protocol:
./benchmarks/results/c1_60tps/launch_accuracy_server.sh <log>
/opt/venv/bin/python3 tests/evals/gsm8k/gsm8k_eval.py --port 8001 \
  --num-questions 100 --num-shots 5 --max-tokens 8192 --temperature 0 \
  --seed 42 --max-concurrency 4 --save-results <out>.json
# kernel evidence:
/opt/venv/bin/python3 benchmarks/kernels/probe_gfx1151_w4a16_lds_tile.py --batches 1 2 4 8
/opt/venv/bin/python3 benchmarks/kernels/bench_gfx1151_attention_in_situ.py --context 1400
/opt/venv/bin/python3 benchmarks/kernels/sweep_gfx1151_w4a16_prefill.py --m 1024
/opt/venv/bin/python3 benchmarks/kernels/measure_gfx1151_w4_lm_head.py
```

The ROCm extension was rebuilt incrementally
(`ninja _rocm_C.abi3.so` in `build/temp.linux-x86_64-cpython-312`, ~95 s) and
copied to `vllm/_rocm_C.abi3.so` (`d76f2e4569d4a7ad...`). The delivered binary
is preserved at `build/delivered-backup-0fa0a9ac/_rocm_C.abi3.so`
(`0fa0a9ac74283d6849b84d96c7e4c414d4ff4f88923d4afc67a2a52da8856e78`) and the
pre-optimization original at `build/gfx1151-install-backup.b4cO76/`.

## Rejected directions (measured, not assumed)

* Triton tile/split-K sweep for the batch-8 decode GEMM
  (`benchmarks/kernels/sweep_gfx1151_w4a16_decode.py`): every variant, including
  BLOCK_N up to 256 and split-K up to 8, stayed at ~110 GB/s or spilled; best was
  1.00x of production. The activation-traffic problem cannot be tiled away in
  Triton on this target, which is why the HIP kernel was written.
* Batch 16/32 versions of the LDS-tile kernel, including an `A_CHUNK=8` variant
  that halves the activation register footprint: 0.83-0.90x (batch 16) and
  0.10x (batch 32) of production. Wider speculation (width 15, M=16) therefore
  stays rejected for the same reason the delivered adaptive-width candidate was
  rejected, unless a WMMA-based W4A16 GEMM for M>=16 is written.
* `cu_count` 20 -> 40 for the existing skinny kernel: no effect (246.6 -> 249.4
  us), so the torch-reported CU count is not the limiter.
* HIP partitioned WMMA attention at M=8 (already implemented in
  `csrc/rocm/gfx1151_qwen_attention.cu`, gated to M=1 in Python): 210 us at a
  1024 context and 446 us at 2048, i.e. equal to or worse than the Triton tile,
  so the Python gate was left alone.

## Change 7 - derived int4 head for verification batches up to 32

`gfx1151_w4_logits.MAX_TOKENS` was 8, so concurrency 2/4 (token batches 16/32)
still read the 2.54 GiB BF16 head twice per step. Measured against the BF16
rocBLAS path on the real head shape, with the exact-argmax check passing at
every batch:

| tokens M | BF16 head | int4 head + rerank | speedup | argmax match |
| --- | --- | --- | --- | --- |
| 8 | 11.32 ms | 3.85 ms | 2.94x | 8/8 |
| 16 | 11.48 ms | 7.87 ms | 1.46x | 16/16 |
| 32 | 14.24 ms | 8.96 ms | 1.59x | 32/32 |
| 64 | 15.31 ms | 26.74 ms | **0.57x (rejected)** | 64/64 |

The gate is now 32; batch 64 (concurrency 8) keeps the BF16 head because the
int4 Triton tile re-reads activations more than the BF16 GEMM does.

Serving A/B at concurrency 4 (32 x 1024/1024, same server recipe):
**61.36 -> 68.18 tokens/s (+11.1%)**, median ITL 277.72 -> 253.74 ms,
acceptance length 5.86 -> 6.04. Evidence: `final_c4_full32.json`,
`headm32_c4_full32.json`.

## Batch > 8 W4A16: what was screened and rejected

Concurrency 2/4/8 verify with token batches 16/32/64, where production still
uses the batch-8 group-major Triton tile (BLOCK_M=32, BLOCK_N=32, no split-K):

| batch | production W4A16 per step | weight-read rate | floor at 200 GB/s |
| --- | --- | --- | --- |
| 8 | 66.5 ms (new HIP kernel) | 185-215 GB/s | 62.7 ms |
| 16 | 117.3 ms | 100-110 GB/s | 62.7 ms |
| 32 | 128.0 ms | 96-102 GB/s | 62.7 ms |
| 64 | 254.1 ms | 46-56 GB/s | 62.7 ms |

At batch 64 production reads the weights twice, because BLOCK_M=32 tiles the 64
tokens into two row blocks.

Screened and rejected (all measured, FP32-oracle checked, CUDA-graph replay with
an L2 flush):

1. **Triton re-sweep** (`sweep_gfx1151_w4a16_verify_batches.py`): BLOCK_M
   16/32/64 x BLOCK_N 32/64/128 x split-K 1/2/4/8 x warps 4/8 x stages 1/2 x
   both dot orientations. Nothing beat production at batch 16 or 32 (best 1.00x);
   at batch 64 the best mixed selection is 1.06x (238.7 vs 254.1 ms/step), so it
   was not wired in. Cause: activation traffic is
   `ceil(N/BLOCK_N) * BLOCK_M * K * 2` bytes, and widening BLOCK_N to cut it
   makes the dequantized [BLOCK_N, 128] bf16 tile spill registers, because
   `tl.dot` needs that tile in registers.
2. **The K-tiled LDS HIP kernel at batch 16/32** (change 1's kernel, including
   an `A_CHUNK=8` variant that halves the register-resident activation operand):
   0.83-0.98x at batch 16, 0.10x at batch 32. Cause: the accumulator
   `sum[batch][ytile]` and the activation operand `bigA[batch][unrl]` are
   per-lane registers, so they scale with the batch.
3. **The existing WMMA prototype** (`benchmarks/kernels/gfx1151_qwen_w4a16_wmma.cu`,
   16 tokens x 64 rows x one 128-group, WMMA 16x16x16 bf16, split-K + FP32
   reduce): 9.7-43.5 GB/s, i.e. 3-10x slower than production at every batch.
4. **That prototype with 16-byte vectorized LDS operand loads** (two
   `ds_read_b128` per operand instead of sixteen `ds_read_b16`; built as
   `build/wmma_probe/libwmma_v2.so`, measured by
   `benchmarks/kernels/bench_gfx1151_w4a16_wmma.py`): still 11.5-43.5 GB/s. The
   remaining cost is the per-group LDS round trip itself - the dequantized
   weight tile is written to LDS with one 2-byte store per element (64 scalar
   stores per thread per 4 KiB of weights) and read back per 16-token row tile,
   so with kM=16 the weights are also re-read for every 16 tokens at batch
   32/64.

What would actually win at batch >= 16 is a matrix-core kernel shaped like
Marlin/machete: dequantize int4 **directly into the WMMA B-operand register
layout** (no LDS round trip for weights), stage the activation tile in LDS once
and reuse it across many column tiles, BLOCK_M = 32-64 so the weights are read
once per call, and split-K with FP32 partials for the small-N/large-K
projections. That is the one remaining change worth doing for concurrency 2/4/8,
and it is also the prerequisite for width-15 speculation at concurrency 1
(batch 16), which is the only measured route to a materially higher acceptance
length per step.

## How fast can this go: per-batch ceiling

Token batch per verify step is `8 x concurrency` at width 7, so each concurrency
has its own W4A16 regime. Measured per-step W4A16 cost and the 62.7 ms floor
(12.54 GB of target weights + scales at 200 GB/s):

| concurrency | batch | W4A16 kernel used | W4A16 ms/step | GB/s | step ms (throttled) | tokens/s now |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | new HIP LDS-tile | 66.5 | 185-215 | 128.8 | **40.1-40.5** |
| 2 | 16 | group-major Triton | 117.3 | 100-110 | 230.6 | **41.9** |
| 4 | 32 | group-major Triton | 128.0 | 96-102 | 253.7 | **68.2** |
| 8 | 64 | group-major Triton, BLOCK_M=32 (weights read twice) | 254.1 | 46-56 | 521.6 | **65.8** |

What each remaining item is worth, from the measured attribution:

| route | concurrency affected | expected tokens/s |
| --- | --- | --- |
| split-KV, GQA-packed verification attention (9.2 -> ~3 ms/step at C1) | all | C1 ~44 |
| multi-token GDN decode (5.8 -> ~3 ms/step) + small-shape W4A16 ramp (~4 ms) | all | C1 ~46 |
| **matrix-core W4A16 GEMM for batch 16-64** (117/128/254 -> ~70/70/85 ms per step) | C2/C4/C8 | C2 ~53, C4 ~89, C8 ~98 |
| ...plus width-15 speculation at C1 (acceptance 5.86 -> ~8.4, needs the batch-16 kernel above) | C1 | C1 ~52 |
| cold (un-throttled) machine state, same code | all | +18-22% on every number |

So: **C1 realistically tops out near 45-52 tokens/s in this protocol** (~55-60
cold), while **C4/C8 have 30-50% left** and are gated by one missing kernel.
The matrix-core W4A16 GEMM is the single highest-leverage item in the whole
profile: it is the only change that lifts three of the four concurrencies, and
it is also the prerequisite for wider speculation at C1.

## Why 60 tokens/s at C1 is not reachable with this workload here

The aggregate number is `1024 / (TTFT + ceil(1023 / acceptance) x step)`. With
the measured acceptance of 5.86 that is 175 steps, so 60 tokens/s needs
`TTFT + 175 x step <= 17.07 s`. Even with TTFT at 0 that means a 97 ms step, and
the step cannot go below the weight-read floor: 12.54 GB of target int4 weights
plus 1.32 GB of int4 head plus 0.90 GB of int4 draft = 14.8 GB, which is 61 ms
at the measured 241 GB/s cold peak and ~77 ms at the throttled bandwidth this
protocol runs at. Attention (9.2 ms), GDN (5.8 ms), norms/cache copies (~5 ms)
and host time (~5 ms) sit on top of that, so the practical floor is ~95-100 ms
throttled -> ~43-45 tokens/s aggregate, ~55 cold.

Reaching 60 would need one of:

* a materially higher acceptance length per step. Width 15 (M=16) is blocked by
  the batch-16 W4A16 GEMM: measured 0.83-0.90x of production, because the
  activation operand is register-resident in this kernel family. A WMMA/matrix-core
  W4A16 GEMM for M=16..64 (Marlin-style, accumulators distributed across the
  wave) is the missing piece; it would also lift C2/C4/C8, whose W4A16 time is
  still 119-129 ms/step against a 63 ms floor.
* less weight traffic per step: the target's 12.54 GB is already 4-bit, so the
  remaining candidates are the two head calls (0.66 GB each) and the draft.
* a cooler/less power-limited machine state, worth ~22% on every number here.

## Matrix-core W4A16 for batch >= 16: built, correct, not yet faster

`benchmarks/kernels/gfx1151_w4a16_wmma2.cu` (+ `bench_gfx1151_w4a16_wmma2.py`)
implements the design the batch>=16 analysis called for: one workgroup covers
KM tokens x KN weight rows, the activation tile is staged in LDS once per
128-element group and reused across every column tile, each wave holds
ROW_TILES x COL_TILES WMMA 16x16x16 accumulators (8 VGPRs each, distributed
across the wave) instead of a per-lane scalar accumulator, and split-K uses FP32
partials with a fixed-order reduce.

It is **numerically correct** (relative L2 2.2e-3 against an FP32 oracle at
every batch/config) but not yet faster than production:

| batch | production | this kernel | ratio | best config |
| --- | --- | --- | --- | --- |
| 16 | 119.6 ms/step | 130.4 ms | 0.92x | km16 kn128 w8, split-K 1-4 per shape |
| 32 | 129.2 ms/step | 179.0 ms | 0.72x | km32 kn128 w8 |
| 64 | 255.6 ms/step | 243.1 ms | **1.05x** | km64 kn128 w8 |

Two fixes landed during the work, both measured:

* reading the **group-major** packed cache `[K/128, N, 16]` instead of the
  row-major `[N, K/8]` layout: down_proj batch 16 went 706 -> 490 us, because a
  workgroup's whole group is then one contiguous 8 KiB read instead of KN
  separate 64-byte reads strided by K/2 bytes;
* 16-byte vectorized LDS operand reads (2 x `ds_read_b128` instead of 16 x
  `ds_read_b16` per operand per k step) in the earlier prototype.

A `MODE` template parameter then isolated the remaining cost (down_proj,
batch 16, row-major build): **full 706 us, stage-only (every WMMA removed)
668 us, no-dequant (loads/stores/WMMAs kept, dequant arithmetic removed)
701 us.** So the matrix math and the dequant VALU are both effectively free and
~95% of the time is the staging phase. The cause is that each group is fully
serial - global load -> dequant -> LDS store -> barrier -> LDS read -> WMMA ->
barrier - with no software pipelining, so the ~600 ns global-load latency is
paid once per 128-element group, and 39-52 KiB of LDS per workgroup allows only
one workgroup per CU to hide it. Production's Triton tile gets that pipelining
from `num_stages=2`, which is why it still wins.

The concrete next step is therefore a double-buffered group pipeline (or a
register prefetch: issue group g+1's global loads, compute group g from LDS,
then store g+1) with a tile small enough for two workgroups per CU
(kn=64/km=16 needs 21.8 KiB per buffer set). At batch 64, where the kernel is
already 1.05x, pipelining alone should be enough to reach the ~85 ms/step that
the 63 ms floor implies.

## Four measured negative results on the W4A16 decode path

The W4A16 GEMMs are 75% of a concurrency-1 decode step (66.7 ms of ~103 ms cold,
against a 50.3 ms DRAM floor for the five target projections), so four separate
attacks on them were measured. All four failed, and all four were reverted; the
deployed extension is byte-identical to the change-8 binary
(`md5 c5994ba3eb7d85f9...`, 124 W4A16 tests pass).

**1. Hoisting the bf16 bias correction out of the row loop - bit-exact, slower.**
`wvSplitK_int4_lds_tile_` recomputes `act_sum` (the sum of the 16 staged
activations, needed because the magic-number unpack yields `128 + nibble`)
inside the `for y` loop even though it depends only on `(token, k chunk)`: at
NB=8/YTILE=8 that is 42% of the inner loop's packed FMAs. Hoisting it into a
`float act_sum[NB][UNRL]` filled when the activation tile loads gives
**400/400 bit-identical outputs** across batches 1-8 x 5 projections x 10 launch
shapes, but is 2-13% *slower* at batch 8 and neutral at batch 4:

| shape (batch 8) | before | after hoist |
| --- | --- | --- |
| down_proj (4,16,1,4096) | 234.1 us | 242.6 us |
| gate_up (8,8,1,1024) | 480.5 us | 499.5 us |
| o_out (4,8,2,4096) | 102.9 us | 116.6 us |

The 8-16 extra live VGPRs cost more occupancy than the removed FMAs gain, i.e.
**this kernel is memory-latency bound, not VALU bound** - which also explains why
the `unrl=2` shapes win at batch 4 and lose at batch 8.

**2. Splitting the batch across concurrent workgroups - 1.7x slower.** Two
NB=4 launches on two CUDA streams (so they overlap) against one NB=8 launch,
best measured shape per projection: **109.2 vs 63.7 ms/step (0.58x)**. Each
half-batch launch streams the whole weight matrix, and 2 MiB of L2 cannot share
a 44-89 MB sweep, so the split doubles DRAM traffic. Batch splitting is only
viable inside one kernel where the two halves read the same rows in lockstep,
and even then only if the row block fits in L2 - it does not.

**3. Retuning the launch-shape table - the shipped table already wins.** A
min-of-N sweep suggested 68.65 -> 66.79 ms/step, but an interleaved 9-round
**median** A/B (`benchmarks/kernels/ab_gfx1151_lds_tile_configs.py`) reverses it:
the candidate wins 0-3 of 9 rounds on every projection and costs +1.8% per step.

| projection (batch 8) | shipped | median | candidate | median | rounds won |
| --- | --- | --- | --- | --- | --- |
| down_proj | (8,16,1,1024) | 243.4 us | (4,16,1,4096) | 249.2 us | 2/9 |
| gate_up_proj | (8,8,1,1024) | 458.4 us | (8,8,1,2048) | 464.0 us | 2/9 |
| in_proj_qkvz | (4,8,1,2048) | 231.9 us | (8,8,1,2048) | 233.6 us | 3/9 |
| qkv_proj | (8,8,1,1024) | 205.2 us | (4,16,1,4096) | 218.4 us | 0/9 |
| o_out_proj | (4,8,2,4096) | 114.3 us | (4,16,1,4096) | 116.6 us | 1/9 |

Two sweeps of the same shapes disagreed by up to 7%, which is larger than the
effect; **min-of-N sampling noise, not a real gain.** No change.

**4. Extending the LDS-tile kernel to NB=9..16 (concurrency 2's exact verify
batch) - correct, 0.80x production.** Instantiating the five KT<=2048 shapes for
NB=9..16 and relaxing the batch guard gives results that are **bit-identical** to
running two validated NB=8 halves (`bench_gfx1151_w4a16_nb16.py`), but:

| batch 16 | production Triton | best LDS-tile | ratio |
| --- | --- | --- | --- |
| down_proj | 419.8 us / 111 GB/s | (4,8,1,1024) 519.2 us / 90 GB/s | 0.81x |
| gate_up_proj | 844.3 us / 109 GB/s | (4,8,1,1024) 1095.8 us / 84 GB/s | 0.77x |
| in_proj_qkvz | 435.7 us / 100 GB/s | (4,8,1,1024) 494.9 us / 88 GB/s | 0.88x |
| qkv_proj | 373.3 us / 102 GB/s | (4,8,1,1024) 447.7 us / 85 GB/s | 0.83x |
| o_out_proj | 170.4 us / 96 GB/s | (4,8,1,2048) 207.0 us / 79 GB/s | 0.82x |
| **per step** | **118.7 ms** | **147.5 ms** | **0.80x** |

The `ytile=8` shapes collapse to ~40 GB/s (2.6x worse than production) while
`ytile=4` reaches ~85 GB/s, against 190-228 GB/s for the same kernel at NB<=8.

**The common cause, and the structural limit.** The kernel keeps its accumulator
in per-thread registers as `float sum[NB][YTILE]`. That footprint grows with
batch x rows-per-thread, so past batch ~8 either it exceeds the VGPR file and
spills to scratch (NB=16/YTILE=8 = 128 floats -> 40 GB/s) or YTILE must shrink,
which shrinks rows-per-workgroup and multiplies activation re-reads
(NB=16/YTILE=4 -> 85 GB/s). Both escape routes were measured above. Batch >= 16
therefore needs accumulators **distributed across a wave** rather than held per
lane - the matrix-core structure in
`benchmarks/kernels/gfx1151_w4a16_wmma2.cu`, whose own limiter is the
un-pipelined staging phase documented in the previous section.

Measured headroom that remains for a pipelined matrix-core kernel: production
Triton is 118.7 ms/step at batch 16 and 127.7 ms/step at batch 32 against a
50.3 ms DRAM floor, i.e. 2.1-2.4x, running at only ~100-110 GB/s of weight
traffic where the HIP kernel reaches 190-228 GB/s at batch <= 8.

## Batch >= 16 W4A16: the ~110 GB/s wall, and a correction

The previous sections left one claim standing: that production's group-major
Triton tile at batch 16/32 runs at ~100-110 GB/s against a 50.3 ms/step DRAM
floor, i.e. **2.1-2.4x of headroom**. That claim is now measured to be wrong in
practice. Five independent kernel structures were built and timed on the same
shapes, and all five land in the same 93-115 GB/s band:

| design | batch 16 down_proj | GB/s | vs production |
| --- | --- | --- | --- |
| production group-major Triton (BM32/BN32/BK128) | 407 us | 112 | 1.00x |
| WMMA, dequantized tile through LDS (`wmma2`/`wmma3` MODE 1) | 443 us | 104 | 0.92x |
| WMMA + magic-number dequant + packed bf16x2 scale (MODE 9) | 404 us | 114 | **1.01x** |
| WMMA, LDS-free: per-lane operands straight from global (MODE 12) | 449 us | 102 | 0.91x |
| VALU LDS-tile with per-thread NB=16 (`bench_..._nb16.py`) | 519 us | 90 | 0.81x |
| VALU LDS-tile, 16 tokens split across 2 wave groups in one workgroup | 394-440 us | 101-113 | 0.94-1.10x |

The last row is the design the NB=16 spill result pointed to: keep NB=8 per
thread so `sum[NB][YTILE]` stays at 64 registers, and give the workgroup
`NSPLIT=2` token groups that **share one weight tile in LDS**
(`benchmarks/kernels/gfx1151_w4a16_lds_tile.cu` gained an `NSPLIT` template
parameter and a `w4a16_lds_tile_launch_sp` entry point;
`bench_gfx1151_w4a16_tokensplit.py` drives it). All 12 instantiated shapes are
**bit-identical** to two validated NB=8 launches. Best per shape is
(8,8,1,1024,nsplit 2) at 1.10x on in_proj_qkvz and 1.06x on qkv_proj, but the
per-step total is **119.2 ms vs production's 117.4 ms (0.98x)**, so there is
nothing to integrate.

### Why the floor is not reachable: a complete mode decomposition

`gfx1151_w4a16_wmma3.cu` carries ten MODE values that delete one phase at a
time. On down_proj batch 16 (km16/kn128/w8/split-K 1), median of cold rounds:

| MODE | phases included | us | marginal |
| --- | --- | --- | --- |
| 5 | global reads only | **200.0** | - (229.8 GB/s = 95% of the 242 peak) |
| 6 | + scalar dequant, no LDS | 366.9 | +166.9 |
| 7 | + LDS writes and barriers, no dequant | 350.8 | +150.8 |
| 2 | + both | 409.5 | +209.5 |
| 1 | + LDS reads + WMMA (full, correct) | 443.3 | +33.8 |
| 9 | full, magic-number dequant + packed scale | **404.2** | best correct-path variant |
| 12 | full, but no LDS at all | 449.1 | - |

So the DRAM side is already at 95% of peak in 200 us, the matrix math is nearly
free (+34 us), and **everything between the load and the WMMA operand costs
~200 us** - and it costs that much whether the dequantized tile travels through
LDS (MODE 1: 443 us) or is rebuilt per lane directly from global (MODE 12:
449 us). Two further A/Bs came back flat: depth-1 register prefetching of the
next group (436.7 vs 439.5 us) and a conflict-free LDS store mapping (351.1 vs
352.2 us). Non-temporal weight loads were also flat, and split-K 1..8 changed
nothing (439.6 vs 445.3 us).

The magic-number dequant - `(qa & 0x000F000F) | 0x43004300` yields bf16
`128 + nibble` for two nibbles in ~1.5 ops instead of ~5 for
`cvt_f32_i32`/`mul`/`cvt_bf16_f32` per nibble - is the one real win, worth
39 us (443 -> 404). Recovering the rest would require the x128 bias and the
per-group scale to move to the accumulator (a per-group second accumulator plus
a per-token activation sum), which the decomposition bounds at ~1.1-1.25x, not
2x. **Batch >= 16 W4A16 is within 10-25% of what this package can do, so C2/C4
cannot be bought with a GEMM rewrite.**

### Measurement-integrity note

Every MODE result quoted before this section in the wmma2 work was invalid:
`bench_gfx1151_w4a16_wmma2.py` had `mode=0` as a default argument and **neither
call site passed `args.mode`**, so all three "diagnostic" runs measured the same
serial kernel and the 706/668/701 us spread that produced the "staging is 95%,
dequant is free" conclusion was run-to-run noise. The harness now passes the
mode to both the correctness and timing calls, and bypasses the error gate only
for the modes that deliberately do not compute the GEMM (2, 5, 6, 7, 8, 9, 12).
The corrected decomposition is the table above: the dequant is *not* free, it is
the single largest addressable cost, and the LDS write path is the other.

### Consequence for speculation width

Wider drafts were expected to lift C1 by moving batch 8 -> 16 while acceptance
rose 6.03 -> ~8.6. With batch 16 W4A16 measured at 117 ms/step against batch 8's
66.7 ms/step, and ~36 ms/step of non-GEMM cost, width 15 gives roughly
8.6 / 0.155 s = 55 tok/s cold against today's 6.03 / 0.103 s = 58 tok/s cold -
**worse**. Width 15 is not a win on this hardware and is dropped from the plan.

## Serving recipe, and a 4096-token cap that silently disables everything

`recipes/gfx1151_qwen38_27b_w4a16.md` plus `recipes/serve_gfx1151_qwen38_agent.sh`
(long context, prefix caching on) and `recipes/serve_gfx1151_qwen38_bench.sh`
(reproduces the numbers in this log) now document the full configuration.

Writing the agent recipe surfaced a real defect. The tuned attention paths are
gated on `1024 <= max_seq_len <= MAX_CONTEXT` with `MAX_CONTEXT = 4096`, and
CUDA-graph capture runs with `max_seq_len` set to the **model maximum**. So any
`--max-model-len` above 4096 captures the generic fallback for every decode step,
silently and with no warning. Measured at concurrency 1 with
`--max-model-len 32768`, same 2-prompt probe:

| | cap 4096 | cap raised to 32768 |
| --- | --- | --- |
| output throughput | 17.36 tok/s | **34.90 tok/s** |
| mean ITL | 140.51 ms | **100.31 ms** |
| accepted tokens/step | 2.57 | **3.91** |
| `M=8` tuned paths in the log | none | split-KV D=256 + draft D=128 |

The acceptance drop is the part that is easy to miss: the **draft** model's
attention falls back too, so the drafts degrade as well as the verification.

Fix: `VLLM_GFX1151_QWEN_MAX_CONTEXT` (default **4096**, i.e. the delivered
behaviour is unchanged) raises the cap; the agent recipe sets it to
`--max-model-len`. Raising it is provably a no-op at 4096 context because the
scratch reservation is `min(max_model_len, cap) / 256` partitions and
`max_seq_len` cannot exceed `max_model_len`, so every number in this log is
unaffected. Validated afterwards: needle retrieval from line 1 of a long prompt
is correct at **5,843 / 10,177 / 11,043 tokens** - a regime these kernels had
never run in - and decode performance matches the 4096 configuration exactly
(34.69-34.90 tok/s, ITL 100.31-100.95 ms, acceptance 3.91).

**Prefix caching is free here and worth 6.1x on TTFT.** An earlier reading
suggested it collapsed acceptance, but that was the 4096 cap confounding the
comparison. With the cap fixed, on and off are indistinguishable on decode
(ITL 100.73 vs 100.31 ms, acceptance 3.91 both) and a repeated 9,111-token prompt
goes **39.94 s -> 6.54 s**. It is on in the agent recipe and off in the benchmark
recipe, where it exists only for run-to-run comparability.

Cold prefill throughput, for planning agent context sizes: ~3.0 s for 1024
tokens, 28 s for 5,843, 40 s for 9,111, 55 s for 11,043 (roughly 230-340 tok/s).

### The agent configuration re-measured on the full protocol: 44.96 tok/s

Because a short probe of the agent recipe reported 34.9 tok/s against this log's
42.69, the agent configuration was re-run as the **full 32-prompt protocol**,
first workload on a fresh server:

| concurrency 1, 32 prompts x 1024/1024 | benchmark config | agent config |
| --- | --- | --- |
| output throughput | 42.69 tok/s | **44.96 tok/s** (+5.3 %) |
| acceptance length | 6.03 | 6.03 |
| acceptance rate | - | 71.86 % |
| mean ITL | 125.39 ms | **113.28 ms** |
| mean TPOT | - | 18.86 ms |
| mean TTFT | 3856 ms | **3485 ms** |
| duration | - | 728.89 s |

Identical acceptance confirms the model and draft behaviour are unchanged; the
TTFT gain is consistent with `--max-num-batched-tokens 4096` halving the prefill
chunk count, and the ITL gain plausibly with `--max-num-seqs 2` capturing smaller
padded batch shapes, though part of it may be throttle-state variation.

The 34.9 tok/s reading was a **prompt-sample artifact, not a regression**:
acceptance length is a property of the drawn prompts, and the same server gives
3.91 on 2 prompts and 6.03 on 32 under `--dataset-name random`. Since
`tok/s = acceptance / TPOT`, that alone accounts for 34.9 -> 45.0. The control
that establishes equivalence is the benchmark configuration run through the same
2-prompt probe: 34.85 tok/s / 100.50 ms / 3.91, matching the agent
configuration's 34.69-34.90 / 100.31-100.95 / 3.91. Short probes are therefore
only comparable to short probes.

### Test regression from change 8, found and fixed

`tests/v1/attention/test_rocm_attention_backends_selection.py` had **14 failures
from change 8** that were never caught: `test_gfx1151_prefix_dispatch_contract`
and `test_gfx1151_capture_dispatch_and_locked_workspace` mock
`gfx1151_qwen_prefill` and the HIP paged-attention op, but not the new
`splitkv_verify_attention`, so the batch<=2 verification cases reached a real
Triton launch on CPU tensors and died with `ValueError: Pointer argument ...
cannot be accessed from Triton (cpu tensor?)`. Both tests now mock the split-KV
launcher and derive the expected dispatch from `backend.splitkv_supports(...)`
and the same gate conditions rather than restating them as literals, so they
assert which path was taken instead of only that the prefix path was. Result:
**375 passed, 3 failed**, the 3 being pre-existing `ROCM_AITER_FA`/MI3xx
selection tests that fail on an `ImportError` in this environment (my diff to
that file touches zero aiter lines). A new
`test_gfx1151_max_context_cap_and_scratch` covers the cap and the reservation it
sizes.

## The merge costs 18% throughput: bisect of the acceptance regression

Merging origin/main dropped accepted tokens per step from **4.42 to 3.66** on the
identical seeded 4-prompt probe (throughput 38.35 -> 31.3 tok/s, ITL 101.8 ->
105.5 ms). Reproducible: two independent merged-tree runs both gave 3.66, and
returning to the pre-merge branch restored exactly 4.42.

The prompts are provably identical, so this is not a sampling artifact:
`RandomDataset` differs between the two trees only in docstring reflow (221 vs
217 lines, no change to any `randint`/`seed`/token-generation line).

Per-position acceptance shows the draft's *first* token degrading too, so it is
not a verification-window effect:

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pre-merge | 74.0% | 58.7% | 49.0% | 41.4% | 35.5% | 30.4% | 27.6% |
| merged | 67.9% | 50.4% | 35.5% | 29.7% | 24.9% | 21.7% | 20.2% |

Four bisect experiments on the merged tree, each a full server restart plus the
same probe. **The entire dflash/draft subsystem is exonerated:**

| restored from the pre-merge branch | acceptance |
| --- | --- |
| nothing (pure merged tree) | 3.66 |
| `qwen3_dflash.py`, `qwen3_dflash2.py`, `dflash/utils.py` | 3.66 |
| eager context K/V, i.e. undo upstream #57632's capture of the precompute inside the draft CUDA graph (added `VLLM_DFLASH_EAGER_CONTEXT_KV`) | 3.66 |
| `dflash/speculator.py` + `dflash/cudagraph.py` (needed two API shims: `maybe_prepare_dcp_local_seq_lens` -> `prepare_dcp_local_seq_lens` guarded by `cp_size > 1`, and a `num_speculative_tokens` kwarg on `propose()` from upstream #57053) | 3.66 |

Also ruled out by inspection: the int4 lm_head hook in `logits_processor.py` is
intact and its only upstream change is an additive `return_log_probs` kwarg
defaulting to False; the rejection sampler's new `invalid_drafts` parameter is
never passed by any caller, so it stays None and inert; and the W4A16 LDS-tile
kernel is bit-identical (150/150 vs pre-merge references).

The remaining suspects are target-side, consistent with ITL also rising 3.5%:
`qwen3_5.py` (auto-merged), the GDN/linear-attention layers that make up 48 of
the 64 layers, the greedy sampler path, or `rejection_sampler.py`'s replacement of
`input_ids[logits_indices]` with the new Triton `gather_draft_sampled` kernel.
The next cheap discriminator is whether the target's greedy continuation is
byte-identical across the two trees at temperature 0: if it is, only draft quality
changed; if it is not, the merge changed model numerics and that is the thing to
chase.

**The target model is numerically unchanged by the merge.** A temperature-0,
96-token greedy continuation of a fixed prompt is byte-identical on both trees.
Because speculative verification always reproduces the target's greedy sequence,
identical text proves identical target logits - so the regression is draft-side,
yet the dflash subsystem is exonerated above. What remains is the *shared*
infrastructure the draft depends on: attention metadata and block-table
construction, sliding-window (2047) handling for the draft's D=128 layers,
RoPE/positions, or the W4A16 draft's weight-loading path. The next cheap
discriminator is to run the **bf16** draft (`incoai/Qwen3.8-27B-DFlash2`) on both
trees: unchanged acceptance implicates the W4A16 draft weight path, a matching
drop implicates the shared draft attention/metadata machinery.

Confirmed serving state after the investigation, on this branch: acceptance
**4.42**, mean ITL **102.31 ms**, **38.15 tok/s** on the 4-prompt probe, both
tuned attention paths engaged (`split-KV verification D=256 M=8`, draft
`Triton prefix D=128 M=8`), and a pi tool-call round trip succeeds.

**Serving recommendation: stay on `strixhalo`.** The merged branch
`sync-main-segmented-attn` is preserved with these findings; switching to it needs
a rebuild because the deployed `_rocm_C.abi3.so` is branch-specific
(`c5994ba3…` pre-merge, `ce0e5797…` merged).

## Remaining headroom (measured, not yet taken)

Concurrency-1 cold decode step, 103.5 ms measured. The per-kernel budget below
sums to 103.4 ms, so it is complete rather than a sample.

| item | ms/step | floor | status |
| --- | --- | --- | --- |
| W4A16 GEMMs, five target projections (batch 8) | 66.7 | 50.3 | **exhausted**: shipped launch table wins an interleaved median A/B (candidates take 0-3 of 9 rounds); hoisting the bias correction is bit-exact but 2-13% slower; NB=16 spills to scratch; splitting 16 tokens across wave groups is 0.98x production |
| W4A16 draft projections | ~9 | ~7 | short kernels (~110 us) whose per-workgroup activation staging is a large fraction of their traffic |
| int4 lm_head x2 | 6.8 | 5.5 | already ~200 GB/s on 0.66 GB per call |
| elementwise / norm / copy, ~430 launches | ~3.0 | ~1.5 | launch-latency bound, needs fusion |
| attention (after change 8) | 2.6 | ~1.5 | was 9.2 before the GQA-packed split-KV kernel |
| GDN delta rule (chunk + recurrent + norms) | 2.0 | ~1.2 | **corrected from 5.8**: the earlier figure had absorbed unrelated small kernels |
| host gaps | ~5.5 | - | serving adds no measurable gap once throttling is accounted for |
| prefill / TTFT | 3.0-3.9 s per request | - | ~12% of a 1K/1K request; prefill W4A16 runs at ~19.5 of 26.1 TFLOPS |

Two items previously listed here are now closed by measurement rather than by
implementation: batch >= 16 W4A16 (five designs, all within 10-25% of
production, see the wall section above) and width-15 speculation (arithmetic
above shows it is net negative once batch-16 GEMM cost is measured).

## Change 8 - GQA-packed split-KV verification attention

`vllm/v1/attention/ops/gfx1151_verify_splitkv.py`, wired into
`Gfx1151QwenAttentionImpl.forward` behind
`VLLM_GFX1151_QWEN_VERIFY_SPLITKV` (default on, gated to `seq_lens.numel() <= 2`).

The shipped prefix kernel launches one workgroup per *query* head, so the six
query heads of a GQA group each stream the same K/V (6x the unique bytes) and a
batch-1 decode exposes only 24 workgroups of 8 warps on 40 CUs. The new kernel
packs the rows of a whole GQA group into one tile (each K/V byte read once) and
splits the context into 256-token partitions across workgroups, combining their
(m, l, acc) states in a fixed partition order - no atomics, bitwise
deterministic. Ragged query lengths (mixed single-token graph rows) are masked
per sequence, so the extra decode launch the prefix path needs is dropped.

Measured in serving conditions (`bench_gfx1151_attention_splitkv.py`, 16 GiB
scattered page pool, L2 flushed by the surrounding weight streaming), best
configuration `heads_per_prog=6, part=256, BLOCK_N=16, 8 warps, 1 stage`:

| context | prefix tile 16/128/8 | split-KV | speedup | rel L2 vs FP32 |
| --- | --- | --- | --- | --- |
| 1024 | 336-343 us | **117 us** | 2.9x | 2.15e-3 (prefix 2.24e-3) |
| 1400 | 435-442 us | **137 us** | 3.2x | 2.13e-3 (prefix 2.23e-3) |
| 2048 | 564-567 us | **157 us** | 3.6x | 2.14e-3 (prefix 2.27e-3) |
| 3072 | 805-809 us | **322 us** | 2.5x | 2.12e-3 |

Configurations rejected on measurement: BLOCK_N=64 with 8 warps exceeds the
64 KiB LDS budget (73,728 B required); `heads_per_prog=2` (three workgroups per
KV head) is 2-5x slower than full packing; `part=128` is slower than `part=256`
because the partition combine grows while per-program work shrinks.

Wide query blocks keep working by packing fewer heads per program
(`pick_heads_per_prog`: 8 tokens -> 6 heads, 16 -> 3, 32 -> 2, so the tile stays
within the 64-row register budget). That is the prerequisite for width-15
speculation.

Serving A/B, fresh server per cell, 32 x 1024/1024:

| concurrency | before change 8 | after | verdict |
| --- | --- | --- | --- |
| 1 | 40.09 / 40.47 | **42.69** (ITL 128.76 -> 125.39 ms) | **+6.5%** |
| 4 | 68.18 | 67.06 (ITL 253.74 -> 257.16 ms, acceptance 6.04 -> 5.90) | neutral/-1.6%, inside acceptance noise |

Because concurrency 4 did not gain (the prefix path already has 4x24 workgroups
there, and the partition combine is not repaid), the path is gated to
`seq_lens.numel() <= 2`; concurrency 4/8 keep the measured-faster prefix tile.

Tests: `tests/v1/attention/test_gfx1151_verify_splitkv.py` (21 passed) - FP32
reference across batch/context/query-width combinations including 4x1024 and
1/2 x 1400 at width 16, a ragged batch (3 + 8 query rows), bitwise determinism
across repeats, argmax agreement with the shipped prefix tile, and the
`splitkv_supports` contract (rejects non-causal, sliding window, sinks, fp16,
D=128, width > 32, short context).

## Memory movement and zero copy on this machine (measured)

Probe: `/tmp/zc/zc.cu` built as `build`-free standalone (`libzc.so`), one 2 GiB
buffer allocated four ways, GPU streaming read timed with HIP events, best of 5.

`hipGetDeviceProperties`: `integrated = 1`, `hostNativeAtomicSupported = 1`,
`unifiedAddressing = 0`, `directManagedMemAccessFromHost = 0`, one KFD memory
bank of **94.2 GiB**, **256-bit** bus, 2 MiB L2, gfx1151.

| allocation | GPU read bandwidth | pointer type | CPU can read it? |
| --- | --- | --- | --- |
| `hipMalloc` (device/GTT) | **242.0 GB/s** | 2 | **yes** - CPU read back the float a GPU kernel wrote |
| `hipHostMalloc` (pinned, CPU-filled) | **239.8 GB/s** | 1 | yes (it is host memory) |
| `hipHostMallocMapped` | 234.8 GB/s | 1 | yes |
| `hipHostRegister` over a file `mmap` | **240.9 GB/s** | 1 | yes |

| copy | bandwidth |
| --- | --- |
| device -> device | 70.4 GB/s |
| pinned host -> device | 82.7 GB/s |
| device -> pinned host | 70.6 GB/s |

Conclusions, all measured rather than assumed:

1. **There is no separate video memory.** One LPDDR5X pool serves the 32 host
   cores and the 40 CUs; 242 GB/s is 94% of the 256-bit LPDDR5X-8000 spec peak,
   so the streaming ceiling is already reached and *cannot* be improved.
2. **Zero copy is complete and free**: pinned host memory and even a registered
   file `mmap` are read by the GPU at 99% of device bandwidth, and the CPU can
   dereference a device pointer directly. Weights could live in a registered
   safetensors mmap with no copy and no penalty - that would remove ~19 GB of
   load-time copying (startup only; the W4A16 layout still has to be repacked,
   which is what most of the 6.5-7.9 s load time actually is).
3. **A copy costs ~3.4x a read here** (70-83 vs 242 GB/s), so copies are worth
   avoiding - but the decode step already contains no host<->device weight
   traffic. The only per-step copy-ish device work measured is
   `__amd_rocclr_copyBuffer` (141/step, 0.24 ms), `direct_copy` elementwise
   (146/step, 0.51 ms) and `CatArrayBatchedCopy` (0.27 ms): ~1.0 ms/step, 0.8%.
   `--cpu-offload-gb` must stay 0: on this machine "host" memory is the same RAM
   and is already GPU-readable at 240 GB/s, so an offload prefetch would add
   82 GB/s copies to move data that never needed to move.

So the remaining memory-movement waste is **redundant device-side traffic**, not
host/device transfer. Ranked by measured size:

| waste | measured | floor | how to remove |
| --- | --- | --- | --- |
| batch>=16 W4A16 activation re-reads: `ceil(N/BN)*BM*K*2` bytes = 178 MB of activation vs 46 MB of weights for down_proj at batch 32 | 117/128/254 ms per step at batch 16/32/64 | 63 ms | matrix-core W4A16 GEMM (weights straight into WMMA B operands, A staged once, BLOCK_M 32-64, split-K) |
| verification attention reads each K/V byte 6x (one workgroup per query head, no GQA packing, no split-KV): 34 MB of requests per layer for 5.7 MB of unique KV, at 39 GB/s | 9.2 ms/step | ~0.5 ms | GQA-packed split-KV flash-decode kernel |
| 6x GQA redundancy also makes it latency-bound: 24 workgroups of 4 warps on 40 CUs | 874 us/layer at a 1.4K context | ~40 us | same kernel |
| derived group-major weight cache duplicates 11.68 GiB of weights; unused at concurrency 1 now that batch<=8 reads the original layout | 11.68 GiB resident | 0 at C1 | drop it at C1, or make the batch>=16 kernel read the original layout |
| BF16 KV cache | 91 MB of unique KV per step | 46 MB | fp8 KV (needs gfx1151 backend support) |

Ruled out by measurement, not assumption:

* **Page scatter / TLB is not a factor.** `bench_gfx1151_attention_in_situ.py
  --layout contiguous` versus `scattered` over a 16 GiB pool: 891 vs 874 us per
  layer for the legacy tile, 432 vs 429 us for the new tile - inside noise. The
  in-situ cost is occupancy/latency, not address translation.
* **`cu_count` 20 vs 40** for the skinny kernel: 246.6 vs 249.4 us, no effect.
* **Split-K in Triton** for batch>=16: swept 1/2/4/8 with BLOCK_N 32-128; never
  beat production.
