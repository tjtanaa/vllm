// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include <torch/all.h>

#include <cmath>
#include <string>

namespace {
constexpr int kWave = 32;
constexpr int kThreads = 256;
constexpr int kPartition = 256;

__device__ float wave_sum(float x) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) x += __shfl_xor(x, offset, kWave);
  return x;
}

__device__ float block_reduce(float x, float* shared, bool maximum) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    float other = __shfl_xor(x, offset, kWave);
    x = maximum ? fmaxf(x, other) : x + other;
  }
  if (threadIdx.x % kWave == 0) shared[threadIdx.x / kWave] = x;
  __syncthreads();
  float value = maximum ? -INFINITY : 0.0f;
#pragma unroll
  for (int i = 0; i < kThreads / kWave; ++i)
    value = maximum ? fmaxf(value, shared[i]) : value + shared[i];
  __syncthreads();
  return value;
}

// One CTA per (sequence, query head, query position, context partition).
// The cache uses the existing x-packed K and token-contiguous V representation.
template <int D>
__global__ __launch_bounds__(kThreads) void gfx1151_qwen_attention_partition(
    const __hip_bfloat16* q, const __hip_bfloat16* k, const __hip_bfloat16* v,
    const int* table, const int* lengths, const int* starts, float* workspace,
    int tokens, int heads, int kv_heads, int pages, int page_size, int parts,
    int max_query, int max_seq, int64_t qs0, int64_t qs1, int64_t ks0,
    int64_t vs0, int64_t ts0, float scale, int window, bool causal) {
  const int seq = blockIdx.x;
  const int head = blockIdx.y;
  const int part = blockIdx.z % parts;
  const int local_q = blockIdx.z / parts;
  const int first_q = starts[seq];
  const int query_len = starts[seq + 1] - first_q;
  const int token = first_q + local_q;
  if (local_q >= query_len || token < 0 || token >= tokens) return;
  const int length = min(max(lengths[seq], 0), max_seq);
  const int position = length - query_len + local_q;
  const int lower = window > 0 ? max(0, position - window + 1) : 0;
  const int upper = causal ? min(length, position + 1) : length;
  const int kv_head = head / (heads / kv_heads);
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  __shared__ float logits[kPartition];
  __shared__ float reduce[kThreads / kWave];
  float query[D / kWave];
#pragma unroll
  for (int i = 0; i < D / kWave; ++i)
    query[i] = float(q[token * qs0 + head * qs1 + lane + i * kWave]);

  for (int t = wave; t < kPartition; t += kThreads / kWave) {
    const int n = part * kPartition + t;
    float score = -INFINITY;
    if (n >= lower && n < upper) {
      const int physical = table[seq * ts0 + n / page_size];
      if (physical >= 0 && physical < pages) {
        const int slot = n % page_size;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < D / kWave; ++i) {
          const int d = lane + i * kWave;
          const int64_t offset = physical * ks0 + kv_head * D * page_size +
                                 (d / 8) * page_size * 8 + slot * 8 + d % 8;
          dot = fmaf(query[i], float(k[offset]), dot);
        }
        score = wave_sum(dot) * scale;
      }
    }
    if (lane == 0) logits[t] = score;
  }
  __syncthreads();
  float score = logits[threadIdx.x];
  const float maximum = block_reduce(score, reduce, true);
  const float probability = maximum == -INFINITY ? 0.0f : expf(score - maximum);
  logits[threadIdx.x] = probability;
  const float sum = block_reduce(probability, reduce, false);
  const int64_t base =
      ((int64_t(token) * heads + head) * parts + part) * (D + 2);
  if (threadIdx.x == 0) {
    workspace[base + D] = maximum;
    workspace[base + D + 1] = sum;
  }
  for (int d = wave; d < D; d += kThreads / kWave) {
    float acc = 0.0f;
    for (int t = lane; t < kPartition; t += kWave) {
      float p = logits[t];
      if (p != 0.0f) {
        const int n = part * kPartition + t;
        const int physical = table[seq * ts0 + n / page_size];
        const int64_t offset = physical * vs0 + kv_head * D * page_size +
                               d * page_size + n % page_size;
        acc = fmaf(p, float(v[offset]), acc);
      }
    }
    acc = wave_sum(acc);
    if (lane == 0) workspace[base + d] = acc;
  }
}

template <int D, bool Direct = false>
__global__ __launch_bounds__(128) void gfx1151_qwen_attention_wmma_partition(
    const __hip_bfloat16* q, const __hip_bfloat16* k, const __hip_bfloat16* v,
    const int* table, const int* lengths, const int* starts, float* workspace,
    int tokens, int heads, int kv_heads, int pages, int page_size, int parts,
    int active_parts, int max_seq, int64_t qs0, int64_t qs1, int64_t ks0,
    int64_t vs0, int64_t ts0, float scale, int window, bool causal,
    const float* sinks, __hip_bfloat16* output) {
#if defined(__gfx1151__) || !defined(__HIP_DEVICE_COMPILE__)
  using bf16x16 = __bf16 __attribute__((ext_vector_type(16)));
  using floatx8 = float __attribute__((ext_vector_type(8)));
  constexpr int rows = 16;
  constexpr int waves = 4;
  const int seq = blockIdx.x, kv_head = blockIdx.y;
  const int part = blockIdx.z % active_parts;
  const int row_tile = blockIdx.z / active_parts;
  const int group = heads / kv_heads;
  const int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave;
  const int lane_lo = lane % 16, lane_hi = lane / 16;
  const int first_q = starts[seq], query_len = starts[seq + 1] - first_q;
  if (row_tile * rows >= query_len * group) return;
  const int length = min(max(lengths[seq], 0), max_seq);
  // Every row in this tile starts at or after its first query's window.
  // Start partitions there instead of launching work over masked prefixes.
  const int first_position = length - query_len + row_tile * rows / group;
  const int context_begin =
      window > 0 ? max(0, first_position - window + 1) : 0;
  const int partition_begin = context_begin + part * kPartition;
  __shared__ __hip_bfloat16 queries[rows][D];
  __shared__ float scores[rows][kPartition];
  __shared__ __hip_bfloat16 probs[rows][kPartition];
  // A row combines a query position and one of the heads sharing this KV head.
  for (int idx = threadIdx.x; idx < rows * D; idx += waves * kWave) {
    const int row = row_tile * rows + idx / D;
    const int token = first_q + row / group;
    const int head = kv_head * group + row % group;
    queries[idx / D][idx % D] =
        row < query_len * group && token >= 0 && token < tokens
            ? q[token * qs0 + head * qs1 + idx % D]
            : __hip_bfloat16(0.0f);
  }
  __syncthreads();
  for (int tile = wave; tile < kPartition / 16; tile += waves) {
    floatx8 accum = {};
    const int n = partition_begin + tile * 16 + lane_lo;
    const int physical = n < length ? table[seq * ts0 + n / page_size] : -1;
    const bool valid = n < length && physical >= 0 && physical < pages;
    for (int kd = 0; kd < D; kd += 16) {
      bf16x16 a, b;
  #pragma unroll
      for (int i = 0; i < 16; ++i) {
        a[i] = __builtin_bit_cast(__bf16, queries[lane_lo][kd + i]);
        const int d = kd + i;
        const int64_t offset = physical * ks0 + kv_head * D * page_size +
                               (d / 8) * page_size * 8 + (n % page_size) * 8 +
                               d % 8;
        b[i] = valid ? __builtin_bit_cast(__bf16, k[offset]) : __bf16(0.0f);
      }
      accum = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, accum);
    }
  #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = 2 * i + lane_hi;
      const int row = row_tile * rows + r;
      const int position = length - query_len + row / group;
      const bool masked = !valid || row >= query_len * group ||
                          (causal && n > position) ||
                          (window > 0 && n < position - window + 1);
      scores[r][tile * 16 + lane_lo] = masked ? -INFINITY : accum[i] * scale;
    }
  }
  __syncthreads();
  for (int r = wave; r < rows; r += waves) {
    float maximum = -INFINITY;
  #pragma unroll
    for (int t = lane; t < kPartition; t += kWave)
      maximum = fmaxf(maximum, scores[r][t]);
  #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      maximum = fmaxf(maximum, __shfl_xor(maximum, offset, kWave));
    float sum = 0.0f;
  #pragma unroll
    for (int t = lane; t < kPartition; t += kWave) {
      const float p =
          maximum == -INFINITY ? 0.0f : expf(scores[r][t] - maximum);
      probs[r][t] = __hip_bfloat16(p);
      sum += p;
    }
    sum = wave_sum(sum);
    if (lane == 0) {
      // QK is no longer needed for this row. Reuse it so D=256 stays within
      // 32 KiB LDS and permits two workgroups per CU.
      scores[r][0] = maximum;
      scores[r][1] = sum;
    }
  }
  __syncthreads();
  for (int tile = wave; tile < D / 16; tile += waves) {
    floatx8 accum = {};
    const int d = tile * 16 + lane_lo;
    for (int kt = 0; kt < kPartition; kt += 16) {
      bf16x16 a, b;
  #pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int n = partition_begin + kt + i;
        const int physical = n < length ? table[seq * ts0 + n / page_size] : -1;
        const int64_t offset = physical * vs0 + kv_head * D * page_size +
                               d * page_size + n % page_size;
        a[i] = __builtin_bit_cast(__bf16, probs[lane_lo][kt + i]);
        b[i] = n < length && physical >= 0 && physical < pages
                   ? __builtin_bit_cast(__bf16, v[offset])
                   : __bf16(0.0f);
      }
      accum = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, accum);
    }
  #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = 2 * i + lane_hi;
      const int row = row_tile * rows + r;
      const int token = first_q + row / group;
      if (row < query_len * group && token >= 0 && token < tokens) {
        const int head = kv_head * group + row % group;
        if constexpr (Direct) {
          // A short context fits in one tile; normalize and store without a
          // workspace round trip or a second launch. Sinks add mass, not V.
          const float maximum =
              sinks ? fmaxf(scores[r][0], sinks[head]) : scores[r][0];
          const float weight =
              scores[r][1] > 0.0f ? expf(scores[r][0] - maximum) : 0.0f;
          const float sink_mass = sinks && maximum != -INFINITY
                                      ? expf(sinks[head] - maximum)
                                      : 0.0f;
          const float denominator = scores[r][1] * weight + sink_mass;
          output[(int64_t(token) * heads + head) * D + d] = __hip_bfloat16(
              denominator > 0.0f ? accum[i] * weight / denominator : 0.0f);
        } else {
          const int64_t base =
              ((int64_t(token) * heads + head) * parts + part) * (D + 2);
          workspace[base + d] = accum[i];
          if (d == 0) {
            workspace[base + D] = scores[r][0];
            workspace[base + D + 1] = scores[r][1];
          }
        }
      }
    }
  }
#endif
}

// Single-token QK stays partition-parallel, but stores logits rather than
// independently normalized partial outputs. The following PV kernel can then
// round BF16 probabilities at the reference's running 32-token maximum.
__global__ __launch_bounds__(128) void gfx1151_qwen_decode_logits(
    const __hip_bfloat16* q, const __hip_bfloat16* k, const int* table,
    const int* lengths, const int* starts, float* workspace, int tokens,
    int pages, int page_size, int parts, int max_seq, int64_t qs0,
    int64_t qs1, int64_t ks0, int64_t ts0, float scale) {
#if defined(__gfx1151__) || !defined(__HIP_DEVICE_COMPILE__)
  using bf16x16 = __bf16 __attribute__((ext_vector_type(16)));
  using floatx8 = float __attribute__((ext_vector_type(8)));
  constexpr int D = 256, heads = 24, group = 6;
  const int seq = blockIdx.x, kv_head = blockIdx.y, part = blockIdx.z;
  const int token = starts[seq];
  if (starts[seq + 1] - token != 1 || token < 0 || token >= tokens) return;
  const int length = min(max(lengths[seq], 0), max_seq);
  if (part * kPartition >= length) return;
  const int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave;
  const int lane_lo = lane % 16, lane_hi = lane / 16;
  __shared__ __hip_bfloat16 queries[16][D];
  for (int idx = threadIdx.x; idx < 16 * D; idx += 128) {
    const int row = idx / D;
    queries[row][idx % D] = row < group
        ? q[token * qs0 + (kv_head * group + row) * qs1 + idx % D]
        : __hip_bfloat16(0.0f);
  }
  __syncthreads();
  for (int tile = wave; tile < kPartition / 16; tile += 4) {
    const int n = part * kPartition + tile * 16 + lane_lo;
    const int physical = n < length ? table[seq * ts0 + n / page_size] : -1;
    const bool valid = n < length && physical >= 0 && physical < pages;
    floatx8 accum = {};
    for (int kd = 0; kd < D; kd += 16) {
      bf16x16 a, b;
  #pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int d = kd + i;
        a[i] = __builtin_bit_cast(__bf16, queries[lane_lo][d]);
        const int64_t offset = physical * ks0 + kv_head * D * page_size +
            (d / 8) * page_size * 8 + (n % page_size) * 8 + d % 8;
        b[i] = valid ? __builtin_bit_cast(__bf16, k[offset]) : __bf16(0.0f);
      }
      accum = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, accum);
    }
  #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = 2 * i + lane_hi;
      if (row < group) {
        const int head = kv_head * group + row;
        const int64_t base =
            ((int64_t(token) * heads + head) * parts + part) * (D + 2);
        workspace[base + tile * 16 + lane_lo] =
            valid ? accum[i] * scale : -INFINITY;
      }
    }
  }
#endif
}

// One wave owns a 32-column V slice. Splitting output columns exposes 32 CTAs
// per request without changing the online-softmax order or splitting its sum.
__global__ __launch_bounds__(32) void gfx1151_qwen_decode_online_pv(
    const float* workspace, const __hip_bfloat16* v, const int* table,
    const int* lengths, const int* starts, __hip_bfloat16* output, int tokens,
    int pages, int page_size, int parts, int max_seq, int64_t vs0, int64_t ts0) {
#if defined(__gfx1151__) || !defined(__HIP_DEVICE_COMPILE__)
  using bf16x16 = __bf16 __attribute__((ext_vector_type(16)));
  using floatx8 = float __attribute__((ext_vector_type(8)));
  constexpr int D = 256, heads = 24, group = 6;
  constexpr float log2e = 1.4426950408889634f;
  const int seq = blockIdx.x, kv_head = blockIdx.y;
  const int token = starts[seq], lane = threadIdx.x;
  if (starts[seq + 1] - token != 1 || token < 0 || token >= tokens) return;
  const int length = min(max(lengths[seq], 0), max_seq);
  const int lane_lo = lane % 16, lane_hi = lane / 16;
  const int first_d = blockIdx.z * 32;
  __shared__ __hip_bfloat16 probabilities[16][32];
  __shared__ float alphas[16];
  float maxima[group], denominators[group];
  #pragma unroll
  for (int row = 0; row < group; ++row) {
    maxima[row] = -INFINITY;
    denominators[row] = 0.0f;
  }
  for (int row = 0; row < 16; ++row) probabilities[row][lane] = __hip_bfloat16(0.0f);
  if (lane < 16) alphas[lane] = 0.0f;
  floatx8 accum[2] = {};
  for (int first_n = 0; first_n < length; first_n += 32) {
    const int n = first_n + lane;
    const int part = n / kPartition;
  #pragma unroll
    for (int row = 0; row < group; ++row) {
      const int head = kv_head * group + row;
      const int64_t base =
          ((int64_t(token) * heads + head) * parts + part) * (D + 2);
      const float score = n < length ? workspace[base + n % kPartition] : -INFINITY;
      float maximum = score;
  #pragma unroll
      for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor(maximum, offset, kWave));
      maximum = fmaxf(maxima[row], maximum);
      const float p = maximum == -INFINITY ? 0.0f
          : exp2f((score - maximum) * log2e);
      const float alpha = maxima[row] == -INFINITY ? 0.0f
          : exp2f((maxima[row] - maximum) * log2e);
      probabilities[row][lane] = __hip_bfloat16(p);
      // The transposed WMMA reference first reduces each parity's 16 values,
      // then combines even/odd lanes. Preserve that FP32 reduction tree.
      float probability_sum = p;
  #pragma unroll
      for (int offset = 2; offset <= 16; offset <<= 1)
        probability_sum += __shfl_xor(probability_sum, offset, kWave);
      probability_sum += __shfl_xor(probability_sum, 1, kWave);
      denominators[row] = fmaf(denominators[row], alpha, probability_sum);
      maxima[row] = maximum;
      if (lane == 0) alphas[row] = alpha;
    }
    __syncthreads();
  #pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
  #pragma unroll
      for (int i = 0; i < 8; ++i)
        accum[dt][i] *= alphas[2 * i + lane_hi];
      const int d = first_d + dt * 16 + lane_lo;
  #pragma unroll
      for (int kt = 0; kt < 32; kt += 16) {
        bf16x16 a, b;
  #pragma unroll
        for (int i = 0; i < 16; ++i) {
          const int pos = first_n + kt + i;
          const int physical = pos < length ? table[seq * ts0 + pos / page_size] : -1;
          const bool valid = pos < length && physical >= 0 && physical < pages;
          const int64_t offset = physical * vs0 + kv_head * D * page_size +
              d * page_size + pos % page_size;
          a[i] = __builtin_bit_cast(__bf16, probabilities[lane_lo][kt + i]);
          b[i] = valid ? __builtin_bit_cast(__bf16, v[offset]) : __bf16(0.0f);
        }
        accum[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, accum[dt]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int dt = 0; dt < 2; ++dt) {
    const int d = first_d + dt * 16 + lane_lo;
  #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = 2 * i + lane_hi;
      if (row < group) {
        const int head = kv_head * group + row;
        output[(int64_t(token) * heads + head) * D + d] =
            __hip_bfloat16(accum[dt][i] / (denominators[row] + 1e-10f));
      }
    }
  }
#endif
}

// Two WMMA row tiles share every K/V operand. Store BF16 probabilities in
// the first half of each retired FP32 score row to keep LDS at 32 KiB.
__global__ __launch_bounds__(128) void gfx1151_qwen_draft_wide_partition(
    const __hip_bfloat16* q, const __hip_bfloat16* k, const __hip_bfloat16* v,
    const int* table, const int* lengths, const int* starts, float* workspace,
    int tokens, int heads, int kv_heads, int pages, int page_size, int parts,
    int active_parts, int max_seq, int64_t qs0, int64_t qs1, int64_t ks0,
    int64_t vs0, int64_t ts0, float scale, int window, bool causal) {
#if defined(__gfx1151__) || !defined(__HIP_DEVICE_COMPILE__)
  using bf16x16 = __bf16 __attribute__((ext_vector_type(16)));
  using floatx8 = float __attribute__((ext_vector_type(8)));
  constexpr int D = 128, rows = 32, waves = 4;
  const int seq = blockIdx.x, kv_head = blockIdx.y;
  const int part = blockIdx.z % active_parts;
  const int row_tile = blockIdx.z / active_parts;
  const int group = heads / kv_heads;
  const int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave;
  const int lane_lo = lane % 16, lane_hi = lane / 16;
  const int first_q = starts[seq], query_len = starts[seq + 1] - first_q;
  if (row_tile * rows >= query_len * group) return;
  const int length = min(max(lengths[seq], 0), max_seq);
  const int first_position = length - query_len + row_tile * rows / group;
  const int context_begin =
      window > 0 ? max(0, first_position - window + 1) : 0;
  const int partition_begin = context_begin + part * kPartition;
  __shared__ uint32_t storage[rows][kPartition];

  for (int tile = wave; tile < kPartition / 16; tile += waves) {
    floatx8 accum[2] = {};
    const int n = partition_begin + tile * 16 + lane_lo;
    const int physical = n < length ? table[seq * ts0 + n / page_size] : -1;
    const bool valid = n < length && physical >= 0 && physical < pages;
    for (int kd = 0; kd < D; kd += 16) {
      bf16x16 b;
  #pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int d = kd + i;
        const int64_t offset = physical * ks0 + kv_head * D * page_size +
                               (d / 8) * page_size * 8 + n % page_size * 8 +
                               d % 8;
        b[i] = valid ? __builtin_bit_cast(__bf16, k[offset]) : __bf16(0.0f);
      }
  #pragma unroll
      for (int qr = 0; qr < 2; ++qr) {
        const int row = row_tile * rows + qr * 16 + lane_lo;
        const int token = first_q + row / group;
        const int head = kv_head * group + row % group;
        bf16x16 a;
  #pragma unroll
        for (int i = 0; i < 16; ++i)
          a[i] = row < query_len * group && token >= 0 && token < tokens
                     ? __builtin_bit_cast(__bf16,
                                          q[token * qs0 + head * qs1 + kd + i])
                     : __bf16(0.0f);
        accum[qr] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, accum[qr]);
      }
    }
  #pragma unroll
    for (int qr = 0; qr < 2; ++qr) {
  #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int r = qr * 16 + 2 * i + lane_hi;
        const int row = row_tile * rows + r;
        const int position = length - query_len + row / group;
        const bool masked = !valid || row >= query_len * group ||
                            (causal && n > position) ||
                            (window > 0 && n < position - window + 1);
        const float score = masked ? -INFINITY : accum[qr][i] * scale;
        storage[r][tile * 16 + lane_lo] = __builtin_bit_cast(uint32_t, score);
      }
    }
  }
  __syncthreads();
  for (int r = wave; r < rows; r += waves) {
    // Every lane loads its whole score slice before any lane overwrites the
    // row. The max reduction consumes all slices before probability stores.
    float values[kPartition / kWave], maximum = -INFINITY;
  #pragma unroll
    for (int i = 0; i < kPartition / kWave; ++i) {
      values[i] = __builtin_bit_cast(float, storage[r][lane + i * kWave]);
      maximum = fmaxf(maximum, values[i]);
    }
  #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      maximum = fmaxf(maximum, __shfl_xor(maximum, offset, kWave));
    float sum = 0.0f;
  #pragma unroll
    for (int i = 0; i < kPartition / kWave; ++i) {
      const float p = maximum == -INFINITY ? 0.0f : expf(values[i] - maximum);
      const uint32_t bits = __builtin_bit_cast(uint16_t, __hip_bfloat16(p));
      const uint32_t packed = bits | (__shfl_down(bits, 1, kWave) << 16);
      // PV reads one probability column across 16 distinct rows. XOR the
      // packed column with the row to avoid a 16-way LDS bank conflict.
      if ((lane & 1) == 0)
        storage[r][((lane + i * kWave) / 2) ^ (r & 15)] = packed;
      sum += p;
    }
    sum = wave_sum(sum);
    if (lane == 0) {
      storage[r][254] = __builtin_bit_cast(uint32_t, maximum);
      storage[r][255] = __builtin_bit_cast(uint32_t, sum);
    }
  }
  __syncthreads();
  for (int tile = wave; tile < D / 16; tile += waves) {
    floatx8 accum[2] = {};
    const int d = tile * 16 + lane_lo;
    for (int kt = 0; kt < kPartition; kt += 16) {
      bf16x16 b;
  #pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int n = partition_begin + kt + i;
        const int physical = n < length ? table[seq * ts0 + n / page_size] : -1;
        const int64_t offset = physical * vs0 + kv_head * D * page_size +
                               d * page_size + n % page_size;
        b[i] = n < length && physical >= 0 && physical < pages
                   ? __builtin_bit_cast(__bf16, v[offset])
                   : __bf16(0.0f);
      }
  #pragma unroll
      for (int qr = 0; qr < 2; ++qr) {
        bf16x16 a;
  #pragma unroll
        for (int i = 0; i < 16; ++i) {
          const int r = qr * 16 + lane_lo;
          const uint32_t packed = storage[r][((kt + i) / 2) ^ (r & 15)];
          const uint16_t bits = uint16_t(packed >> ((i & 1) * 16));
          a[i] = __builtin_bit_cast(__bf16, bits);
        }
        accum[qr] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, accum[qr]);
      }
    }
  #pragma unroll
    for (int qr = 0; qr < 2; ++qr) {
  #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int r = qr * 16 + 2 * i + lane_hi;
        const int row = row_tile * rows + r;
        const int token = first_q + row / group;
        if (row < query_len * group && token >= 0 && token < tokens) {
          const int head = kv_head * group + row % group;
          const int64_t base =
              ((int64_t(token) * heads + head) * parts + part) * (D + 2);
          workspace[base + d] = accum[qr][i];
          if (d == 0) {
            workspace[base + D] = __builtin_bit_cast(float, storage[r][254]);
            workspace[base + D + 1] =
                __builtin_bit_cast(float, storage[r][255]);
          }
        }
      }
    }
  }
#endif
}

template <int D>
__global__ void gfx1151_qwen_attention_reduce(
    const float* workspace, const int* starts, const float* sinks,
    __hip_bfloat16* output, int tokens, int heads, int parts, int active_parts,
    int64_t os0, int64_t os1) {
  const int seq = blockIdx.x;
  const int head = blockIdx.y;
  const int local_q = blockIdx.z;
  const int token = starts[seq] + local_q;
  if (local_q >= starts[seq + 1] - starts[seq] || token < 0 || token >= tokens)
    return;
  const int d = threadIdx.x;
  if (d >= D) return;
  const int64_t base = (int64_t(token) * heads + head) * parts * (D + 2);
  float maximum = sinks ? sinks[head] : -INFINITY;
  for (int p = 0; p < active_parts; ++p)
    maximum = fmaxf(maximum, workspace[base + p * (D + 2) + D]);
  float denom =
      sinks && maximum != -INFINITY ? expf(sinks[head] - maximum) : 0.0f;
  float value = 0.0f;
  for (int p = 0; p < active_parts; ++p) {
    const int64_t entry = base + p * (D + 2);
    const float sum = workspace[entry + D + 1];
    if (sum > 0.0f) {
      const float weight = expf(workspace[entry + D] - maximum);
      denom += sum * weight;
      value += workspace[entry + d] * weight;
    }
  }
  output[token * os0 + head * os1 + d] =
      __hip_bfloat16(denom > 0.0f ? value / denom : 0.0f);
}
}  // namespace

void gfx1151_qwen_paged_attention(
    const torch::Tensor& query, const torch::Tensor& key_cache,
    const torch::Tensor& value_cache, const torch::Tensor& block_table,
    const torch::Tensor& seq_lens, const torch::Tensor& query_start_loc,
    const std::optional<torch::Tensor>& sinks, torch::Tensor& output,
    torch::Tensor& workspace, int64_t max_seq_len, int64_t max_query_len,
    double scale, int64_t sliding_window, bool causal) {
  TORCH_CHECK(query.is_cuda(), "gfx1151 attention requires GPU tensors");
  const at::cuda::OptionalCUDAGuard guard(device_of(query));
  const auto* prop = at::cuda::getDeviceProperties(query.get_device());
  TORCH_CHECK(std::string(prop->gcnArchName).find("gfx1151") == 0,
              "gfx1151 attention requires gfx1151");
  for (const auto* t : std::initializer_list<const torch::Tensor*>{
           &key_cache, &value_cache, &block_table, &seq_lens, &query_start_loc,
           &output, &workspace})
    TORCH_CHECK(t->device() == query.device(),
                "all tensors must share a device");
  TORCH_CHECK(
      query.dim() == 3 && key_cache.dim() == 5 && value_cache.dim() == 4,
      "invalid Q/K/V ranks");
  TORCH_CHECK(query.scalar_type() == torch::kBFloat16 &&
                  key_cache.scalar_type() == torch::kBFloat16 &&
                  value_cache.scalar_type() == torch::kBFloat16 &&
                  output.scalar_type() == torch::kBFloat16,
              "gfx1151 attention requires BF16 Q/K/V/output");
  const int64_t d = query.size(2), h = query.size(1), kh = key_cache.size(1);
  TORCH_CHECK(
      (d == 256 && h == 24 && kh == 4) || (d == 128 && h == 32 && kh == 8),
      "unsupported Qwen attention head geometry");
  TORCH_CHECK(max_seq_len > 0 && max_seq_len <= 32768 && max_query_len > 0 &&
                  max_query_len <= 32 && std::isfinite(scale) &&
                  sliding_window >= 0,
              "unsupported sequence/query length, scale, or window");
  const int64_t pages = key_cache.size(0), page = key_cache.size(3);
  TORCH_CHECK(pages > 0 && page > 0 && key_cache.size(2) == d / 8 &&
                  key_cache.size(4) == 8 && value_cache.size(0) == pages &&
                  value_cache.size(1) == kh && value_cache.size(2) == d &&
                  value_cache.size(3) == page,
              "unsupported cache shapes");
  TORCH_CHECK(
      key_cache.stride(4) == 1 && key_cache.stride(3) == 8 &&
          key_cache.stride(2) == page * 8 && key_cache.stride(1) == d * page &&
          key_cache.stride(0) >= kh * d * page && value_cache.stride(3) == 1 &&
          value_cache.stride(2) == page && value_cache.stride(1) == d * page &&
          value_cache.stride(0) >= kh * d * page,
      "unsupported cache strides");
  TORCH_CHECK(query.stride(2) == 1 && query.stride(1) >= d &&
                  query.stride(0) >= h * query.stride(1) &&
                  output.sizes() == query.sizes() && output.is_contiguous(),
              "unsupported query/output layout");
  TORCH_CHECK(
      seq_lens.dim() == 1 && seq_lens.size(0) > 0 && seq_lens.size(0) <= 8 &&
          seq_lens.is_contiguous() && seq_lens.scalar_type() == torch::kInt32 &&
          query_start_loc.dim() == 1 &&
          query_start_loc.size(0) == seq_lens.size(0) + 1 &&
          query_start_loc.is_contiguous() &&
          query_start_loc.scalar_type() == torch::kInt32 &&
          block_table.dim() == 2 && block_table.size(0) == seq_lens.size(0) &&
          block_table.size(1) >= (max_seq_len + page - 1) / page &&
          block_table.stride(1) == 1 &&
          block_table.stride(0) >= block_table.size(1) &&
          block_table.scalar_type() == torch::kInt32,
      "unsupported attention metadata layout");
  const int parts = (max_seq_len + kPartition - 1) / kPartition;
  const int window = std::min(sliding_window, max_seq_len);
  const int context_span =
      window > 0 ? std::min(max_seq_len, window + max_query_len - 1)
                 : max_seq_len;
  const int active_parts = (context_span + kPartition - 1) / kPartition;
  TORCH_CHECK(workspace.scalar_type() == torch::kFloat32 &&
                  workspace.is_contiguous() && workspace.dim() == 4 &&
                  workspace.size(0) == query.size(0) &&
                  workspace.size(1) == h && workspace.size(2) == parts &&
                  workspace.size(3) == d + 2,
              "workspace must be FP32 [tokens, heads, partitions, head_dim+2]");
  if (sinks)
    TORCH_CHECK(sinks->device() == query.device() && sinks->dim() == 1 &&
                    sinks->size(0) == h && sinks->is_contiguous() &&
                    sinks->scalar_type() == torch::kFloat32,
                "sinks must be FP32 [heads] on the query device");
  if (query.size(0) == 0) return;
  const auto stream = at::cuda::getCurrentCUDAStream();
  // Non-power-of-two hybrid pages use 32-token tiles in the original decode.
  // Preserve that order for target M1; verification/draft dispatch is unchanged.
  if (d == 256 && max_query_len == 1 && active_parts > 1 && window == 0 && !sinks &&
      (page & (page - 1)) != 0) {
    const dim3 logits_grid(seq_lens.size(0), kh, parts);
    gfx1151_qwen_decode_logits<<<logits_grid, 128, 0, stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(query.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(key_cache.data_ptr()),
        block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
        query_start_loc.data_ptr<int>(), workspace.data_ptr<float>(),
        query.size(0), pages, page, parts, max_seq_len, query.stride(0),
        query.stride(1), key_cache.stride(0), block_table.stride(0), scale);
    const dim3 pv_grid(seq_lens.size(0), kh, 8);
    gfx1151_qwen_decode_online_pv<<<pv_grid, 32, 0, stream>>>(
        workspace.data_ptr<float>(),
        reinterpret_cast<const __hip_bfloat16*>(value_cache.data_ptr()),
        block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
        query_start_loc.data_ptr<int>(),
        reinterpret_cast<__hip_bfloat16*>(output.data_ptr()), query.size(0),
        pages, page, parts, max_seq_len, value_cache.stride(0),
        block_table.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const dim3 grid(seq_lens.size(0), kh,
                  active_parts * ((max_query_len * (h / kh) + 15) / 16));
  const dim3 reduce_grid(seq_lens.size(0), h, max_query_len);
  const auto launch = [&]<int D, bool Direct>() {
    gfx1151_qwen_attention_wmma_partition<D, Direct><<<grid, 128, 0, stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(query.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(key_cache.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(value_cache.data_ptr()),
        block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
        query_start_loc.data_ptr<int>(), workspace.data_ptr<float>(),
        query.size(0), h, kh, pages, page, parts, active_parts, max_seq_len,
        query.stride(0), query.stride(1), key_cache.stride(0),
        value_cache.stride(0), block_table.stride(0), scale, window, causal,
        sinks ? sinks->data_ptr<float>() : nullptr,
        reinterpret_cast<__hip_bfloat16*>(output.data_ptr()));
    if constexpr (!Direct)
      gfx1151_qwen_attention_reduce<D><<<reduce_grid, kThreads, 0, stream>>>(
          workspace.data_ptr<float>(), query_start_loc.data_ptr<int>(),
          sinks ? sinks->data_ptr<float>() : nullptr,
          reinterpret_cast<__hip_bfloat16*>(output.data_ptr()), query.size(0),
          h, parts, active_parts, output.stride(0), output.stride(1));
  };
  if (d == 256) {
    if (active_parts == 1)
      launch.template operator()<256, true>();
    else
      launch.template operator()<256, false>();
  } else {
    if (active_parts == 1)
      launch.template operator()<128, true>();
    else if (max_query_len >= 16) {
      const dim3 wide_grid(
          seq_lens.size(0), kh,
          active_parts * ((max_query_len * (h / kh) + 31) / 32));
      gfx1151_qwen_draft_wide_partition<<<wide_grid, 128, 0, stream>>>(
          reinterpret_cast<const __hip_bfloat16*>(query.data_ptr()),
          reinterpret_cast<const __hip_bfloat16*>(key_cache.data_ptr()),
          reinterpret_cast<const __hip_bfloat16*>(value_cache.data_ptr()),
          block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
          query_start_loc.data_ptr<int>(), workspace.data_ptr<float>(),
          query.size(0), h, kh, pages, page, parts, active_parts, max_seq_len,
          query.stride(0), query.stride(1), key_cache.stride(0),
          value_cache.stride(0), block_table.stride(0), scale, window, causal);
      gfx1151_qwen_attention_reduce<128><<<reduce_grid, kThreads, 0, stream>>>(
          workspace.data_ptr<float>(), query_start_loc.data_ptr<int>(),
          sinks ? sinks->data_ptr<float>() : nullptr,
          reinterpret_cast<__hip_bfloat16*>(output.data_ptr()), query.size(0),
          h, parts, active_parts, output.stride(0), output.stride(1));
    } else
      launch.template operator()<128, false>();
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
