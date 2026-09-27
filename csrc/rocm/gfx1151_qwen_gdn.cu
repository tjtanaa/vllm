// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include <torch/all.h>

#include <cmath>
#include <string>
#include <type_traits>

namespace {
constexpr int kDim = 128;
constexpr int kHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kThreads = 1024;
constexpr int kWaves = kThreads / 32;
constexpr int kRows = kDim / kWaves;
constexpr int kMixed = (2 * kHeads + kValueHeads) * kDim;

__device__ float wave_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_xor(value, offset, 32);
  return value;
}

__device__ float sigmoid(float value) { return 1.0f / (1.0f + expf(-value)); }

// Packed M1 uses contiguous groups of four, two local pairs, then a row-wise
// reduction. Only lane 31 contains the full sum before broadcasting it.
__device__ float packed_dot_sum(const float (&lhs)[4], const float (&rhs)[4]) {
  const float lo = fmaf(lhs[0], rhs[0], lhs[1] * rhs[1]);
  const float hi = fmaf(lhs[2], rhs[2], lhs[3] * rhs[3]);
  float sum = lo + hi;
#pragma unroll
  for (int offset = 8; offset; offset >>= 1)
    sum += __shfl_up(sum, offset, 32);
  sum += __shfl_xor(sum, 16, 32);
  return __shfl(sum, 31, 32);
}

__device__ float packed_norm_sqrt(float value) {
  float result;
  asm("v_sqrt_f32 %0, %1" : "=v"(result) : "v"(value));
  return result;
}

__device__ float packed_softplus(float x) {
  if (x > 20.0f) return x;
  const float value = 1.0f + exp2f(x * 1.4426950408889634f);
  float log2_value;
  asm("v_log_f32 %0, %1" : "=v"(log2_value) : "v"(value));
  // value is >= 1. Match the packed reference's contracted final log sum.
  constexpr float ln2_hi = 0x1.62e42ep-1f, ln2_lo = 0x1.efa39ep-25f;
  const float high = log2_value * ln2_hi;
  float low = fmaf(log2_value, ln2_hi, -high);
  low = fmaf(log2_value, ln2_lo, low);
  return fmaf(log2_value, ln2_hi, low);
}

template <typename State>
__device__ void store_state4(State* destination, const float (&values)[4],
                             bool aligned) {
  if (aligned) {
    using Bits = std::conditional_t<sizeof(State) == 4, uint32_t, uint16_t>;
    using Packed = Bits __attribute__((ext_vector_type(4)));
    Packed packed;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      packed[i] = __builtin_bit_cast(Bits, State(values[i]));
    *reinterpret_cast<Packed*>(destination) = packed;
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) destination[i] = State(values[i]);
  }
}

__device__ float conv_product(float lhs, float rhs) {
#if defined(__gfx1151__)
  // Match the executed Triton BF16-multiply lowering. FP32 multiplication
  // followed by a BF16 cast does not reproduce this instruction's result.
  using short2 = short __attribute__((ext_vector_type(2)));
  const short2 a = {__builtin_bit_cast(short, __hip_bfloat16(lhs)), 0};
  const short2 b = {__builtin_bit_cast(short, __hip_bfloat16(rhs)), 0};
  const short product = __builtin_amdgcn_fdot2_bf16_bf16(a, b, 0);
  return float(__builtin_bit_cast(__hip_bfloat16, product));
#else
  return float(__hip_bfloat16(lhs * rhs));
#endif
}

struct Strides {
  int64_t mixed, a, b, gate, state;
};

struct Convolution {
  __hip_bfloat16* state;
  const __hip_bfloat16* weight;
  const __hip_bfloat16* bias;
  int64_t slot_stride, channel_stride, token_stride, weight_stride;
  int slots;
};

// A wave32 workgroup owns one value head, or all three heads sharing Q/K when
// convolution is fused. Keep FP32 recurrence registers across tokens, even
// when checkpoint states are BF16.
template <typename State, typename Bias, typename Weight, bool FuseConv = false,
          bool FuseNorm = true>
__global__ __launch_bounds__(kThreads) void gfx1151_qwen_gdn_post_conv_kernel(
    const __hip_bfloat16* mixed, const __hip_bfloat16* a,
    const __hip_bfloat16* b, const float* a_log, const Bias* dt_bias,
    const int* indices, const int* starts, const int* accepted, State* state,
    const __hip_bfloat16* gate, const Weight* norm_weight,
    __hip_bfloat16* output, int tokens, int slots, int input_width, float scale,
    float epsilon, bool sigmoid_gate, Strides strides, Convolution conv) {
  constexpr int heads_per_block = FuseConv ? 3 : 1;
  constexpr int local_channels = 5 * kDim;
  // The raw-core entry validates M=1 before launch.
  const int width = FuseNorm ? input_width : 1;
  const int request = blockIdx.x, first_head = blockIdx.y * heads_per_block;
  const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  const bool aligned_state =
      reinterpret_cast<uintptr_t>(state) % (4 * sizeof(State)) == 0 &&
      strides.state % 4 == 0;
  const int begin = starts[request], end = starts[request + 1];
  if (begin < 0 || end > tokens || end <= begin) return;
  const int count = end - begin;
  const int num_accepted = accepted[request];
  const int source = num_accepted > 0 && num_accepted <= width
                         ? indices[request * width + num_accepted - 1]
                         : 0;
  const int conv_slot = FuseConv ? indices[request * width] : 0;
  if (source <= 0 || source >= slots || count > width ||
      (FuseConv && (conv_slot <= 0 || conv_slot >= conv.slots))) {
    for (int h = 0; h < heads_per_block; ++h)
      for (int i = threadIdx.x; i < count * kDim; i += kThreads)
        output[(int64_t(begin + i / kDim) * kValueHeads + first_head + h) *
                   kDim +
               i % kDim] = __hip_bfloat16(0.0f);
    return;
  }

  // All three value heads sharing Q/K belong to this workgroup. No other
  // workgroup can overwrite their convolution history while it is being read.
  __shared__ __hip_bfloat16 convolved[FuseConv ? 32 * local_channels : 1];
  if constexpr (FuseConv) {
    for (int c = threadIdx.x; c < local_channels; c += kThreads) {
      const int local_head = c / kDim, d = c % kDim;
      const int channel =
          local_head < 2
              ? (local_head * kHeads + blockIdx.y) * kDim + d
              : (2 * kHeads + first_head + local_head - 2) * kDim + d;
      const int64_t base =
          int64_t(conv_slot) * conv.slot_stride + channel * conv.channel_stride;
      float history[3], weights[4];
#pragma unroll
      for (int i = 0; i < 3; ++i)
        history[i] = float(
            conv.state[base + (num_accepted - 1 + i) * conv.token_stride]);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        weights[i] = float(conv.weight[channel * conv.weight_stride + i]);
      // Speculative history keeps the last two accepted history entries and
      // every new token. Ragged requests update only count + 2 entries.
      conv.state[base] = __hip_bfloat16(history[1]);
      conv.state[base + conv.token_stride] = __hip_bfloat16(history[2]);
      for (int t = 0; t < count; ++t) {
        const float x = float(mixed[(begin + t) * strides.mixed + channel]);
        float value = conv.bias ? float(conv.bias[channel]) : 0.0f;
#pragma unroll
        for (int i = 0; i < 3; ++i)
          value += conv_product(history[i], weights[i]);
        value += conv_product(x, weights[3]);
        convolved[t * local_channels + c] = __hip_bfloat16(
            value / (1.0f + exp2f(-value * 1.4426950408889634f)));
        conv.state[base + (t + 2) * conv.token_stride] = __hip_bfloat16(x);
        history[0] = history[1];
        history[1] = history[2];
        history[2] = x;
      }
    }
    __syncthreads();
  }
  __shared__ float query[kDim], key[kDim];
  __shared__ float decay, beta;
  __shared__ __hip_bfloat16 raw_output[kDim];
  for (int head_offset = 0; head_offset < heads_per_block; ++head_offset) {
    const int head = first_head + head_offset;
    float h[kRows][4];
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        h[row][i] = float(
            state[int64_t(source) * strides.state +
                  (head * kDim + wave + row * kWaves) * kDim + lane * 4 + i]);
    }
    for (int token = begin; token < end; ++token) {
      if (wave == 0) {
        float q[4], k[4], q2 = 0.0f, k2 = 0.0f;
        const bool packed_norm = FuseConv && width == 1;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int d = packed_norm ? lane * 4 + i : lane + i * 32;
          if constexpr (FuseConv) {
            q[i] = float(convolved[(token - begin) * local_channels + d]);
            k[i] =
                float(convolved[(token - begin) * local_channels + kDim + d]);
          } else {
            q[i] = float(mixed[token * strides.mixed + (head / 3) * kDim + d]);
            k[i] = float(
                mixed[token * strides.mixed + (kHeads + head / 3) * kDim + d]);
          }
          if (!packed_norm) {
            q2 += q[i] * q[i];
            k2 += k[i] * k[i];
          }
        }
        const float q_norm = packed_norm
            ? packed_norm_sqrt(packed_dot_sum(q, q) + 1.0e-6f)
            : rsqrtf(wave_sum(q2) + 1.0e-6f) * scale;
        const float k_norm = packed_norm
            ? packed_norm_sqrt(packed_dot_sum(k, k) + 1.0e-6f)
            : rsqrtf(wave_sum(k2) + 1.0e-6f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int d = packed_norm ? lane * 4 + i : lane + i * 32;
          query[d] = packed_norm ? (q[i] / q_norm) * scale : q[i] * q_norm;
          key[d] = packed_norm ? k[i] / k_norm : k[i] * k_norm;
        }
        if (lane == 0) {
          const float x =
              float(a[token * strides.a + head]) + float(dt_bias[head]);
          if constexpr (FuseConv) {
            // Preserve the packed decode reference's exp2/log expression.
            constexpr float log2e = 1.4426950408889634f;
            const float softplus = width == 1
                ? packed_softplus(x)
                : (x > 20.0f ? x : logf(1.0f + exp2f(x * log2e)));
            decay = exp2f((-exp2f(a_log[head] * log2e) * softplus) * log2e);
            beta = 1.0f /
                   (1.0f + exp2f(-float(b[token * strides.b + head]) * log2e));
          } else {
            const float softplus = x > 20.0f ? x : log1pf(expf(x));
            decay = expf(-expf(a_log[head]) * softplus);
            beta = sigmoid(float(b[token * strides.b + head]));
          }
        }
      }
      __syncthreads();
      float q[4], k[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        q[i] = query[lane * 4 + i];
        k[i] = key[lane * 4 + i];
      }
      const int destination = indices[request * width + token - begin];
      const bool packed_recurrence = FuseConv && width == 1;
#pragma unroll
      for (int row = 0; row < kRows; ++row) {
        const int value = wave + row * kWaves;
        float hk = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= decay;
          if (!packed_recurrence) hk += h[row][i] * k[i];
        }
        const float v = FuseConv
                            ? float(convolved[(token - begin) * local_channels +
                                              (2 + head_offset) * kDim + value])
                            : float(mixed[token * strides.mixed +
                                          (2 * kHeads + head) * kDim + value]);
        hk = packed_recurrence ? packed_dot_sum(h[row], k) : wave_sum(hk);
        const float delta = (v - hk) * beta;
        float hq = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += delta * k[i];
          if (!packed_recurrence) hq += h[row][i] * q[i];
        }
        if (destination > 0 && destination < slots)
          store_state4(state + int64_t(destination) * strides.state +
                           (head * kDim + value) * kDim + lane * 4,
                       h[row], aligned_state);
        hq = packed_recurrence ? packed_dot_sum(h[row], q) : wave_sum(hq);
        if (lane == 0) {
          if constexpr (FuseNorm)
            raw_output[value] = __hip_bfloat16(hq);
          else
            output[(int64_t(token) * kValueHeads + head) * kDim + value] =
                __hip_bfloat16(hq);
        }
      }
      __syncthreads();
      if (FuseNorm && wave == 0) {
        float values[4], sum_square = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          values[i] = float(raw_output[lane + i * 32]);
          sum_square += values[i] * values[i];
        }
        const float rstd = rsqrtf(wave_sum(sum_square) / kDim + epsilon);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int d = lane + i * 32;
          const float z = float(gate[token * strides.gate + head * kDim + d]);
          // Inductor's SiLU uses direct division, not a rounded reciprocal.
          const float activation = FuseConv && width == 1 && !sigmoid_gate
              ? z / (1.0f + expf(-z))
              : (sigmoid_gate ? sigmoid(z) : z * sigmoid(z));
          output[(int64_t(token) * kValueHeads + head) * kDim + d] =
              __hip_bfloat16(values[i] * rstd * float(norm_weight[d]) *
                             activation);
        }
      }
      // Protect shared query/key/output before the next token reuses them.
      __syncthreads();
    }
  }
}
}  // namespace

static void dispatch_gfx1151_qwen_gdn(
    const torch::Tensor& mixed, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_log, const torch::Tensor& dt_bias,
    const torch::Tensor& indices, const torch::Tensor& starts,
    const torch::Tensor& accepted, torch::Tensor& state,
    const std::optional<torch::Tensor>& gate,
    const std::optional<torch::Tensor>& norm_weight,
    torch::Tensor& output, double scale, double epsilon, bool sigmoid_gate,
    const std::optional<torch::Tensor>& conv_state,
    const std::optional<torch::Tensor>& conv_weight,
    const std::optional<torch::Tensor>& conv_bias) {
  TORCH_CHECK(mixed.is_cuda(), "mixed_qkv must be on a ROCm device");
  const c10::cuda::CUDAGuard guard(mixed.device());
  const auto* properties = at::cuda::getDeviceProperties(mixed.get_device());
  TORCH_CHECK(std::string(properties->gcnArchName).substr(0, 7) == "gfx1151",
              "GDN specialization requires gfx1151");
  for (const auto* tensor : std::initializer_list<const torch::Tensor*>{
           &a, &b, &a_log, &dt_bias, &indices, &starts, &accepted, &state,
           &output})
    TORCH_CHECK(tensor->device() == mixed.device(),
                "all GDN tensors must be on the same device");
  for (const auto* tensor : std::initializer_list<const torch::Tensor*>{
           &mixed, &a, &b, &output})
    TORCH_CHECK(tensor->scalar_type() == at::kBFloat16,
                "GDN activations and output must be bfloat16");
  TORCH_CHECK(mixed.dim() == 2 && mixed.size(0) > 0 && mixed.size(0) <= 256 &&
                  mixed.size(1) == kMixed && mixed.stride(1) == 1 &&
                  mixed.stride(0) >= kMixed,
              "mixed_qkv requires [1..256, 10240] with contiguous channels");
  const int tokens = mixed.size(0);
  for (const auto* tensor : {&a, &b})
    TORCH_CHECK(tensor->dim() == 2 && tensor->size(0) == tokens &&
                    tensor->size(1) == kValueHeads && tensor->stride(1) == 1 &&
                    tensor->stride(0) >= kValueHeads,
                "a and b require [tokens, 48] with contiguous heads");
  TORCH_CHECK(state.dim() == 4 && state.size(0) > 0 &&
                  state.size(1) == kValueHeads && state.size(2) == kDim &&
                  state.size(3) == kDim && state.stride(3) == 1 &&
                  state.stride(2) == kDim && state.stride(1) == kDim * kDim &&
                  state.stride(0) >= kValueHeads * kDim * kDim &&
                  (state.scalar_type() == at::kFloat ||
                   state.scalar_type() == at::kBFloat16),
              "state requires float32/bfloat16 [slots, 48, 128, 128] with "
              "contiguous slot contents");
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) >= 1 &&
                  indices.size(0) <= 8 && indices.size(1) >= 1 &&
                  indices.size(1) <= 32,
              "state_indices requires [1..8, 1..32]");
  for (const auto* tensor : {&indices, &starts, &accepted})
    TORCH_CHECK(tensor->scalar_type() == at::kInt && tensor->is_contiguous(),
                "GDN metadata must be contiguous int32");
  const int requests = indices.size(0);
  TORCH_CHECK(starts.dim() == 1 && starts.numel() == requests + 1 &&
                  accepted.dim() == 1 && accepted.numel() == requests,
              "query starts and accepted counts must match requests");
  TORCH_CHECK(a_log.scalar_type() == at::kFloat && a_log.dim() == 1 &&
                  a_log.numel() == kValueHeads && a_log.is_contiguous(),
              "A_log requires contiguous float32 [48]");
  TORCH_CHECK(gate.has_value() == norm_weight.has_value(),
              "GDN normalization requires both gate and norm_weight");
  const bool fuse_norm = gate.has_value();
  TORCH_CHECK(fuse_norm || (conv_state.has_value() && indices.size(1) == 1),
              "raw GDN decode requires fused convolution and M=1");
  for (const auto* tensor : {&dt_bias})
    TORCH_CHECK(tensor->dim() == 1 && tensor->is_contiguous() &&
                    (tensor->scalar_type() == at::kFloat ||
                     tensor->scalar_type() == at::kBFloat16),
                "dt_bias and norm_weight require contiguous float32/bfloat16");
  TORCH_CHECK(dt_bias.numel() == kValueHeads, "dt_bias requires [48]");
  if (fuse_norm) {
    TORCH_CHECK(gate->device() == mixed.device() &&
                    norm_weight->device() == mixed.device() &&
                    gate->scalar_type() == at::kBFloat16,
                "GDN gate requires bfloat16 on the input device");
    TORCH_CHECK(norm_weight->dim() == 1 && norm_weight->numel() == kDim &&
                    norm_weight->is_contiguous() &&
                    (norm_weight->scalar_type() == at::kFloat ||
                     norm_weight->scalar_type() == at::kBFloat16),
                "norm_weight requires contiguous float32/bfloat16 [128]");
  }
  for (const auto* tensor :
       std::initializer_list<const torch::Tensor*>{
           fuse_norm ? &*gate : &output, &output})
    TORCH_CHECK(
        tensor->dim() == 3 && tensor->size(0) == tokens &&
            tensor->size(1) == kValueHeads && tensor->size(2) == kDim &&
            tensor->stride(2) == 1 && tensor->stride(1) == kDim &&
            tensor->stride(0) >= kValueHeads * kDim,
        "gate and output require [tokens, 48, 128] with contiguous heads");
  TORCH_CHECK(output.is_contiguous(), "GDN output must be contiguous");
  TORCH_CHECK(std::isfinite(scale) && scale > 0 && std::isfinite(epsilon) &&
                  epsilon > 0,
              "scale and epsilon must be positive and finite");

  Convolution conv{};
  if (conv_state.has_value()) {
    TORCH_CHECK(conv_weight.has_value(), "convolution requires weights");
    const auto& cs = *conv_state;
    const auto& cw = *conv_weight;
    TORCH_CHECK(
        cs.device() == mixed.device() && cw.device() == mixed.device() &&
            cs.scalar_type() == at::kBFloat16 &&
            cw.scalar_type() == at::kBFloat16,
        "convolution state and weights require bfloat16 on the input device");
    TORCH_CHECK(cs.dim() == 3 && cs.size(0) > 0 && cs.size(1) == kMixed &&
                    cs.size(2) >= indices.size(1) + 2,
                "convolution state requires [slots, 10240, >= M + 2]");
    const bool ds = cs.stride(2) == 1 && cs.stride(1) >= cs.size(2) &&
                    cs.stride(0) >= kMixed * cs.stride(1);
    const bool sd = cs.stride(1) == 1 && cs.stride(2) >= kMixed &&
                    cs.stride(0) >= cs.size(2) * cs.stride(2);
    TORCH_CHECK(ds || sd,
                "convolution state requires non-overlapping DS or SD layout");
    TORCH_CHECK(cw.dim() == 2 && cw.size(0) == kMixed && cw.size(1) == 4 &&
                    cw.stride(1) == 1 && cw.stride(0) >= 4,
                "convolution weight requires [10240, 4] with contiguous taps");
    if (conv_bias.has_value()) {
      const auto& cb = *conv_bias;
      TORCH_CHECK(cb.device() == mixed.device() &&
                      cb.scalar_type() == at::kBFloat16 && cb.dim() == 1 &&
                      cb.numel() == kMixed && cb.is_contiguous(),
                  "convolution bias requires contiguous bfloat16 [10240]");
      conv.bias = reinterpret_cast<const __hip_bfloat16*>(cb.data_ptr());
    }
    conv.state = reinterpret_cast<__hip_bfloat16*>(cs.data_ptr());
    conv.weight = reinterpret_cast<const __hip_bfloat16*>(cw.data_ptr());
    conv.slot_stride = cs.stride(0);
    conv.channel_stride = cs.stride(1);
    conv.token_stride = cs.stride(2);
    conv.weight_stride = cw.stride(0);
    conv.slots = cs.size(0);
  }

  const auto stream = at::cuda::getCurrentCUDAStream(mixed.get_device());
  const Strides strides{mixed.stride(0), a.stride(0), b.stride(0),
                        fuse_norm ? gate->stride(0) : 0, state.stride(0)};
  const auto launch =
      [&]<typename State, typename Bias, typename Weight, bool FuseConv,
          bool FuseNorm = true>() {
        gfx1151_qwen_gdn_post_conv_kernel<State, Bias, Weight, FuseConv, FuseNorm>
            <<<dim3(requests, FuseConv ? kHeads : kValueHeads), kThreads, 0,
               stream>>>(
                reinterpret_cast<const __hip_bfloat16*>(mixed.data_ptr()),
                reinterpret_cast<const __hip_bfloat16*>(a.data_ptr()),
                reinterpret_cast<const __hip_bfloat16*>(b.data_ptr()),
                a_log.data_ptr<float>(),
                static_cast<const Bias*>(dt_bias.data_ptr()),
                indices.data_ptr<int>(), starts.data_ptr<int>(),
                accepted.data_ptr<int>(), static_cast<State*>(state.data_ptr()),
                fuse_norm ? reinterpret_cast<const __hip_bfloat16*>(
                                gate->data_ptr()) : nullptr,
                fuse_norm ? static_cast<const Weight*>(norm_weight->data_ptr())
                          : nullptr,
                reinterpret_cast<__hip_bfloat16*>(output.data_ptr()), tokens,
                state.size(0), indices.size(1), scale, epsilon, sigmoid_gate,
                strides, conv);
      };
  const auto dispatch_conv =
      [&]<typename State, typename Bias, typename Weight>() {
        if (conv_state.has_value())
          launch.template operator()<State, Bias, Weight, true>();
        else
          launch.template operator()<State, Bias, Weight, false>();
      };
  const auto dispatch_weight = [&]<typename State, typename Bias>() {
    if (!fuse_norm)
      launch.template operator()<State, Bias, float, true, false>();
    else if (norm_weight->scalar_type() == at::kFloat)
      dispatch_conv.template operator()<State, Bias, float>();
    else
      dispatch_conv.template operator()<State, Bias, __hip_bfloat16>();
  };
  const auto dispatch_bias = [&]<typename State>() {
    if (dt_bias.scalar_type() == at::kFloat)
      dispatch_weight.template operator()<State, float>();
    else
      dispatch_weight.template operator()<State, __hip_bfloat16>();
  };
  if (state.scalar_type() == at::kFloat)
    dispatch_bias.template operator()<float>();
  else
    dispatch_bias.template operator()<__hip_bfloat16>();
  const auto error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess,
              "gfx1151 GDN launch failed: ", hipGetErrorString(error));
}

void gfx1151_qwen_gdn_post_conv(
    const torch::Tensor& mixed, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_log, const torch::Tensor& dt_bias,
    const torch::Tensor& indices, const torch::Tensor& starts,
    const torch::Tensor& accepted, torch::Tensor& state,
    const torch::Tensor& gate, const torch::Tensor& norm_weight,
    torch::Tensor& output, double scale, double epsilon, bool sigmoid_gate) {
  dispatch_gfx1151_qwen_gdn(mixed, a, b, a_log, dt_bias, indices, starts,
                            accepted, state, gate, norm_weight, output, scale,
                            epsilon, sigmoid_gate, std::nullopt, std::nullopt,
                            std::nullopt);
}

void gfx1151_qwen_gdn_decode(
    const torch::Tensor& mixed, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_log, const torch::Tensor& dt_bias,
    const torch::Tensor& indices, const torch::Tensor& starts,
    const torch::Tensor& accepted, torch::Tensor& state,
    const torch::Tensor& gate, const torch::Tensor& norm_weight,
    torch::Tensor& output, double scale, double epsilon, bool sigmoid_gate,
    torch::Tensor& conv_state, const torch::Tensor& conv_weight,
    const std::optional<torch::Tensor>& conv_bias) {
  dispatch_gfx1151_qwen_gdn(mixed, a, b, a_log, dt_bias, indices, starts,
                            accepted, state, gate, norm_weight, output, scale,
                            epsilon, sigmoid_gate, conv_state, conv_weight,
                            conv_bias);
}

void gfx1151_qwen_gdn_decode_core(
    const torch::Tensor& mixed, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_log, const torch::Tensor& dt_bias,
    const torch::Tensor& indices, const torch::Tensor& starts,
    const torch::Tensor& accepted, torch::Tensor& state,
    torch::Tensor& output, double scale, torch::Tensor& conv_state,
    const torch::Tensor& conv_weight,
    const std::optional<torch::Tensor>& conv_bias) {
  dispatch_gfx1151_qwen_gdn(mixed, a, b, a_log, dt_bias, indices, starts,
                            accepted, state, std::nullopt, std::nullopt, output,
                            scale, 1.0e-6, false, conv_state, conv_weight,
                            conv_bias);
}
