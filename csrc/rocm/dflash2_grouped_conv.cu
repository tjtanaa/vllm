#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include <torch/all.h>

namespace {

constexpr int kThreads = 256;
constexpr int kChannelsPerThread = 4;

__global__ void dflash2_grouped_conv_bf16_kernel(
    const __hip_bfloat16* hidden, const __hip_bfloat16* delta,
    const __hip_bfloat16* base, __hip_bfloat16* output, int64_t num_elements,
    int hidden_size, int group_size, int num_groups, int64_t delta_token_stride,
    int block_size, int side) {
  const int64_t first =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kChannelsPerThread;

#pragma unroll
  for (int offset = 0; offset < kChannelsPerThread; ++offset) {
    const int64_t index = first + offset;
    if (index >= num_elements) return;

    const int token = index / hidden_size;
    const int channel = index - static_cast<int64_t>(token) * hidden_size;
    const int group = channel / group_size;
    const int base_offset = side * 2 * hidden_size + channel;
    const int64_t delta_offset = token * delta_token_stride + group;

    float value = static_cast<float>(base[base_offset]) +
                  static_cast<float>(delta[delta_offset]);
    value *= static_cast<float>(hidden[index]);
    if (token % block_size != 0) {
      float previous = static_cast<float>(base[base_offset + hidden_size]) +
                       static_cast<float>(delta[delta_offset + num_groups]);
      value += previous * static_cast<float>(hidden[index - hidden_size]);
    }
    output[index] = __hip_bfloat16(value);
  }
}

}  // namespace

torch::Tensor dflash2_grouped_conv(
    const torch::Tensor& hidden, const torch::Tensor& delta,
    const torch::Tensor& base, int64_t block_size, int64_t group_size,
    int64_t num_groups, int64_t side) {
  TORCH_CHECK(hidden.is_cuda() && delta.is_cuda() && base.is_cuda(),
              "dflash2_grouped_conv requires GPU tensors");
  TORCH_CHECK(hidden.scalar_type() == torch::kBFloat16 &&
                  delta.scalar_type() == torch::kBFloat16 &&
                  base.scalar_type() == torch::kBFloat16,
              "dflash2_grouped_conv requires bfloat16 tensors");
  TORCH_CHECK(hidden.is_contiguous() && base.is_contiguous(),
              "dflash2_grouped_conv requires contiguous hidden and base tensors");
  TORCH_CHECK(delta.dim() == 3 && delta.size(1) == 2 &&
                  delta.size(2) == num_groups && delta.stride(2) == 1 &&
                  delta.stride(1) == num_groups,
              "delta must have shape [tokens, 2, num_groups] with contiguous "
              "inner dimensions");
  TORCH_CHECK(base.dim() == 3 && base.size(0) == 2 && base.size(1) == 2,
              "base must have shape [2, 2, hidden_size]");
  TORCH_CHECK(side == 0 || side == 1, "side must be 0 or 1");

  const auto hidden_size = hidden.size(-1);
  TORCH_CHECK(hidden_size % group_size == 0 &&
                  hidden_size / group_size == num_groups,
              "group dimensions do not match hidden_size");
  TORCH_CHECK(delta.numel() ==
                  hidden.numel() / hidden_size * 2 * num_groups,
              "delta shape does not match hidden_states");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(hidden));
  auto output = torch::empty_like(hidden);
  const int64_t work_items =
      (hidden.numel() + kChannelsPerThread - 1) / kChannelsPerThread;
  const int blocks = (work_items + kThreads - 1) / kThreads;
  const auto stream = at::cuda::getCurrentCUDAStream();
  dflash2_grouped_conv_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const __hip_bfloat16*>(hidden.data_ptr()),
      reinterpret_cast<const __hip_bfloat16*>(delta.data_ptr()),
      reinterpret_cast<const __hip_bfloat16*>(base.data_ptr()),
      reinterpret_cast<__hip_bfloat16*>(output.data_ptr()), hidden.numel(),
      hidden_size, group_size, num_groups, delta.stride(0), block_size, side);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
