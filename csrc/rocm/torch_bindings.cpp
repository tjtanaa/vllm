#include "core/registration.h"
#include "rocm/ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) {
  // vLLM custom ops for rocm

  rocm_ops.def(
      "gfx1151_qwen_gdn_decode_core(Tensor mixed, Tensor a, Tensor b, "
      "Tensor a_log, Tensor dt_bias, Tensor indices, Tensor starts, "
      "Tensor accepted, Tensor(s!) state, Tensor(o!) output, float scale, "
      "Tensor(c!) conv_state, Tensor conv_weight, Tensor? conv_bias) -> ()");
  rocm_ops.impl("gfx1151_qwen_gdn_decode_core", torch::kCUDA,
                &gfx1151_qwen_gdn_decode_core);

  rocm_ops.def(
      "gfx1151_qwen_gdn_decode(Tensor mixed, Tensor a, Tensor b, Tensor a_log, "
      "Tensor dt_bias, Tensor indices, Tensor starts, Tensor accepted, "
      "Tensor(s!) state, Tensor gate, Tensor norm_weight, Tensor(o!) output, "
      "float scale, float epsilon, bool sigmoid_gate, Tensor(c!) conv_state, "
      "Tensor conv_weight, Tensor? conv_bias) -> ()");
  rocm_ops.impl("gfx1151_qwen_gdn_decode", torch::kCUDA,
                &gfx1151_qwen_gdn_decode);

  rocm_ops.def(
      "gfx1151_qwen_gdn_post_conv(Tensor mixed, Tensor a, Tensor b, Tensor a_log, "
      "Tensor dt_bias, Tensor indices, Tensor starts, Tensor accepted, "
      "Tensor(s!) state, Tensor gate, Tensor norm_weight, Tensor(o!) output, "
      "float scale, float epsilon, bool sigmoid_gate) -> ()");
  rocm_ops.impl("gfx1151_qwen_gdn_post_conv", torch::kCUDA,
                &gfx1151_qwen_gdn_post_conv);

  rocm_ops.def(
      "gfx1151_qwen_paged_attention(Tensor query, Tensor key_cache, "
      "Tensor value_cache, Tensor block_table, Tensor seq_lens, "
      "Tensor query_start_loc, Tensor? sinks, Tensor(a!) output, "
      "Tensor(b!) workspace, int max_seq_len, int max_query_len, float scale, "
      "int sliding_window, bool causal) -> ()");
  rocm_ops.impl("gfx1151_qwen_paged_attention", torch::kCUDA,
                &gfx1151_qwen_paged_attention);

  rocm_ops.def(
      "dflash2_grouped_conv(Tensor hidden, Tensor delta, Tensor base, "
      "int block_size, int group_size, int num_groups, int side) -> Tensor");
  rocm_ops.impl("dflash2_grouped_conv", torch::kCUDA,
                &dflash2_grouped_conv);

// skinny_gemms.cu (LLMM1/wvSplitK/wvSplitKrc/wvSplitKQ) is excluded on gfx1250
// (gfx9/gfx11 ISA, unsupported there); skip these registrations to avoid
// undefined symbols. vLLM uses default/Triton GEMM for these ops on gfx1250.
#ifndef VLLM_SKIP_SKINNY_GEMMS
  // Custom gemm op for matrix-vector multiplication
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> "
      "Tensor");
  rocm_ops.impl("LLMM1", torch::kCUDA, &LLMM1);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitK", torch::kCUDA, &wvSplitK);

  // W4A16 grouped skinny GEMM: packed int4 weights, per-group scales,
  // optional zero points [M/8, K/group_size] int32 for asymmetric
  // quantization
  rocm_ops.def(
      "wvSplitK_int4_g(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_zero_points, Tensor? in_bias, int CuCount, "
      "int group_size) -> Tensor");
  rocm_ops.impl("wvSplitK_int4_g", torch::kCUDA, &wvSplitK_int4_g);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitKrc(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitKrc", torch::kCUDA, &wvSplitKrc);

  // wvSplitK for fp8
  rocm_ops.def(
      "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor? in_bias, Tensor! out_c, "
      "Tensor scale_a, "
      "          Tensor scale_b, int CuCount) -> ()");
  rocm_ops.impl("wvSplitKQ", torch::kCUDA, &wvSplitKQ);
#endif  // VLLM_SKIP_SKINNY_GEMMS

#ifdef VLLM_ROCM_GFX1100
  // W4A16 GPTQ kernels for AMD RDNA3 (gfx1100).
  rocm_ops.def(
      "gptq_gemm_rdna3(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3", torch::kCUDA, &gptq_gemm_rdna3);

  rocm_ops.def(
      "gptq_gemm_rdna3_wmma(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3_wmma", torch::kCUDA, &gptq_gemm_rdna3_wmma);

  rocm_ops.def(
      "moe_gptq_gemm_rdna3(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor b_scales, Tensor b_qzeros, Tensor topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk) -> ()");
  rocm_ops.impl("moe_gptq_gemm_rdna3", torch::kCUDA, &moe_gptq_gemm_rdna3);
#endif

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor seq_lens,"
      "                Tensor? query_start_loc,"
      "                int block_size,"
      "                int max_seq_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                str mfma_type) -> ()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
