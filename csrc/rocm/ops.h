#pragma once

#include <torch/all.h>

void gfx1151_qwen_gdn_decode_core(
    const torch::Tensor& mixed, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_log, const torch::Tensor& dt_bias,
    const torch::Tensor& indices, const torch::Tensor& starts,
    const torch::Tensor& accepted, torch::Tensor& state,
    torch::Tensor& output, double scale, torch::Tensor& conv_state,
    const torch::Tensor& conv_weight,
    const std::optional<torch::Tensor>& conv_bias);

void gfx1151_qwen_gdn_decode(
    const torch::Tensor& mixed, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_log, const torch::Tensor& dt_bias,
    const torch::Tensor& indices, const torch::Tensor& starts,
    const torch::Tensor& accepted, torch::Tensor& state,
    const torch::Tensor& gate, const torch::Tensor& norm_weight,
    torch::Tensor& output, double scale, double epsilon, bool sigmoid_gate,
    torch::Tensor& conv_state, const torch::Tensor& conv_weight,
    const std::optional<torch::Tensor>& conv_bias);

void gfx1151_qwen_gdn_post_conv(
    const torch::Tensor& mixed, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_log, const torch::Tensor& dt_bias,
    const torch::Tensor& indices, const torch::Tensor& starts,
    const torch::Tensor& accepted, torch::Tensor& state,
    const torch::Tensor& gate, const torch::Tensor& norm_weight,
    torch::Tensor& output, double scale, double epsilon, bool sigmoid_gate);

void gfx1151_qwen_paged_attention(
    const torch::Tensor& query, const torch::Tensor& key_cache,
    const torch::Tensor& value_cache, const torch::Tensor& block_table,
    const torch::Tensor& seq_lens, const torch::Tensor& query_start_loc,
    const std::optional<torch::Tensor>& sinks, torch::Tensor& output,
    torch::Tensor& workspace, int64_t max_seq_len, int64_t max_query_len,
    double scale, int64_t sliding_window, bool causal);

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

torch::Tensor wvSplitK_int4_g(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_zero_points,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size);

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor dflash2_grouped_conv(
    const torch::Tensor& hidden, const torch::Tensor& delta,
    const torch::Tensor& base, int64_t block_size, int64_t group_size,
    int64_t num_groups, int64_t side);

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              bool use_v2_format);

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales, bool use_v2_format);

void moe_gptq_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type);
