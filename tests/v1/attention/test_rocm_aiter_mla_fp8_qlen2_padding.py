# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_MLA fp8 2-token verify blocks against a float32 reference.

On gfx950 AITER serves fp8 qlen-2 MLA decode at 16 heads (and every padded
count below 16) with its qseqlen-2 kernel, which returns wrong rows (head 0
of the second query) when a request's KV length sits just past a tile edge.
DeepSeek-V3 at TP8 with MTP-1 and an fp8 cache hits it. The backend pads
these blocks to the qseqlen-4 kernel; rows with no query tokens must not
switch the padding off.

The test drives a real ``MLAAttention`` layer bound to the backend: the
latent is written through ``update_kv_cache`` and ``forward_impl`` runs the
query absorption, fp8 query quantization, ``forward_mqa`` and the ``W_UV``
up-projection, as the model runner does.
"""

import pytest
import torch

from vllm.platforms import current_platform


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm._aiter_ops import is_aiter_found
    from vllm.platforms.rocm import on_gfx950

    return is_aiter_found() and on_gfx950()


pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the fp8 qseqlen-2 padding is gfx950-only"
)

BLOCK_SIZE = 16
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
SCALE = (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5
# Non-unit, so a dropped scale shows up as an error.
K_SCALE = 0.5
Q_SCALE = 0.25
LAYER = "model.layers.0.self_attn.attn"

# KV lengths just past AITER's tile edges, where the qseqlen-2 kernel returned
# rows off by up to 0.9; correct rows stay within fp8 rounding (~0.06).
SEQ_LENS = [[9, 10, 11, 12], [33, 900, 129, 5000], [17, 65, 257, 1025]]
MAX_ABS_ERR = 0.2


def _fp8_round(x: torch.Tensor, scale: float) -> torch.Tensor:
    fp8 = current_platform.fp8_dtype()
    return (x.float() / scale).to(fp8).float() * scale


class _Case:
    """One MLA layer at ``num_heads`` per rank with an fp8 latent cache."""

    def __init__(self, num_heads: int, num_blocks: int):
        from tests.v1.attention.utils import dense_kv_cache_views
        from vllm.config import (
            CacheConfig,
            ModelConfig,
            SchedulerConfig,
            SpeculativeConfig,
            VllmConfig,
            set_current_vllm_config,
        )
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        from vllm.model_executor.layers.linear import ColumnParallelLinear
        from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
            AiterMLABackend,
            AiterMLAMetadataBuilder,
        )
        from vllm.v1.kv_cache_layout import KVCacheLayout

        self.num_heads = num_heads
        self.device = torch.device("cuda")
        model_config = ModelConfig(
            model="deepseek-ai/DeepSeek-R1",
            dtype=torch.bfloat16,
            max_model_len=8192,
            hf_overrides={
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_heads,
            },
        )
        cache_config = CacheConfig(block_size=BLOCK_SIZE, cache_dtype="fp8")
        cache_config.num_gpu_blocks = num_blocks
        cache_config.num_cpu_blocks = 0
        cache_config.kv_cache_layout = "LBHNC"
        self.config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            scheduler_config=SchedulerConfig(
                max_num_seqs=16,
                max_num_batched_tokens=8192,
                max_model_len=8192,
                is_encoder_decoder=False,
            ),
        )
        self.config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=1
        )
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with set_current_vllm_config(self.config), torch.device(self.device):
                kv_b_proj = ColumnParallelLinear(
                    KV_LORA_RANK,
                    num_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
                    bias=False,
                    params_dtype=torch.bfloat16,
                    prefix="model.layers.0.self_attn.kv_b_proj",
                    disable_tp=True,
                )
                # Unit-variance K_nope and V per head for a unit-variance latent.
                kv_b_proj.weight.data.normal_(std=KV_LORA_RANK**-0.5)
                self.layer = MLAAttention(
                    num_heads=num_heads,
                    scale=SCALE,
                    qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                    qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                    v_head_dim=V_HEAD_DIM,
                    q_lora_rank=None,
                    kv_lora_rank=KV_LORA_RANK,
                    kv_b_proj=kv_b_proj,
                    cache_config=cache_config,
                    prefix=LAYER,
                    attn_backend=AiterMLABackend,
                )
                self.layer.process_weights_after_loading(torch.bfloat16)
                spec = AiterMLABackend.customize_spec(
                    self.layer.get_kv_cache_spec(self.config)
                )
                self.builder = AiterMLAMetadataBuilder(
                    spec, [LAYER], self.config, self.device
                )
        finally:
            torch.set_default_dtype(prev_dtype)
        # The latent cache as the model runner allocates and views it.
        raw = torch.zeros(
            num_blocks * spec.page_size_bytes, dtype=torch.int8, device=self.device
        )
        (view,) = dense_kv_cache_views(
            raw, spec, num_blocks, num_layers=1, layout=KVCacheLayout.LBHNC
        )
        self.layer.bind_kv_cache(view)
        kv_cache = self.kv_cache = self.layer.kv_cache
        self.layer._k_scale.fill_(K_SCALE)
        self.layer._q_scale.fill_(Q_SCALE)
        self.layer._k_scale_float = K_SCALE
        self.layer._q_scale_float = Q_SCALE
        num_slots = num_blocks * BLOCK_SIZE
        # Values the fp8 cache stores exactly, so only accumulation error is left.
        self.latent = _fp8_round(
            torch.randn(num_slots, KV_LORA_RANK + QK_ROPE_HEAD_DIM, device="cuda"),
            K_SCALE,
        ).to(torch.bfloat16)
        kv_c, k_pe = self._split(self.latent)
        self.layer.update_kv_cache(
            kv_c,
            k_pe,
            kv_cache,
            torch.arange(num_slots, device=self.device),
            None,
            "fp8",
            self.layer._k_scale,
        )

    @staticmethod
    def _split(latent: torch.Tensor):
        kv_c, k_pe = latent.split([KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)
        return kv_c.contiguous(), k_pe.unsqueeze(1).contiguous()

    def run(self, query_lens: list[int], seq_lens: list[int], seed: int):
        """Forward one batch; return (max abs error of the real rows, metadata)."""
        from vllm.v1.attention.backend import CommonAttentionMetadata

        gen = torch.Generator().manual_seed(seed)
        num_reqs = len(seq_lens)
        max_blocks = max(1, *(-(-s // BLOCK_SIZE) for s in seq_lens))
        # Random, non-aliasing pages; block 0 stays free as the pad value.
        perm = torch.randperm(self.kv_cache.shape[0] - 1, generator=gen) + 1
        block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32)
        slots, query_slots = [], []
        cursor = 0
        for i, (q, s) in enumerate(zip(query_lens, seq_lens)):
            n = -(-s // BLOCK_SIZE)
            block_table[i, :n] = perm[cursor : cursor + n].to(torch.int32)
            cursor += n
            pos = torch.arange(s)
            req = block_table[i, pos // BLOCK_SIZE].long() * BLOCK_SIZE
            slots.append(req + pos % BLOCK_SIZE)
            query_slots.append(slots[-1][s - q :])
        qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
        qsl[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
        num_tokens = sum(query_lens)
        common = CommonAttentionMetadata(
            query_start_loc=qsl.to(self.device),
            query_start_loc_cpu=qsl,
            seq_lens=seq_lens_t.to(self.device),
            seq_lens_cpu_upper_bound=seq_lens_t,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max(query_lens),
            max_seq_len=max(seq_lens),
            block_table_tensor=block_table.to(self.device),
            slot_mapping=torch.cat(query_slots).to(self.device),
        )
        metadata = self.builder.build(0, common)
        query = torch.randn(
            num_tokens,
            self.num_heads,
            QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        kv_c, k_pe = self._split(self.latent[torch.cat(query_slots).cuda()])
        self.layer.update_kv_cache(
            kv_c,
            k_pe,
            self.kv_cache,
            common.slot_mapping,
            metadata,
            "fp8",
            self.layer._k_scale,
        )
        output = torch.full(
            (num_tokens, self.num_heads * V_HEAD_DIM),
            float("nan"),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.layer.forward_impl(query, kv_c, k_pe, self.kv_cache, metadata, output)
        actual = output.view(num_tokens, self.num_heads, V_HEAD_DIM).float()
        expected = self._reference(query, query_lens, seq_lens, slots)
        assert torch.isfinite(actual).all()
        return (actual - expected).abs().max().item(), metadata

    def _reference(self, query, query_lens, seq_lens, slots) -> torch.Tensor:
        """Float32 absorbed-form MLA over the fp8-rounded query and latent."""
        w = self.layer.kv_b_proj.weight.float().view(
            self.num_heads, QK_NOPE_HEAD_DIM + V_HEAD_DIM, KV_LORA_RANK
        )
        w_uk, w_uv = w[:, :QK_NOPE_HEAD_DIM], w[:, QK_NOPE_HEAD_DIM:]
        q_nope, q_pe = query.float().split([QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM], -1)
        q_lat = torch.cat([torch.einsum("thp,hpl->thl", q_nope, w_uk), q_pe], -1)
        q_lat = _fp8_round(q_lat, Q_SCALE)
        out, offset = [], 0
        for q_len, s_len, req_slots in zip(query_lens, seq_lens, slots):
            k = self.latent[req_slots.cuda()].float()
            q = q_lat[offset : offset + q_len]
            logits = torch.einsum("thd,kd->htk", q, k) * SCALE
            q_pos = torch.arange(s_len - q_len, s_len, device="cuda")
            mask = torch.arange(s_len, device="cuda")[None, :] <= q_pos[:, None]
            probs = logits.masked_fill(~mask[None], float("-inf")).softmax(-1)
            o_lat = torch.einsum("htk,kl->thl", probs, k[:, :KV_LORA_RANK])
            out.append(torch.einsum("thl,hvl->thv", o_lat, w_uv))
            offset += q_len
        return torch.cat(out)


@pytest.mark.parametrize("num_heads", [16, 8], ids=["tp8-h16", "tp16-h8"])
@pytest.mark.parametrize(
    "tail", [[], [0, 0], [700, 37]], ids=["no-tail", "pad-seq0", "kv-load"]
)
@torch.inference_mode()
def test_fp8_two_token_verify_matches_reference(
    dist_init, workspace_init, num_heads, tail
):
    """MTP-1 verify blocks, optionally followed by rows with no query tokens
    (padding requests whose tokens were not padded, or requests whose KV an
    async connector is loading)."""
    torch.manual_seed(0)
    case = _Case(num_heads, num_blocks=1024)
    worst = 0.0
    for seq_lens in SEQ_LENS:
        for seed in range(4):
            err, metadata = case.run(
                [2] * len(seq_lens) + [0] * len(tail), seq_lens + tail, seed
            )
            assert metadata.decode is not None and metadata.num_prefills == 0
            assert metadata.decode.max_qo_len == 4
            worst = max(worst, err)
    assert worst <= MAX_ABS_ERR, f"max abs error {worst:.3f}"
