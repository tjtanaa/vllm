# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention-layer shapes of current flagship models, for backend test matrices.

Values are copied from each model's Hugging Face ``config.json`` (read on
2026-10-24). Only models whose layers run through a dense MHA/GQA backend are
listed; MLA and sparse-attention models (DeepSeek V3/V4/V4.1, GLM-5.x, MiniMax
M3's LM layers) belong to the MLA/sparse backend suites. Re-read the configs
rather than hand-editing a number.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AttnShape:
    """One attention layer kind of one model, at a given TP degree."""

    model: str
    num_heads: int
    num_kv_heads: int
    head_size: int
    sliding_window: int | None = None
    tp: int = 1

    @property
    def id(self) -> str:
        name = self.model.split("/")[-1]
        window = f"-swa{self.sliding_window}" if self.sliding_window else ""
        return (
            f"{name}-tp{self.tp}-h{self.num_heads}kv{self.num_kv_heads}"
            f"d{self.head_size}{window}"
        )

    def shard(self, tp: int) -> "AttnShape":
        """Per-rank shape; KV heads are replicated once tp exceeds them."""
        assert self.num_heads % tp == 0
        return AttnShape(
            self.model,
            self.num_heads // tp,
            max(self.num_kv_heads // tp, 1),
            self.head_size,
            self.sliding_window,
            tp,
        )


QWEN3_235B = AttnShape("Qwen/Qwen3-235B-A22B", 64, 4, 128)
QWEN3_8_27B = AttnShape("Qwen/Qwen3.8-27B", 24, 4, 256)
QWEN3_8_2_4T = AttnShape("Qwen/Qwen3.8-2.4T-A95B", 64, 4, 256)
# Served by its own QSA backend; kept as a GQA-12 head-256 shape.
QWEN3_8_FLASH_NEXT = AttnShape("Qwen/Qwen3.8-Flash-Next", 24, 2, 256)
GEMMA4_31B_SLIDING = AttnShape("google/gemma-4-31B-it", 32, 16, 256, 1024)
GEMMA4_31B_GLOBAL = AttnShape("google/gemma-4-31B-it", 32, 4, 512)
GEMMA4_26B_SLIDING = AttnShape("google/gemma-4-26B-A4B-it", 16, 8, 256, 1024)
GEMMA4_26B_GLOBAL = AttnShape("google/gemma-4-26B-A4B-it", 16, 2, 512)
# Gated repo; values from the public mlx-community conversion of its config.
MUSE_SPARK_SLIDING = AttnShape(
    "meta-models/Muse-Spark-Open-816B-A42B", 128, 16, 64, 2048
)
MUSE_SPARK_FULL = AttnShape("meta-models/Muse-Spark-Open-816B-A42B", 128, 16, 64)
# MiniMax M3's LM layers are block-sparse; its dense shape is still a common
# GQA-16 head-128 case and matches the ViT path that can use dense backends.
MINIMAX_M3 = AttnShape("MiniMaxAI/MiniMax-M3", 64, 4, 128)

ALL_SHAPES = [
    QWEN3_235B,
    QWEN3_8_27B,
    QWEN3_8_2_4T,
    QWEN3_8_FLASH_NEXT,
    GEMMA4_31B_SLIDING,
    GEMMA4_31B_GLOBAL,
    GEMMA4_26B_SLIDING,
    GEMMA4_26B_GLOBAL,
    MUSE_SPARK_SLIDING,
    MUSE_SPARK_FULL,
    MINIMAX_M3,
]

# Per-rank shapes that change the H/KV ratio or replicate KV heads.
TP_SHARDED_SHAPES = [
    QWEN3_235B.shard(8),
    QWEN3_8_27B.shard(4),
    MUSE_SPARK_FULL.shard(8),
    MUSE_SPARK_SLIDING.shard(8),
    GEMMA4_31B_SLIDING.shard(2),
]


def shapes_supported_by(head_sizes: list[int]) -> list[AttnShape]:
    """Shapes a backend with the given head sizes must run (TP=1 and sharded)."""
    return [s for s in ALL_SHAPES + TP_SHARDED_SHAPES if s.head_size in head_sizes]


def shapes_rejected_by(head_sizes: list[int]) -> list[AttnShape]:
    return [s for s in ALL_SHAPES if s.head_size not in head_sizes]
