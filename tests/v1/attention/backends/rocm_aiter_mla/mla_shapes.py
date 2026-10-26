# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank dense MLA shapes that pick each ROCM_AITER_MLA decode kernel.

MLA caches one latent head shared by every rank, so TP only splits the query
heads, and the per-rank head count is what selects the decode kernel: AITER's
asm decode needs a multiple of 16 heads, 1-15 heads take Gluon (divisors of
16, gfx950 only) or a padded asm decode, unaligned counts up to 128 are padded
to the next multiple of 16, and 24 runs natively when the AITER build has the
H24 reducer and planner.
"""

from tests.v1.attention.model_shapes import (
    DEEPSEEK_V2_LITE,
    DEEPSEEK_V3,
    GLM_4_7_FLASH,
    KIMI_K2,
    KIMI_K3_MLA,
    KIMI_LINEAR,
    MLAShape,
)

DSV3_TP1 = DEEPSEEK_V3.shard(1)  # 128: DP attention
DSV3_TP4 = DEEPSEEK_V3.shard(4)  # 32
DSV3_TP8 = DEEPSEEK_V3.shard(8)  # 16: the asm minimum
DSV3_TP16 = DEEPSEEK_V3.shard(16)  # 8: Gluon divisor
KIMI_K2_TP8 = KIMI_K2.shard(8)  # 8
KIMI_K3_TP1 = KIMI_K3_MLA.shard(1)  # 96: DP attention
KIMI_K3_TP2 = KIMI_K3_MLA.shard(2)  # 48
KIMI_K3_TP4 = KIMI_K3_MLA.shard(4)  # 24: native H24 or padded to 32
KIMI_K3_TP8 = KIMI_K3_MLA.shard(8)  # 12: non-divisor, padded asm
KIMI_LINEAR_TP1 = KIMI_LINEAR.shard(1)  # 32
KIMI_LINEAR_TP8 = KIMI_LINEAR.shard(8)  # 4: Gluon divisor
DSV2_LITE_TP1 = DEEPSEEK_V2_LITE.shard(1)  # 16
DSV2_LITE_TP4 = DEEPSEEK_V2_LITE.shard(4)  # 4
GLM47_FLASH_TP1 = GLM_4_7_FLASH.shard(1)  # 20: padded to 32, nope 192 / v 256
GLM47_FLASH_TP2 = GLM_4_7_FLASH.shard(2)  # 10: non-divisor
GLM47_FLASH_TP4 = GLM_4_7_FLASH.shard(4)  # 5: non-divisor

ALL_SHAPES: list[MLAShape] = [
    DSV3_TP1,
    DSV3_TP4,
    DSV3_TP8,
    DSV3_TP16,
    KIMI_K2_TP8,
    KIMI_K3_TP1,
    KIMI_K3_TP2,
    KIMI_K3_TP4,
    KIMI_K3_TP8,
    KIMI_LINEAR_TP1,
    KIMI_LINEAR_TP8,
    DSV2_LITE_TP1,
    DSV2_LITE_TP4,
    GLM47_FLASH_TP1,
    GLM47_FLASH_TP2,
    GLM47_FLASH_TP4,
]

# One shape per decode-kernel regime, for the cross products: 16 aligned,
# 128 aligned, 8 Gluon divisor, 12 padded non-divisor, 24 native-or-padded,
# 20 padded with GLM dims.
REGIME_SHAPES = [
    DSV3_TP8,
    DSV3_TP1,
    DSV3_TP16,
    KIMI_K3_TP8,
    KIMI_K3_TP4,
    GLM47_FLASH_TP1,
]
