# src/models/relative_position.py
"""
relative_position.py
---------------------
实现相对位置偏置（Relative Position Bias）模块，
采用 T5 风格的 bucket 映射机制，将距离信息编码为离散桶，
为多头注意力提供可学习的相对位置信息。
"""

import math
import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, num_buckets, max_distance):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position):
        rp = relative_position
        sign = (rp < 0).to(torch.long)
        dist = torch.abs(rp)

        max_exact = self.num_buckets // 2
        is_small = dist < max_exact
        small_bucket = dist

        large_bucket = max_exact + (
            torch.log(dist.float()/max_exact + 1e-6) /
            math.log(self.max_distance/max_exact + 1e-6) *
            (self.num_buckets - max_exact)
        ).to(torch.long)
        large_bucket = torch.min(
            large_bucket,
            torch.full_like(large_bucket, self.num_buckets - 1)
        )

        buckets = torch.where(is_small, small_bucket, large_bucket)
        half = self.num_buckets // 2
        buckets = torch.where(sign.bool(), buckets, buckets + half)
        buckets = torch.clamp(buckets, max=self.num_buckets - 1)
        return buckets

    def forward(self, qlen, klen, device):
        q_pos = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
        k_pos = torch.arange(klen, dtype=torch.long, device=device)[None, :]
        rel_pos = q_pos - k_pos
        buckets = self._relative_position_bucket(rel_pos)
        values = self.relative_attention_bias(buckets)
        values = values.permute(2, 0, 1)
        return values
