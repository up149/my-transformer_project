# src/models/layers.py
"""
layers.py
----------
定义 Transformer 的核心子层结构：
- RelPosMultiHeadAttention：带相对位置偏置的自注意力；
- CrossMultiHeadAttention：编码器-解码器交叉注意力；
- FeedForward：前馈网络层；
- EncoderLayer / DecoderLayer：编码器与解码器的完整层结构；
- InputEmbedding：词嵌入与位置嵌入层。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .relative_position import RelativePositionBias


class RelPosMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout,
                 num_buckets, max_distance):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.drop = nn.Dropout(dropout)
        self.rel_pos_bias = RelativePositionBias(
            num_heads=n_heads,
            num_buckets=num_buckets,
            max_distance=max_distance
        )

    def forward(self, x, mask=None):
        B, L, d = x.shape
        h, dk = self.h, self.dk
        Q = self.Wq(x).view(B, L, h, dk).transpose(1, 2)
        K = self.Wk(x).view(B, L, h, dk).transpose(1, 2)
        V = self.Wv(x).view(B, L, h, dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        scores = scores + self.rel_pos_bias(L, L, x.device).unsqueeze(0)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).contiguous().view(B, L, d)
        return self.Wo(out)


class CrossMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Lq, d = q.shape
        _, Lk, _ = k.shape
        h, dk = self.h, self.dk

        Q = self.Wq(q).view(B, Lq, h, dk).transpose(1, 2)
        K = self.Wk(k).view(B, Lk, h, dk).transpose(1, 2)
        V = self.Wv(v).view(B, Lk, h, dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).contiguous().view(B, Lq, d)
        return self.Wo(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout,
                 num_buckets, max_distance):
        super().__init__()
        self.self_attn = RelPosMultiHeadAttention(
            d_model, n_heads, dropout,
            num_buckets=num_buckets,
            max_distance=max_distance
        )
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn_out = self.self_attn(x, src_mask)
        x = self.norm1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout,
                 num_buckets, max_distance):
        super().__init__()
        self.self_attn = RelPosMultiHeadAttention(
            d_model, n_heads, dropout,
            num_buckets=num_buckets,
            max_distance=max_distance
        )
        self.cross_attn = CrossMultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, tgt_mask, mem_mask):
        sa_out = self.self_attn(x, tgt_mask)
        x = self.norm1(x + self.drop(sa_out))
        ca_out = self.cross_attn(x, mem, mem, mem_mask)
        x = self.norm2(x + self.drop(ca_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.drop(ff_out))
        return x


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, pad_id, dropout, max_len):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids):
        B, L = ids.size()
        tok = self.embed_tokens(ids)
        pos_ids = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        pos = self.pos_embed(pos_ids)
        return self.drop(tok + pos)
