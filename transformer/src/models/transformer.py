# src/models/transformer.py
"""
transformer.py
---------------
构建完整的 Seq2Seq Transformer 模型：
- TransformerEncoder / TransformerDecoder：堆叠多层注意力与前馈模块；
- Seq2SeqTransformer：封装整体前向传播、mask 构建及损失计算；
实现编码器-解码器结构的端到端翻译模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG
from ..utils import label_smoothed_nll_loss
from .layers import (
    EncoderLayer,
    DecoderLayer,
    InputEmbedding,
)


class TransformerEncoder(nn.Module):
    def __init__(self, cfg, shared_embed):
        super().__init__()
        self.embed = shared_embed
        self.layers = nn.ModuleList([
            EncoderLayer(
                cfg["d_model"], cfg["n_heads"], cfg["d_ff"], cfg["dropout"],
                num_buckets=cfg["relative_num_buckets"],
                max_distance=cfg["relative_max_distance"]
            )
            for _ in range(cfg["num_enc_layers"])
        ])

    def forward(self, src_ids, src_mask):
        x = self.embed(src_ids)
        m = (src_mask == 1).unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            x = layer(x, m)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, cfg, shared_embed):
        super().__init__()
        self.embed = shared_embed
        self.layers = nn.ModuleList([
            DecoderLayer(
                cfg["d_model"], cfg["n_heads"], cfg["d_ff"], cfg["dropout"],
                num_buckets=cfg["relative_num_buckets"],
                max_distance=cfg["relative_max_distance"]
            )
            for _ in range(cfg["num_dec_layers"])
        ])
        self.out_proj = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)

    def forward(self, tgt_ids, mem, tgt_mask, mem_mask):
        x = self.embed(tgt_ids)
        for layer in self.layers:
            x = layer(x, mem, tgt_mask, mem_mask)
        logits = self.out_proj(x)
        return logits


class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        shared_embed = InputEmbedding(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            pad_id=cfg["pad_id"],
            dropout=cfg["dropout"],
            max_len=cfg["max_len"]
        )
        self.encoder = TransformerEncoder(cfg, shared_embed)
        self.decoder = TransformerDecoder(cfg, shared_embed)
        # weight tying
        self.decoder.out_proj.weight = self.encoder.embed.embed_tokens.weight

        self.cfg = cfg
        self.pad_id = cfg["pad_id"]
        self.bos_id = cfg["bos_id"]
        self.eos_id = cfg["eos_id"]

    def _build_tgt_mask(self, tgt_ids):
        B, T = tgt_ids.size()
        pad_mask = (tgt_ids != self.pad_id).unsqueeze(1).unsqueeze(2)
        causal = torch.tril(torch.ones((T, T), device=tgt_ids.device)).bool()
        causal = causal.unsqueeze(0).unsqueeze(1)
        return pad_mask & causal

    def _build_mem_mask(self, src_mask):
        return (src_mask == 1).unsqueeze(1).unsqueeze(2)

    def forward(self, input_ids, attention_mask,
                decoder_input_ids, decoder_attention_mask, labels=None):
        mem = self.encoder(input_ids, attention_mask)
        tgt_mask = self._build_tgt_mask(decoder_input_ids)
        mem_mask = self._build_mem_mask(attention_mask)
        logits = self.decoder(decoder_input_ids, mem, tgt_mask, mem_mask)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = label_smoothed_nll_loss(
                logits,
                labels,
                ignore_index=-100,
                smoothing=CONFIG["label_smoothing"]
            )
        return out
