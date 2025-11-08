# src/dataset.py
"""
dataset.py
-----------
定义平行语料数据集与批处理函数：
- ParallelTextDataset：按行读取英文与德文平行语料；
- make_collate_fn：在 DataLoader 中对 batch 进行 padding、添加 <bos>，
  生成输入、标签与掩码张量。
该模块实现了训练所需的输入预处理与动态批对齐。
"""

from torch.utils.data import Dataset
from typing import Dict
import torch
from config import CONFIG, device
from .tokenizer import SentencePieceTokenizer


class ParallelTextDataset(Dataset):
    def __init__(self, src_file: str, tgt_file: str):
        with open(src_file, "r", encoding="utf-8") as f_src:
            src_lines = [l.strip() for l in f_src.readlines()]
        with open(tgt_file, "r", encoding="utf-8") as f_tgt:
            tgt_lines = [l.strip() for l in f_tgt.readlines()]
        assert len(src_lines) == len(tgt_lines)
        self.src_list = src_lines
        self.tgt_list = tgt_lines

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):
        return {"src": self.src_list[idx], "tgt": self.tgt_list[idx]}


def make_collate_fn(tokenizer: SentencePieceTokenizer):
    pad_id = CONFIG["pad_id"]
    bos_id = CONFIG["bos_id"]

    def collate_fn(batch):
        src_texts = [b["src"] for b in batch]
        tgt_texts = [b["tgt"] for b in batch]

        src_ids, src_mask = tokenizer.batch_encode(src_texts)
        tgt_ids, _ = tokenizer.batch_encode(tgt_texts)

        B, L = tgt_ids.size()
        decoder_in = torch.full((B, L), pad_id, dtype=torch.long)
        labels     = torch.full((B, L), -100, dtype=torch.long)

        for i in range(B):
            full_seq = tgt_ids[i].tolist()
            try:
                cutoff = full_seq.index(pad_id)
            except ValueError:
                cutoff = L
            valid = full_seq[:cutoff]  # 含 <eos>

            di = [bos_id] + valid[:-1]
            if len(di) < L:
                di += [pad_id] * (L - len(di))
            decoder_in[i] = torch.tensor(di, dtype=torch.long)

            for j in range(L):
                labels[i, j] = valid[j] if j < len(valid) else -100

        return {
            "input_ids": src_ids.to(device),
            "attention_mask": src_mask.to(device),
            "decoder_input_ids": decoder_in.to(device),
            "decoder_attention_mask": (decoder_in != pad_id).long().to(device),
            "labels": labels.to(device),
        }

    return collate_fn
