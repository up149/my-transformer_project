# src/tokenizer.py
"""
tokenizer.py
-------------
封装 SentencePiece 分词器，提供文本与 ID 之间的双向转换接口。
实现：
- encode_one / batch_encode：文本转为带 <pad>、<bos>、<eos> 的张量；
- decode_ids：将生成的 token ID 序列还原成文本；
- build_tokenizer_and_update_config：初始化分词器并回写配置。
"""

from typing import List, Tuple
import torch
import sentencepiece as spm
from config import CONFIG, device


class SentencePieceTokenizer:
    def __init__(self, spm_model_path: str, max_len: int):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)

        self.max_len = max_len

        base_vocab_size = self.sp.get_piece_size()

        # spm 自带 unk
        unk_id = self.sp.unk_id()
        if unk_id < 0:
            unk_id = base_vocab_size
            base_vocab_size += 1

        pad_id = base_vocab_size; base_vocab_size += 1
        bos_id = base_vocab_size; base_vocab_size += 1
        eos_id = base_vocab_size; base_vocab_size += 1

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.vocab_size = base_vocab_size

    def encode_one(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def batch_encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        all_ids = []
        for s in texts:
            ids = self.encode_one(s)
            ids = ids[: self.max_len - 1] + [self.eos_id]
            if len(ids) < self.max_len:
                ids += [self.pad_id] * (self.max_len - len(ids))
            all_ids.append(ids)

        ids_tensor = torch.tensor(all_ids, dtype=torch.long)
        attn_mask = (ids_tensor != self.pad_id).long()
        return ids_tensor, attn_mask

    def decode_ids(self, ids: List[int]) -> str:
        out_ids = []
        sp_vocab = self.sp.get_piece_size()
        for t in ids:
            if t == self.eos_id: break
            if t == self.pad_id: break
            if t == self.bos_id: continue
            if t == self.unk_id:
                out_ids.append(self.unk_id)
            elif t < sp_vocab:
                out_ids.append(t)
            else:
                continue
        if len(out_ids) == 0:
            return ""
        return self.sp.decode(out_ids)


def build_tokenizer_and_update_config():
    tok = SentencePieceTokenizer(CONFIG["spm_model"], CONFIG["max_len"])
    CONFIG["pad_id"] = tok.pad_id
    CONFIG["bos_id"] = tok.bos_id
    CONFIG["eos_id"] = tok.eos_id
    CONFIG["unk_id"] = tok.unk_id
    CONFIG["vocab_size"] = tok.vocab_size

    print("[Tokenizer] SentencePiece loaded")
    print(" vocab_size =", tok.vocab_size)
    return tok
