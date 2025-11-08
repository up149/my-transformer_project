# src/utils.py
"""
utils.py
---------
通用工具函数模块，包括：
- 随机种子设定（seed_everything）；
- 标签平滑损失函数（label_smoothed_nll_loss）；
- sacreBLEU 自动评估函数（evaluate_bleu_sacrebleu）；
- 模型参数统计打印与保存函数（print_and_save_model_param_stats）。
该模块用于辅助训练、验证与模型分析。
"""

import os, random, csv, math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import sacrebleu
from config import CONFIG, device


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_smoothed_nll_loss(logits, labels, ignore_index=-100, smoothing=0.1):
    B, T, V = logits.size()
    logits_flat = logits.view(B * T, V)
    labels_flat = labels.view(B * T)
    mask = (labels_flat != ignore_index)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    logits_sel = logits_flat[mask]
    target_sel = labels_flat[mask]
    log_probs = F.log_softmax(logits_sel, dim=-1)
    nll = -log_probs[torch.arange(log_probs.size(0), device=logits.device), target_sel]
    smooth_loss = -log_probs.mean(dim=-1)
    loss = (1.0 - smoothing) * nll + smoothing * smooth_loss
    return loss.mean()


@torch.no_grad()
def evaluate_bleu_sacrebleu(model, loader, tokenizer, beam_for_eval=4):
    model.eval()
    hyps, refs = [], []
    for batch in tqdm(loader, desc="BLEU"):
        gen = model.beam_search_generate(
            batch["input_ids"],
            batch["attention_mask"],
            max_len=CONFIG["max_len"],
            beam_size=beam_for_eval
        )
        for seq in gen:
            hyps.append(tokenizer.decode_ids(seq.tolist()))
        for lab in batch["labels"]:
            gold_ids = []
            for tid in lab.tolist():
                if tid == -100 or tid == CONFIG["pad_id"]:
                    break
                gold_ids.append(tid)
            refs.append(tokenizer.decode_ids(gold_ids))
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    return bleu


def print_and_save_model_param_stats(model: torch.nn.Module, filepath: str):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = []
    lines.append("=== Model Parameter Statistics ===")
    lines.append(f"Total params      : {total_params:,}")
    lines.append(f"Trainable params  : {trainable_params:,}")
    lines.append(f"Non-trainable     : {total_params - trainable_params:,}")
    lines.append("")
    lines.append("By top-level submodule:")
    for name, module in model.named_children():
        sub_params = sum(p.numel() for p in module.parameters())
        sub_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        lines.append(f" - {name:<12} total={sub_params:,} trainable={sub_train:,}")
    text = "\n".join(lines)
    print(text)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"[Info] Parameter stats saved to {filepath}")
