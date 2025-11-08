# src/train.py
"""
train.py
---------
训练过程核心函数：
- WarmupInverseSqrtScheduler：实现 warmup + 反平方根衰减学习率调度；
- train_one_epoch：执行单轮训练、反向传播与梯度裁剪；
- evaluate_loss：在验证集上计算平均损失。
该模块封装了训练阶段的所有优化逻辑。
"""

import torch
from tqdm.auto import tqdm
from config import CONFIG
from .utils import label_smoothed_nll_loss


class WarmupInverseSqrtScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self.step_num = 0
        self.scale = base_lr / ((d_model ** -0.5) * (warmup_steps ** -0.5))

    def step(self):
        self.step_num += 1
        step = self.step_num
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup ** -1.5)
        lr = self.scale * (self.d_model ** -0.5) * min(arg1, arg2)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def train_one_epoch(model, loader, optimizer, scheduler, epoch_idx, total_loss_so_far):
    model.train()
    pbar = tqdm(enumerate(loader, start=1), total=len(loader))
    epoch_loss_sum = 0.0
    epoch_loss_count = 0

    for step, batch in pbar:
        out = model(**batch)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_now = scheduler.step()

        loss_val = loss.item()
        total_loss_so_far += loss_val
        epoch_loss_sum += loss_val
        epoch_loss_count += 1

        avg_loss_global = total_loss_so_far / ((epoch_idx - 1) * len(loader) + step)
        pbar.set_description(f"loss {avg_loss_global:.4f} | lr {lr_now:.2e}")

    epoch_avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
    return total_loss_so_far, epoch_avg_loss


@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    total, count = 0.0, 0
    for batch in tqdm(loader, desc="ValLoss"):
        out = model(**batch)
        total += out["loss"].item()
        count += 1
    return total / max(count, 1)
