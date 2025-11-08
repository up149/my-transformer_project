# main.py
"""
main.py
--------
项目主入口脚本。负责整体训练与评估流程的组织：
1. 加载配置与随机种子；
2. 构建分词器、数据集与 DataLoader；
3. 初始化模型、优化器与学习率调度器；
4. 训练主循环、验证与早停策略；
5. 记录损失与 BLEU 指标，保存最优模型并在测试集上评估。
"""

import csv
import torch
from torch.utils.data import DataLoader

from config import CONFIG, device
from src.tokenizer import build_tokenizer_and_update_config
from src.dataset import ParallelTextDataset, make_collate_fn
from src.models.transformer import Seq2SeqTransformer
from src.train import WarmupInverseSqrtScheduler, train_one_epoch, evaluate_loss
from src.utils import (
    seed_everything,
    evaluate_bleu_sacrebleu,
    print_and_save_model_param_stats,
)


def main():
    seed_everything(42)

    # 1) tokenizer
    tokenizer = build_tokenizer_and_update_config()

    # 2) dataset / dataloader
    train_set = ParallelTextDataset(CONFIG["train_src"], CONFIG["train_tgt"])
    dev_set   = ParallelTextDataset(CONFIG["dev_src"],   CONFIG["dev_tgt"])
    test_set  = ParallelTextDataset(CONFIG["test_src"],  CONFIG["test_tgt"])

    collate_fn = make_collate_fn(tokenizer)

    train_loader = DataLoader(train_set,
                              batch_size=CONFIG["batch_size"],
                              shuffle=True,
                              collate_fn=collate_fn,
                              drop_last=True)
    dev_loader = DataLoader(dev_set,
                            batch_size=CONFIG["batch_size"],
                            shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_set,
                             batch_size=CONFIG["batch_size"],
                             shuffle=False,
                             collate_fn=collate_fn)

    # 3) model
    model = Seq2SeqTransformer(CONFIG).to(device)

    # 参数统计
    print_and_save_model_param_stats(model, CONFIG["param_log_file"])

    # 4) optimizer & scheduler
    opt_name = CONFIG.get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["base_lr"],
            betas=(0.9, 0.999),
            eps=1e-9,
            weight_decay=CONFIG.get("weight_decay", 0.01)
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CONFIG["base_lr"],
            betas=(0.9, 0.999),
            eps=1e-9
        )

    scheduler = WarmupInverseSqrtScheduler(
        optimizer,
        d_model=CONFIG["d_model"],
        warmup_steps=CONFIG["warmup_steps"],
        base_lr=CONFIG["base_lr"]
    )

    # 训练状态
    total_loss_so_far = 0.0
    best_bleu = 0.0
    patience_left = CONFIG["patience"]
    best_state = None
    train_loss_per_epoch = []
    dev_loss_per_epoch = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n===== Epoch {epoch}/{CONFIG['epochs']} =====")
        total_loss_so_far, epoch_train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, epoch, total_loss_so_far
        )
        dev_loss = evaluate_loss(model, dev_loader)
        print(f"[Dev] loss = {dev_loss:.4f}")

        train_loss_per_epoch.append(epoch_train_loss)
        dev_loss_per_epoch.append(dev_loss)

        dev_bleu = evaluate_bleu_sacrebleu(
            model, dev_loader, tokenizer, beam_for_eval=1
        )
        print(f"[Dev] sacreBLEU (beam=1) = {dev_bleu:.2f}")

        if dev_bleu > best_bleu:
            best_bleu = dev_bleu
            patience_left = CONFIG["patience"]
            best_state = {
                "model_state": model.state_dict(),
                "config": CONFIG,
                "bleu": best_bleu,
                "epoch": epoch,
            }
            torch.save(best_state, "best_model.pt")
            print(f"... new best model saved (epoch {epoch}, BLEU {dev_bleu:.2f})")
        else:
            patience_left -= 1
            print(f"No BLEU improvement. patience_left={patience_left}")
            if patience_left <= 0:
                print("Early stop triggered.")
                break

    # 存 loss
    with open(CONFIG["loss_log_csv"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "dev_loss"])
        for i, (tr, dv) in enumerate(zip(train_loss_per_epoch, dev_loss_per_epoch), start=1):
            writer.writerow([i, tr, dv])

    # 最终 test
    print("\n===== Final Test Eval =====")
    ckpt = torch.load("best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss = evaluate_loss(model, test_loader)
    test_bleu = evaluate_bleu_sacrebleu(
        model, test_loader, tokenizer, beam_for_eval=4
    )
    print(f"[Test] loss = {test_loss:.4f}")
    print(f"[Test] sacreBLEU (beam=4) = {test_bleu:.2f}")


if __name__ == "__main__":
    main()
