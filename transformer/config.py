# config.py
"""
config.py
----------
全局配置文件。包含数据路径、模型结构超参数、训练参数（学习率、batch size 等）、
以及 tokenizer 相关设置。用于在训练与推理时统一管理所有实验参数。
"""

import torch

CONFIG = {
    # ====== 平行语料，逐行一一对应 ======
    "train_src": "./iwslt_raw/en-de/train.clean.en",
    "train_tgt": "./iwslt_raw/en-de/train.clean.de",

    "dev_src":   "./iwslt_raw/en-de/dev.en",
    "dev_tgt":   "./iwslt_raw/en-de/dev.de",

    "test_src":  "./iwslt_raw/en-de/test.en",
    "test_tgt":  "./iwslt_raw/en-de/test.de",

    # SentencePiece 模型路径
    "spm_model": "./iwslt_raw/en-de/iwslt_bpe_16k.model",

    # Transformer 结构超参
    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "num_enc_layers": 6,
    "num_dec_layers": 6,
    "dropout": 0.1,

    # 最大序列长度
    "max_len": 160,

    # 训练超参
    "batch_size": 64,
    "epochs": 30,
    "base_lr": 3e-4,
    "warmup_steps": 4000,

    # 优化器选择
    "optimizer": "adamw",
    "weight_decay": 0.01,

    # label smoothing
    "label_smoothing": 0.05,

    # early stopping
    "patience": 10,

    # 相对位置偏置
    "relative_max_distance": 128,
    "relative_num_buckets": 32,

    # 日志
    "loss_log_csv": "loss_log.csv",
    "param_log_file": "model_param_stats.txt",

    # tokenizer 初始化后写回
    "pad_id": None,
    "bos_id": None,
    "eos_id": None,
    "unk_id": None,
    "vocab_size": None,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
