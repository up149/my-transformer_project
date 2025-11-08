# 从零实现 Transformer 模型 —— 实验运行说明

## 一、项目简介
本项目基于 PyTorch 从零实现了完整的 Transformer 模型，涵盖编码器—解码器结构、多头注意力机制、相对位置偏置（Relative Positional Bias）、前馈网络、层归一化以及标签平滑（Label Smoothing）等关键模块。  
在 IWSLT 英德平行语料上进行训练与评估，实验包含基线模型与消融实验，用于分析不同结构组件对模型性能的影响。

---

## 二、硬件环境要求
- **CPU**：Intel Xeon E5-2680 v4 @ 2.40GHz  
- **GPU**：NVIDIA GeForce RTX 2080 Ti / RTX 3090（显存 ≥ 24GB）  
- **CUDA 版本**：11.8 及以上  
- **Python 版本**：3.9 及以上  
- **PyTorch 版本**：2.0 及以上  
- **操作系统**：Linux（Ubuntu 20.04 LTS）

---

## 三、依赖环境安装
项目依赖库已在 `requirements.txt` 文件中列出。  
可通过以下命令配置运行环境：
```bash
conda create -n transformer python=3.9
conda activate transformer
pip install -r requirements.txt
```
---

## 四、可复现实验命令
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --seed 42
```
本项目的完整目录结构及复现脚本如上，执行 bash scripts/run.sh 可直接复现实验