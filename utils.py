"""
Utility functions dùng chung cho toàn bộ project.
Bao gồm: seed, đếm params, logging, metrics, learning rate scheduler.
"""

import os
import random
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import math


# ============================================================
# 1. SEED & REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    """Đặt seed cho tất cả nguồn random để reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. PARAMETER COUNTING
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """Đếm tổng số trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_detailed(model: nn.Module) -> dict:
    """Đếm params chi tiết theo từng module."""
    result = {}
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            result[name] = params
    return result


def format_params(n: int) -> str:
    """Format số params cho dễ đọc. Ví dụ: 1234567 -> '1.23M'"""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ============================================================
# 3. LEARNING RATE SCHEDULER - Cosine with Warmup
# ============================================================

class CosineWarmupScheduler:
    """
    Cosine annealing LR scheduler with linear warmup.
    
    LR schedule:
    - Warmup (0 → warmup_epochs): LR tăng tuyến tính từ 0 → base_lr
    - Cosine (warmup_epochs → total_epochs): LR giảm theo cosine từ base_lr → min_lr
    """

    def __init__(self, optimizer, base_lr, total_epochs, warmup_epochs=5, min_lr=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _get_lr(self):
        epoch = self.current_epoch
        if epoch <= self.warmup_epochs:
            # Linear warmup
            return self.base_lr * epoch / max(1, self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
# 4. METRICS & LOGGING
# ============================================================

class AverageMeter:
    """Theo dõi trung bình và giá trị hiện tại của một metric."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Tính accuracy cho top-k predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


# ============================================================
# 5. EXPERIMENT LOGGING
# ============================================================

class ExperimentLogger:
    """
    Logger để ghi lại kết quả thí nghiệm vào file JSON.
    Sử dụng:
        logger = ExperimentLogger("results/gmlp_small_seed42.json")
        logger.log_epoch(epoch=1, train_loss=0.5, train_acc=80.0, val_loss=0.6, val_acc=75.0, lr=1e-3)
        logger.save()
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self.data = json.load(f)
        else:
            self.data = {
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": {},
                "epochs": [],
                "best_val_acc": 0.0,
                "best_epoch": 0,
            }

    def log_config(self, config: dict):
        self.data["config"] = config

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "lr": round(lr, 8),
        }
        self.data["epochs"].append(entry)

        if val_acc > self.data["best_val_acc"]:
            self.data["best_val_acc"] = round(val_acc, 2)
            self.data["best_epoch"] = epoch

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.data["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Best Val Accuracy: {self.data['best_val_acc']:.2f}% (Epoch {self.data['best_epoch']})")
        print(f"{'='*60}\n")


# ============================================================
# 6. CHECKPOINT
# ============================================================

def save_checkpoint(model, optimizer, epoch, val_acc, filepath):
    """Lưu checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["val_acc"]
