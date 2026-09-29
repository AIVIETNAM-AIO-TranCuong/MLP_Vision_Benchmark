"""
Training Script dùng chung cho tất cả 5 model.

Sử dụng:
    # Train gMLP small với seed 42
    python train.py --model gmlp --size small --seed 42

    # Train gMLP medium, 50 epochs, batch 256
    python train.py --model gmlp --size medium --seed 42 --epochs 50 --batch_size 256

    # Train với data subset (50% dữ liệu) — cho câu hỏi nghiên cứu 2
    python train.py --model gmlp --size small --seed 42 --train_subset 0.5

    # Chạy trên Google Colab — giảm num_workers
    python train.py --model gmlp --size small --seed 42 --num_workers 0
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from data.cifar10 import get_cifar10_loaders
from configs.default import get_model_config, TRAIN_CONFIG
from models import build_model
from utils import (
    set_seed, count_parameters, format_params,
    CosineWarmupScheduler, AverageMeter, accuracy,
    ExperimentLogger, save_checkpoint, load_checkpoint
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model on CIFAR-10")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        choices=["gmlp", "vit", "mlp_mixer", "resmlp", "convmixer"],
                        help="Model name")
    parser.add_argument("--size", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Model size")

    # Training
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_subset", type=float, default=1.0,
                        help="Fraction of training data to use (for RQ2)")

    # System
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detect if not specified)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from best checkpoint if available")

    # Label smoothing
    parser.add_argument("--label_smoothing", type=float, default=None,
                        help="Label smoothing factor")

    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (avg_loss, avg_accuracy)."""
    model.train()
    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Acc")

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1, images.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on val/test set. Returns (avg_loss, avg_accuracy)."""
    model.eval()
    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Acc")

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1, images.size(0))

    return loss_meter.avg, acc_meter.avg


def main():
    args = parse_args()

    # --- Setup ---
    set_seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")

    # --- Config ---
    train_cfg = TRAIN_CONFIG.copy()
    model_cfg = get_model_config(args.model, args.size)

    # Override from command line
    epochs = args.epochs or train_cfg["epochs"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    lr = args.lr or train_cfg["lr"]
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else train_cfg["label_smoothing"]

    # --- Data ---
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        num_workers=args.num_workers,
        train_subset=args.train_subset,
        seed=args.seed,
    )

    # --- Model ---
    model = build_model(args.model, model_cfg)
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"[Model] {args.model}-{args.size}: {format_params(n_params)} parameters ({n_params:,})")

    # --- Loss, Optimizer, Scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg["weight_decay"],
        betas=train_cfg["betas"],
    )

    scheduler = CosineWarmupScheduler(
        optimizer,
        base_lr=lr,
        total_epochs=epochs,
        warmup_epochs=train_cfg["warmup_epochs"],
        min_lr=train_cfg["min_lr"],
    )

    # --- Logger ---
    exp_name = f"{args.model}_{args.size}_seed{args.seed}"
    if args.train_subset < 1.0:
        exp_name += f"_data{int(args.train_subset*100)}pct"
    log_path = os.path.join(args.output_dir, f"{exp_name}.json")

    logger = ExperimentLogger(log_path)
    logger.log_config({
        "model": args.model,
        "size": args.size,
        "model_config": model_cfg,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": args.seed,
        "train_subset": args.train_subset,
        "label_smoothing": label_smoothing,
        "n_params": n_params,
        "device": str(device),
    })

    # --- Training Loop ---
    print(f"\n{'='*60}")
    print(f"Training {args.model}-{args.size} for {epochs} epochs")
    print(f"{'='*60}\n")

    best_val_acc = 0.0
    start_epoch = 0
    ckpt_path = os.path.join(args.output_dir, "checkpoints", f"{exp_name}_best.pt")

    if args.resume and os.path.exists(ckpt_path):
        start_epoch, best_val_acc = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resuming from epoch {start_epoch} with Best Val Acc: {best_val_acc:.2f}%")
        # Advance scheduler state
        for _ in range(start_epoch):
            scheduler.step()

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        current_lr = scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)

        # Print progress
        print(
            f"Epoch [{epoch:3d}/{epochs}] "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%  |  "
            f"LR: {current_lr:.6f}  |  "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, ckpt_path)
            print(f"  -> New best! Saved checkpoint ({val_acc:.2f}%)")

    # --- Final Evaluation on Test Set ---
    print(f"\n{'='*60}")
    print("Evaluating best model on TEST set (chỉ dùng 1 lần cuối!)")
    print(f"{'='*60}")

    # Load best checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.2f}%")

    # Save final results
    logger.data["test_acc"] = round(test_acc, 2)
    logger.data["test_loss"] = round(test_loss, 4)
    logger.save()
    logger.print_summary()

    print(f"Results saved to: {log_path}")
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
