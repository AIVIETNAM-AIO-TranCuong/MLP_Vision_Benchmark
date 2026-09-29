"""
Evaluation Script — Đánh giá model đã train trên test set.

Sử dụng:
    # Evaluate một checkpoint
    python evaluate.py --checkpoint results/checkpoints/gmlp_small_seed42_best.pt \
                       --model gmlp --size small

    # So sánh tất cả checkpoints trong thư mục
    python evaluate.py --results_dir results/
"""

import argparse
import os
import json
import glob

import torch
import torch.nn as nn

from data.cifar10 import get_cifar10_loaders
from configs.default import get_model_config, TRAIN_CONFIG
from models import build_model
from utils import accuracy, AverageMeter, count_parameters, format_params


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on CIFAR-10")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name")
    parser.add_argument("--size", type=str, default=None,
                        help="Model size")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory containing result JSON files — tổng hợp bảng")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)

    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    """Evaluate model, return loss and accuracy."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1 = accuracy(outputs, targets, topk=(1,))[0]

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1, images.size(0))

    return loss_meter.avg, acc_meter.avg


def evaluate_checkpoint(args):
    """Evaluate a single checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_cfg = get_model_config(args.model, args.size)
    model = build_model(args.model, model_cfg).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_acc={checkpoint['val_acc']:.2f}%)")

    # Data
    _, _, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

    n_params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"Model: {args.model}-{args.size} ({format_params(n_params)})")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*50}")


def summarize_results(results_dir):
    """Tổng hợp tất cả kết quả từ JSON files thành bảng."""
    json_files = glob.glob(os.path.join(results_dir, "*.json"))

    if not json_files:
        print(f"No result files found in {results_dir}")
        return

    results = []
    for jf in sorted(json_files):
        with open(jf) as f:
            data = json.load(f)

        cfg = data.get("config", {})
        results.append({
            "model": cfg.get("model", "?"),
            "size": cfg.get("size", "?"),
            "seed": cfg.get("seed", "?"),
            "params": cfg.get("n_params", 0),
            "best_val_acc": data.get("best_val_acc", 0),
            "test_acc": data.get("test_acc", "N/A"),
            "epochs": cfg.get("epochs", 0),
        })

    # Print table
    print(f"\n{'='*90}")
    print(f"{'Model':<12} {'Size':<8} {'Seed':<6} {'Params':<10} "
          f"{'Best Val%':<10} {'Test%':<10} {'Epochs':<8}")
    print(f"{'-'*90}")

    for r in sorted(results, key=lambda x: (x["model"], x["size"], x["seed"])):
        print(
            f"{r['model']:<12} {r['size']:<8} {r['seed']:<6} "
            f"{format_params(r['params']):<10} "
            f"{r['best_val_acc']:<10.2f} "
            f"{str(r['test_acc']):<10} "
            f"{r['epochs']:<8}"
        )

    print(f"{'='*90}")
    print(f"Total experiments: {len(results)}")


def main():
    args = parse_args()

    if args.results_dir:
        summarize_results(args.results_dir)
    elif args.checkpoint and args.model and args.size:
        evaluate_checkpoint(args)
    else:
        print("Usage:")
        print("  Evaluate checkpoint:")
        print("    python evaluate.py --checkpoint PATH --model MODEL --size SIZE")
        print("  Summarize all results:")
        print("    python evaluate.py --results_dir results/")


if __name__ == "__main__":
    main()
