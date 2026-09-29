"""
CIFAR-10 Data Loader dùng chung cho cả 5 model.

Split: Train 45,000 / Val 5,000 / Test 10,000
Augmentation (train): RandomCrop(32, padding=4) + RandomHorizontalFlip
Normalization: CIFAR-10 mean/std

Sử dụng:
    from data.cifar10 import get_cifar10_loaders
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=128, data_dir="./dataset", num_workers=2
    )
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


# CIFAR-10 normalization statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_cifar10_transforms(train: bool = True):
    """
    Trả về transform cho train hoặc val/test.
    
    Train: RandomCrop + RandomHorizontalFlip + Normalize
    Val/Test: Chỉ Normalize
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


def get_cifar10_loaders(
    batch_size: int = 128,
    data_dir: str = "./dataset",
    num_workers: int = 2,
    val_size: int = 5000,
    train_subset: float = 1.0,
    seed: int = 42,
):
    """
    Tạo DataLoader cho CIFAR-10 với split train/val/test.

    Args:
        batch_size: Batch size
        data_dir: Thư mục lưu dataset
        num_workers: Số worker cho DataLoader
        val_size: Số sample cho validation set (lấy từ train)
        train_subset: Tỷ lệ dữ liệu train sử dụng (1.0 = 100%, 0.5 = 50%)
                      Dùng cho thí nghiệm giảm dữ liệu (câu hỏi nghiên cứu 2)
        seed: Random seed cho split

    Returns:
        (train_loader, val_loader, test_loader)
    """

    # --- Full training set (50,000 samples) ---
    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=get_cifar10_transforms(train=True)
    )

    # --- Val set dùng transform không augment ---
    full_val_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=get_cifar10_transforms(train=False)
    )

    # --- Split train/val ---
    total_train = len(full_train_dataset)  # 50,000
    indices = list(range(total_train))

    # Shuffle với seed cố định để reproducible
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    val_indices = indices[:val_size]          # Đầu tiên 5,000 → val
    train_indices = indices[val_size:]        # Còn lại 45,000 → train

    # --- Áp dụng train_subset nếu < 1.0 ---
    if train_subset < 1.0:
        n_keep = int(len(train_indices) * train_subset)
        train_indices = train_indices[:n_keep]
        print(f"[Data] Using {train_subset*100:.0f}% of training data: {n_keep} samples")

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    # --- Test set (10,000 samples) ---
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=get_cifar10_transforms(train=False)
    )

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[Data] CIFAR-10 loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    return train_loader, val_loader, test_loader
