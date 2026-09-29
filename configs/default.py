"""
Configurations cho tất cả model và thí nghiệm.

Mỗi model có 3 mức kích thước: small (~1M), medium (~3M), large (~8M).
Training config dùng chung cho cả 5 model để đảm bảo fair comparison.

Sử dụng:
    from configs.default import get_model_config, TRAIN_CONFIG
    cfg = get_model_config("gmlp", "small")
"""


# ============================================================
# TRAINING CONFIG — DÙNG CHUNG CHO CẢ 5 MODEL
# ============================================================

TRAIN_CONFIG = {
    # Optimizer
    "optimizer": "adamw",
    "lr": 1e-3,
    "weight_decay": 0.05,
    "betas": (0.9, 0.999),

    # Scheduler
    "scheduler": "cosine_warmup",
    "warmup_epochs": 5,
    "min_lr": 1e-6,

    # Training
    "epochs": 100,
    "batch_size": 128,

    # Augmentation
    "random_crop": True,
    "random_flip": True,
    "label_smoothing": 0.1,

    # CIFAR-10
    "num_classes": 10,
    "image_size": 32,
    "patch_size": 4,       # 32/4 = 8×8 = 64 patches
    "in_channels": 3,

    # Misc
    "num_workers": 2,
    "val_size": 5000,
}


# ============================================================
# MODEL CONFIGS — 3 MỨC KÍCH THƯỚC MỖI MODEL
# ============================================================
# Ghi chú: Số params ước lượng, cần verify bằng count_parameters()
# Target: small ~1M, medium ~3M, large ~8M

MODEL_CONFIGS = {

    # --------------------------------------------------------
    # gMLP (Cường)
    # --------------------------------------------------------
    "gmlp": {
        "small": {
            "depth": 8,
            "dim": 192,
            "ffn_dim": 384,
            "patch_size": 4,
            "num_classes": 10,
        },
        "medium": {
            "depth": 16,
            "dim": 256,
            "ffn_dim": 512,
            "patch_size": 4,
            "num_classes": 10,
        },
        "large": {
            "depth": 24,
            "dim": 384,
            "ffn_dim": 768,
            "patch_size": 4,
            "num_classes": 10,
        },
    },

    # --------------------------------------------------------
    # ViT (Giang) — placeholder, Giang sẽ điền
    # --------------------------------------------------------
    "vit": {
        "small": {
            "depth": 6,
            "dim": 192,
            "num_heads": 3,
            "mlp_dim": 384,
            "patch_size": 4,
            "num_classes": 10,
        },
        "medium": {
            "depth": 9,
            "dim": 288,
            "num_heads": 6,
            "mlp_dim": 576,
            "patch_size": 4,
            "num_classes": 10,
        },
        "large": {
            "depth": 12,
            "dim": 384,
            "num_heads": 6,
            "mlp_dim": 768,
            "patch_size": 4,
            "num_classes": 10,
        },
    },

    # --------------------------------------------------------
    # MLP-Mixer (Thy) — placeholder, Thy sẽ điền
    # --------------------------------------------------------
    "mlp_mixer": {
        "small": {
            "depth": 8,
            "dim": 192,
            "token_dim": 96,
            "channel_dim": 384,
            "patch_size": 4,
            "num_classes": 10,
        },
        "medium": {
            "depth": 12,
            "dim": 288,
            "token_dim": 144,
            "channel_dim": 576,
            "patch_size": 4,
            "num_classes": 10,
        },
        "large": {
            "depth": 16,
            "dim": 384,
            "token_dim": 192,
            "channel_dim": 768,
            "patch_size": 4,
            "num_classes": 10,
        },
    },

    # --------------------------------------------------------
    # ResMLP (Minh) — placeholder, Minh sẽ điền
    # --------------------------------------------------------
    "resmlp": {
        "small": {
            "depth": 8,
            "dim": 192,
            "ffn_dim": 384,
            "patch_size": 4,
            "num_classes": 10,
        },
        "medium": {
            "depth": 16,
            "dim": 256,
            "ffn_dim": 512,
            "patch_size": 4,
            "num_classes": 10,
        },
        "large": {
            "depth": 24,
            "dim": 384,
            "ffn_dim": 768,
            "patch_size": 4,
            "num_classes": 10,
        },
    },

    # --------------------------------------------------------
    # ConvMixer (Tiên) — placeholder, Tiên sẽ điền
    # --------------------------------------------------------
    "convmixer": {
        "small": {
            "depth": 8,
            "dim": 256,
            "kernel_size": 5,
            "patch_size": 4,
            "num_classes": 10,
        },
        "medium": {
            "depth": 16,
            "dim": 256,
            "kernel_size": 7,
            "patch_size": 4,
            "num_classes": 10,
        },
        "large": {
            "depth": 20,
            "dim": 512,
            "kernel_size": 9,
            "patch_size": 4,
            "num_classes": 10,
        },
    },
}


def get_model_config(model_name: str, size: str) -> dict:
    """
    Lấy config cho model cụ thể.

    Args:
        model_name: "gmlp", "vit", "mlp_mixer", "resmlp", "convmixer"
        size: "small", "medium", "large"

    Returns:
        dict chứa hyperparameters của model
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")
    if size not in MODEL_CONFIGS[model_name]:
        raise ValueError(f"Unknown size: {size}. Choose from {list(MODEL_CONFIGS[model_name].keys())}")
    return MODEL_CONFIGS[model_name][size].copy()
