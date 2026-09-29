"""
Model Registry — import và tạo model theo tên.

Sử dụng:
    from models import build_model
    model = build_model("gmlp", {"depth": 8, "dim": 192, "ffn_dim": 384, ...})

Thêm model mới:
    1. Tạo file models/your_model.py với function build_your_model(config)
    2. Import và đăng ký trong MODEL_REGISTRY bên dưới
"""

from models.gmlp import build_gmlp
# from models.vit import build_vit

# ============================================================
# MODEL REGISTRY
# ============================================================
# Khi thêm model mới, import build function và thêm vào đây

MODEL_REGISTRY = {
    "gmlp": build_gmlp,
    # "vit": build_vit,              # Giang sẽ thêm
    # "mlp_mixer": build_mlp_mixer,  # Thy sẽ thêm
    # "resmlp": build_resmlp,        # Minh sẽ thêm
    # "convmixer": build_convmixer,  # Tiên sẽ thêm
}


def build_model(model_name: str, config: dict):
    """
    Tạo model từ tên và config.

    Args:
        model_name: Tên model ("gmlp", "vit", "mlp_mixer", "resmlp", "convmixer")
        config: Dict chứa hyperparameters

    Returns:
        nn.Module
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {available}. "
            f"Hãy implement model và đăng ký trong models/__init__.py"
        )
    return MODEL_REGISTRY[model_name](config)


def list_available_models():
    """Liệt kê các model đã implement."""
    return list(MODEL_REGISTRY.keys())
