"""ResMLP for CIFAR-10 (Minh).

Image -> patch embedding -> residual MLP blocks -> affine -> mean -> logits.
Architecture reference: https://github.com/facebookresearch/deit/blob/main/resmlp_models.py
The CIFAR-10 presets use custom widths/depths, no dropout or stochastic depth,
and an explicit LayerScale initialization shared across sizes.
"""

import torch
from torch import nn


class Affine(nn.Module):
    """Learned scale and bias per channel; no normalization statistics."""

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.scale + self.bias


class ResMLPBlock(nn.Module):
    """Mix patches with a shared linear map, then mix channels with an MLP."""

    def __init__(self, dim, ffn_dim, num_patches, init_scale):
        super().__init__()
        self.token_affine = Affine(dim)
        self.token_mixer = nn.Linear(num_patches, num_patches)
        self.channel_affine = Affine(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim)
        )
        self.token_scale = nn.Parameter(torch.full((dim,), init_scale))
        self.channel_scale = nn.Parameter(torch.full((dim,), init_scale))

    def forward(self, x):
        # (B, N, D) -> (B, D, N): the same N x N map serves every channel.
        mixed = self.token_mixer(self.token_affine(x).transpose(1, 2))
        x = x + self.token_scale * mixed.transpose(1, 2)
        return x + self.channel_scale * self.channel_mlp(self.channel_affine(x))


class ResMLP(nn.Module):
    """Return unnormalized class logits for a fixed square image size."""

    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 num_classes=10, dim=120, ffn_dim=480, depth=8,
                 init_scale=1e-4):
        super().__init__()
        for name, value in dict(image_size=image_size, patch_size=patch_size,
                                in_channels=in_channels, num_classes=num_classes,
                                dim=dim, ffn_dim=ffn_dim, depth=depth).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        self.blocks = nn.Sequential(*[
            ResMLPBlock(dim, ffn_dim, self.num_patches, init_scale)
            for _ in range(depth)
        ])
        self.affine = Affine(dim)
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        expected = (self.in_channels, self.image_size, self.image_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected (B, {expected}), got {tuple(x.shape)}")
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = self.affine(self.blocks(x)).mean(dim=1)
        return self.head(x)


def build_resmlp(config: dict) -> ResMLP:
    """Factory compatible with the shared training/evaluation scripts."""
    return ResMLP(
        image_size=config.get("image_size", 32),
        in_channels=config.get("in_channels", 3),
        patch_size=config.get("patch_size", 4),
        num_classes=config.get("num_classes", 10),
        dim=config["dim"], ffn_dim=config["ffn_dim"], depth=config["depth"],
        init_scale=config.get("init_scale", 1e-4),
    )


if __name__ == "__main__":
    from configs.default import get_model_config
    from models import build_model

    torch.manual_seed(42)
    for size, target in (("small", 1_000_000), ("medium", 3_000_000),
                         ("large", 8_000_000)):
        model = build_model("resmlp", get_model_config("resmlp", size))
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert abs(params - target) / target < 0.05, (size, params)
        model.eval()
        with torch.no_grad():
            for batch_size in (1, 2):
                output = model(torch.randn(batch_size, 3, 32, 32))
                assert output.shape == (batch_size, 10)
                assert torch.isfinite(output).all()
        # Check gradient flow without an optimizer step or dataset download.
        model.train()
        loss = nn.functional.cross_entropy(
            model(torch.randn(2, 3, 32, 32)), torch.tensor([0, 9])
        )
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Missing gradient: {name}"
            assert torch.isfinite(param.grad).all(), f"Invalid gradient: {name}"
        for block in model.blocks:
            assert block.token_mixer.weight.grad.abs().sum() > 0
        print(f"{size}: {params:,} params | Forward pass OK | Backward pass OK")
    print("All ResMLP checks passed (synthetic inputs only).")
