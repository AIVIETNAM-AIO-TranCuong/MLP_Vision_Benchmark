"""
gMLP — Pay Attention to MLPs (Liu et al., NeurIPS 2021)

Kiến trúc gMLP cho Image Classification trên CIFAR-10.
Sử dụng Spatial Gating Unit (SGU) thay thế self-attention.

Kiến trúc:
    Image → PatchEmbed → [gMLP Block × L] → LayerNorm → Mean Pool → Linear → Classes

gMLP Block:
    x → LayerNorm → Linear(d → ffn_dim) → GeLU → SGU → Linear(ffn_dim/2 → d) → + residual

SGU (Spatial Gating Unit):
    Z → split(Z₁, Z₂) along channel
    Z₂ → LayerNorm → Linear_spatial(n_patches × n_patches) → + bias
    output = Z₁ ⊙ f(Z₂)

Đặc điểm:
    - Static spatial weights (không phụ thuộc input, khác self-attention)
    - Init W ≈ 0, b = 1 → ban đầu hoạt động như FFN bình thường
    - Không cần position embedding (spatial projection học vị trí ngầm)

Reference: https://arxiv.org/abs/2105.08050
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ============================================================
# 1. PATCH EMBEDDING
# ============================================================

class PatchEmbed(nn.Module):
    """
    Chuyển ảnh thành sequence of patch embeddings.
    
    Input:  (B, C, H, W)    — ví dụ (B, 3, 32, 32)
    Output: (B, N, D)       — ví dụ (B, 64, 192) với patch_size=4
    
    Trong đó N = (H/P) × (W/P) = số patches.
    """

    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 64 cho 32/4
        self.embed_dim = embed_dim

        # Conv2d với kernel=stride=patch_size ≡ linear projection trên mỗi patch
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) → (B, D, H/P, W/P)
        x = self.proj(x)
        # → (B, D, N) → (B, N, D)
        x = x.flatten(2).transpose(1, 2)
        return x


# ============================================================
# 2. SPATIAL GATING UNIT (SGU) — Core of gMLP
# ============================================================

class SpatialGatingUnit(nn.Module):
    """
    Spatial Gating Unit (SGU) — cơ chế trộn spatial thay thế self-attention.
    
    Hoạt động:
    1. Split input Z thành 2 nửa: Z₁ (content), Z₂ (gate) theo chiều channel
    2. Z₂ → LayerNorm → Linear spatial projection (n × n)
    3. Output = Z₁ ⊙ f(Z₂)   (element-wise multiplication)
    
    Khởi tạo đặc biệt:
    - W (spatial weight) ≈ 0  → f(Z₂) ≈ bias ≈ 1
    - b (bias) = 1
    → Ban đầu SGU(Z) ≈ Z₁ ⊙ 1 = Z₁ → block hoạt động như FFN thông thường
    → Dần dần học spatial interactions trong quá trình training
    
    Args:
        dim: Chiều input (= ffn_dim, sẽ bị split thành dim/2)
        num_patches: Số patches (N = 64 cho CIFAR-10 patch 4×4)
    """

    def __init__(self, dim, num_patches):
        super().__init__()
        self.norm = nn.LayerNorm(dim // 2)

        # Spatial projection: (num_patches, num_patches)
        # Shared across channels — chỉ 1 matrix W cho tất cả channels
        self.spatial_proj = nn.Linear(num_patches, num_patches)

        # === KHỞI TẠO ĐẶC BIỆT ===
        # W ≈ 0: ban đầu không có spatial interaction
        nn.init.constant_(self.spatial_proj.weight, 0.0)
        # b = 1: f(Z₂) ≈ 1 → SGU(Z) ≈ Z₁ (identity-like)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        """
        Args:
            x: (B, N, ffn_dim)
        Returns:
            (B, N, ffn_dim // 2)
        """
        # Split thành 2 nửa along channel dimension
        # z1: content branch, z2: gate branch
        z1, z2 = x.chunk(2, dim=-1)  # Mỗi nửa: (B, N, ffn_dim/2)

        # Gate branch: normalize → spatial projection
        z2 = self.norm(z2)
        # Transpose để áp spatial projection theo chiều patches:
        # (B, N, D/2) → (B, D/2, N) → Linear(N→N) → (B, D/2, N) → (B, N, D/2)
        z2 = z2.transpose(1, 2)           # (B, D/2, N)
        z2 = self.spatial_proj(z2)         # (B, D/2, N) — spatial mixing
        z2 = z2.transpose(1, 2)           # (B, N, D/2)

        # Element-wise gating: content × gate
        return z1 * z2


# ============================================================
# 3. gMLP BLOCK
# ============================================================

class gMLPBlock(nn.Module):
    """
    Một block của gMLP.
    
    Structure:
        x → LayerNorm → Linear(d → ffn_dim) → GeLU → SGU(ffn_dim → ffn_dim/2) 
          → Linear(ffn_dim/2 → d) → + residual
    
    Khác Transformer: KHÔNG CÓ self-attention, thay bằng SGU.
    Khác MLP-Mixer: Dùng gating thay vì 2-layer MLP cho spatial mixing.
    
    Args:
        dim: Model dimension (embedding dim)
        ffn_dim: Hidden dimension trong FFN (thường = 2 × dim hoặc 4 × dim)
        num_patches: Số patches
    """

    def __init__(self, dim, ffn_dim, num_patches):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # Channel projection UP: d → ffn_dim
        self.channel_proj_up = nn.Linear(dim, ffn_dim)
        self.act = nn.GELU()

        # Spatial Gating Unit: ffn_dim → ffn_dim/2
        self.sgu = SpatialGatingUnit(ffn_dim, num_patches)

        # Channel projection DOWN: ffn_dim/2 → d
        self.channel_proj_down = nn.Linear(ffn_dim // 2, dim)

    def forward(self, x):
        """
        Args:
            x: (B, N, D) — patch embeddings
        Returns:
            (B, N, D) — updated patch embeddings
        """
        residual = x

        # Pre-norm
        x = self.norm(x)

        # Channel projection up + activation
        x = self.channel_proj_up(x)     # (B, N, ffn_dim)
        x = self.act(x)                 # GeLU activation

        # Spatial Gating Unit — đây là nơi spatial mixing xảy ra
        x = self.sgu(x)                # (B, N, ffn_dim/2)

        # Channel projection down
        x = self.channel_proj_down(x)   # (B, N, D)

        # Residual connection
        return x + residual


# ============================================================
# 4. gMLP MODEL (Full Architecture)
# ============================================================

class gMLP(nn.Module):
    """
    gMLP — Full model for image classification.
    
    Architecture:
        Image → PatchEmbed → [gMLPBlock × depth] → LayerNorm → MeanPool → Linear → logits
    
    Args:
        image_size: Kích thước ảnh input (32 cho CIFAR-10)
        patch_size: Kích thước mỗi patch (4 → 64 patches)
        in_channels: Số kênh input (3 cho RGB)
        num_classes: Số classes output (10 cho CIFAR-10)
        dim: Embedding dimension
        ffn_dim: Hidden dimension trong FFN (thường 2× hoặc 4× dim)
        depth: Số gMLP blocks
    
    Example:
        model = gMLP(image_size=32, patch_size=4, num_classes=10,
                     dim=192, ffn_dim=384, depth=8)
        x = torch.randn(2, 3, 32, 32)
        logits = model(x)  # (2, 10)
    """

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        dim=192,
        ffn_dim=384,
        depth=8,
    ):
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
        )

        # 2. Stack of gMLP Blocks
        self.blocks = nn.Sequential(*[
            gMLPBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_patches=self.num_patches,
            )
            for _ in range(depth)
        ])

        # 3. Final LayerNorm
        self.norm = nn.LayerNorm(dim)

        # 4. Classification Head
        self.head = nn.Linear(dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Khởi tạo weights theo chuẩn."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # SGU spatial_proj đã init riêng, skip nếu bias = 1
                if hasattr(m, 'bias') and m.bias is not None:
                    if m.bias.data.mean().item() > 0.9:
                        continue  # Đã init đặc biệt cho SGU
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 32, 32) — batch of CIFAR-10 images
        Returns:
            (B, num_classes) — classification logits
        """
        # Patch embedding: (B, 3, 32, 32) → (B, 64, dim)
        x = self.patch_embed(x)

        # gMLP blocks: (B, 64, dim) → (B, 64, dim)
        x = self.blocks(x)

        # Final norm
        x = self.norm(x)

        # Global average pooling: (B, 64, dim) → (B, dim)
        x = x.mean(dim=1)

        # Classification: (B, dim) → (B, num_classes)
        x = self.head(x)

        return x


# ============================================================
# 5. FACTORY FUNCTION
# ============================================================

def build_gmlp(config: dict) -> gMLP:
    """
    Tạo gMLP model từ config dict.
    
    Args:
        config: dict với keys: depth, dim, ffn_dim, patch_size, num_classes
    
    Returns:
        gMLP model
    """
    return gMLP(
        image_size=32,
        patch_size=config.get("patch_size", 4),
        in_channels=3,
        num_classes=config.get("num_classes", 10),
        dim=config["dim"],
        ffn_dim=config["ffn_dim"],
        depth=config["depth"],
    )


# ============================================================
# 6. QUICK TEST
# ============================================================

if __name__ == "__main__":
    from configs.default import MODEL_CONFIGS

    print("=" * 60)
    print("gMLP Architecture Verification")
    print("=" * 60)

    for size_name, cfg in MODEL_CONFIGS["gmlp"].items():
        model = build_gmlp(cfg)
        n_params = sum(p.numel() for p in model.parameters())

        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        y = model(x)

        print(f"\n[{size_name.upper()}]")
        print(f"  Config: depth={cfg['depth']}, dim={cfg['dim']}, ffn_dim={cfg['ffn_dim']}")
        print(f"  Params: {n_params:,} ({n_params/1e6:.2f}M)")
        print(f"  Input:  {tuple(x.shape)}")
        print(f"  Output: {tuple(y.shape)}")
        assert y.shape == (2, 10), f"Expected (2, 10), got {y.shape}"
        print(f"  [OK] Forward pass OK")

    print(f"\n{'='*60}")
    print("All tests passed!")
    print(f"{'='*60}")
