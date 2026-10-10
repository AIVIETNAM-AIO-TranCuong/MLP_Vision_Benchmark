"""
Vision Transformer (ViT) — An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
(Dosovitskiy et al., ICLR 2021)

Kiến trúc ViT cơ bản:
    Image → PatchEmbed → + Positional Embedding (+ CLS token) → [Transformer Block × L] → LayerNorm → Linear (Classification Head)

Các thành phần chính:
    1. PatchEmbed: Cắt ảnh thành các patch, dùng Conv2d để nhúng thành các token vectors.
    2. Positional Embedding: Thêm thông tin vị trí vào từng token (vì Self-Attention không phân biệt thứ tự).
    3. Self-Attention (Multi-head Self Attention - MSA): Trích xuất thông tin tương tác giữa các patch.
    4. FFN (Feed-Forward Network): Xử lý cục bộ từng token.
    5. CLS Token: Một token học được (learnable) dùng để tổng hợp thông tin của toàn bộ ảnh cho classification.

Reference: https://arxiv.org/abs/2010.11929
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. PATCH EMBEDDING
class PatchEmbed(nn.Module):
    """
    Chuyển ảnh thành sequence of patch embeddings.
    """
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Tính attention score
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        # Nhân với V, gom các head lại và chiếu tuyến tính đầu ra
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_dim, drop=drop)

    def forward(self, x):
        # Áp dụng cơ chế residual connection (skip connection)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=4,
                 in_channels=3,
                 num_classes=10,
                 dim=192,
                 depth=6,
                 num_heads=3,
                 mlp_dim=384,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim

        # 1. Patch embedding
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches

        # 2. CLS Token & Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim = dim,
                num_heads = num_heads,
                mlp_dim = mlp_dim,
                qkv_bias = qkv_bias,
                drop = drop_rate,
                attn_drop = attn_drop_rate
            ) for _ in range(depth)
        ])

        # 4. Classification Head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        # Khởi tạo trọng số
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        # - Extract patch embeddings: (B, N, D)
        x = self.patch_embed(x)

        # - Nối CLS token vào patch sequence: (B, 1 + N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # - Thêm Positional embedding và Dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # - Đẩy qua các block Transformer
        for block in self.blocks:
            x = block(x)

        # - Lấy token CLS cuối cùng và Norm
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def build_vit(config: dict):
    return VisionTransformer(
        image_size=config.get("image_size", 32),
        patch_size=config.get("patch_size", 4),
        in_channels=config.get("in_channels", 3),
        num_classes=config.get("num_classes", 10),
        dim=config.get("dim", 192),
        depth=config.get("depth", 6),
        num_heads=config.get("num_heads", 3),
        mlp_dim=config.get("mlp_dim", 384),
        qkv_bias=config.get("qkv_bias", True),
        drop_rate=config.get("drop_rate", 0.),
        attn_drop_rate=config.get("attn_drop_rate", 0.)
    )
