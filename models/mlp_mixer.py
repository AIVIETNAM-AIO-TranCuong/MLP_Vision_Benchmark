import torch
import torch.nn as nn

class MlpBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MixerBlock(nn.Module):
    def __init__(self, num_patches, dim, token_dim, channel_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = MlpBlock(num_patches, token_dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = MlpBlock(dim, channel_dim)

    def forward(self, x):
        # Token mixing
        res = x
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.token_mix(x)
        x = x.transpose(1, 2)
        x = x + res
        
        # Channel mixing
        res = x
        x = self.norm2(x)
        x = self.channel_mix(x)
        x = x + res
        return x

class MLPMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config['patch_size']
        self.dim = config['dim']
        self.depth = config['depth']
        self.num_classes = config['num_classes']
        self.token_dim = config.get('token_dim', config['dim'] // 2)
        self.channel_dim = config.get('channel_dim', config['dim'] * 4)
        
        img_size = 32
        self.num_patches = (img_size // self.patch_size) ** 2
        
        self.stem = nn.Conv2d(3, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        self.blocks = nn.ModuleList([
            MixerBlock(self.num_patches, self.dim, self.token_dim, self.channel_dim)
            for _ in range(self.depth)
        ])
        
        self.norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.num_classes)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.stem(x)  # (B, dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, dim)
        x = self.head(x)   # (B, num_classes)
        return x

def build_mlp_mixer(config):
    return MLPMixer(config)

if __name__ == '__main__':
    config = {
        "depth": 8,
        "dim": 192,
        "token_dim": 96,
        "channel_dim": 384,
        "patch_size": 4,
        "num_classes": 10,
    }
    model = build_mlp_mixer(config)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("Forward pass OK")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
