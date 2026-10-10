import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config['dim']
        self.depth = config['depth']
        self.kernel_size = config['kernel_size']
        self.patch_size = config['patch_size']
        self.num_classes = config['num_classes']
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.dim, kernel_size=self.patch_size, stride=self.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(self.dim)
        )
        
        blocks = []
        for _ in range(self.depth):
            blocks.append(
                nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(self.dim, self.dim, self.kernel_size, groups=self.dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(self.dim)
                    )),
                    nn.Conv2d(self.dim, self.dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(self.dim)
                )
            )
        self.blocks = nn.Sequential(*blocks)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.dim, self.num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

def build_convmixer(config):
    return ConvMixer(config)

if __name__ == '__main__':
    config = {
        "depth": 8,
        "dim": 256,
        "kernel_size": 5,
        "patch_size": 4,
        "num_classes": 10,
    }
    model = build_convmixer(config)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("Forward pass OK")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
