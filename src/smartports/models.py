import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# SimpleCNN

class SimpleCNN(nn.Module):
    """
    Lightweight CNN trained from scratch for binary classification.

    Architecture:
        3 convolutional blocks (Conv → BN → ReLU → MaxPool)
        followed by a fully connected classifier head.

    Args:
        dropout: Dropout probability before the final linear layer.
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224 → 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 112x112 → 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 56x56 → 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ConvNeXt-Tiny

class ConvNeXtBlock(nn.Module):
    """
    Single ConvNeXt block:
        Depthwise Conv 7x7 → LayerNorm → Linear 4x → GELU → Linear 1x
    with a residual connection and layer scale.
    """

    def __init__(self, dim: int, layer_scale: float = 1e-6):
        super().__init__()
        self.dwconv  = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # Different filter for features
        self.norm    = nn.LayerNorm(dim, eps=1e-6) 
        self.pwconv1 = nn.Linear(dim, 4 * dim) # Channel expansion
        self.act     = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim) # Channel reduction
        self.gamma   = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale > 0 else None

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)       # (N, C, H, W) → (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)       # (N, H, W, C) → (N, C, H, W)
        return residual + x

class _LayerNorm2d(nn.LayerNorm):
    """LayerNorm for (N, C, H, W) tensors."""
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x

class MiniConvNeXt(nn.Module):
    """
    Lightweight ConvNeXt-inspired architecture trained from scratch.

    Follows the ConvNeXt design (Liu et al., 2022) with reduced depth
    and width suitable for small datasets (~300 images).

    Stage channels : [40, 80, 160, 320]
    Blocks per stage: [2,  2,   6,   2]

    Args:
        dropout: Dropout probability in the classifier head.
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        dims   = [40, 80, 160, 320]
        depths = [2, 2, 6, 2]

        # Stem: patchify 4x4
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6) if False else _LayerNorm2d(dims[0]),
        )

        # Stages + downsampling
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage) # Feature extraction
            if i < 3:
                self.downsamples.append(nn.Sequential(
                    _LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )) # Downsampling

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dims[-1], eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(dims[-1], 1),
        )

    def forward(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < 3:
                x = self.downsamples[i](x)
        x = self.classifier(x)
        return x


# ResNet18 - pretrained on ImageNet, fine-tuned

def get_resnet18(freeze_backbone: bool = True) -> nn.Module:
    """
    ResNet18 pretrained on ImageNet, adapted for binary classification.

    The final FC layer is replaced with a single-output linear layer.
    Backbone is optionally frozen for the first training phase,
    then unfrozen for full fine-tuning.

    Args:
        freeze_backbone: If True, only the FC layer is trainable initially.

    Returns:
        Modified ResNet18 model.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreezes all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def get_model(name: str, freeze_backbone: bool = True) -> nn.Module:
    """
    Model factory.

    Args:
        name: One of 'simplecnn', 'convnext', 'resnet18'.
        freeze_backbone: Only relevant for resnet18.

    Returns:
        Instantiated model.
    """
    if name == 'simplecnn':
        return SimpleCNN()
    elif name == 'convnext':
        return MiniConvNeXt()
    elif name == 'resnet18':
        return get_resnet18(freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model: {name}. Choose from ['simplecnn', 'convnext', 'resnet18']")