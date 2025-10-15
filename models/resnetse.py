from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn import functional as F
import torch.nn as nn

# SEblock
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

# 添加SE后的ResNet18
class ResNet18_SE(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.se = SEBlock(512)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.se(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
