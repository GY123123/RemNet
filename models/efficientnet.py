from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class EfficientNet5C(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super().__init__()
        self.model = EfficientNet.from_name('efficientnet-b0')
        self.model._conv_stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
