import torchvision.models
import torch.nn as nn

class ResNET18(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
