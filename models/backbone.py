import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ResNet18Backbone(nn.Module):
    """
    ResNet-18 backbone with all layers frozen except the final residual block.
    """
    def __init__(self, pretrained=True):
        super().__init__()

        # Load pretrained ResNet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

        self.backbone = models.resnet18(weights=weights)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Freeze everything except layer4
        for param in self.parameters():
            param.requires_grad = False

        # Train layer4
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)