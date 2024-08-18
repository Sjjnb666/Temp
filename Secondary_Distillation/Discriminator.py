import segmentation_models_pytorch as smp
import torch.nn as nn
import torch

class PretrainedDiscriminator(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(PretrainedDiscriminator, self).__init__()
        self.deeplabv3plus = smp.DeepLabV3Plus(
            encoder_name="resnet18", 
            encoder_weights="imagenet", 
            in_channels=in_channels, 
            classes=num_classes
        )


    def forward(self, x):
        x = self.deeplabv3plus(x)
        return torch.sigmoid(x).view(-1, 1).squeeze(1)