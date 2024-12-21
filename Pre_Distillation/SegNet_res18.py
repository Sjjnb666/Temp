import torchvision.models as models
from torchvision.models import VGG16_Weights
import torch.nn as nn
import torch
class SegNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SegNet, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(
            resnet18.conv1,   # (N, 3, 1024, 1024) -> (N, 64, 512, 512)
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool, # (N, 64, 512, 512) -> (N, 64, 256, 256)
            resnet18.layer1,  # (N, 64, 256, 256) -> (N, 64, 256, 256)
            resnet18.layer2,  # (N, 64, 256, 256) -> (N, 128, 128, 128)
            resnet18.layer3,  # (N, 128, 128, 128) -> (N, 256, 64, 64)
            resnet18.layer4   # (N, 256, 64, 64) -> (N, 512, 32, 32)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), # (N, 512, 32, 32) -> (N, 256, 64, 64)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),         # (N, 256, 64, 64) -> (N, 256, 64, 64)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # (N, 256, 64, 64) -> (N, 128, 128, 128)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),         # (N, 128, 128, 128) -> (N, 128, 128, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # (N, 128, 128, 128) -> (N, 64, 256, 256)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),           # (N, 64, 256, 256) -> (N, 64, 256, 256)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),   # (N, 64, 256, 256) -> (N, 64, 512, 512)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),           # (N, 64, 512, 512) -> (N, 64, 512, 512)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # (N, 64, 512, 512) -> (N, 32, 1024, 1024)
            nn.Conv2d(32, num_classes, kernel_size=1)              # (N, 32, 1024, 1024) -> (N, num_classes, 1024, 1024)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_segnet_model(pretrained_weights_path = None):
    student_model = SegNet()
    if pretrained_weights_path is not None:
        student_model.load_state_dict(torch.load(pretrained_weights_path))
    return student_model
