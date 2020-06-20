'''encoder.py'''

import torch
import torch.nn as nn

from utils import settings

from dataclasses import dataclass


@dataclass
class Features:
    global_: torch.tensor
    spatial: torch.tensor


class ClippedResnet(nn.Module):
    def __init__(self):
        '''A pretrained resnet152 without the classification layers.

        See torchvision/models/resnet.py for how pytorch handles loading
        ResNet-152 pretrained.
        '''
        super(ClippedResnet, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.5.0', 'resnet152', pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for param in self.parameters():
            param.requires_grad = False

    def set_tunable(self):
        for param in [*self.layer3.parameters(), *self.layer4.parameters()]:
            param.requires_grad = True

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = ClippedResnet()
        self.project_global = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 512)
        )
        self.project_spatial = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 512)
        )

    def forward(self, x):
        if not x.is_cuda:
            x = x.to(settings.device)
        x = self.resnet(x)
        # Permute the features for projection
        x = x.permute((0, 2, 3, 1))
        V = x.view(-1, 49, 2048)
        vᵍ = torch.mean(V, dim=1)
        vᵍ = self.project_global(vᵍ)
        V = self.project_spatial(V)
        return Features(vᵍ, V)
