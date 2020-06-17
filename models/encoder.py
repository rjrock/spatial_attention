'''encoder.py'''

import torch
import torch.nn as nn

from utils import settings

from dataclasses import dataclass


@dataclass
class Features:
    global_: torch.tensor
    spatial: torch.tensor


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.5.0', 'resnet152',
                                     pretrained=True)
        # Remove avg_pool and fc layer
        modules = list(self.resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.project_global = nn.Sequential(
            nn.Linear(2048, 512)
        )
        self.project_spatial = nn.Sequential(
            nn.Linear(2048, 512)
        )

    def forward(self, x):
        if not x.is_cuda:
            x = x.to(settings.device)
        with torch.no_grad():
            x = self.resnet(x)
        # Permute the features for projection
        x = x.permute((0, 2, 3, 1))
        V = x.view(-1, 49, 2048)
        vᵍ = torch.mean(V, dim=1)
        vᵍ = self.project_global(vᵍ)
        V = self.project_spatial(V)
        return Features(vᵍ, V)
