"""
VGG Implementation in PyTorch

Paper Link: https://arxiv.org/pdf/1409.1556.pdf
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


model_configs = {
    "VGG11": [
        64,
        "pooling",
        128,
        "pooling",
        256,
        256,
        "pooling",
        512,
        512,
        "pooling",
        512,
        512,
        "pooling",
    ],
    "VGG13": [
        64,
        64,
        "pooling",
        128,
        128,
        "pooling",
        256,
        256,
        "pooling",
        512,
        512,
        "pooling",
        512,
        512,
        "pooling",
    ],
    "VGG16": [
        64,
        64,
        "pooling",
        128,
        128,
        "pooling",
        256,
        256,
        256,
        "pooling",
        512,
        512,
        512,
        "pooling",
        512,
        512,
        512,
        "pooling",
    ],
    "VGG19": [
        64,
        64,
        "pooling",
        128,
        128,
        "pooling",
        256,
        256,
        256,
        256,
        "pooling",
        512,
        512,
        512,
        512,
        "pooling",
        512,
        512,
        512,
        512,
        "pooling",
    ],
}


class VGG(nn.Module):
    def __init__(self, model_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(model_configs[model_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    # Main model creation function
    def _make_layers(self, model_configs):

        layers = []
        in_channels = 3

        for x in model_configs:
            if x == "pooling":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        # Unpack layers for adding to the final Sequential container
        return nn.Sequential(*layers)


# VGG16 creation using generic class
def VGG16():
    return VGG("VGG16")


# VGG19 creation using generic class
def VGG19():
    return VGG("VGG19")

