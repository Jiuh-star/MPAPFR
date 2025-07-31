from __future__ import annotations

import torch
import torch.nn as nn


HEAD_NAME = "classifier"  # NOTE: This is essential for some algorithms


class CnnModel(nn.Module):
    def __init__(self, in_channels=1, num_class=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.classifier(out)
        return out
