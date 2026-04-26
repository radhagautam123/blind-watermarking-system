import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stn import STN


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderNet(nn.Module):
    def __init__(self, wm_size=32):
        super().__init__()
        self.wm_size = wm_size

        # 🔥 STN module
        self.stn = STN()

        self.net = nn.Sequential(
            ConvBlock(3, 64),
            nn.MaxPool2d(2),

            ConvBlock(64, 128),
            nn.MaxPool2d(2),

            ConvBlock(128, 256),
            nn.MaxPool2d(2),

            ConvBlock(256, 512),

            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 🔥 Correct geometric distortions
        x = self.stn(x)

        y = self.net(x)

        y = F.interpolate(
            y,
            size=(self.wm_size, self.wm_size),
            mode='bilinear',
            align_corners=False
        )

        return y