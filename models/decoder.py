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
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class DecoderNet(nn.Module):
    def __init__(self, wm_size=32):
        super().__init__()

        self.wm_size = wm_size
        self.stn = STN()

        # 3 image + 1 key = 4 channels
        self.net = nn.Sequential(
            ConvBlock(4, 64),
            nn.MaxPool2d(2),

            ConvBlock(64, 128),
            nn.MaxPool2d(2),

            ConvBlock(128, 256),
            nn.MaxPool2d(2),

            ConvBlock(256, 512),
            nn.Conv2d(512, 1, 1)
        )

    def forward(self, x, key):

        # STN alignment
        x = self.stn(x)

        # Resize key correctly
        key_map = F.interpolate(key, size=x.shape[2:])
        key_map = key_map.expand(x.size(0), -1, -1, -1)

        # Concatenate
        x = torch.cat([x, key_map], dim=1)

        y = self.net(x)

        return F.interpolate(y, size=(self.wm_size, self.wm_size))
