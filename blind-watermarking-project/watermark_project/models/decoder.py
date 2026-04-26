import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stn import STN
from models.attention import ChannelAttention, SpatialAttention


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

        # -------- STN (geometry correction) --------
        self.stn = STN()

        # -------- FEATURE EXTRACTION --------
        self.c1 = ConvBlock(3, 64)
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()

        self.c2 = ConvBlock(64, 128)
        self.ca2 = ChannelAttention(128)
        self.sa2 = SpatialAttention()

        self.c3 = ConvBlock(128, 256)
        self.ca3 = ChannelAttention(256)
        self.sa3 = SpatialAttention()

        self.c4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # -------- FINAL --------
        self.final = nn.Conv2d(512, 1, 1)

    def forward(self, x):

        # -------- STN --------
        x = self.stn(x)

        # -------- FEATURE EXTRACTION + ATTENTION --------
        x = self.c1(x)
        x = x + self.sa1(self.ca1(x)) * 0.5
        x = self.pool(x)

        x = self.c2(x)
        x = x + self.sa2(self.ca2(x)) * 0.5
        x = self.pool(x)

        x = self.c3(x)
        x = x + self.sa3(self.ca3(x)) * 0.5
        x = self.pool(x)

        x = self.c4(x)

        # -------- OUTPUT (NO SIGMOID) --------
        x = self.final(x)

        # -------- RESIZE TO WATERMARK --------
        x = F.interpolate(
            x,
            size=(self.wm_size, self.wm_size),
            mode='bilinear',
            align_corners=False
        )

        return x