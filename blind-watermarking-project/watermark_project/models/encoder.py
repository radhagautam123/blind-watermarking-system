import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import ChannelAttention, SpatialAttention


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # -------- DOWN --------
        self.c1 = ConvBlock(4, 64)
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()

        self.c2 = ConvBlock(64, 128)
        self.ca2 = ChannelAttention(128)
        self.sa2 = SpatialAttention()

        self.c3 = ConvBlock(128, 256)
        self.ca3 = ChannelAttention(256)
        self.sa3 = SpatialAttention()

        # -------- BOTTLENECK --------
        self.bottleneck = ConvBlock(256, 512)

        # -------- UP --------
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.u1 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u2 = ConvBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.u3 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, img, wm):
        # Resize watermark to match image
        wm = F.interpolate(wm, size=img.shape[2:])
        x = torch.cat([img, wm], dim=1)

        # -------- DOWN + STABLE ATTENTION --------
        c1 = self.c1(x)
        c1 = c1 + self.sa1(self.ca1(c1)) * 0.5

        c2 = self.c2(self.pool(c1))
        c2 = c2 + self.sa2(self.ca2(c2)) * 0.5

        c3 = self.c3(self.pool(c2))
        c3 = c3 + self.sa3(self.ca3(c3)) * 0.5

        # -------- BOTTLENECK --------
        b = self.bottleneck(self.pool(c3))

        # -------- UP --------
        x = self.up1(b)
        x = torch.cat([x, c3], dim=1)
        x = self.u1(x)

        x = self.up2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.u2(x)

        x = self.up3(x)
        x = torch.cat([x, c1], dim=1)
        x = self.u3(x)

        # -------- FINAL (CONTROLLED RESIDUAL) --------
        return 0.1 * torch.tanh(self.final(x))