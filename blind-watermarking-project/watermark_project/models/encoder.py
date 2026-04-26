import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, wm_size=32):
        super().__init__()
        self.wm_size = wm_size

        self.pool = nn.MaxPool2d(2)

        # Down
        self.c1 = ConvBlock(3, 64)
        self.c2 = ConvBlock(64, 128)
        self.c3 = ConvBlock(128, 256)

        # Bottleneck (inject watermark here)
        self.bottleneck = ConvBlock(256 + 1, 512)

        # Up
        self.u1 = ConvBlock(512 + 256, 256)
        self.u2 = ConvBlock(256 + 128, 128)
        self.u3 = ConvBlock(128 + 64, 64)

        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x, wm):
        c1 = self.c1(x)
        c2 = self.c2(self.pool(c1))
        c3 = self.c3(self.pool(c2))

        # Resize watermark to bottleneck size
        wm = F.interpolate(wm, size=c3.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([c3, wm], dim=1)
        x = self.bottleneck(x)

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, c2], dim=1)
        x = self.u2(x)

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, c1], dim=1)
        x = self.u3(x)

        return torch.sigmoid(self.final(x))