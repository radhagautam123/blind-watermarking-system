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
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # 3 (img) + 1 (wm) + 1 (key)
        self.c1 = ConvBlock(5, 64)
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()

        self.c2 = ConvBlock(64, 128)
        self.ca2 = ChannelAttention(128)
        self.sa2 = SpatialAttention()

        self.c3 = ConvBlock(128, 256)
        self.ca3 = ChannelAttention(256)
        self.sa3 = SpatialAttention()

        self.bottleneck = ConvBlock(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.u1 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.u2 = ConvBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.u3 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, img, wm, key):

        # ================= FIX WATERMARK SHAPE =================
        # Accepts (32,32), (1,32,32), (B,32,32), or (B,1,32,32)
        if wm.dim() == 2:
            wm = wm.unsqueeze(0).unsqueeze(0)

        elif wm.dim() == 3:
            wm = wm.unsqueeze(1)

        # Ensure correct shape
        wm = wm.view(wm.size(0), 1, 32, 32)

        # Resize watermark to match image
        wm = F.interpolate(wm, size=img.shape[2:], mode='bilinear', align_corners=False)

        # ================= FIX KEY SHAPE =================
        key_map = F.interpolate(key, size=img.shape[2:], mode='bilinear', align_corners=False)
        key_map = key_map.expand(img.size(0), -1, -1, -1)

        # ================= CONCAT =================
        x = torch.cat([img, wm, key_map], dim=1)

        # ================= ENCODER =================
        c1 = self.sa1(self.ca1(self.c1(x)))
        c2 = self.sa2(self.ca2(self.c2(self.pool(c1))))
        c3 = self.sa3(self.ca3(self.c3(self.pool(c2))))

        b = self.bottleneck(self.pool(c3))

        x = self.up1(b)
        x = torch.cat([x, c3], dim=1)
        x = self.u1(x)

        x = self.up2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.u2(x)

        x = self.up3(x)
        x = torch.cat([x, c1], dim=1)
        x = self.u3(x)

        # ================= OUTPUT =================
        delta = torch.tanh(self.final(x)) * 0.05
        return delta