import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stn import STN


# ---------------- BASIC BLOCK ----------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ---------------- DECODER ----------------
class DecoderNet(nn.Module):
    def __init__(self, wm_size=32):
        super().__init__()

        self.wm_size = wm_size
        self.stn = STN()

        # ---------- ENCODER ----------
        self.c1 = ConvBlock(4, 64)
        self.p1 = nn.MaxPool2d(2)

        self.c2 = ConvBlock(64, 128)
        self.p2 = nn.MaxPool2d(2)

        self.c3 = ConvBlock(128, 256)
        self.p3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(256, 512)

        # ---------- DECODER ----------
        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.u1 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.u2 = ConvBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.u3 = ConvBlock(128, 64)

        # ---------- OUTPUT ----------
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x, key):

        # ---------------- STN ALIGNMENT ----------------
        x = self.stn(x)

        # ---------------- KEY HANDLING ----------------
        key_map = F.interpolate(key, size=x.shape[2:], mode='bilinear', align_corners=False)
        key_map = key_map.expand(x.size(0), -1, -1, -1)

        # concat image + key
        x = torch.cat([x, key_map], dim=1)

        # ---------------- ENCODER ----------------
        c1 = self.c1(x)
        c2 = self.c2(self.p1(c1))
        c3 = self.c3(self.p2(c2))

        b = self.bottleneck(self.p3(c3))

        # ---------------- DECODER ----------------
        x = self.up1(b)
        x = torch.cat([x, c3], dim=1)
        x = self.u1(x)

        x = self.up2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.u2(x)

        x = self.up3(x)
        x = torch.cat([x, c1], dim=1)
        x = self.u3(x)

        # ---------------- OUTPUT ----------------
        y = self.final(x)

        # resize to watermark size
        y = F.interpolate(y, size=(self.wm_size, self.wm_size), mode='bilinear', align_corners=False)

        return y
