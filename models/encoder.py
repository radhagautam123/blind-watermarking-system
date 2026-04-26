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
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # DOWN (fix: 4 channels input)
        self.c1 = ConvBlock(4, 64)
        self.c2 = ConvBlock(64, 128)
        self.c3 = ConvBlock(128, 256)

        # BOTTLENECK
        self.bottleneck = ConvBlock(256, 512)

        # UP (correct structure)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.u1 = ConvBlock(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u2 = ConvBlock(128 + 128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.u3 = ConvBlock(64 + 64, 64)

        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, img, wm):

        wm = F.interpolate(wm, size=img.shape[2:])
        x = torch.cat([img, wm], dim=1)

        c1 = self.c1(x)
        c2 = self.c2(self.pool(c1))
        c3 = self.c3(self.pool(c2))

        b = self.bottleneck(self.pool(c3))

        # UP 1
        x = self.up1(b)
        x = torch.cat([x, c3], dim=1)
        x = self.u1(x)

        # UP 2
        x = self.up2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.u2(x)

        # UP 3
        x = self.up3(x)
        x = torch.cat([x, c1], dim=1)
        x = self.u3(x)

        return torch.tanh(self.final(x))