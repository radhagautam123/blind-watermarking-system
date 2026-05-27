import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import CBAMBlock


def norm2d(channels):
    groups = 8
    if channels < groups:
        groups = 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            norm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            norm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=True)


class AttnResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res = ResidualBlock(channels)
        self.attn = CBAMBlock(channels, reduction=16, spatial_kernel=7, residual=True)

    def forward(self, x):
        x = self.res(x)
        x = self.attn(x)
        return x


class EdgeStrengthMap(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0,  0,  0],
             [1,  2,  1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, img):
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        g = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
        g = g / (g.amax(dim=(2, 3), keepdim=True) + 1e-8)
        return g


class EncoderNet(nn.Module):
    def __init__(self, wm_size=32):
        super().__init__()
        self.wm_size = wm_size

        self.img_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            norm2d(64),
            nn.ReLU(inplace=True),
            AttnResidualBlock(64),
            AttnResidualBlock(64),
            CBAMBlock(64, reduction=16, spatial_kernel=7, residual=True),
        )

        self.wm_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            norm2d(32),
            nn.ReLU(inplace=True),
            AttnResidualBlock(32),
            AttnResidualBlock(32),
            CBAMBlock(32, reduction=8, spatial_kernel=7, residual=True),
        )

        self.key_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            norm2d(32),
            nn.ReLU(inplace=True),
            AttnResidualBlock(32),
            AttnResidualBlock(32),
            CBAMBlock(32, reduction=8, spatial_kernel=7, residual=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(64 + 32 + 32, 128, 3, padding=1, bias=False),
            norm2d(128),
            nn.ReLU(inplace=True),
            AttnResidualBlock(128),
            AttnResidualBlock(128),
            CBAMBlock(128, reduction=16, spatial_kernel=7, residual=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            norm2d(64),
            nn.ReLU(inplace=True),
            CBAMBlock(64, reduction=16, spatial_kernel=7, residual=True),
        )

        self.residual_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            norm2d(32),
            nn.ReLU(inplace=True),
            CBAMBlock(32, reduction=8, spatial_kernel=7, residual=True),
            nn.Conv2d(32, 3, 1)
        )

        self.alpha_head = nn.Sequential(
            nn.Conv2d(64 + 1, 32, 3, padding=1, bias=False),
            norm2d(32),
            nn.ReLU(inplace=True),
            CBAMBlock(32, reduction=8, spatial_kernel=7, residual=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        self.edge_map = EdgeStrengthMap()

        self.strength = nn.Parameter(torch.tensor(-1.0))
        self.register_buffer("strength_min", torch.tensor(0.04))
        self.register_buffer("strength_max", torch.tensor(0.14))

    def forward(self, img, wm, key):
        h, w = img.shape[-2:]

        if wm.shape[-2:] != (h, w):
            wm = F.interpolate(wm, size=(h, w), mode="bilinear", align_corners=False)

        if key.shape[-2:] != (h, w):
            key = F.interpolate(key, size=(h, w), mode="bilinear", align_corners=False)

        wm = (wm > 0.5).float()
        key = key.clamp(0, 1)

        img_feat = self.img_branch(img)
        wm_feat = self.wm_branch(wm)
        key_feat = self.key_branch(key)

        fused = self.fusion(torch.cat([img_feat, wm_feat, key_feat], dim=1))
        residual = torch.tanh(self.residual_head(fused))

        edge = self.edge_map(img)
        alpha_local = self.alpha_head(torch.cat([fused, edge], dim=1))

        s = torch.sigmoid(self.strength)
        s = self.strength_min + (self.strength_max - self.strength_min) * s

        strength_map = s * (0.55 + 0.45 * edge) * (0.70 + 0.30 * alpha_local)
        return residual * strength_map