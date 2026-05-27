import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stn import STN
from models.attention import CBAMBlock


def gn(ch, groups=8):
    g = min(groups, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


class ConvGNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
            gn(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = ConvGNAct(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            gn(out_c)
        )
        self.act = nn.SiLU(inplace=True)
        self.skip = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class AttnResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, reduction=16):
        super().__init__()
        self.res = ResidualBlock(in_c, out_c)
        self.attn = CBAMBlock(out_c, reduction=reduction, spatial_kernel=7, residual=True)

    def forward(self, x):
        x = self.res(x)
        x = self.attn(x)
        return x


class KeyEncoder(nn.Module):
    def __init__(self, wm_size=32, base=24):
        super().__init__()
        self.wm_size = wm_size
        self.net = nn.Sequential(
            ConvGNAct(1, base, 3, 1, 1),
            AttnResidualBlock(base, base, reduction=8),
            AttnResidualBlock(base, base, reduction=8),
            nn.Conv2d(base, base, 1, bias=False),
            gn(base),
            nn.SiLU(inplace=True),
            CBAMBlock(base, reduction=8, spatial_kernel=7, residual=True)
        )

    def forward(self, key):
        if key.dim() == 2:
            key = key.unsqueeze(0).unsqueeze(0)
        elif key.dim() == 3:
            key = key.unsqueeze(1)
        key = key.view(key.size(0), 1, self.wm_size, self.wm_size)
        return self.net(key)


class FeatureModulator(nn.Module):
    def __init__(self, feat_c, key_c):
        super().__init__()
        self.to_gamma = nn.Conv2d(key_c, feat_c, 1)
        self.to_beta = nn.Conv2d(key_c, feat_c, 1)

    def forward(self, feat, key_feat):
        gamma = torch.tanh(self.to_gamma(key_feat))
        beta = self.to_beta(key_feat)
        return feat * (1.0 + 0.18 * gamma) + 0.18 * beta


class PyramidFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            gn(out_channels),
            nn.SiLU(inplace=True),
            CBAMBlock(out_channels, reduction=8, spatial_kernel=7, residual=True)
        )

    def forward(self, feats, out_size):
        resized = [
            F.interpolate(f, size=out_size, mode="bilinear", align_corners=False)
            if f.shape[2:] != out_size else f
            for f in feats
        ]
        return self.proj(torch.cat(resized, dim=1))


class DecoderNet(nn.Module):
    def __init__(self, wm_size=32, base=24, use_stn=True):
        super().__init__()
        self.wm_size = wm_size
        self.use_stn = use_stn
        self.stn = STN()

        self.key_encoder = KeyEncoder(wm_size=wm_size, base=base)

        self.img_stem = nn.Sequential(
            ConvGNAct(3, base, 3, 1, 1),
            AttnResidualBlock(base, base, reduction=8),
            CBAMBlock(base, reduction=8, spatial_kernel=7, residual=True)
        )

        self.down1 = nn.Sequential(
            ConvGNAct(base, 64, 3, 2, 1),
            AttnResidualBlock(64, 64, reduction=8)
        )
        self.down2 = nn.Sequential(
            ConvGNAct(64, 128, 3, 2, 1),
            AttnResidualBlock(128, 128, reduction=16)
        )
        self.down3 = nn.Sequential(
            ConvGNAct(128, 192, 3, 2, 1),
            AttnResidualBlock(192, 192, reduction=16)
        )
        self.down4 = nn.Sequential(
            ConvGNAct(192, 256, 3, 2, 1),
            AttnResidualBlock(256, 256, reduction=16),
            AttnResidualBlock(256, 256, reduction=16)
        )

        self.mod0 = FeatureModulator(base, base)
        self.mod1 = FeatureModulator(64, base)
        self.mod2 = FeatureModulator(128, base)
        self.mod3 = FeatureModulator(192, base)
        self.mod4 = FeatureModulator(256, base)

        self.up3 = nn.Sequential(
            ConvGNAct(256 + 192, 192, 3, 1, 1),
            AttnResidualBlock(192, 192, reduction=16)
        )
        self.up2 = nn.Sequential(
            ConvGNAct(192 + 128, 128, 3, 1, 1),
            AttnResidualBlock(128, 128, reduction=16)
        )
        self.up1 = nn.Sequential(
            ConvGNAct(128 + 64, 64, 3, 1, 1),
            AttnResidualBlock(64, 64, reduction=8)
        )
        self.up0 = nn.Sequential(
            ConvGNAct(64 + base, base, 3, 1, 1),
            AttnResidualBlock(base, base, reduction=8)
        )

        self.pyramid = PyramidFusion(
            in_channels=base + 64 + 128 + 192 + 256,
            out_channels=base
        )

        self.fuse_attn = CBAMBlock(base, reduction=8, spatial_kernel=7, residual=True)

        self.local_head = nn.Sequential(
            AttnResidualBlock(base, base, reduction=8),
            ConvGNAct(base, base, 3, 1, 1),
            CBAMBlock(base, reduction=8, spatial_kernel=7, residual=True)
        )

        self.confidence_head = nn.Sequential(
            nn.Conv2d(base, max(base // 2, 8), 3, padding=1, bias=False),
            gn(max(base // 2, 8)),
            nn.SiLU(inplace=True),
            CBAMBlock(max(base // 2, 8), reduction=8, spatial_kernel=7, residual=True),
            nn.Conv2d(max(base // 2, 8), 1, 1),
            nn.Sigmoid()
        )

        self.spatial_refine = nn.Sequential(
            AttnResidualBlock(base, base, reduction=8),
            ConvGNAct(base, base, 3, 1, 1),
        )

        self.logit_head = nn.Sequential(
            nn.Conv2d(base + 1, base, 3, padding=1, bias=False),
            gn(base),
            nn.SiLU(inplace=True),
            CBAMBlock(base, reduction=8, spatial_kernel=7, residual=True),
            nn.Conv2d(base, 1, 1)
        )

    def resize_key(self, key_feat, feat):
        return F.interpolate(key_feat, size=feat.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x, key):
        if self.use_stn:
            x = self.stn(x)

        if key.size(0) != x.size(0):
            key = key.expand(x.size(0), -1, -1, -1)

        key_feat = self.key_encoder(key)

        x0 = self.img_stem(x)
        x0 = self.mod0(x0, self.resize_key(key_feat, x0))

        x1 = self.down1(x0)
        x1 = self.mod1(x1, self.resize_key(key_feat, x1))

        x2 = self.down2(x1)
        x2 = self.mod2(x2, self.resize_key(key_feat, x2))

        x3 = self.down3(x2)
        x3 = self.mod3(x3, self.resize_key(key_feat, x3))

        x4 = self.down4(x3)
        x4 = self.mod4(x4, self.resize_key(key_feat, x4))

        y3 = F.interpolate(x4, size=x3.shape[2:], mode="bilinear", align_corners=False)
        y3 = self.up3(torch.cat([y3, x3], dim=1))

        y2 = F.interpolate(y3, size=x2.shape[2:], mode="bilinear", align_corners=False)
        y2 = self.up2(torch.cat([y2, x2], dim=1))

        y1 = F.interpolate(y2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        y1 = self.up1(torch.cat([y1, x1], dim=1))

        y0 = F.interpolate(y1, size=x0.shape[2:], mode="bilinear", align_corners=False)
        y0 = self.up0(torch.cat([y0, x0], dim=1))

        pyr = self.pyramid([x0, x1, x2, x3, x4], out_size=y0.shape[2:])
        fused = self.fuse_attn(y0 + pyr)

        local_feat = self.local_head(fused)
        local_feat = self.spatial_refine(local_feat)

        if local_feat.shape[2:] != (self.wm_size, self.wm_size):
            local_feat = F.interpolate(
                local_feat,
                size=(self.wm_size, self.wm_size),
                mode="bilinear",
                align_corners=False
            )

        confidence = self.confidence_head(local_feat)
        logits = self.logit_head(torch.cat([local_feat, confidence], dim=1))
        return logits