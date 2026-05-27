import torch
import torch.nn as nn
import torch.nn.functional as F


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


class STN(nn.Module):
    def __init__(
        self,
        max_rot_deg=25.0,
        max_scale_delta=0.22,
        max_aniso_delta=0.14,
        max_trans=0.18,
        max_shear=0.10
    ):
        super().__init__()
        self.max_rot_deg = max_rot_deg
        self.max_scale_delta = max_scale_delta
        self.max_aniso_delta = max_aniso_delta
        self.max_trans = max_trans
        self.max_shear = max_shear

        self.localization = nn.Sequential(
            ConvGNAct(3, 32, 5, 1, 2),
            nn.AvgPool2d(2),
            ConvGNAct(32, 64, 3, 1, 1),
            nn.AvgPool2d(2),
            ConvGNAct(64, 96, 3, 1, 1),
            nn.AvgPool2d(2),
            ConvGNAct(96, 128, 3, 1, 1),
            nn.AvgPool2d(2),
            ConvGNAct(128, 160, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(160, 96),
            nn.SiLU(inplace=True),
            nn.Linear(96, 6)
        )

        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        feat = self.localization(x).flatten(1)
        params = self.fc(feat)

        rot = torch.tanh(params[:, 0]) * (self.max_rot_deg * torch.pi / 180.0)
        scale_base = 1.0 + torch.tanh(params[:, 1]) * self.max_scale_delta
        scale_x = scale_base * (1.0 + torch.tanh(params[:, 2]) * self.max_aniso_delta)
        scale_y = scale_base * (1.0 + torch.tanh(params[:, 3]) * self.max_aniso_delta)
        tx = torch.tanh(params[:, 4]) * self.max_trans
        ty = torch.tanh(params[:, 5]) * self.max_trans

        cos_r = torch.cos(rot)
        sin_r = torch.sin(rot)

        a00 = scale_x * cos_r
        a01 = -scale_y * sin_r
        a10 = scale_x * sin_r
        a11 = scale_y * cos_r

        theta = torch.zeros(x.size(0), 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = a00
        theta[:, 0, 1] = a01
        theta[:, 1, 0] = a10
        theta[:, 1, 1] = a11
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_aligned = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )
        return x_aligned