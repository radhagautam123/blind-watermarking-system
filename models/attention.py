import torch
import torch.nn as nn


def _get_groups(channels, max_groups=8):
    g = min(max_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, act=True, use_gn=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        ]
        if use_gn:
            layers.append(nn.GroupNorm(_get_groups(out_channels), out_channels))
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, min_hidden=8):
        super().__init__()

        hidden = max(in_channels // reduction, min_hidden)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")

        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)

        x_cat = torch.cat([avg_map, max_map], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7, residual=True):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=spatial_kernel)
        self.residual = residual

    def forward(self, x):
        out = self.channel_attn(x)
        out = self.spatial_attn(out)
        if self.residual:
            out = 0.5 * out + 0.5 * x
        return out


class AttentionGate(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 8)

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        g = self.gate(x)
        return x * g