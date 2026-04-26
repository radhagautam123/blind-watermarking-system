import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= STN =================
class STN(nn.Module):
    def __init__(self):
        super().__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7),
            nn.MaxPool2d(2),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 6)
        )

        # identity transform
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(x.size(0), -1)

        # 🔥 CRITICAL: match training FC input
        xs = xs[:, :128]

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x


# ================= CONV BLOCK =================
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


# ================= DECODER =================
class DecoderNet(nn.Module):
    def __init__(self, wm_size=32):
        super().__init__()
        self.wm_size = wm_size

        self.stn = STN()

        self.net = nn.Sequential(
            ConvBlock(3, 64),      # net.0
            nn.MaxPool2d(2),

            ConvBlock(64, 128),    # net.2
            nn.MaxPool2d(2),

            ConvBlock(128, 256),   # net.4
            nn.MaxPool2d(2),

            ConvBlock(256, 512),   # net.6
            nn.Conv2d(512, 1, 1)   # net.7 ✔
        )

    def forward(self, x):
        x = self.stn(x)
        x = self.net(x)

        x = F.interpolate(
            x,
            size=(self.wm_size, self.wm_size),
            mode='bilinear',
            align_corners=False
        )

        return x