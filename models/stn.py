import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self):
        super().__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(x.size(0), -1)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        theta = torch.clamp(theta, -1.0, 1.0)

        grid = F.affine_grid(theta, x.size(), align_corners=False)

        x = F.grid_sample(
            x,
            grid,
            padding_mode='border',
            align_corners=False
        )

        return x
