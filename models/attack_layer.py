import torch
import torch.nn.functional as F
import math
import random


def add_gaussian_noise(x, sigma):
    return torch.clamp(x + sigma * torch.randn_like(x), 0.0, 1.0)


def blur3x3(x):
    kernel = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=x.dtype, device=x.device) / 16.0
    kernel = kernel.view(1,1,3,3).repeat(x.size(1),1,1,1)
    return F.conv2d(x, kernel, padding=1, groups=x.size(1))


def resize_restore(x, scale):
    h, w = x.shape[-2:]
    y = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
    return F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)


def rotate_tensor(x, angle):
    theta = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle),  math.cos(angle), 0]
    ], dtype=x.dtype, device=x.device)

    theta = theta.unsqueeze(0).repeat(x.size(0),1,1)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, padding_mode='border', align_corners=False)


def random_crop(x, crop_scale):
    B, C, H, W = x.shape
    new_h = int(H * crop_scale)
    new_w = int(W * crop_scale)

    top = random.randint(0, H - new_h)
    left = random.randint(0, W - new_w)

    cropped = x[:, :, top:top+new_h, left:left+new_w]
    return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)


def random_attack(x, epoch=None):

    if epoch is None:
        epoch = 50

    if epoch < 8:
        return x

    elif epoch < 18:
        x = add_gaussian_noise(x, random.uniform(0.01, 0.03))

        if random.random() < 0.7:
            x = blur3x3(x)

        if random.random() < 0.7:
            x = resize_restore(x, random.uniform(0.7, 1.3))

    else:
        x = add_gaussian_noise(x, random.uniform(0.02, 0.05))

        if random.random() < 0.8:
            x = blur3x3(x)

        if random.random() < 0.8:
            x = resize_restore(x, random.uniform(0.6, 1.4))

        if random.random() < 0.7:
            x = rotate_tensor(x, random.uniform(-25, 25) * math.pi / 180)

        if random.random() < 0.7:
            x = random_crop(x, random.uniform(0.6, 0.9))

        if random.random() < 0.7:
            x = torch.round(x * 255) / 255.0

    return torch.clamp(x, 0.0, 1.0)