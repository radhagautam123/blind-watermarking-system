import torch
import torch.nn.functional as F
import math
import random


# ---------------- NOISE ----------------
def add_gaussian_noise(x, sigma):
    return torch.clamp(x + sigma * torch.randn_like(x), 0.0, 1.0)


# ---------------- BLUR ----------------
def blur3x3(x):
    kernel = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=x.dtype, device=x.device) / 16.0
    kernel = kernel.view(1,1,3,3).repeat(x.size(1),1,1,1)
    return F.conv2d(x, kernel, padding=1, groups=x.size(1))


# ---------------- RESIZE ----------------
def resize_restore(x, scale):
    h, w = x.shape[-2:]
    y = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
    return F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)


# ---------------- ROTATION ----------------
def rotate_tensor(x, angle):
    theta = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle),  math.cos(angle), 0]
    ], dtype=x.dtype, device=x.device)

    theta = theta.unsqueeze(0).repeat(x.size(0),1,1)

    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, padding_mode='border', align_corners=False)


# ---------------- RANDOM CROP ----------------
def random_crop(x, crop_scale):
    B, C, H, W = x.shape
    new_h = int(H * crop_scale)
    new_w = int(W * crop_scale)

    top = random.randint(0, H - new_h)
    left = random.randint(0, W - new_w)

    cropped = x[:, :, top:top+new_h, left:left+new_w]
    return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)


# ---------------- RANDOM ATTACK PIPELINE ----------------
def random_attack(x, epoch=None):
    """
    Progressive multi-attack pipeline
    """

    if epoch is None:
        epoch = 50  # default (for inference safety)

    # ---------------- STAGE CONTROL ----------------
    if epoch < 10:
        # Only slight noise → improve invisibility
        if random.random() < 0.5:
            x = add_gaussian_noise(x, sigma=0.01)
        return x

    elif epoch < 30:
        # Moderate attacks
        if random.random() < 0.7:
            x = add_gaussian_noise(x, sigma=0.02)

        if random.random() < 0.5:
            x = blur3x3(x)

        if random.random() < 0.5:
            scale = random.uniform(0.7, 1.2)
            x = resize_restore(x, scale)

    else:
        # 🔥 Strong attacks (real-world simulation)

        # Noise
        if random.random() < 0.7:
            x = add_gaussian_noise(x, sigma=0.03)

        # Blur
        if random.random() < 0.6:
            x = blur3x3(x)

        # Resize / Zoom
        if random.random() < 0.7:
            scale = random.uniform(0.6, 1.4)
            x = resize_restore(x, scale)

        # Rotation
        if random.random() < 0.7:
            angle = random.uniform(-30, 30) * math.pi / 180
            x = rotate_tensor(x, angle)

        # Crop (VERY IMPORTANT for robustness)
        if random.random() < 0.6:
            crop_scale = random.uniform(0.6, 0.9)
            x = random_crop(x, crop_scale)

    return torch.clamp(x, 0.0, 1.0)