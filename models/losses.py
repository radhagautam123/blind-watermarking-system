import torch
import torch.nn.functional as F


# ---------------- SSIM LOSS ----------------
def ssim_loss(x, y):
    x = torch.clamp(x, 0, 1)
    y = torch.clamp(y, 0, 1)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    return 1 - ssim_map.mean()


# ---------------- IMAGE LOSS ----------------
def image_loss(host, watermarked):
    mse = F.mse_loss(watermarked, host)
    ssim = ssim_loss(watermarked, host)
    return 0.6 * mse + 0.4 * ssim


# ---------------- DICE LOSS ----------------
def dice_loss(pred, target):
    smooth = 1e-8
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


# ---------------- WATERMARK LOSS ----------------
def watermark_loss(pred_wm, true_wm):
    bce = F.binary_cross_entropy_with_logits(pred_wm, true_wm)

    pred_prob = torch.sigmoid(pred_wm)
    l1 = F.l1_loss(pred_prob, true_wm)
    dice = dice_loss(pred_prob, true_wm)

    return 0.6 * bce + 0.2 * l1 + 0.2 * dice
