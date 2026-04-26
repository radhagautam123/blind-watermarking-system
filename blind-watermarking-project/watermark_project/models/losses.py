import torch
import torch.nn.functional as F


# ---------------- SSIM LOSS ----------------
def ssim_loss(x, y):
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

    # 🔥 Combine → improves PSNR + perceptual quality
    return 0.7 * mse + 0.3 * ssim


# ---------------- WATERMARK LOSS ----------------
def watermark_loss(pred_wm, true_wm):
    bce = F.binary_cross_entropy(pred_wm, true_wm)

    # Optional: add L1 for stability
    l1 = F.l1_loss(pred_wm, true_wm)

    return 0.8 * bce + 0.2 * l1