import torch
import torch.nn.functional as F


def ssim_loss(x, y):
    x = torch.clamp(x, 0, 1)
    y = torch.clamp(y, 0, 1)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)

    ssim_map = numerator / (denominator + 1e-8)
    return 1.0 - ssim_map.mean()


def image_loss(host, watermarked):
    mse = F.mse_loss(watermarked, host)
    l1 = F.l1_loss(watermarked, host)
    ssim = ssim_loss(watermarked, host)
    return 0.45 * mse + 0.35 * l1 + 0.20 * ssim


def bce_logits_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target.float())


def bit_accuracy_from_logits(logits, target):
    pred = (logits > 0.0).float()
    return pred.eq(target.float()).float().mean()


def ber_from_logits(logits, target):
    pred = (logits > 0.0).float()
    return 1.0 - pred.eq(target.float()).float().mean()


def ncc_loss_from_logits(logits, target):
    pred = torch.sigmoid(logits).flatten(1)
    tgt = target.float().flatten(1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    tgt = tgt - tgt.mean(dim=1, keepdim=True)
    num = (pred * tgt).sum(dim=1)
    den = torch.sqrt((pred * pred).sum(dim=1) * (tgt * tgt).sum(dim=1) + 1e-8)
    ncc = num / den
    return 1.0 - ncc.mean()


def watermark_loss(pred_wm_logits, true_wm):
    true_wm = true_wm.float()
    bce = bce_logits_loss(pred_wm_logits, true_wm)
    probs = torch.sigmoid(pred_wm_logits)
    l1 = F.l1_loss(probs, true_wm)
    ncc_l = ncc_loss_from_logits(pred_wm_logits, true_wm)
    return 0.70 * bce + 0.15 * l1 + 0.15 * ncc_l


def wrong_key_loss(wrong_pred_logits):
    """Push wrong-key outputs toward 0.5 (AMP-safe: uses logits for BCE)."""
    target_uniform = torch.full_like(wrong_pred_logits, 0.5)
    wrong_probs = torch.sigmoid(wrong_pred_logits)
    mse = F.mse_loss(wrong_probs, target_uniform)
    bce_to_half = F.binary_cross_entropy_with_logits(wrong_pred_logits, target_uniform)
    return 0.6 * mse + 0.4 * bce_to_half


def total_loss(
    host,
    watermarked,
    pred_logits,
    true_wm,
    wrong_pred_logits=None,
    wm_weight=3.0,
    img_weight=0.8,
    key_weight=0.1
):
    wm_l = watermark_loss(pred_logits, true_wm)
    img_l = image_loss(host, watermarked)

    if wrong_pred_logits is None:
        key_l = torch.tensor(0.0, device=host.device, dtype=host.dtype)
    else:
        key_l = wrong_key_loss(wrong_pred_logits)

    loss = wm_weight * wm_l + img_weight * img_l + key_weight * key_l
    return loss, wm_l, img_l, key_l
