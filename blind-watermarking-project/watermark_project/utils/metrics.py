import numpy as np
from skimage.metrics import structural_similarity as ssim


def psnr(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 99.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1, img2):
    if img1.ndim == 3:
        return ssim(img1, img2, channel_axis=2, data_range=255)
    return ssim(img1, img2, data_range=255)


def normalized_correlation(wm_true, wm_pred):
    a = wm_true.astype(np.float32).flatten()
    b = wm_pred.astype(np.float32).flatten()
    num = np.sum(a * b)
    den = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-8
    return float(num / den)


def ber(wm_true, wm_pred):
    a = wm_true.astype(np.uint8).flatten()
    b = wm_pred.astype(np.uint8).flatten()
    return float(np.mean(a != b))
