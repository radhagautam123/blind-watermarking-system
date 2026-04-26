import numpy as np
import cv2


# ================= PSNR =================
def psnr(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100

    return 20 * np.log10(255.0 / np.sqrt(mse))


# ================= SSIM =================
def compute_ssim(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


# ================= NCC =================
def normalized_correlation(original, extracted):
    original = original.flatten().astype(np.float32)
    extracted = extracted.flatten().astype(np.float32)

    numerator = np.sum(original * extracted)
    denominator = np.sqrt(np.sum(original ** 2) * np.sum(extracted ** 2)) + 1e-8

    return numerator / denominator


# ================= BER =================
def ber(original, extracted):
    original = original.flatten().astype(np.uint8)
    extracted = extracted.flatten().astype(np.uint8)

    errors = np.sum(original != extracted)
    total = len(original)

    return errors / total


# ================= ACCURACY =================
def watermark_accuracy(original, extracted):
    original = original.flatten().astype(np.uint8)
    extracted = extracted.flatten().astype(np.uint8)

    correct = np.sum(original == extracted)
    total = len(original)

    return correct / total


# ================= OVERALL SCORE =================
def overall_performance(psnr_val, ssim_val, nc_val, acc_val):
    # Normalize PSNR (0–50 dB → 0–1)
    psnr_norm = min(psnr_val / 50.0, 1.0)

    return (
        0.25 * psnr_norm +
        0.25 * ssim_val +
        0.25 * nc_val +
        0.25 * acc_val
    )