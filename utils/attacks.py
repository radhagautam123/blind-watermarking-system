import cv2
import numpy as np


# ---------------- JPEG ----------------
def jpeg_compress(img, quality=50):
    quality = int(np.clip(quality, 5, 95))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)


# ---------------- GAUSSIAN NOISE ----------------
def gaussian_noise(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------- BLUR ----------------
def gaussian_blur(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# ---------------- ROTATION ----------------
def rotate_attack(img, angle=0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated, angle


# ---------------- SCALING ----------------
def scaling_attack(img, scale=1.0):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    return cv2.resize(resized, (w, h))


# ---------------- CROP ----------------
def crop_attack(img, ratio=0.2):
    h, w = img.shape[:2]
    dh = int(h * ratio / 2)
    dw = int(w * ratio / 2)
    cropped = img[dh:h - dh, dw:w - dw]
    return cv2.resize(cropped, (w, h))


# ---------------- ZOOM ----------------
def zoom_attack(img, zoom_factor=1.2):
    h, w = img.shape[:2]
    zh, zw = int(h / zoom_factor), int(w / zoom_factor)
    y1 = (h - zh) // 2
    x1 = (w - zw) // 2
    cropped = img[y1:y1 + zh, x1:x1 + zw]
    return cv2.resize(cropped, (w, h))


# ---------------- SALT & PEPPER ----------------
def salt_pepper_noise(img, amount=0.02):
    noisy = img.copy()
    h, w, c = img.shape
    num_pixels = int(amount * h * w)

    coords = (
        np.random.randint(0, h, num_pixels),
        np.random.randint(0, w, num_pixels)
    )
    noisy[coords[0], coords[1], :] = 255

    coords = (
        np.random.randint(0, h, num_pixels),
        np.random.randint(0, w, num_pixels)
    )
    noisy[coords[0], coords[1], :] = 0

    return noisy


# ---------------- 🔥 COMBINED ATTACK ----------------
def combined_attack(
    img,
    selected_attacks=None,
    jpeg_quality=50,
    noise_sigma=10,
    blur_kernel=5,
    scale=1.0,
    angle=0,
    crop_ratio=0.2,
    zoom_factor=1.2,
    sp_amount=0.02
):
    if selected_attacks is None:
        selected_attacks = []

    applied = []

    if "jpeg" in selected_attacks:
        img = jpeg_compress(img, quality=jpeg_quality)
        applied.append(f"JPEG(q={jpeg_quality})")

    if "gaussian" in selected_attacks:
        img = gaussian_noise(img, sigma=noise_sigma)
        applied.append(f"Noise(σ={noise_sigma})")

    if "blur" in selected_attacks:
        img = gaussian_blur(img, kernel_size=blur_kernel)
        applied.append(f"Blur(k={blur_kernel})")

    if "resize" in selected_attacks:
        img = scaling_attack(img, scale)
        applied.append(f"Scale({scale:.2f})")

    if "rotate" in selected_attacks:
        img, _ = rotate_attack(img, angle)
        applied.append(f"Rotate({angle}°)")

    if "crop" in selected_attacks:
        img = crop_attack(img, crop_ratio)
        applied.append(f"Crop({crop_ratio:.2f})")

    if "zoom" in selected_attacks:
        img = zoom_attack(img, zoom_factor)
        applied.append(f"Zoom({zoom_factor:.2f})")

    if "salt_pepper" in selected_attacks:
        img = salt_pepper_noise(img, amount=sp_amount)
        applied.append(f"SP({sp_amount:.2f})")

    return img, applied