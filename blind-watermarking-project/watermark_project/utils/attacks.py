import cv2
import numpy as np
import random


# ---------------- JPEG ----------------
def jpeg_compress(img, quality=50):
    quality = int(np.clip(quality, 10, 95))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)


# ---------------- NOISE ----------------
def gaussian_noise(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


# ---------------- BLUR ----------------
def gaussian_blur(img, sigma=1.5):
    return cv2.GaussianBlur(img, (0, 0), sigma)


# ---------------- ROTATION ----------------
def rotate_attack(img, angle=None):
    if angle is None:
        angle = np.random.uniform(-30, 30)

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return rotated, angle


# ---------------- ROTATION CORRECTION ----------------
def rotate_correction(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# ---------------- SCALING ----------------
def scaling_attack(img, scale=None):
    if scale is None:
        scale = np.random.uniform(0.5, 1.5)

    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w*scale), int(h*scale)))

    return cv2.resize(resized, (w, h))


# ---------------- RANDOM CROP ----------------
def crop_attack(img, ratio=None):
    if ratio is None:
        ratio = np.random.uniform(0.1, 0.4)

    h, w = img.shape[:2]
    dh = int(h * ratio)
    dw = int(w * ratio)

    cropped = img[dh:h-dh, dw:w-dw]
    return cv2.resize(cropped, (w, h))


# ---------------- ZOOM ----------------
def zoom_attack(img, zoom_factor=None):
    if zoom_factor is None:
        zoom_factor = np.random.uniform(1.1, 1.5)

    h, w = img.shape[:2]

    zh, zw = int(h/zoom_factor), int(w/zoom_factor)
    y1 = (h - zh) // 2
    x1 = (w - zw) // 2

    cropped = img[y1:y1+zh, x1:x1+zw]
    return cv2.resize(cropped, (w, h))


# ---------------- SALT PEPPER ----------------
def salt_pepper_noise(img, amount=0.02):
    noisy = img.copy()
    h, w, _ = img.shape

    num = int(amount * h * w)

    coords = (np.random.randint(0, h, num), np.random.randint(0, w, num))
    noisy[coords] = 255

    coords = (np.random.randint(0, h, num), np.random.randint(0, w, num))
    noisy[coords] = 0

    return noisy


# ---------------- 🔥 COMBINED ATTACK ----------------
def combined_attack(img):
    """
    Simulates real-world distortions
    """

    if random.random() < 0.7:
        img = jpeg_compress(img, quality=random.randint(20, 70))

    if random.random() < 0.5:
        img = gaussian_noise(img, sigma=random.randint(5, 20))

    if random.random() < 0.5:
        img = gaussian_blur(img, sigma=random.uniform(0.5, 2.0))

    if random.random() < 0.5:
        img = scaling_attack(img)

    if random.random() < 0.5:
        img, _ = rotate_attack(img)

    if random.random() < 0.5:
        img = crop_attack(img)

    return img