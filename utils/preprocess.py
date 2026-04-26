import cv2
import numpy as np


def load_image_rgb(path, size=(256, 256)):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img


def load_binary_watermark(path, size=(16, 16), threshold=127):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read watermark image: {path}")

    img = cv2.resize(img, size)
    wm = (img > threshold).astype(np.uint8)
    return wm