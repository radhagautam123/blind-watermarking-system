import cv2
import numpy as np


DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_WM_SIZE = (32, 32)


def load_image_rgb(path, size=DEFAULT_IMAGE_SIZE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def load_image_rgb_float(path, size=DEFAULT_IMAGE_SIZE):
    img = load_image_rgb(path, size=size)
    return img.astype(np.float32) / 255.0


def _tight_crop_binary(wm):
    ys, xs = np.where(wm > 0)
    if len(xs) == 0 or len(ys) == 0:
        return wm
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return wm[y0:y1, x0:x1]


def _pad_to_square(img, pad_value=0):
    h, w = img.shape[:2]
    s = max(h, w)

    if img.ndim == 2:
        canvas = np.full((s, s), pad_value, dtype=img.dtype)
    else:
        canvas = np.full((s, s, img.shape[2]), pad_value, dtype=img.dtype)

    top = (s - h) // 2
    left = (s - w) // 2
    canvas[top:top + h, left:left + w] = img
    return canvas


def _clean_binary_logo(wm):
    wm = (wm > 0).astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)
    wm = cv2.morphologyEx(wm, cv2.MORPH_OPEN, kernel)
    wm = cv2.morphologyEx(wm, cv2.MORPH_CLOSE, kernel)
    return (wm > 127).astype(np.uint8)


def load_binary_watermark(path, size=DEFAULT_WM_SIZE, threshold=127, cleanup=True, center=True):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read watermark image: {path}")

    _, wm = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)

    if cleanup:
        wm = _clean_binary_logo(wm)

    if center:
        wm = _tight_crop_binary(wm)
        wm = _pad_to_square(wm, pad_value=0)

    wm = cv2.resize(wm.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    wm = (wm > 0).astype(np.uint8)
    return wm


def load_grayscale_watermark(path, size=DEFAULT_WM_SIZE):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read watermark image: {path}")

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def load_color_watermark(path, size=DEFAULT_WM_SIZE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read watermark image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def watermark_to_tensor_shape(wm):
    wm = np.asarray(wm)

    if wm.ndim == 2:
        return wm[None, :, :]
    elif wm.ndim == 3:
        return np.transpose(wm, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported watermark shape: {wm.shape}")


def ensure_binary_uint8(wm):
    wm = np.asarray(wm)
    wm = (wm > 0).astype(np.uint8)
    return wm