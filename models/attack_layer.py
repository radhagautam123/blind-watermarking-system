import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode


def _to_numpy_image(tensor_img):
    img = tensor_img.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255.0).astype(np.uint8)


def _from_numpy_image(np_img, device):
    if np_img.ndim == 2:
        np_img = np.expand_dims(np_img, axis=-1)
    if np_img.shape[2] == 1:
        np_img = np.repeat(np_img, 3, axis=2)
    tensor = torch.from_numpy(np_img.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor.to(device)


def _jpeg_attack(img, quality_range=(50, 90)):
    device = img.device
    out = []
    for i in range(img.size(0)):
        np_img = _to_numpy_image(img[i])
        np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        quality = random.randint(*quality_range)
        ok, enc = cv2.imencode(".jpg", np_img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            out.append(img[i].detach().cpu())
            continue
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        out.append(_from_numpy_image(dec, device).cpu())
    return torch.stack(out, dim=0).to(device)


def _gaussian_noise(img, sigma=0.02):
    noise = torch.randn_like(img) * sigma
    return torch.clamp(img + noise, 0, 1)


def _salt_pepper(img, prob=0.01):
    rnd = torch.rand_like(img)
    out = img.clone()
    out[rnd < (prob / 2.0)] = 0.0
    out[rnd > (1.0 - prob / 2.0)] = 1.0
    return out


def _gaussian_blur(img, kernel_size=5, sigma=(0.8, 1.5)):
    return TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=list(sigma))


def _rotate(img, angle=5.0):
    return TF.rotate(
        img,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        expand=False,
        fill=0.5
    )


def _translate(img, max_ratio=0.05):
    _, _, h, w = img.shape
    tx = int(random.uniform(-max_ratio, max_ratio) * w)
    ty = int(random.uniform(-max_ratio, max_ratio) * h)
    return TF.affine(
        img,
        angle=0.0,
        translate=[tx, ty],
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.5
    )


def _crop_and_resize(img, crop_ratio=0.90):
    _, _, h, w = img.shape
    ch = max(32, int(h * crop_ratio))
    cw = max(32, int(w * crop_ratio))
    top = random.randint(0, max(0, h - ch))
    left = random.randint(0, max(0, w - cw))
    cropped = img[:, :, top:top + ch, left:left + cw]
    if cropped.shape[-2:] != (h, w):
        cropped = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
    return torch.clamp(cropped, 0, 1)


def _resize_attack(img, scale=0.85):
    _, _, h, w = img.shape
    nh = max(32, int(h * scale))
    nw = max(32, int(w * scale))
    resized = F.interpolate(img, size=(nh, nw), mode="bilinear", align_corners=False)
    restored = F.interpolate(resized, size=(h, w), mode="bilinear", align_corners=False)
    return torch.clamp(restored, 0, 1)


def _zoom_attack(img, scale=1.10):
    _, _, h, w = img.shape
    nh = max(32, int(h * scale))
    nw = max(32, int(w * scale))
    zoomed = F.interpolate(img, size=(nh, nw), mode="bilinear", align_corners=False)
    top = max(0, (nh - h) // 2)
    left = max(0, (nw - w) // 2)
    cropped = zoomed[:, :, top:top + h, left:left + w]
    if cropped.shape[-2:] != (h, w):
        cropped = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
    return torch.clamp(cropped, 0, 1)


def _brightness(img, factor=1.1):
    return torch.clamp(img * factor, 0, 1)


def _contrast(img, factor=1.1):
    mean = img.mean(dim=(2, 3), keepdim=True)
    return torch.clamp((img - mean) * factor + mean, 0, 1)


def _sharpen(img):
    device = img.device
    ch = img.shape[1]
    kernel = torch.tensor(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]],
        dtype=img.dtype,
        device=device
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(ch, 1, 1, 1)
    out = F.conv2d(img, kernel, padding=1, groups=ch)
    return torch.clamp(out, 0, 1)


def _apply_attack_by_name(img, atk, params):
    if atk == "jpeg":
        return _jpeg_attack(img, quality_range=params["jpeg_range"])
    if atk == "noise":
        return _gaussian_noise(img, sigma=params["noise_sigma"])
    if atk == "blur":
        return _gaussian_blur(img, kernel_size=params["blur_kernel"], sigma=params["blur_sigma"])
    if atk == "rotation":
        return _rotate(img, angle=params["rotation_angle"])
    if atk == "crop":
        return _crop_and_resize(img, crop_ratio=params["crop_ratio"])
    if atk == "resize":
        return _resize_attack(img, scale=params["resize_scale"])
    if atk == "zoom":
        return _zoom_attack(img, scale=params["zoom_scale"])
    if atk == "contrast":
        return _contrast(img, factor=params["contrast_factor"])
    if atk == "brightness":
        return _brightness(img, factor=params["brightness_factor"])
    if atk == "salt_pepper":
        return _salt_pepper(img, prob=params["sp_prob"])
    if atk == "sharpen":
        return _sharpen(img)
    if atk == "translate":
        return _translate(img, max_ratio=params["translate_ratio"])
    return torch.clamp(img, 0, 1)


def _combo_attack(img, params, min_ops=2, max_ops=3, include_sharpen=False):
    out = img
    available = [
        "jpeg", "noise", "blur", "rotation", "crop",
        "resize", "zoom", "contrast", "brightness",
        "salt_pepper", "translate"
    ]
    if include_sharpen:
        available.append("sharpen")
    k = random.randint(min_ops, max_ops)
    ops = random.sample(available, k=k)
    for atk in ops:
        out = _apply_attack_by_name(out, atk, params)
    return torch.clamp(out, 0, 1)


def targeted_attack(img, name="jpeg50"):
    img = img.float()

    if name == "jpeg50":
        return _jpeg_attack(img, quality_range=(50, 50))
    if name == "jpeg70":
        return _jpeg_attack(img, quality_range=(70, 70))
    if name == "jpeg30":
        return _jpeg_attack(img, quality_range=(30, 30))
    if name == "blur5":
        return _gaussian_blur(img, kernel_size=5, sigma=(0.8, 1.2))
    if name == "blur7":
        return _gaussian_blur(img, kernel_size=7, sigma=(1.0, 1.5))
    if name == "rotate10":
        return _rotate(img, angle=10.0)
    if name == "rotate15":
        return _rotate(img, angle=15.0)
    if name == "crop90":
        return _crop_and_resize(img, crop_ratio=0.90)
    if name == "crop80":
        return _crop_and_resize(img, crop_ratio=0.80)
    if name == "crop70":
        return _crop_and_resize(img, crop_ratio=0.70)
    if name == "resize70":
        return _resize_attack(img, scale=0.70)
    if name == "zoom105":
        return _zoom_attack(img, scale=1.05)
    if name == "zoom120":
        return _zoom_attack(img, scale=1.20)
    if name == "zoom140":
        return _zoom_attack(img, scale=1.40)
    if name == "noise003":
        return _gaussian_noise(img, sigma=0.03)
    if name == "noise005":
        return _gaussian_noise(img, sigma=0.05)
    if name == "sp002":
        return _salt_pepper(img, prob=0.02)
    if name == "contrast":
        return _contrast(img, factor=1.20)
    if name == "brightness":
        return _brightness(img, factor=1.15)
    if name == "translate5":
        return _translate(img, max_ratio=0.05)

    if name == "translate10":
        return _translate(img, max_ratio=0.10)

    if name == "translate15":
        return _translate(img, max_ratio=0.15)

    if name == "translate":
        return _translate(img, max_ratio=0.08)

    return torch.clamp(img, 0, 1)


def random_attack(img, strength="medium"):
    img = img.float()

    if strength == "light":
        choices = [
            "noise", "jpeg", "blur", "brightness", "contrast",
            "resize", "zoom", "rotation", "translate"
        ]
        params = {
            "noise_sigma": 0.010,
            "jpeg_range": (72, 95),
            "blur_kernel": 3,
            "blur_sigma": (0.4, 0.8),
            "rotation_angle": random.uniform(-4, 4),
            "crop_ratio": random.uniform(0.92, 0.97),
            "resize_scale": random.uniform(0.88, 0.98),
            "zoom_scale": random.uniform(1.02, 1.08),
            "brightness_factor": random.uniform(0.96, 1.06),
            "contrast_factor": random.uniform(0.96, 1.06),
            "sp_prob": 0.003,
            "translate_ratio": 0.03,
        }

    elif strength == "medium":
        choices = [
            "noise", "jpeg", "blur", "rotation", "crop", "resize",
            "zoom", "brightness", "contrast", "salt_pepper", "translate"
        ]
        params = {
            "noise_sigma": 0.020,
            "jpeg_range": (45, 85),
            "blur_kernel": 5,
            "blur_sigma": (0.7, 1.4),
            "rotation_angle": random.uniform(-8, 8),
            "crop_ratio": random.uniform(0.82, 0.92),
            "resize_scale": random.uniform(0.75, 0.95),
            "zoom_scale": random.uniform(1.04, 1.16),
            "brightness_factor": random.uniform(0.90, 1.12),
            "contrast_factor": random.uniform(0.90, 1.12),
            "sp_prob": 0.008,
            "translate_ratio": 0.05,
        }

    elif strength == "strong":
        choices = [
            "noise", "jpeg", "blur", "rotation", "crop", "resize",
            "zoom", "brightness", "contrast", "salt_pepper",
            "sharpen", "translate", "combined"
        ]
        params = {
            "noise_sigma": 0.030,
            "jpeg_range": (25, 65),
            "blur_kernel": 7,
            "blur_sigma": (0.9, 1.8),
            "rotation_angle": random.uniform(-12, 12),
            "crop_ratio": random.uniform(0.70, 0.86),
            "resize_scale": random.uniform(0.60, 0.88),
            "zoom_scale": random.uniform(1.08, 1.25),
            "brightness_factor": random.uniform(0.82, 1.22),
            "contrast_factor": random.uniform(0.82, 1.22),
            "sp_prob": 0.015,
            "translate_ratio": 0.08,
        }

    else:
        raise ValueError(f"Unknown strength: {strength}")

    attack_type = random.choice(choices)

    if attack_type == "combined":
        if strength == "strong":
            out = _combo_attack(img, params, min_ops=2, max_ops=3, include_sharpen=True)
        else:
            out = _combo_attack(img, params, min_ops=2, max_ops=2, include_sharpen=False)
    else:
        out = _apply_attack_by_name(img, attack_type, params)

    return torch.clamp(out, 0, 1).float()


def mixed_attack_batch(img, epoch=0):
    attacked = []

    for i in range(img.size(0)):
        x = img[i:i + 1]

        if epoch < 5:
            if random.random() < 0.70:
                y = x
            else:
                y = random.choice([
                    targeted_attack(x, "jpeg50"),
                    targeted_attack(x, "blur5"),
                    targeted_attack(x, "noise003"),
                    targeted_attack(x, "brightness"),
                    targeted_attack(x, "contrast"),
                ])

        elif epoch < 20:
            if random.random() < 0.55:
                y = x
            elif random.random() < 0.75:
                y = random.choice([
                    targeted_attack(x, "jpeg50"),
                    targeted_attack(x, "blur5"),
                    targeted_attack(x, "noise003"),
                    targeted_attack(x, "brightness"),
                    targeted_attack(x, "contrast"),
                    targeted_attack(x, "translate"),
                ])
            else:
                y = random_attack(x, strength="light")

        elif epoch < 35:
            if random.random() < 0.75:
                y = random.choice([
                    targeted_attack(x, "jpeg50"),
                    targeted_attack(x, "blur5"),
                    targeted_attack(x, "noise003"),
                    targeted_attack(x, "brightness"),
                    targeted_attack(x, "contrast"),
                    targeted_attack(x, "translate"),
                ])
            else:
                y = random_attack(x, strength="light")

        elif epoch < 50:
            if random.random() < 0.65:
                y = random.choice([
                    targeted_attack(x, "jpeg30"),
                    targeted_attack(x, "rotate10"),
                    targeted_attack(x, "crop80"),
                    targeted_attack(x, "resize70"),
                    targeted_attack(x, "zoom120"),
                    targeted_attack(x, "noise005"),
                    targeted_attack(x, "translate"),
                ])
            else:
                y = random_attack(x, strength="medium")

        else:
            if random.random() < 0.55:
                y = random.choice([
                    targeted_attack(x, "jpeg30"),
                    targeted_attack(x, "rotate15"),
                    targeted_attack(x, "crop70"),
                    targeted_attack(x, "noise005"),
                    targeted_attack(x, "sp002"),
                    targeted_attack(x, "translate"),
                ])
            else:
                y = random_attack(x, strength="strong")

        attacked.append(torch.clamp(y, 0, 1))

    return torch.cat(attacked, dim=0)


def attack_suite(img):
    return {
        "clean": img,
        "jpeg50": targeted_attack(img, "jpeg50"),
        "jpeg30": targeted_attack(img, "jpeg30"),
        "blur5": targeted_attack(img, "blur5"),
        "rotate10": targeted_attack(img, "rotate10"),
        "rotate15": targeted_attack(img, "rotate15"),
        "resize70": targeted_attack(img, "resize70"),
        "zoom120": targeted_attack(img, "zoom120"),
        "crop80": targeted_attack(img, "crop80"),
        "crop70": targeted_attack(img, "crop70"),
        "noise003": targeted_attack(img, "noise003"),
        "noise005": targeted_attack(img, "noise005"),
        "sp002": targeted_attack(img, "sp002"),
        "contrast": targeted_attack(img, "contrast"),
        "brightness": targeted_attack(img, "brightness"),
        "translate": targeted_attack(img, "translate"),
    }