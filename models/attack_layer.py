import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import cv2
import numpy as np


def random_attack(img):

    device = img.device
    img = img.float()

    attack_type = random.choice([
        "noise",
        "jpeg",
        "blur",
        "rotation",
        "crop",
        "resize",
        "brightness",
        "contrast",
        "salt_pepper",
        "combined"
    ])

    # -------- NOISE --------
    if attack_type == "noise":
        noise = torch.randn_like(img) * 0.05
        img = torch.clamp(img + noise, 0, 1)

    # -------- JPEG --------
    elif attack_type == "jpeg":
        imgs = []
        for i in range(img.size(0)):
            np_img = img[i].detach().permute(1,2,0).cpu().numpy()
            np_img = (np_img * 255).astype(np.uint8)

            _, enc = cv2.imencode(
                '.jpg',
                np_img,
                [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 70)]
            )
            dec = cv2.imdecode(enc, 1)

            dec = torch.tensor(dec / 255.0, dtype=torch.float32).permute(2,0,1)
            imgs.append(dec)

        img = torch.stack(imgs)

    # -------- BLUR --------
    elif attack_type == "blur":
        img = TF.gaussian_blur(img, kernel_size=5)

    # -------- ROTATION --------
    elif attack_type == "rotation":
        img = TF.rotate(img, random.uniform(-15, 15))

    # -------- CROP --------
    elif attack_type == "crop":
        _, _, h, w = img.shape
        crop_size = int(h * random.uniform(0.6, 0.9))
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        cropped = img[:, :, i:i+crop_size, j:j+crop_size]
        img = F.interpolate(cropped, size=(h, w))

    # -------- RESIZE --------
    elif attack_type == "resize":
        scale = random.uniform(0.5, 1.5)
        new_size = int(img.shape[2] * scale)
        resized = F.interpolate(img, size=(new_size, new_size))
        img = F.interpolate(resized, size=(img.shape[2], img.shape[3]))

    # -------- BRIGHTNESS --------
    elif attack_type == "brightness":
        img = torch.clamp(img * random.uniform(0.6, 1.4), 0, 1)

    # -------- CONTRAST --------
    elif attack_type == "contrast":
        mean = img.mean()
        img = torch.clamp((img - mean) * random.uniform(0.6, 1.4) + mean, 0, 1)

    # -------- SALT & PEPPER --------
    elif attack_type == "salt_pepper":
        noise = torch.rand_like(img)
        img = img.clone()
        img[noise < 0.02] = 0
        img[noise > 0.98] = 1

    # -------- COMBINED --------
    elif attack_type == "combined":
        img = random_attack(img)
        img = random_attack(img)

    return img.to(device).float()
