import cv2
import numpy as np
import torch

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from models.attack_layer import targeted_attack
from utils.metrics import normalized_correlation
from utils.wm_pipeline import (
    decode_from_logits,
    encode_for_embedding,
    extract_with_crop_search,
    key_to_tensor,
    prepare_plain_watermark,
    wm_secure_to_tensor,
)

DEVICE = 'cpu'
WM_SIZE = 32
KEY = 'secure_key'
ck = torch.load('checkpoints/checkpoint_epoch_25.pth', map_location=DEVICE, weights_only=False)
encoder = EncoderNet(wm_size=WM_SIZE).to(DEVICE)
decoder = DecoderNet(wm_size=WM_SIZE).to(DEVICE)
encoder.load_state_dict(ck['encoder'], strict=True)
decoder.load_state_dict(ck['decoder'], strict=True)
encoder.eval(); decoder.eval()

img = cv2.imread('data/host_images/dog.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

wm = cv2.imread('data/watermarks/apple_logo.png', cv2.IMREAD_GRAYSCALE)
wm = cv2.resize(wm, (32, 32))
_, wm = cv2.threshold(wm, 127, 1, cv2.THRESH_BINARY)
wm = prepare_plain_watermark(wm)

wm_secure = encode_for_embedding(wm, KEY)
wm_t = wm_secure_to_tensor(wm_secure, device=DEVICE)
key_t = key_to_tensor(KEY, device=DEVICE)

with torch.no_grad():
    residual = encoder(img, wm_t, key_t)
    watermarked = torch.clamp(img + residual, 0, 1)

for name in ['clean', 'crop80', 'crop70', 'translate', 'jpeg30']:
    if name == 'clean':
        x = watermarked
    else:
        x = targeted_attack(watermarked.clone(), name)
    if name in ('crop80', 'crop70', 'translate'):
        _, pred, out, _ = extract_with_crop_search(decoder, x, KEY, key_t, use_decrypt=True, use_ecc=True)
    else:
        pred = decoder(x, key_t)
        out = decode_from_logits(pred, KEY, use_decrypt=True, use_ecc=True)
    ncc = normalized_correlation(wm, out['final_bits'])
    ber = (wm != out['final_bits']).mean()
    print(f'{name:>8s} NCC={ncc:.4f} BER={ber:.4f}')
