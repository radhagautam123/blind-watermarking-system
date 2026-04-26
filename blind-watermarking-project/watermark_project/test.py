import torch
import numpy as np
import cv2
import os

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from utils.security import secure_encode, secure_decode

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ENC_PATH = "checkpoints/encoder_latest.pth"
DEC_PATH = "checkpoints/decoder_latest.pth"

encoder = EncoderNet().to(DEVICE)
decoder = DecoderNet().to(DEVICE)

encoder.load_state_dict(torch.load(ENC_PATH, map_location=DEVICE))
decoder.load_state_dict(torch.load(DEC_PATH, map_location=DEVICE))

encoder.eval()
decoder.eval()

img = cv2.imread("data/host_images/dog.png")
img = cv2.resize(img, (256,256)) / 255.0
img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

wm_np = np.random.randint(0,2,(1,32,32))
wm_secure = np.stack([secure_encode(wm_np[i],"key") for i in range(1)])
wm = torch.tensor(wm_secure).float().unsqueeze(1).to(DEVICE)

with torch.no_grad():
    residual = encoder(img, wm)
    watermarked = torch.clamp(img + 0.05 * residual,0,1)
    pred = decoder(watermarked)

pred_bin = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()

decoded = secure_decode(pred_bin[0,0], "key")

os.makedirs("results", exist_ok=True)
cv2.imwrite("results/extracted.png", decoded*255)