import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from models.attack_layer import (
    add_gaussian_noise,
    blur3x3,
    resize_restore,
    rotate_tensor,
    random_crop
)
from models.losses import watermark_loss
from utils.security import secure_encode, secure_decode


# ================= CONFIG =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 35
BATCH_SIZE = 4
LR = 2e-4
KEY = "secure_train_key"
WRONG_KEY = "wrong_key_123"

DATA_PATH = "/content/drive/MyDrive/blind-watermarking-system/data/DIV2K_train_HR"
CHECKPOINT_DIR = "/content/drive/MyDrive/blind-watermarking-system/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EMBED_STRENGTH = 0.20
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "training_state.pth")


# ================= DATASET =================
class ImageDataset(Dataset):
    def __init__(self, path):
        self.files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith('.png')
        ]

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)


dataset = ImageDataset(DATA_PATH)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# ================= MODEL =================
encoder = EncoderNet().to(DEVICE)
decoder = DecoderNet().to(DEVICE)

optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=LR
)

scaler = torch.amp.GradScaler('cuda')

start_epoch = 0

# ================= LOAD CHECKPOINT =================
if os.path.exists(CHECKPOINT_PATH):
    print("🔄 Resuming from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    start_epoch = checkpoint['epoch'] + 1


# ================= TRAIN =================
for epoch in range(start_epoch, EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    encoder.train()
    decoder.train()

    total_loss = 0

    # 🔥 FIXED ATTACK PER EPOCH
    if epoch < 5:
        epoch_attack = "none"
    elif epoch < 15:
        epoch_attack = random.choice(["noise", "blur", "resize"])
    else:
        epoch_attack = random.choice(["rotate", "crop"])

    print(f"Using attack: {epoch_attack}")

    for x in tqdm(train_loader):

        x = x.to(DEVICE)

        wm_np = np.random.randint(0, 2, (x.size(0), 32, 32))

        # 🔐 SECURITY PHASE
        if epoch < 5:
            wm_secure = wm_np
        else:
            wm_secure = np.stack([
                secure_encode(wm_np[i], KEY)
                for i in range(len(wm_np))
            ])

        wm = torch.tensor(wm_secure).float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):

            residual = encoder(x, wm)
            watermarked = torch.clamp(x + EMBED_STRENGTH * residual, 0, 1)

            # 🔥 APPLY FIXED ATTACK
            if epoch_attack == "none":
                attacked = watermarked

            elif epoch_attack == "noise":
                attacked = add_gaussian_noise(watermarked, 0.02)

            elif epoch_attack == "blur":
                attacked = blur3x3(watermarked)

            elif epoch_attack == "resize":
                attacked = resize_restore(watermarked, 0.85)

            elif epoch_attack == "rotate":
                attacked = rotate_tensor(
                    watermarked,
                    random.uniform(-10, 10) * math.pi / 180
                )

            elif epoch_attack == "crop":
                attacked = random_crop(watermarked, 0.85)

            pred_clean = decoder(watermarked)
            pred_attack = decoder(attacked)

            loss_clean = watermark_loss(pred_clean, wm)
            loss_attack = watermark_loss(pred_attack, wm)
            loss_img = F.mse_loss(watermarked, x)

            loss_consistency = F.mse_loss(
                torch.sigmoid(pred_clean),
                torch.sigmoid(pred_attack)
            )

            if epoch < 5:
                loss = 5.0 * loss_clean + 1.0 * loss_img
            elif epoch < 15:
                loss = 4.0 * loss_clean + 2.0 * loss_attack + 0.5 * loss_img
            else:
                loss = 2.5 * loss_clean + 3.5 * loss_attack + 0.5 * loss_img + 1.0 * loss_consistency

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Train Loss: {total_loss/len(train_loader):.4f}")

    # ================= VALIDATION =================
    encoder.eval()
    decoder.eval()

    val_acc_total = 0
    bit_error_total = 0
    wrong_acc_total = 0

    with torch.no_grad():
        for x in val_loader:

            x = x.to(DEVICE)

            wm_np = np.random.randint(0, 2, (x.size(0), 32, 32))

            if epoch < 5:
                wm_secure = wm_np
            else:
                wm_secure = np.stack([
                    secure_encode(wm_np[i], KEY)
                    for i in range(len(wm_np))
                ])

            wm = torch.tensor(wm_secure).float().unsqueeze(1).to(DEVICE)

            residual = encoder(x, wm)
            watermarked = torch.clamp(x + EMBED_STRENGTH * residual, 0, 1)

            if epoch < 5:
                attacked = watermarked
            else:
                attacked = add_gaussian_noise(watermarked, 0.03)

            pred = decoder(attacked)

            pred_bin = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()
            wm_bin = wm.cpu().numpy()

            decoded_pred = []
            decoded_gt = []
            wrong_decoded = []

            for i in range(pred_bin.shape[0]):
                decoded_pred.append(secure_decode(pred_bin[i, 0], KEY))
                decoded_gt.append(secure_decode(wm_bin[i, 0], KEY))
                wrong_decoded.append(secure_decode(pred_bin[i, 0], WRONG_KEY))

            decoded_pred = np.array(decoded_pred)
            decoded_gt = np.array(decoded_gt)
            wrong_decoded = np.array(wrong_decoded)

            acc = (decoded_pred == decoded_gt).mean()
            bit_error = (decoded_pred != decoded_gt).mean()
            wrong_acc = (wrong_decoded == decoded_gt).mean()

            val_acc_total += acc
            bit_error_total += bit_error
            wrong_acc_total += wrong_acc

    print(f"Val Acc: {val_acc_total/len(val_loader):.4f} | Bit Error: {bit_error_total/len(val_loader):.4f} | Wrong Key Acc: {wrong_acc_total/len(val_loader):.4f}")

    # ================= SAVE CHECKPOINT =================
    torch.save({
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict()
    }, CHECKPOINT_PATH)