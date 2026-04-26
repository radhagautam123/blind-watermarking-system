import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from models.losses import image_loss, watermark_loss
from models.attack_layer import random_attack
from utils.ecc import encrypt_watermark_bits, decrypt_watermark_bits


# ================= CONFIG =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", DEVICE)

BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4

EMBED_STRENGTH = 0.05
KEY = "secure_train_key"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DATA_PATH = "/content/data"   # <-- use this


# ================= DATA =================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATA_PATH, transform=transform)

print("Dataset size:", len(dataset))
print("Classes:", dataset.classes)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ================= MODELS =================
encoder = EncoderNet().to(DEVICE)
decoder = DecoderNet().to(DEVICE)

optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=LR
)


# ================= RESUME =================
start_epoch = 0
ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pth")

if os.path.exists(ckpt_path):
    print("🔄 Resuming training...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1


# ================= TRAIN =================
for epoch in range(start_epoch, EPOCHS):

    encoder.train()
    decoder.train()

    total_loss = 0
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for imgs, _ in tqdm(loader):
        imgs = imgs.to(DEVICE)

        # -------- WATERMARK --------
        wm_np = np.random.randint(0, 2, (imgs.size(0), 32, 32))

        wm_secure = np.stack([
            encrypt_watermark_bits(wm_np[i], KEY).reshape(32, 32)
            for i in range(len(wm_np))
        ])

        wm = torch.tensor(wm_secure).float().unsqueeze(1).to(DEVICE)

        # -------- EMBEDDING --------
        residual = encoder(imgs, wm)
        watermarked = torch.clamp(imgs + EMBED_STRENGTH * residual, 0, 1)

        # -------- ATTACK --------
        if epoch < 5:
            attacked = watermarked
        elif epoch < 10:
            attacked = random_attack(watermarked, epoch=10)
        else:
            attacked = random_attack(watermarked, epoch=25)

        # -------- DECODER --------
        pred = decoder(attacked)

        # -------- LOSS --------
        wm_loss = watermark_loss(pred, wm)
        img_loss = image_loss(imgs, watermarked)

        loss = wm_loss + 2.5 * img_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Train Loss:", round(total_loss / max(len(loader), 1), 4))


    # ================= VALIDATION =================
    encoder.eval()
    decoder.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)

            wm_np = np.random.randint(0, 2, (imgs.size(0), 32, 32))

            wm_secure = np.stack([
                encrypt_watermark_bits(wm_np[i], KEY).reshape(32, 32)
                for i in range(len(wm_np))
            ])

            wm = torch.tensor(wm_secure).float().unsqueeze(1).to(DEVICE)

            residual = encoder(imgs, wm)
            watermarked = torch.clamp(imgs + EMBED_STRENGTH * residual, 0, 1)

            attacked = random_attack(watermarked, epoch=15)

            pred = decoder(attacked)
            pred_bin = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()

            for i in range(len(pred_bin)):
                decoded = decrypt_watermark_bits(pred_bin[i, 0], KEY, (32, 32))
                true = wm_np[i]

                correct += (decoded == true).sum()
                total += true.size

    acc = correct / total
    print(f"Val Acc: {acc:.4f}")


    # ================= SAVE =================
    torch.save({
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict()
    }, ckpt_path)

    torch.save(encoder.state_dict(), os.path.join(CHECKPOINT_DIR, "encoder_latest.pth"))
    torch.save(decoder.state_dict(), os.path.join(CHECKPOINT_DIR, "decoder_latest.pth"))

    print("💾 Saved checkpoint")