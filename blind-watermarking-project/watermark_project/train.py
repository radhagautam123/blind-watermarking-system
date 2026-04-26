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
from utils.security import secure_encode

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4
EMBED_STRENGTH = 0.05
KEY = "secure_train_key"

CHECKPOINT_DIR = "checkpoints"
DRIVE_DIR = "/content/drive/MyDrive/watermark_weights"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DRIVE_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("data", transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

encoder = EncoderNet().to(DEVICE)
decoder = DecoderNet().to(DEVICE)

optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=LR
)

start_epoch = 0
ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pth")

if os.path.exists(ckpt_path):
    print("🔄 Resuming training...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1

for epoch in range(start_epoch, EPOCHS):
    encoder.train()
    decoder.train()

    total_loss = 0
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for imgs, _ in tqdm(loader):
        imgs = imgs.to(DEVICE)

        wm_np = np.random.randint(0, 2, (imgs.size(0), 32, 32))
        wm = torch.tensor(wm_np).float().unsqueeze(1).to(DEVICE)

        wm_secure = torch.tensor(
            np.stack([secure_encode(wm_np[i], KEY) for i in range(len(wm_np))])
        ).float().unsqueeze(1).to(DEVICE)

        residual = encoder(imgs, wm_secure)
        watermarked = torch.clamp(imgs + EMBED_STRENGTH * residual, 0, 1)

        attacked = random_attack(watermarked, epoch=epoch)

        pred = decoder(attacked)

        loss = watermark_loss(pred, wm_secure) + 2.5 * image_loss(imgs, watermarked)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Train Loss:", total_loss / len(loader))

    torch.save({
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict()
    }, ckpt_path)

    torch.save(encoder.state_dict(), os.path.join(CHECKPOINT_DIR, "encoder_latest.pth"))
    torch.save(decoder.state_dict(), os.path.join(CHECKPOINT_DIR, "decoder_latest.pth"))

    # SAVE TO DRIVE (IMPORTANT)
    torch.save(encoder.state_dict(), os.path.join(DRIVE_DIR, "encoder_latest.pth"))
    torch.save(decoder.state_dict(), os.path.join(DRIVE_DIR, "decoder_latest.pth"))