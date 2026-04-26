import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from models.losses import image_loss, watermark_loss
from models.attack_layer import random_attack
from utils.security import secure_encode, secure_decode


# ================= CONFIG =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4

EMBED_STRENGTH = 0.05
KEY = "secure_train_key"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================= DATA =================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("data", transform=transform)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0   # CPU safe
)

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
            secure_encode(wm_np[i], KEY)
            for i in range(len(wm_np))
        ])

        wm = torch.tensor(wm_secure).float().unsqueeze(1).to(DEVICE)

        # -------- ENCODER --------
        residual = encoder(imgs, wm)
        watermarked = torch.clamp(imgs + EMBED_STRENGTH * residual, 0, 1)

        # -------- ATTACK SCHEDULING --------
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

    print("Train Loss:", round(total_loss / len(loader), 4))

    # ================= VALIDATION =================
    encoder.eval()
    decoder.eval()

    correct = 0
    total = 0
    wrong_correct = 0

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)

            wm_np = np.random.randint(0, 2, (imgs.size(0), 32, 32))

            wm_secure = np.stack([
                secure_encode(wm_np[i], KEY)
                for i in range(len(wm_np))
            ])

            wm = torch.tensor(wm_secure).float().unsqueeze(1).to(DEVICE)

            residual = encoder(imgs, wm)
            watermarked = torch.clamp(imgs + EMBED_STRENGTH * residual, 0, 1)

            pred = decoder(watermarked)
            pred_bin = (torch.sigmoid(pred) > 0.5).float().cpu().numpy()

            for i in range(len(pred_bin)):
                decoded = secure_decode(pred_bin[i, 0], KEY)
                true = secure_decode(wm_np[i], KEY)

                correct += (decoded == true).sum()
                total += true.size

                wrong = secure_decode(pred_bin[i, 0], "wrong_key")
                wrong_correct += (wrong == true).sum()

    acc = correct / total
    wrong_acc = wrong_correct / total

    print(f"Val Acc: {acc:.4f} | Wrong Key Acc: {wrong_acc:.4f}")

    # ================= SAVE =================
    torch.save({
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict()
    }, ckpt_path)

    torch.save(encoder.state_dict(), os.path.join(CHECKPOINT_DIR, "encoder_latest.pth"))
    torch.save(decoder.state_dict(), os.path.join(CHECKPOINT_DIR, "decoder_latest.pth"))