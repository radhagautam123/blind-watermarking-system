import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from models.losses import watermark_loss, image_loss
from models.attack_layer import random_attack
from utils.ecc import encrypt_watermark_bits
from utils.error_correction import encode_bch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", DEVICE)

BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4

SAVE_DIR = "/kaggle/working/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint.pth")

import hashlib
import random
import string

def random_key():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=16))

def key_to_tensor(key, size=32):
    hash_bytes = b''
    current = key.encode()

    while len(hash_bytes) < size * size:
        current = hashlib.sha256(current).digest()
        hash_bytes += current

    arr = np.frombuffer(hash_bytes[:size*size], dtype=np.uint8)
    arr = arr.astype(np.float32) / 255.0

    return torch.tensor(arr).view(1, 1, size, size)

class DIV2KDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.images = [f for f in os.listdir(root)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.images[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.images)

DATA_PATH = "/kaggle/input/datasets/soumikrakshit/div2k-high-resolution-images/DIV2K_train_HR/DIV2K_train_HR"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = DIV2KDataset(DATA_PATH, transform=transform)
print("Dataset size:", len(dataset))

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

encoder = EncoderNet().to(DEVICE)
decoder = DecoderNet().to(DEVICE)

optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=LR
)

start_epoch = 0

# ---------------- RESUME ----------------
if os.path.exists(CHECKPOINT_PATH):
    print("🔄 Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = checkpoint['epoch'] + 1
    print("✅ Resumed from epoch", start_epoch)

# ---------------- TRAIN ----------------
for epoch in range(start_epoch, EPOCHS):

    encoder.train()
    decoder.train()

    print("Epoch {}/{}".format(epoch+1, EPOCHS))

    total_loss = 0

    for imgs, _ in tqdm(loader):
        imgs = imgs.to(DEVICE)

        wm_np = np.random.randint(0, 2, (imgs.size(0), 32, 32)).astype(np.uint8)

        key_tensors = []
        wm_secure_list = []

        for i in range(imgs.size(0)):
            key = random_key()

            key_tensor = key_to_tensor(key)
            key_tensors.append(key_tensor)

            wm_encoded = encode_bch(wm_np[i])
            wm_secure = encrypt_watermark_bits(wm_encoded, key)

            wm_secure_list.append(wm_secure)

        key_tensor = torch.cat(key_tensors, dim=0).to(DEVICE)
        wm_secure = np.stack(wm_secure_list)

        wm = torch.from_numpy(wm_secure).float().unsqueeze(1).to(DEVICE)

        residual = encoder(imgs, wm, key_tensor)
        watermarked = torch.clamp(imgs + residual, 0, 1)

        attacked = random_attack(watermarked).float()

        pred = decoder(attacked, key_tensor)

        wm_loss = watermark_loss(pred, wm)
        img_loss = image_loss(imgs, watermarked)

        loss = 8.0 * wm_loss + 1.0 * img_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch Loss:", total_loss / len(loader))

    # -------- SAVE CHECKPOINT --------
    torch.save({
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict()
    }, CHECKPOINT_PATH)

    # -------- ALSO SAVE FINAL WEIGHTS --------
    torch.save(encoder.state_dict(), os.path.join(SAVE_DIR, "encoder.pth"))
    torch.save(decoder.state_dict(), os.path.join(SAVE_DIR, "decoder.pth"))
