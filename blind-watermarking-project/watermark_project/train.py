from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
import torchvision.models as models

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from models.losses import image_loss, watermark_loss
from models.attack_layer import random_attack

# ================= CONFIG =================
IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100

DATASET_PATH = '/content/drive/MyDrive/DIV2K_train_HR'
WEIGHTS_DIR = '/content/drive/MyDrive/watermark_project/weights'
CHECKPOINT_PATH = '/content/drive/MyDrive/watermark_project/checkpoints/latest.pth'

# ================= DATASET =================
class DIV2KDataset(Dataset):
    def __init__(self, root):
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)

# ================= PERCEPTUAL LOSS =================
class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()

    def forward(self, x, y):
        return F.mse_loss(self.vgg(x), self.vgg(y))

# ================= LOAD CHECKPOINT =================
def load_checkpoint(encoder, decoder, optimizer, scheduler):
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint found.")
        return 0, float('inf')

    checkpoint = torch.load(CHECKPOINT_PATH)

    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])

    print("Checkpoint loaded.")
    return checkpoint['epoch'] + 1, checkpoint.get('best_loss', float('inf'))

# ================= TRAIN =================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using:", device)

    encoder = EncoderNet().to(device)
    decoder = DecoderNet().to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    scaler = torch.cuda.amp.GradScaler()
    perceptual_loss_fn = PerceptualLoss().to(device)

    dataset = DIV2KDataset(DATASET_PATH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)

    start_epoch, best_loss = load_checkpoint(encoder, decoder, optimizer, scheduler)

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        total_loss = 0

        for x in loader:
            x = x.to(device, non_blocking=True)

            wm = torch.randint(0, 2, (x.size(0), 1, 32, 32), device=device).float()

            with torch.cuda.amp.autocast():
                watermarked = encoder(x, wm)
                attacked = random_attack(watermarked, epoch)
                pred = decoder(attacked)

                loss_img = image_loss(x, watermarked)
                loss_wm = watermark_loss(pred, wm)
                loss_perc = perceptual_loss_fn(watermarked, x)

                corrected = decoder.stn(attacked)
                stn_reg = torch.mean((attacked - corrected) ** 2)

                loss = (
                    1.5 * loss_img +
                    1.0 * loss_wm +
                    0.3 * loss_perc +
                    0.1 * stn_reg
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        scheduler.step(avg_loss)

        # SAVE CHECKPOINT
        torch.save({
            'epoch': epoch,
            'encoder_state': encoder.state_dict(),
            'decoder_state': decoder.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_loss': best_loss
        }, CHECKPOINT_PATH)

        # SAVE BEST MODEL
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), f"{WEIGHTS_DIR}/best_encoder.pth")
            torch.save(decoder.state_dict(), f"{WEIGHTS_DIR}/best_decoder.pth")
            print("✅ Best model saved")

    print("🎉 Training Completed")

if __name__ == "__main__":
    main()