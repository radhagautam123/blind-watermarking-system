import os
import sys
from pathlib import Path

# Ensure project root is on sys.path (required for Kaggle / non-package runs).
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import string

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from models.losses import watermark_loss, wrong_key_loss, image_loss, ber_from_logits
from models.attack_layer import mixed_attack_batch
from utils.wm_pipeline import encode_for_embedding, key_to_tensor, decode_from_logits, prepare_plain_watermark


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

SEED = 42
BATCH_SIZE = 1
EPOCHS = 70
LR = 1e-4
NUM_WORKERS = 1
VAL_RATIO = 0.1

WM_SIZE = 32
KEY_POOL_SIZE = 64
ARCH_VERSION = "attn_v4_extraction_robust_eccfix_v1"

ATTACK_START_EPOCH = 3
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 5e-4
EARLY_STOP_MIN_EPOCH = 35
SAVE_EVERY = 5

IS_KAGGLE = os.path.exists("/kaggle")
DEFAULT_KAGGLE_DATA = "/kaggle/input/datasets/soumikrakshit/div2k-high-resolution-images/DIV2K_train_HR/DIV2K_train_HR"
DEFAULT_LOCAL_DATA = "./data/DIV2K_train_HR"

DATA_PATH = os.environ.get("WM_DATA_PATH", DEFAULT_KAGGLE_DATA if IS_KAGGLE else DEFAULT_LOCAL_DATA)
SAVE_DIR = os.environ.get("WM_SAVE_DIR", "/kaggle/working/watermark_project/models" if IS_KAGGLE else "./checkpoints")

os.makedirs(SAVE_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_latest.pth")
BEST_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_best.pth")
BEST_BER_PATH = os.path.join(SAVE_DIR, "checkpoint_best_ber.pth")
BEST_PSNR_PATH = os.path.join(SAVE_DIR, "checkpoint_best_psnr.pth")
BEST_CLEAN_BER_PATH = os.path.join(SAVE_DIR, "checkpoint_best_clean_ber.pth")
BEST_ATTACK_BER_PATH = os.path.join(SAVE_DIR, "checkpoint_best_attack_ber.pth")
BEST_ENCODER_PATH = os.path.join(SAVE_DIR, "best_encoder.pth")
BEST_DECODER_PATH = os.path.join(SAVE_DIR, "best_decoder.pth")

RESUME = True
RESUME_PATH = CHECKPOINT_PATH

print("Data path:", DATA_PATH)
print("Save dir :", SAVE_DIR)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


def random_key():
    return "".join(random.choices(string.ascii_letters + string.digits, k=16))


def compute_psnr_torch(x, y, eps=1e-8):
    mse = F.mse_loss(x, y)
    return (10.0 * torch.log10(1.0 / (mse + eps))).item()


class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.num_bad_epochs = 0

    def load_state(self, best, num_bad_epochs):
        self.best = best
        self.num_bad_epochs = num_bad_epochs

    def state_dict(self):
        return {
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }

    def step(self, current):
        if current < self.best - self.min_delta:
            self.best = current
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


class DIV2KDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.root_path = Path(root)
        if not self.root_path.exists():
            raise FileNotFoundError(f"DATA_PATH does not exist: {root}")
        self.images = sorted([f for f in os.listdir(root) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in: {root}")
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.images[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, idx

    def __len__(self):
        return len(self.images)


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = DIV2KDataset(DATA_PATH, transform=transform)
print("Dataset size:", len(dataset))

val_size = max(1, int(len(dataset) * VAL_RATIO))
train_size = len(dataset) - val_size

split_gen = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=split_gen
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

encoder = EncoderNet(wm_size=WM_SIZE).to(DEVICE)
decoder = DecoderNet(wm_size=WM_SIZE).to(DEVICE)

optimizer = optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=LR,
    weight_decay=1e-5
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
KEY_POOL = [random_key() for _ in range(KEY_POOL_SIZE)]


def get_loss_weights(epoch):
    if epoch < 5:
        return {"wm": 3.0, "img": 0.9, "key": 0.10}
    elif epoch < 20:
        return {"wm": 4.0, "img": 0.7, "key": 0.15}
    elif epoch < 40:
        return {"wm": 5.0, "img": 0.55, "key": 0.20}
    else:
        return {"wm": 5.5, "img": 0.45, "key": 0.25}


def sample_training_watermark(batch_size, epoch, rng=None):
    """Mix random bits with sparse structured patterns (logo-like density)."""
    if rng is None:
        rng = np.random.default_rng()

    wm_list = []
    for _ in range(batch_size):
        if rng.random() < 0.35:
            density = float(rng.uniform(0.08, 0.35))
            wm = (rng.random((WM_SIZE, WM_SIZE)) < density).astype(np.uint8)
        else:
            wm = rng.integers(0, 2, (WM_SIZE, WM_SIZE), dtype=np.uint8)
        wm_list.append(wm)
    return np.stack(wm_list, axis=0)


def build_batch_payload(batch_size, epoch_seed=0, fixed=False, epoch=0):
    rng = np.random.default_rng(epoch_seed if fixed else None)

    if fixed:
        wm_plain_np = rng.integers(0, 2, (batch_size, WM_SIZE, WM_SIZE), dtype=np.uint8)
    else:
        wm_plain_np = sample_training_watermark(batch_size, epoch, rng=rng)

    key_tensors = []
    wrong_key_tensors = []
    wm_secure_list = []
    key_strings = []

    for i in range(batch_size):
        key = KEY_POOL[i % len(KEY_POOL)] if fixed else random.choice(KEY_POOL)
        wrong_key = random.choice(KEY_POOL)
        while wrong_key == key:
            wrong_key = random.choice(KEY_POOL)

        key_strings.append(key)
        key_tensors.append(key_to_tensor(key, size=WM_SIZE, device=DEVICE))
        wrong_key_tensors.append(key_to_tensor(wrong_key, size=WM_SIZE, device=DEVICE))

        wm_plain = prepare_plain_watermark(wm_plain_np[i])
        wm_secure = encode_for_embedding(wm_plain, key)
        wm_secure_list.append(wm_secure)

    key_tensor = torch.cat(key_tensors, dim=0)
    wrong_key_tensor = torch.cat(wrong_key_tensors, dim=0)

    wm_secure_np = np.stack(wm_secure_list)
    wm_target_bits = torch.from_numpy(wm_secure_np).float().unsqueeze(1).to(DEVICE, non_blocking=True)
    wm_plain = torch.from_numpy(wm_plain_np).float().unsqueeze(1).to(DEVICE, non_blocking=True)

    return wm_target_bits, wm_plain, key_tensor, wrong_key_tensor, key_strings


def apply_epoch_attack(x, epoch):
    if epoch < ATTACK_START_EPOCH:
        return x
    return mixed_attack_batch(x, epoch=epoch).float()


def compute_combined_loss(imgs, watermarked, pred_clean, pred_attacked, wm_target_bits, wrong_pred, weights):
    clean_wm_l = watermark_loss(pred_clean, wm_target_bits)
    attacked_wm_l = watermark_loss(pred_attacked, wm_target_bits)

    if weights["attack_mix"] <= 0.0:
        wm_l = clean_wm_l
    else:
        wm_l = (1.0 - weights["attack_mix"]) * clean_wm_l + weights["attack_mix"] * attacked_wm_l

    img_l = image_loss(imgs, watermarked)

    if wrong_pred is None:
        key_l = torch.tensor(0.0, device=imgs.device, dtype=imgs.dtype)
    else:
        key_l = wrong_key_loss(wrong_pred)

    total = (
        weights["wm"] * wm_l +
        weights["img"] * img_l +
        weights["key"] * key_l
    )
    return total, wm_l, img_l, key_l


def get_rng_state():
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_random" in state:
        try:
            tr = state["torch_random"]
            if not isinstance(tr, torch.ByteTensor):
                tr = torch.tensor(tr, dtype=torch.uint8)
            torch.set_rng_state(tr)
        except Exception:
            # If legacy/foreign RNG format, skip restoring PyTorch RNG.
            pass
    if torch.cuda.is_available() and "torch_cuda_random" in state:
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda_random"])
        except Exception:
            # Skip restoring CUDA RNG if format is incompatible.
            pass


def save_checkpoint(path, epoch, best_score, best_ber, best_psnr, best_clean_ber, best_attack_ber,
                    encoder, decoder, optimizer, scheduler, scaler, val_stats, early_stopper):
    checkpoint = {
        "arch_version": ARCH_VERSION,
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_score": best_score,
        "best_ber": best_ber,
        "best_psnr": best_psnr,
        "best_clean_ber": best_clean_ber,
        "best_attack_ber": best_attack_ber,
        "val_psnr": val_stats["psnr"],
        "val_loss": val_stats["loss"],
        "val_ber_clean": val_stats["ber_clean"],
        "val_ber_attack": val_stats["ber_attack"],
        "early_stopper": early_stopper.state_dict(),
        "rng_state": get_rng_state(),
        "seed": SEED,
        "attack_start_epoch": ATTACK_START_EPOCH,
    }
    torch.save(checkpoint, path)


def _torch_load(path):
    """Load training checkpoint (includes optimizer/RNG numpy state — not weights-only)."""
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def try_resume(path, encoder, decoder, optimizer, scheduler, scaler, early_stopper):
    if not os.path.exists(path):
        print("No checkpoint found to resume. Starting fresh.")
        return 0, -1e9, float("inf"), 0.0, float("inf"), float("inf")

    checkpoint = _torch_load(path)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    if "early_stopper" in checkpoint and checkpoint["early_stopper"] is not None:
        es = checkpoint["early_stopper"]
        early_stopper.load_state(
            best=es.get("best", float("inf")),
            num_bad_epochs=es.get("num_bad_epochs", 0),
        )

    if "rng_state" in checkpoint:
        set_rng_state(checkpoint["rng_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_score = checkpoint.get("best_score", -1e9)
    best_ber = checkpoint.get("best_ber", float("inf"))
    best_psnr = checkpoint.get("best_psnr", 0.0)
    best_clean_ber = checkpoint.get("best_clean_ber", float("inf"))
    best_attack_ber = checkpoint.get("best_attack_ber", float("inf"))

    print(f"Resumed from: {path}")
    print(f"Start epoch  : {start_epoch}")
    print(f"Best score   : {best_score:.4f}")
    print(f"Best BER     : {best_ber:.4f}")
    print(f"Best clean   : {best_clean_ber:.4f}")
    print(f"Best attack  : {best_attack_ber:.4f}")
    print(f"Best PSNR    : {best_psnr:.2f}")

    return start_epoch, best_score, best_ber, best_psnr, best_clean_ber, best_attack_ber


@torch.no_grad()
def evaluate(encoder, decoder, loader, epoch):
    encoder.eval()
    decoder.eval()

    total_loss_val = 0.0
    total_wm_loss = 0.0
    total_img_loss = 0.0
    total_key_loss = 0.0
    total_psnr = 0.0
    total_ber_clean = 0.0
    total_ber_attack = 0.0
    total_wrong_conf = 0.0
    total_batches = 0

    weights = get_loss_weights(epoch)
    attack_mix = 0.0 if epoch < ATTACK_START_EPOCH else 0.75
    weights = {**weights, "attack_mix": attack_mix}

    total_plain_ber = 0.0

    for batch_idx, (imgs, _) in enumerate(tqdm(loader, leave=False)):
        imgs = imgs.to(DEVICE, non_blocking=True)

        wm_target_bits, wm_plain, key_tensor, wrong_key_tensor, key_strings = build_batch_payload(
            imgs.size(0), epoch_seed=epoch * 1000 + batch_idx, fixed=True, epoch=epoch
        )

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            residual = encoder(imgs, wm_target_bits, key_tensor)
            watermarked = torch.clamp(imgs + residual, 0, 1)
            attacked = apply_epoch_attack(watermarked, epoch)

            pred_clean = decoder(watermarked, key_tensor)
            pred_attacked = decoder(attacked, key_tensor)
            wrong_pred = decoder(attacked, wrong_key_tensor)

            loss, wm_l, img_l, key_l = compute_combined_loss(
                imgs=imgs,
                watermarked=watermarked,
                pred_clean=pred_clean,
                pred_attacked=pred_attacked,
                wm_target_bits=wm_target_bits,
                wrong_pred=wrong_pred,
                weights=weights
            )

        total_loss_val += loss.item()
        total_wm_loss += wm_l.item()
        total_img_loss += img_l.item()
        total_key_loss += key_l.item()
        total_psnr += compute_psnr_torch(imgs.float(), watermarked.float())
        total_ber_clean += ber_from_logits(pred_clean.float(), wm_target_bits.float()).item()
        total_ber_attack += ber_from_logits(pred_attacked.float(), wm_target_bits.float()).item()
        total_wrong_conf += torch.abs(torch.sigmoid(wrong_pred.float()) - 0.5).mean().item()

        for i in range(imgs.size(0)):
            stages = decode_from_logits(pred_attacked[i:i + 1], key_strings[i])
            plain_np = wm_plain[i, 0].detach().cpu().numpy().astype(np.uint8)
            final_np = stages["final_bits"]
            total_plain_ber += float(np.mean(plain_np != final_np))

        total_batches += 1

    return {
        "loss": total_loss_val / max(total_batches, 1),
        "wm_loss": total_wm_loss / max(total_batches, 1),
        "img_loss": total_img_loss / max(total_batches, 1),
        "key_loss": total_key_loss / max(total_batches, 1),
        "psnr": total_psnr / max(total_batches, 1),
        "ber_clean": total_ber_clean / max(total_batches, 1),
        "ber_attack": total_ber_attack / max(total_batches, 1),
        "wrong_conf": total_wrong_conf / max(total_batches, 1),
        "plain_ber_attack": total_plain_ber / max(total_batches, 1),
    }


start_epoch = 0
best_score = -1e9
best_ber = float("inf")
best_psnr = 0.0
best_clean_ber = float("inf")
best_attack_ber = float("inf")

early_stopper = EarlyStopper(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA)

if RESUME:
    start_epoch, best_score, best_ber, best_psnr, best_clean_ber, best_attack_ber = try_resume(
        RESUME_PATH, encoder, decoder, optimizer, scheduler, scaler, early_stopper
    )

for epoch in range(start_epoch, EPOCHS):
    encoder.train()
    decoder.train()

    base_weights = get_loss_weights(epoch)
    attack_mix = 0.0 if epoch < ATTACK_START_EPOCH else 0.75
    weights = {**base_weights, "attack_mix": attack_mix}

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(
        f"Loss weights -> WM: {weights['wm']:.2f}, IMG: {weights['img']:.2f}, "
        f"KEY: {weights['key']:.2f}, ATTACK_MIX: {weights['attack_mix']:.2f}"
    )
    print(f"Current LR    -> {scheduler.get_last_lr()[0]:.8f}")

    running_loss = 0.0
    running_wm_loss = 0.0
    running_img_loss = 0.0
    running_key_loss = 0.0
    running_psnr = 0.0
    running_ber_clean = 0.0
    running_ber_attack = 0.0
    total_batches = 0

    for imgs, _ in tqdm(train_loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        wm_target_bits, wm_plain, key_tensor, wrong_key_tensor, _ = build_batch_payload(
            imgs.size(0), epoch=epoch
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            residual = encoder(imgs, wm_target_bits, key_tensor)
            watermarked = torch.clamp(imgs + residual, 0, 1)
            attacked = apply_epoch_attack(watermarked, epoch)

            pred_clean = decoder(watermarked, key_tensor)
            pred_attacked = decoder(attacked, key_tensor)
            wrong_pred = decoder(attacked, wrong_key_tensor)

            loss, wm_l, img_l, key_l = compute_combined_loss(
                imgs=imgs,
                watermarked=watermarked,
                pred_clean=pred_clean,
                pred_attacked=pred_attacked,
                wm_target_bits=wm_target_bits,
                wrong_pred=wrong_pred,
                weights=weights
            )

        if not torch.isfinite(loss):
            raise RuntimeError("Loss became NaN or Inf")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_wm_loss += wm_l.item()
        running_img_loss += img_l.item()
        running_key_loss += key_l.item()
        running_psnr += compute_psnr_torch(imgs.float(), watermarked.float())
        running_ber_clean += ber_from_logits(pred_clean.detach().float(), wm_target_bits.float()).item()
        running_ber_attack += ber_from_logits(pred_attacked.detach().float(), wm_target_bits.float()).item()
        total_batches += 1

    scheduler.step()
    val_stats = evaluate(encoder, decoder, val_loader, epoch)

    train_loss = running_loss / max(total_batches, 1)
    train_wm_loss = running_wm_loss / max(total_batches, 1)
    train_img_loss = running_img_loss / max(total_batches, 1)
    train_key_loss = running_key_loss / max(total_batches, 1)
    train_psnr = running_psnr / max(total_batches, 1)
    train_ber_clean = running_ber_clean / max(total_batches, 1)
    train_ber_attack = running_ber_attack / max(total_batches, 1)

    print(f"Train Loss       : {train_loss:.4f}")
    print(f"Train WM Loss    : {train_wm_loss:.4f}")
    print(f"Train IMG Loss   : {train_img_loss:.4f}")
    print(f"Train KEY Loss   : {train_key_loss:.4f}")
    print(f"Train BER Clean  : {train_ber_clean:.4f}")
    print(f"Train BER Attack : {train_ber_attack:.4f}")
    print(f"Train PSNR       : {train_psnr:.2f} dB")

    print(f"Val Loss         : {val_stats['loss']:.4f}")
    print(f"Val WM Loss      : {val_stats['wm_loss']:.4f}")
    print(f"Val IMG Loss     : {val_stats['img_loss']:.4f}")
    print(f"Val KEY Loss     : {val_stats['key_loss']:.4f}")
    print(f"Val BER Clean    : {val_stats['ber_clean']:.4f}")
    print(f"Val BER Attack   : {val_stats['ber_attack']:.4f}")
    print(f"Val Plain BER    : {val_stats['plain_ber_attack']:.4f}")
    print(f"Val PSNR         : {val_stats['psnr']:.2f} dB")
    print(f"Wrong-key conf   : {val_stats['wrong_conf']:.4f}")

    primary_ber = (
        val_stats["ber_clean"]
        if epoch < ATTACK_START_EPOCH
        else 0.5 * val_stats["ber_clean"] + 0.5 * val_stats["ber_attack"]
    )

    plain_ber = val_stats.get("plain_ber_attack", primary_ber)
    score = (
        5.0 * (1.0 - primary_ber) +
        3.0 * (1.0 - plain_ber) +
        0.02 * val_stats["psnr"] -
        0.6 * val_stats["wm_loss"] -
        0.10 * val_stats["img_loss"] -
        0.5 * val_stats["wrong_conf"]
    )

    save_checkpoint(
        CHECKPOINT_PATH,
        epoch,
        best_score,
        best_ber,
        best_psnr,
        best_clean_ber,
        best_attack_ber,
        encoder,
        decoder,
        optimizer,
        scheduler,
        scaler,
        val_stats,
        early_stopper
    )

    if val_stats["ber_clean"] < best_clean_ber:
        best_clean_ber = val_stats["ber_clean"]
        save_checkpoint(
            BEST_CLEAN_BER_PATH,
            epoch,
            best_score,
            best_ber,
            best_psnr,
            best_clean_ber,
            best_attack_ber,
            encoder,
            decoder,
            optimizer,
            scheduler,
            scaler,
            val_stats,
            early_stopper
        )

    if val_stats["ber_attack"] < best_attack_ber:
        best_attack_ber = val_stats["ber_attack"]
        save_checkpoint(
            BEST_ATTACK_BER_PATH,
            epoch,
            best_score,
            best_ber,
            best_psnr,
            best_clean_ber,
            best_attack_ber,
            encoder,
            decoder,
            optimizer,
            scheduler,
            scaler,
            val_stats,
            early_stopper
        )

    if primary_ber < best_ber:
        best_ber = primary_ber
        save_checkpoint(
            BEST_BER_PATH,
            epoch,
            best_score,
            best_ber,
            best_psnr,
            best_clean_ber,
            best_attack_ber,
            encoder,
            decoder,
            optimizer,
            scheduler,
            scaler,
            val_stats,
            early_stopper
        )
        print("Best BER checkpoint updated.")

    if val_stats["psnr"] > best_psnr:
        best_psnr = val_stats["psnr"]
        save_checkpoint(
            BEST_PSNR_PATH,
            epoch,
            best_score,
            best_ber,
            best_psnr,
            best_clean_ber,
            best_attack_ber,
            encoder,
            decoder,
            optimizer,
            scheduler,
            scaler,
            val_stats,
            early_stopper
        )

    if score > best_score:
        best_score = score
        save_checkpoint(
            BEST_CHECKPOINT_PATH,
            epoch,
            best_score,
            best_ber,
            best_psnr,
            best_clean_ber,
            best_attack_ber,
            encoder,
            decoder,
            optimizer,
            scheduler,
            scaler,
            val_stats,
            early_stopper
        )
        torch.save(encoder.state_dict(), BEST_ENCODER_PATH)
        torch.save(decoder.state_dict(), BEST_DECODER_PATH)
        print("Best score checkpoint updated.")

    if (epoch + 1) % SAVE_EVERY == 0:
        periodic_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch + 1}.pth")
        save_checkpoint(
            periodic_path,
            epoch,
            best_score,
            best_ber,
            best_psnr,
            best_clean_ber,
            best_attack_ber,
            encoder,
            decoder,
            optimizer,
            scheduler,
            scaler,
            val_stats,
            early_stopper
        )

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    stop_metric = val_stats.get("plain_ber_attack", primary_ber)
    if (epoch + 1) >= EARLY_STOP_MIN_EPOCH and early_stopper.step(stop_metric):
        print(f"Early stopping triggered at epoch {epoch + 1} (metric={stop_metric:.4f}).")
        break

print("Training complete.")
print("Latest checkpoint     :", CHECKPOINT_PATH)
print("Best checkpoint       :", BEST_CHECKPOINT_PATH)
print("Best BER checkpoint   :", BEST_BER_PATH)
print("Best clean BER        :", BEST_CLEAN_BER_PATH)
print("Best attack BER       :", BEST_ATTACK_BER_PATH)
print("Best encoder          :", BEST_ENCODER_PATH)
print("Best decoder          :", BEST_DECODER_PATH)