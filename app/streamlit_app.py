import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torchvision.transforms.functional import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.encoder import EncoderNet
from models.decoder import DecoderNet
from models.attack_layer import targeted_attack, random_attack
from utils.preprocess import load_binary_watermark
from utils.metrics import ber as ber_metric
from utils.metrics import compute_ssim as ssim_metric
from utils.metrics import normalized_correlation
from utils.metrics import psnr as psnr_metric
from utils.wm_pipeline import (
    decode_from_logits,
    encode_for_embedding,
    extract_with_crop_search,
    key_to_tensor,
    prepare_plain_watermark,
    wm_secure_to_tensor,
)

st.set_page_config(
    page_title="Blind Watermarking System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {
    max-width: 1450px !important;
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}
section[data-testid="stSidebar"] {
    width: 340px !important;
    min-width: 340px !important;
}
section[data-testid="stSidebar"] > div {
    width: 340px !important;
}
h1 {
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.2rem !important;
}
h2, h3 {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
}
.stButton > button, .stDownloadButton > button {
    width: 100%;
    min-height: 3rem;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
}
div[data-testid="stMetric"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    padding: 0.9rem 1rem !important;
    border-radius: 14px !important;
}
img {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WM_SIZE = 32

CHECKPOINT_SEARCH_DIRS = [
    PROJECT_ROOT / "checkpoints",
    PROJECT_ROOT / "models",
]
_env_ckpt_dir = os.environ.get("WM_CHECKPOINT_DIR")
if _env_ckpt_dir:
    CHECKPOINT_SEARCH_DIRS.insert(0, Path(_env_ckpt_dir))

KNOWN_CHECKPOINT_NAMES = [
    "checkpoint_best_ber.pth",
    "checkpoint_best_attack_ber.pth",
    "checkpoint_best_clean_ber.pth",
    "checkpoint_best_psnr.pth",
    "checkpoint_best.pth",
    "checkpoint_latest.pth",
    "best_encoder.pth",
    "best_decoder.pth",
]

ATTACK_CHOICES = [
    "None",

    "JPEG70",
    "JPEG50",
    "JPEG30",

    "Crop10",
    "Crop20",
    "Crop30",

    "Translation5",
    "Translation10",
    "Translation15",

    "Zoom105",
    "Zoom120",
    "Zoom140",

    # Keep all other existing attacks
    "Blur5",
    "Blur7",
    "Noise 0.03",
    "Noise 0.05",
    "Rotate 10",
    "Rotate 15",
    "Resize 70",
    "Brightness",
    "Contrast",
    "Salt & Pepper",
    "Random Medium",
    "Random Strong",
]

# ── helpers ──────────────────────────────────────────────────────────────────

def pil_to_tensor(img, size=(256, 256)):
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize(size, Image.Resampling.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    return tensor, np.array(img)


def tensor_to_rgb_np(t):
    x = t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (x * 255).astype(np.uint8)


def psnr(x, y, eps=1e-8):
    return psnr_metric(x, y)


def ssim(img1, img2):
    return ssim_metric(img1, img2)


def ber_and_accuracy(true_wm, pred_wm):
    ber_val = ber_metric(true_wm, pred_wm)
    return float(ber_val), float(1.0 - ber_val)


def prepare_watermark(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        wm = load_binary_watermark(
            tmp_path,
            size=(WM_SIZE, WM_SIZE),
            threshold=127,
            cleanup=True,
            center=True,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return wm.astype(np.uint8)


def png_bytes_from_array(arr):
    bio = BytesIO()
    Image.fromarray(arr).save(bio, format="PNG")
    return bio.getvalue()


# ── rotation-robust extraction ────────────────────────────────────────────────

_SEARCH_ANGLES = [-15, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 15]


def derotate_and_decode(attacked_tensor, decoder, key_tensor, wrong_key_tensor):
    """
    Sweep inverse rotations over a candidate grid and pick the orientation
    where decoder confidence is highest (bits furthest from 0.5).
    Returns (best_logits, wrong_logits, estimated_angle_deg).
    """
    best_score = -1.0
    best_logits = None
    best_angle = 0.0

    for angle in _SEARCH_ANGLES:
        if abs(angle) < 1e-3:
            rotated = attacked_tensor
        else:
            rotated = TF.rotate(
                attacked_tensor,
                angle=-angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=0.5,
            )
        with torch.no_grad():
            logits = decoder(rotated, key_tensor)
        conf = torch.mean(torch.abs(torch.sigmoid(logits) - 0.5)).item()
        if conf > best_score:
            best_score = conf
            best_logits = logits
            best_angle = angle

    # Decode wrong key at the best orientation
    wrong_logits = None
    if wrong_key_tensor is not None:
        if abs(best_angle) < 1e-3:
            best_rotated = attacked_tensor
        else:
            best_rotated = TF.rotate(
                attacked_tensor,
                angle=-best_angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=0.5,
            )
        with torch.no_grad():
            wrong_logits = decoder(best_rotated, wrong_key_tensor)

    return best_logits, wrong_logits, best_angle


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _torch_load(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def _peek_checkpoint_meta(path: Path):
    meta = {"epoch": None, "best_ber": None, "val_psnr": None, "val_ber_attack": None}
    try:
        ckpt = _torch_load(path)
        if isinstance(ckpt, dict):
            meta.update({k: ckpt.get(k) for k in meta})
            meta["kind"] = "full" if ("encoder" in ckpt and "decoder" in ckpt) else "partial"
        else:
            meta["kind"] = "state_dict"
    except Exception as exc:
        meta["kind"] = "error"
        meta["error"] = str(exc)
    return meta


def _format_checkpoint_label(path: Path):
    meta = _peek_checkpoint_meta(path)
    parts = [path.name]
    if meta.get("epoch") is not None:
        parts.append(f"epoch {int(meta['epoch']) + 1}")
    best_ber = meta.get("best_ber")
    if best_ber is not None and best_ber == best_ber and best_ber < 1e6:
        parts.append(f"best_ber {best_ber:.4f}")
    val_atk = meta.get("val_ber_attack")
    if val_atk is not None and val_atk == val_atk:
        parts.append(f"val_atk {val_atk:.4f}")
    val_psnr = meta.get("val_psnr")
    if val_psnr is not None and val_psnr == val_psnr:
        parts.append(f"psnr {val_psnr:.1f}")
    if meta.get("kind") == "partial":
        parts.append(meta["kind"])
    return " | ".join(parts)


@st.cache_data(ttl=30)
def discover_checkpoints():
    found = {}
    seen = set()
    for directory in CHECKPOINT_SEARCH_DIRS:
        if not directory.exists():
            continue
        candidates = [directory / n for n in KNOWN_CHECKPOINT_NAMES]
        candidates += sorted(directory.glob("checkpoint_epoch_*.pth"))
        candidates += sorted(directory.glob("*.pth"))
        for path in candidates:
            if not path.is_file():
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            label = _format_checkpoint_label(path)
            if label in found:
                label = f"{label} ({directory.name})"
            found[label] = path
    return dict(sorted(found.items(), key=lambda kv: kv[1].name))


def _load_split_encoder_decoder(encoder, decoder, path: Path):
    name = path.name.lower()
    parent = path.parent
    enc_path = path if "encoder" in name else parent / "best_encoder.pth"
    dec_path = path if "decoder" in name else parent / "best_decoder.pth"
    if not enc_path.exists():
        raise FileNotFoundError(f"Encoder weights not found: {enc_path}")
    if not dec_path.exists():
        raise FileNotFoundError(f"Decoder weights not found: {dec_path}")
    enc_info = encoder.load_state_dict(_torch_load(enc_path), strict=False)
    dec_info = decoder.load_state_dict(_torch_load(dec_path), strict=False)
    return enc_info, dec_info, {}


def _load_full_checkpoint(encoder, decoder, ckpt_path: Path):
    ckpt = _torch_load(ckpt_path)
    if isinstance(ckpt, dict) and "encoder" in ckpt and "decoder" in ckpt:
        enc_info = encoder.load_state_dict(ckpt["encoder"], strict=False)
        dec_info = decoder.load_state_dict(ckpt["decoder"], strict=False)
        return enc_info, dec_info, {
            "epoch": ckpt.get("epoch"),
            "best_ber": ckpt.get("best_ber"),
            "val_psnr": ckpt.get("val_psnr"),
            "val_ber_attack": ckpt.get("val_ber_attack"),
            "arch_version": ckpt.get("arch_version"),
        }
    raise RuntimeError(
        f"Unsupported checkpoint format: {ckpt_path}. "
        "Expected dict with 'encoder' and 'decoder' keys."
    )


@st.cache_resource
def load_models(checkpoint_path_str: str):
    ckpt_path = Path(checkpoint_path_str)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    encoder = EncoderNet(wm_size=WM_SIZE).to(DEVICE)
    decoder = DecoderNet(wm_size=WM_SIZE).to(DEVICE)
    name_lower = ckpt_path.name.lower()
    if "encoder" in name_lower or "decoder" in name_lower:
        enc_info, dec_info, meta = _load_split_encoder_decoder(encoder, decoder, ckpt_path)
    else:
        enc_info, dec_info, meta = _load_full_checkpoint(encoder, decoder, ckpt_path)
    encoder.eval()
    decoder.eval()
    return encoder, decoder, {
        "path": str(ckpt_path.resolve()),
        "file_name": ckpt_path.name,
        **meta,
        "encoder_missing": enc_info.missing_keys,
        "encoder_unexpected": enc_info.unexpected_keys,
        "decoder_missing": dec_info.missing_keys,
        "decoder_unexpected": dec_info.unexpected_keys,
    }


def apply_selected_attack(img_tensor, attack_name):

    if attack_name == "None":
        return img_tensor

    # JPEG
    if attack_name == "JPEG70":
        return targeted_attack(img_tensor, "jpeg70")

    if attack_name == "JPEG50":
        return targeted_attack(img_tensor, "jpeg50")

    if attack_name == "JPEG30":
        return targeted_attack(img_tensor, "jpeg30")

    # Crop
    if attack_name == "Crop10":
        return targeted_attack(img_tensor, "crop90")

    if attack_name == "Crop20":
        return targeted_attack(img_tensor, "crop80")

    if attack_name == "Crop30":
        return targeted_attack(img_tensor, "crop70")

    # Translation
    if attack_name == "Translation5":
        return targeted_attack(img_tensor, "translate5")

    if attack_name == "Translation10":
        return targeted_attack(img_tensor, "translate10")

    if attack_name == "Translation15":
        return targeted_attack(img_tensor, "translate15")

    # Zoom
    if attack_name == "Zoom105":
        return targeted_attack(img_tensor, "zoom105")

    if attack_name == "Zoom120":
        return targeted_attack(img_tensor, "zoom120")

    if attack_name == "Zoom140":
        return targeted_attack(img_tensor, "zoom140")

    mapping = {
        "Blur5": "blur5",
        "Blur7": "blur7",
        "Noise 0.03": "noise003",
        "Noise 0.05": "noise005",
        "Rotate 10": "rotate10",
        "Rotate 15": "rotate15",
        "Resize 70": "resize70",
        "Brightness": "brightness",
        "Contrast": "contrast",
        "Salt & Pepper": "sp002",
    }

    if attack_name == "Random Medium":
        return random_attack(img_tensor, strength="medium")

    if attack_name == "Random Strong":
        return random_attack(img_tensor, strength="strong")

    key = mapping.get(attack_name)

    if key:
        return targeted_attack(img_tensor, key)

    return img_tensor

    


def _is_rotation_attack(attack_name: str) -> bool:
    return "Rotate" in attack_name or "Random" in attack_name


def choose_default_checkpoint() -> Path:
    """Prefer the best available checkpoint, then latest, then the newest epoch checkpoint."""
    candidates = []

    preferred_names = [
        "checkpoint_best.pth",
        "checkpoint_best_ber.pth",
        "checkpoint_best_attack_ber.pth",
        "checkpoint_best_clean_ber.pth",
        "checkpoint_best_psnr.pth",
        "checkpoint_latest.pth",
    ]

    for directory in CHECKPOINT_SEARCH_DIRS:
        if not directory.exists():
            continue
        for name in preferred_names:
            path = directory / name
            if path.exists():
                candidates.append(path)

        epoch_files = sorted(directory.glob("checkpoint_epoch_*.pth"), key=lambda p: p.name)
        candidates.extend(epoch_files)

    if candidates:
        # Choose the newest epoch checkpoint if present, otherwise the first preferred one.
        epoch_candidates = [p for p in candidates if "checkpoint_epoch_" in p.name]
        if epoch_candidates:
            return max(epoch_candidates, key=lambda p: int(p.stem.split("_")[-1]))
        return candidates[0]

    return PROJECT_ROOT / "checkpoints" / "checkpoint_latest.pth"


# ── session state ─────────────────────────────────────────────────────────────

for key, default in {
    "watermarked_tensor": None,
    "watermarked_np": None,
    "host_np": None,
    "wm_true": None,
    "checkpoint_path": None,
    "last_results": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")

    # Auto-select the best available checkpoint from the project folders.
    selected_ckpt_path = choose_default_checkpoint()

    # Keys
    st.subheader("Keys")

    embed_key = st.text_input(
        "Embedding Key",
        placeholder="Enter embedding key",
        type="password"
    )

    extract_key = st.text_input(
        "Extraction Key",
        placeholder="Enter extraction key",
        type="password"
    )

    st.divider()

    # Attack + robustness
    st.subheader("Attack")
    attack_name = st.selectbox("Attack type", ATTACK_CHOICES)
    use_rotation_search = True

    st.divider()

    # Always enabled internally
    use_decrypt = True
    use_bch = True

# ── model loading ─────────────────────────────────────────────────────────────

ckpt_path_str = str(selected_ckpt_path.resolve())

if st.session_state["checkpoint_path"] != ckpt_path_str:
    for k in ("watermarked_tensor", "watermarked_np", "host_np", "wm_true", "last_results"):
        st.session_state[k] = None
    st.session_state["checkpoint_path"] = ckpt_path_str

if selected_ckpt_path is None:
    st.info("Select or upload a checkpoint in the sidebar to load models.")
    st.stop()

try:
    encoder, decoder, load_info = load_models(ckpt_path_str)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.exception(e)
    st.stop()

# ── title + checkpoint summary ────────────────────────────────────────────────

st.title("Blind Watermarking System")
# ── upload panel ──────────────────────────────────────────────────────────────

col_upload_1, col_upload_2 = st.columns(2)
with col_upload_1:
    host_file = st.file_uploader("Upload Host Image", type=["png", "jpg", "jpeg"], key="host_file")
with col_upload_2:
    wm_file = st.file_uploader("Upload Binary Watermark", type=["png", "jpg", "jpeg"], key="wm_file")

col_btn_1, col_btn_2 = st.columns(2)
embed_clicked = col_btn_1.button("Embed Watermark", use_container_width=True)
extract_clicked = col_btn_2.button("Extract Watermark", use_container_width=True)

# ── embed ─────────────────────────────────────────────────────────────────────

if embed_clicked:
    if host_file is None or wm_file is None:
        st.warning("Please upload both a host image and a watermark image.")
    else:
        host_pil = Image.open(host_file)
        host_tensor, host_np = pil_to_tensor(host_pil)

        wm_true = prepare_plain_watermark(prepare_watermark(wm_file))
        wm_secure = encode_for_embedding(wm_true, embed_key)
        wm_tensor = wm_secure_to_tensor(wm_secure, device=DEVICE)
        key_tensor = key_to_tensor(embed_key, size=WM_SIZE, device=DEVICE)

        with torch.no_grad():
            residual = encoder(host_tensor, wm_tensor, key_tensor)
            watermarked_tensor = torch.clamp(host_tensor + residual, 0, 1)

        watermarked_np = tensor_to_rgb_np(watermarked_tensor)
        psnr_embed = psnr(host_np, watermarked_np)

        st.session_state["watermarked_tensor"] = watermarked_tensor
        st.session_state["watermarked_np"] = watermarked_np
        st.session_state["host_np"] = host_np
        st.session_state["wm_true"] = wm_true
        st.session_state["psnr_embed"] = psnr_embed
        st.session_state["last_results"] = None

        st.success(f"Watermark embedded — PSNR: {psnr_embed:.2f} dB")

        ec1, ec2 = st.columns(2)

        with ec1:
            st.image(
                host_np,
                caption="Host Image",
                width=280
            )

        with ec2:
            st.image(
                watermarked_np,
                caption=f"Watermarked ({psnr_embed:.2f} dB)",
                width=280
            )

        st.download_button(
            "Download Watermarked Image",
            data=png_bytes_from_array(watermarked_np),
            file_name="watermarked_image.png",
            mime="image/png",
            use_container_width=False,
        )

# ── extract ───────────────────────────────────────────────────────────────────

if extract_clicked:
    if st.session_state["watermarked_tensor"] is None:
        st.warning("Please embed a watermark first.")
    else:
        attacked_tensor = apply_selected_attack(
            st.session_state["watermarked_tensor"].clone(),
            attack_name,
        )
        attacked_np = tensor_to_rgb_np(attacked_tensor)

        key_t = key_to_tensor(extract_key, size=WM_SIZE, device=DEVICE)
        wrong_key_t = key_to_tensor("wrong_key_probe", size=WM_SIZE, device=DEVICE)

        is_rotation = _is_rotation_attack(attack_name)
        estimated_angle = 0.0

        if use_rotation_search and is_rotation:
            logits, wrong_logits, estimated_angle = derotate_and_decode(
                attacked_tensor, decoder, key_t, wrong_key_t
            )
            best_candidate = attacked_tensor
        else:
            best_candidate, logits, decoded, _ = extract_with_crop_search(
                decoder,
                attacked_tensor,
                extract_key,
                key_t,
                use_decrypt=use_decrypt,
                use_ecc=use_bch,
            )
            with torch.no_grad():
                wrong_logits = decoder(best_candidate, wrong_key_t)
            estimated_angle = 0.0

        if "decoded" not in locals():
            decoded = decode_from_logits(logits, extract_key, use_decrypt=use_decrypt, use_ecc=use_bch)
        wrong_decoded = decode_from_logits(wrong_logits, "wrong_key_probe",
                                           use_decrypt=use_decrypt, use_ecc=use_bch)

        final_bits = decoded["final_bits"]
        raw_bits = decoded["raw_bits"]
        decrypted_bits = decoded["decrypted_bits"]
        prob_map = decoded["prob_map"]
        wrong_final = wrong_decoded["final_bits"]

        final_ber, final_acc = ber_and_accuracy(st.session_state["wm_true"], final_bits)
        raw_ber, raw_acc = ber_and_accuracy(st.session_state["wm_true"], raw_bits)
        dec_ber, _ = ber_and_accuracy(st.session_state["wm_true"], decrypted_bits)
        ncc_val = normalized_correlation(st.session_state["wm_true"], final_bits)
        wrong_key_sim = float(np.mean(wrong_final == st.session_state["wm_true"]))
        
        # For "No Attack", use imperceptibility PSNR from embedding; for other attacks, measure degradation
        if attack_name == "None":
            psnr_val = st.session_state.get("psnr_embed", 35.0)
            ssim_val = ssim_metric(st.session_state["host_np"], st.session_state["watermarked_np"])
        else:
            psnr_val = psnr_metric(st.session_state["host_np"], attacked_np)
            ssim_val = ssim_metric(st.session_state["host_np"], attacked_np)

        # Store for re-rendering graphs without re-running
        st.session_state["last_results"] = {
            "attack_name": attack_name,
            "estimated_angle": estimated_angle,
            "is_rotation": is_rotation,
            "psnr_val": psnr_val,
            "ssim_val": ssim_val,
            "ncc_val": ncc_val,
            "final_ber": final_ber,
            "final_acc": final_acc,
            "raw_ber": raw_ber,
            "raw_acc": raw_acc,
            "dec_ber": dec_ber,
            "wrong_key_sim": wrong_key_sim,
            "attacked_np": attacked_np,
            "final_bits": final_bits,
            "raw_bits": raw_bits,
            "decrypted_bits": decrypted_bits,
            "prob_map": prob_map,
        }

# ── results display ───────────────────────────────────────────────────────────

res = st.session_state.get("last_results")
if res is not None:
    attack_name_r = res["attack_name"]
    final_ber = res["final_ber"]
    final_acc = res["final_acc"]
    raw_ber = res["raw_ber"]
    dec_ber = res["dec_ber"]
    ncc_val = res["ncc_val"]
    psnr_val = res["psnr_val"]
    ssim_val = res["ssim_val"]
    wrong_key_sim = res["wrong_key_sim"]
    attacked_np = res["attacked_np"]
    final_bits = res["final_bits"]
    raw_bits = res["raw_bits"]
    decrypted_bits = res["decrypted_bits"]
    prob_map = res["prob_map"]

    st.divider()
    st.subheader("Quality & Robustness Metrics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PSNR", f"{psnr_val:.2f} dB", help="Higher is better (>30 dB is imperceptible).")
    m2.metric("SSIM", f"{ssim_val:.4f}", help="Structural similarity (1 = identical).")
    m3.metric("NCC", f"{ncc_val:.4f}", help="Normalized cross-correlation of extracted vs original bits.")
    m4.metric("Final BER", f"{final_ber:.4f}", help="Bit error rate after full pipeline (lower is better).")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Raw BER", f"{raw_ber:.4f}", help="BER before decryption/ECC.")
    d2.metric("Decrypted BER", f"{dec_ber:.4f}", help="BER after decryption, before ECC.")
    d3.metric("Wrong-key sim", f"{wrong_key_sim:.4f}",
              help="Similarity with wrong key (≈0.5 means no leakage).")
    d4.metric("Attack", attack_name_r)

    if res["is_rotation"] and res["estimated_angle"] != 0.0:
        st.info(
            f"Anti-rotation search estimated the image was rotated by "
            f"**{res['estimated_angle']:.0f}°** and compensated automatically."
        )

    # ── Visual comparison ─────────────────────────────────────────────────────

    st.subheader("Visual Comparison")
    row2 = st.columns(5)

    with row2[0]:
        st.image(
            st.session_state["host_np"],
            caption="Host Image",
            use_container_width=True
        )

    with row2[1]:
        st.image(
            st.session_state["watermarked_np"],
            caption="Watermarked",
            use_container_width=True
    )

    with row2[2]:
        st.image(
            attacked_np,
            caption=f"Attacked: {attack_name}",
            use_container_width=True
        )

    with row2[3]:
        st.image(
            (st.session_state["wm_true"] * 255).astype(np.uint8),
            caption="Original Watermark",
            use_container_width=True,
        )

    with row2[4]:
        st.image(
            (final_bits * 255).astype(np.uint8),
            caption="Final Extracted",
            use_container_width=True
        )


    # # ── Charts for report / PPT ───────────────────────────────────────────────

    # st.divider()
    # st.subheader("Charts")

    # tab_pipeline, tab_comparison = st.tabs(
    #     ["Pipeline BER breakdown", "Attack robustness comparison"]
    # )

    # with tab_pipeline:
    #     st.markdown(
    #         "**BER at each decoding stage** — shows how decryption and error-correction "
    #         "progressively reduce the bit error rate."
    #     )
    #     pipeline_df = pd.DataFrame(
    #         {
    #             "Stage": ["Raw (after decoder)", "After Decryption", "Final (after ECC)"],
    #             "BER": [raw_ber, dec_ber, final_ber],
    #             "Accuracy (%)": [
    #                 (1 - raw_ber) * 100,
    #                 (1 - dec_ber) * 100,
    #                 final_acc * 100,
    #             ],
    #         }
    #     )
    #     # Bar chart
    #     st.bar_chart(
    #         pipeline_df.set_index("Stage")["BER"],
    #         color="#EF553B",
    #         use_container_width=True,
    #     )
    #     st.dataframe(
    #         pipeline_df.style.format({"BER": "{:.4f}", "Accuracy (%)": "{:.2f}"}),
    #         use_container_width=True,
    #         hide_index=True,
    #     )

    # with tab_comparison:
    #     st.markdown(
    #         "**Per-attack robustness** — run extraction under different attacks "
    #         "to fill in this comparison chart."
    #     )

    #     # Accumulate results across runs into session state
    #     if "attack_history" not in st.session_state:
    #         st.session_state["attack_history"] = {}

    #     atk_key = attack_name_r if attack_name_r != "None" else "No Attack"
    #     st.session_state["attack_history"][atk_key] = {
    #         "Accuracy (%)": final_acc * 100,
    #         "BER": final_ber,
    #         "NCC": ncc_val,
    #     }

    #     hist = st.session_state["attack_history"]
    #     if hist:
    #         hist_df = pd.DataFrame(hist).T.reset_index().rename(columns={"index": "Attack"})
    #         col_chart1, col_chart2 = st.columns(2)

    #         with col_chart1:
    #             st.markdown("**Accuracy (%) per attack**")
    #             st.bar_chart(
    #                 hist_df.set_index("Attack")["Accuracy (%)"],
    #                 color="#00CC96",
    #                 use_container_width=True,
    #             )

    #         with col_chart2:
    #             st.markdown("**NCC per attack**")
    #             st.bar_chart(
    #                 hist_df.set_index("Attack")["NCC"],
    #                 color="#636EFA",
    #                 use_container_width=True,
    #             )

    #         st.dataframe(
    #             hist_df.style.format({"Accuracy (%)": "{:.2f}", "BER": "{:.4f}", "NCC": "{:.4f}"}),
    #             use_container_width=True,
    #             hide_index=True,
    #         )

    #         if st.button("Clear attack history", use_container_width=False):
    #             st.session_state["attack_history"] = {}
    #             st.rerun()

    #     else:
    #         st.info("Run extraction under different attacks to compare robustness here.")

    # ─────────────────────────────────────────────

    import matplotlib.pyplot as plt
    import seaborn as sns

    # ── attack history ─────────────────────────

    if "attack_history" not in st.session_state:
        st.session_state["attack_history"] = {}

    atk_key = attack_name_r if attack_name_r != "None" else "No Attack"

    st.session_state["attack_history"][atk_key] = {
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "BER": final_ber,
        "NCC": ncc_val,
    }

    hist_df = pd.DataFrame(
        st.session_state["attack_history"]
    ).T.reset_index()

    hist_df.columns = [
        "Attack",
        "PSNR",
        "SSIM",
        "BER",
        "NCC"
    ]

    # ── HEATMAP ───────────────────────────────

    st.markdown("Attack Robustness Heatmap")

    heatmap_df = hist_df.set_index("Attack")

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=1,
        cbar=True,
        ax=ax
    )

    st.pyplot(fig, use_container_width=True)

    # ── BER GRAPH ─────────────────────────────

    st.markdown("## 📉 BER vs Attack")

    fig2, ax2 = plt.subplots(figsize=(12, 4))

    ax2.plot(
        hist_df["Attack"],
        hist_df["BER"],
        marker="o",
        linewidth=3
    )

    ax2.set_xlabel("Attack")
    ax2.set_ylabel("BER")
    ax2.set_title("BER vs Attack")

    plt.xticks(rotation=30)

    st.pyplot(fig2, use_container_width=True)

        # ── Download ──────────────────────────────────────────────────────────────

    st.divider()
    st.download_button(
        "Download Extracted Watermark (PNG)",
        data=png_bytes_from_array((final_bits * 255).astype(np.uint8)),
        file_name="extracted_watermark.png",
        mime="image/png",
        use_container_width=True,
    )
