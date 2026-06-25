import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from src.models.segmentation import build_deeplabv3plus
from src.anomaly.scoring import compute_anomaly_score
from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD

# Konfiguracija
CHECKPOINT = "checkpoints/baseline_imagenet/best.pth"
INPUT_VIDEO_PATH = "anomalies/video/elk.mp4"
OUTPUT_VIDEO_PATH = "anomalies/video/results/baseline_elk.mp4"
METHOD = "energy"


def load_model(checkpoint_path, device):
    model = build_deeplabv3plus(
        encoder_name="resnet50", encoder_weights=None, num_classes=19
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


@torch.no_grad()
def get_energy_score(model, frame_bgr, device):
    """Prima BGR frejm (OpenCV), vraća (H,W) numpy normalized score array."""
    # OpenCV koristi BGR, prebacujemo u RGB i PIL Image za transformacije
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    w, h = img_pil.size
    t = TF.to_tensor(img_pil)
    t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    t = t.unsqueeze(0).to(device)

    # Padding na višekratnik od 16 ako model to zahtijeva
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")

    logits = model(t)
    if pad_h or pad_w:
        logits = logits[:, :, :h, :w]

    # Računanje Energy score-a
    s = compute_anomaly_score(logits, method=METHOD)
    score = s[0].cpu().numpy()
    
    return score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {CHECKPOINT}")
    model = load_model(CHECKPOINT, device)

    # Otvaranje ulaznog videa
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Greška: Nije moguće otvoriti video na lokaciji {INPUT_VIDEO_PATH}")
        return

    # Dohvaćanje svojstava videa
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Priprema izlaznog direktorija i video writera
    Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec za MP4
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print(f"Procesiranje videa: {INPUT_VIDEO_PATH} ({total_frames} frejmova)...")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Kraj videa

        # 1. Izračunaj anomalijske rezultate za trenutni frejm
        score = get_energy_score(model, frame, device)

        # 2. Normalizacija score-a na raspon 0-255 za prikaz (min-max normalizacija)
        # Napomena: Ako želiš fiksnu skalu kroz cijeli video, ovdje postavi fiksni min i max
        score_min, score_max = score.min(), score.max()
        if score_max - score_min > 1e-5:
            score_norm = (score - score_min) / (score_max - score_min)
        else:
            score_norm = np.zeros_like(score)
        
        score_img = (score_norm * 255).astype(np.uint8)

        # 3. Primijeni JET mapu boja (color map) na score sliku
        heatmap = cv2.applyColorMap(score_img, cv2.COLORMAP_JET)

        # 4. Preklapanje (Overlay) heatmapu preko originalnog frejma
        # alpha je prozirnost originalne slike, beta je prozirnost heatmape
        alpha = 0.6
        beta = 0.4
        overlayed_frame = cv2.addWeighted(frame, alpha, heatmap, beta, 0)

        # Zapiši frejm u izlazni video
        out.write(overlayed_frame)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Procesirano: {frame_idx}/{total_frames} frejmova")

    # Čišćenje resursa
    cap.release()
    out.release()
    print(f"\nNovi video s heatmapom je spremljen na: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()