"""
Pipeline completa numa imagem: YOLO OBB -> SAM -> perspective warp -> EfficientAd.

Detecta as etiquetas na imagem, planifica cada uma e mede a chance de defeito.

Uso:
    python src/predict_anomaly.py samples/errado.jpeg
    python src/predict_anomaly.py samples/correto.jpeg --device cpu
    python src/predict_anomaly.py --crop crops_dataset/test/defeito/errado__0.png
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (
    YOLO_WEIGHTS, SAM_WEIGHTS, ANOMALIB_CKPT,
    SamRefiner, perspective_warp,
)


IMG_SIZE = 512
OUT_DIR = Path("runs/anomaly")


def score_crop(model, crop, device):
    """Passa um crop pelo EfficientAd e devolve (score, heatmap 0..1)."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        out = model(t)
    score = float(out.pred_score.item())
    heat = out.anomaly_map.squeeze().cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return score, heat


def save_visual(crop_bgr, heat, out_path, score):
    heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    resized = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE))
    overlay = cv2.addWeighted(resized, 0.6, heat_color, 0.4, 0)
    cv2.imwrite(str(out_path), np.hstack([resized, overlay]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=None, help="Imagem original com etiqueta(s).")
    parser.add_argument("--crop", type=str, default=None,
                        help="Crop já planificado: pula YOLO/SAM.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             ("cuda" if args.device == "cuda" else "cpu")
    print(f"Device      : {device}")
    print(f"Anomalib    : {ANOMALIB_CKPT}")

    from anomalib.models import EfficientAd
    model = EfficientAd.load_from_checkpoint(str(ANOMALIB_CKPT), map_location=device)
    model.eval().to(device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Modo 1: crop já pronto ------------------------------------------
    if args.crop:
        crop = cv2.imread(args.crop)
        if crop is None:
            raise FileNotFoundError(args.crop)
        score, heat = score_crop(model, crop, device)
        pct = score * 100
        label = "ANOMALO" if score > args.threshold else "normal"
        print(f"\n{args.crop}")
        print(f"  {pct:6.2f}%  [{label}]")
        save_visual(crop, heat, OUT_DIR / f"{Path(args.crop).stem}_score{int(pct):02d}.png", score)
        return

    # --- Modo 2: pipeline completa ---------------------------------------
    if not args.image:
        parser.error("Passe uma imagem ou --crop <caminho>.")
    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_WEIGHTS))
    print(f"YOLO OBB    : {YOLO_WEIGHTS}")
    results = yolo.predict(source=img, conf=args.conf, verbose=False)
    obb = results[0].obb
    if obb is None or len(obb) == 0:
        print("Nenhuma etiqueta detectada.")
        return
    print(f"Detectou {len(obb)} etiqueta(s).\n")

    sam = SamRefiner(SAM_WEIGHTS, mode="quad")
    for i, raw in enumerate(obb.xyxyxyxy.cpu().numpy()):
        quad = sam.refine(img, raw)
        if quad is None:
            quad = raw
        warp = perspective_warp(img, quad)
        if warp is None:
            continue
        score, heat = score_crop(model, warp, device)
        pct = score * 100
        label = "ANOMALO" if score > args.threshold else "normal"
        print(f"  crop {i}: {pct:6.2f}%  [{label}]")
        save_visual(warp, heat, OUT_DIR / f"{img_path.stem}__crop{i}_score{int(pct):02d}.png", score)

    print(f"\nHeatmaps salvos em: {OUT_DIR}/")


if __name__ == "__main__":
    main()
