"""
Aplica YOLO OBB + SAM + perspective warp em imagens brutas e salva crops
planificados no formato do anomalib (para treinar o EfficientAd).

Entrada:
    raw_dataset/train/good/          imagens normais
    raw_dataset/test/good/           normais de teste
    raw_dataset/test/defeito/        anômalas (opcional)

Saída:
    crops_dataset/<mesma_estrutura>/*.png

Uso:
    python src/data_prep/prepare_anomaly_crops.py --src raw_dataset --dst crops_dataset
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import YOLO_WEIGHTS, SAM_WEIGHTS, SamRefiner, perspective_warp

from ultralytics import YOLO


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def process(src_dir: Path, dst_dir: Path, yolo, sam, conf, imgsz, min_side):
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_imgs = n_crops = 0
    for img_path in sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_EXTS):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        n_imgs += 1
        obb = yolo.predict(source=img, conf=conf, imgsz=imgsz, verbose=False)[0].obb
        if obb is None or len(obb) == 0:
            continue
        for k, raw in enumerate(obb.xyxyxyxy.cpu().numpy()):
            quad = sam.refine(img, raw)
            if quad is None:
                quad = raw
            warp = perspective_warp(img, quad)
            if warp is None or min(warp.shape[:2]) < min_side:
                continue
            cv2.imwrite(str(dst_dir / f"{img_path.stem}__{k}.png"), warp)
            n_crops += 1
    print(f"  {src_dir} -> {dst_dir}: {n_imgs} imgs, {n_crops} crops")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("raw_dataset"))
    parser.add_argument("--dst", type=Path, default=Path("crops_dataset"))
    parser.add_argument("--weights", type=str, default=str(YOLO_WEIGHTS))
    parser.add_argument("--sam", type=str, default=str(SAM_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-side", type=int, default=32)
    parser.add_argument("--splits", nargs="+",
                        default=["train/good", "test/good", "test/defeito"])
    args = parser.parse_args()

    print(f"YOLO OBB : {args.weights}")
    print(f"SAM      : {args.sam}\n")
    yolo = YOLO(args.weights)
    sam = SamRefiner(args.sam, mode="quad")

    for split in args.splits:
        src, dst = args.src / split, args.dst / split
        if src.exists():
            process(src, dst, yolo, sam, args.conf, args.imgsz, args.min_side)
        else:
            print(f"[skip] {src}")


if __name__ == "__main__":
    main()
