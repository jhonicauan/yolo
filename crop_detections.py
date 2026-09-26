"""
Roda o YOLO em todas as imagens de um dataset e salva cada detecção
(com confiança >= threshold) como um recorte separado numa pasta.

Uso:
    python crop_detections.py bottles_ok
    python crop_detections.py bottles_ok --dataset bottledata --conf 0.7
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_latest_weights():
    candidates = list(Path("runs").rglob("best.pt"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def main():
    parser = argparse.ArgumentParser(description="Recortar detecções do YOLO em pasta.")
    parser.add_argument("output", type=str, help="Pasta de saída dos recortes.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="bottledata",
        help="Pasta do dataset (percorre images/train, images/val, images/test).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Alternativa: passe uma pasta com imagens em vez de um dataset YOLO.",
    )
    parser.add_argument("--weights", type=str, default=None, help="best.pt (padrão: mais recente).")
    parser.add_argument("--conf", type=float, default=0.70, help="Confiança mínima.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--padding", type=int, default=0, help="Pixels extras ao redor da bbox.")
    parser.add_argument("--min-size", type=int, default=32, help="Descarta recortes menores que isto.")
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="Cria subpastas por classe dentro da pasta de saída.",
    )
    args = parser.parse_args()

    weights = args.weights or find_latest_weights()
    if not weights:
        raise FileNotFoundError("Nenhum best.pt em runs/. Treine o modelo primeiro.")
    print(f"Pesos: {weights}")

    model = YOLO(weights)

    # Coleta imagens
    if args.source:
        src = Path(args.source)
        images = sorted([p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS])
    else:
        ds = Path(args.dataset)
        images = []
        for split in ("train", "val", "test"):
            d = ds / "images" / split
            if d.exists():
                images.extend(sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS))
    if not images:
        raise RuntimeError("Nenhuma imagem encontrada.")
    print(f"Imagens a processar: {len(images)}")

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    total_crops = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  falha ao ler: {img_path}")
            continue
        H, W = img.shape[:2]

        results = model.predict(
            source=img,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf < args.conf:
                continue
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            # padding com clip nas bordas
            x1 = max(0, x1 - args.padding)
            y1 = max(0, y1 - args.padding)
            x2 = min(W, x2 + args.padding)
            y2 = min(H, y2 + args.padding)

            w, h = x2 - x1, y2 - y1
            if w < args.min_size or h < args.min_size:
                continue

            crop = img[y1:y2, x1:x2]

            if args.per_class:
                dest_dir = out_root / cls_name
                dest_dir.mkdir(exist_ok=True)
            else:
                dest_dir = out_root

            out_name = f"{img_path.stem}_{i:02d}_{cls_name}_c{int(conf * 100)}.jpg"
            cv2.imwrite(str(dest_dir / out_name), crop)
            total_crops += 1

        print(f"  {img_path.name}: {len(boxes)} detecção(ões)")

    print(f"\nOK. {total_crops} recortes salvos em {out_root}")


if __name__ == "__main__":
    main()
