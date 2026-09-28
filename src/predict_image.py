"""
Roda o YOLO-OBB em uma imagem, pasta ou vídeo. Salva a saída anotada.

Uso:
    python src/predict_image.py samples/errado.jpeg
    python src/predict_image.py minha_pasta/
    python src/predict_image.py video.mp4 --conf 0.6
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import YOLO_WEIGHTS

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="Imagem, pasta ou vídeo.")
    parser.add_argument("--weights", type=str, default=str(YOLO_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=str, default="runs/predict")
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise FileNotFoundError(src)

    print(f"Pesos: {args.weights}\nFonte: {src}\n")
    model = YOLO(args.weights)
    results = model.predict(
        source=str(src), conf=args.conf, imgsz=args.imgsz,
        save=True, project=args.out, name="obb", exist_ok=True,
    )
    for r in results:
        n = 0 if r.obb is None else len(r.obb)
        print(f"{Path(r.path).name}: {n} detecção(ões)")
        if not n:
            continue
        for i in range(n):
            cls = model.names[int(r.obb.cls[i])]
            conf = float(r.obb.conf[i])
            cx, cy, w, h, ang = [float(v) for v in r.obb.xywhr[i]]
            print(f"  - {cls} conf={conf:.2f} pos=({cx:.0f},{cy:.0f}) "
                  f"tam={w:.0f}x{h:.0f} ang={math.degrees(ang):+.1f}°")


if __name__ == "__main__":
    main()
