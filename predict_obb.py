"""
Predict com modelo YOLO-OBB (bboxes rotacionadas).

Uso:
    python predict_obb.py foto.jpg
    python predict_obb.py foto.jpg --conf 0.5
    python predict_obb.py pasta/  --conf 0.7
    python predict_obb.py video.mp4
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def find_latest_obb_weights():
    root = Path("runs/obb")
    if not root.exists():
        return None
    candidates = list(root.rglob("best.pt"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def main():
    parser = argparse.ArgumentParser(description="Predict YOLO-OBB em imagem/pasta/video.")
    parser.add_argument("source", type=str, help="Imagem, pasta ou vídeo.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Caminho dos pesos OBB (padrão: mais recente em runs/obb).")
    parser.add_argument("--conf", type=float, default=0.50, help="Confiança mínima.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save-dir", type=str, default="runs/obb_predict")
    args = parser.parse_args()

    weights = args.weights or find_latest_obb_weights()
    if not weights:
        raise FileNotFoundError("Nenhum best.pt em runs/obb/. Treine com train_obb.py primeiro.")
    print(f"Usando pesos: {weights}")

    src = Path(args.source)
    if not src.exists():
        raise FileNotFoundError(src)

    model = YOLO(weights)

    results = model.predict(
        source=str(src),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=args.save_dir,
        name="obb",
        exist_ok=True,
    )

    for r in results:
        name = Path(r.path).name
        obb = r.obb
        n = 0 if obb is None else len(obb)
        print(f"\n{name}: {n} detecção(ões)")
        if obb is None or n == 0:
            continue
        for i in range(n):
            cls_id = int(obb.cls[i])
            cls_name = model.names[cls_id]
            conf = float(obb.conf[i])
            cx, cy, w, h, r_ang = [float(v) for v in obb.xywhr[i]]
            import math
            ang_deg = math.degrees(r_ang)
            print(f"  - {cls_name} conf={conf:.2f} centro=({cx:.0f},{cy:.0f}) "
                  f"tam={w:.0f}x{h:.0f} ang={ang_deg:+.1f}°")


if __name__ == "__main__":
    main()
