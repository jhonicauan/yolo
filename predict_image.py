import argparse
from pathlib import Path

from ultralytics import YOLO


def find_latest_weights():
    """Retorna o best.pt mais recente dentro de runs/, ou None."""
    candidates = list(Path("runs").rglob("best.pt"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def main():
    parser = argparse.ArgumentParser(description="Testar YOLO em uma imagem.")
    parser.add_argument("image", type=str, help="Caminho da imagem de entrada.")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Caminho dos pesos treinados (padrão: best.pt mais recente).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confiança mínima.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem.")
    parser.add_argument(
        "--save-dir", type=str, default="runs/predict", help="Pasta de saída."
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    weights = args.weights or find_latest_weights()
    if not weights:
        raise FileNotFoundError("Nenhum best.pt encontrado em runs/. Treine o modelo primeiro.")
    print(f"Usando pesos: {weights}")

    model = YOLO(weights)

    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=args.save_dir,
        name="img",
        exist_ok=True,
    )

    for r in results:
        print(f"Detecções em {image_path.name}: {len(r.boxes)}")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = [float(x) for x in box.xyxy[0]]
            print(f"  - {cls_name} conf={conf:.2f} bbox={xyxy}")


if __name__ == "__main__":
    main()
