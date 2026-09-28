"""
Treina YOLO-OBB (bounding boxes rotacionadas) para detectar etiquetas.

Uso:
    python src/train_yolo.py caminho/do/dataset
    python src/train_yolo.py etiquetas --epochs 50 --batch 8
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Pasta do dataset OBB ou data.yaml.")
    parser.add_argument("--model", type=str, default="yolo26n-obb.pt",
                        help="Modelo base (pretrained) ou .pt pra fine-tuning.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    data = Path(args.dataset)
    data_yaml = data if data.suffix in (".yaml", ".yml") else data / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    run_name = args.name or f"obb-{data.stem}"

    print(f"Dataset : {data_yaml}")
    print(f"Modelo  : {args.model}")
    print(f"Device  : {device}\n")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=device, workers=args.workers,
        project="runs/obb", name=run_name, patience=args.patience,
    )
    print("\nPesos salvos em runs/obb/<name>/weights/best.pt")
    print("Copie para models/yolo_obb.pt pra usar como modelo padrão.")


if __name__ == "__main__":
    main()
