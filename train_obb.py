"""
Treina YOLO-OBB (oriented bounding box) — captura garrafas em qualquer ângulo
com uma bbox rotacionada, permitindo recortes precisos.

Uso:
    python train_obb.py bottledata_obb
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="bottledata_obb",
                        help="Pasta do dataset OBB ou caminho pro data.yaml.")
    parser.add_argument("--model", type=str, default="yolo26n-obb.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    data_arg = Path(args.dataset)
    if data_arg.suffix in (".yaml", ".yml") and data_arg.exists():
        data_path = data_arg
        dataset_name = data_arg.parent.name
    else:
        data_path = data_arg / "data.yaml"
        dataset_name = data_arg.name
        if not data_path.exists():
            raise FileNotFoundError(data_path)

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    run_name = args.name or f"obb-{dataset_name}"

    print(f"Dataset : {data_path}")
    print(f"Modelo  : {args.model}")
    print(f"Device  : {device}")
    print(f"Run     : runs/obb/{run_name}\n")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project="runs/obb",
        name=run_name,
        patience=args.patience,
        save=True,
    )
    metrics = model.val(verbose=False)
    print(f"\nR (Recall médio): {float(metrics.box.mr):.4f}")


if __name__ == "__main__":
    main()
