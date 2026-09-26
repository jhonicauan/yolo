import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Treinar YOLO em um dataset.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="my_dataset",
        help="Nome da pasta do dataset (ex: my_dataset, bottledata) OU caminho direto pro data.yaml.",
    )
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Modelo base ou peso .pt.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0, help="0 = sem early stopping.")
    parser.add_argument("--name", type=str, default=None, help="Nome do experimento (padrão: nome do dataset).")
    parser.add_argument("--device", type=str, default=None, help="cpu, 0, 0,1 etc. Padrão: auto.")
    args = parser.parse_args()

    # Resolve caminho do data.yaml
    data_arg = Path(args.dataset)
    if data_arg.suffix in (".yaml", ".yml") and data_arg.exists():
        data_path = data_arg
        dataset_name = data_arg.parent.name
    else:
        data_path = data_arg / "data.yaml"
        dataset_name = data_arg.name
        if not data_path.exists():
            raise FileNotFoundError(f"Não achei {data_path}")

    if args.device is not None:
        device = args.device
    else:
        device = 0 if torch.cuda.is_available() else "cpu"

    run_name = args.name or f"exp-{dataset_name}"
    print(f"Dataset : {data_path}")
    print(f"Modelo  : {args.model}")
    print(f"Device  : {device} (cuda: {torch.cuda.is_available()})")
    print(f"Run     : runs/train/{run_name}")
    print(f"Epochs  : {args.epochs}, batch: {args.batch}, imgsz: {args.imgsz}\n")

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project="runs/train",
        name=run_name,
        patience=args.patience,
        save=True,
        # augmentation moderado — o suficiente pra inclinação sem confundir o modelo
        degrees=30,        # rotação até ±30° (inclinações e deitadas leves)
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )

    metrics = model.val(verbose=False)
    recall = float(metrics.box.mr)
    print(f"\nR (Recall médio): {recall:.4f}")


if __name__ == "__main__":
    main()
