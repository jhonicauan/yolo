"""
Treina EfficientAd (anomalib) nos crops planificados das etiquetas.

Estrutura esperada em crops_dataset/:
    train/good/     imagens normais
    test/good/      normais de teste
    test/defeito/   imagens com defeito (opcional)

Uso:
    python src/train_anomalib.py
    python src/train_anomalib.py --epochs 200 --size medium
"""
import argparse
import os
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

from pathlib import Path

from torchvision.transforms.v2 import Compose, Resize

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.pre_processing import PreProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("crops_dataset"))
    parser.add_argument("--size", choices=["small", "medium"], default="small")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    datamodule = Folder(
        name="etiqueta",
        root=str(args.data),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/defeito" if (args.data / "test/defeito").exists() else None,
        train_batch_size=1, eval_batch_size=1, num_workers=0,
    )

    pre = PreProcessor(transform=Compose([Resize((args.imgsz, args.imgsz))]))
    model = EfficientAd(model_size=args.size, pre_processor=pre)

    engine = Engine(max_epochs=args.epochs, accelerator="auto", devices=1)
    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)

    print("\nCheckpoint salvo em results/EfficientAd/etiqueta/<vN>/weights/lightning/model.ckpt")
    print("Copie para models/anomalib.ckpt pra usar como modelo padrão.")


if __name__ == "__main__":
    main()
