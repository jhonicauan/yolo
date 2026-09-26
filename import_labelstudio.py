"""
Importa um export do Label Studio (formato YOLO, .zip) e distribui as imagens
em train/val/test dentro de my_dataset/, tratando backgrounds corretamente.

Espera dentro do zip:
  images/  (as fotos)
  labels/  (arquivos .txt no formato YOLO — imagens sem objeto ficam sem .txt)
  classes.txt  (opcional — usado pra atualizar names no data.yaml)

Backgrounds = imagens sem .txt correspondente OU com .txt vazio.
Elas são copiadas normalmente e recebem um .txt vazio no destino
(que é como o YOLO reconhece uma imagem de background).
"""
import argparse
import random
import shutil
import zipfile
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=str, help="Caminho do zip exportado pelo Label Studio.")
    parser.add_argument("--dataset", type=str, default="my_dataset", help="Pasta do dataset destino.")
    parser.add_argument("--train", type=float, default=0.7, help="Proporção de treino.")
    parser.add_argument("--val", type=float, default=0.2, help="Proporção de validação.")
    parser.add_argument("--test", type=float, default=0.1, help="Proporção de teste.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="ls_", help="Prefixo pros arquivos novos.")
    parser.add_argument("--clean", action="store_true", help="Limpa splits antes de importar.")
    args = parser.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train+val+test deve somar 1.0 (soma atual: {total})")

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    dataset = Path(args.dataset)
    tmp_dir = dataset.parent / f".ls_extract_{zip_path.stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    print(f"Extraindo {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp_dir)

    # Localiza pastas images/ e labels/ (podem estar na raiz ou dentro de subpasta)
    images_dir = None
    labels_dir = None
    for p in tmp_dir.rglob("images"):
        if p.is_dir():
            images_dir = p
            break
    for p in tmp_dir.rglob("labels"):
        if p.is_dir():
            labels_dir = p
            break
    if not images_dir:
        raise RuntimeError("Não achei a pasta 'images/' no zip.")
    if not labels_dir:
        print("Aviso: nenhuma pasta 'labels/' encontrada — todas as imagens serão background.")
        labels_dir = tmp_dir / "labels_vazio"
        labels_dir.mkdir(exist_ok=True)

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        raise RuntimeError("Nenhuma imagem encontrada.")

    # Separa em (com anotação) e background
    annotated, backgrounds = [], []
    for img in images:
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists() and lbl.read_text().strip():
            annotated.append((img, lbl))
        else:
            backgrounds.append((img, None))

    print(f"Total: {len(images)} imagens ({len(annotated)} anotadas, {len(backgrounds)} backgrounds)")

    rng = random.Random(args.seed)
    rng.shuffle(annotated)
    rng.shuffle(backgrounds)

    def split_list(items):
        n = len(items)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        # o resto vai pro teste (evita perder itens por arredondamento)
        return {
            "train": items[:n_train],
            "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:],
        }

    splits_ann = split_list(annotated)
    splits_bg = split_list(backgrounds)

    # Cria/limpa pastas destino
    for split in ("train", "val", "test"):
        img_out = dataset / "images" / split
        lbl_out = dataset / "labels" / split
        if args.clean and img_out.exists():
            shutil.rmtree(img_out)
        if args.clean and lbl_out.exists():
            shutil.rmtree(lbl_out)
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    def copy_items(items, split, has_label):
        img_out = dataset / "images" / split
        lbl_out = dataset / "labels" / split
        for i, (img, lbl) in enumerate(items):
            tag = "ann" if has_label else "bg"
            new_name = f"{args.prefix}{tag}_{split}_{i:04d}{img.suffix.lower()}"
            shutil.copy2(img, img_out / new_name)
            dst_lbl = lbl_out / (Path(new_name).stem + ".txt")
            if has_label:
                shutil.copy2(lbl, dst_lbl)
            else:
                dst_lbl.write_text("")  # background = txt vazio

    for split in ("train", "val", "test"):
        copy_items(splits_ann[split], split, has_label=True)
        copy_items(splits_bg[split], split, has_label=False)
        print(f"  {split}: {len(splits_ann[split])} anotadas + {len(splits_bg[split])} backgrounds")

    # Remove caches antigos
    for cache in dataset.glob("labels/*.cache"):
        cache.unlink()
        print(f"cache removido: {cache}")

    # Atualiza data.yaml com nomes das classes, se classes.txt existir
    classes_txt = None
    for p in tmp_dir.rglob("classes.txt"):
        classes_txt = p
        break
    if classes_txt:
        names = [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        yaml_path = dataset / "data.yaml"
        lines = [
            "path: ./" + dataset.name,
            "",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
        ]
        for i, n in enumerate(names):
            lines.append(f"  {i}: {n}")
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"data.yaml atualizado com {len(names)} classes: {names}")

    shutil.rmtree(tmp_dir)
    print("\nOK. Rode 'python train.py'.")


if __name__ == "__main__":
    main()
