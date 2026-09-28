"""
Extrai de um zip apenas as imagens que TEM arquivo .txt (mesmo vazio =
background revisado). Ignora só as sem .txt (não revisadas).

Por padrão já gera um dataset pronto pro YOLO com train/val/test + data.yaml.

Uso:
    python extract_annotated.py export.zip etiquetas
    python extract_annotated.py export.zip etiquetas --train 0.8 --val 0.15 --test 0.05
    python extract_annotated.py export.zip etiquetas --no-split       # só copia plano
"""
import argparse
import random
import shutil
import zipfile
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=str, help="Caminho do zip.")
    parser.add_argument("output", type=str, help="Pasta destino.")
    parser.add_argument("--train", type=float, default=0.7, help="Proporção de treino.")
    parser.add_argument("--val", type=float, default=0.2, help="Proporção de validação.")
    parser.add_argument("--test", type=float, default=0.1, help="Proporção de teste.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="", help="Prefixo pros arquivos copiados.")
    parser.add_argument("--keep-names", action="store_true", help="Mantém nomes originais.")
    parser.add_argument("--no-split", action="store_true",
                        help="Não divide em train/val/test — só copia plano em images/ + labels/.")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_dir.parent / f".extract_ann_{zip_path.stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    print(f"Extraindo {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp_dir)

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
        shutil.rmtree(tmp_dir)
        raise RuntimeError("Não achei a pasta 'images/' no zip.")
    if not labels_dir:
        shutil.rmtree(tmp_dir)
        raise RuntimeError("Não achei a pasta 'labels/' no zip.")

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        shutil.rmtree(tmp_dir)
        raise RuntimeError("Nenhuma imagem encontrada.")

    print(f"Total de imagens no zip: {len(images)}")

    annotated = []
    ignored = 0
    empty_labels = 0
    for img in images:
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            annotated.append((img, lbl))
            if not lbl.read_text().strip():
                empty_labels += 1
        else:
            ignored += 1

    print(f"  Com .txt   : {len(annotated)} ({empty_labels} vazios = backgrounds revisados)")
    print(f"  Ignoradas  : {ignored} (sem .txt = não revisadas)")

    if not annotated:
        shutil.rmtree(tmp_dir)
        print("Nenhuma imagem pra copiar.")
        return

    # === Modo simples (sem split) ===
    if args.no_split:
        img_out = out_dir / "images"
        lbl_out = out_dir / "labels"
        img_out.mkdir(exist_ok=True)
        lbl_out.mkdir(exist_ok=True)
        for i, (img, lbl) in enumerate(annotated):
            base = img.stem if args.keep_names else f"{args.prefix}{i:04d}"
            shutil.copy2(img, img_out / f"{base}{img.suffix.lower()}")
            shutil.copy2(lbl, lbl_out / f"{base}.txt")
        shutil.rmtree(tmp_dir)
        print(f"\nOK. {len(annotated)} imagens + labels copiados em {out_dir}")
        return

    # === Modo dataset YOLO (com split + data.yaml) ===
    total_prop = args.train + args.val + args.test
    if abs(total_prop - 1.0) > 1e-6:
        shutil.rmtree(tmp_dir)
        raise ValueError(f"train+val+test deve somar 1.0 (atual: {total_prop})")

    # separa anotadas de backgrounds pra distribuir proporcional em cada split
    ann_only = [(i, l) for i, l in annotated if l.read_text().strip()]
    bg_only = [(i, l) for i, l in annotated if not l.read_text().strip()]
    rng = random.Random(args.seed)
    rng.shuffle(ann_only)
    rng.shuffle(bg_only)

    def split_list(items):
        n = len(items)
        n_tr = int(n * args.train)
        n_v = int(n * args.val)
        return {
            "train": items[:n_tr],
            "val": items[n_tr:n_tr + n_v],
            "test": items[n_tr + n_v:],
        }

    splits_ann = split_list(ann_only)
    splits_bg = split_list(bg_only)

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def copy_items(items, split, tag):
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        for i, (img, lbl) in enumerate(items):
            base = img.stem if args.keep_names else f"{args.prefix}{tag}_{split}_{i:04d}"
            shutil.copy2(img, img_out / f"{base}{img.suffix.lower()}")
            shutil.copy2(lbl, lbl_out / f"{base}.txt")

    for split in ("train", "val", "test"):
        copy_items(splits_ann[split], split, "ann")
        copy_items(splits_bg[split], split, "bg")
        print(f"  {split}: {len(splits_ann[split])} anotadas + {len(splits_bg[split])} backgrounds")

    # data.yaml com nomes das classes do classes.txt do zip
    classes_txt = None
    for p in tmp_dir.rglob("classes.txt"):
        classes_txt = p
        break

    names = ["object"]
    if classes_txt:
        names = [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines()
                 if line.strip()]

    yaml_path = out_dir / "data.yaml"
    lines = [
        "path: " + str(out_dir.resolve()).replace("\\", "/"),
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
    print(f"\n  data.yaml criado com {len(names)} classe(s): {names}")

    shutil.rmtree(tmp_dir)
    print(f"\nOK. Dataset pronto em {out_dir}. Rode: python train_obb.py {out_dir.name}")


if __name__ == "__main__":
    main()
