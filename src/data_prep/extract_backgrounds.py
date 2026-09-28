"""
Extrai de um zip (export do Label Studio ou similar) APENAS as imagens
que são background — ou seja, imagens sem anotação (.txt inexistente
ou vazio) — e copia pra uma pasta destino.

Uso:
    python extract_backgrounds.py export.zip backgrounds_out
    python extract_backgrounds.py export.zip bg_quarto --prefix quarto_
"""
import argparse
import shutil
import zipfile
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=str, help="Caminho do zip.")
    parser.add_argument("output", type=str, help="Pasta destino dos backgrounds.")
    parser.add_argument("--prefix", type=str, default="bg_",
                        help="Prefixo pros arquivos copiados (padrão: 'bg_').")
    parser.add_argument("--keep-names", action="store_true",
                        help="Mantém os nomes originais em vez de renumerar.")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extrai pra pasta temporária ao lado do output
    tmp_dir = out_dir.parent / f".extract_bg_{zip_path.stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    print(f"Extraindo {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp_dir)

    # Localiza pastas images/ e labels/ (podem estar na raiz ou em subpasta)
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

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        shutil.rmtree(tmp_dir)
        raise RuntimeError("Nenhuma imagem encontrada.")

    print(f"Total de imagens no zip: {len(images)}")

    # Separa backgrounds: sem .txt correspondente OU .txt vazio
    backgrounds = []
    annotated = []
    for img in images:
        if labels_dir:
            lbl = labels_dir / (img.stem + ".txt")
            if lbl.exists() and lbl.read_text().strip():
                annotated.append(img)
            else:
                backgrounds.append(img)
        else:
            # sem pasta labels/, tudo vira background
            backgrounds.append(img)

    print(f"  Anotadas   : {len(annotated)}")
    print(f"  Backgrounds: {len(backgrounds)}")

    if not backgrounds:
        shutil.rmtree(tmp_dir)
        print("Nenhum background pra copiar.")
        return

    # Copia pro destino
    for i, img in enumerate(backgrounds):
        if args.keep_names:
            new_name = img.name
        else:
            new_name = f"{args.prefix}{i:04d}{img.suffix.lower()}"
        shutil.copy2(img, out_dir / new_name)

    shutil.rmtree(tmp_dir)
    print(f"\nOK. {len(backgrounds)} backgrounds copiados em {out_dir}")


if __name__ == "__main__":
    main()
