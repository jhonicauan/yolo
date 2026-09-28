"""
Detecta etiquetas numa imagem, planifica cada uma e salva os crops isolados.
Sem análise de anomalia.

Uso:
    python src/predict_crop.py samples/errado.jpeg
    python src/predict_crop.py samples/errado.jpeg --device intel
    python src/predict_crop.py samples/errado.jpeg --sam --device intel
    python src/predict_crop.py samples/errado.jpeg --show

Saídas em runs/crops/:
    <nome>__annotated.jpg   imagem original com as detecções desenhadas
    <nome>__crop0.png       crop planificado da detecção 0
    <nome>__crop1.png       crop planificado da detecção 1
    ...
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (
    YOLO_WEIGHTS, SAM_WEIGHTS,
    SamRefiner, perspective_warp, draw_quad, resolve_devices,
)


OUT_DIR = Path("runs/crops")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Imagem de entrada.")
    parser.add_argument("--weights", type=str, default=str(YOLO_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--sam", action="store_true",
                        help="Refina o polígono com SAM antes de planificar.")
    parser.add_argument("--sam-model", type=str, default=str(SAM_WEIGHTS))
    parser.add_argument("--sam-mode", choices=["rect", "quad"], default="quad")
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "intel", "intel:gpu", "intel:npu", "nvidia", "cuda"],
        help="Onde rodar o YOLO. SAM usa CUDA se disponível, senão CPU.",
    )
    parser.add_argument("--out", type=str, default=str(OUT_DIR),
                        help="Pasta de saída.")
    parser.add_argument("--show", action="store_true",
                        help="Abre janelas com a imagem anotada e os crops.")
    args = parser.parse_args()

    yolo_device, sam_device = resolve_devices(args.device)

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Não consegui ler {img_path}")

    using_ov = Path(args.weights).name.endswith("_openvino_model")
    print(f"Pesos      : {args.weights}  {'[OpenVINO]' if using_ov else '[PyTorch]'}")
    print(f"YOLO device: {yolo_device}")
    print(f"SAM        : {'ligado (' + args.sam_mode + ') em ' + sam_device if args.sam else 'desligado'}")
    print(f"Imagem     : {img_path}\n")

    model = YOLO(args.weights, task="obb")
    results = model.predict(source=img, conf=args.conf, imgsz=args.imgsz,
                            device=yolo_device, verbose=False)
    obb = results[0].obb
    if obb is None or len(obb) == 0:
        print("Nenhuma etiqueta detectada.")
        return

    print(f"Detectou {len(obb)} etiqueta(s).")

    sam = SamRefiner(args.sam_model, mode=args.sam_mode, device=sam_device) if args.sam else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated = img.copy()
    corners_all = obb.xyxyxyxy.cpu().numpy()
    confs = obb.conf.cpu().numpy()
    clss = obb.cls.int().cpu().numpy()

    crops = []
    for i, raw in enumerate(corners_all):
        label = f"{model.names[int(clss[i])]} {float(confs[i]):.2f}"

        quad = sam.refine(img, raw) if sam is not None else None
        used = quad if quad is not None else raw
        color = (0, 255, 0) if quad is not None else (0, 200, 200)
        draw_quad(annotated, used, label, color)

        warp = perspective_warp(img, used)
        if warp is None:
            print(f"  crop {i}: falhou no warp, ignorado.")
            continue
        crop_path = out_dir / f"{img_path.stem}__crop{i}.png"
        cv2.imwrite(str(crop_path), warp)
        crops.append((i, crop_path, warp))
        print(f"  crop {i}: {warp.shape[1]}x{warp.shape[0]}  ->  {crop_path}")

    ann_path = out_dir / f"{img_path.stem}__annotated.jpg"
    cv2.imwrite(str(ann_path), annotated)
    print(f"\nAnotada em : {ann_path}")
    print(f"Crops em   : {out_dir}/")

    if args.show:
        cv2.namedWindow("Deteccoes", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Deteccoes", 1200, 800)
        cv2.imshow("Deteccoes", annotated)
        for i, path, warp in crops:
            win = f"Crop {i}"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, min(600, warp.shape[1] * 2), min(400, warp.shape[0] * 2))
            cv2.imshow(win, warp)
        print("\nPressione qualquer tecla nas janelas para fechar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
