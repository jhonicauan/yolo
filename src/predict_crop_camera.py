"""
Detecção ao vivo com YOLO-OBB + planificação (crop isolado) das etiquetas.
Mostra a câmera anotada e um mosaico com os crops isolados atualizados a cada frame.

Uso:
    python src/predict_crop_camera.py
    python src/predict_crop_camera.py --device intel
    python src/predict_crop_camera.py --device intel --sam
    python src/predict_crop_camera.py --save-crops   # salva cada crop em runs/crops_live/

Teclas:
    q     sair
    s     salvar o frame atual + crops em runs/crops_live/
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (
    YOLO_WEIGHTS, SAM_WEIGHTS,
    SamRefiner, perspective_warp, draw_quad, resolve_devices,
)


CROP_TILE = 256    # cada crop no mosaico é redimensionado pra caber num quadro CROP_TILE x CROP_TILE
MOSAIC_COLS = 4    # colunas do mosaico de crops
OUT_DIR = Path("runs/crops_live")


def fit_into_tile(crop, tile):
    """Redimensiona mantendo aspect ratio e centraliza num tile preto tile x tile."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((tile, tile, 3), dtype=np.uint8)
    scale = tile / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
    y0 = (tile - nh) // 2
    x0 = (tile - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def build_mosaic(crops, tile=CROP_TILE, cols=MOSAIC_COLS):
    if not crops:
        canvas = np.zeros((tile, tile * cols, 3), dtype=np.uint8)
        cv2.putText(canvas, "Sem deteccoes", (10, tile // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        return canvas

    tiles = [fit_into_tile(c, tile) for c in crops]
    for idx, t in enumerate(tiles):
        cv2.putText(t, f"#{idx}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros((tile, tile, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(YOLO_WEIGHTS))
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cam-w", type=int, default=1280)
    parser.add_argument("--cam-h", type=int, default=720)
    parser.add_argument("--view-w", type=int, default=1280,
                        help="Largura da janela da câmera (px).")
    parser.add_argument("--view-h", type=int, default=720,
                        help="Altura da janela da câmera (px).")
    parser.add_argument("--sam", action="store_true", help="Refina com SAM.")
    parser.add_argument("--sam-model", type=str, default=str(SAM_WEIGHTS))
    parser.add_argument("--sam-mode", choices=["rect", "quad"], default="quad")
    parser.add_argument("--refine-every", type=int, default=3,
                        help="Refina SAM a cada N frames (>=1).")
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "intel", "intel:gpu", "intel:npu", "nvidia", "cuda"],
        help="Onde rodar o YOLO. SAM usa CUDA se disponível, senão CPU.",
    )
    parser.add_argument("--save-crops", action="store_true",
                        help="Salva TODO crop de TODO frame em runs/crops_live/.")
    args = parser.parse_args()

    yolo_device, sam_device = resolve_devices(args.device)

    using_ov = Path(args.weights).name.endswith("_openvino_model")
    print(f"Pesos      : {args.weights}  {'[OpenVINO]' if using_ov else '[PyTorch]'}")
    print(f"YOLO device: {yolo_device}")
    print(f"SAM        : {'ligado (' + args.sam_mode + ') em ' + sam_device if args.sam else 'desligado'}")

    model = YOLO(args.weights, task="obb")
    sam = SamRefiner(args.sam_model, mode=args.sam_mode, device=sam_device) if args.sam else None

    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    if not cap.isOpened():
        raise RuntimeError(f"Câmera {args.source} indisponível.")

    cv2.namedWindow("YOLO-OBB", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO-OBB", args.view_w, args.view_h)
    cv2.namedWindow("Crops", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crops", CROP_TILE * MOSAIC_COLS, CROP_TILE * 2)
    print("Teclas: q sai | s salva frame + crops")

    if args.save_crops or True:  # cria a pasta pra tecla 's' também
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    last_refined = []
    fps_t0 = time.time()
    fps_n = 0
    fps_txt = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            results = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz,
                                    device=yolo_device, verbose=False)
            annotated = frame.copy()
            crops = []
            obb = results[0].obb
            if obb is not None and len(obb) > 0:
                corners_all = obb.xyxyxyxy.cpu().numpy()
                confs = obb.conf.cpu().numpy()
                clss = obb.cls.int().cpu().numpy()

                if sam is not None and (
                    frame_idx % max(1, args.refine_every) == 0
                    or len(last_refined) != len(corners_all)
                ):
                    last_refined = [sam.refine(frame, c) for c in corners_all]

                for i, raw in enumerate(corners_all):
                    label = f"{model.names[int(clss[i])]} {float(confs[i]):.2f}"
                    refined = last_refined[i] if (sam is not None and i < len(last_refined)) else None
                    used = refined if refined is not None else raw
                    color = (0, 255, 0) if refined is not None else (0, 200, 200)
                    draw_quad(annotated, used, label, color)

                    warp = perspective_warp(frame, used)
                    if warp is not None:
                        crops.append(warp)
                        if args.save_crops:
                            cv2.imwrite(str(OUT_DIR / f"f{frame_idx:06d}_c{i}.png"), warp)

            # FPS
            fps_n += 1
            if fps_n >= 10:
                dt = time.time() - fps_t0
                fps_txt = f"{fps_n/dt:.1f} FPS"
                fps_t0 = time.time()
                fps_n = 0
            if fps_txt:
                cv2.putText(annotated, fps_txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("YOLO-OBB", annotated)
            cv2.imshow("Crops", build_mosaic(crops))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(OUT_DIR / f"{ts}_frame.jpg"), annotated)
                for i, warp in enumerate(crops):
                    cv2.imwrite(str(OUT_DIR / f"{ts}_crop{i}.png"), warp)
                print(f"Salvo em {OUT_DIR}/ ({len(crops)} crops).")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
