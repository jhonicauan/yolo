"""
Detecção em tempo real com YOLO-OBB (opcionalmente com refinamento por SAM).

Uso:
    python src/predict_camera.py
    python src/predict_camera.py --sam --sam-mode quad
    python src/predict_camera.py --source 1 --conf 0.6
"""
import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import YOLO_WEIGHTS, SAM_WEIGHTS, SamRefiner, draw_quad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(YOLO_WEIGHTS))
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cam-w", type=int, default=1280)
    parser.add_argument("--cam-h", type=int, default=720)
    parser.add_argument("--sam", action="store_true", help="Ativa refinamento por SAM.")
    parser.add_argument("--sam-model", type=str, default=str(SAM_WEIGHTS))
    parser.add_argument("--sam-mode", choices=["rect", "quad"], default="rect")
    parser.add_argument("--refine-every", type=int, default=3)
    args = parser.parse_args()

    print(f"Pesos: {args.weights}")
    print(f"SAM  : {'ligado (' + args.sam_mode + ')' if args.sam else 'desligado'}")

    model = YOLO(args.weights)
    sam = SamRefiner(args.sam_model, mode=args.sam_mode) if args.sam else None

    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    if not cap.isOpened():
        raise RuntimeError(f"Câmera {args.source} indisponível.")

    cv2.namedWindow("YOLO-OBB", cv2.WINDOW_NORMAL)
    print("Pressione 'q' pra sair.")

    frame_idx = 0
    last_refined = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            results = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

            if sam is None:
                annotated = results[0].plot()
            else:
                annotated = frame.copy()
                obb = results[0].obb
                if obb is not None and len(obb) > 0:
                    corners = obb.xyxyxyxy.cpu().numpy()
                    confs = obb.conf.cpu().numpy()
                    clss = obb.cls.int().cpu().numpy()
                    if frame_idx % max(1, args.refine_every) == 0 or len(last_refined) != len(corners):
                        last_refined = [sam.refine(frame, c) for c in corners]
                    for i, raw in enumerate(corners):
                        label = f"{model.names[int(clss[i])]} {float(confs[i]):.2f}"
                        refined = last_refined[i] if i < len(last_refined) else None
                        if refined is not None:
                            draw_quad(annotated, refined, label, (0, 255, 0))
                        else:
                            draw_quad(annotated, raw, label, (0, 200, 200))

            cv2.imshow("YOLO-OBB", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
