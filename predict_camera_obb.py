"""
Câmera em tempo real com YOLO-OBB (bboxes rotacionadas).

Uso:
    python predict_camera_obb.py
    python predict_camera_obb.py --conf 0.5 --source 1
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def find_latest_obb_weights():
    root = Path("runs/obb")
    if not root.exists():
        return None
    candidates = list(root.rglob("best.pt"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def main():
    parser = argparse.ArgumentParser(description="Câmera com YOLO-OBB.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Pesos OBB (padrão: mais recente em runs/obb).")
    parser.add_argument("--source", type=int, default=0, help="Índice da câmera.")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cam-w", type=int, default=1280)
    parser.add_argument("--cam-h", type=int, default=720)
    parser.add_argument("--window-w", type=int, default=1280)
    parser.add_argument("--window-h", type=int, default=720)
    args = parser.parse_args()

    weights = args.weights or find_latest_obb_weights()
    if not weights:
        raise FileNotFoundError("Nenhum best.pt em runs/obb/. Treine com train_obb.py primeiro.")
    print(f"Usando pesos: {weights}")

    model = YOLO(weights)

    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Não consegui abrir a câmera {args.source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    print(f"Câmera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    win = "YOLO-OBB - Camera"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.window_w, args.window_h)

    print("Pressione 'q' pra sair.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Falha ao ler frame.")
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )

            annotated = results[0].plot()
            annotated = cv2.resize(annotated, (args.window_w, args.window_h))
            cv2.imshow(win, annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
