"""
Tracking OBB em tempo real com contagem de garrafas passando.
Mesmo comportamento do track_camera.py mas com bboxes rotacionadas.

Uso:
    python track_camera_obb.py                                # webcam
    python track_camera_obb.py --source video.mp4             # vídeo
    python track_camera_obb.py --save saida.mp4               # salva
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--source", default="0", help="Índice da câmera ou caminho do vídeo.")
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cam-w", type=int, default=1280)
    parser.add_argument("--cam-h", type=int, default=720)
    parser.add_argument("--window-w", type=int, default=1280)
    parser.add_argument("--window-h", type=int, default=720)
    parser.add_argument("--tracker", type=str, default="bytetrack_custom.yaml",
                        help="bytetrack.yaml (padrão Ultralytics), bytetrack_custom.yaml (segura ID mais tempo) ou botsort.yaml.")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--persist-frames", type=int, default=150,
                        help="Frames sem detecção antes de considerar 'passou' (~5s a 30fps).")
    args = parser.parse_args()

    weights = args.weights or find_latest_obb_weights()
    if not weights:
        raise FileNotFoundError("Nenhum best.pt em runs/obb/. Treine com train_obb.py.")
    print(f"Pesos   : {weights}")
    print(f"Tracker : {args.tracker}")

    model = YOLO(weights)

    src = args.source
    try:
        src = int(src)
    except ValueError:
        pass
    is_camera = isinstance(src, int)

    if is_camera:
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    else:
        cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Não consegui abrir: {args.source}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Fonte   : {'webcam' if is_camera else args.source}  ({W}x{H} @ {fps:.1f} fps)")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (args.window_w, args.window_h))

    win = "YOLO-OBB Track"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.window_w, args.window_h)

    seen_ids = set()
    last_seen_frame = {}
    finished_ids = set()
    frame_idx = 0

    print("Pressione 'q' pra sair.\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            results = model.track(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                tracker=args.tracker,
                persist=True,
                verbose=False,
            )

            current_ids = set()
            obb = results[0].obb
            if obb is not None and obb.id is not None:
                ids = obb.id.int().cpu().tolist()
                for bid in ids:
                    current_ids.add(bid)
                    seen_ids.add(bid)
                    last_seen_frame[bid] = frame_idx

            for bid, last in list(last_seen_frame.items()):
                if bid not in current_ids and (frame_idx - last) > args.persist_frames:
                    finished_ids.add(bid)

            annotated = results[0].plot()

            hud = [
                f"Na tela : {len(current_ids)}",
                f"Total vistas: {len(seen_ids)}",
                f"Passaram : {len(finished_ids)}",
            ]
            for i, line in enumerate(hud):
                y = 30 + i * 30
                cv2.putText(annotated, line, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(annotated, line, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            annotated = cv2.resize(annotated, (args.window_w, args.window_h))
            cv2.imshow(win, annotated)
            if writer:
                writer.write(annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\nResumo:")
    print(f"  Únicas vistas  : {len(seen_ids)}")
    print(f"  Passaram total : {len(finished_ids)}")


if __name__ == "__main__":
    main()
