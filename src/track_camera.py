"""
Tracking OBB em tempo real com contagem de etiquetas.

Uso:
    python src/track_camera.py
    python src/track_camera.py --source video.mp4 --save out.mp4
    python src/track_camera.py --tracker botsort
"""
import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import YOLO_WEIGHTS, CONFIGS_DIR, resolve_devices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(YOLO_WEIGHTS))
    parser.add_argument("--source", default="0", help="Câmera (int) ou vídeo (path).")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--persist-frames", type=int, default=150)
    parser.add_argument("--cam-w", type=int, default=1280)
    parser.add_argument("--cam-h", type=int, default=720)
    parser.add_argument("--view-w", type=int, default=1600,
                        help="Largura da janela (px). 0 = tela cheia.")
    parser.add_argument("--view-h", type=int, default=900,
                        help="Altura da janela (px). 0 = tela cheia.")
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "intel", "intel:gpu", "intel:npu", "nvidia", "cuda"],
        help="Onde rodar o YOLO. 'intel' exige modelo exportado com src/export_openvino.py.",
    )
    args = parser.parse_args()

    yolo_device, _ = resolve_devices(args.device)

    tracker_yaml = CONFIGS_DIR / f"{args.tracker}.yaml"
    using_ov = Path(args.weights).name.endswith("_openvino_model")
    print(f"Pesos      : {args.weights}  {'[OpenVINO]' if using_ov else '[PyTorch]'}")
    print(f"YOLO device: {yolo_device}")
    print(f"Tracker    : {tracker_yaml}")

    model = YOLO(args.weights, task="obb")
    src = args.source
    try:
        src = int(src)
    except ValueError:
        pass

    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    else:
        cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Fonte indisponível: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (args.cam_w, args.cam_h))

    cv2.namedWindow("Track", cv2.WINDOW_NORMAL)
    if args.view_w == 0 or args.view_h == 0:
        cv2.setWindowProperty("Track", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow("Track", args.view_w, args.view_h)
    seen, finished, last_seen = set(), set(), {}
    frame_idx = 0
    print("Pressione 'q' pra sair.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            results = model.track(source=frame, conf=args.conf, imgsz=args.imgsz,
                                  tracker=str(tracker_yaml), persist=True,
                                  device=yolo_device, verbose=False)
            annotated = results[0].plot()
            current = set()
            obb = results[0].obb
            if obb is not None and obb.id is not None:
                for tid in obb.id.int().cpu().tolist():
                    current.add(tid)
                    seen.add(tid)
                    last_seen[tid] = frame_idx

            for tid, last in list(last_seen.items()):
                if tid not in current and frame_idx - last > args.persist_frames:
                    finished.add(tid)

            hud = [f"Na tela: {len(current)}", f"Total: {len(seen)}", f"Passaram: {len(finished)}"]
            for i, line in enumerate(hud):
                y = 30 + i * 30
                cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Track", annotated)
            if writer:
                writer.write(annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\nÚnicas: {len(seen)}  |  Passaram: {len(finished)}")


if __name__ == "__main__":
    main()
