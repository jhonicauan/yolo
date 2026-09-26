import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Cortar região de um vídeo (ex: só a webcam).")
    parser.add_argument("video", type=str, help="Caminho do vídeo de entrada.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho de saída (padrão: <video>_crop.mp4 ao lado do original).",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Coordenadas x,y,w,h (pula seleção interativa).",
    )
    args = parser.parse_args()

    in_path = Path(args.video)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_crop.mp4")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError("Não consegui abrir o vídeo.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Vídeo: {W}x{H}, {fps:.2f} fps, {total} frames")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Vídeo sem frames.")

    if args.roi:
        x, y, w, h = [int(v) for v in args.roi.split(",")]
    else:
        print("\nDesenhe o retângulo com o mouse (a área da câmera).")
        print("Pressione ENTER ou ESPAÇO pra confirmar, C pra cancelar.\n")
        x, y, w, h = cv2.selectROI("Selecione a area", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        if w == 0 or h == 0:
            print("Cancelado.")
            return

    print(f"ROI: x={x} y={y} w={w} h={h}")

    # dimensões pares (exigido por alguns encoders)
    w -= w % 2
    h -= h % 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Não consegui criar o vídeo de saída.")

    # reinicia o vídeo do zero
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y:y + h, x:x + w]
        writer.write(crop)
        n += 1
        if n % 60 == 0:
            print(f"  processados {n}/{total}")

    cap.release()
    writer.release()
    print(f"\nOK: {out_path}  ({n} frames)")


if __name__ == "__main__":
    main()
