import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def find_latest_weights():
    """Retorna o best.pt mais recente dentro de runs/, ou None."""
    candidates = list(Path("runs").rglob("best.pt"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def main():
    parser = argparse.ArgumentParser(description="Testar YOLO usando webcam.")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Caminho dos pesos treinados (padrão: best.pt mais recente).",
    )
    parser.add_argument("--source", type=int, default=0, help="Índice da câmera.")
    parser.add_argument("--conf", type=float, default=0.70, help="Confiança mínima.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem.")
    args = parser.parse_args()

    weights = args.weights or find_latest_weights()
    if not weights:
        raise FileNotFoundError("Nenhum best.pt encontrado em runs/. Treine o modelo primeiro.")
    print(f"Usando pesos: {weights}")

    model = YOLO(weights)

    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera {args.source}")

    print("Pressione 'q' para sair.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Falha ao ler frame da câmera.")
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )

            annotated = results[0].plot()

            cv2.imshow("YOLO - Camera", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
