"""
Exporta o YOLO-OBB para OpenVINO (rápido em CPU Intel).

Uso:
    python src/export_openvino.py
    python src/export_openvino.py --imgsz 480 --half
"""
import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import MODELS_DIR, YOLO_WEIGHTS_PT, YOLO_WEIGHTS_OV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(YOLO_WEIGHTS_PT))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true",
                        help="Exporta em FP16 (mais rápido, ligeiramente menos preciso).")
    args = parser.parse_args()

    src = Path(args.weights)
    if not src.exists():
        raise SystemExit(f"Não achei os pesos: {src}")

    print(f"Exportando {src} para OpenVINO (imgsz={args.imgsz}, half={args.half})...")
    model = YOLO(str(src))
    exported = model.export(format="openvino", imgsz=args.imgsz, half=args.half)
    exported = Path(exported)
    print(f"Exportado em: {exported}")

    # Ultralytics gera <weights>_openvino_model/ ao lado do .pt.
    # Move pro nome esperado pelo pipeline (models/yolo_obb_openvino_model/).
    if exported.resolve() != YOLO_WEIGHTS_OV.resolve():
        if YOLO_WEIGHTS_OV.exists():
            shutil.rmtree(YOLO_WEIGHTS_OV)
        shutil.move(str(exported), str(YOLO_WEIGHTS_OV))
        print(f"Movido para: {YOLO_WEIGHTS_OV}")

    print("Pronto. O predict_camera.py já usa este modelo automaticamente.")


if __name__ == "__main__":
    main()
