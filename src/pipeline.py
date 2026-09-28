"""
Utilidades compartilhadas: caminhos dos modelos, SAM refiner, perspective warp.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"

YOLO_WEIGHTS_PT = MODELS_DIR / "yolo_obb.pt"
YOLO_WEIGHTS_OV = MODELS_DIR / "yolo_obb_openvino_model"
ANOMALIB_CKPT = MODELS_DIR / "anomalib.ckpt"
SAM_WEIGHTS = MODELS_DIR / "mobile_sam.pt"


def resolve_yolo_weights(prefer_openvino: bool = True):
    """Prefere o modelo OpenVINO (rápido em CPU Intel) se existir; senão volta pro .pt."""
    if prefer_openvino and YOLO_WEIGHTS_OV.exists():
        return YOLO_WEIGHTS_OV
    return YOLO_WEIGHTS_PT


YOLO_WEIGHTS = resolve_yolo_weights()


def resolve_devices(choice: str):
    """
    Converte 'cpu' | 'intel' | 'intel:gpu' | 'intel:npu' | 'nvidia' | 'cuda'
    em (yolo_device, sam_device).

    YOLO aceita 'cpu', 'cuda', 'intel:gpu', 'intel:npu' (OpenVINO).
    SAM (PyTorch puro) só aceita 'cpu' ou 'cuda' — se o usuário pediu Intel,
    o SAM cai pra 'cpu' automaticamente.
    """
    c = choice.lower().strip()
    if c in ("intel", "intel:gpu"):
        return "intel:gpu", "cpu"
    if c == "intel:npu":
        return "intel:npu", "cpu"
    if c in ("nvidia", "cuda"):
        return "cuda", "cuda"
    if c == "cpu":
        return "cpu", "cpu"
    raise ValueError(f"device desconhecido: {choice}")


class SamRefiner:
    """Refina bboxes do YOLO usando SAM: 'rect' = OBB, 'quad' = quadrilátero livre."""

    def __init__(self, model_path=SAM_WEIGHTS, device=None, mode="rect"):
        from ultralytics import SAM
        self.model = SAM(str(model_path))
        self.device = device
        self.mode = mode

    def _rect_from_contour(self, contour):
        rect = cv2.minAreaRect(contour)
        return cv2.boxPoints(rect).astype(np.float32)

    def _quad_from_contour(self, contour):
        peri = cv2.arcLength(contour, True)
        for eps in (0.02, 0.03, 0.04, 0.05, 0.01, 0.06, 0.08):
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
        hull = cv2.convexHull(contour)
        if len(hull) >= 4:
            for eps in (0.05, 0.08, 0.1, 0.15, 0.2):
                approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
                if len(approx) == 4:
                    return approx.reshape(4, 2).astype(np.float32)
        return self._rect_from_contour(contour)

    def refine(self, img, corners, min_area_ratio=0.1):
        pts = corners.astype(np.float32)
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        try:
            results = self.model.predict(
                source=img, bboxes=[[x1, y1, x2, y2]],
                verbose=False, device=self.device,
            )
        except Exception as e:
            print(f"[SAM] falhou: {e}")
            return None
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            return None
        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < (x2 - x1) * (y2 - y1) * min_area_ratio:
            return None
        return self._quad_from_contour(largest) if self.mode == "quad" else self._rect_from_contour(largest)


def order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def perspective_warp(img, quad, out_w=None, out_h=None):
    """Planifica um quadrilátero em uma imagem retangular frontal."""
    ordered = order_quad(quad)
    tl, tr, br, bl = ordered
    W = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))) if out_w is None else out_w
    H = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))) if out_h is None else out_h
    if W < 5 or H < 5:
        return None
    if W > H:
        W, H = H, W
        ordered = np.array([bl, tl, tr, br], dtype=np.float32)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H))


def draw_quad(img, quad, label, color, show_raw=None, raw_color=(120, 120, 120)):
    if show_raw is not None:
        cv2.polylines(img, [show_raw.astype(np.int32)], True, raw_color, 1)
    pts = quad.astype(np.int32)
    cv2.polylines(img, [pts], True, color, 2)
    x, y = pts[0]
    cv2.putText(img, label, (int(x), max(15, int(y) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
