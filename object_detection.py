"""
object_detection.py  —  YOLO-based target detection for the ZED pipeline
=========================================================================
Loads a YOLOv8 model, runs per-frame inference on a BGR image, samples
depth from the ZED depth map to estimate distance to each detected object,
and renders labelled bounding boxes + a detection HUD panel on the frame.

Design principles
-----------------
  • No ZED SDK calls — operates entirely on NumPy arrays passed in by main.py.
  • No SLAM logic — purely detection and visualisation.
  • Drop-in: call ObjectDetector.run(frame_bgr, depth_clean) once per frame.

Usage (from main.py or standalone testing)
------------------------------------------
  from object_detection import ObjectDetector, DetectionConfig

  cfg      = DetectionConfig()
  cfg.MODEL_PATH  = "models/best.pt"
  cfg.CONF_THRESH = 0.35

  detector = ObjectDetector(cfg)

  # In your main loop:
  result = detector.run(frame_bgr, depth_clean, frame_idx)
  vis    = detector.draw_overlay(frame_bgr, result, depth_clean, cfg)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

class DetectionConfig:
    """All tuneable detection parameters — edit attributes at runtime."""

    # Model
    MODEL_PATH   : str   = "models/best.pt"   # path to YOLOv8 .pt weights

    # Inference thresholds
    CONF_THRESH  : float = 0.35   # minimum confidence to report a detection
    IOU_THRESH   : float = 0.45   # NMS IoU overlap threshold

    # Depth sampling: use the centre crop of the bounding box
    # (fraction of box width/height)
    DEPTH_SAMPLE_FRAC : float = 0.30

    # Display
    BOX_THICKNESS    : int   = 2
    LABEL_FONT_SCALE : float = 0.55
    SHOW_CONF        : bool  = True    # include confidence in label
    SHOW_DEPTH       : bool  = True    # include depth estimate in label
    PANEL_HEIGHT     : int   = 60      # height of bottom HUD panel (pixels)

    # Per-class box colours (BGR).  Falls back to BOX_COLOR_DEFAULT for
    # any class_id not listed here.
    CLASS_COLORS     : dict  = field(default_factory=dict)
    BOX_COLOR_DEFAULT: tuple = (0, 200, 255)   # amber / yellow-orange

    def __init__(self):
        # Populate default colour map (can be overridden after construction)
        self.CLASS_COLORS = {}


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class DetectionResult:
    """Per-frame detection output."""

    # Each element: (x1, y1, x2, y2, conf, class_id, label, depth_m)
    boxes       : list  = field(default_factory=list)
    count       : int   = 0
    frame_idx   : int   = 0
    any_detected: bool  = False


# =============================================================================
# Detector
# =============================================================================

class ObjectDetector:
    """
    Wraps a YOLOv8 model.  Construct once; call run() every frame.

    Parameters
    ----------
    cfg : DetectionConfig
        Configuration object.  MODEL_PATH must point to a valid .pt file.

    Raises
    ------
    RuntimeError
        If the model file is not found or ultralytics is not installed.
    """

    def __init__(self, cfg: DetectionConfig):
        model_path = Path(cfg.MODEL_PATH)
        if not model_path.exists():
            raise RuntimeError(
                f"[ObjectDetector] Model weights not found: {model_path}\n"
                f"Place your .pt file at that path or pass --model <path>."
            )

        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "[ObjectDetector] ultralytics is not installed.\n"
                "Run: pip install ultralytics"
            )

        self._model      = YOLO(str(model_path))
        self._cfg        = cfg
        self._class_names: list[str] = (
            list(self._model.names.values())
            if hasattr(self._model, "names") else []
        )
        print(f"[ObjectDetector] Loaded: {model_path.name}  "
              f"({len(self._class_names)} classes)")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self,
            frame_bgr   : np.ndarray,
            depth_clean : np.ndarray,
            frame_idx   : int = 0) -> DetectionResult:
        """
        Run YOLO inference on one frame and sample depth for each detection.

        Parameters
        ----------
        frame_bgr   : (H, W, 3) uint8 BGR image from ZED left camera.
        depth_clean : (H, W) float32 depth array (NaNs for invalid pixels),
                      as returned by zed_scene_core.preprocess_depth().
        frame_idx   : Current frame counter (for logging).

        Returns
        -------
        DetectionResult
        """
        cfg = self._cfg
        results = self._model.predict(
            frame_bgr,
            conf       = cfg.CONF_THRESH,
            iou        = cfg.IOU_THRESH,
            verbose    = False,
        )

        boxes: list = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                # Bounding box in pixel coords
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf     = float(box.conf[0])
                class_id = int(box.cls[0])
                label    = (self._class_names[class_id]
                            if class_id < len(self._class_names)
                            else str(class_id))

                # Depth estimate from centre crop of the box
                depth_m = self._sample_depth(depth_clean, x1, y1, x2, y2)

                boxes.append((x1, y1, x2, y2, conf, class_id, label, depth_m))

        return DetectionResult(
            boxes        = boxes,
            count        = len(boxes),
            frame_idx    = frame_idx,
            any_detected = len(boxes) > 0,
        )

    # ------------------------------------------------------------------
    # Depth sampling helper
    # ------------------------------------------------------------------

    def _sample_depth(self,
                      depth  : np.ndarray,
                      x1, y1, x2, y2: int) -> float:
        """
        Return the median valid depth inside the centre crop of a bounding box.
        Falls back to inf if no valid pixels exist.
        """
        frac = self._cfg.DEPTH_SAMPLE_FRAC
        h_d  = depth.shape[0]
        w_d  = depth.shape[1]

        # Centre crop
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        dw = max(int(bw * frac / 2), 1)
        dh = max(int(bh * frac / 2), 1)

        # Clip to image bounds
        rx1 = max(cx - dw, 0)
        rx2 = min(cx + dw, w_d)
        ry1 = max(cy - dh, 0)
        ry2 = min(cy + dh, h_d)

        roi   = depth[ry1:ry2, rx1:rx2]
        valid = roi[np.isfinite(roi)]
        return float(np.median(valid)) if len(valid) > 0 else float("inf")

    # ------------------------------------------------------------------
    # Overlay rendering
    # ------------------------------------------------------------------

    @staticmethod
    def draw_overlay(frame      : np.ndarray,
                     result     : DetectionResult,
                     depth_clean: np.ndarray,
                     cfg        : DetectionConfig) -> np.ndarray:
        """
        Draw labelled bounding boxes and a detection HUD panel on a copy of frame.

        Parameters
        ----------
        frame       : (H, W, 3) uint8 BGR — the camera image to annotate.
        result      : DetectionResult from ObjectDetector.run().
        depth_clean : Depth array (used only for future per-pixel overlays).
        cfg         : DetectionConfig controlling colours and labels.

        Returns
        -------
        New (H + PANEL_HEIGHT, W, 3) uint8 BGR image.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        font     = cv2.FONT_HERSHEY_SIMPLEX
        font_s   = cfg.LABEL_FONT_SCALE
        thickness = cfg.BOX_THICKNESS

        # --- Class tally for HUD ----------------------------------------
        class_counts: dict[str, int] = {}

        for (x1, y1, x2, y2, conf, class_id, label, depth_m) in result.boxes:
            color = cfg.CLASS_COLORS.get(class_id, cfg.BOX_COLOR_DEFAULT)

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            # Label text
            parts = [label]
            if cfg.SHOW_CONF:
                parts.append(f"{conf:.2f}")
            if cfg.SHOW_DEPTH and depth_m < float("inf"):
                parts.append(f"{depth_m:.2f}m")
            text = "  ".join(parts)

            # Background pill for readability
            (tw, th), _ = cv2.getTextSize(text, font, font_s, 1)
            ty = max(y1 - 6, th + 4)
            cv2.rectangle(out,
                          (x1, ty - th - 4),
                          (x1 + tw + 6, ty + 2),
                          color, -1)
            cv2.putText(out, text, (x1 + 3, ty - 2),
                        font, font_s, (0, 0, 0), 1, cv2.LINE_AA)

            # Tally
            class_counts[label] = class_counts.get(label, 0) + 1

        # --- Bottom HUD panel -------------------------------------------
        panel = np.zeros((cfg.PANEL_HEIGHT, w, 3), dtype=np.uint8)

        def put(text: str, row: int,
                col: tuple = (200, 230, 200), x_off: int = 10):
            cv2.putText(panel, text, (x_off, 20 + row * 26),
                        font, 0.58, col, 1, cv2.LINE_AA)

        if result.any_detected:
            tally = "  |  ".join(
                f"{lbl}: {cnt}" for lbl, cnt in class_counts.items()
            )
            put(f"TARGETS  {result.count}  ─  {tally}",
                0, (0, 230, 100))
        else:
            put("TARGETS  0  ─  no detections this frame", 0, (80, 80, 80))

        put(f"Frame: {result.frame_idx}", 1, (150, 150, 150))

        return np.vstack([out, panel])


# =============================================================================
# Console printing helper (used by main.py)
# =============================================================================

def print_detections(result: DetectionResult) -> None:
    """
    Print a compact detection summary to stdout.
    Call once per frame (or every N frames) from main.py.
    """
    if not result.any_detected:
        return
    lines = []
    for (x1, y1, x2, y2, conf, class_id, label, depth_m) in result.boxes:
        depth_str = f"{depth_m:.2f} m" if depth_m < float("inf") else "  n/a  "
        lines.append(f"    [{label}]  conf={conf:.2f}  depth={depth_str}  "
                     f"box=({x1},{y1})-({x2},{y2})")
    print(f"  DETECTIONS (frame {result.frame_idx}):")
    print("\n".join(lines))