"""Person detection module — DETECT AGENT "Eyes".

Provides frame-level person detection using Ultralytics YOLO.
Model selection:  YOLO11n on CPU, YOLO11s when a CUDA GPU is present.
Fallback order:   YOLOv8n -> torchvision Faster R-CNN (not implemented here;
                  swap at the module boundary if needed).

Input contract
--------------
One decoded image frame as a NumPy ndarray in OpenCV BGR layout
(shape ``(H, W, 3)``, dtype ``uint8``).

Output contract
---------------
``list[list[float]]`` where each inner list is exactly
``[x1, y1, x2, y2, conf]`` with:
  - coordinates ordered (x1 <= x2, y1 <= y2) and clamped to image dimensions,
  - confidence in [0, 1],
  - the full return value passes ``json.dumps``.

Runtime rules
-------------
- **Offline at runtime**: model weights must already be present locally.
  A missing-weights error is raised immediately; no download is attempted
  during frame processing.
- **CPU-only operation** is always supported.
- Device / model selection is injectable for testing.
"""

from __future__ import annotations

import os

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ULTRALYTICS_OFFLINE", "true")
os.environ.setdefault("YOLO_OFFLINE", "true")

import json
import math
from typing import Any, List, Optional, Protocol


# ---------------------------------------------------------------------------
# Injectable model protocol (for testing / swapping without real weights)
# ---------------------------------------------------------------------------

class DetectionModel(Protocol):
    """Minimal duck-type expected from any detection backend."""

    def __call__(self, frame: Any, **kwargs: Any) -> Any:
        """Run inference on a single frame."""
        ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PersonDetector:
    """Stateful person detector wrapping an Ultralytics YOLO model.

    Parameters
    ----------
    model : DetectionModel | None
        An already-loaded model instance (useful for tests or custom
        backends).  When *None* the class loads Ultralytics YOLO from
        local weights automatically.
    device : str | None
        Force a device string (``"cpu"``, ``"cuda"``, ``"cuda:0"`` ...).
        When *None* the class auto-selects: ``cuda`` if available, else
        ``cpu``.
    conf_threshold : float
        Minimum detection confidence.  Default ``0.35`` per spec.
    model_variant : str | None
        Override which weight file to load (``"yolo11n"``, ``"yolo11s"``,
        ``"yolov8n"`` ...).  When *None* the class picks ``yolo11s`` for
        GPU and ``yolo11n`` for CPU.
    weights_dir : str | None
        Directory where ``.pt`` weight files live.  When *None* the
        project-locked ``models/`` directory is used.
    """

    # YOLO COCO class index for "person"
    _PERSON_CLASS: int = 0

    def __init__(
        self,
        *,
        model: Optional[DetectionModel] = None,
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        model_variant: Optional[str] = None,
        weights_dir: Optional[str] = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self._device = device
        self._model = model

        # Default to the project's locked models directory so callers do not
        # have to remember to point at the verified artifacts.
        if weights_dir is None:
            weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

        if self._model is None:
            self._model = self._load_model(
                device=device,
                model_variant=model_variant,
                weights_dir=weights_dir,
            )

    # ------------------------------------------------------------------
    # Model loading (only when no model injected)
    # ------------------------------------------------------------------

    def _resolve_device(self, device: Optional[str]) -> str:
        """Return the effective device string."""
        if device is not None:
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _resolve_variant(self, device: str, model_variant: Optional[str]) -> str:
        """Pick model variant based on device when not explicitly set."""
        if model_variant is not None:
            return model_variant
        return "yolo11s" if device.startswith("cuda") else "yolo11n"

    def _load_model(
        self,
        device: Optional[str],
        model_variant: Optional[str],
        weights_dir: Optional[str],
    ) -> Any:
        """Load Ultralytics YOLO model from local weights.

        Raises
        ------
        FileNotFoundError
            If the weight file does not exist locally.
        ImportError
            If the ``ultralytics`` package is not installed.
        """
        try:
            from ultralytics import YOLO, settings
        except ImportError as exc:
            raise ImportError(
                "The 'ultralytics' package is required for person detection.  "
                "Install it with:  pip install ultralytics"
            ) from exc

        settings.update({"sync": False})

        from scripts.verify_models import main as verify_models

        if verify_models() != 0:
            raise RuntimeError("Model integrity verification failed; inference refused")

        resolved_device = self._resolve_device(device)
        self._device = resolved_device
        variant = self._resolve_variant(resolved_device, model_variant)

        # Build weight path --------------------------------------------------
        import os

        if weights_dir is not None:
            weight_path = os.path.join(weights_dir, f"{variant}.pt")
        else:
            # Ultralytics default: look in its own cache.  We pass the
            # variant name and let YOLO resolve it.  But we must NOT let
            # it silently download — check existence first.
            #
            # Ultralytics stores weights under YOLO's settings dir; the
            # simplest reliable check is to attempt a *local-only* load.
            weight_path = f"{variant}.pt"

        # Verify the file exists locally before handing it to YOLO.
        # For bare names (no dir), also check the Ultralytics default cache.
        resolved_path = self._find_weights(weight_path, variant)

        model = YOLO(resolved_path)
        model.to(resolved_device)
        return model

    @staticmethod
    def _find_weights(weight_path: str, variant: str) -> str:
        """Return an existing local path for the weights or raise."""
        import os

        # Absolute / relative path provided explicitly
        if os.path.isfile(weight_path):
            return weight_path

        # Check Ultralytics default cache locations
        from pathlib import Path
        candidates = [
            Path.home() / ".config" / "Ultralytics" / f"{variant}.pt",
            Path.home() / ".ultralytics" / f"{variant}.pt",
            # Sometimes stored right in cwd by Ultralytics
            Path(f"{variant}.pt"),
        ]

        # Also check the YOLO settings dir if ultralytics exposes it
        try:
            from ultralytics import settings as ul_settings
            settings_dir = Path(ul_settings.get("weights_dir", ""))
            if settings_dir.is_dir():
                candidates.append(settings_dir / f"{variant}.pt")
        except Exception:
            pass

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        raise FileNotFoundError(
            f"YOLO weight file '{variant}.pt' not found locally.  "
            f"Searched: {weight_path} and default Ultralytics cache locations.  "
            f"Download weights during setup (not at runtime):  "
            f"yolo export model={variant}.pt"
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def detect(self, frame: "numpy.ndarray") -> List[List[float]]:
        """Detect persons in a single BGR frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Shape ``(H, W, 3)``, dtype ``uint8``, BGR colour order.

        Returns
        -------
        list[list[float]]
            Each inner list: ``[x1, y1, x2, y2, conf]``.
            Coordinates are clamped to ``[0, W)`` / ``[0, H)`` and
            ordered so that ``x1 <= x2``, ``y1 <= y2``.
            Confidence is in ``[0.0, 1.0]``.
            The full return value is ``json.dumps``-safe.

        Raises
        ------
        ValueError
            If *frame* is not a valid 3-channel uint8 image.
        """
        import numpy as np

        # ---- Input validation ------------------------------------------------
        if not isinstance(frame, np.ndarray):
            raise ValueError(
                f"Expected numpy.ndarray, got {type(frame).__name__}"
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel image (H, W, 3), got shape {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(
                f"Expected dtype uint8, got {frame.dtype}"
            )

        h, w = frame.shape[:2]

        # ---- Inference -------------------------------------------------------
        results = self._model(
            frame,
            conf=self.conf_threshold,
            classes=[self._PERSON_CLASS],
            imgsz=960 if self._device == "cpu" else 640,
            verbose=False,
        )

        # ---- Post-process ----------------------------------------------------
        detections: List[List[float]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].tolist()       # [x1, y1, x2, y2]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = (float(v) for v in xyxy)

                # Check finiteness BEFORE clamping (NaN/Inf corrupt
                # min/max in Python).  Kill criterion: malformed boxes.
                if not all(
                    math.isfinite(v) for v in (x1, y1, x2, y2, conf)
                ):
                    continue  # silently drop non-finite detections

                # Clamp to image bounds
                x1 = max(0.0, min(x1, float(w)))
                y1 = max(0.0, min(y1, float(h)))
                x2 = max(0.0, min(x2, float(w)))
                y2 = max(0.0, min(y2, float(h)))

                # Ensure ordering
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                # Clamp confidence
                conf = max(0.0, min(1.0, conf))

                detections.append([x1, y1, x2, y2, conf])

        # ---- Output-contract sanity check ------------------------------------
        _validate_output(detections, h, w)

        return detections

    # Convenience: make the instance callable
    def __call__(self, frame: "numpy.ndarray") -> List[List[float]]:
        """Alias for :meth:`detect`."""
        return self.detect(frame)


# ---------------------------------------------------------------------------
# Output-contract validator (also usable in tests)
# ---------------------------------------------------------------------------

def _validate_output(
    detections: List[List[float]], img_h: int, img_w: int
) -> None:
    """Raise ``ValueError`` if *detections* violate the output contract.

    Checks:
      1. Outer type is list.
      2. Each inner element is a list of exactly 5 floats.
      3. Coordinates are ordered and within image bounds.
      4. Confidence is in [0, 1].
      5. All values are finite.
      6. The whole structure passes ``json.dumps``.
    """
    if not isinstance(detections, list):
        raise ValueError(f"Output must be list, got {type(detections).__name__}")

    for i, det in enumerate(detections):
        if not isinstance(det, list):
            raise ValueError(f"Detection {i}: expected list, got {type(det).__name__}")
        if len(det) != 5:
            raise ValueError(f"Detection {i}: expected 5 values, got {len(det)}")

        x1, y1, x2, y2, conf = det

        for j, v in enumerate(det):
            if not isinstance(v, (int, float)):
                raise ValueError(
                    f"Detection {i}[{j}]: expected float, got {type(v).__name__}"
                )
            if not math.isfinite(v):
                raise ValueError(f"Detection {i}[{j}]: non-finite value {v}")

        if x1 > x2:
            raise ValueError(f"Detection {i}: x1 ({x1}) > x2 ({x2})")
        if y1 > y2:
            raise ValueError(f"Detection {i}: y1 ({y1}) > y2 ({y2})")
        if not (0.0 <= x1 <= img_w and 0.0 <= x2 <= img_w):
            raise ValueError(
                f"Detection {i}: x coords [{x1}, {x2}] outside [0, {img_w}]"
            )
        if not (0.0 <= y1 <= img_h and 0.0 <= y2 <= img_h):
            raise ValueError(
                f"Detection {i}: y coords [{y1}, {y2}] outside [0, {img_h}]"
            )
        if not (0.0 <= conf <= 1.0):
            raise ValueError(
                f"Detection {i}: confidence {conf} outside [0, 1]"
            )

    # JSON-serializable check
    try:
        json.dumps(detections)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Output is not JSON-serializable: {exc}") from exc
