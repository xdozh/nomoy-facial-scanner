"""Unit tests for detect.py — DETECT AGENT "Eyes".

These tests exercise the PersonDetector class using injected fake models
so that no real YOLO weights are required.  They verify:

  - Output contract (shape, types, ordering, clamping, finiteness, JSON)
  - Input validation (wrong dtype, wrong dims, non-ndarray)
  - Empty-frame / no-detection handling
  - Multi-detection handling
  - Coordinate clamping and ordering correction
  - Confidence clamping
  - Non-finite value filtering
  - Device / variant selection logic
  - Model-missing error path
  - The _validate_output helper
  - json.dumps on all outputs
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional
from unittest import mock

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detect import PersonDetector, _validate_output


# ============================================================================
# Fake model infrastructure — simulates Ultralytics YOLO result objects
# ============================================================================

@dataclass
class _FakeBox:
    """Mimics a single ultralytics box object."""
    _xyxy: List[float]
    _conf: float
    _cls: int = 0

    @property
    def xyxy(self) -> Any:
        """Return xyxy as a 2-D array-like (outer dim = batch)."""
        return np.array([self._xyxy], dtype=np.float32)

    @property
    def conf(self) -> Any:
        return np.array([self._conf], dtype=np.float32)

    @property
    def cls(self) -> Any:
        return np.array([self._cls], dtype=np.float32)


@dataclass
class _FakeResult:
    """Mimics a single ultralytics result object."""
    _boxes: Optional[List[_FakeBox]] = None

    @property
    def boxes(self) -> Optional[Any]:
        if self._boxes is None:
            return None
        return self._boxes

    def __len__(self) -> int:
        return len(self._boxes) if self._boxes else 0


class FakeModel:
    """Injectable fake model that returns pre-configured detections.

    Pass a list of ``_FakeBox`` instances (or ``None`` for no detections).
    The model ignores the actual frame content.
    """

    def __init__(self, boxes: Optional[List[_FakeBox]] = None) -> None:
        self._boxes = boxes
        self.last_kwargs: dict = {}

    def __call__(self, frame: Any, **kwargs: Any) -> List[_FakeResult]:
        self.last_kwargs = kwargs
        result = _FakeResult(self._boxes)
        return [result]


# ============================================================================
# Helpers
# ============================================================================

def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a synthetic BGR uint8 frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ============================================================================
# Tests: output contract
# ============================================================================

class TestOutputContract:
    """Verify the output satisfies all contract requirements."""

    def test_returns_list(self):
        model = FakeModel(boxes=[])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        assert isinstance(result, list)

    def test_empty_frame_returns_empty_list(self):
        model = FakeModel(boxes=[])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        assert result == []

    def test_none_boxes_returns_empty_list(self):
        model = FakeModel(boxes=None)
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        assert result == []

    def test_single_detection_shape(self):
        model = FakeModel(boxes=[_FakeBox([10.0, 20.0, 100.0, 200.0], 0.9)])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        assert len(result) == 1
        assert len(result[0]) == 5

    def test_all_values_are_float(self):
        model = FakeModel(boxes=[_FakeBox([10.0, 20.0, 100.0, 200.0], 0.9)])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        for val in result[0]:
            assert isinstance(val, float), f"Expected float, got {type(val)}"

    def test_coordinates_are_ordered(self):
        model = FakeModel(boxes=[_FakeBox([10.0, 20.0, 100.0, 200.0], 0.8)])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        x1, y1, x2, y2, _ = result[0]
        assert x1 <= x2
        assert y1 <= y2

    def test_confidence_in_unit_interval(self):
        model = FakeModel(boxes=[_FakeBox([10.0, 20.0, 100.0, 200.0], 0.42)])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        conf = result[0][4]
        assert 0.0 <= conf <= 1.0

    def test_json_serializable(self):
        model = FakeModel(boxes=[
            _FakeBox([10.0, 20.0, 100.0, 200.0], 0.9),
            _FakeBox([50.0, 60.0, 150.0, 250.0], 0.7),
        ])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        # Must not raise
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
        # Round-trip
        deserialized = json.loads(serialized)
        assert deserialized == result

    def test_all_values_are_finite(self):
        model = FakeModel(boxes=[_FakeBox([10.0, 20.0, 100.0, 200.0], 0.5)])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        for box in result:
            for val in box:
                assert math.isfinite(val), f"Non-finite value: {val}"


# ============================================================================
# Tests: coordinate clamping and ordering
# ============================================================================

class TestClamping:
    """Verify coordinates are clamped to image bounds."""

    def test_clamp_negative_coords(self):
        model = FakeModel(boxes=[_FakeBox([-10.0, -20.0, 100.0, 200.0], 0.9)])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame(480, 640))
        x1, y1, x2, y2, _ = result[0]
        assert x1 >= 0.0
        assert y1 >= 0.0

    def test_clamp_coords_exceeding_image(self):
        model = FakeModel(boxes=[_FakeBox([10.0, 20.0, 700.0, 500.0], 0.9)])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame(480, 640))
        x1, y1, x2, y2, _ = result[0]
        assert x2 <= 640.0
        assert y2 <= 480.0

    def test_swap_reversed_coords(self):
        # Model returns x1 > x2 (reversed): detect.py must swap them
        model = FakeModel(boxes=[_FakeBox([200.0, 300.0, 100.0, 150.0], 0.9)])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame(480, 640))
        x1, y1, x2, y2, _ = result[0]
        assert x1 <= x2
        assert y1 <= y2

    def test_clamp_confidence_above_one(self):
        model = FakeModel(boxes=[_FakeBox([10.0, 20.0, 100.0, 200.0], 1.5)])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        conf = result[0][4]
        assert conf <= 1.0

    def test_clamp_confidence_below_zero(self):
        model = FakeModel(boxes=[_FakeBox([10.0, 20.0, 100.0, 200.0], -0.1)])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        conf = result[0][4]
        assert conf >= 0.0


# ============================================================================
# Tests: non-finite filtering
# ============================================================================

class TestNonFiniteFiltering:
    """Non-finite values (NaN, Inf) must be dropped."""

    def test_nan_coordinate_dropped(self):
        model = FakeModel(boxes=[
            _FakeBox([float("nan"), 20.0, 100.0, 200.0], 0.9),
        ])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        assert result == [], "Box with NaN coordinate should be dropped"

    def test_inf_coordinate_dropped(self):
        model = FakeModel(boxes=[
            _FakeBox([float("inf"), 20.0, 100.0, 200.0], 0.9),
        ])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        assert result == [], "Box with Inf coordinate should be dropped"

    def test_nan_confidence_dropped(self):
        model = FakeModel(boxes=[
            _FakeBox([10.0, 20.0, 100.0, 200.0], float("nan")),
        ])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        assert result == [], "Box with NaN confidence should be dropped"

    def test_valid_box_alongside_nan_box(self):
        model = FakeModel(boxes=[
            _FakeBox([float("nan"), 20.0, 100.0, 200.0], 0.9),
            _FakeBox([10.0, 20.0, 100.0, 200.0], 0.8),
        ])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        assert len(result) == 1
        assert result[0][4] == pytest.approx(0.8)


# ============================================================================
# Tests: multiple detections
# ============================================================================

class TestMultiDetection:
    """Verify handling of multiple persons in one frame."""

    def test_two_persons(self):
        model = FakeModel(boxes=[
            _FakeBox([10.0, 20.0, 100.0, 200.0], 0.9),
            _FakeBox([300.0, 100.0, 400.0, 350.0], 0.6),
        ])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        assert len(result) == 2
        for box in result:
            assert len(box) == 5

    def test_five_persons(self):
        boxes = [
            _FakeBox([i * 50.0, i * 30.0, i * 50.0 + 80.0, i * 30.0 + 120.0], 0.5 + i * 0.05)
            for i in range(5)
        ]
        model = FakeModel(boxes=boxes)
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        assert len(result) == 5


# ============================================================================
# Tests: input validation
# ============================================================================

class TestInputValidation:
    """Verify proper errors on invalid inputs."""

    def test_non_ndarray_raises(self):
        model = FakeModel(boxes=[])
        det = PersonDetector(model=model)
        with pytest.raises(ValueError, match="numpy.ndarray"):
            det.detect([[0, 0, 0]])

    def test_wrong_dims_raises(self):
        model = FakeModel(boxes=[])
        det = PersonDetector(model=model)
        with pytest.raises(ValueError, match="3-channel"):
            det.detect(np.zeros((480, 640), dtype=np.uint8))

    def test_four_channel_raises(self):
        model = FakeModel(boxes=[])
        det = PersonDetector(model=model)
        with pytest.raises(ValueError, match="3-channel"):
            det.detect(np.zeros((480, 640, 4), dtype=np.uint8))

    def test_wrong_dtype_raises(self):
        model = FakeModel(boxes=[])
        det = PersonDetector(model=model)
        with pytest.raises(ValueError, match="uint8"):
            det.detect(np.zeros((480, 640, 3), dtype=np.float32))


# ============================================================================
# Tests: model kwargs forwarding
# ============================================================================

class TestModelKwargs:
    """Verify that inference kwargs are correctly forwarded."""

    def test_conf_threshold_forwarded(self):
        model = FakeModel(boxes=[])
        det = PersonDetector(model=model, conf_threshold=0.42)
        det.detect(_make_frame())
        assert model.last_kwargs.get("conf") == 0.42

    def test_person_class_filter_forwarded(self):
        model = FakeModel(boxes=[])
        det = PersonDetector(model=model)
        det.detect(_make_frame())
        assert model.last_kwargs.get("classes") == [0]

    def test_verbose_false_forwarded(self):
        model = FakeModel(boxes=[])
        det = PersonDetector(model=model)
        det.detect(_make_frame())
        assert model.last_kwargs.get("verbose") is False

    def test_default_conf_threshold_is_035(self):
        model = FakeModel(boxes=[])
        det = PersonDetector(model=model)
        assert det.conf_threshold == 0.35


# ============================================================================
# Tests: device / variant resolution
# ============================================================================

class TestDeviceVariantResolution:
    """Verify CPU/GPU model variant selection without loading actual weights."""

    def test_cpu_selects_yolo11n(self):
        det = PersonDetector(model=FakeModel(boxes=[]))
        variant = det._resolve_variant("cpu", None)
        assert variant == "yolo11n"

    def test_gpu_selects_yolo11s(self):
        det = PersonDetector(model=FakeModel(boxes=[]))
        variant = det._resolve_variant("cuda", None)
        assert variant == "yolo11s"

    def test_cuda0_selects_yolo11s(self):
        det = PersonDetector(model=FakeModel(boxes=[]))
        variant = det._resolve_variant("cuda:0", None)
        assert variant == "yolo11s"

    def test_explicit_variant_overrides(self):
        det = PersonDetector(model=FakeModel(boxes=[]))
        variant = det._resolve_variant("cpu", "yolov8n")
        assert variant == "yolov8n"

    def test_resolve_device_explicit(self):
        det = PersonDetector(model=FakeModel(boxes=[]))
        assert det._resolve_device("cpu") == "cpu"
        assert det._resolve_device("cuda:1") == "cuda:1"

    def test_resolve_device_no_torch(self):
        """Without torch installed, default device should be cpu."""
        det = PersonDetector(model=FakeModel(boxes=[]))
        # Mock ImportError for torch
        with mock.patch.dict(sys.modules, {"torch": None}):
            device = det._resolve_device(None)
            assert device == "cpu"


# ============================================================================
# Tests: missing weights error path
# ============================================================================

class TestMissingWeights:
    """Loading without weights must fail clearly, not silently download."""

    def test_missing_ultralytics_raises_import_error(self):
        """If ultralytics is not installed, ImportError must be raised."""
        with mock.patch.dict(sys.modules, {"ultralytics": None}):
            with pytest.raises(ImportError, match="ultralytics"):
                PersonDetector(device="cpu")

    def test_missing_weight_file_raises_file_not_found(self):
        """If weight file does not exist, FileNotFoundError must be raised."""
        # Create a fake ultralytics module with a YOLO class
        fake_ul = type(sys)("ultralytics")
        fake_ul.YOLO = lambda path: None  # won't be reached
        fake_ul.settings = mock.Mock()

        with mock.patch.dict(sys.modules, {"ultralytics": fake_ul}):
            with pytest.raises(FileNotFoundError, match="not found locally"):
                PersonDetector(
                    device="cpu",
                    model_variant="yolo11n",
                    weights_dir="/nonexistent/dir",
                )


# ============================================================================
# Tests: callable interface
# ============================================================================

class TestCallable:
    """PersonDetector instance should be callable as an alias for detect()."""

    def test_call_equals_detect(self):
        model = FakeModel(boxes=[_FakeBox([10.0, 20.0, 100.0, 200.0], 0.9)])
        det = PersonDetector(model=model)
        frame = _make_frame()
        assert det(frame) == det.detect(frame)


# ============================================================================
# Tests: _validate_output helper
# ============================================================================

class TestValidateOutput:
    """Test the standalone output validator."""

    def test_valid_output(self):
        _validate_output([[10.0, 20.0, 100.0, 200.0, 0.5]], 480, 640)

    def test_empty_output(self):
        _validate_output([], 480, 640)

    def test_non_list_outer(self):
        with pytest.raises(ValueError, match="list"):
            _validate_output("not a list", 480, 640)  # type: ignore

    def test_non_list_inner(self):
        with pytest.raises(ValueError, match="list"):
            _validate_output([(10, 20, 100, 200, 0.5)], 480, 640)  # type: ignore

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="5 values"):
            _validate_output([[10.0, 20.0, 100.0, 200.0]], 480, 640)

    def test_non_float_value(self):
        with pytest.raises(ValueError, match="float"):
            _validate_output([[10.0, 20.0, 100.0, 200.0, "hi"]], 480, 640)  # type: ignore

    def test_nan_value(self):
        with pytest.raises(ValueError, match="non-finite"):
            _validate_output([[float("nan"), 20.0, 100.0, 200.0, 0.5]], 480, 640)

    def test_x1_gt_x2(self):
        with pytest.raises(ValueError, match="x1"):
            _validate_output([[200.0, 20.0, 100.0, 200.0, 0.5]], 480, 640)

    def test_y1_gt_y2(self):
        with pytest.raises(ValueError, match="y1"):
            _validate_output([[10.0, 300.0, 100.0, 200.0, 0.5]], 480, 640)

    def test_x_out_of_bounds(self):
        with pytest.raises(ValueError, match="x coords"):
            _validate_output([[10.0, 20.0, 700.0, 200.0, 0.5]], 480, 640)

    def test_y_out_of_bounds(self):
        with pytest.raises(ValueError, match="y coords"):
            _validate_output([[10.0, 20.0, 100.0, 500.0, 0.5]], 480, 640)

    def test_confidence_below_zero(self):
        with pytest.raises(ValueError, match="confidence"):
            _validate_output([[10.0, 20.0, 100.0, 200.0, -0.1]], 480, 640)

    def test_confidence_above_one(self):
        with pytest.raises(ValueError, match="confidence"):
            _validate_output([[10.0, 20.0, 100.0, 200.0, 1.1]], 480, 640)


# ============================================================================
# Tests: edge cases
# ============================================================================

class TestEdgeCases:
    """Boundary / edge-case scenarios."""

    def test_tiny_frame(self):
        """1x1 frame with no detections."""
        model = FakeModel(boxes=[])
        det = PersonDetector(model=model)
        result = det.detect(np.zeros((1, 1, 3), dtype=np.uint8))
        assert result == []

    def test_large_frame_dimensions(self):
        """4K-ish frame dimensions are handled correctly."""
        model = FakeModel(boxes=[_FakeBox([100.0, 200.0, 3000.0, 2000.0], 0.7)])
        det = PersonDetector(model=model)
        result = det.detect(np.zeros((2160, 3840, 3), dtype=np.uint8))
        assert len(result) == 1
        x1, y1, x2, y2, conf = result[0]
        assert x2 <= 3840.0
        assert y2 <= 2160.0

    def test_box_at_exact_image_boundary(self):
        """Box sitting exactly on image edges."""
        model = FakeModel(boxes=[_FakeBox([0.0, 0.0, 640.0, 480.0], 0.99)])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame(480, 640))
        assert len(result) == 1
        x1, y1, x2, y2, _ = result[0]
        assert x1 == 0.0
        assert y1 == 0.0
        assert x2 == 640.0
        assert y2 == 480.0

    def test_zero_area_box_after_clamp(self):
        """A detection that collapses to a point after clamping is still valid
        per contract (no minimum area requirement)."""
        model = FakeModel(boxes=[_FakeBox([0.0, 0.0, 0.0, 0.0], 0.5)])
        det = PersonDetector(model=model)
        result = det.detect(_make_frame())
        assert len(result) == 1
        x1, y1, x2, y2, _ = result[0]
        assert x1 == x2 == 0.0
        assert y1 == y2 == 0.0

    def test_multiple_result_objects(self):
        """If the model returns multiple result objects (batch), they should
        all be processed (defensive: YOLO normally returns one result for
        one frame)."""
        class MultiBatchModel:
            def __call__(self, frame, **kwargs):
                return [
                    _FakeResult([_FakeBox([10.0, 10.0, 50.0, 50.0], 0.9)]),
                    _FakeResult([_FakeBox([60.0, 60.0, 120.0, 120.0], 0.8)]),
                ]

        det = PersonDetector(model=MultiBatchModel())
        result = det.detect(_make_frame())
        assert len(result) == 2

    def test_custom_conf_threshold(self):
        """Custom confidence threshold is used."""
        model = FakeModel(boxes=[_FakeBox([10.0, 20.0, 100.0, 200.0], 0.9)])
        det = PersonDetector(model=model, conf_threshold=0.5)
        det.detect(_make_frame())
        assert model.last_kwargs["conf"] == 0.5


class TestOfflineInitialization:
    def test_offline_environment_is_configured_before_loading(self):
        assert os.environ["DO_NOT_TRACK"] == "1"
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["ULTRALYTICS_OFFLINE"] == "true"
        assert os.environ["YOLO_OFFLINE"] == "true"

    def test_integrity_failure_refuses_model_load(self):
        fake_ul = type(sys)("ultralytics")
        fake_ul.YOLO = mock.Mock()
        fake_ul.settings = mock.Mock()
        with mock.patch.dict(sys.modules, {"ultralytics": fake_ul}), \
                mock.patch("scripts.verify_models.main", return_value=1):
            with pytest.raises(RuntimeError, match="integrity verification failed"):
                PersonDetector(device="cpu", weights_dir="models")
        fake_ul.YOLO.assert_not_called()

    def test_telemetry_disabled_before_local_model_construction(self, tmp_path):
        weight = tmp_path / "yolo11n.pt"
        weight.write_bytes(b"test")
        loaded_model = mock.Mock()
        fake_ul = type(sys)("ultralytics")
        fake_ul.YOLO = mock.Mock(return_value=loaded_model)
        fake_ul.settings = mock.Mock()
        with mock.patch.dict(sys.modules, {"ultralytics": fake_ul}), \
                mock.patch("scripts.verify_models.main", return_value=0), \
                mock.patch("socket.socket.connect") as connect, \
                mock.patch("socket.create_connection") as create_connection:
            PersonDetector(device="cpu", weights_dir=str(tmp_path))
        fake_ul.settings.update.assert_called_once_with({"sync": False})
        fake_ul.YOLO.assert_called_once_with(str(weight))
        connect.assert_not_called()
        create_connection.assert_not_called()
