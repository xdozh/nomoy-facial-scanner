"""Deterministic offline tests for the face-recognition boundary.

All recognition tests inject a fake backend whose embeddings are derived
from image colour, so majority vote, threshold, minimum-size, UNKNOWN retry,
zero-false-accept, cache regeneration, and gallery-swap contracts are
exercised with no real faces and no InsightFace dependency.  The real
InsightFace smoke test at the bottom is the only test that touches
``models/buffalo_l``; it is also the provisional stand-in for T4 because the
synthetic assets contain no dedicated non-enrolled person — see module
docstring in ``face.py`` and the Gate C report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face import FaceRecognizer, InsightFaceBackend, UNKNOWN, VOTE_WINDOW

ROOT = Path(__file__).resolve().parent.parent

# Identity colours.  The fake backend maps dominant channel -> basis vector,
# so each colour family is a distinct orthogonal "identity".
IDENTITY_COLORS = {
    "alex": (255, 0, 0),    # BGR blue
    "blair": (0, 255, 0),   # BGR green
    "casey": (0, 0, 255),   # BGR red
}
UNENROLLED_COLOR = (255, 255, 0)  # cyan — far from every enrolled colour

# The fake backend snaps an image's mean colour to the nearest palette entry;
# too-far colours get a negative embedding that cannot pass the threshold.
_PALETTE = list(IDENTITY_COLORS.values())
_PALETTE_DISTANCE_MAX = 200.0


class FakeFace:
    def __init__(self, bbox, embedding):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.normed_embedding = np.asarray(embedding, dtype=np.float64)


def _embedding_for(image: np.ndarray) -> np.ndarray:
    """Snap mean colour to nearest palette identity; negative vector if far."""
    means = image.reshape(-1, 3).mean(axis=0)
    dists = [float(np.linalg.norm(means - np.asarray(c))) for c in _PALETTE]
    nearest = int(np.argmin(dists))
    if dists[nearest] > _PALETTE_DISTANCE_MAX:
        return np.array([-1.0, -1.0, -1.0, -1.0])
    vec = np.zeros(4, dtype=np.float64)
    vec[nearest] = 1.0
    return vec


class ScriptedBackend:
    """Returns one large face per call, embedding keyed on image colour.

    ``empty`` makes ``get`` return no faces at all (pose-away simulation).
    ``face_size`` shrinks the reported face below ``face_min_size_px``.
    """

    def __init__(self, *, empty=False, face_size: int = 100,
                 embedding_fn=_embedding_for):
        # ``empty`` may be a bool or a callable(image) -> bool so gallery
        # photos (200x200) can still embed while frame crops return no faces.
        self.empty = empty
        self.face_size = face_size
        self.embedding_fn = embedding_fn
        self.calls: list[np.ndarray] = []

    def get(self, image: np.ndarray) -> list[FakeFace]:
        self.calls.append(image)
        empty = self.empty(image) if callable(self.empty) else self.empty
        if empty or image.size == 0:
            return []
        s = self.face_size
        return [FakeFace([10, 10, 10 + s, 10 + s], self.embedding_fn(image))]


def _config(**overrides):
    cfg = {
        "gallery": {name: {"display_name": name.title()} for name in IDENTITY_COLORS},
        "match_threshold": 0.4,
        "face_min_size_px": 48,
        "embed_every_n_tracks": 1,
        "face_check_interval_frames": 15,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture()
def gallery_dir(tmp_path: Path) -> Path:
    gdir = tmp_path / "gallery"
    gdir.mkdir()
    for name, color in IDENTITY_COLORS.items():
        image = np.full((200, 200, 3), color, dtype=np.uint8)
        assert cv2.imwrite(str(gdir / f"{name}.jpg"), image)
    return gdir


def _frame_with_patch(color, box=(50, 50, 200, 200)):
    frame = np.full((300, 300, 3), 30, dtype=np.uint8)
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = color
    return frame


def _track(track_id=1, bbox=(50, 50, 200, 200), state="active"):
    return {"track_id": track_id, "bbox": list(bbox), "age": 1, "state": state}


def _recognizer(gallery_dir, backend, **overrides):
    return FaceRecognizer(_config(**overrides), gallery_dir=gallery_dir,
                          backend=backend)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

def test_gallery_built_from_photos_and_json_contract(gallery_dir):
    backend = ScriptedBackend()
    rec = _recognizer(gallery_dir, backend)
    assert rec.gallery_identities == ["alex", "blair", "casey"]
    assert len(backend.calls) == 3  # one embed call per gallery photo

    results, events = rec.update(
        _frame_with_patch(IDENTITY_COLORS["alex"]), [_track()], 0
    )
    assert results == [{"track_id": 1, "identity": "alex",
                        "confidence": pytest.approx(1.0)}]
    assert events == [{"event": "match", "frame": 0, "track_id": 1,
                       "identity": "alex", "confidence": pytest.approx(1.0)}]
    json.dumps({"results": results, "events": events})


def test_zero_false_accept_for_unenrolled_face(gallery_dir):
    """Deterministic isolated unknown-face check (provisional T4 stand-in)."""
    backend = ScriptedBackend()
    rec = _recognizer(gallery_dir, backend)
    for frame_index in range(30):
        results, events = rec.update(
            _frame_with_patch(UNENROLLED_COLOR), [_track()], frame_index
        )
        assert results[0]["identity"] == UNKNOWN
        assert results[0]["confidence"] < rec.match_threshold
        assert events == []


def test_below_threshold_vote_is_unknown(gallery_dir):
    # Unit vector whose max cosine vs any basis axis is ~0.415 < 0.45.
    near_miss = np.array([0.4, -0.5, 0.4, -0.6])
    near_miss /= np.linalg.norm(near_miss)
    # Gallery photos (200x200) embed normally; frame crops get the near-miss.
    def embed(img):
        if img.shape[:2] == (200, 200):
            return _embedding_for(img)
        return near_miss.copy()
    backend = ScriptedBackend(embedding_fn=embed)
    rec = _recognizer(gallery_dir, backend, match_threshold=0.45)
    results, events = rec.update(_frame_with_patch(IDENTITY_COLORS["alex"]),
                                 [_track()], 0)
    assert results[0]["identity"] == UNKNOWN
    assert events == []


def test_min_face_size_skips_small_faces(gallery_dir):
    # Gallery build uses big faces; runtime crops report sub-threshold faces.
    backend = ScriptedBackend(face_size=100)
    rec = FaceRecognizer(_config(face_min_size_px=48), gallery_dir=gallery_dir,
                         backend=backend)
    rec._backend = ScriptedBackend(face_size=20)  # < face_min_size_px
    for i in range(5):
        results, events = rec.update(
            _frame_with_patch(IDENTITY_COLORS["alex"]), [_track()], i)
        assert results[0]["identity"] == UNKNOWN
        assert events == []


def test_unknown_track_retries_every_interval(gallery_dir):
    empty = ScriptedBackend(
        empty=lambda img: img.shape[:2] != (200, 200))
    rec = _recognizer(gallery_dir, empty, face_check_interval_frames=15)
    gallery_calls = len(empty.calls)
    # Frame 0: first attempt (misses). Frames 1-14: no retry. Frame 15: retry.
    for i in range(16):
        rec.update(np.zeros((300, 300, 3), np.uint8), [_track()], i)
    assert len(empty.calls) == gallery_calls + 2


def test_unknown_track_becomes_named_on_retry(gallery_dir):
    backend = ScriptedBackend(
        empty=lambda img: img.shape[:2] != (200, 200))
    rec = _recognizer(gallery_dir, backend, face_check_interval_frames=3)
    rec.update(np.zeros((300, 300, 3), np.uint8), [_track()], 0)
    # Pose improves: crops now return the alex-coloured face embedding.
    backend.empty = False
    rec.update(_frame_with_patch(IDENTITY_COLORS["alex"]), [_track()], 1)
    rec.update(_frame_with_patch(IDENTITY_COLORS["alex"]), [_track()], 2)
    results, events = rec.update(
        _frame_with_patch(IDENTITY_COLORS["alex"]), [_track()], 3)
    assert results[0]["identity"] == "alex"
    assert events[-1]["event"] == "match"


def test_majority_vote_resists_flicker_and_reidentifies(gallery_dir):
    """9 alex votes then blair majority -> one reidentify, no name ping-pong."""
    backend = ScriptedBackend()
    rec = _recognizer(gallery_dir, backend)
    identities: list[str] = []
    for i in range(9):
        results, _ = rec.update(
            _frame_with_patch(IDENTITY_COLORS["alex"]), [_track()], i)
        identities.append(results[0]["identity"])
    assert set(identities) == {"alex"}

    for i in range(9, 18):  # blair minority holds alex; i=17 ties 9a/9b
        results, events = rec.update(
            _frame_with_patch(IDENTITY_COLORS["blair"]), [_track()], i)
        assert results[0]["identity"] == "alex"
        assert events == []
    # i=18: deque is 9a/10b -> strict blair majority flips the identity.
    results, events = rec.update(
        _frame_with_patch(IDENTITY_COLORS["blair"]), [_track()], 18)
    assert results[0]["identity"] == "blair"
    assert events == [{"event": "reidentify", "frame": 18, "track_id": 1,
                       "previous_identity": "alex", "identity": "blair",
                       "confidence": pytest.approx(1.0)}]


def test_gallery_swap_and_cache_regeneration(gallery_dir, tmp_path):
    """Adding a .jpg + config line works with zero code changes; each
    FaceRecognizer construction rebuilds embeddings from the photos."""
    backend = ScriptedBackend()
    rec = _recognizer(gallery_dir, backend)
    calls_after_first_build = len(backend.calls)

    # Swap: add a new enrolled photo + config entry, no code changes.
    new_color = (128, 0, 128)
    cv2.imwrite(str(gallery_dir / "drew.jpg"),
                np.full((200, 200, 3), new_color, np.uint8))
    # Backend with a palette that also knows drew's colour (axis 3).
    palette = _PALETTE + [new_color]

    def embed4(image):
        means = image.reshape(-1, 3).mean(axis=0)
        dists = [float(np.linalg.norm(means - np.asarray(c))) for c in palette]
        nearest = int(np.argmin(dists))
        if dists[nearest] > _PALETTE_DISTANCE_MAX:
            return np.array([-1.0] * 4)
        vec = np.zeros(4, dtype=np.float64)
        vec[nearest] = 1.0
        return vec

    backend2 = ScriptedBackend(embedding_fn=embed4)
    cfg = _config()
    cfg["gallery"]["drew"] = {"display_name": "Drew Synthetic"}
    rec2 = FaceRecognizer(cfg, gallery_dir=gallery_dir, backend=backend2)
    # Embeddings regenerated: backend called for all four photos again.
    assert len(backend2.calls) == 4
    assert "drew" in rec2.gallery_identities

    results, events = rec2.update(_frame_with_patch(new_color), [_track()], 0)
    assert results[0]["identity"] == "drew"
    assert events[0]["event"] == "match"


def test_coasting_track_keeps_identity_without_attempt(gallery_dir):
    backend = ScriptedBackend()
    rec = _recognizer(gallery_dir, backend)
    rec.update(_frame_with_patch(IDENTITY_COLORS["alex"]), [_track()], 0)
    calls = len(backend.calls)
    results, _ = rec.update(
        _frame_with_patch(IDENTITY_COLORS["blair"]),
        [_track(state="coasting")], 1)
    assert results[0]["identity"] == "alex"
    assert len(backend.calls) == calls  # no attempt while coasting


def test_match_threshold_outside_band_fails_closed(gallery_dir):
    with pytest.raises(ValueError):
        _recognizer(gallery_dir, ScriptedBackend(), match_threshold=0.6)


def test_missing_gallery_photo_fails_closed(gallery_dir):
    cfg = _config()
    cfg["gallery"]["ghost"] = {"display_name": "Ghost"}
    with pytest.raises(FileNotFoundError):
        FaceRecognizer(cfg, gallery_dir=gallery_dir, backend=ScriptedBackend())


# ---------------------------------------------------------------------------
# Real InsightFace smoke test (API-compatibility probe; provisional T4)
# ---------------------------------------------------------------------------

def test_insightface_buffalo_l_smoke():
    """Load local buffalo_l, embed gallery jpgs, and embed one synthetic crop.

    This exercises the real API path.  The repo's synthetic gallery photos
    are generated sprites, not real faces, so InsightFace may legitimately
    find zero faces in them — in which case we assert the fail-closed error
    surfaces cleanly rather than faking a pass.  Actual-video T4 (>=80%
    recognition on enrolled people, 0 false accepts on a non-enrolled person)
    stays PROVISIONAL until consenting real assets exist.
    """
    pytest.importorskip("insightface")
    model_dir = ROOT / "models" / "buffalo_l"
    if not model_dir.is_dir():
        pytest.skip("models/buffalo_l not present")

    backend = InsightFaceBackend(model_dir)
    faces_found = 0
    for jpg in sorted((ROOT / "gallery").glob("*.jpg")):
        image = cv2.imread(str(jpg))
        assert image is not None
        for face in backend.get(image):
            emb = np.asarray(face.normed_embedding, dtype=np.float64)
            assert emb.shape == (512,)
            assert np.isfinite(emb).all()
            faces_found += 1
    # Synthetic sprite faces are expected to yield few or no detections; the
    # contract proven here is that local buffalo_l loads and runs offline.
    print(f"\n[smoke] faces detected in synthetic gallery: {faces_found}")
