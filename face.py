"""Face recognition module — FACE AGENT "The Identifier".

Per-track identity resolution for the "It Scans & Identifies" demo.

Pipeline per active track
-------------------------
crop person bbox -> detect faces in crop -> keep largest face >=
``face_min_size_px`` -> 512-D embedding -> cosine match against the gallery
-> majority vote over the last ``vote_window`` attempts.

UNKNOWN tracks are re-attempted every ``face_check_interval_frames`` so a
track that starts face-away can still be named once the pose improves.

Output contract
---------------
``update(frame, tracks, frame_index)`` returns ``(results, events)`` where
``results`` is a list of ``{"track_id": int, "identity": str, "confidence":
float}`` (``identity`` is the config key, or ``"UNKNOWN"``) and ``events`` is
a list of JSON-compatible dicts:
  - ``{"event": "match",      "frame": n, "track_id": t, "identity": name,
       "confidence": c}`` on the first majority-confirmed identity;
  - ``{"event": "reidentify", "frame": n, "track_id": t,
       "previous_identity": a, "identity": b, "confidence": c}`` when the
       majority-confirmed identity changes from one enrolled name to another.

Runtime rules
-------------
- **Offline at runtime**: the InsightFace backend only ever loads the local,
  checksum-verified ``models/buffalo_l`` directory. A missing model
  directory, a failed ``models.lock`` verification, or an absent InsightFace
  install all fail closed with an exception — no download is ever attempted.
- Gallery embeddings are regenerated from ``gallery/*.jpg`` on every
  construction and cached only in memory for the run (no persisted identity
  store). Swapping the gallery = dropping a ``.jpg`` plus one config line.
- InsightFace pretrained models are research/non-commercial licensed;
  demo/internal use only.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent

UNKNOWN = "UNKNOWN"
VOTE_WINDOW = 20
# Kill-criterion tuning band from the build spec: thresholds outside this
# range are a config error, not a silent fallback.
THRESHOLD_MIN = 0.35
THRESHOLD_MAX = 0.50

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Injectable backend protocol (deterministic tests / fallback engines)
# ---------------------------------------------------------------------------

class FaceLike(Protocol):
    """Duck-type shared by InsightFace ``Face`` objects and test fakes."""

    bbox: Any  # (x1, y1, x2, y2) in crop coordinates
    normed_embedding: Any  # unit-norm 512-D vector


class FaceBackend(Protocol):
    """Minimal interface required from a swappable face-analysis backend."""

    def get(self, image: np.ndarray) -> list[FaceLike]:
        """Detect faces in a BGR image and return faces with embeddings."""
        ...


# ---------------------------------------------------------------------------
# Real backend: InsightFace buffalo_l, local-only
# ---------------------------------------------------------------------------

class InsightFaceBackend:
    """Wrap ``insightface.app.FaceAnalysis`` pinned to local buffalo_l files.

    InsightFace 2.x accepts a directory path as ``name`` and loads every
    ``*.onnx`` inside it; pointing that at the verified ``models/buffalo_l``
    directory bypasses ``ensure_available``'s download path entirely.
    """

    def __init__(self, model_dir: os.PathLike[str] | str) -> None:
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"buffalo_l model directory not found: {model_dir}. "
                "Run scripts/download_models.py and verify_models.py first; "
                "no runtime downloads are permitted."
            )
        if not any(model_dir.glob("*.onnx")):
            raise RuntimeError(
                f"buffalo_l model directory has no .onnx files: {model_dir}"
            )
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(name=str(model_dir))
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    def get(self, image: np.ndarray) -> list[FaceLike]:
        return list(self._app.get(image))


def _verify_models() -> None:
    """Run the pinned-checksum verifier boundary; fail closed on mismatch."""
    scripts = ROOT / "scripts"
    lock = ROOT / "models.lock"
    if not lock.is_file():
        raise FileNotFoundError(f"models.lock not found at {lock}")
    import sys

    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import verify_models  # type: ignore[import-not-found]

    if verify_models.main() != 0:
        raise RuntimeError("models.lock verification failed; refusing to run")


def load_config(config_path: os.PathLike[str] | str) -> dict[str, Any]:
    """Load ``config.yaml``."""
    import yaml

    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config file did not contain a mapping: {config_path}")
    return config


# ---------------------------------------------------------------------------
# Per-track identity state
# ---------------------------------------------------------------------------

@dataclass
class _IdentityState:
    votes: deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    identity: str = UNKNOWN
    confidence: float = 0.0
    last_attempt_frame: int | None = None
    has_matched: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FaceRecognizer:
    """Resolve track IDs to gallery identities.

    Parameters
    ----------
    config : mapping | str | Path
        Parsed config mapping, or a path to ``config.yaml``.
    gallery_dir : str | Path
        Directory containing one ``<identity>.jpg`` per enrolled person.
    backend : FaceBackend | None
        Injectable face-analysis engine. ``None`` loads InsightFace from the
        local ``models/buffalo_l`` directory (after ``models.lock``
        verification unless ``verify=False``).
    model_dir : str | Path | None
        Location of the local buffalo_l ONNX files.
    verify : bool
        Run the ``models.lock`` checksum verifier before loading real models.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | str | Path,
        *,
        gallery_dir: os.PathLike[str] | str = ROOT / "gallery",
        backend: FaceBackend | None = None,
        model_dir: os.PathLike[str] | str = ROOT / "models" / "buffalo_l",
        verify: bool = True,
    ) -> None:
        if isinstance(config, (str, os.PathLike)):
            config = load_config(config)
        self._config = dict(config)

        self.match_threshold = float(self._config.get("match_threshold", 0.4))
        if not THRESHOLD_MIN <= self.match_threshold <= THRESHOLD_MAX:
            raise ValueError(
                f"match_threshold {self.match_threshold} outside allowed "
                f"tuning band [{THRESHOLD_MIN}, {THRESHOLD_MAX}]"
            )
        self.face_min_size_px = int(self._config.get("face_min_size_px", 48))
        self.embed_every_n_tracks = int(self._config.get("embed_every_n_tracks", 1))
        if self.embed_every_n_tracks < 1:
            raise ValueError("embed_every_n_tracks must be >= 1")
        self.face_check_interval_frames = int(
            self._config.get("face_check_interval_frames", 15)
        )
        if self.face_check_interval_frames < 1:
            raise ValueError("face_check_interval_frames must be >= 1")

        self.gallery_dir = Path(gallery_dir)
        if backend is None:
            if verify:
                _verify_models()
            backend = InsightFaceBackend(model_dir)
        self._backend = backend

        # Gallery embeddings are always rebuilt from photos on construction —
        # in-memory cache for this run only, so gallery swaps need no code
        # changes (add a .jpg + one config.yaml line and rerun).
        self._gallery = self._build_gallery()

        self._tracks: dict[int, _IdentityState] = {}
        self._last_frame: int | None = None

    # -- gallery -----------------------------------------------------------

    def _build_gallery(self) -> dict[str, np.ndarray]:
        """Embed every configured identity's photo; fail closed on bad photos."""
        import cv2

        gallery_cfg = self._config.get("gallery", {})
        if not isinstance(gallery_cfg, dict):
            raise ValueError("config 'gallery' must be a mapping of identity -> metadata")
        embeddings: dict[str, np.ndarray] = {}
        for identity in sorted(gallery_cfg):
            photo = self.gallery_dir / f"{identity}.jpg"
            if not photo.is_file():
                # Allow other image suffixes without config edits.
                candidates = sorted(
                    p for p in self.gallery_dir.glob(f"{identity}.*")
                    if p.suffix.lower() in IMAGE_SUFFIXES
                )
                if not candidates:
                    raise FileNotFoundError(
                        f"gallery photo missing for enrolled identity "
                        f"{identity!r} in {self.gallery_dir}"
                    )
                photo = candidates[0]
            image = cv2.imread(str(photo))
            if image is None:
                raise ValueError(f"gallery photo unreadable: {photo}")
            faces = [f for f in self._backend.get(image)
                     if self._face_size(f) >= self.face_min_size_px]
            if not faces:
                raise RuntimeError(
                    f"no usable face (>= {self.face_min_size_px}px) in "
                    f"gallery photo {photo}"
                )
            best = max(faces, key=self._face_size)
            embedding = np.asarray(best.normed_embedding, dtype=np.float64)
            norm = float(np.linalg.norm(embedding))
            if not np.isfinite(norm) or norm == 0.0:
                raise RuntimeError(f"degenerate embedding for {photo}")
            embeddings[str(identity)] = embedding / norm
        return embeddings

    @property
    def gallery_identities(self) -> list[str]:
        return sorted(self._gallery)

    # -- frame loop ---------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        tracks: Sequence[Mapping[str, Any]],
        frame_index: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Resolve identities for the tracks present at ``frame_index``."""
        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if self._last_frame is not None and frame_index != self._last_frame + 1:
            raise ValueError("frame_index must increase by exactly one")

        events: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        present: set[int] = set()

        for track in tracks:
            track_id = int(track["track_id"])
            present.add(track_id)
            state = self._tracks.setdefault(track_id, _IdentityState())

            if track.get("state", "active") == "active" and self._due(state, track_id, frame_index):
                self._attempt(frame, track, state, frame_index)

            previous = state.identity
            decided = self._decide(state)
            if decided != previous:
                state.identity = decided
                if decided != UNKNOWN:
                    if previous != UNKNOWN:
                        events.append({
                            "event": "reidentify",
                            "frame": frame_index,
                            "track_id": track_id,
                            "previous_identity": previous,
                            "identity": decided,
                            "confidence": state.confidence,
                        })
                    else:
                        events.append({
                            "event": "match",
                            "frame": frame_index,
                            "track_id": track_id,
                            "identity": decided,
                            "confidence": state.confidence,
                        })
                    state.has_matched = True

            results.append({
                "track_id": track_id,
                "identity": state.identity,
                "confidence": float(state.confidence),
            })

        for track_id in [t for t in self._tracks if t not in present]:
            del self._tracks[track_id]

        self._last_frame = frame_index
        json.dumps({"results": results, "events": events})  # contract check
        return results, events

    # -- internals -----------------------------------------------------------

    def _due(self, state: _IdentityState, track_id: int, frame_index: int) -> bool:
        """Whether this track should get a face attempt this frame."""
        if state.last_attempt_frame is None:
            return True
        if state.identity == UNKNOWN:
            # Re-embed UNKNOWN tracks periodically as pose improves.
            return frame_index - state.last_attempt_frame >= self.face_check_interval_frames
        # Known tracks: stagger work across tracks per embed_every_n_tracks.
        return (track_id + frame_index) % self.embed_every_n_tracks == 0

    def _attempt(
        self,
        frame: np.ndarray,
        track: Mapping[str, Any],
        state: _IdentityState,
        frame_index: int,
    ) -> None:
        crop = self._crop(frame, track["bbox"])
        if crop is None:
            return
        faces = [f for f in self._backend.get(crop)
                 if self._face_size(f) >= self.face_min_size_px]
        state.last_attempt_frame = frame_index
        if not faces:
            return
        face = max(faces, key=self._face_size)
        embedding = np.asarray(face.normed_embedding, dtype=np.float64)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm == 0.0:
            return
        embedding /= norm

        best_identity, best_sim = UNKNOWN, -1.0
        for identity, gallery_vec in self._gallery.items():
            sim = float(np.dot(embedding, gallery_vec))
            if sim > best_sim:
                best_identity, best_sim = identity, sim
        if best_sim >= self.match_threshold:
            vote = best_identity
            state.confidence = best_sim
        else:
            vote = UNKNOWN
            state.confidence = max(0.0, best_sim)
        state.votes.append(vote)

    def _decide(self, state: _IdentityState) -> str:
        """Majority vote over the rolling attempt window.

        A label must hold a strict majority of recorded attempts; otherwise
        the previous decision stands (anti-flicker).
        """
        if not state.votes:
            return UNKNOWN
        top, count = Counter(state.votes).most_common(1)[0]
        if count * 2 > len(state.votes):
            return str(top)
        return state.identity

    @staticmethod
    def _face_size(face: FaceLike) -> float:
        x1, y1, x2, y2 = [float(v) for v in np.asarray(face.bbox).ravel()[:4]]
        return min(abs(x2 - x1), abs(y2 - y1))

    @staticmethod
    def _crop(frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1 = max(0, int(math.floor(float(bbox[0]))))
        y1 = max(0, int(math.floor(float(bbox[1]))))
        x2 = min(width, int(math.ceil(float(bbox[2]))))
        y2 = min(height, int(math.ceil(float(bbox[3]))))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else None

    def reset(self) -> None:
        """Drop all per-track state (gallery cache is kept for the run)."""
        self._tracks.clear()
        self._last_frame = None


Identifier = FaceRecognizer
