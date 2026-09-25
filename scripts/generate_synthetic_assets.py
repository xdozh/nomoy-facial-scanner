#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np
from skimage import data

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
GALLERY = ROOT / "gallery"
SAMPLES.mkdir(exist_ok=True)
GALLERY.mkdir(exist_ok=True)

astronaut = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
faces = data.lfw_subset()
face_indices = [0, 40, 80, 20]
names = ["alex", "blair", "casey"]
face_tiles = []
for person_index, index in enumerate(face_indices):
    face = np.clip(faces[index] * 255, 0, 255).astype(np.uint8)
    face = cv2.resize(face, (256, 256), interpolation=cv2.INTER_CUBIC)
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
    gallery_image = cv2.copyMakeBorder(face, 128, 128, 128, 128, cv2.BORDER_CONSTANT, value=(160, 160, 160))
    if person_index < len(names):
        cv2.imwrite(str(GALLERY / f"{names[person_index]}.jpg"), gallery_image)
    face_tiles.append(face)

sprite = cv2.resize(astronaut, (300, 300))
height, width, fps, frames = 720, 1280, 10, 300
writer = cv2.VideoWriter(str(SAMPLES / "test.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
if not writer.isOpened():
    raise RuntimeError("Could not create samples/test.mp4")

for frame_index in range(frames):
    frame = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.putText(frame, "SYNTHETIC STOPGAP - NOT RECOGNITION EVIDENCE", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    swap_one = 0.5 * (1 + np.tanh((frame_index - 75) / 4))
    swap_two = 0.5 * (1 + np.tanh((frame_index - 225) / 4))
    swap = swap_one - swap_two
    positions = [
        (int(40 + 480 * swap), 130),
        (int(520 - 480 * swap), 300),
        (920, 190),
    ]
    order = [0, 2, 1] if frame_index % 80 < 40 else [1, 2, 0]
    for person_index in order:
        x, y = positions[person_index]
        person = sprite.copy()
        face = cv2.resize(face_tiles[person_index], (72, 72))
        person[40:112, 114:186] = face
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + 300), min(height, y + 300)
        person_x1, person_y1 = x1 - x, y1 - y
        person_x2, person_y2 = person_x1 + (x2 - x1), person_y1 + (y2 - y1)
        frame[y1:y2, x1:x2] = person[person_y1:person_y2, person_x1:person_x2]
    writer.write(frame)
writer.release()

recognition_writer = cv2.VideoWriter(str(SAMPLES / "recognition_test.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
if not recognition_writer.isOpened():
    raise RuntimeError("Could not create samples/recognition_test.mp4")
recognition_sprite = cv2.resize(astronaut, (260, 260))
for frame_index in range(120):
    frame = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.putText(frame, "SYNTHETIC RECOGNITION STOPGAP", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    positions = [(20, 100), (340, 330), (660, 100), (980, 330)]
    for person_index, (base_x, base_y) in enumerate(positions):
        x = base_x + int(12 * np.sin((frame_index + person_index * 13) / 10))
        y = base_y + int(8 * np.cos((frame_index + person_index * 7) / 12))
        person = recognition_sprite.copy()
        face = cv2.resize(face_tiles[person_index], (96, 96))
        person[28:124, 82:178] = face
        frame[y:y + 260, x:x + 260] = person
    recognition_writer.write(frame)
recognition_writer.release()
print(SAMPLES / "test.mp4")
print(SAMPLES / "recognition_test.mp4")
