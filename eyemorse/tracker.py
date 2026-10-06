"""Webcam capture + MediaPipe Face Landmarker (landmarks, iris, blink scores)."""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# 6-point eye contours for the eye-aspect-ratio fallback (Soukupova & Cech 2016).
EAR_POINTS = {
    "left": (362, 385, 387, 263, 373, 380),
    "right": (33, 160, 158, 133, 153, 144),
}


def ensure_model(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmark model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def eye_aspect_ratio(lm: np.ndarray, idx: tuple[int, ...]) -> float:
    p = lm[list(idx), :2]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = 2 * np.linalg.norm(p[0] - p[3]) or 1e-6
    return float(vertical / horizontal)


def ear_to_closure(ear: float, open_ear: float = 0.30, closed_ear: float = 0.12) -> float:
    return float(np.clip((open_ear - ear) / (open_ear - closed_ear), 0.0, 1.0))


@dataclass
class FaceFrame:
    landmarks: np.ndarray  # (478, 3) in pixels
    blink_left: float  # 0 = open, 1 = closed
    blink_right: float


class FaceTracker:
    def __init__(self, model_path: str, camera: int = 0, width: int = 640, height: int = 480):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._last_ts = -1

    def read(self) -> tuple[float, np.ndarray, FaceFrame | None]:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera frame grab failed")
        t = time.monotonic()
        return t, frame, self.process(frame, t)

    def process(self, frame_bgr: np.ndarray, t: float) -> FaceFrame | None:
        ts = max(int(t * 1000), self._last_ts + 1)  # VIDEO mode needs increasing ms
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        lm = np.array([[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]])

        scores = {}
        if result.face_blendshapes:
            scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        if "eyeBlinkLeft" in scores and "eyeBlinkRight" in scores:
            left, right = scores["eyeBlinkLeft"], scores["eyeBlinkRight"]
        else:  # fall back to geometry
            left = ear_to_closure(eye_aspect_ratio(lm, EAR_POINTS["left"]))
            right = ear_to_closure(eye_aspect_ratio(lm, EAR_POINTS["right"]))
        return FaceFrame(lm, float(left), float(right))

    def close(self) -> None:
        self.cap.release()
        self.landmarker.close()
