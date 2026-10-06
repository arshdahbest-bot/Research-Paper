"""Gaze features from face landmarks and a calibrated feature -> screen mapping.

Features (all scale-invariant, i.e. divided by eye width or inter-ocular
distance, so moving closer to / further from the camera matters less):

* eye_h, eye_v : iris centre relative to the eye-corner midpoint, along and
                 perpendicular to the corner-to-corner axis (both eyes averaged)
* yaw, pitch   : head-rotation proxies from the nose tip relative to the eyes

A ridge-regularised polynomial regression, fitted during calibration, maps the
features to screen pixels.
"""

from __future__ import annotations

import numpy as np

# MediaPipe Face Landmarker indices (478-point mesh incl. iris).
# "image-left" eye = the subject's right eye.
EYES = (
    {"c0": 33, "c1": 133, "iris": 468},  # image-left eye
    {"c0": 362, "c1": 263, "iris": 473},  # image-right eye
)
NOSE_TIP = 1
N_FEATURES = 4


def gaze_features(landmarks: np.ndarray) -> np.ndarray:
    """landmarks: (478, >=2) array in *pixel* units. Returns [h, v, yaw, pitch]."""
    lm = landmarks[:, :2]
    hs, vs, centers = [], [], []
    for eye in EYES:
        p0, p1 = lm[eye["c0"]], lm[eye["c1"]]
        axis = p1 - p0
        width = float(np.linalg.norm(axis)) or 1e-6
        u = axis / width
        n = np.array([-u[1], u[0]])
        center = (p0 + p1) / 2
        rel = lm[eye["iris"]] - center
        hs.append(float(rel @ u) / width)
        vs.append(float(rel @ n) / width)
        centers.append(center)

    left_c, right_c = centers
    iod_vec = right_c - left_c
    iod = float(np.linalg.norm(iod_vec)) or 1e-6
    u = iod_vec / iod
    n = np.array([-u[1], u[0]])
    nose_rel = lm[NOSE_TIP] - (left_c + right_c) / 2
    yaw = float(nose_rel @ u) / iod
    pitch = float(nose_rel @ n) / iod
    return np.array([np.mean(hs), np.mean(vs), yaw, pitch])


class GazeMapper:
    """Feature vector -> (x, y) screen pixels via ridge polynomial regression."""

    def __init__(self, alpha: float = 0.01, quadratic: bool = True):
        self.alpha = alpha
        self.quadratic = quadratic
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.W: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.W is not None

    def _design(self, F: np.ndarray) -> np.ndarray:
        Z = (F - self.mu) / self.sd
        cols = [np.ones(len(Z)), *Z.T]
        if self.quadratic:
            h, v = Z[:, 0], Z[:, 1]
            cols += [h * h, v * v, h * v]
        return np.column_stack(cols)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "GazeMapper":
        F = np.asarray(features, float)
        Y = np.asarray(targets, float)
        self.mu = F.mean(axis=0)
        self.sd = F.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        X = self._design(F)
        penalty = self.alpha * len(X) * np.eye(X.shape[1])
        penalty[0, 0] = 0.0  # don't shrink the intercept
        self.W = np.linalg.solve(X.T @ X + penalty, X.T @ Y)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("GazeMapper is not calibrated")
        F = np.atleast_2d(np.asarray(features, float))
        out = self._design(F) @ self.W
        return out[0] if np.ndim(features) == 1 else out

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "quadratic": self.quadratic,
            "mu": self.mu.tolist(),
            "sd": self.sd.tolist(),
            "W": self.W.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GazeMapper":
        m = cls(d["alpha"], d["quadratic"])
        m.mu, m.sd, m.W = (np.array(d[k]) for k in ("mu", "sd", "W"))
        return m


def trim_outliers(features: np.ndarray, k: float = 2.5) -> np.ndarray:
    """Boolean mask keeping samples within k robust SDs (MAD) of the median."""
    F = np.asarray(features, float)
    med = np.median(F, axis=0)
    mad = np.median(np.abs(F - med), axis=0) * 1.4826
    mad[mad < 1e-9] = 1e-9
    return np.all(np.abs(F - med) <= k * mad, axis=1)
