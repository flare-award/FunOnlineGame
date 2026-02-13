from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mss
import numpy as np


@dataclass
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int


class ScreenCapture:
    """Fast rectangular screen capture using mss."""

    def __init__(self, region: CaptureRegion, image_size: Tuple[int, int], grayscale: bool = False) -> None:
        self.region = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }
        self.image_size = image_size
        self.grayscale = grayscale
        self.sct: Optional[mss.mss] = None

    def __enter__(self) -> "ScreenCapture":
        self.sct = mss.mss()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.sct:
            self.sct.close()
        self.sct = None

    def grab_raw(self) -> np.ndarray:
        if not self.sct:
            self.sct = mss.mss()
        frame = np.array(self.sct.grab(self.region))
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
        if self.grayscale:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            resized = np.expand_dims(resized, axis=-1)
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def grab_preprocessed(self) -> np.ndarray:
        frame = self.grab_raw()
        return self.preprocess(frame)
