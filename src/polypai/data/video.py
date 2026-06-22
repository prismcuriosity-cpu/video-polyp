"""Streaming colonoscopy-video frame reader for the frame selector.

Yields RGB frames lazily so a full procedure (tens of thousands of frames) never
needs to be held in RAM — feeding the streaming-dataset requirement and the
informative-frame-selection front-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


class VideoFrameReader:
    def __init__(self, path: str | Path, stride: int = 1, max_frames: int | None = None,
                 resize_to: int | None = None):
        self.path = str(path)
        self.stride = max(1, stride)
        self.max_frames = max_frames
        self.resize_to = resize_to

    def __iter__(self) -> Iterator[np.ndarray]:
        import cv2

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")
        idx = yielded = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % self.stride == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.resize_to:
                        h, w = frame.shape[:2]
                        scale = self.resize_to / max(h, w)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    yield frame
                    yielded += 1
                    if self.max_frames and yielded >= self.max_frames:
                        break
                idx += 1
        finally:
            cap.release()

    @property
    def fps(self) -> float:
        import cv2

        cap = cv2.VideoCapture(self.path)
        f = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(f)
