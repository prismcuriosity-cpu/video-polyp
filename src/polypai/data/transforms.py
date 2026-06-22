"""Augmentation/normalisation transforms.

Uses Albumentations when available (recommended for training — rich endoscopy-
appropriate augmentations); otherwise a small NumPy implementation keeps the
``Dataset`` usable.  All transforms operate jointly on image+mask and return
CHW float tensors normalised with ImageNet statistics.
"""

from __future__ import annotations

import numpy as np
import torch

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

try:
    import albumentations as A  # type: ignore

    _HAS_ALB = True
except Exception:  # pragma: no cover
    _HAS_ALB = False


class Compose:
    """Minimal callable wrapper with a unified ``(image, mask)`` interface."""

    def __init__(self, size: int = 512, train: bool = True):
        self.size = size
        self.train = train
        self._alb = self._build_alb() if _HAS_ALB else None

    def _build_alb(self):
        import albumentations as A

        if self.train:
            return A.Compose([
                A.LongestMaxSize(self.size),
                A.PadIfNeeded(self.size, self.size, border_mode=0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.15, rotate_limit=20, p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.HueSaturationValue(10, 15, 10, p=0.3),
                A.GaussNoise(p=0.2),
                A.MotionBlur(blur_limit=5, p=0.15),  # endoscopy motion blur
            ])
        return A.Compose([A.LongestMaxSize(self.size), A.PadIfNeeded(self.size, self.size, border_mode=0)])

    def _np_resize(self, image, mask):
        from polypai.frame_selection.imageops import resize

        h, w = image.shape[:2]
        scale = self.size / max(h, w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        image = resize(image, (nh, nw))
        mask = resize(mask.astype(np.float32), (nh, nw))
        # pad to square
        canvas = np.zeros((self.size, self.size, 3), np.float32)
        mcanvas = np.zeros((self.size, self.size), np.float32)
        canvas[:nh, :nw] = image[..., :3] if image.ndim == 3 else image[..., None]
        mcanvas[:nh, :nw] = mask
        if self.train and np.random.rand() < 0.5:
            canvas, mcanvas = canvas[:, ::-1].copy(), mcanvas[:, ::-1].copy()
        return canvas, (mcanvas > 0.5).astype(np.uint8)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict:
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        if self._alb is not None:
            res = self._alb(image=(image * 255).astype(np.uint8), mask=mask)
            image = res["image"].astype(np.float32) / 255.0
            mask = res["mask"]
        else:
            image, mask = self._np_resize(image, mask)
        image = (image - _MEAN) / _STD
        image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return {"image": image_t.float(), "mask": torch.from_numpy(np.ascontiguousarray(mask))}


def default_train_transforms(size: int = 512) -> Compose:
    return Compose(size=size, train=True)


def default_val_transforms(size: int = 512) -> Compose:
    return Compose(size=size, train=False)
