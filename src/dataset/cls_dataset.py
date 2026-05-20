"""
cls_dataset.py — CBIS-DDSM classification dataset  (v5)
=========================================================
Changes vs v4:
  [ART-2] Multi-scale input: randomly resize to 224 or 260 during training
           Shia et al. 2025 trained at both 224×224 and 260×260.
           Forces scale invariance — calcifications appear at different
           effective sizes across scanner models and acquisition settings.
           At inference: always 224 (consistent with extract_crops.py).

  [ART-3] Random translation ±10% during training
           Shia et al. used random affine transforms including translation.
           Shifting the lesion away from centre forces the model not to
           rely on spatial position of the calcification cluster.

Augmentation stack (training only):
  Spatial   : HFlip, VFlip, Rot90/180/270
  Scale     : random resize 224 or 260 → back to 224       [ART-2]
  Translate : random shift ±10%                             [ART-3]
  Elastic   : small random elastic distortion (p=0.3)
  Intensity : brightness/contrast, gamma, noise, blur
"""

import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from pathlib import Path


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ClsDataset(Dataset):
    """
    Reads 224×224 grayscale PNG patches from:
      root_dir / split / benign    / *.png  → label 0
      root_dir / split / malignant / *.png  → label 1
    """

    def __init__(self, root_dir: Path, split: str,
                 augment: bool = False):
        self.root    = Path(root_dir) / split
        self.augment = augment
        self.split   = split
        self.samples: list[tuple[Path, int]] = []

        for label_name, label_val in [("benign", 0), ("malignant", 1)]:
            label_dir = self.root / label_name
            if not label_dir.exists():
                print(f"  WARNING: {label_dir} does not exist — skipping")
                continue
            for p in sorted(label_dir.glob("*.png")):
                self.samples.append((p, label_val))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root}. "
                f"Run patch_extraction_cls.py first."
            )

        n_b = sum(1 for _, l in self.samples if l == 0)
        n_m = sum(1 for _, l in self.samples if l == 1)
        self.pos_weight: float = n_b / max(n_m, 1)

        print(f"  [{split:5s}] Total={len(self.samples):4d} | "
              f"Benign={n_b:4d} | Malignant={n_m:4d} | "
              f"pos_weight={self.pos_weight:.3f}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Cannot read image: {path}")

        if self.augment:
            img = self._augment(img)

        tensor = self._to_tensor(img)
        return tensor, torch.tensor(label, dtype=torch.float32)

    # ------------------------------------------------------------------
    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Full augmentation stack for training."""

        # ── Spatial ───────────────────────────────────────────────────
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k).copy()

        # ── [ART-2] Multi-scale: randomly resize to 224 or 260 ────────
        # Simulates calcifications appearing at different effective
        # resolutions across scanner models (Shia et al. 2025).
        # Always returns 224×224 for model compatibility.
        if random.random() < 0.5:
            scale = 260
            img = cv2.resize(img, (scale, scale),
                             interpolation=cv2.INTER_LINEAR)
            img = cv2.resize(img, (224, 224),
                             interpolation=cv2.INTER_AREA)

        # ── [ART-3] Random translation ±10% ───────────────────────────
        # Forces model not to rely on lesion being centred.
        if random.random() < 0.5:
            h, w  = img.shape
            max_dx = int(0.10 * w)
            max_dy = int(0.10 * h)
            dx = random.randint(-max_dx, max_dx)
            dy = random.randint(-max_dy, max_dy)
            M  = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(
                img, M, (w, h),
                borderMode=cv2.BORDER_REFLECT_101
            )

        # ── Elastic distortion (p=0.3) ────────────────────────────────
        if random.random() < 0.3:
            img = self._elastic(img, alpha=15, sigma=4)

        # ── Brightness / contrast (p=0.5) ────────────────────────────
        if random.random() > 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta  = random.uniform(-10.0, 10.0)
            img   = np.clip(
                img.astype(np.float32) * alpha + beta, 0, 255
            ).astype(np.uint8)

        # ── Gamma (p=0.4) ─────────────────────────────────────────────
        if random.random() < 0.4:
            gamma = random.uniform(0.80, 1.25)
            table = np.array(
                [((i / 255.0) ** gamma) * 255 for i in range(256)],
                dtype=np.uint8,
            )
            img = cv2.LUT(img, table)

        # ── Gaussian noise (p=0.4) ────────────────────────────────────
        if random.random() < 0.4:
            sigma = random.uniform(0.0, 4.0)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img   = np.clip(
                img.astype(np.float32) + noise, 0, 255
            ).astype(np.uint8)

        # ── Gaussian blur (p=0.3) ─────────────────────────────────────
        if random.random() < 0.3:
            sigma = random.uniform(0.3, 0.8)
            img   = cv2.GaussianBlur(img, (3, 3), sigma)

        return img

    # ------------------------------------------------------------------
    @staticmethod
    def _elastic(img: np.ndarray,
                 alpha: float = 15,
                 sigma: float = 4) -> np.ndarray:
        h, w = img.shape
        dx = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32),
            (0, 0), sigma,
        ) * alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32),
            (0, 0), sigma,
        ) * alpha
        x, y  = np.meshgrid(np.arange(w), np.arange(h))
        map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
        return cv2.remap(img, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        """Grayscale uint8 → ImageNet-normalised 3-channel float tensor."""
        img_f = img.astype(np.float32) / 255.0
        t = torch.from_numpy(img_f).unsqueeze(0).repeat(3, 1, 1)
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3,1,1)
        std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3,1,1)
        return (t - mean) / std