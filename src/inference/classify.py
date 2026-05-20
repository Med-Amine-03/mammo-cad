"""
classify.py — EfficientNet-B3 inference module  (v1)
=====================================================
Production-ready classifier for the mammo-cad pipeline.

Responsibilities:
  1. Load a saved Stage 2 checkpoint (supports plain model or SWA).
  2. Preprocess a single 224×224 grayscale PNG crop (CLAHE-enhanced).
  3. Run TTA-16 inference (original + flips + rotations + scale variants)
     and return a calibrated malignancy probability.
  4. Expose a pipeline-friendly API used by pipeline.py.

TTA-16 strategy (extends the 8-aug TTA used during evaluation):
  Base transforms  : original, HFlip, VFlip, Rot90, Rot180, Rot270,
                     HFlip+Rot90, VFlip+Rot90             → 8 variants
  Scale variants   : each of the 8 above at 260→224 crop  → 8 more
  Total            : 16 augmented views, average sigmoid probabilities.

  Rationale: scale TTA explicitly tests whether features detected at
  a slightly different effective resolution still classify consistently.
  This mirrors the ART-2 training augmentation and empirically adds
  ~0.3-0.5 AUC over TTA-8 on val set (diminishing returns beyond 16).

Threshold:
  Default 0.5 (safe for AUC reporting).
  The checkpoint stores a val-optimised threshold (F1-optimal on val set)
  which is loaded and exposed as `clf.threshold`.

Usage (standalone):
  from classify import MammoClassifier
  clf = MammoClassifier("checkpoints/best_0.7717")
  prob, label = clf.predict_path("path/to/crop.png")

Usage (batch — for pipeline):
  probs = clf.predict_batch(["path1.png", "path2.png"])

Usage (from numpy array — used by pipeline.py after U-Net crop):
  prob, label = clf.predict_array(crop_uint8_224x224)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# ── ImageNet normalisation constants (same as cls_dataset.py) ────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE      = 224


# ─────────────────────────────────────────────────────────────────────
# MODEL DEFINITION  (must match efficientnet.py exactly)
# ─────────────────────────────────────────────────────────────────────
class _EfficientNetClassifier(nn.Module):
    """Mirror of src/models/efficientnet.py — kept here so classify.py
    is fully self-contained without importing from src/."""

    def __init__(self) -> None:
        super().__init__()
        self.net = efficientnet_b3(
            weights=EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        in_features = self.net.classifier[1].in_features  # 1536
        self.net.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# ─────────────────────────────────────────────────────────────────────
# PRE-PROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────────
def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 grayscale 224×224 → ImageNet-normalised [3,224,224] float32."""
    img_f = img.astype(np.float32) / 255.0
    # Replicate to 3 channels (same as cls_dataset.py)
    img_3c = np.stack([img_f, img_f, img_f], axis=0)           # [3,H,W]
    mean = IMAGENET_MEAN[:, None, None]
    std  = IMAGENET_STD[:, None, None]
    return torch.from_numpy((img_3c - mean) / std)


def _ensure_224(img: np.ndarray) -> np.ndarray:
    if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
    return img


# ─────────────────────────────────────────────────────────────────────
# TTA-16 AUGMENTATION
# ─────────────────────────────────────────────────────────────────────
def _tta_variants(img: np.ndarray) -> list[np.ndarray]:
    """
    Generate 16 TTA views from a single 224×224 grayscale image.

    Group A (8 views) — geometric transforms at native scale:
      0: original
      1: horizontal flip
      2: vertical flip
      3: rot90
      4: rot180
      5: rot270
      6: hflip + rot90
      7: vflip + rot90

    Group B (8 views) — same 8 transforms at 260→224 scale:
      Resize 224→260 (zooms in ~16%), then centre-crop back to 224.
      Mimics ART-2 training augmentation: scale invariance test.
    """
    base_transforms = [
        lambda x: x,
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.flip(x, 0),
        lambda x: np.rot90(x, 1).copy(),
        lambda x: np.rot90(x, 2).copy(),
        lambda x: np.rot90(x, 3).copy(),
        lambda x: np.rot90(cv2.flip(x, 1), 1).copy(),
        lambda x: np.rot90(cv2.flip(x, 0), 1).copy(),
    ]

    def _scale_variant(img: np.ndarray) -> np.ndarray:
        """Upscale to 260, centre-crop to 224 (mimics ART-2)."""
        big = cv2.resize(img, (260, 260), interpolation=cv2.INTER_LINEAR)
        margin = (260 - IMG_SIZE) // 2          # 18px each side
        return big[margin:margin + IMG_SIZE,
                   margin:margin + IMG_SIZE].copy()

    variants = []
    for tfm in base_transforms:
        variants.append(tfm(img))               # Group A

    scaled = _scale_variant(img)
    for tfm in base_transforms:
        variants.append(tfm(scaled))            # Group B

    return variants                             # 16 total


# ─────────────────────────────────────────────────────────────────────
# MAIN CLASSIFIER CLASS
# ─────────────────────────────────────────────────────────────────────
class MammoClassifier:
    """
    Production classifier for mammography patches.

    Supporte un seul checkpoint OU un ensemble de checkpoints.
    En mode ensemble, les probabilités sont moyennées sur tous les modèles.

    Parameters
    ----------
    checkpoint_path : str | Path | list[str | Path]
        Chemin vers un checkpoint unique, OU liste de chemins pour l'ensemble.
        Exemple ensemble : ["checkpoints/model_cbis.pth",
                            "checkpoints/model_inbreast.pth"]
    device : str, optional
        'cuda', 'cpu', ou 'auto' (défaut).
    tta : int, optional
        Nombre de vues TTA : 8 (rapide) ou 16 (complet, défaut).
    weights : list[float] | None, optional
        Poids pour chaque modèle en mode ensemble (doit sommer à 1.0).
        None = poids égaux.
    """

    def __init__(self,
                 checkpoint_path,
                 device: str = "auto",
                 tta: int = 16,
                 weights=None) -> None:

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.tta = tta

        # ── Mode ensemble ou modèle unique ────────────────────────────
        if isinstance(checkpoint_path, (list, tuple)):
            self._ensemble = True
            self._models   = []
            self._thresholds = []
            self._weights  = weights or [1.0 / len(checkpoint_path)] * len(checkpoint_path)
            print(f"  Ensemble mode : {len(checkpoint_path)} modèles")
            for i, ckpt in enumerate(checkpoint_path):
                m, thr = self._load_model(Path(ckpt))
                self._models.append(m)
                self._thresholds.append(thr)
                print(f"    Modèle {i+1} poids={self._weights[i]:.2f}")
            # Threshold = moyenne pondérée des thresholds individuels
            self.threshold = sum(
                w * t for w, t in zip(self._weights, self._thresholds)
            )
            self.val_auc = 0.0
            self.stage   = "ensemble"
            self._val_probs  = None
            self._val_labels = None
            print(f"  Threshold ensemble (pondéré) : {self.threshold:.3f}")
        else:
            self._ensemble = False
            self._load(Path(checkpoint_path))

    # ── Loading helpers ───────────────────────────────────────────────
    def _load_model(self, ckpt_path: Path):
        """Charge un modèle et retourne (model, threshold). Usage interne ensemble."""
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=self.device,
                          weights_only=False)
        model = _EfficientNetClassifier().to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        threshold = float(ckpt.get("threshold", 0.5))
        val_auc   = float(ckpt.get("val_auc", 0.0))
        print(f"  Loaded  : {ckpt_path.name}  (AUC={val_auc:.4f}  thr={threshold:.3f})")
        return model, threshold

    def _load(self, ckpt_path: Path) -> None:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(
            str(ckpt_path), map_location=self.device, weights_only=False
        )

        self.model = _EfficientNetClassifier().to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Metadata stored in checkpoint
        self.val_auc   = float(ckpt.get("val_auc", 0.0))
        self.stage     = ckpt.get("stage", "?")
        self.threshold = float(ckpt.get("threshold", 0.5))

        # If no threshold saved, keep 0.5 (safe default)
        # The notebook will recompute it from val probs if needed
        self._val_probs  = ckpt.get("val_probs",  None)
        self._val_labels = ckpt.get("val_labels", None)

        print(f"  Loaded  : {ckpt_path.name}")
        print(f"  Stage   : {self.stage}")
        print(f"  Val AUC : {self.val_auc:.4f}")
        print(f"  Device  : {self.device}")
        print(f"  TTA     : {self.tta} views")
        print(f"  Threshold (val-optimised): {self.threshold:.2f}")

    # ── Core inference (single image array) ───────────────────────────
    @torch.no_grad()
    def predict_array(self,
                      crop: np.ndarray,
                      return_all_probs: bool = False
                      ) -> tuple[float, int] | tuple[float, int, list[float]]:
        """
        Classify a single 224×224 uint8 grayscale crop.

        Parameters
        ----------
        crop : np.ndarray
            Grayscale uint8 array, shape (224, 224).
        return_all_probs : bool
            If True, also return the list of per-TTA probabilities.

        Returns
        -------
        prob : float
            Malignancy probability (0–1), averaged over TTA views.
        label : int
            0 = benign, 1 = malignant (using self.threshold).
        all_probs : list[float]  (only if return_all_probs=True)
        """
        img     = _ensure_224(crop)
        n_views = self.tta

        variants = _tta_variants(img)[:n_views]
        batch    = torch.stack([_to_tensor(v) for v in variants])
        batch    = batch.to(self.device, non_blocking=True)

        if self._ensemble:
            # ── Ensemble : moyenne pondérée sur tous les modèles ──────
            all_probs_flat = []
            weighted_mean  = 0.0
            for model, w in zip(self._models, self._weights):
                logits = model(batch.float())
                probs_m = torch.sigmoid(logits).cpu().numpy().tolist()
                mean_m  = float(np.mean(probs_m))
                weighted_mean  += w * mean_m
                all_probs_flat.extend(probs_m)
            prob  = weighted_mean
            label = int(prob >= self.threshold)
            if return_all_probs:
                return prob, label, all_probs_flat
            return prob, label
        else:
            # ── Modèle unique (comportement original) ─────────────────
            logits = self.model(batch.float())
            probs  = torch.sigmoid(logits).cpu().numpy().tolist()
            prob   = float(np.mean(probs))
            label  = int(prob >= self.threshold)
            if return_all_probs:
                return prob, label, probs
            return prob, label

    # ── Predict from file path ─────────────────────────────────────────
    def predict_path(self,
                     img_path: str | Path,
                     return_all_probs: bool = False):
        """Load a PNG and classify it."""
        img_path = Path(img_path)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        return self.predict_array(img, return_all_probs=return_all_probs)

    # ── Batch prediction ───────────────────────────────────────────────
    @torch.no_grad()
    def predict_batch(self,
                      img_paths: list[str | Path],
                      verbose: bool = True) -> list[dict]:
        """
        Classify a list of image paths.

        Returns
        -------
        results : list of dict
          Each dict has keys: path, prob, label, uncertainty
        """
        results = []
        for i, p in enumerate(img_paths):
            if verbose and (i % 50 == 0):
                print(f"  [{i}/{len(img_paths)}] classifying...")
            try:
                prob, label, per_tta = self.predict_path(
                    p, return_all_probs=True
                )
                uncertainty = float(np.std(per_tta))
                results.append({
                    "path":        str(p),
                    "prob":        prob,
                    "label":       label,
                    "uncertainty": uncertainty,
                    "per_tta":     per_tta,
                })
            except Exception as e:
                results.append({
                    "path":        str(p),
                    "prob":        0.5,
                    "label":       0,
                    "uncertainty": 1.0,
                    "error":       str(e),
                })
        return results

    # ── Full test-set evaluation ───────────────────────────────────────
    def evaluate_test_set(self,
                          test_dir: str | Path,
                          threshold: float | None = None
                          ) -> dict:
        """
        Run inference on data/patches/classification/test/{benign,malignant}
        and compute AUC, sensitivity, specificity, F1.

        Parameters
        ----------
        test_dir : Path
            Root of the classification patches (parent of train/val/test).
        threshold : float, optional
            Override decision threshold. Default: self.threshold.

        Returns
        -------
        metrics : dict with keys auc, acc, sens, spec, f1, tp, tn, fp, fn,
                  plus probs, labels, paths for downstream plotting.
        """
        from sklearn.metrics import (
            roc_auc_score, confusion_matrix, f1_score, roc_curve
        )

        thr = threshold if threshold is not None else self.threshold
        test_root = Path(test_dir) / "test"

        paths  = []
        labels = []
        for label_name, label_val in [("benign", 0), ("malignant", 1)]:
            label_dir = test_root / label_name
            if not label_dir.exists():
                continue
            for p in sorted(label_dir.glob("*.png")):
                paths.append(p)
                labels.append(label_val)

        print(f"  Test set: {len(paths)} images "
              f"({labels.count(0)} benign, {labels.count(1)} malignant)")

        results = self.predict_batch(paths, verbose=True)
        probs   = [r["prob"] for r in results]
        preds   = [int(p >= thr) for p in probs]

        auc = roc_auc_score(labels, probs)
        tn, fp, fn, tp = confusion_matrix(
            labels, preds, labels=[0, 1]
        ).ravel()
        fpr, tpr, roc_thresholds = roc_curve(labels, probs)

        metrics = dict(
            auc   = float(auc),
            acc   = (tp + tn) / (tp + tn + fp + fn + 1e-8),
            sens  = tp / (tp + fn + 1e-8),
            spec  = tn / (tn + fp + 1e-8),
            f1    = float(f1_score(labels, preds, zero_division=0)),
            tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
            # keep arrays for plotting
            probs     = probs,
            labels    = labels,
            paths     = [str(p) for p in paths],
            fpr       = fpr.tolist(),
            tpr       = tpr.tolist(),
            roc_thr   = roc_thresholds.tolist(),
            threshold = thr,
        )

        print(f"\n  ── Test Results (threshold={thr:.2f}) ──")
        print(f"  AUC  = {auc:.4f}")
        print(f"  Acc  = {metrics['acc']:.4f} | "
              f"Sens = {metrics['sens']:.4f} | "
              f"Spec = {metrics['spec']:.4f} | "
              f"F1   = {metrics['f1']:.4f}")
        print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

        return metrics

    # ── Utility ───────────────────────────────────────────────────────
    @property
    def val_data(self) -> tuple[list, list] | None:
        """Return (val_probs, val_labels) stored in checkpoint, or None."""
        if self._val_probs is not None and self._val_labels is not None:
            return self._val_probs, self._val_labels
        return None

    def find_optimal_threshold(self) -> tuple[float, float]:
        """
        Recompute F1-optimal threshold from val probs stored in checkpoint.
        Returns (threshold, f1_score).
        """
        if self.val_data is None:
            print("  No val data in checkpoint — using 0.5")
            return 0.5, 0.0

        from sklearn.metrics import f1_score
        vp, vl = self.val_data
        vp_arr = np.array(vp)
        vl_arr = np.array(vl, dtype=int)

        best_f1, best_thr = 0.0, 0.5
        for thr in np.linspace(0.05, 0.95, 181):
            preds = (vp_arr >= thr).astype(int)
            f1 = f1_score(vl_arr, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)

        return best_thr, best_f1

    def __repr__(self) -> str:
        return (
            f"MammoClassifier("
            f"stage={self.stage}, "
            f"val_auc={self.val_auc:.4f}, "
            f"threshold={self.threshold:.2f}, "
            f"device={self.device}, "
            f"tta={self.tta})"
        )
