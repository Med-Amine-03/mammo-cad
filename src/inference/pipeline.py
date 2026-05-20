"""
pipeline.py — mammo-cad end-to-end inference pipeline  (v3)
============================================================

FIXES vs v1/v2
--------------
[FIX-1] preprocess=False par défaut
         Les images de data/processed/inbreast/AllPng/ sont DÉJÀ prétraitées.

[FIX-2] Suppression du threshold-200 dans preprocess_mammogram()
         Remplacé par Otsu + largest blob, safe pour le tissu dense.

[FIX-3] cluster_dist: 100 -> 60px (cohérent avec définition clinique ~4mm)

[FIX-4] — LE VRAI BUG PRINCIPAL —
         Suppression du roundtrip write->read disque pour la segmentation.
         Avant : pipeline écrivait img_proc sur disque, puis run_segmentation()
                 relisait ce fichier avec _load_and_normalise().
                 Problème : si le preprocessing change la taille de l'image
                 (crop, flip), le fichier écrit ≠ l'image originale.
                 Sur certaines images INbreast full-res non croppées, cela
                 donnait 696 raw MC et 22 secondes au lieu de 6 MC et 300ms.
         Après : pipeline utilise run_segmentation_array(img_proc) directement,
                 zero I/O intermédiaire, résultats identiques à segment.py CLI.

Usage
-----
  # Image de data/processed/ (déjà prétraitée) — cas normal :
  python src/inference/pipeline.py \\
      --input  data/processed/inbreast/AllPng/22678622.png \\
      --seg_ckpt  checkpoints/unet_best_fold.pth \\
      --cls_ckpt  checkpoints/best_0.7717/efficientnet_stage2.pth \\
      --output_dir outputs/predictions/

  # Image brute jamais vue par preprocessing.py :
  python src/inference/pipeline.py --input raw.png --preprocess ...

  # Sans Grad-CAM (plus rapide) :
  python src/inference/pipeline.py --input ... --no_gradcam
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inference.segment import (
    load_model as load_seg_model,
    run_segmentation,
    run_segmentation_array,   # [FIX-4] direct array API
    SegmentationResult,
    BoundingBox,
)
from src.inference.classify import MammoClassifier

# medical-image-std (Hamza Gbada / LATIS lab)
from medical_image.algorithms.breast_mask import BreastMaskAlgorithm as _BreastMaskAlg
from medical_image.algorithms.FEBDS import FebdsAlgorithm as _FebdsAlg
from medical_image.data.in_memory_image import InMemoryImage as _InMemImg


# ─────────────────────────────────────────────────────────────────────
# COLOURS  (BGR for OpenCV)
# ─────────────────────────────────────────────────────────────────────
RED    = (68,  68,  255)   # MALIGNANT  #FF4444
YELLOW = (0,   215, 255)   # BENIGN     #FFD700
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)


# ─────────────────────────────────────────────────────────────────────
# PREPROCESSING  [FIX-2] — safe, no tissue destruction
# ─────────────────────────────────────────────────────────────────────
def preprocess_mammogram(img_gray: np.ndarray,
                         apply_clahe: bool = True,
                         clahe_clip: float = 2.0) -> np.ndarray:
    """
    Safe preprocessing for mammogram images.

    Steps:
      1. Flip breast to LEFT  (brighter half heuristic)
      2. Otsu threshold -> keep largest blob -> dilate mask
         (removes border/nipple/collimation artifacts)
      3. Crop black background
      4. CLAHE clip=clahe_clip (optional — set apply_clahe=False for DMID)

    Parameters
    ----------
    apply_clahe : bool
        True  = standard CBIS-DDSM preprocessing (full pipeline)
        False = light preprocessing for DMID (border removal only, no CLAHE)
    clahe_clip : float
        CLAHE clip limit (default 2.0). Lower = less aggressive.
    """
    img = img_gray.copy()

    # 1. Flip to LEFT
    H, W = img.shape
    if img[:, W // 2:].sum() > img[:, :W // 2].sum():
        img = cv2.flip(img, 1)

    # 2. Breast region extraction via medical-image-std BreastMaskAlgorithm
    #    (Otsu + largest connected component — same logic, standardized framework)
    _img_t = _InMemImg(array=torch.from_numpy(img.astype(np.float32)))
    _out_t = _img_t.clone()
    _BreastMaskAlg(mask_only=True, device="cpu")(_img_t, _out_t)
    mask   = (_out_t.pixel_data.numpy() > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask   = cv2.dilate(mask, kernel, iterations=2)
    img    = cv2.bitwise_and(img, img, mask=mask)

    # 3. Crop black background
    cols = np.any(img > 5, axis=0)
    rows = np.any(img > 5, axis=1)
    if cols.any() and rows.any():
        c0, c1 = np.where(cols)[0][[0, -1]]
        r0, r1 = np.where(rows)[0][[0, -1]]
        img    = img[r0:r1+1, c0:c1+1]

    # 4. CLAHE (optional)
    if apply_clahe:
        clahe_op = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        img = clahe_op.apply(img)

    return img


# ─────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────
class GradCAM:
    """GradCAM targeting net.features[7] of EfficientNet-B3 (7×7 spatial)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model  = model
        self._feats: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._fwd = target_layer.register_forward_hook(self._save_feats)
        self._bwd = target_layer.register_full_backward_hook(self._save_grads)

    def _save_feats(self, m, i, o): self._feats = o.detach()
    def _save_grads(self, m, gi, go): self._grads = go[0].detach()
    def remove(self): self._fwd.remove(); self._bwd.remove()

    def generate(self, img_tensor: torch.Tensor,
                 out_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        self.model.eval()
        score = torch.sigmoid(self.model(img_tensor))
        self.model.zero_grad()
        score.backward()
        w   = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._feats).sum(dim=1).squeeze(0))
        # Clip top 5% -> supprime les artefacts extrêmes
        cam_np = cam.cpu().numpy()
        p95    = np.percentile(cam_np, 95)
        cam_np = np.clip(cam_np, 0, p95)
        mn, mx = cam_np.min(), cam_np.max()
        cam_np = (cam_np - mn) / (mx - mn + 1e-8)
        # Smooth -> supprime les rectangles artificiels
        cam_np = cv2.GaussianBlur(cam_np, (7, 7), 0)
        cam_np = cv2.resize(cam_np,
                            (out_size[1], out_size[0]),
                            interpolation=cv2.INTER_CUBIC)
        return cam_np.astype(np.float32)


def _make_gradcam_overlay(crop_gray: np.ndarray,
                          cam: np.ndarray,
                          alpha: float = 0.55) -> np.ndarray:
    # Normaliser le CAM final
    mn, mx = cam.min(), cam.max()
    cam_n  = (cam - mn) / (mx - mn + 1e-8)
    hm = cv2.applyColorMap((cam_n * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    # Blend seulement là où le CAM est chaud (> 0.3) -> fond reste visible
    mask = (cam_n > 0.3).astype(np.float32)[:, :, None]
    result = base.copy().astype(np.float32)
    result = result * (1 - mask * alpha) + hm.astype(np.float32) * (mask * alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# CROP HELPER
# ─────────────────────────────────────────────────────────────────────
def _crop_bbox(img: np.ndarray, bb: BoundingBox,
               pad: int = 32, target: int = 224) -> np.ndarray:
    H, W = img.shape
    x1   = max(0, bb.x1 - pad);  y1 = max(0, bb.y1 - pad)
    x2   = min(W, bb.x2 + pad);  y2 = min(H, bb.y2 + pad)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img[bb.y1:bb.y2, bb.x1:bb.x2]
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────
@dataclass
class RegionResult:
    region_id   : int
    bbox        : List[int]
    label       : str
    prob        : float
    uncertainty : float
    cam_overlay : Optional[np.ndarray] = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────
# ANNOTATION
# ─────────────────────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICK  = 3
FONT_SCALE = 0.55
FONT_THICK = 2


def _annotate_image(canvas: np.ndarray,
                    regions: List[RegionResult],
                    show_prob: bool = True,
                    show_gradcam: bool = True) -> np.ndarray:
    out = canvas.copy()
    for r in regions:
        x1, y1, x2, y2 = r.bbox
        color     = RED if r.label == "MALIGNANT" else YELLOW
        main_text = f"R{r.region_id} {r.label[:3]}"
        sub_text  = f"P={r.prob*100:.1f}% ±{r.uncertainty*100:.1f}%"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_THICK)
        (tw, th), _ = cv2.getTextSize(main_text, FONT, FONT_SCALE, FONT_THICK)
        cv2.rectangle(out, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, main_text, (x1 + 3, y1 - 4),
                    FONT, FONT_SCALE, BLACK, FONT_THICK, cv2.LINE_AA)
        cv2.putText(out, sub_text, (x1 + 3, y1 + 16),
                    FONT, 0.42, color, 1, cv2.LINE_AA)

        if show_gradcam and r.cam_overlay is not None:
            ts  = max(60, min(x2 - x1, y2 - y1, 120))
            th2 = cv2.resize(r.cam_overlay, (ts, ts))
            tx1 = x1 + BOX_THICK + 2
            ty1 = y1 + BOX_THICK + 20
            tx2 = min(tx1 + ts, out.shape[1])
            ty2 = min(ty1 + ts, out.shape[0])
            ta, tb = tx2 - tx1, ty2 - ty1
            if ta > 0 and tb > 0:
                out[ty1:ty2, tx1:tx2] = cv2.addWeighted(
                    out[ty1:ty2, tx1:tx2], 0.3,
                    th2[:tb, :ta], 0.7, 0
                )
                cv2.rectangle(out, (tx1, ty1), (tx2, ty2), WHITE, 1)
    return out


def _build_legend(canvas: np.ndarray) -> np.ndarray:
    H, W = canvas.shape[:2]
    for i, (color, text) in enumerate([(RED, "MALIGNANT"), (YELLOW, "BENIGN")]):
        y = H - 60 + i * 25
        cv2.rectangle(canvas, (W - 200, y - 12), (W - 184, y + 4), color, -1)
        cv2.putText(canvas, text, (W - 178, y),
                    FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    return canvas


# ─────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# [FIX-1] preprocess=False  [FIX-3] cluster_dist=60  [FIX-4] array API
# ─────────────────────────────────────────────────────────────────────
def run_pipeline(
    image_path       : str | Path,
    seg_ckpt_path    : str | Path,
    cls_ckpt_path    : str | Path,
    output_dir       : str | Path,
    seg_threshold    : float = 0.45,
    seg_tta          : bool  = False,
    min_area         : int   = 50,
    cluster_dist     : int   = 60,        # [FIX-3]
    cluster_pad      : int   = 32,
    cls_threshold    : Optional[float] = None,
    fallback_threshold : float = 0.39,
    cls_tta          : int   = 16,
    crop_pad         : int   = 32,
    gradcam          : bool  = True,
    gradcam_alpha    : float = 0.50,
    clf_preloaded    : Optional["MammoClassifier"] = None,
    save_seg_overlay : bool  = True,
    save_crops       : bool  = True,
    preprocess       : bool  = False,     # [FIX-1]
    preprocess_clahe : bool  = True,      # False = border removal only (for DMID)
    clahe_clip       : float = 2.0,
    device           : str   = "auto",
) -> dict:
    t_total    = time.perf_counter()
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  mammo-cad Pipeline  (v3)")
    print(f"  Input      : {image_path.name}")
    print(f"  Device     : {device}")
    print(f"  Preprocess : {preprocess}")
    print(f"  Seg thr    : {seg_threshold}  cluster_dist={cluster_dist}px")
    print(f"{'='*60}")

    # ── Load image ────────────────────────────────────────────────────
    img_raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    # ── Step 1: Preprocessing ─────────────────────────────────────────
    if preprocess:
        mode = "flip+blob+CLAHE" if preprocess_clahe else "flip+blob (no CLAHE)"
        print(f"\n[1/6] Preprocessing ({mode})...")
        img_proc = preprocess_mammogram(img_raw,
                                        apply_clahe=preprocess_clahe,
                                        clahe_clip=clahe_clip)
        print(f"      {img_raw.shape[1]}x{img_raw.shape[0]}"
              f" -> {img_proc.shape[1]}x{img_proc.shape[0]}")
    else:
        print("\n[1/6] Preprocessing SKIPPED (already preprocessed)")
        img_proc = img_raw

    # Save for reference (but NOT used as input to segmentation)
    cv2.imwrite(str(output_dir / f"{stem}_preprocessed.png"), img_proc)

    # ── Step 2: Segmentation — [FIX-4] direct array, no disk roundtrip ──
    print(f"\n[2/6] Segmentation (U-Net 256×256 sliding window, TTA={seg_tta})...")
    seg_model, seg_device = load_seg_model(seg_ckpt_path, device=device)

    seg_result: SegmentationResult = run_segmentation_array(
        img_uint8    = img_proc,          # ← numpy array directly, no imread
        model        = seg_model,
        device       = seg_device,
        threshold    = seg_threshold,
        tta          = seg_tta,
        min_area     = min_area,
        cluster_dist = cluster_dist,
        cluster_pad  = cluster_pad,
    )

    print(f"      Raw MC detections : {len(seg_result.raw_bboxes)}")
    print(f"      Lesion clusters   : {len(seg_result.bboxes)}")
    print(f"      Inference time    : {seg_result.inference_ms:.0f} ms")

    # Save binary mask + seg overlay
    cv2.imwrite(str(output_dir / f"{stem}_seg_mask.png"), seg_result.binary_mask)

    if save_seg_overlay:
        hm  = cv2.applyColorMap(
            (seg_result.prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        ov  = cv2.addWeighted(
            cv2.cvtColor(img_proc, cv2.COLOR_GRAY2BGR), 0.6, hm, 0.4, 0
        )
        for bb in seg_result.bboxes:
            cv2.rectangle(ov, (bb.x1, bb.y1), (bb.x2, bb.y2), (0, 255, 0), 2)
            cv2.putText(ov, f"C{bb.region_id}",
                        (bb.x1 + 3, bb.y1 + 16),
                        FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        ov_path = output_dir / f"{stem}_seg_overlay.png"
        cv2.imwrite(str(ov_path), ov)
        print(f"      Seg overlay -> {ov_path.name}")

    if not seg_result.bboxes:
        print(f"\n  [FALLBACK] No MC clusters — trying FEBDS enhancement (medical-image-std)...")
        clf = clf_preloaded or MammoClassifier(cls_ckpt_path, device=device, tta=cls_tta)
        # Use stricter threshold on fallback path — FEBDS crops are noisier than U-Net's.
        # Save & restore so the shared preloaded classifier isn't mutated for subsequent calls.
        _orig_thr = clf.threshold
        clf.threshold = fallback_threshold
        print(f"  [FALLBACK] Using stricter threshold={fallback_threshold} (normal={_orig_thr:.2f})")

        # FEBDS "dog": enhance MC candidates then extract bboxes
        _img_f  = _InMemImg(array=torch.from_numpy(img_proc.astype(np.float32) / 255.0))
        _out_f  = _img_f.clone()
        _FebdsAlg(method="dog", device="cpu")(_img_f, _out_f)
        febds_mask = (_out_f.pixel_data.numpy() * 255).astype(np.uint8)

        # Build eroded breast interior mask to reject border-edge artifacts
        H_p, W_p = img_proc.shape
        breast_mask = (img_proc > 5).astype(np.uint8) * 255
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        breast_interior = cv2.erode(breast_mask, erode_k, iterations=2)
        febds_mask = cv2.bitwise_and(febds_mask, febds_mask, mask=breast_interior)

        n_f, lbl_f, st_f, _ = cv2.connectedComponentsWithStats(febds_mask, connectivity=8)
        febds_cands = []  # (area, x1, y1, x2, y2)
        margin = 20
        for i in range(1, n_f):
            a = int(st_f[i, cv2.CC_STAT_AREA])
            if not (30 < a < 8000):
                continue
            x = int(st_f[i, cv2.CC_STAT_LEFT]); y = int(st_f[i, cv2.CC_STAT_TOP])
            w = int(st_f[i, cv2.CC_STAT_WIDTH]); h = int(st_f[i, cv2.CC_STAT_HEIGHT])
            # reject components touching image edge
            if x <= margin or y <= margin or (x + w) >= W_p - margin or (y + h) >= H_p - margin:
                continue
            febds_cands.append((a, x, y, x + w, y + h))

        # Sort by area descending -> take the strongest candidates, not just top-of-image
        febds_cands.sort(key=lambda c: -c[0])
        febds_bboxes = [(c[1], c[2], c[3], c[4]) for c in febds_cands]

        print(f"  [FALLBACK] FEBDS found {len(febds_bboxes)} candidate region(s) after interior+edge filtering")

        # Grad-CAM engine for fallback regions
        fb_gradcam_engine: Optional[GradCAM] = None
        if gradcam:
            _grad_model = clf._models[0] if clf._ensemble else clf.model
            fb_gradcam_engine = GradCAM(_grad_model, _grad_model.net.features[5])

        _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if save_crops:
            (output_dir / "crops").mkdir(exist_ok=True)

        if febds_bboxes:
            febds_bboxes = febds_bboxes[:8]   # limit to avoid over-detection
            H_p, W_p = img_proc.shape
            fallback_regions = []
            for fi, (fx1, fy1, fx2, fy2) in enumerate(febds_bboxes):
                pad = 20
                cx1 = int(max(0, fx1 - pad));  cy1 = int(max(0, fy1 - pad))
                cx2 = int(min(W_p, fx2 + pad));cy2 = int(min(H_p, fy2 + pad))
                fcrop = cv2.resize(img_proc[cy1:cy2, cx1:cx2], (224, 224),
                                   interpolation=cv2.INTER_AREA)
                prob, lbl, per_tta = clf.predict_array(fcrop, return_all_probs=True)
                label_str   = "MALIGNANT" if lbl == 1 else "BENIGN"
                uncertainty = float(np.std(per_tta))

                cam_overlay = None
                if fb_gradcam_engine is not None:
                    img_f = fcrop.astype(np.float32) / 255.0
                    t_in = torch.from_numpy(
                        (np.stack([img_f]*3) - _MEAN[:,None,None]) / _STD[:,None,None]
                    ).unsqueeze(0).float().to(torch.device(device))
                    cam = fb_gradcam_engine.generate(t_in, (224, 224))
                    cam_overlay = _make_gradcam_overlay(fcrop, cam, gradcam_alpha)

                if save_crops:
                    crop_dir = output_dir / "crops"
                    cv2.imwrite(str(crop_dir / f"{stem}_R{fi:02d}_crop.png"), fcrop)
                    if cam_overlay is not None:
                        cv2.imwrite(
                            str(crop_dir / f"{stem}_R{fi:02d}_gradcam.png"),
                            cam_overlay
                        )

                print(f"      FEBDS R{fi:02d}  -> {label_str}  P={prob:.3f} ±{uncertainty:.3f}")
                fallback_regions.append(RegionResult(
                    region_id=fi, bbox=[cx1, cy1, cx2, cy2],
                    label=label_str, prob=prob, uncertainty=uncertainty,
                    cam_overlay=cam_overlay,
                ))
        else:
            print("  [FALLBACK] FEBDS also found nothing — using center crop")
            H_p, W_p = img_proc.shape
            half = min(W_p, H_p) // 3
            cx, cy = W_p // 2, H_p // 2
            x1 = int(max(0, cx - half));  y1 = int(max(0, cy - half))
            x2 = int(min(W_p, cx + half));y2 = int(min(H_p, cy + half))
            global_crop = cv2.resize(img_proc[y1:y2, x1:x2], (224, 224),
                                     interpolation=cv2.INTER_AREA)
            prob, lbl, per_tta = clf.predict_array(global_crop, return_all_probs=True)
            label_str   = "MALIGNANT" if lbl == 1 else "BENIGN"
            uncertainty = float(np.std(per_tta))

            cam_overlay = None
            if fb_gradcam_engine is not None:
                img_f = global_crop.astype(np.float32) / 255.0
                t_in = torch.from_numpy(
                    (np.stack([img_f]*3) - _MEAN[:,None,None]) / _STD[:,None,None]
                ).unsqueeze(0).float().to(torch.device(device))
                cam = fb_gradcam_engine.generate(t_in, (224, 224))
                cam_overlay = _make_gradcam_overlay(global_crop, cam, gradcam_alpha)

            if save_crops:
                crop_dir = output_dir / "crops"
                cv2.imwrite(str(crop_dir / f"{stem}_R00_crop.png"), global_crop)
                if cam_overlay is not None:
                    cv2.imwrite(str(crop_dir / f"{stem}_R00_gradcam.png"), cam_overlay)

            print(f"  [FALLBACK] Center crop -> {label_str}  P={prob:.3f} ±{uncertainty:.3f}")
            fallback_regions = [RegionResult(
                region_id=0, bbox=[x1, y1, x2, y2],
                label=label_str, prob=prob, uncertainty=uncertainty,
                cam_overlay=cam_overlay,
            )]

        if fb_gradcam_engine:
            fb_gradcam_engine.remove()

        # Annotated image (overlay with bboxes + labels)
        canvas = _annotate_image(
            cv2.cvtColor(img_proc, cv2.COLOR_GRAY2BGR),
            fallback_regions, show_prob=True, show_gradcam=False
        )
        canvas = _build_legend(canvas)
        n_mal = sum(1 for r in fallback_regions if r.label == "MALIGNANT")
        n_ben = sum(1 for r in fallback_regions if r.label == "BENIGN")
        cv2.putText(canvas, f"[FEBDS] Clusters:{len(fallback_regions)} MAL={n_mal} BEN={n_ben}",
                    (10, 30), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
        ann_path = output_dir / f"{stem}_annotated.png"
        cv2.imwrite(str(ann_path), canvas)
        print(f"      Annotated -> {ann_path.name}")

        _save_panel(img_proc, seg_result, fallback_regions, stem, output_dir)
        clf.threshold = _orig_thr  # restore to not leak fallback threshold to later calls
        return _save_report(image_path, stem, output_dir,
                            seg_result, fallback_regions, t_total)

    # ── Step 3: Classifier ────────────────────────────────────────────
    # cls_ckpt_path peut être un chemin unique OU une liste pour l'ensemble
    print(f"\n[3/6] Loading classifier (TTA-{cls_tta})...")
    clf = clf_preloaded or MammoClassifier(cls_ckpt_path, device=device, tta=cls_tta)
    if cls_threshold is not None:
        clf.threshold = cls_threshold
    print(f"      Threshold : {clf.threshold:.2f}")

    # ── Step 4: Grad-CAM ──────────────────────────────────────────────
    gradcam_engine: Optional[GradCAM] = None
    if gradcam:
        print(f"\n[4/6] Grad-CAM (net.features[5])...")
        _grad_model = clf._models[0] if clf._ensemble else clf.model
        gradcam_engine = GradCAM(_grad_model, _grad_model.net.features[5])

    # ── Step 5: Classify each cluster ────────────────────────────────
    print(f"\n[5/6] Classifying {len(seg_result.bboxes)} cluster(s)...")

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    regions: List[RegionResult] = []
    if save_crops:
        (output_dir / "crops").mkdir(exist_ok=True)

    for bb in seg_result.bboxes:
        crop                = _crop_bbox(img_proc, bb, pad=crop_pad)
        prob, lbl, per_tta  = clf.predict_array(crop, return_all_probs=True)
        label_str           = "MALIGNANT" if lbl == 1 else "BENIGN"
        uncertainty         = float(np.std(per_tta))

        cam_overlay = None
        if gradcam_engine is not None:
            img_f  = crop.astype(np.float32) / 255.0
            t_in   = torch.from_numpy(
                (np.stack([img_f]*3) - _MEAN[:,None,None]) / _STD[:,None,None]
            ).unsqueeze(0).float().to(torch.device(device))
            cam         = gradcam_engine.generate(t_in, (224, 224))
            cam_overlay = _make_gradcam_overlay(crop, cam, gradcam_alpha)

        if save_crops:
            crop_dir = output_dir / "crops"
            cv2.imwrite(str(crop_dir / f"{stem}_R{bb.region_id:02d}_crop.png"),
                        crop)
            if cam_overlay is not None:
                cv2.imwrite(
                    str(crop_dir / f"{stem}_R{bb.region_id:02d}_gradcam.png"),
                    cam_overlay
                )

        tag = "RED" if label_str == "MALIGNANT" else "YEL"
        print(f"      R{bb.region_id:02d}  [{bb.x1},{bb.y1}->{bb.x2},{bb.y2}]"
              f"  {bb.width}×{bb.height}px"
              f"  -> {label_str}  P={prob:.3f} ±{uncertainty:.3f}  [{tag}]")

        regions.append(RegionResult(
            region_id   = bb.region_id,
            bbox        = bb.as_list,
            label       = label_str,
            prob        = prob,
            uncertainty = uncertainty,
            cam_overlay = cam_overlay,
        ))

    if gradcam_engine:
        gradcam_engine.remove()

    # ── Step 6: Annotate ─────────────────────────────────────────────
    print(f"\n[6/6] Annotating final image...")
    canvas = _annotate_image(
        cv2.cvtColor(img_proc, cv2.COLOR_GRAY2BGR),
        regions, show_prob=True, show_gradcam=False
    )
    canvas = _build_legend(canvas)
    n_mal  = sum(1 for r in regions if r.label == "MALIGNANT")
    n_ben  = sum(1 for r in regions if r.label == "BENIGN")
    cv2.putText(canvas, f"Clusters:{len(regions)} MAL={n_mal} BEN={n_ben}",
                (10, 30), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
    ann_path = output_dir / f"{stem}_annotated.png"
    cv2.imwrite(str(ann_path), canvas)
    print(f"      Annotated -> {ann_path.name}")

    _save_panel(img_proc, seg_result, regions, stem, output_dir)
    report  = _save_report(image_path, stem, output_dir,
                           seg_result, regions, t_total)
    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.1f}s — "
          f"{len(regions)} cluster(s)  MAL={n_mal}  BEN={n_ben}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*60}\n")
    return report


# ─────────────────────────────────────────────────────────────────────
# SUMMARY PANEL
# ─────────────────────────────────────────────────────────────────────
def _save_panel(img_proc, seg_result, regions, stem, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    n  = min(len(regions), 4)
    fig = plt.figure(
        figsize=(18, 12 if n > 0 else 6), facecolor='#1a1a2e'
    )

    gs = fig.add_gridspec(1, 3, left=0.02, right=0.98,
                          top=0.97, bottom=0.52 if n > 0 else 0.02,
                          wspace=0.04)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img_proc, cmap='gray', vmin=0, vmax=255)
    ax0.set_title('Preprocessed', color='white', fontsize=11); ax0.axis('off')

    ax1 = fig.add_subplot(gs[1])
    im  = ax1.imshow(seg_result.prob_map, cmap='hot', vmin=0, vmax=1)
    ax1.set_title(
        f'U-Net Probability Map\n'
        f'{len(seg_result.raw_bboxes)} raw -> {len(seg_result.bboxes)} clusters',
        color='white', fontsize=11
    )
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, fraction=0.03, pad=0.01)
    for bb in seg_result.bboxes:
        ax1.add_patch(mpatches.Rectangle(
            (bb.x1, bb.y1), bb.width, bb.height,
            lw=1.5, edgecolor='lime', facecolor='none'
        ))

    ax2 = fig.add_subplot(gs[2])
    canvas = _annotate_image(
        cv2.cvtColor(img_proc, cv2.COLOR_GRAY2BGR),
        regions, show_prob=True, show_gradcam=False
    )
    n_mal = sum(1 for r in regions if r.label == 'MALIGNANT')
    assess = 'SUSPICIOUS' if n_mal > 0 else 'BENIGN' if regions else 'NORMAL'
    ax2.imshow(cv2.cvtColor(_build_legend(canvas), cv2.COLOR_BGR2RGB))
    ax2.set_title(
        f'Result — {assess}  (RED=Malignant  YELLOW=Benign)',
        color='#FF4444' if n_mal > 0 else '#FFD700',
        fontsize=11
    )
    ax2.axis('off')

    if n > 0:
        gs2 = fig.add_gridspec(1, n * 2, left=0.02, right=0.98,
                               top=0.48, bottom=0.02, wspace=0.06)
        crop_dir = output_dir / "crops"
        for ci, r in enumerate(regions[:4]):
            color_hex = '#FF4444' if r.label == 'MALIGNANT' else '#FFD700'
            ax_c = fig.add_subplot(gs2[ci * 2])
            cp   = crop_dir / f"{stem}_R{r.region_id:02d}_crop.png"
            if cp.exists():
                ax_c.imshow(cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE),
                            cmap='gray')
            ax_c.set_title(f'R{r.region_id} {r.label}\nP={r.prob:.3f}',
                           color=color_hex, fontsize=9)
            ax_c.axis('off')
            for sp in ax_c.spines.values():
                sp.set_visible(True); sp.set_color(color_hex); sp.set_linewidth(3)

            ax_g = fig.add_subplot(gs2[ci * 2 + 1])
            gp   = crop_dir / f"{stem}_R{r.region_id:02d}_gradcam.png"
            if gp.exists():
                ax_g.imshow(cv2.cvtColor(cv2.imread(str(gp)),
                                         cv2.COLOR_BGR2RGB))
                ax_g.set_title('Grad-CAM', color='white', fontsize=9)
            elif r.cam_overlay is not None:
                ax_g.imshow(cv2.cvtColor(r.cam_overlay, cv2.COLOR_BGR2RGB))
                ax_g.set_title('Grad-CAM', color='white', fontsize=9)
            ax_g.axis('off')

    plt.suptitle(f'mammo-cad — {stem}\n'
                 f'U-Net (AUC=0.9642) + EfficientNet-B3 (AUC=0.7812)',
                 color='white', fontsize=13, y=0.999)
    out = output_dir / f"{stem}_panel.png"
    plt.savefig(str(out), dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"      Panel -> {out.name}")


# ─────────────────────────────────────────────────────────────────────
# JSON REPORT
# ─────────────────────────────────────────────────────────────────────
def _save_report(image_path, stem, output_dir, seg_result, regions, t_start):
    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
    n_mal = sum(1 for r in regions if r.label == "MALIGNANT")
    n_ben = sum(1 for r in regions if r.label == "BENIGN")
    report = {
        "image"              : str(image_path),
        "model_auc_cls"      : 0.7812,
        "model_auc_seg"      : 0.9642,
        "regions_found"      : len(regions),
        "malignant"          : n_mal,
        "benign"             : n_ben,
        "raw_mc_detections"  : len(seg_result.raw_bboxes),
        "seg_threshold"      : seg_result.threshold,
        "tta_seg"            : seg_result.tta_used,
        "inference_ms"       : elapsed_ms,
        "overall_assessment" : (
            "SUSPICIOUS — malignant cluster(s) detected" if n_mal > 0
            else "BENIGN — no malignant clusters" if n_ben > 0
            else "NORMAL — no microcalcification clusters detected"
        ),
        "results": [
            {
                "region_id"  : r.region_id,
                "location"   : r.bbox,
                "label"      : r.label,
                "probability": round(r.prob, 4),
                "uncertainty": round(r.uncertainty, 4),
                "confidence" : f"{r.prob*100:.1f}%",
            }
            for r in regions
        ],
    }
    rp = output_dir / f"{stem}_report.json"
    rp.write_text(json.dumps(report, indent=2))
    print(f"      Report -> {rp.name}")
    return report


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="mammo-cad pipeline v3")
    p.add_argument("--input",       required=True)
    p.add_argument("--seg_ckpt",    required=True)
    p.add_argument("--cls_ckpt",    required=True)
    p.add_argument("--output_dir",  default="outputs/predictions")
    p.add_argument("--preprocess",  action="store_true",
                   help="Only for raw images. Do NOT use with data/processed/.")
    p.add_argument("--seg_threshold", type=float, default=0.45)
    p.add_argument("--seg_tta",       action="store_true")
    p.add_argument("--min_area",      type=int,   default=50)
    p.add_argument("--cluster_dist",  type=int,   default=60)
    p.add_argument("--cluster_pad",   type=int,   default=32)
    p.add_argument("--cls_threshold", type=float, default=None)
    p.add_argument("--cls_tta",       type=int,   default=16, choices=[8, 16])
    p.add_argument("--crop_pad",      type=int,   default=32)
    p.add_argument("--no_gradcam",    action="store_true")
    p.add_argument("--gradcam_alpha", type=float, default=0.50)
    p.add_argument("--no_seg_overlay", action="store_true")
    p.add_argument("--no_crops",       action="store_true")
    p.add_argument("--device",         default="auto")
    return p.parse_args()


def main():
    args = _parse_args()
    run_pipeline(
        image_path       = args.input,
        seg_ckpt_path    = args.seg_ckpt,
        cls_ckpt_path    = args.cls_ckpt,
        output_dir       = args.output_dir,
        preprocess       = args.preprocess,
        seg_threshold    = args.seg_threshold,
        seg_tta          = args.seg_tta,
        min_area         = args.min_area,
        cluster_dist     = args.cluster_dist,
        cluster_pad      = args.cluster_pad,
        cls_threshold    = args.cls_threshold,
        cls_tta          = args.cls_tta,
        gradcam          = not args.no_gradcam,
        gradcam_alpha    = args.gradcam_alpha,
        save_seg_overlay = not args.no_seg_overlay,
        save_crops       = not args.no_crops,
        device           = args.device,
    )


if __name__ == "__main__":
    main()