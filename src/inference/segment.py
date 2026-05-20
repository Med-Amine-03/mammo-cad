"""

    python src/inference/segment.py \
        --input  data/processed/inbreast/AllPng/some_image.png \
        --ckpt   checkpoints/unet_best_fold.pth \
        --output outputs/pred_mask.png \
        --viz    outputs/overlay.png
 
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
import torch.nn.functional as F
 
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.unet import UNet
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Public data structures
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class BoundingBox:
    
    region_id : int
    x1        : int
    y1        : int
    x2        : int
    y2        : int
    area      : int               
    centroid  : Tuple[float, float]  
 
    @property
    def width(self)  -> int: return self.x2 - self.x1
    @property
    def height(self) -> int: return self.y2 - self.y1
    @property
    def aspect_ratio(self) -> float:
        
        return self.width / (self.height + 1e-8)
    @property
    def as_list(self)-> List[int]: return [self.x1, self.y1, self.x2, self.y2]
 
 
@dataclass
class SegmentationResult:
    binary_mask  : np.ndarray          
    prob_map     : np.ndarray          
    bboxes       : List[BoundingBox]   
    raw_bboxes   : List[BoundingBox]   
    threshold    : float
    tta_used     : bool
    inference_ms : float
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Model loader (cached singleton for use in pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────
 
_LOADED_MODEL: Optional[Tuple[UNet, torch.device]] = None
 
def load_model(
    ckpt_path : str | Path,
    device    : str | torch.device = "cuda",
    force_reload: bool = False,
) -> Tuple[UNet, torch.device]:
   
    global _LOADED_MODEL
    if _LOADED_MODEL is not None and not force_reload:
        return _LOADED_MODEL
 
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
 
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
 
    model = UNet(in_channels=1, out_channels=1, base_ch=64).to(device)
 
    ckpt = torch.load(ckpt_path, map_location=device)
    
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
 
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    if meta:
        info = "  ".join(f"{k}={v}" for k, v in meta.items())
        print(f"[segment] Loaded checkpoint — {info}")
 
    _LOADED_MODEL = (model, device)
    return _LOADED_MODEL
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Core preprocessing — mirrors training preprocessing.py
# ─────────────────────────────────────────────────────────────────────────────
 
def _load_and_normalise(image_path: str | Path) -> np.ndarray:
    
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = img.astype(np.float32) / 255.0
    return img   # (H, W)
 
 
def _pad_to_multiple(img: np.ndarray, multiple: int = 256) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
  
    H, W = img.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    # reflect padding on top/left, zero on bottom/right  →  same as np.pad default
    pad_top   = 0
    pad_bottom = pad_h
    pad_left  = 0
    pad_right  = pad_w
    padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
    return padded, (pad_top, pad_bottom, pad_left, pad_right)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window inference
# ─────────────────────────────────────────────────────────────────────────────
 
PATCH_SIZE = 256   
 
@torch.no_grad()
def _infer_patches(
    model     : UNet,
    device    : torch.device,
    img_padded: np.ndarray,    
) -> np.ndarray:
    
    pH, pW = img_padded.shape
    assert pH % PATCH_SIZE == 0 and pW % PATCH_SIZE == 0
 
    logit_map = np.zeros((pH, pW), dtype=np.float32)
 
    for row in range(0, pH, PATCH_SIZE):
        for col in range(0, pW, PATCH_SIZE):
            patch = img_padded[row:row+PATCH_SIZE, col:col+PATCH_SIZE]
 
            
            t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
 
            logits = model(t)                 # (1, 1, 256, 256) raw logits
            logit_map[row:row+PATCH_SIZE, col:col+PATCH_SIZE] = (
                logits.squeeze().cpu().numpy()
            )
 
    return logit_map
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TTA — mirrors augmentations used in training (HFlip, VFlip, Rot90)
#
# ─────────────────────────────────────────────────────────────────────────────
 
@torch.no_grad()
def _infer_with_tta(
    model     : UNet,
    device    : torch.device,
    img_padded: np.ndarray,
) -> np.ndarray:
    """
    4-pass TTA: original + hflip + vflip + rot180
    """
   
    augmented = [
        img_padded.copy(),
        np.fliplr(img_padded).copy(),
        np.flipud(img_padded).copy(),
        np.rot90(img_padded, 2).copy(),   
    ]
 
    prob_sum = np.zeros_like(img_padded, dtype=np.float32)
 
    for i, aug_img in enumerate(augmented):
        logit_map = _infer_patches(model, device, aug_img)
        prob = 1.0 / (1.0 + np.exp(-logit_map))   # sigmoid
 
        # Inverse transform to original orientation
        if i == 1:   prob = np.fliplr(prob)
        elif i == 2: prob = np.flipud(prob)
        elif i == 3: prob = np.rot90(prob, -2)
 
        prob_sum += prob
 
    return prob_sum / len(augmented)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Connected components → bounding boxes
# ─────────────────────────────────────────────────────────────────────────────
 
MIN_AREA       = 50     
MAX_ASPECT     = 3.0   
MAX_AREA       = 5000  
 
def _extract_bboxes(
    binary_mask : np.ndarray,   
    min_area    : int   = MIN_AREA,
    max_area    : int   = MAX_AREA,
    max_aspect  : float = MAX_ASPECT,
) -> List[BoundingBox]:
    
    # CC analysis
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
 
    bboxes: List[BoundingBox] = []
    region_id = 0
 
    for label_idx in range(1, n_labels):   # skip background (0)
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if area > max_area:
            continue   
 
        x1 = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y1 = int(stats[label_idx, cv2.CC_STAT_TOP])
        w  = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h  = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
 
        # Aspect ratio filter — wires/streaks are highly elongated
        ar = w / (h + 1e-8)
        if ar > max_aspect or ar < (1.0 / max_aspect):
            continue
 
        cx, cy = centroids[label_idx]
 
        region_id += 1
        bboxes.append(BoundingBox(
            region_id = region_id,
            x1        = x1,
            y1        = y1,
            x2        = x1 + w,
            y2        = y1 + h,
            area      = area,
            centroid  = (float(cx), float(cy)),
        ))
 
    bboxes.sort(key=lambda b: b.area, reverse=True)
    return bboxes
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Cluster grouping — merge individual MC boxes into one box per lesion cluster
# ─────────────────────────────────────────────────────────────────────────────
 
CLUSTER_DIST = 100  
 
def _cluster_bboxes(
    bboxes      : List[BoundingBox],
    cluster_dist: int = CLUSTER_DIST,
    img_w       : int = 0,   
    img_h       : int = 0,   
    pad         : int = 36,  
) -> List[BoundingBox]:
   
    if not bboxes:
        return []
 
    n = len(bboxes)
 
    
    parent = list(range(n))
 
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]   # path compression
            i = parent[i]
        return i
 
    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)
 
    for i in range(n):
        for j in range(i + 1, n):
            cx_i, cy_i = bboxes[i].centroid
            cx_j, cy_j = bboxes[j].centroid
            dist = ((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2) ** 0.5
            if dist <= cluster_dist:
                union(i, j)
 
    
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(bboxes[i])
 
   
    merged: List[BoundingBox] = []
    cluster_id = 0
 
    for members in groups.values():
        x1 = min(b.x1 for b in members) - pad
        y1 = min(b.y1 for b in members) - pad
        x2 = max(b.x2 for b in members) + pad
        y2 = max(b.y2 for b in members) + pad
 
        # Clamp to image bounds
        if img_w > 0: x1, x2 = max(0, x1), min(img_w, x2)
        if img_h > 0: y1, y2 = max(0, y1), min(img_h, y2)
 
        total_area = sum(b.area for b in members)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
 
        cluster_id += 1
        merged.append(BoundingBox(
            region_id = cluster_id,
            x1        = x1,
            y1        = y1,
            x2        = x2,
            y2        = y2,
            area      = total_area,
            centroid  = (cx, cy),
        ))
 
    merged.sort(key=lambda b: b.area, reverse=True)
    return merged
 
 

 
def centroid_detection_metrics(
    pred_bboxes : List[BoundingBox],
    gt_mask     : np.ndarray,            
    radius_px   : float = 10.0,       
) -> dict:
   
    
    gt_bin = (gt_mask > 127).astype(np.uint8)
    n_gt, _, _, gt_cents = cv2.connectedComponentsWithStats(gt_bin, connectivity=8)
    gt_centroids = [(float(gt_cents[i][0]), float(gt_cents[i][1]))
                    for i in range(1, n_gt)]   # skip background
 
    if not gt_centroids and not pred_bboxes:
        return dict(TP=0, FP=0, FN=0, precision=1.0, recall=1.0, f1=1.0)
    if not gt_centroids:
        return dict(TP=0, FP=len(pred_bboxes), FN=0,
                    precision=0.0, recall=1.0, f1=0.0)
    if not pred_bboxes:
        return dict(TP=0, FP=0, FN=len(gt_centroids),
                    precision=1.0, recall=0.0, f1=0.0)
 
    pred_cents = [bb.centroid for bb in pred_bboxes]
 
    matched_gt   = set()
    matched_pred = set()
 
    
    for gi, gc in enumerate(gt_centroids):
        for pi, pc in enumerate(pred_cents):
            dist = ((gc[0]-pc[0])**2 + (gc[1]-pc[1])**2) ** 0.5
            if dist <= radius_px and gi not in matched_gt and pi not in matched_pred:
                matched_gt.add(gi)
                matched_pred.add(pi)
 
    tp  = len(matched_gt)
    fn  = len(gt_centroids) - tp
    fp  = len(pred_cents)   - len(matched_pred)
 
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
 
    return dict(TP=tp, FP=fp, FN=fn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4))
 
 

def run_segmentation_array(
    img_uint8    : np.ndarray,
    model        : UNet,
    device       : torch.device,
    threshold    : float = 0.5,
    tta          : bool  = False,
    min_area     : int   = MIN_AREA,
    max_area     : int   = MAX_AREA,
    max_aspect   : float = MAX_ASPECT,
    cluster_dist : int   = CLUSTER_DIST,
    cluster_pad  : int   = 32,
) -> SegmentationResult:
    """
    Identical to run_segmentation() but accepts a uint8 numpy array directly
    instead of a file path. Used by pipeline.py to avoid the write→read roundtrip
    that could alter image statistics and size.

    Parameters
    ----------
    img_uint8 : np.ndarray
        Grayscale uint8 image array, shape (H, W). Already preprocessed.
    """
    t0 = time.perf_counter()

    img = img_uint8.astype(np.float32) / 255.0
    orig_H, orig_W = img.shape

    img_padded, (pt, pb, pl, pr) = _pad_to_multiple(img, PATCH_SIZE)

    if tta:
        prob_map_padded = _infer_with_tta(model, device, img_padded)
    else:
        logit_map = _infer_patches(model, device, img_padded)
        prob_map_padded = 1.0 / (1.0 + np.exp(-logit_map))

    pH, pW = img_padded.shape
    prob_map = prob_map_padded[
        pt : pH - pb if pb else pH,
        pl : pW - pr if pr else pW,
    ]

    assert prob_map.shape == (orig_H, orig_W), (
        f"Unpad mismatch: {prob_map.shape} vs expected ({orig_H}, {orig_W})"
    )

    binary     = (prob_map >= threshold).astype(np.uint8) * 255
    raw_bboxes = _extract_bboxes(binary, min_area=min_area,
                                 max_area=max_area, max_aspect=max_aspect)

    if cluster_dist > 0:
        bboxes = _cluster_bboxes(raw_bboxes, cluster_dist=cluster_dist,
                                 img_w=orig_W, img_h=orig_H, pad=cluster_pad)
    else:
        bboxes = raw_bboxes

    inference_ms = (time.perf_counter() - t0) * 1000.0

    return SegmentationResult(
        binary_mask  = binary,
        prob_map     = prob_map.astype(np.float32),
        bboxes       = bboxes,
        raw_bboxes   = raw_bboxes,
        threshold    = threshold,
        tta_used     = tta,
        inference_ms = inference_ms,
    )


def run_segmentation(
    image_path   : str | Path,
    model        : UNet,
    device       : torch.device,
    threshold    : float = 0.5,
    tta          : bool  = False,
    min_area     : int   = MIN_AREA,
    max_area     : int   = MAX_AREA,
    max_aspect   : float = MAX_ASPECT,
    cluster_dist : int   = CLUSTER_DIST,  # 0 = disable clustering
    cluster_pad  : int   = 32,
) -> SegmentationResult:
   
    t0 = time.perf_counter()
 
    
    img = _load_and_normalise(image_path)
    orig_H, orig_W = img.shape
 
    
    img_padded, (pt, pb, pl, pr) = _pad_to_multiple(img, PATCH_SIZE)
 

    if tta:
        prob_map_padded = _infer_with_tta(model, device, img_padded)
    else:
        logit_map = _infer_patches(model, device, img_padded)
        prob_map_padded = 1.0 / (1.0 + np.exp(-logit_map))
 
    
    pH, pW = img_padded.shape
    prob_map = prob_map_padded[
        pt : pH - pb if pb else pH,
        pl : pW - pr if pr else pW,
    ]
    
    assert prob_map.shape == (orig_H, orig_W), (
        f"Unpad mismatch: {prob_map.shape} vs expected ({orig_H}, {orig_W})"
    )
 
    
    binary = (prob_map >= threshold).astype(np.uint8) * 255
 
    
    raw_bboxes = _extract_bboxes(binary, min_area=min_area,
                                 max_area=max_area, max_aspect=max_aspect)
 
    
    if cluster_dist > 0:
        bboxes = _cluster_bboxes(raw_bboxes, cluster_dist=cluster_dist,
                                 img_w=orig_W, img_h=orig_H, pad=cluster_pad)
    else:
        bboxes = raw_bboxes  
    inference_ms = (time.perf_counter() - t0) * 1000.0
 
    return SegmentationResult(
        binary_mask  = binary,
        prob_map     = prob_map.astype(np.float32),
        bboxes       = bboxes,
        raw_bboxes   = raw_bboxes,
        threshold    = threshold,
        tta_used     = tta,
        inference_ms = inference_ms,
    )
 
 

def save_overlay(
    image_path  : str | Path,
    result      : SegmentationResult,
    out_path    : str | Path,
    box_color   : Tuple[int,int,int] = (0, 255, 0),   # BGR green
    mask_alpha  : float = 0.35,
) -> None:
  
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Cannot read: {image_path}")
 
    canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
 
    
    prob_uint8 = (result.prob_map * 255).astype(np.uint8)
    heatmap    = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    canvas     = cv2.addWeighted(canvas, 1 - mask_alpha, heatmap, mask_alpha, 0)
 
    
    for bb in result.bboxes:
        cv2.rectangle(canvas, (bb.x1, bb.y1), (bb.x2, bb.y2), box_color, 2)
        label = f"R{bb.region_id} ({bb.area}px)"
        cv2.putText(
            canvas, label,
            (bb.x1, max(bb.y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA
        )
 
    cv2.imwrite(str(out_path), canvas)
    print(f"[segment] Overlay saved → {out_path}")
 
 
def save_binary_mask(result: SegmentationResult, out_path: str | Path) -> None:
    cv2.imwrite(str(out_path), result.binary_mask)
    print(f"[segment] Mask saved    → {out_path}")
 
 
def save_report(
    result     : SegmentationResult,
    image_path : str | Path,
    out_path   : str | Path,
) -> None:
    
    report = {
        "image"          : str(image_path),
        "clusters_found" : len(result.bboxes),
        "raw_mc_count"   : len(result.raw_bboxes),
        "threshold"      : result.threshold,
        "tta_used"       : result.tta_used,
        "inference_ms"   : round(result.inference_ms, 1),
        "clusters"       : [
            {
                "cluster_id"  : bb.region_id,
                "location"    : bb.as_list,
                "total_area_px": bb.area,
                "centroid"    : [round(bb.centroid[0], 1), round(bb.centroid[1], 1)],
                "width"       : bb.width,
                "height"      : bb.height,
            }
            for bb in result.bboxes
        ],
    }
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"[segment] Report saved  → {out_path}")
 
 
 
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mammogram segmentation inference — U-Net sliding window"
    )
    p.add_argument("--input",      required=True,  help="Path to preprocessed mammogram PNG")
    p.add_argument("--ckpt",       required=True,  help="Path to unet_best_fold.pth checkpoint")
    p.add_argument("--output",     default=None,   help="Save binary mask PNG here (optional)")
    p.add_argument("--viz",        default=None,   help="Save overlay PNG here (optional)")
    p.add_argument("--report",     default=None,   help="Save JSON report here (optional)")
    p.add_argument("--threshold",  type=float, default=0.45,
                   help="Binarisation threshold (default 0.45 — optimal from sweep)")
    p.add_argument("--tta",        action="store_true",
                   help="Enable TTA (4× slower, +0.004 IoU — not recommended for full mammograms)")
    p.add_argument("--min-area",   type=int,   default=MIN_AREA,
                   help=f"Min CC area px (default {MIN_AREA})")
    p.add_argument("--max-area",   type=int,   default=MAX_AREA,
                   help=f"Max CC area px (default {MAX_AREA})")
    p.add_argument("--max-aspect",    type=float, default=MAX_ASPECT,
                   help=f"Max bbox aspect ratio — wire filter (default {MAX_ASPECT})")
    p.add_argument("--cluster-dist",  type=int,   default=CLUSTER_DIST,
                   help=f"Cluster grouping distance px (default {CLUSTER_DIST}, 0=disable)")
    p.add_argument("--cluster-pad",   type=int,   default=32,
                   help="Padding around each cluster bbox (default 32)")
    p.add_argument("--device",        default="cuda", help="cuda or cpu")
    return p.parse_args()
 
 
def main() -> None:
    args = _parse_args()
 
    print(f"[segment] Loading model from {args.ckpt}")
    model, device = load_model(args.ckpt, device=args.device)
 
    print(f"[segment] Running inference on {args.input}")
    result = run_segmentation(
        image_path   = args.input,
        model        = model,
        device       = device,
        threshold    = args.threshold,
        tta          = args.tta,
        min_area     = args.min_area,
        max_area     = args.max_area,
        max_aspect   = args.max_aspect,
        cluster_dist = args.cluster_dist,
        cluster_pad  = args.cluster_pad,
    )
 
    sep = '-' * 50
    print(f"\n{sep}")
    print(f"  Raw MC detections : {len(result.raw_bboxes)}")
    print(f"  Clusters found    : {len(result.bboxes)}")
    print(f"  TTA               : {result.tta_used}")
    print(f"  Threshold         : {result.threshold}")
    print(f"  Inference         : {result.inference_ms:.1f} ms")
    if result.bboxes:
        print(f"\n  Cluster bboxes (classifier input):")
        for bb in result.bboxes:
            print(f"    C{bb.region_id:02d}  [{bb.x1},{bb.y1} -> {bb.x2},{bb.y2}]"
                  f"  {bb.width}x{bb.height}px  total_area={bb.area}px"
                  f"  centroid=({bb.centroid[0]:.0f},{bb.centroid[1]:.0f})")
    print(f"{sep}\n")
 
    if args.output:
        save_binary_mask(result, args.output)
    if args.viz:
        save_overlay(args.input, result, args.viz)
    if args.report:
        save_report(result, args.input, args.report)
 
 
if __name__ == "__main__":
    main()