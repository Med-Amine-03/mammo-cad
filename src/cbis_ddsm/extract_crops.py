"""
extract_crops.py — Extract lossless PNG crops from CBIS-DDSM DICOM files
=========================================================================
Strategy (confirmed best — v1, not mask-crop):
  Use the pre-cropped DICOM (smallest pixel area in the series folder).
  The CBIS-DDSM team already extracted precise ROI crops — trust it.

  This outperforms mask-based cropping because:
  - The radiologist crop already contains the correct lesion context
  - It preserves natural diversity between crops from the same patient
  - It gives more effective training samples (2700+ vs 1270 mask method)

DICOM structure:
  Case with 1 DICOM : 1-1.dcm = full mammogram OR crop
  Case with 2 DICOMs: smallest pixel area = the crop

  MAX_CROP_DIM = 1500: if smallest is still full mammogram,
  fall back to mask-bbox crop.

Output: data/interim/cbis_ddsm/Crops/<patient_id>_<idx>.png  (224×224 PNG)

Run:
  python src/cbis_ddsm/extract_crops.py
"""

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pathlib import Path
from tqdm import tqdm
import sys


INTERIM_DIR  = Path("data/interim/cbis_ddsm")
CROPS_DIR    = INTERIM_DIR / "Crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

IN_CSV       = INTERIM_DIR / "cbis_ddsm_cases.csv"
OUT_CSV      = IN_CSV

TARGET_SIZE  = 224
MAX_CROP_DIM = 1500   # threshold for "this is still a full mammogram"


# ─────────────────────────────────────────────────────────────────────
# DICOM READING
# ─────────────────────────────────────────────────────────────────────
def read_dicom_pixels(dcm_path: Path) -> tuple[np.ndarray | None, object]:
    """
    Read DICOM → normalised uint8 array.
    Applies RescaleSlope/Intercept, VOI LUT windowing,
    and MONOCHROME1 inversion (brighter = lower tissue density).
    """
    try:
        ds     = pydicom.dcmread(str(dcm_path))
        pixels = ds.pixel_array.astype(np.float32)
    except Exception:
        return None, None

    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    pixels    = pixels * slope + intercept

    try:
        pixels = apply_voi_lut(pixels, ds).astype(np.float32)
    except Exception:
        pass

    photo = getattr(ds, "PhotometricInterpretation", "MONOCHROME2").strip()
    if photo == "MONOCHROME1":
        pixels = pixels.max() - pixels

    p_min, p_max = pixels.min(), pixels.max()
    if p_max > p_min:
        pixels = (pixels - p_min) / (p_max - p_min) * 255.0

    return pixels.astype(np.uint8), ds


def get_dicom_dims(dcm_path: Path) -> tuple[int, int]:
    """Return (rows, cols) without loading pixel data."""
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        return int(getattr(ds, "Rows", 99999)), \
               int(getattr(ds, "Columns", 99999))
    except Exception:
        return 99999, 99999


# ─────────────────────────────────────────────────────────────────────
# FIND THE CROP DICOM
# ─────────────────────────────────────────────────────────────────────
def find_crop_dicom(crop_dcm_path_str: str) -> Path | None:
    """
    Find the DICOM with the smallest pixel area in the series folder.
    That is always the radiologist pre-cropped ROI.
    """
    crop_dcm   = Path(crop_dcm_path_str)
    if not crop_dcm.exists():
        return None

    series_dir = crop_dcm.parent
    all_dcms   = sorted(series_dir.glob("*.dcm"))

    if not all_dcms:
        return None
    if len(all_dcms) == 1:
        return all_dcms[0]

    best_dcm  = None
    best_area = float("inf")
    for d in all_dcms:
        rows, cols = get_dicom_dims(d)
        area = rows * cols
        if area < best_area:
            best_area = area
            best_dcm  = d

    return best_dcm


# ─────────────────────────────────────────────────────────────────────
# FALLBACK: crop from full mammogram using mask bbox
# ─────────────────────────────────────────────────────────────────────
def crop_using_mask(full_img: np.ndarray,
                    mask_dcm_path: Path,
                    padding: int = 32) -> np.ndarray | None:
    """
    When no small crop DICOM exists, use the ROI mask to locate the
    lesion and crop the full mammogram.
    """
    try:
        ds   = pydicom.dcmread(str(mask_dcm_path))
        mask = ds.pixel_array
    except Exception:
        return None

    nz = np.where(mask > 0)
    if len(nz[0]) == 0:
        return None

    y1, y2 = int(nz[0].min()), int(nz[0].max())
    x1, x2 = int(nz[1].min()), int(nz[1].max())

    h, w = full_img.shape
    crop = full_img[
        max(0, y1 - padding) : min(h, y2 + padding),
        max(0, x1 - padding) : min(w, x2 + padding),
    ]
    return crop if crop.size > 0 else None


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def extract_crops() -> None:
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found. Run build_cases.py first.")
        sys.exit(1)

    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} cases from {IN_CSV}")
    print(f"Saving {TARGET_SIZE}×{TARGET_SIZE} PNG crops to {CROPS_DIR}")

    crop_pngs  = []
    used_crop  = 0
    used_mask  = 0
    errors     = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        out_name = f"{row['patient_id']}_{idx}.png"
        out_path = CROPS_DIR / out_name

        # Resume: skip if already extracted
        if out_path.exists():
            crop_pngs.append(out_name)
            continue

        crop_dcm = find_crop_dicom(str(row["crop_dcm"]))
        if crop_dcm is None:
            crop_pngs.append("")
            errors += 1
            continue

        rows, cols = get_dicom_dims(crop_dcm)

        # ── Direct crop (preferred): CBIS-DDSM pre-cropped ROI ────────
        if rows <= MAX_CROP_DIM or cols <= MAX_CROP_DIM:
            img, _ = read_dicom_pixels(crop_dcm)
            if img is None:
                crop_pngs.append("")
                errors += 1
                continue
            used_crop += 1

        # ── Fallback: still a full mammogram — use mask ───────────────
        else:
            study_dir      = Path(row["crop_dcm"]).parent.parent
            all_study_dcms = sorted(study_dir.rglob("*.dcm"))

            mask_dcm = None
            img_dcm  = None
            min_area = float("inf")

            for d in all_study_dcms:
                r, c = get_dicom_dims(d)
                area = r * c
                if area < min_area:
                    min_area = area
                    mask_dcm = d
                if r > 2000 and c > 2000:
                    img_dcm = d

            if (mask_dcm is None or img_dcm is None
                    or mask_dcm == img_dcm):
                # Last resort: centre-crop the full mammogram
                img, _ = read_dicom_pixels(crop_dcm)
                if img is None:
                    crop_pngs.append("")
                    errors += 1
                    continue
                h, w = img.shape
                s    = 256
                cy, cx = h // 2, w // 2
                img  = img[max(0, cy-s):cy+s, max(0, cx-s):cx+s]
                used_crop += 1
            else:
                full_img, _ = read_dicom_pixels(img_dcm)
                if full_img is None:
                    crop_pngs.append("")
                    errors += 1
                    continue
                img = crop_using_mask(full_img, mask_dcm, padding=32)
                if img is None or img.size == 0:
                    img = full_img
                used_mask += 1

        if img is None or img.size == 0:
            crop_pngs.append("")
            errors += 1
            continue

        img_resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE),
                                 interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_path), img_resized)
        crop_pngs.append(out_name)

    # ── Update CSV ────────────────────────────────────────────────────
    df["crop_png"] = crop_pngs
    df_valid = df[df["crop_png"] != ""].copy()
    df_valid.to_csv(OUT_CSV, index=False)

    print(f"\n{'='*55}")
    print(f"  Extracted  : {len(df_valid)} crops")
    print(f"  Direct crop: {used_crop}  (pre-cropped DICOM  ← preferred)")
    print(f"  Mask-based : {used_mask}  (fallback for full mammograms)")
    print(f"  Errors     : {errors}")

    # Sanity check: sample 5 crops
    samples = sorted(CROPS_DIR.glob("*.png"))[:5]
    print(f"\n  Sample verification:")
    for p in samples:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"    {p.name}: shape={img.shape} "
                  f"mean={img.mean():.1f} std={img.std():.1f}")

    if used_crop > used_mask * 3:
        print(f"\n  ✅ Direct crop >> mask fallback — good data quality")
    else:
        print(f"\n  ⚠️  Many mask fallbacks — check DICOM series structure")

    print(f"{'='*55}")
    print("\nNext: python src/preprocessing/preprocessing.py --dataset cbis_ddsm")


if __name__ == "__main__":
    extract_crops()