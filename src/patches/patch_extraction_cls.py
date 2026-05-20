"""
patch_extraction_cls.py — Organize CBIS-DDSM crops for classification
======================================================================

Reads:  data/interim/cbis_ddsm/cbis_ddsm_cases.csv
        (split column already assigned by build_cases.py: train/val/test)
Source: data/processed/cbis_ddsm/Crops/  (CLAHE-enhanced, from preprocessing.py)

Organizes into:
  data/patches/classification/train/benign/
  data/patches/classification/train/malignant/
  data/patches/classification/val/benign/
  data/patches/classification/val/malignant/
  data/patches/classification/test/benign/
  data/patches/classification/test/malignant/

Run:
  python src/patches/patch_extraction_cls.py
"""

import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd

CSV_PATH  = Path("data/interim/cbis_ddsm/cbis_ddsm_cases.csv")
CROPS_DIR = Path("data/processed/cbis_ddsm/Crops")
OUT_DIR   = Path("data/patches/classification")
IMG_SIZE  = 224
LABEL_MAP = {0: "benign", 1: "malignant"}


def main():
    df = pd.read_csv(str(CSV_PATH))

    # Keep only rows with a valid crop
    df = df[df["crop_png"].notna() & (df["crop_png"] != "")].copy()
    print(f"Valid cases: {len(df)}")

    # Verify split column exists
    if "split" not in df.columns:
        print("ERROR: 'split' column not found in CSV.")
        print("Re-run build_cases.py to regenerate the CSV with split column.")
        return

    # Create all output directories
    for split in ["train", "val", "test"]:
        for label in ["benign", "malignant"]:
            (OUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

    counts = {s: {l: 0 for l in ["benign", "malignant"]}
              for s in ["train", "val", "test"]}
    failed  = 0
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Organizing patches"):

        split_name = str(row["split"]).strip()
        if split_name not in ("train", "val", "test"):
            failed += 1
            continue

        label_name = LABEL_MAP.get(int(row["label"]))
        if label_name is None:
            failed += 1
            continue

        crop_file = str(row["crop_png"]).strip()
        src       = CROPS_DIR / crop_file

        if not src.exists():
            missing += 1
            if missing <= 3:
                print(f"  MISSING: {src}")
            continue

        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            failed += 1
            continue

        # Ensure 224x224 (should already be, but resize just in case)
        if img.shape != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_AREA)

        out_path = OUT_DIR / split_name / label_name / crop_file
        cv2.imwrite(str(out_path), img)
        counts[split_name][label_name] += 1

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  CLASSIFICATION PATCHES COMPLETE")
    print(f"{'='*50}")
    grand_total = 0
    for split in ["train", "val", "test"]:
        b = counts[split]["benign"]
        m = counts[split]["malignant"]
        t = b + m
        grand_total += t
        ratio = b / max(m, 1)
        print(f"  {split:5s}: benign={b:4d}  malignant={m:4d}  "
              f"total={t:4d}  ratio={ratio:.2f}")
    print(f"  Grand total : {grand_total}")
    if missing:
        print(f"  Missing src : {missing}")
    if failed:
        print(f"  Failed      : {failed}")
    print(f"{'='*50}")
    print("\nNext steps:")
    print("  python src/training/train_classifier.py --stage 1")
    print("  python src/training/train_classifier.py --stage 2 --eval_test")


if __name__ == "__main__":
    main()