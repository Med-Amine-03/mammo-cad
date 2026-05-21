
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

PROCESSED_PNG  = Path("data/processed/inbreast/AllPng")
PROCESSED_MASK = Path("data/processed/inbreast/Masks")
CSV_PATH       = Path("data/interim/inbreast/inbreast_mc.csv")
OUT_DIR        = Path("data/patches/segmentation")
PATCH_SIZE     = 256


def pad_to_multiple(img: np.ndarray, size: int) -> np.ndarray:
    """Pad image so H and W are divisible by size."""
    h, w   = img.shape[:2]
    pad_h  = (size - h % size) % size
    pad_w  = (size - w % size) % size
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                               cv2.BORDER_CONSTANT, value=0)


def extract_patches(img: np.ndarray,
                    mask: np.ndarray,
                    size: int):
    """Extract all size×size patches with step=size (no overlap)."""
    h, w     = img.shape
    patches  = []
    for y in range(0, h - size + 1, size):
        for x in range(0, w - size + 1, size):
            img_p  = img[y:y+size, x:x+size]
            mask_p = mask[y:y+size, x:x+size]
            patches.append((img_p, mask_p, y, x))
    return patches


def save_patches(patches, img_out: Path, mask_out: Path,
                 stem: str) -> tuple:
    """Save patches and return (saved, skipped) counts."""
    saved   = 0
    skipped = 0
    for img_p, mask_p, y, x in patches:
        # reduce_patches: skip if no MC pixels
        if mask_p.max() == 0:
            skipped += 1
            continue
        name = f"{stem}_{y}_{x}.png"
        cv2.imwrite(str(img_out  / name), img_p)
        cv2.imwrite(str(mask_out / name), mask_p)
        saved += 1
    return saved, skipped


def main():
    df = pd.read_csv(str(CSV_PATH))
    print(f"Total MC cases: {len(df)}")
    print(f"  CV pool (all): {(df.split=='all').sum()}")
    print(f"  Test (held)  : {(df.split=='test').sum()}")

    # Create output dirs
    for split in ["all", "test"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "masks").mkdir(parents=True, exist_ok=True)

    total_saved   = {"all": 0, "test": 0}
    total_skipped = {"all": 0, "test": 0}
    failed        = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Extracting patches"):
        png_name  = row["png_name"]
        split     = row["split"]

        png_path  = PROCESSED_PNG  / png_name
        mask_path = PROCESSED_MASK / png_name

        if not png_path.exists() or not mask_path.exists():
            failed.append(png_name)
            continue

        img  = cv2.imread(str(png_path),  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            failed.append(png_name)
            continue

        # Pad to multiple of 256
        img  = pad_to_multiple(img,  PATCH_SIZE)
        mask = pad_to_multiple(mask, PATCH_SIZE)

        # Extract patches
        patches = extract_patches(img, mask, PATCH_SIZE)

        # Save positive patches only
        stem     = Path(png_name).stem
        img_out  = OUT_DIR / split / "images"
        mask_out = OUT_DIR / split / "masks"

        saved, skipped = save_patches(patches, img_out, mask_out, stem)
        total_saved[split]   += saved
        total_skipped[split] += skipped

    # Summary
    print(f"\n{'='*50}")
    print(f"PATCH EXTRACTION COMPLETE")
    print(f"{'='*50}")
    for split in ["all", "test"]:
        print(f"\n  Split '{split}':")
        print(f"    Positive patches saved : {total_saved[split]}")
        print(f"    Negative patches removed: {total_skipped[split]}")
    print(f"\n  Total positive patches: "
          f"{sum(total_saved.values())}")
    if failed:
        print(f"\n  Failed ({len(failed)}): {failed[:5]}")

    # Verify
    print(f"\nVerification:")
    for split in ["all", "test"]:
        imgs  = list((OUT_DIR/split/"images").glob("*.png"))
        masks = list((OUT_DIR/split/"masks").glob("*.png"))
        print(f"  {split}/images: {len(imgs)} patches")
        print(f"  {split}/masks : {len(masks)} patches")

    
    print(f"\nChecking all masks have MC pixels...")
    all_ok = True
    for split in ["all", "test"]:
        for p in (OUT_DIR/split/"masks").glob("*.png"):
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None or m.max() == 0:
                print(f"  WARNING: empty mask found: {p.name}")
                all_ok = False
    if all_ok:
        print(f"  All masks verified ")


if __name__ == "__main__":
    main()