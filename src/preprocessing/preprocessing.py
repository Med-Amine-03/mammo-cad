
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd



def clahe(img: np.ndarray) -> np.ndarray:
    
    img_u8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cl.apply(img_u8)


def crop_background(img: np.ndarray,
                    mask: np.ndarray = None,
                    tol: int = 0):
    
    binary = img > tol
    rows   = binary.any(axis=1)
    cols   = binary.any(axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    img_c  = img[r0:r1+1, c0:c1+1]
    msk_c  = mask[r0:r1+1, c0:c1+1] if mask is not None else None
    return img_c, msk_c


def keep_largest_blob(img: np.ndarray,
                      binary: np.ndarray) -> np.ndarray:
    
    binary_u8 = (binary * 255).astype(np.uint8)
    kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
    closed     = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, kernel)
    dilated    = cv2.morphologyEx(closed,    cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return img

    largest    = max(contours, key=cv2.contourArea)
    blob_mask  = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(blob_mask, [largest], -1, 1, thickness=-1)

    result        = img.copy()
    result[blob_mask == 0] = 0
    return result


def roi_center(mask: np.ndarray):
    """Find center of ROI (white pixels) in mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [mask.shape[1] // 2, mask.shape[0] // 2]
    return [int(xs.mean()), int(ys.mean())]


def find_border(binary: np.ndarray, axis_slice, from_center):
    """Find mammogram border distance from center."""
    dists = []
    for i in range(binary.shape[0 if axis_slice == 'row' else 1]):
        if axis_slice == 'row':
            line = binary[i, from_center:]
        else:
            line = binary[from_center:, i]
        zeros = np.where(line == 0)[0]
        dists.append(zeros[0] if len(zeros) > 0 else 0)
    return max(dists) if dists else 0


def remove_artifacts(img: np.ndarray,
                     mask: np.ndarray,
                     binary: np.ndarray,
                     cx: int, cy: int):
    h, w = img.shape

    # RIGHT side
    dist_right = []
    for i in range(h):
        line  = binary[i, cx:]
        zeros = np.where(line == 0)[0]
        dist_right.append(zeros[0] if len(zeros) > 0 else 0)
    cut_right = round((w - cx - max(dist_right)) * 0.9)
    if cut_right > 0:
        img[:, w - cut_right:]    = 0
        binary[:, w - cut_right:] = 0

    
    dist_down = []
    for j in range(w):
        line  = binary[cy:, j]
        zeros = np.where(line == 0)[0]
        dist_down.append(zeros[0] if len(zeros) > 0 else 0)
    cut_down = round((h - cy - max(dist_down)) * 0.9)
    if cut_down > 0:
        img[h - cut_down:, :]    = 0
        binary[h - cut_down:, :] = 0

    
    img_f    = np.flipud(img)
    binary_f = np.flipud(binary)
    mask_f   = np.flipud(mask) if mask is not None else None
    cy_f     = h - cy - 1

    dist_up = []
    for j in range(w):
        line  = binary_f[cy_f:, j]
        zeros = np.where(line == 0)[0]
        dist_up.append(zeros[0] if len(zeros) > 0 else 0)
    cut_up = round((h - cy_f - max(dist_up)) * 0.9)
    if cut_up > 0:
        img_f[h - cut_up:, :]    = 0
        binary_f[h - cut_up:, :] = 0
    img    = np.flipud(img_f)
    binary = np.flipud(binary_f)

    
    left_strip = img[:, :max(1, round(w * 0.001))]
    if (len(np.where(left_strip > 0.9)[0]) > 0.1 * left_strip.size or
            len(np.where(left_strip < 0.1)[0]) > 0.1 * left_strip.size):
        cut = round(w * 0.01)
        img    = img[:, cut:]
        binary = binary[:, cut:]
        if mask is not None:
            mask = mask[:, cut:]

    return img, mask, binary


# ─────────────────────────────────────────────
# INBREAST PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_inbreast(df: pd.DataFrame,
                        png_dir: Path,
                        mask_dir: Path,
                        out_png_dir: Path,
                        out_mask_dir: Path):
    out_png_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Preprocessing INbreast"):
        png_name  = row["png_name"]
        png_path  = png_dir  / png_name
        mask_path = mask_dir / png_name

        if not png_path.exists() or not mask_path.exists():
            failed.append(png_name)
            continue

        img  = cv2.imread(str(png_path),  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        
        img  = img.astype(np.float32) / 255.0

       
        left_sum  = img[:, :img.shape[1]//2].sum()
        right_sum = img[:, img.shape[1]//2:].sum()
        if right_sum > left_sum:
            img  = np.fliplr(img)
            mask = np.fliplr(mask)

        
        cx, cy = roi_center(mask)

        _, binary = cv2.threshold(img, 0.01, 1, cv2.THRESH_BINARY)

        img, mask, binary = remove_artifacts(img, mask, binary, cx, cy)

        
        _, binary2 = cv2.threshold(img, 0.01, 1, cv2.THRESH_BINARY)
        img = keep_largest_blob(img, binary2)

        img, mask = crop_background(img, mask, tol=0)

        
        img = clahe(img)

        cv2.imwrite(str(out_png_dir  / png_name), img)
        if mask is not None:
            cv2.imwrite(str(out_mask_dir / png_name), mask)

    print(f"  Done. Failed: {len(failed)}")
    if failed:
        print(f"  Failed files: {failed[:5]}")


# ─────────────────────────────────────────────
# CBIS-DDSM PREPROCESSING (CLAHE only)
# ─────────────────────────────────────────────

def preprocess_cbis(crops_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(crops_dir.glob("*.png"))
    print(f"Found {len(files)} CBIS-DDSM crops")

    for p in tqdm(files, desc="Preprocessing CBIS-DDSM"):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = clahe(img)
        cv2.imwrite(str(out_dir / p.name), img)

    print(f"  Done: {len(files)} processed")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["inbreast", "cbis_ddsm"],
                        required=True)
    args = parser.parse_args()

    if args.dataset == "inbreast":
        df = pd.read_csv("data/interim/inbreast/inbreast_mc.csv")
        print(f"Processing {len(df)} INbreast MC cases...")
        preprocess_inbreast(
            df         = df,
            png_dir    = Path("data/interim/inbreast/AllPng"),
            mask_dir   = Path("data/interim/inbreast/Masks"),
            out_png_dir  = Path("data/processed/inbreast/AllPng"),
            out_mask_dir = Path("data/processed/inbreast/Masks"),
        )

    elif args.dataset == "cbis_ddsm":
        preprocess_cbis(
            crops_dir = Path("data/interim/cbis_ddsm/Crops"),
            out_dir   = Path("data/processed/cbis_ddsm/Crops"),
        )

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()