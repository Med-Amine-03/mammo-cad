
import pydicom
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

RAW_DIR  = Path("data/raw/inbreast/INbreast Release 1.0/AllDICOMs")
OUT_DIR  = Path("data/interim/inbreast/AllPng")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dicom_to_8bit(dcm_path: Path) -> np.ndarray:
    ds    = pydicom.dcmread(str(dcm_path))
    img   = ds.pixel_array.astype(np.float32)
    img   = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img   = (img * 255).astype(np.uint8)
    return img


def main():
    dcm_files = sorted(RAW_DIR.glob("*.dcm"))
    print(f"Found {len(dcm_files)} DICOM files")

    failed = []
    for dcm_path in tqdm(dcm_files, desc="Converting DICOM → PNG"):
        try:
            img      = dicom_to_8bit(dcm_path)
            out_path = OUT_DIR / (dcm_path.stem + ".png")
            cv2.imwrite(str(out_path), img)
        except Exception as e:
            print(f"  FAILED: {dcm_path.name} → {e}")
            failed.append(dcm_path.name)

    print(f"\nDone: {len(dcm_files) - len(failed)} converted")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")

    pngs = list(OUT_DIR.glob("*.png"))
    print(f"PNG files in output: {len(pngs)}")
    print(f"Expected: {len(dcm_files)}")


if __name__ == "__main__":
    main()