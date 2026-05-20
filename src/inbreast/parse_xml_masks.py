
import plistlib
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

RAW_XML_DIR = Path("data/raw/inbreast/INbreast Release 1.0/AllXML")
PNG_DIR     = Path("data/interim/inbreast/AllPng")
OUT_DIR     = Path("data/interim/inbreast/Masks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOT_RADIUS  = 3


def build_id_to_png(png_dir: Path) -> dict:
    
    mapping = {}
    for p in png_dir.glob("*.png"):
        short_id = p.stem.split("_")[0]
        mapping[short_id] = p
    return mapping


def parse_points(point_list: list) -> list:
    points = []
    for p in point_list:
        p = p.strip("()")
        coords = [float(c.strip()) for c in p.split(",")]
        points.append((int(coords[0]), int(coords[1])))
    return points


def xml_to_mask(xml_path: Path, img_shape: tuple) -> np.ndarray:
    mask = np.zeros(img_shape, dtype=np.uint8)

    with open(str(xml_path), "rb") as f:
        plist_data = plistlib.load(f)

    images = plist_data.get("Images", [])
    if not images:
        return mask

    rois = images[0].get("ROIs", [])

    for roi in rois:
        roi_type = roi.get("Type", -1)
        name     = roi.get("Name", "")
        points   = roi.get("Point_px", [])

        if "alcification" not in name:
            continue
        if not points:
            continue

        parsed = parse_points(points)

        if len(parsed) == 1 or roi_type == 19:
            cx, cy = parsed[0]
            cv2.circle(mask, (cx, cy), DOT_RADIUS, 255, -1)
        elif len(parsed) >= 3:
            pts = np.array(parsed, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)

    return mask


def main():
    xml_files  = sorted(RAW_XML_DIR.glob("*.xml"))
    id_to_png  = build_id_to_png(PNG_DIR)

    print(f"Found {len(xml_files)} XML files")
    print(f"Found {len(id_to_png)} PNG files")

    no_png    = []
    no_mc     = []
    converted = 0

    for xml_path in tqdm(xml_files, desc="Parsing XML → masks"):
        short_id = xml_path.stem
        png_path = id_to_png.get(short_id)

        if png_path is None:
            no_png.append(short_id)
            continue

        img  = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        mask = xml_to_mask(xml_path, (h, w))

        out_path = OUT_DIR / png_path.name
        cv2.imwrite(str(out_path), mask)
        converted += 1

        if mask.max() == 0:
            no_mc.append(short_id)

    print(f"\nDone: {converted} masks created")
    print(f"No PNG found for: {len(no_png)} XML files")
    print(f"Masks with NO MC annotations : {len(no_mc)}")
    print(f"Masks WITH MC annotations    : {converted - len(no_mc)}")

   
    print("\nVerifying 3 MC masks:")
    mc_masks = [p for p in OUT_DIR.glob("*.png")
                if cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).max() > 0]
    for p in mc_masks[:3]:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        print(f"  {p.name}: shape={m.shape} "
              f"MC_pixels={np.sum(m>0)}")


if __name__ == "__main__":
    main()