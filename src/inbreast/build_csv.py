
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

RAW_CSV  = Path("data/raw/inbreast/INbreast Release 1.0/INbreast.csv")
PNG_DIR  = Path("data/interim/inbreast/AllPng")
MASK_DIR = Path("data/interim/inbreast/Masks")
OUT_CSV  = Path("data/interim/inbreast/inbreast_mc.csv")

TEST_SIZE   = 0.10
RANDOM_SEED = 42


def main():
    df = pd.read_csv(str(RAW_CSV), sep=";", encoding="ISO-8859-1")
    print(f"Total INbreast cases : {len(df)}")
    print(f"Columns              : {list(df.columns)}")

    
    id_to_png = {}
    for p in PNG_DIR.glob("*.png"):
        short_id = p.stem.split("_")[0]
        id_to_png[short_id] = p

    
    records = []
    for _, row in df.iterrows():
        short_id = str(int(row["File Name"]))
        png_path = id_to_png.get(short_id)
        if png_path is None:
            continue

        mask_path = MASK_DIR / png_path.name
        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.max() == 0:
            continue  

        records.append({
            "file_name"  : short_id,
            "png_name"   : png_path.name,
            "patient_id" : row["Patient ID"] if row["Patient ID"] != "removed"
                           else short_id,  
            "birads"     : row["Bi-Rads"],
            "mc_pixels"  : int(np.sum(mask > 0)),
        })

    df_mc = pd.DataFrame(records).reset_index(drop=True)
    print(f"\nCases with MC masks  : {len(df_mc)}")

    
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )
    groups   = df_mc["file_name"].values  
    cv_idx, test_idx = next(splitter.split(df_mc, groups=groups))

    df_mc["split"] = "all"
    df_mc.iloc[test_idx, df_mc.columns.get_loc("split")] = "test"

    df_mc.to_csv(str(OUT_CSV), index=False)

    print(f"\nSaved: {OUT_CSV}")
    print(f"  CV pool (all) : {(df_mc.split=='all').sum()}")
    print(f"  Test (held)   : {(df_mc.split=='test').sum()}")
    print(f"\nBI-RADS distribution:")
    print(df_mc["birads"].value_counts().sort_index())
    print(f"\nMC pixels stats:")
    print(f"  min    : {df_mc.mc_pixels.min()}")
    print(f"  median : {df_mc.mc_pixels.median():.0f}")
    print(f"  max    : {df_mc.mc_pixels.max()}")

    print(f"\nSample rows:")
    print(df_mc.head(3).to_string())


if __name__ == "__main__":
    main()