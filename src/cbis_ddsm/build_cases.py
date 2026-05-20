"""
build_cases.py — Build unified CBIS-DDSM cases CSV (calc + mass)
=================================================================
v4 fix:
  [FIX-1] Patient leakage eliminated.
           Any patient_id that appears in ANY official test CSV is
           quarantined — sent to test regardless of orig_split.
           Guarantees: train∩test = 0, val∩test = 0.

Run:
  python src/cbis_ddsm/build_cases.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


# ── Paths ─────────────────────────────────────────────────────────────
RAW_DIR     = Path("data/raw/cbis-ddsm-calcification-dataset")
CALC_DICOM  = RAW_DIR / "Calcification" / "Calcification"
MASS_DICOM  = Path("data/raw/cbis-ddsm-mass-dataset")
INTERIM_DIR = Path("data/interim/cbis_ddsm")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV     = INTERIM_DIR / "cbis_ddsm_cases.csv"

CSVS = {
    "calc": {
        "train": RAW_DIR / "calc_case_description_train_set.csv",
        "test":  RAW_DIR / "calc_case_description_test_set.csv",
    },
    "mass": {
        "train": RAW_DIR / "mass_case_description_train_set.csv",
        "test":  RAW_DIR / "mass_case_description_test_set.csv",
    },
}


# ── Path resolution ───────────────────────────────────────────────────
def resolve_calc_crop(csv_rel_path: str) -> Path | None:
    clean = csv_rel_path.strip().replace("\\", "/")
    full  = CALC_DICOM / clean
    if full.exists():
        return full
    parts      = clean.split("/")
    subject_id = parts[0]
    filename   = parts[-1]
    subj_dir   = CALC_DICOM / subject_id
    if subj_dir.exists():
        for f in subj_dir.rglob(filename):
            return f
    return None


def resolve_mass_crop(csv_rel_path: str) -> Path | None:
    clean = csv_rel_path.strip().replace("\\", "/")
    parts = clean.split("/")
    if len(parts) < 3:
        return None
    series_uid = parts[-2]
    filename   = parts[-1]
    candidate  = MASS_DICOM / series_uid / filename
    if candidate.exists():
        return candidate
    series_dir = MASS_DICOM / series_uid
    if series_dir.exists():
        dcms = sorted(series_dir.glob("*.dcm"))
        if dcms:
            for d in dcms:
                if "000000" in d.name or "1-1" in d.name:
                    return d
            return dcms[0]
    return None


# ── Main ─────────────────────────────────────────────────────────────
def build_cases() -> None:
    records = []
    missing = 0

    for case_type, splits in CSVS.items():
        for split_name, csv_path in splits.items():
            if not csv_path.exists():
                print(f"  WARNING: {csv_path} not found — skipping")
                continue

            df = pd.read_csv(csv_path)
            df.columns = (df.columns
                          .str.strip()
                          .str.lower()
                          .str.replace(" ", "_"))

            print(f"\n[{case_type:4s}/{split_name:5s}] "
                  f"{len(df)} rows from {csv_path.name}")

            for _, row in df.iterrows():
                patient_id = str(row.get("patient_id", "")).strip()

                pathology = str(row.get("pathology", "")).strip().upper()
                if pathology == "MALIGNANT":
                    label = 1
                elif pathology in ("BENIGN", "BENIGN_WITHOUT_CALLBACK"):
                    label = 0
                else:
                    continue

                csv_crop = str(row.get(
                    "cropped_image_file_path",
                    row.get("cropped_image_path", "")
                )).strip()
                if not csv_crop:
                    missing += 1
                    continue

                if case_type == "calc":
                    crop_dcm = resolve_calc_crop(csv_crop)
                else:
                    crop_dcm = resolve_mass_crop(csv_crop)

                if crop_dcm is None:
                    missing += 1
                    continue

                records.append({
                    "patient_id": patient_id,
                    "crop_dcm":   str(crop_dcm),
                    "label":      label,
                    "pathology":  pathology,
                    "case_type":  case_type,
                    "orig_split": split_name,
                })

    if not records:
        print("\nERROR: No records built. Check DICOM paths.")
        sys.exit(1)

    df_out = pd.DataFrame(records)

    # ── [FIX-1] Leakage-free patient split ───────────────────────────
    # Step 1: collect every patient that appears in ANY test CSV.
    #         These are quarantined — always go to test.
    np.random.seed(42)

    test_patients = set(
        df_out[df_out["orig_split"] == "test"]["patient_id"].unique()
    )
    print(f"\n  Quarantined test patients: {len(test_patients)}")

    # Step 2: 80/20 split ONLY on patients that never appear in test.
    trainval_pats = np.array(sorted(
        df_out[
            (df_out["orig_split"] == "train") &
            (~df_out["patient_id"].isin(test_patients))
        ]["patient_id"].unique()
    ))
    np.random.shuffle(trainval_pats)
    n_train = int(0.80 * len(trainval_pats))
    p_train = set(trainval_pats[:n_train])

    def assign_split(row: pd.Series) -> str:
        # Any patient appearing in test CSVs → test (even if orig=train)
        if row["patient_id"] in test_patients:
            return "test"
        return "train" if row["patient_id"] in p_train else "val"

    df_out["split"] = df_out.apply(assign_split, axis=1)
    df_out.to_csv(OUT_CSV, index=False)

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Saved : {OUT_CSV}")
    print(f"  Total : {len(df_out)} cases | Missing DICOMs: {missing}")
    print()

    for sp in ["train", "val", "test"]:
        sub  = df_out[df_out["split"] == sp]
        b    = (sub["label"] == 0).sum()
        m    = (sub["label"] == 1).sum()
        calc = (sub["case_type"] == "calc").sum()
        mass = (sub["case_type"] == "mass").sum()
        print(f"  [{sp:5s}] {len(sub):4d} | B={b:4d} M={m:4d} "
              f"ratio={b/max(m,1):.2f} | calc={calc} mass={mass}")

    # ── Leakage verification (must all be 0) ─────────────────────────
    print()
    train_p = set(df_out[df_out["split"] == "train"]["patient_id"])
    val_p   = set(df_out[df_out["split"] == "val"]["patient_id"])
    test_p  = set(df_out[df_out["split"] == "test"]["patient_id"])

    tv = len(train_p & val_p)
    tt = len(train_p & test_p)
    vt = len(val_p   & test_p)

    print(f"  Leakage check:")
    print(f"    train∩val  = {tv}  (must be 0)")
    print(f"    train∩test = {tt}  (must be 0)")
    print(f"    val∩test   = {vt}  (must be 0)")

    if tt == 0 and vt == 0:
        print(f"\n  ✅ Clean split — no patient leakage")
    else:
        print(f"\n  ❌ Leakage detected — check assign_split logic")

    total = len(df_out)
    if total >= 3000:
        print(f"  ✅ {total} cases — calc+mass combined")
    elif total >= 1800:
        print(f"  ⚠️  {total} cases — looks calc-only")
    else:
        print(f"  ❌ Only {total} cases — check paths")

    print(f"{'='*60}")
    print("\nNext: python src/cbis_ddsm/extract_crops.py")


if __name__ == "__main__":
    build_cases()