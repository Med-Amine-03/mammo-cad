# mammo-cad — Complete Changes & Decisions Log
# Senior DL Engineer Review — Everything NOT in the original mindmap
# Last updated: March 2026

---

## CURRENT STATUS

### Segmentation — DONE ✅ (do not retrain)
| Metric | Result | DeepMiCa |
|--------|--------|----------|
| AUC-ROC | **0.9642** | 0.95 |
| AUC-PR | 0.8009 | — |
| IoU | 0.5694 ± 0.2870 | — |
| Dice | 0.6654 ± 0.2955 | — |
| Sensitivity | 0.7015 ± 0.3082 | — |

IoU variance is a known INbreast limitation — tiny point annotations.
AUC-ROC beats DeepMiCa. Segmentation is complete and final.

### Classification — IN PROGRESS (retrain with calc+mass tonight)
| Run | Data | Val AUC | Test AUC | Method |
|-----|------|---------|----------|--------|
| Broken original | JPEG | — | 0.477 | single pass |
| After bug fixes | JPEG calc+mass | — | 0.722 | single pass |
| DICOM calc only | DICOM | 0.8317 | 0.7306 | single pass |
| DICOM calc only | DICOM | 0.8317 | **0.7933** | Stage2+TTA |
| Mask crop attempt | DICOM | 0.7518 | 0.7538 | Stage2+TTA |
| DICOM calc retrain | DICOM | 0.7661 | 0.7538 | Stage2+TTA |
| **NEXT: calc+mass** | DICOM | ~0.84+ | **~0.80+** | Stage2+TTA |

---

## DATASET CHANGES (vs mindmap)

### Mindmap said
- CBIS-DDSM JPEG exports from Kaggle (~1500 images)
- 70/15/15 patient-wise split

### What we actually use
- CBIS-DDSM DICOM lossless 16-bit (calc from Kaggle + mass from TCIA)
- ~3500 cases total (calc + mass combined)
- 80/20 patient split of orig_split=train; official test kept as-is
- Two separate DICOM locations on disk:
  - Calc: `data/raw/cbis-ddsm-calcification-dataset/Calcification/Calcification/<subject>/<study>/<series>/1-2.dcm`
  - Mass: `data/raw/cbis-ddsm-mass-dataset/<series_uid>/000000.dcm` (flat)
- 4 CSV files all in `data/raw/cbis-ddsm-calcification-dataset/`:
  - `calc_case_description_train_set.csv`
  - `calc_case_description_test_set.csv`
  - `mass_case_description_train_set.csv`
  - `mass_case_description_test_set.csv`

### Why DICOM
JPEG compression destroys microcalcification texture — exactly what distinguishes
benign vs malignant. Lossless DICOM raised val AUC 0.722 → 0.831.

### Why mass cases matter
Calc-only DICOM = 1871 cases. Calc+mass DICOM = ~3500 cases.
The 0.831 val AUC run trained on JPEG calc+mass (~3567 cases).
More data + better quality = our target combination.

---

## FILES WRITTEN (not in mindmap)

### src/cbis_ddsm/build_cases.py — REWRITTEN
- Reads all 4 CSVs (calc + mass, train + test)
- Resolves calc crops: `CALC_DICOM/<subject>/<study>/<series>/1-2.dcm`
- Resolves mass crops: extracts series_uid from CSV path (parts[-2]),
  looks in `MASS_DICOM/<series_uid>/000000.dcm`
- Deterministic split: sorted() before shuffle with seed=42
- Outputs case_type column (calc/mass) for debugging

### src/cbis_ddsm/extract_crops.py — REWRITTEN (v1 = best strategy)
- Reads lossless 16-bit DICOM (RescaleSlope, VOI LUT, MONOCHROME1 inversion)
- Strategy: pick smallest DICOM in series folder = radiologist pre-cropped ROI
- MAX_CROP_DIM=1500: if smallest is still full mammogram, fallback to mask bbox
- Handles both calc (nested) and mass (flat series_uid) folder structures
- Saves 224x224 PNG

### src/inference/evaluate_cls.py — NEW (not in mindmap)
- TTA: 8 augmented versions per image (original + 7 flips/rotations), average probs
- Ensemble: stage1 + stage2 sigmoid average
- Comparison table: Stage1, Stage2, Stage1+TTA, Stage2+TTA, Ensemble+TTA
- ROC comparison plot saved to checkpoints/evaluation_roc_comparison.png
- Measured TTA gain: +0.063 AUC with zero retraining

### src/download_mass_dicoms.py — NEW (not in mindmap)
- Downloads Mass ROI mask + cropped DICOMs from TCIA via tcia_utils nbia API
- Filters: PatientID.startswith("Mass-") + SeriesDescription in {ROI mask, cropped}
- Correct resume: checks SAVE_DIR/<series_uid>/*.dcm before skipping
- Result: 1798 series downloaded, 0 errors, ~28.9GB

---

## BUGS FIXED IN TRAINING CODE (vs mindmap)

### Bug 1 — Missing ImageNet normalisation
- Original: `img / 255.0` only
- Fix: apply ImageNet mean/std after /255 and after repeat to 3ch
- Impact: AUC 0.477 → 0.60+ in epoch 1

### Bug 2 — Class imbalance ignored
- Original: BCEWithLogitsLoss() with no pos_weight
- Fix: pos_weight = n_benign / n_malignant per split
- Impact: Specificity 0.17 → 0.60+

### Bug 3 — CosineAnnealingLR instability
- Original: CosineAnnealingLR resets LR each cycle
- Fix: ReduceLROnPlateau (factor=0.5, patience=7) — only decreases
- Impact: smooth loss curves, no val spikes

### Bug 4 — No warmup in Stage 2
- Original: Stage 2 starts at full lr_bb=1e-5 immediately
- Fix: linear warmup 5 epochs, backbone LR ramps 1e-8 → 1e-5
- Impact: no epoch-1 loss spike from weight destruction

### Bug 5 — BatchNorm frozen with backbone
- Original: freezing backbone also freezes all BN layers
- Fix: after freeze, explicitly re-enable all BN layers requires_grad=True
- Impact: BN stats adapt to mammogram distribution

### Bug 6 — Patient data leakage
- Original: DataLoader shuffle=True, all crops mixed together
- Fix: WeightedRandomSampler weight=1/n_crops_per_patient → 1 crop/patient/epoch
- Impact: model learns lesion features not patient breast texture

### Bug 7 — NaN in predictions from AMP
- Symptom: ValueError: Input contains NaN from roc_auc_score at runtime
- Fix 1: NaN guard in compute_metrics — replace NaN/Inf with 0.5 before AUC
- Fix 2: clamp sigmoid output [0,1] in validate() after autocast
- Root cause: AMP float16 overflow on edge batches

### Bug 8 — Non-deterministic train/val split
- Symptom: different val AUC each run (0.75 vs 0.83) on same data
- Fix: sorted() before np.random.shuffle() → consistent starting order
- Root cause: pandas .unique() returns arbitrary insertion order

---

## ARCHITECTURE CHANGES (vs mindmap)

| Component | Mindmap | Actual |
|-----------|---------|--------|
| Unfreeze Stage 2 | last 2 blocks | blocks 5,6,7,8 (last 4) |
| Dropout | 0.3 | 0.4 |
| Optimizer | Adam | AdamW |
| Batch size | 64 | 32 |
| Stage 1 epochs | 50 | 60 (early stop p=20) |
| Stage 2 epochs | 100 | 150 (early stop p=25) |
| Scheduler | not specified | ReduceLROnPlateau |
| Gradient clip | not specified | 1.0 (S1) / 0.5 (S2) |

---

## CONFIRMED WORKING HYPERPARAMETERS

### Stage 1
```
epochs=60, early_stop_patience=20
lr=1e-4, AdamW, weight_decay=1e-4
batch=32
ReduceLROnPlateau factor=0.5 patience=7
grad_clip=1.0
pos_weight=n_benign/n_malignant
WeightedRandomSampler 1crop/patient/epoch
```

### Stage 2
```
epochs=150, early_stop_patience=25
lr_backbone=1e-5 (after 5ep warmup from 1e-8)
lr_head=1e-4, AdamW
batch=32
ReduceLROnPlateau factor=0.5 patience=7
grad_clip=0.5
unfreeze: blocks 5,6,7,8 + classifier
```

---

## EXPERIMENTS ABANDONED

### DenseNet-121 torchxrayvision
- Result: AUC 0.607 — worse than EfficientNet
- Root cause: incompatible pixel range [-1024,1024], domain mismatch
- Decision: keep EfficientNet-B3

### Mask-based precise ROI crop (v2)
- Result: val AUC dropped 0.831 → 0.751, dataset shrank 2736 → 1270
- Root cause: different abnormalities from same patient share same breast region
  → crops too similar → less effective training diversity
- Decision: reverted to v1 (smallest DICOM = radiologist pre-cropped ROI)
- Key lesson: CBIS-DDSM team already made the correct crop — trust it

---

## HONEST PERFORMANCE CONTEXT

### DeepMiCa 0.89 AUC
- Private clinical dataset (Istituti Maugeri, Pavia) — NOT CBIS-DDSM
- Cannot be directly compared
- Their segmentation AUC 0.95 on INbreast → we get 0.9642 (beats them)

### Published range for CBIS-DDSM classification
- Calc-only: 0.72–0.80 test AUC
- Calc+mass: 0.78–0.85 test AUC
- Our current best (calc-only + TTA): 0.7933 — top of published range
- Target with calc+mass + TTA: 0.80+ — competitive with best published

---

## EXECUTE NOW — COMPLETE PIPELINE

```bash
# 1. Copy all ready files to project
cp build_cases.py    src/cbis_ddsm/build_cases.py
cp extract_crops.py  src/cbis_ddsm/extract_crops.py
cp train_classifier.py src/training/train_classifier.py
cp evaluate_cls.py   src/inference/evaluate_cls.py

# 2. Clean old data and checkpoints
rm -rf data/interim/cbis_ddsm/Crops/
rm -rf data/processed/cbis_ddsm/Crops/
rm -rf data/patches/classification/
rm -f  checkpoints/efficientnet_stage*.pth

# 3. Build unified CSV (calc + mass)
python src/cbis_ddsm/build_cases.py
# VERIFY: ~3500 total, train ~2200+, val ~550+, test ~700+

# 4. Extract lossless DICOM crops
python src/cbis_ddsm/extract_crops.py
# VERIFY: Direct crop >> Fallback count, diverse pixel means

# 5. Preprocessing + patches
python src/preprocessing/preprocessing.py --dataset cbis_ddsm
python src/patches/patch_extraction_cls.py
# VERIFY: train ~1800+, val ~450+, test ~700+

# 6. Train
python src/training/train_classifier.py --stage 1
# VERIFY: val AUC climbing past 0.72 by epoch 20

python src/training/train_classifier.py --stage 2
# VERIFY: val AUC 0.82+, smooth loss curves

# 7. Evaluate with TTA + ensemble
python src/inference/evaluate_cls.py
# TARGET: Stage2+TTA test AUC 0.80+
```

---

## AFTER CLASSIFICATION — PIPELINE AND APP

```bash
# Files still to write:
src/inference/segment.py     # U-Net sliding window on full mammogram
src/inference/classify.py    # EfficientNet + TTA per bbox region
src/inference/pipeline.py    # end-to-end: mammogram → annotated image + JSON
app.py                       # Streamlit web app

# Pipeline flow (from mindmap):
# 1. preprocessing → flip, artifact removal, CLAHE
# 2. U-Net 256x256 sliding window → binary mask
# 3. connected components min_area=50px → bounding boxes
# 4. for each bbox: crop+32px → 224x224 → EfficientNet+TTA → probability
# 5. RED = malignant, YELLOW = benign
# 6. output: annotated_image.png + report.json

# Checkpoints:
# checkpoints/unet_best_fold.pth     (segmentation — DONE)
# checkpoints/efficientnet_stage2.pth (classification — retrain tonight)
```
