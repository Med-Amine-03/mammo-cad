# mammo-cad — Final Mind Map & Pipeline
## Our Own Implementation (Inspired by DeepMiCa 2023)
### RTX 4090 | Linux | PyTorch 2.x

---

## DATASETS

```
INbreast (410 DICOM + XML)     → SEGMENTATION ONLY
  Filter: Micros=1 cases only  → ~100-150 images
  Split:  10-fold CV            → patient-wise
  Why:    small dataset needs maximum data usage
          directly comparable to DeepMiCa results

CBIS-DDSM (calc + mass cases)  → CLASSIFICATION ONLY
  Cases:  ~1500 cropped images + BENIGN/MALIGNANT labels
  Split:  70% train / 15% val / 15% test patient-wise
  Masks:  IGNORED (imprecise for calc cases)
  Why:    large enough for simple split
          10-fold would take days of training
```

---

## PROJECT STRUCTURE

```
mammo-cad/
│
├── data/
│   ├── raw/                              ← NEVER TOUCH
│   │   ├── inbreast/
│   │   │   └── INbreast Release 1.0/
│   │   │       ├── AllDICOMs/            410 .dcm
│   │   │       ├── AllXML/               410 .xml plist
│   │   │       └── INbreast.csv          metadata
│   │   └── cbis_ddsm/
│   │       ├── jpeg/                     all images
│   │       └── csv/
│   │           ├── calc_case_description_train_set.csv
│   │           ├── calc_case_description_test_set.csv
│   │           ├── mass_case_description_train_set.csv
│   │           ├── mass_case_description_test_set.csv
│   │           └── dicom_info.csv
│   │
│   ├── interim/                          ← converted raw data
│   │   ├── inbreast/
│   │   │   ├── AllPng/                   DICOM → 8bit PNG
│   │   │   ├── Masks/                    XML → binary masks
│   │   │   └── inbreast_mc.csv           Micros=1 + patient_id
│   │   └── cbis_ddsm/
│   │       ├── Crops/                    JPEG crops → PNG
│   │       └── cbis_ddsm_cases.csv       unified CSV + labels
│   │
│   ├── processed/                        ← after preprocessing
│   │   ├── inbreast/
│   │   │   ├── AllPng/                   flip+artifacts+crop+CLAHE
│   │   │   └── Masks/                    aligned masks
│   │   └── cbis_ddsm/
│   │       └── Crops/                    CLAHE applied
│   │
│   └── patches/
│       ├── segmentation/                 INbreast 256×256
│       │   ├── all/                      ALL positive patches
│       │   │   ├── images/               used for 10-fold CV
│       │   │   └── masks/
│       │   └── test/                     held-out test set
│       │       ├── images/               10% patients
│       │       └── masks/
│       └── classification/               CBIS-DDSM 224×224
│           ├── train/
│           │   ├── benign/
│           │   └── malignant/
│           ├── val/
│           │   ├── benign/
│           │   └── malignant/
│           └── test/
│               ├── benign/
│               └── malignant/
│
├── src/
│   │
│   ├── inbreast/
│   │   ├── dicom_to_png.py
│   │   │     pydicom → normalize 16bit→8bit → save PNG
│   │   │     OUTPUT: data/interim/inbreast/AllPng/
│   │   │
│   │   ├── parse_xml_masks.py
│   │   │     Apple plist XML parsing
│   │   │     Type 15 polygon → cv2.fillPoly
│   │   │     Type 19 dot     → cv2.circle radius=3
│   │   │     OUTPUT: data/interim/inbreast/Masks/
│   │   │
│   │   └── build_csv.py
│   │         filter Micros=1 only
│   │         keep patient_id for CV grouping
│   │         OUTPUT: data/interim/inbreast/inbreast_mc.csv
│   │
│   ├── cbis_ddsm/
│   │   ├── build_cases.py
│   │   │     calc_train+calc_test+mass_train+mass_test
│   │   │     map UIDs → JPEG paths
│   │   │     MALIGNANT=1, else=0
│   │   │     patient-wise 70/15/15 split
│   │   │     OUTPUT: data/interim/cbis_ddsm/cbis_ddsm_cases.csv
│   │   │
│   │   └── extract_crops.py
│   │         JPEG → PNG conversion
│   │         clean naming: {patient_id}_{idx}.png
│   │         OUTPUT: data/interim/cbis_ddsm/Crops/
│   │
│   ├── preprocessing/
│   │   └── preprocessing.py
│   │         --dataset inbreast:
│   │           1. flip breast to LEFT
│   │           2. remove artifacts (right/down/up/left)
│   │           3. keep largest blob
│   │           4. crop black background
│   │           5. CLAHE clip=2.0 tile=8×8
│   │           OUTPUT: data/processed/inbreast/
│   │         --dataset cbis_ddsm:
│   │           1. CLAHE only
│   │           OUTPUT: data/processed/cbis_ddsm/
│   │
│   ├── patches/
│   │   ├── patch_extraction_seg.py
│   │   │     pad image to divisible by 256
│   │   │     sliding window 256×256 step=256
│   │   │     save all patches
│   │   │     reduce_patches: DELETE mask.sum()=0
│   │   │     keep POSITIVE patches only
│   │   │     separate: 10% patients → test/
│   │   │               90% patients → all/ (for CV folds)
│   │   │     OUTPUT: data/patches/segmentation/
│   │   │
│   │   └── patch_extraction_cls.py
│   │         resize 224×224
│   │         organize by split/label
│   │         OUTPUT: data/patches/classification/
│   │
│   ├── models/
│   │   ├── unet.py
│   │   │     standard U-Net (like DeepMiCa)
│   │   │     encoder:    1→64→128→256→512
│   │   │     bottleneck: 512→1024
│   │   │     decoder:    1024→512→256→128→64
│   │   │     output:     Conv2d(64,1,1) raw logits
│   │   │     input:      (B, 1, 256, 256)
│   │   │     output:     (B, 1, 256, 256)
│   │   │
│   │   ├── losses.py
│   │   │     topk_CE (OHEM):
│   │   │       BCEWithLogitsLoss reduction=none
│   │   │       keep ALL positive pixels
│   │   │       keep TOP 3×n_pos hardest negatives
│   │   │       mean over selected pixels
│   │   │
│   │   └── efficientnet.py               ← NEW (upgraded from ResNet50)
│   │         EfficientNet-B3 pretrained ImageNet
│   │         freeze ALL backbone
│   │         replace classifier:
│   │           Linear(1536→256)→ReLU→Dropout(0.3)→Linear(256→1)
│   │         output: raw logit
│   │
│   ├── dataset/
│   │   ├── seg_dataset.py
│   │   │     reads image+mask paths
│   │   │     normalize /255
│   │   │     augmentation train only:
│   │   │       HFlip, VFlip, Rotate90 random
│   │   │     returns (image_tensor, mask_tensor)
│   │   │
│   │   └── cls_dataset.py
│   │         reads image path + label
│   │         resize 224×224
│   │         normalize /255
│   │         repeat grayscale → 3 channels
│   │         augmentation train only:
│   │           HFlip, VFlip, Rotate90
│   │         returns (image_tensor, label_tensor)
│   │
│   ├── training/
│   │   ├── train_unet.py
│   │   │     10-FOLD CROSS VALIDATION:
│   │   │       GroupKFold(n_splits=10) by patient_id
│   │   │       for fold in range(10):
│   │   │         train on 9 folds
│   │   │         validate on 1 fold
│   │   │         save best model per fold
│   │   │       report: mean±std IoU across folds
│   │   │       final evaluation on held-out test set
│   │   │     PER FOLD CONFIG:
│   │   │       loss:      topk_CE (OHEM)
│   │   │       optimizer: SGD lr=0.001 momentum=0.99
│   │   │       scheduler: MultiStepLR milestone=150 gamma=0.1
│   │   │       batch:     8
│   │   │       epochs:    300
│   │   │       save by:   best val IoU
│   │   │     OUTPUT: checkpoints/unet_fold{1-10}.pth
│   │   │             checkpoints/unet_best_fold.pth
│   │   │
│   │   └── train_classifier.py
│   │         STAGE 1 — Feature Extraction (50 epochs):
│   │           freeze ALL EfficientNet-B3 backbone
│   │           train classifier head only
│   │           optimizer: Adam lr=1e-4
│   │           loss: BCEWithLogitsLoss
│   │           batch: 64
│   │           save by: best val AUC
│   │         STAGE 2 — Fine Tuning (100 epochs):
│   │           load stage1 best checkpoint
│   │           unfreeze last 2 blocks + classifier
│   │           optimizer: Adam lr=1e-5 / head lr=1e-4
│   │           batch: 64
│   │           save by: best val AUC
│   │         metrics: AUC-ROC, Sensitivity, Specificity, F1
│   │         OUTPUT: checkpoints/efficientnet_stage1.pth
│   │                 checkpoints/efficientnet_stage2.pth
│   │
│   └── inference/
│       ├── segment.py
│       │     load best U-Net checkpoint
│       │     preprocess mammogram
│       │     pad to divisible by 256
│       │     sliding window 256×256 step=256
│       │     U-Net → sigmoid → binary per patch
│       │     unpatchify → full mask
│       │     return binary mask
│       │
│       ├── classify.py
│       │     load EfficientNet checkpoint
│       │     find connected components min_area=50px
│       │     for each region:
│       │       crop + padding=32px
│       │       resize 224×224 → repeat 3ch
│       │       EfficientNet → sigmoid → probability
│       │     return regions + labels + confidence
│       │
│       └── pipeline.py
│             INPUT:  mammogram PNG
│             STEP 1: preprocessing
│             STEP 2: segment.py → full MC mask
│             STEP 3: classify.py → regions + labels
│             STEP 4: draw RED/YELLOW boxes on image
│             STEP 5: save annotated image + JSON report
│             OUTPUT: annotated_image.png + report.json
│
├── checkpoints/
│   ├── unet_fold1.pth  ...  unet_fold10.pth
│   ├── unet_best_fold.pth        best fold for inference
│   ├── efficientnet_stage1.pth
│   └── efficientnet_stage2.pth   used for inference
│
├── outputs/
│   ├── predictions/              inference results
│   └── evaluation/               metrics + plots + GradCAM
│
└── notebooks/
    ├── 01_verify_inbreast.ipynb  verify DICOM+masks
    ├── 02_verify_cbis.ipynb      verify crops+labels
    ├── 03_evaluate_seg.ipynb     segmentation metrics
    └── 04_evaluate_cls.ipynb     classification metrics + GradCAM
```

---

## FULL PIPELINE EXECUTION ORDER

```
═══════════════════════════════════════════════════════════
PHASE 1 — INbreast Data Pipeline
═══════════════════════════════════════════════════════════

[1.1] python src/inbreast/dicom_to_png.py
      IN:  data/raw/inbreast/AllDICOMs/*.dcm
      OUT: data/interim/inbreast/AllPng/*.png  (410 files)
      CHK: 410 PNG files created and readable

[1.2] python src/inbreast/parse_xml_masks.py
      IN:  data/raw/inbreast/AllXML/*.xml
           data/interim/inbreast/AllPng/ (for image dimensions)
      OUT: data/interim/inbreast/Masks/*.png
      CHK: open 5 masks → MC dots visible as white pixels

[1.3] python src/inbreast/build_csv.py
      IN:  data/raw/inbreast/INbreast.csv
      OUT: data/interim/inbreast/inbreast_mc.csv
           columns: file_name, patient_id, birads, split=all
      CHK: print how many Micros=1 cases found

[1.4] python src/preprocessing/preprocessing.py --dataset inbreast
      IN:  data/interim/inbreast/AllPng/ + Masks/
      OUT: data/processed/inbreast/AllPng/ + Masks/
      CHK: images face LEFT, no artifacts, CLAHE applied

[1.5] python src/patches/patch_extraction_seg.py
      IN:  data/processed/inbreast/ + inbreast_mc.csv
      OUT: data/patches/segmentation/all/images+masks/
           data/patches/segmentation/test/images+masks/
      HOW: 10% patients → test/ (held-out, never used in CV)
           90% patients → all/ (used in 10-fold CV)
           reduce_patches: delete mask.sum()=0
      CHK: ALL patches in both folders have non-zero masks


═══════════════════════════════════════════════════════════
PHASE 2 — CBIS-DDSM Data Pipeline
═══════════════════════════════════════════════════════════

[2.1] python src/cbis_ddsm/build_cases.py
      IN:  data/raw/cbis_ddsm/csv/*.csv + jpeg/
      OUT: data/interim/cbis_ddsm/cbis_ddsm_cases.csv
      CHK: ~1500 cases, print label distribution

[2.2] python src/cbis_ddsm/extract_crops.py
      IN:  cbis_ddsm_cases.csv
      OUT: data/interim/cbis_ddsm/Crops/*.png
      CHK: ~1500 PNG files readable

[2.3] python src/preprocessing/preprocessing.py --dataset cbis_ddsm
      IN:  data/interim/cbis_ddsm/Crops/
      OUT: data/processed/cbis_ddsm/Crops/
      CHK: CLAHE applied, images look enhanced

[2.4] python src/patches/patch_extraction_cls.py
      IN:  data/processed/cbis_ddsm/ + cbis_ddsm_cases.csv
      OUT: data/patches/classification/{train,val,test}/{benign,malignant}/
      CHK: print counts per split per class


═══════════════════════════════════════════════════════════
PHASE 3 — U-Net Segmentation (10-fold CV)
═══════════════════════════════════════════════════════════

[3.1] python src/training/train_unet.py
      SPLITS:    GroupKFold(10) on patient_id
      PER FOLD:
        loss:      topk_CE (OHEM)
        optimizer: SGD lr=0.001 momentum=0.99
        scheduler: MultiStepLR milestone=150 gamma=0.1
        batch:     8
        epochs:    300
        save:      checkpoints/unet_fold{k}.pth
      AFTER ALL FOLDS:
        report mean±std IoU across 10 folds
        evaluate best fold on held-out test/
        save: checkpoints/unet_best_fold.pth
      TIME:      ~20 hours on RTX 4090 (overnight)
      TARGET:    mean IoU > 0.70
                 test IoU > 0.65


═══════════════════════════════════════════════════════════
PHASE 4 — EfficientNet-B3 Classification
═══════════════════════════════════════════════════════════

[4.1] python src/training/train_classifier.py --stage 1
      MODEL:   EfficientNet-B3 pretrained
               freeze ALL backbone
               train head only
      CONFIG:  batch=64 epochs=50 lr=1e-4
      TARGET:  val AUC > 0.75
      SAVE:    checkpoints/efficientnet_stage1.pth

[4.2] python src/training/train_classifier.py --stage 2
      MODEL:   load stage1 checkpoint
               unfreeze last 2 blocks + head
      CONFIG:  batch=64 epochs=100 lr=1e-5/1e-4
      TARGET:  val AUC > 0.88
      SAVE:    checkpoints/efficientnet_stage2.pth
      TIME:    ~3 hours on RTX 4090


═══════════════════════════════════════════════════════════
PHASE 5 — Evaluation
═══════════════════════════════════════════════════════════

[5.1] notebooks/03_evaluate_seg.ipynb
      - IoU mean±std across 10 folds
      - Dice, Sensitivity, Specificity on test set
      - Best/worst prediction visualizations
      - Reconstruct full mammogram mask
      SAVE: outputs/evaluation/seg_results.txt

[5.2] notebooks/04_evaluate_cls.ipynb
      - AUC-ROC curve
      - Confusion matrix
      - Sensitivity, Specificity, F1
      - GradCAM visualization (medcam library)
      SAVE: outputs/evaluation/cls_results.txt


═══════════════════════════════════════════════════════════
PHASE 6 — Full Pipeline Inference
═══════════════════════════════════════════════════════════

[6.1] python src/inference/pipeline.py --input mammogram.png

      INPUT:  any mammogram PNG/JPEG

      STEP 1: preprocessing
              flip → artifact removal → crop → CLAHE

      STEP 2: pad image to divisible by 256

      STEP 3: sliding window 256×256 step=256
              U-Net → sigmoid → binary mask per patch

      STEP 4: unpatchify → full resolution MC mask

      STEP 5: connected components
              filter min_area=50 pixels

      STEP 6: for each region:
              crop + 32px padding
              resize 224×224 → 3 channels
              EfficientNet-B3 → sigmoid → probability

      STEP 7: draw boxes on original image
              RED    = MALIGNANT (prob > 0.5)
              YELLOW = BENIGN    (prob <= 0.5)

      OUTPUT:
        annotated_image.png
        report.json:
        {
          "regions_found": N,
          "results": [
            {
              "region_id": 1,
              "location": [x1,y1,x2,y2],
              "label": "MALIGNANT",
              "confidence": 0.94
            }
          ]
        }
```

---

## KEY TECHNICAL DECISIONS — FINAL

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEGMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model     : U-Net base_ch=64 (standard, no residual)
Loss      : topk_CE (OHEM)
            ALL positive pixels + TOP 3x hard negatives
            forces model to focus on tiny MC dots
Optimizer : SGD lr=0.001 momentum=0.99
Scheduler : MultiStepLR milestone=150 gamma=0.1
Batch     : 8
Epochs    : 300 per fold
Split     : 10-fold GroupKFold by patient_id
Patches   : 256×256 POSITIVE ONLY (reduce_patches)
Augment   : HFlip VFlip Rotate90 (train only)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASSIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model     : EfficientNet-B3 pretrained ImageNet
            (upgraded from ResNet50 — better results)
Head      : Linear(1536→256)→ReLU→Drop(0.3)→Linear(256→1)
Loss      : BCEWithLogitsLoss
Stage 1   : freeze backbone, train head, lr=1e-4, 50ep
Stage 2   : unfreeze last 2 blocks, lr=1e-5, 100ep
Batch     : 64
Split     : 70/15/15 patient-wise
Input     : 224×224 grayscale→3ch
Metric    : AUC-ROC (primary), Sensitivity, F1
```

---

## EXPECTED RESULTS

```
Segmentation (U-Net 10-fold on INbreast):
  IoU mean±std  : 0.72 ± 0.05   (target, realistic)
  Sensitivity   : > 0.75
  Comparable to : DeepMiCa (0.95 AUC, different metric)

Classification (EfficientNet-B3 on CBIS-DDSM):
  AUC-ROC       : 0.88 - 0.92   (matches/beats DeepMiCa 0.89)
  Sensitivity   : > 0.82
  Specificity   : > 0.78
  F1 Score      : > 0.83
```

---

## VALIDATION GATES

```
After [1.2] parse_xml_masks:
  □ 5 masks show MC dots as white pixels
  □ mask size matches image size exactly

After [1.5] patch_extraction_seg:
  □ ZERO patches with all-black masks
  □ print patch counts per split

Before [3.1] train_unet:
  □ sanity check: overfit 10 samples
  □ IoU must reach > 0.90 in 100 steps
  □ if fails → check topk_CE implementation

After [3.1] fold 1 epoch 50:
  □ debug images show white prediction on MC regions
  □ NOT all black predictions

After [4.1] stage 1 epoch 20:
  □ val AUC > 0.65
  □ if stuck at 0.5 → data loading bug
```

---

## HOW WE COMPARE TO DEEPMICA

```
DeepMiCa                     mammo-cad (ours)
───────────────────────────────────────────────────────
U-Net standard           →   U-Net standard (same)
topk_CE loss             →   topk_CE loss (same)
SGD momentum=0.99        →   SGD momentum=0.99 (same)
10-fold CV               →   10-fold CV (same)
256×256 patches          →   256×256 patches (same)
positive patches only    →   positive patches only (same)
ResNet18 + VGG16         →   EfficientNet-B3 (BETTER)
Accuracy metric          →   AUC-ROC (BETTER)
No full pipeline         →   pipeline.py end-to-end (BETTER)
wandb required           →   local logging (SIMPLER)
```

---

## DEPLOYMENT (AFTER TRAINING)

```
Streamlit Web App:
  User uploads mammogram
  Pipeline runs automatically
  Shows annotated image + report
  Deploy on Linux server

Input:  mammogram PNG/JPEG
Output: annotated image + JSON report
        MALIGNANT regions in RED
        BENIGN regions in YELLOW
        confidence % per region
```

---

*Reference: DeepMiCa (Castellino et al., CMPB 2023)*
*Datasets: INbreast (Moreira 2012) + CBIS-DDSM (Lee 2017)*
*Hardware: RTX 4090 Linux | Framework: PyTorch 2.x*
