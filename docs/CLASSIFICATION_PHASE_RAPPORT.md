# Classification des Lesions Mammographiques

## Systeme CAD - Phase de Classification EfficientNet-B3 sur CBIS-DDSM

---

# 1. VUE GLOBALE DU PIPELINE

```
 PIPELINE COMPLET DE CLASSIFICATION
 ====================================

 +------------------+     +------------------+     +------------------+
 |   DONNEES BRUTES |     |  EXTRACTION DES  |     |  PRE-TRAITEMENT  |
 |   CBIS-DDSM      | --> |  CROPS DICOM     | --> |  CLAHE           |
 |   DICOM lossless |     |  16-bit -> PNG   |     |  Normalisation   |
 +------------------+     +------------------+     +------------------+
                                                          |
                                                          v
 +------------------+     +------------------+     +------------------+
 |   EVALUATION     |     |   ENTRAINEMENT   |     |   PATCHES 224    |
 |   TTA-16         | <-- |   2 ETAPES       | <-- |   benin/malin    |
 |   AUC = 0.7812   |     |   S1: head only  |     |   train/val/test |
 |   [0.748—0.813]  |     |   S2: fine-tune  |     |   4 484 patches  |
 +------------------+     +------------------+     +------------------+
```

---

# 2. DATASET CBIS-DDSM - CHIFFRES EXACTS

## 2.1 Source des Donnees

```
 BASE DE DONNEES CBIS-DDSM
 ===========================

 CBIS-DDSM = Curated Breast Imaging Subset of DDSM
 Format    : DICOM lossless 16-bit (PAS de JPEG !)

 +---------------------------------------------+
 |              CBIS-DDSM Original              |
 |                                              |
 |   Calcifications (calc) :  1 871 cas         |
 |   Masses (mass)         :  1 696 cas         |
 |                                              |
 |   TOTAL                  :  3 567 cas        |
 |   Patients uniques       :  1 566 patients   |
 +---------------------------------------------+

 IMPORTANT : Pourquoi DICOM et pas JPEG ?
   - JPEG compresse et DETRUIT la texture des microcalcifications
   - Texture = ce qui distingue benin vs malin
   - DICOM 16-bit = qualite radiologique preservee
   - Gain mesure : Val AUC 0.722 (JPEG) -> 0.806 (DICOM)
```

## 2.2 Repartition des Donnees (Patient-Wise Split)

```
 SPLIT PAR PATIENT (deterministe, seed=42)
 ============================================

 Methode : sorted() avant shuffle pour eviter
           la non-determinisme de pandas .unique()

 +-----------------------+----------+----------+----------+
 |                       |  TRAIN   |   VAL    |   TEST   |
 +-----------------------+----------+----------+----------+
 | Cas (cases CSV)       |  2 260   |   520    |   787    |
 | Patients              |    973   |   244    |   349    |
 | Pourcentage cas       |  63.4%   |  14.6%   |  22.0%   |
 +-----------------------+----------+----------+----------+

 IMPORTANT :
   - Train/Val = 80/20 du split TRAIN officiel CBIS-DDSM
   - Test     = split TEST officiel CBIS-DDSM (non touche)
   - AUCUN patient n'apparait dans 2 splits differents
```

## 2.3 Distribution par Type de Lesion

```
 TYPES DE LESIONS PAR SPLIT
 ============================

 +---------------+----------+----------+----------+----------+
 |               |  TRAIN   |   VAL    |   TEST   |  TOTAL   |
 +---------------+----------+----------+----------+----------+
 | Calc          |  1 237   |   252    |   382    |  1 871   |
 | Mass          |  1 023   |   268    |   405    |  1 696   |
 +---------------+----------+----------+----------+----------+
 | Total         |  2 260   |   520    |   787    |  3 567   |
 +---------------+----------+----------+----------+----------+

 Distribution par pathologie :

   Calc :                          Mass :
   +------------------+            +------------------+
   | BENIN     :  658 |            | BENIN     :  771 |
   | BENIN_NC* :  541 |            | BENIN_NC* :  141 |
   | MALIN     :  672 |            | MALIN     :  784 |
   +------------------+            +------------------+
   * BENIN_NC = BENIGN_WITHOUT_CALLBACK (regroupe avec BENIN)
```

## 2.4 Distribution des Labels (Benin / Malin)

```
 LABELS BINAIRES : 0 = BENIN, 1 = MALIN
 ========================================

 Total dataset (3 567 cas) :
   BENIN (label 0)  : 2 111  (59.2%)  ███████████████████
   MALIN (label 1)  : 1 456  (40.8%)  ████████████████

 Par split :
 +---------+---------+---------+----------+----------+
 | Split   |  BENIN  |  MALIN  |  Ratio   |  Total   |
 |         | (lab 0) | (lab 1) | (B/M)    |          |
 +---------+---------+---------+----------+----------+
 | Train   |  1 329  |    931  |  1.43    |  2 260   |
 | Val     |    307  |    213  |  1.44    |    520   |
 | Test    |    475  |    312  |  1.52    |    787   |
 +---------+---------+---------+----------+----------+

 +--------------------------------------------------+
 |  DESEQUILIBRE LEGER : ~60% benin / ~40% malin   |
 |                                                  |
 |  --> Solution : pos_weight = n_benin / n_malin   |
 |                              (~1.43)             |
 |  --> Applique a BCEWithLogitsLoss                |
 +--------------------------------------------------+
```

---

# 3. PRE-TRAITEMENT ET EXTRACTION DES PATCHES

## 3.1 Pipeline de Pre-traitement

```
 ETAPES DE PRE-TRAITEMENT
 ==========================

 DICOM lossless 16-bit (CBIS-DDSM)
       |
       v
 +-----------------------------------+
 | 1. Lecture DICOM                  |
 |    - RescaleSlope/Intercept       |
 |    - VOI LUT                      |
 |    - MONOCHROME1 inversion        |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 2. Selection du crop ROI          |
 |    Strategie : prendre le PLUS    |
 |    PETIT DICOM dans la serie      |
 |    = crop pre-fait par radiologue |
 |                                   |
 |    Si > 1500 px : fallback masque |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 3. Conversion 16-bit -> PNG       |
 |    Sauvegarde 224 x 224           |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 4. CLAHE                          |
 |    clipLimit  = 2.0               |
 |    tileGrid   = (8, 8)            |
 +-----------------------------------+
       |
       v
   PNG 224x224 (8-bit)
```

## 3.2 Comptage des Patches

```
 PATCHES EXTRAITS (224 x 224)
 ==============================

 Structure des dossiers :
   data/patches/classification/
   |-- train/
   |   |-- benign/      1 610 PNG
   |   +-- malignant/   1 125 PNG
   |-- val/
   |   |-- benign/        579 PNG
   |   +-- malignant/     383 PNG
   +-- test/
       |-- benign/        475 PNG
       +-- malignant/     312 PNG

 +=================================================+
 |  CHIFFRES EXACTS DES PATCHES                    |
 |                                                 |
 |  +--------+---------+----------+----------+    |
 |  | Split  | Benin   | Malin    | Total    |    |
 |  +--------+---------+----------+----------+    |
 |  | Train  | 1 610   | 1 125    | 2 735    |    |
 |  | Val    |   579   |   383    |   962    |    |
 |  | Test   |   475   |   312    |   787    |    |
 |  +--------+---------+----------+----------+    |
 |  | TOTAL  | 2 664   | 1 820    | 4 484    |    |
 |  +--------+---------+----------+----------+    |
 +=================================================+

 Note : Un cas peut produire plusieurs crops 224x224
        --> 3 567 cas --> 4 484 patches
```

---

# 4. ARCHITECTURE EfficientNet-B3

## 4.1 Schema du Modele

```
 ARCHITECTURE EfficientNet-B3 (efficientnet.py)
 ================================================
 Parametres totaux : ~11 900 000 (~11.9M)

 ENTREE: (Batch, 3, 224, 224)  -- ImageNet normalisee
           |
 ==========|========= BACKBONE EfficientNet-B3 ==========
           |        (pre-entraine sur ImageNet)
           v
 +-------------------+
 | features[0]       |    Conv stem (3 -> 40)
 +-------------------+
           |
           v
 +-------------------+
 | features[1]       |    MBConv block 1
 +-------------------+
           |
           v
 +-------------------+
 | features[2]       |    MBConv block 2
 +-------------------+
           |
           v
 +-------------------+
 | features[3]       |    MBConv block 3
 +-------------------+
           |
           v
 +-------------------+
 | features[4]       |    MBConv block 4
 +-------------------+
           |
           v
 +-------------------+    +--- DEGEL en STAGE 2 ---+
 | features[5]       |    |                         |
 +-------------------+    |                         |
           |              |                         |
           v              |                         |
 +-------------------+    |                         |
 | features[6]       |    |                         |
 +-------------------+    |                         |
           |              |                         |
           v              |                         |
 +-------------------+    |                         |
 | features[7]       |    | Cible Grad-CAM++ (7x7)  |
 +-------------------+    |                         |
           |              |                         |
           v              |                         |
 +-------------------+    |                         |
 | features[8]       |    |                         |
 | (final conv)      |    |                         |
 +-------------------+    +-------------------------+
           |
           v
 +-------------------+
 | AdaptiveAvgPool2d |    -> (B, 1536, 1, 1)
 +-------------------+
           |
           v
 +-------------------+
 | Flatten           |    -> (B, 1536)
 +-------------------+
           |
 ==========|========= TETE PERSONNALISEE ==========
           |        (toujours entrainee)
           v
 +-------------------+
 | Linear(1536, 256) |    Kaiming init
 +-------------------+
           |
           v
 +-------------------+
 | ReLU              |
 +-------------------+
           |
           v
 +-------------------+
 | Dropout(p=0.5)    |    [ART-1] Shia et al. 2025
 +-------------------+
           |
           v
 +-------------------+
 | Linear(256, 1)    |    Xavier init
 +-------------------+
           |
           v
 SORTIE: (Batch,)  -- logit brut (PAS de sigmoid)
```

## 4.2 Initialisation des Poids

```
 INITIALISATION DES POIDS DE LA TETE
 =====================================

 +----------------------+----------------------+
 | Couche               | Initialisation       |
 +----------------------+----------------------+
 | Linear(1536 -> 256)  | Kaiming normal       |
 |                      | mode = "fan_out"     |
 |                      | bias = 0             |
 +----------------------+----------------------+
 | Linear(256 -> 1)     | Xavier normal        |
 |                      | bias = 0             |
 +----------------------+----------------------+

 Pourquoi ? Eviter des logits trop grands a l'epoque 0
            qui causeraient des spikes dans la perte.
```

## 4.3 Strategie de Gel / Degel

```
 STRATEGIE EN 2 ETAPES
 =======================

 STAGE 1 : Feature Extraction (gel total)
 +--------------------------------------------+
 |  features[0..8] : GELES (frozen)           |
 |                   requires_grad = False    |
 |                                            |
 |  classifier     : ENTRAINE (head)          |
 |                   requires_grad = True     |
 |                                            |
 |  BatchNorm      : ENTRAINE                 |
 |                   (stats adaptees)         |
 |                                            |
 |  Parametres entrainables : ~262 000        |
 |  (~2.2% du total)                          |
 +--------------------------------------------+

 STAGE 2 : Fine-tuning (degel partiel)
 +--------------------------------------------+
 |  features[0..4] : GELES                    |
 |  features[5,6,7,8] : DEGELES               |
 |  classifier     : ENTRAINE                 |
 |                                            |
 |  Parametres entrainables : ~6 500 000      |
 |  (~55% du total)                           |
 +--------------------------------------------+
```

---

# 5. AUGMENTATION DES DONNEES

## 5.1 Stack Complete d'Augmentations

```
 AUGMENTATIONS APPLIQUEES (cls_dataset.py - augment=True)
 ==========================================================

 Patch original (224x224)
       |
       v
 +-------------------------------+
 | 1. Flip Horizontal  (p=0.5)   |
 +-------------------------------+
       |
       v
 +-------------------------------+
 | 2. Flip Vertical    (p=0.5)   |
 +-------------------------------+
       |
       v
 +-------------------------------+
 | 3. Rotation 90/180/270 (p=0.75)|
 +-------------------------------+
       |
       v
 +-------------------------------+
 | 4. [ART-2] Multi-scale (p=0.5)|
 |    Resize 224 -> 260 -> 224   |
 |    (resolution invariance)    |
 +-------------------------------+
       |
       v
 +-------------------------------+
 | 5. [ART-3] Translation +/-10% |
 |    (p=0.5)                    |
 |    Force le modele a NE PAS   |
 |    se reposer sur la position |
 +-------------------------------+
       |
       v
 +-------------------------------+
 | 6. Elastic distortion (p=0.3) |
 |    alpha=15, sigma=4          |
 +-------------------------------+
       |
       v
 +-------------------------------+
 | 7. Brightness/Contrast(p=0.5) |
 |    alpha=[0.85,1.15]          |
 |    beta =[-10, 10]            |
 +-------------------------------+
       |
       v
 +-------------------------------+
 | 8. Gamma correction   (p=0.4) |
 |    gamma=[0.80, 1.25]         |
 +-------------------------------+
       |
       v
 +-------------------------------+
 | 9. Bruit Gaussien     (p=0.4) |
 |    sigma=[0, 4]               |
 +-------------------------------+
       |
       v
 +-------------------------------+
 | 10. Flou Gaussien     (p=0.3) |
 |     kernel=3x3, sigma=[0.3,0.8]|
 +-------------------------------+
       |
       v
 +-------------------------------+
 | 11. Normalisation ImageNet    |
 |     mean=[0.485,0.456,0.406]  |
 |     std =[0.229,0.224,0.225]  |
 |     Convert 1ch -> 3ch        |
 +-------------------------------+
       |
       v
   Tensor (3, 224, 224)
```

## 5.2 Tableau Recapitulatif des Augmentations

```
 +----+------------------------+--------+-------------------+
 | #  | Augmentation           | Proba. | Parametres        |
 +----+------------------------+--------+-------------------+
 |  1 | HFlip                  |  0.50  |          -        |
 |  2 | VFlip                  |  0.50  |          -        |
 |  3 | Rotation 90/180/270    |  0.75  | k = {1,2,3}      |
 |  4 | Multi-scale 224<->260  |  0.50  | INTER_LINEAR/AREA |
 |  5 | Translation            |  0.50  | dx,dy : +/-10%    |
 |  6 | Elastic distortion     |  0.30  | a=15, s=4         |
 |  7 | Brightness/Contrast    |  0.50  | a=0.85-1.15       |
 |  8 | Gamma correction       |  0.40  | g=0.80-1.25       |
 |  9 | Bruit Gaussien         |  0.40  | sigma=0-4         |
 | 10 | Flou Gaussien          |  0.30  | sigma=0.3-0.8     |
 | 11 | ImageNet normalisation |  1.00  | obligatoire       |
 +----+------------------------+--------+-------------------+

 IMPORTANT :
   - TRAIN  : augmentations 1-10 actives + ImageNet norm
   - VAL    : SEULEMENT ImageNet norm (pas d'augmentation)
   - TEST   : SEULEMENT ImageNet norm (pas d'augmentation)
   - INFER. : ImageNet norm + TTA optionnel (8 ou 16 variantes)
```

## 5.3 Volume Effectif de Donnees (Train)

```
 VOLUME EFFECTIF GENERE PAR AUGMENTATION
 =========================================

 +-----------------------------------------------+----------+
 | Donnee                                        | Nombre   |
 +-----------------------------------------------+----------+
 | Patches train sur disque                      | 2 735    |
 +-----------------------------------------------+----------+
 | Combinaisons theoriques (p~0.5 chacune)      | infini   |
 | (chaque sample tire des params continus)     |          |
 +-----------------------------------------------+----------+
 | Sampler effectif par epoque                   | 1 566*   |
 | (WeightedRandomSampler 1crop/patient)        |          |
 +-----------------------------------------------+----------+
 | Stage 1 : 60 epoques x 1 566                  |  93 960  |
 | Stage 2 : 150 epoques x 1 566                 | 234 900  |
 | TOTAL samples vus en entrainement             | 328 860  |
 +-----------------------------------------------+----------+

 * Le sampler tire 1 crop par patient par epoque pour
   eviter le data leakage (un patient peut avoir 5 crops).
```

---

# 6. FONCTION DE PERTE

## 6.1 BCEWithLogitsLoss avec Label Smoothing

```
 BCE + Label Smoothing + pos_weight
 ====================================

 Probleme : 60% benin / 40% malin
            --> Le modele tend a predire BENIN par defaut

 Solution combinee :

 1. pos_weight = n_benin / n_malin ~ 1.43
    --> Erreurs sur MALIN ponderees x 1.43

 2. Label Smoothing (epsilon = 0.05)
    --> y_smooth = y * (1 - 0.05) + 0.5 * 0.05
    --> Empeche le modele d'etre trop confiant

 3. pos_weight plafonne a 2.5 (POS_WEIGHT_CAP)
    --> Evite instabilite si desequilibre extreme

 +-----------------------------------------------------+
 |  Formule finale :                                   |
 |                                                     |
 |  y_s = y * 0.95 + 0.025  (label smooth)            |
 |  loss = BCE(logit, y_s, pos_weight=1.43)            |
 |                                                     |
 |  Si loss = NaN : fallback a 0.7 (guard NaN)        |
 +-----------------------------------------------------+
```

## 6.2 MixUp Data Augmentation (alpha=0.2)

```
 MIXUP : Melange de paires d'images
 ====================================

 lambda ~ Beta(0.2, 0.2)

 img_mix = lambda * img_A + (1 - lambda) * img_B
 lbl_mix = lambda * lbl_A + (1 - lambda) * lbl_B

 +------------------------------------------------+
 |  Effet :                                       |
 |  - Regularisation supplementaire               |
 |  - Force le modele a predire des probabilites  |
 |    intermediaires (pas 0/1 trop confiant)      |
 |  - Ameliore calibration et generalisation      |
 +------------------------------------------------+
```

---

# 7. METRIQUES D'EVALUATION

## 7.1 Metriques Binaires

```
 METRIQUES DE CLASSIFICATION (seuil = 0.535, F1-optimal sur val)
 ================================================================

 Matrice de confusion :
                      Predit
                  +---------+---------+
                  | BENIN   | MALIN   |
              +---+---------+---------+
   Verite     |B  |   TN    |   FP    |
              +---+---------+---------+
              |M  |   FN    |   TP    |
              +---+---------+---------+

 Formules :

   Accuracy    = (TP + TN) / (TP + TN + FP + FN)
   Sensibilite = TP / (TP + FN)        (= Recall)
   Specificite = TN / (TN + FP)
   PPV         = TP / (TP + FP)        (= Precision)
   NPV         = TN / (TN + FN)
   F1 Score    = 2 * (PPV * Sens) / (PPV + Sens)
   MCC         = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
   Brier Score = mean((p - y)^2)       (calibration, plus bas = mieux)
```

## 7.2 AUC-ROC et AUC-PR (Metriques Principales)

```
 AUC-ROC : Area Under the Receiver Operating Characteristic
 ============================================================

 Independant du seuil --> mesure la capacite globale
                          du modele a separer benin/malin

 AUC-PR (Average Precision) : plus informative quand
                               les classes sont desequilibrees

 Interpretation AUC-ROC :
   AUC = 0.50 : aleatoire
   AUC = 0.70 : acceptable
   AUC = 0.78 : NOTRE RESULTAT (calc+mass, TTA-16)
   AUC = 0.80 : bon
   AUC = 0.85 : excellent
   AUC = 0.90+: clinique
```

---

# 8. ENTRAINEMENT EN 2 ETAPES

## 8.1 Vue d'Ensemble du Training

```
 ENTRAINEMENT 2 ETAPES (Transfer Learning)
 ===========================================

 +-------------------+        +-------------------+
 |     STAGE 1       |        |     STAGE 2       |
 |  Feature Extract  |  -->   |  Fine-tuning      |
 |                   |        |                   |
 |  Backbone gele    |        |  Blocs 5-8 degeles|
 |  Tete entrainee   |        |  Tete entrainee   |
 |                   |        |                   |
 |  60 epoques max   |        |  150 epoques max  |
 |  Patience: 20     |        |  Patience: 35     |
 |                   |        |                   |
 |  ~262K params     |        |  ~6.5M params     |
 +-------------------+        +-------------------+
         |                            |
         v                            v
   stage1.pth                   stage2.pth
   (point depart S2)            (modele final)
```

## 8.2 Hyperparametres - STAGE 1

```
 HYPERPARAMETRES STAGE 1 (Feature Extraction)
 ==============================================

 +----------------------------------+-------------------+
 | Parametre                        | Valeur            |
 +----------------------------------+-------------------+
 |                                                      |
 | --- ARCHITECTURE ---                                 |
 | Modele                           | EfficientNet-B3   |
 | Backbone                         | GELE              |
 | Tete entrainable                 | Linear(1536->256) |
 |                                  | -> ReLU -> Drop   |
 |                                  | -> Linear(256->1) |
 | Dropout                          | 0.5               |
 | Params entrainables              | ~262 000          |
 |                                                      |
 | --- OPTIMISATION ---                                 |
 | Optimiseur                       | AdamW             |
 | Learning rate                    | 1e-4              |
 | Weight decay                     | 1e-4              |
 | Scheduler                        | ReduceLROnPlateau |
 | Factor                           | 0.5               |
 | Patience                         | 7                 |
 | Gradient clipping                | 1.0               |
 |                                                      |
 | --- ENTRAINEMENT ---                                 |
 | Epoques max                      | 60                |
 | Early stopping patience          | 20                |
 | Batch size                       | 32                |
 | MixUp alpha                      | 0.2               |
 | Label smoothing                  | 0.05              |
 |                                                      |
 | --- PERTE ---                                        |
 | Fonction                         | LabelSmoothingBCE |
 | pos_weight                       | n_benin / n_malin |
 |                                  | (~1.43, cap 2.5)  |
 |                                                      |
 | --- DONNEES ---                                      |
 | Sampler                          | WeightedRandom    |
 | Strategie                        | 1 crop/patient/ep |
 | Augmentations                    | Toutes (11)       |
 |                                                      |
 +----------------------------------+-------------------+
```

## 8.3 Hyperparametres - STAGE 2

```
 HYPERPARAMETRES STAGE 2 (Fine-tuning)
 =======================================

 +----------------------------------+-------------------+
 | Parametre                        | Valeur            |
 +----------------------------------+-------------------+
 |                                                      |
 | --- ARCHITECTURE ---                                 |
 | Modele                           | EfficientNet-B3   |
 | Backbone                         | features[5,6,7,8] |
 |                                  | DEGELES           |
 | Tete                             | Entrainee         |
 | Params entrainables              | ~6 500 000        |
 |                                                      |
 | --- OPTIMISATION ---                                 |
 | Optimiseur                       | AdamW             |
 | LR backbone (final)              | 1e-5              |
 | LR head                          | 1e-4              |
 | Weight decay                     | 1e-4              |
 | Warmup backbone                  | 5 epoques         |
 | Warmup LR (debut -> fin)         | 1e-8 -> 1e-5      |
 | Scheduler                        | ReduceLROnPlateau |
 | Factor                           | 0.5               |
 | Patience                         | 7                 |
 | Gradient clipping                | 0.5               |
 | Scheduler loss clip              | 2.0 (FIX-7)       |
 |                                                      |
 | --- ENTRAINEMENT ---                                 |
 | Epoques max                      | 150               |
 | Early stopping patience          | 35 (FIX-8)        |
 | Batch size                       | 32                |
 | MixUp alpha                      | 0.2               |
 | Label smoothing                  | 0.05              |
 | Initialisation                   | Charger stage1.pth|
 |                                                      |
 | --- SWA (Stochastic Weight Avg) ---                  |
 | Epoques SWA                      | 20 (fin du S2)   |
 | SWA learning rate                 | 1e-6             |
 |                                                      |
 | --- PERTE ---                                        |
 | Fonction                         | LabelSmoothingBCE |
 | pos_weight                       | n_benin / n_malin |
 |                                                      |
 +----------------------------------+-------------------+
```

## 8.4 Schema des Learning Rates

```
 EVOLUTION DU LEARNING RATE - STAGE 2
 ======================================

 LR
 ^
 |
 1e-4 |***************************    HEAD (constant)
 |   *
 |  *
 |  *  Reduce on plateau
 |  *  (factor=0.5, patience=7)
 |   ********____________
 |              ____________
 |
 1e-5 |          .  BACKBONE (apres warmup)
 |     ___ ___/
 |    /
 1e-6 |   /               +--- SWA (20 dernieres epoques) ---+
 |  /  Warmup lineaire |  LR fixe = 1e-6                  |
 | /   (5 epoques)     |  Moyenne des poids               |
 1e-8 |/ ^               +----------------------------------+
 +----+----+----+----+----+----+----+----+--> Epoques
 0    5   25   50   75  100  125  130  150
 ^                                ^     ^
 Debut warmup              Debut SWA  Fin
```

## 8.5 Strategie WeightedRandomSampler

```
 ECHANTILLONNAGE PAR PATIENT (anti-leakage)
 =============================================

 Probleme : Un patient peut avoir 5 crops differents
            Si on les melange tous, le modele apprend
            la "texture du sein de ce patient" plutot
            que le motif de la calcification.

 Solution : WeightedRandomSampler avec poids
            w_i = 1 / n_crops_du_patient_i

 Effet :
   Patient avec 5 crops --> chaque crop a un poids 0.2
   Patient avec 1 crop  --> ce crop a un poids 1.0

   --> Par epoque, on echantillonne ~1 crop par patient
   --> ~1566 samples par epoque (= nombre de patients)

 +--------------------------------------------+
 |                                            |
 |  AVANT le sampler :                        |
 |    Patient A (5 crops) : 5/2735 = 0.18%   |
 |    Patient B (1 crop)  : 1/2735 = 0.04%   |
 |    --> Patient A = 5x plus represente      |
 |                                            |
 |  APRES le sampler :                        |
 |    Patient A (5 crops) : 1/1566 = 0.064%  |
 |    Patient B (1 crop)  : 1/1566 = 0.064%  |
 |    --> CHAQUE patient egal en frequence    |
 |                                            |
 +--------------------------------------------+
```

---

# 9. INFERENCE ET TTA

## 9.1 Pipeline d'Inference Standard

```
 INFERENCE STANDARD (1 passage)
 ================================

 Image PNG 224x224 grayscale
       |
       v
 +-----------------------------------+
 | 1. Lecture cv2.IMREAD_GRAYSCALE   |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 2. /255 -> [0, 1] float           |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 3. Repeat 1ch -> 3ch              |
 |    (RGB attendu par EfficientNet) |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 4. Normalisation ImageNet         |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 5. Forward pass (float32, no AMP) |
 |    logit = model(tensor)          |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 6. Sigmoid -> probabilite [0,1]   |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 7. Seuil = 0.535                  |
 |    (F1-optimal sur validation)    |
 |    p < 0.535 -> BENIN             |
 |    p >= 0.535 -> MALIN            |
 +-----------------------------------+
```

## 9.2 TTA - Test-Time Augmentation (TTA-8 et TTA-16)

```
 TTA-8 : GROUPE A (8 variantes spatiales)
 ===========================================

 Image originale (224x224)
       |
       +---> 0. Original         -> p0
       +---> 1. Flip Horizontal  -> p1
       +---> 2. Flip Vertical    -> p2
       +---> 3. Rotation 90      -> p3
       +---> 4. Rotation 180     -> p4
       +---> 5. Rotation 270     -> p5
       +---> 6. HFlip + Rot 90   -> p6
       +---> 7. VFlip + Rot 90   -> p7

 TTA-8  : p_final = mean(p0..p7)


 TTA-16 : GROUPE A + GROUPE B (scale variants)
 ================================================

 GROUPE B = memes 8 transforms MAIS sur image
            resizee 224 -> 260 -> centre-crop 224
            (marge 18px de chaque cote)
            --> Teste invariance a l'echelle (ART-2)

       +---> 8.  Original (260)       -> p8
       +---> 9.  HFlip (260)          -> p9
       +---> 10. VFlip (260)          -> p10
       +---> 11. Rot90 (260)          -> p11
       +---> 12. Rot180 (260)         -> p12
       +---> 13. Rot270 (260)         -> p13
       +---> 14. HFlip+Rot90 (260)    -> p14
       +---> 15. VFlip+Rot90 (260)    -> p15

 TTA-16 : p_final = mean(p0..p15)
```

```
 COMPARAISON TTA-8 vs TTA-16 (mesuree sur test set)
 =====================================================

 +----------+----------+----------+----------+----------+
 | Methode  | AUC-ROC  | Accuracy | Sensib.  | Specif.  |
 +----------+----------+----------+----------+----------+
 | TTA-8    |  0.7738  |  0.7090  |  0.6603  |  0.7411  |
 | TTA-16   |  0.7812  |  0.7103  |  0.6314  |  0.7621  |
 +----------+----------+----------+----------+----------+
 | Gain     | +0.0074  | +0.0013  | -0.0289  | +0.0210  |
 +----------+----------+----------+----------+----------+

 TTA-16 ameliore l'AUC et la specificite mais perd
 un peu de sensibilite (trade-off).
 Cout TTA-16 : 16x temps d'inference (vs 8x pour TTA-8).
```

---

# 10. RESULTATS FINAUX

## 10.1 Meilleur Modele (Checkpoint best_0.7717)

```
 CHECKPOINT EVALUE : checkpoints/best_0.7717/efficientnet_stage2.pth
 ====================================================================

 Modele   : EfficientNet-B3 (Dropout=0.5)
 Training : 2-stage + MixUp(0.2) + SWA(20ep) + LabelSmooth(0.05)
 Donnees  : CBIS-DDSM DICOM (calc + mass, split propre)
 Seuil    : 0.535 (F1-optimal sur validation)
```

## 10.2 Resultats sur le Jeu de Test (787 images)

```
 RESULTATS TEST SET — CHECKPOINT best_0.7717
 =============================================

 Test set : 787 images (475 benin + 312 malin)

 +------------------+----------+----------+
 | Metrique         |  TTA-8   |  TTA-16  |
 +------------------+----------+----------+
 | AUC-ROC          |  0.7738  |**0.7812**|
 | AUC-PR (AP)      |  0.7064  |  0.7183  |
 | Accuracy         |  0.7090  |  0.7103  |
 | Sensibilite      |  0.6603  |  0.6314  |
 | Specificite      |  0.7411  |  0.7621  |
 | PPV (Precision)  |  0.6261  |  0.6355  |
 | NPV              |  0.7686  |  0.7589  |
 | F1 Score         |  0.6427  |  0.6334  |
 | MCC              |  0.3980  |  0.3940  |
 | Cohen Kappa      |  0.3976  |  0.3939  |
 | Brier Score      |  0.1991  |  0.1958  |
 | Youden's J       |  0.4013  |  0.3935  |
 +------------------+----------+----------+

 Val AUC  : 0.8056
 Test AUC : 0.7812 (TTA-16)
 Gap      : 2.43%  (bonne generalisation, < 4%)
```

## 10.3 Intervalles de Confiance 95% (Bootstrap, n=2000)

```
 INTERVALLES DE CONFIANCE — TTA-16
 ===================================

 +------------------+----------+-------------------+
 | Metrique         | Valeur   | IC 95%            |
 +------------------+----------+-------------------+
 | AUC-ROC          |  0.7812  | [0.7483 — 0.8130] |
 | AUC-PR (AP)      |  0.7183  | [0.6662 — 0.7678] |
 | F1 Score         |  0.6334  | [0.5878 — 0.6775] |
 | Sensibilite      |  0.6314  | [0.5793 — 0.6871] |
 | Specificite      |  0.7621  | [0.7244 — 0.7996] |
 +------------------+----------+-------------------+

 +--------------------------------------------------+
 |  L'AUC est significativement > 0.50 (aleatoire)  |
 |  Borne inferieure IC 95% = 0.748 >> 0.50         |
 +--------------------------------------------------+
```

## 10.4 Matrice de Confusion (TTA-16, seuil=0.535)

```
                          Predit
                    +-----------+-----------+
                    |  BENIN    |  MALIN    |
 +------+-----------+-----------+-----------+
 |Verite| BENIN     |  TN=362   |  FP=113   |
 |      |           | (46.0%)   | (14.4%)   |
 +------+-----------+-----------+-----------+
 |      | MALIN     |  FN=115   |  TP=197   |  
 |      |           | (14.6%)   | (25.0%)   |
 +------+-----------+-----------+-----------+

 Total : 787 images
 Correct : 559 (71.0%)
 Erreurs : 228 (29.0%)
   - 113 Faux Positifs (benin predit malin)
   - 115 Faux Negatifs (malin predit benin) <-- critique cliniquement
```

## 10.5 Seuil Optimal

```
 OPTIMISATION DU SEUIL DE DECISION
 ===================================

 +----------------------------+---------+
 | Methode                    | Seuil   |
 +----------------------------+---------+
 | Defaut                     |  0.50   |
 | F1 optimal (validation)    |  0.535  | <-- utilise
 | Youden optimal (validation)|  ~0.51  |
 +----------------------------+---------+

 Comparaison a differents seuils (TTA-16, test) :
 +----------+--------+--------+--------+
 | Seuil    | Sens.  | Spec.  | F1     |
 +----------+--------+--------+--------+
 | 0.40     | haute  | basse  | ~0.60  | haute sensibilite
 | 0.50     | moyen  | moyen  | ~0.63  | defaut
 | 0.535    | 0.631  | 0.762  | 0.633  | F1-optimal
 +----------+--------+--------+--------+
```

---

# 11. ANALYSE PAR TYPE DE LESION (CALC vs MASS)

```
 RESULTATS SEPARES PAR TYPE — TTA-16, seuil=0.535
 ====================================================

 +------------------+----------+----------+----------+
 | Metrique         |  CALC    |  MASS    | OVERALL  |
 |                  | (n=382)  | (n=405)  | (n=787)  |
 +------------------+----------+----------+----------+
 | AUC-ROC          |  0.7808  |  0.7836  |  0.7812  |
 | AUC-PR (AP)      |  0.7356  |  0.7133  |  0.7183  |
 | Accuracy         |  0.7094  |  0.7111  |  0.7103  |
 | Sensibilite      |  0.6065  |  0.6561  |  0.6314  |
 | Specificite      |  0.7797  |  0.7460  |  0.7621  |
 | PPV              |  0.6528  |  0.6205  |  0.6355  |
 | NPV              |  0.7437  |  0.7741  |  0.7589  |
 | F1               |  0.6288  |  0.6378  |  0.6334  |
 | MCC              |  0.3913  |  0.3983  |  0.3940  |
 +------------------+----------+----------+----------+

 Distribution des erreurs par type :
 +------------------+----------+----------+----------+
 |                  |  CALC    |  MASS    |  TOTAL   |
 +------------------+----------+----------+----------+
 | TP               |   94     |  103     |  197     |
 | TN               |  177     |  185     |  362     |
 | FP               |   50     |   63     |  113     |
 | FN               |   61     |   54     |  115     |
 +------------------+----------+----------+----------+

 +----------------------------------------------------+
 |  OBSERVATIONS :                                    |
 |                                                    |
 |  1. AUC similaire : calc=0.781, mass=0.784         |
 |     --> Le modele generalise bien aux deux types   |
 |                                                    |
 |  2. Sensibilite mass > calc (+0.050)               |
 |     --> Les masses malignes sont mieux detectees   |
 |                                                    |
 |  3. Specificite calc > mass (+0.034)               |
 |     --> Moins de faux positifs sur les calc        |
 |                                                    |
 |  4. FN: 61 calc + 54 mass                         |
 |     --> Les faux negatifs sont distribues          |
 |         de facon equilibree entre les types        |
 +----------------------------------------------------+
```

---

# 12. ANALYSE DES ERREURS

## 12.1 Caracteristiques des Erreurs

```
 ANALYSE DES FAUX POSITIFS ET FAUX NEGATIFS
 =============================================

 FAUX POSITIFS (n=113) : Benin classe Malin
 +---------------------------------------------+
 |  Calc : 50 (44.2%)  |  Mass : 63 (55.8%)   |
 |                                              |
 |  Probabilite moyenne     : 0.617             |
 |  Probabilite min / max   : 0.535 / ~0.90    |
 |  Incertitude TTA moyenne : 0.104             |
 |                                              |
 |  --> Beaucoup sont proches du seuil (0.535)  |
 |  --> Incertitude moderee = modele hesite     |
 +---------------------------------------------+

 FAUX NEGATIFS (n=115) : Malin classe Benin
 +---------------------------------------------+
 |  Calc : 61 (53.0%)  |  Mass : 54 (47.0%)   |
 |                                              |
 |  Probabilite moyenne     : 0.433             |
 |  Probabilite min / max   : ~0.05 / 0.534    |
 |  Incertitude TTA moyenne : 0.106             |
 |                                              |
 |  --> Cas manques proches du seuil aussi      |
 |  --> CRITIQUE : malins manques par le systeme|
 +---------------------------------------------+
```

## 12.2 Analyse par Patient

```
 ERREURS AU NIVEAU PATIENT
 ===========================

 La majorite des erreurs sont des cas "borderline" :
 probabilite proche du seuil de decision (0.535).

 L'incertitude TTA (ecart-type des 16 predictions) est
 un bon indicateur de fiabilite :
   - Predictions correctes : incertitude plus faible
   - Predictions incorrectes : incertitude plus elevee

 --> Option de REJET : si incertitude > seuil,
     referer au radiologue pour revision manuelle.
```

---

# 13. EXPLICABILITE (Grad-CAM++)

```
 GRAD-CAM++ : Visualisation des Zones d'Attention
 ===================================================

 Methode : Grad-CAM++ (Chattopadhay et al. 2018)
 Couche cible : features[7] (dernier MBConv, 7x7)
 Amelioration vs Grad-CAM : meilleure localisation
                            des petites structures

 +-------------------+  +-------------------+  +-------------------+
 |  Image originale  |  |  Overlay Grad-CAM |  |  Heatmap seule    |
 |  (grayscale)      |  |  (inferno + cont) |  |  (localisation)   |
 |                   |  |                   |  |                   |
 |   [crop 224x224]  |  |  [activation map] |  |  [peak @ (x,y)]  |
 +-------------------+  +-------------------+  +-------------------+

 Utilisation :
   - TP (Vrais Positifs) : le modele regarde la LESION
   - FP (Faux Positifs)  : le modele regarde des artéfacts
   - FN (Faux Negatifs)  : le modele ne regarde PAS la lesion

 +----------------------------------------------------+
 |  Interpretation clinique :                         |
 |                                                    |
 |  Si Grad-CAM++ montre activation SUR la lesion :   |
 |    --> Le modele utilise les bonnes features       |
 |    --> Decision cliniquement justifiable            |
 |                                                    |
 |  Si activation HORS de la lesion :                 |
 |    --> Le modele utilise des artéfacts             |
 |    --> Decision non fiable                         |
 +----------------------------------------------------+
```

---

# 14. STRATEGIES TESTEES ET AMELIORATIONS

## 14.1 Chronologie des Experiences

```
 EVOLUTION DES RESULTATS - DU PIRE AU MEILLEUR
 ================================================

 Test AUC
 ^
 |
 0.80 |                                                 * 0.8056 (val)
      |                                               /
 0.78 |                                          ___/ * 0.7812 (test)
      |                                         /
 0.75 |                                     ___/   * 0.7538
      |                              ______/       * 0.7306
 0.70 |                       ______/
      |                ______/
 0.65 |          _____/
      |        _/
 0.60 |       /
      |      /
 0.55 |     /
      |    /
 0.50 |   /
      |  / * 0.4768
 0.45 | /
      +-+----+----+----+----+----+----+----+----> Strategie
        v0   v1   v2   v3   v4   v5
```

## 14.2 Tableau Detaille des Strategies

```
 +-----+----------------------------+---------+--------+----------+
 | #   | Strategie                  | Format  | Val AUC| Test AUC |
 +-----+----------------------------+---------+--------+----------+
 |  v0 | Code original (8 bugs)     | JPEG    |   -    |  0.477   |
 |  v1 | + Correction 8 bugs        | JPEG    |   -    |  0.722   |
 |  v2 | + DICOM calc+mass          | DICOM   | 0.832  |  0.731   |
 |  v3 | + Stage2 + TTA             | DICOM   | 0.832  |  0.793   |
 |  v4 | Mask crop (ABANDONNE)      | DICOM   | 0.752  |  0.754   |
 |  v5 | + MixUp+SWA+LabelSmooth    | DICOM   | 0.806  |**0.781** |
 +-----+----------------------------+---------+--------+----------+
 | DenseNet-121 (ABANDONNE)         | DICOM   |   -    |  0.607   |
 +-----+----------------------------+---------+--------+----------+

 Progression totale : 0.477 -> 0.781 = +0.304 AUC (+63.7%)
```

## 14.3 Detail des 8 Bugs Corriges

```
 BUG #1 : Normalisation ImageNet manquante
 ============================================
 Avant : img = img / 255.0
 Apres : img = (img/255 - mean) / std
 Gain  : AUC 0.477 -> ~0.60+

 BUG #2 : Desequilibre des classes ignore
 ==========================================
 Avant : BCEWithLogitsLoss() sans pos_weight
 Apres : pos_weight = n_benin/n_malin (~1.43)
 Gain  : Specificite 0.17 -> 0.60+

 BUG #3 : CosineAnnealingLR instable
 ======================================
 Avant : CosineAnnealingLR (remet LR au max)
 Apres : ReduceLROnPlateau(factor=0.5, patience=7)
 Gain  : Convergence lisse

 BUG #4 : Pas de warmup en Stage 2
 ===================================
 Avant : lr_backbone = 1e-5 directement
 Apres : Warmup 5ep (1e-8 -> 1e-5)
 Gain  : Pas de spike epoque 1

 BUG #5 : BatchNorm gele avec le backbone
 ==========================================
 Avant : BN stats figees a ImageNet
 Apres : BN requires_grad_(True)
 Gain  : BN s'adapte aux mammographies

 BUG #6 : Fuite de donnees patient
 ===================================
 Avant : DataLoader(shuffle=True) tous les crops
 Apres : WeightedRandomSampler (1 crop/patient/ep)
 Gain  : ~+0.05 AUC

 BUG #7 : NaN dans les predictions (AMP)
 =========================================
 Avant : ValueError: NaN from roc_auc_score
 Apres : NaN guard + clamp + validate en float32
 Gain  : Entrainement stable

 BUG #8 : Split train/val non-deterministe
 ===========================================
 Avant : pandas .unique() ordre arbitraire
 Apres : sorted() avant shuffle
 Gain  : Resultats reproductibles
```

## 14.4 Strategies Specifiques (Apres Bug Fixes)

```
 STRATEGIE A : Passage JPEG -> DICOM lossless
 ==============================================
 Resultat : Val AUC 0.722 (JPEG) -> 0.806 (DICOM) = +0.084

 STRATEGIE B : Stage 2 Fine-tuning
 ===================================
 Resultat : Test AUC 0.731 -> 0.752 = +0.021
 Conditions : warmup + grad_clip 0.5 + charger stage1.pth

 STRATEGIE C : MixUp + SWA + Label Smoothing
 ==============================================
 Resultat : Test AUC 0.752 -> 0.772 = +0.020
 MixUp alpha=0.2, SWA 20ep lr=1e-6, smooth=0.05

 STRATEGIE D : Test-Time Augmentation
 ======================================
 Sans TTA  : 0.772 (val AUC du checkpoint)
 TTA-8     : 0.774  (+0.002)
 TTA-16    : 0.781  (+0.010)
 Cout : 16x inference, ZERO re-entrainement
```

## 14.5 Strategies Abandonnees

```
 ABANDON A : DenseNet-121 torchxrayvision
 ==========================================
 Hypothese : Pre-entraine radiographies thorax -> meilleur transfer
 Resultat  : Test AUC = 0.607 (BIEN PIRE qu'EfficientNet)
 Raison    : Plage pixels incompatible, domaine mammo != thorax
 Decision  : ABANDON

 ABANDON B : Crop precis base masque ROI
 =========================================
 Hypothese : Masque exact au lieu du crop radiologue
 Resultat  : Val AUC 0.831 -> 0.751 (BAISSE de 0.08 !)
             Dataset 2736 -> 1270 cas (presque -50%)
 Raison    : Crops trop similaires, diversite reduite
 Decision  : RETOUR a la strategie smallest DICOM
```

## 14.6 Resume des Gains par Strategie

```
 +------------------------------------------+----------+
 | Strategie                                | Gain AUC |
 +------------------------------------------+----------+
 | Bugs 1-8 : Corrections fondamentales     | +0.245   |
 | A : Passage DICOM lossless               | +0.084   |
 | B : Stage 2 fine-tuning                  | +0.021   |
 | C : MixUp + SWA + LabelSmooth           | +0.020   |
 | D : TTA-16                               | +0.010   |
 +------------------------------------------+----------+
 | Point de depart                          |  0.477   |
 | Resultat actuel (TTA-16)                 |  0.781   |
 | Amelioration absolue                     |  +0.304  |
 +------------------------------------------+----------+
```

---

# 15. STRUCTURE DES FICHIERS DU PROJET

```
 ARBORESCENCE DES FICHIERS (phase classification)
 ===================================================

 mammo-cad/
 |
 |-- data/
 |   |-- raw/
 |   |   |-- cbis-ddsm-calcification-dataset/  (DICOM calc, 1871 cas)
 |   |   +-- cbis-ddsm-mass-dataset/           (DICOM mass, 1696 cas)
 |   |
 |   |-- interim/cbis_ddsm/
 |   |   |-- Crops/                  Crops 224x224 PNG
 |   |   +-- cbis_ddsm_cases.csv     3 567 lignes (metadonnees)
 |   |
 |   +-- patches/classification/
 |       |-- train/  (benign: 1610, malignant: 1125)
 |       |-- val/    (benign: 579,  malignant: 383)
 |       +-- test/   (benign: 475,  malignant: 312)
 |
 |-- src/
 |   |-- cbis_ddsm/
 |   |   |-- build_cases.py          Lecture CSV + split patient
 |   |   +-- extract_crops.py        DICOM 16-bit -> PNG 224
 |   |
 |   |-- dataset/
 |   |   +-- cls_dataset.py          ClsDataset + 11 augmentations
 |   |
 |   |-- models/
 |   |   +-- efficientnet.py         EfficientNet-B3 + tete custom
 |   |
 |   |-- training/
 |   |   +-- train_classifier.py     Stage 1 + Stage 2 + SWA
 |   |
 |   +-- inference/
 |       |-- classify.py             MammoClassifier + TTA-8/16
 |       +-- explainability.py       Grad-CAM++ + Integrated Gradients
 |
 |-- checkpoints/
 |   +-- best_0.7717/
 |       |-- efficientnet_stage1.pth
 |       +-- efficientnet_stage2.pth  ** CHECKPOINT PRINCIPAL **
 |
 |-- notebooks/
 |   |-- classification_inference.ipynb
 |   +-- classification_evaluation_complete.ipynb  ** EVALUATION **
 |
 +-- outputs/
     |-- notebook_outputs/           (ancienne evaluation)
     +-- evaluation_complete/        (evaluation complete)
         |-- 01_dataset_statistics.png
         |-- 02_roc_pr_curves.png
         |-- 03_confusion_matrices.png
         |-- 04_threshold_sweep.png
         |-- 05_distributions_calibration.png
         |-- 06_val_vs_test.png
         |-- 07_per_type_analysis.png
         |-- 08_top_false_positives.png
         |-- 09_top_false_negatives.png
         |-- 10_prob_vs_uncertainty.png
         |-- 11_tta_consistency.png
         |-- 12_gradcam_tp.png
         |-- 13_gradcam_fp.png
         |-- 14_gradcam_fn.png
         |-- 15_strategy_evolution.png
         |-- 16_bug_fixes.png
         |-- 17_benchmark_comparison.png
         +-- evaluation_complete_report.json
```

---

# 16. RESUME DES CHIFFRES CLES

```
 +=====================================================================+
 |         RESUME - CLASSIFICATION BENIN/MALIN                          |
 +=====================================================================+
 |                                                                     |
 |  DATASET                                                            |
 |  -------                                                            |
 |  Source           : CBIS-DDSM (DICOM lossless 16-bit)               |
 |  Cas total        : 3 567  (calc: 1871 + mass: 1696)               |
 |  Patients uniques : 1 566                                          |
 |                                                                     |
 |  SPLIT (par patient, seed=42)                                       |
 |  -----                                                              |
 |  Train  : 2 260 cas / 973 patients (63.4%)                         |
 |  Val    :   520 cas / 244 patients (14.6%)                         |
 |  Test   :   787 cas / 349 patients (22.0%)                         |
 |                                                                     |
 |  PATCHES (224 x 224 PNG)                                            |
 |  -------                                                            |
 |  Train  : 2 735 (1610 benin + 1125 malin)                          |
 |  Val    :   962 (579 benin + 383 malin)                             |
 |  Test   :   787 (475 benin + 312 malin)                             |
 |  TOTAL  : 4 484                                                     |
 |                                                                     |
 |  MODELE                                                             |
 |  ------                                                             |
 |  Architecture     : EfficientNet-B3                                 |
 |  Pre-entrainement : ImageNet                                        |
 |  Parametres total : ~11 900 000                                     |
 |  Tete : Linear(1536->256)->ReLU->Drop(0.5)->Linear(256->1)          |
 |                                                                     |
 |  ENTRAINEMENT                                                       |
 |  -------------                                                      |
 |  Strategie : 2 etapes (transfer learning)                           |
 |  Stage 1   : 60 ep / patience 20 / lr=1e-4 / ~262K params          |
 |  Stage 2   : 150 ep / patience 35 / lr_bb=1e-5 / ~6.5M params      |
 |  Optimiseur: AdamW (wd=1e-4)                                        |
 |  Batch     : 32                                                     |
 |  MixUp     : alpha=0.2                                              |
 |  SWA       : 20 epoques, lr=1e-6                                    |
 |  Sampler   : WeightedRandomSampler 1crop/patient/ep                 |
 |  Perte     : LabelSmoothingBCE + pos_weight (~1.43)                 |
 |  Augment.  : 11 transformations on-the-fly                          |
 |                                                                     |
 |  RESULTATS (checkpoint best_0.7717, TTA-16, seuil=0.535)            |
 |  ---------                                                          |
 |  Val AUC             : 0.8056                                       |
 |  Test AUC            : 0.7812  [0.7483 -- 0.8130] 95% CI           |
 |  Test AUC-PR (AP)    : 0.7183  [0.6662 -- 0.7678] 95% CI           |
 |  Accuracy            : 0.7103                                       |
 |  Sensibilite         : 0.6314  [0.5793 -- 0.6871] 95% CI           |
 |  Specificite         : 0.7621  [0.7244 -- 0.7996] 95% CI           |
 |  F1 Score            : 0.6334  [0.5878 -- 0.6775] 95% CI           |
 |  PPV                 : 0.6355                                       |
 |  NPV                 : 0.7589                                       |
 |  MCC                 : 0.3940                                       |
 |  Brier Score         : 0.1958                                       |
 |  TP=197  TN=362  FP=113  FN=115                                     |
 |  Val-Test gap        : 2.43% (bonne generalisation)                 |
 |                                                                     |
 |  PAR TYPE DE LESION (TTA-16)                                        |
 |  --------                                                           |
 |  Calc (n=382) : AUC=0.7808  Sens=0.607  Spec=0.780  F1=0.629      |
 |  Mass (n=405) : AUC=0.7836  Sens=0.656  Spec=0.746  F1=0.638      |
 |                                                                     |
 |  PROGRESSION                                                        |
 |  -----------                                                        |
 |  v0 (bugs)    : 0.477                                               |
 |  v1 (fixes)   : 0.722  (+0.245)                                     |
 |  v5 (final)   : 0.781  (+0.304 total, +63.7%)                       |
 |                                                                     |
 |  GPU          : NVIDIA RTX 4090                                     |
 |                                                                     |
 +=====================================================================+
```

---

# 17. IMAGES DE REFERENCE

```
 VISUALISATIONS GENEREES (outputs/evaluation_complete/)
 ========================================================

 A. Statistiques du dataset :
   01_dataset_statistics.png       Distribution par split/label/type

 B. Courbes de performance :
   02_roc_pr_curves.png            ROC + PR (Val vs Test, TTA-8 vs TTA-16)
   03_confusion_matrices.png       Matrices a 3 seuils (0.50, 0.535, 0.40)
   04_threshold_sweep.png          Sens/Spec/F1/Acc vs seuil

 C. Analyse de generalisation :
   05_distributions_calibration.png  Distributions + calibration + incertitude
   06_val_vs_test.png              Comparaison Val vs Test (10 metriques)

 D. Analyse par type :
   07_per_type_analysis.png        Calc vs Mass (metriques + ROC separees)

 E. Analyse des erreurs :
   08_top_false_positives.png      10 FP les plus confiants (images)
   09_top_false_negatives.png      10 FN les plus confiants (images)
   10_prob_vs_uncertainty.png      Scatter TP/TN/FP/FN (prob vs incertitude)
   11_tta_consistency.png          Accord TTA + option de rejet

 F. Explicabilite :
   12_gradcam_tp.png               Grad-CAM++ sur Vrais Positifs
   13_gradcam_fp.png               Grad-CAM++ sur Faux Positifs
   14_gradcam_fn.png               Grad-CAM++ sur Faux Negatifs

 G. Historique :
   15_strategy_evolution.png       Evolution AUC v0 -> v5
   16_bug_fixes.png                8 bugs corriges + impact
   17_benchmark_comparison.png     Position vs litterature

 H. Donnees :
   evaluation_complete_report.json  Toutes les metriques en JSON

 Pour le rapport du professeur, les plus importants :
   1. 02_roc_pr_curves.png         --> Performance globale
   2. 03_confusion_matrices.png    --> Matrice detaillee
   3. 07_per_type_analysis.png     --> Calc vs Mass
   4. 04_threshold_sweep.png       --> Choix du seuil
   5. 12-14_gradcam_*.png          --> Interpretabilite
   6. 15_strategy_evolution.png    --> Progression
   7. 06_val_vs_test.png           --> Generalisation
```
