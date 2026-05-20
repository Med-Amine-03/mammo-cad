# Segmentation des Microcalcifications Mammographiques

## Systeme CAD - Phase de Segmentation U-Net sur INbreast

---

# 1. VUE GLOBALE DU PIPELINE

```
 PIPELINE COMPLET DE SEGMENTATION
 =================================

 +------------------+     +------------------+     +------------------+
 |   DONNEES BRUTES |     |  PRE-TRAITEMENT  |     |   EXTRACTION     |
 |   INbreast DB    | --> |  CLAHE + Nettoy. | --> |   DE PATCHES     |
 |   410 DICOM      |     |  313 images      |     |   256x256        |
 +------------------+     +------------------+     +------------------+
                                                          |
                                                          v
 +------------------+     +------------------+     +------------------+
 |   INFERENCE      |     |   ENTRAINEMENT   |     |   DONNEES        |
 |   Fenetre        | <-- |   10-Fold CV     | <-- |   PATCHEES       |
 |   Glissante      |     |   300 epoques    |     |   2601 patches   |
 +------------------+     +------------------+     +------------------+
          |
          v
 +------------------+
 |   RESULTATS      |
 |   AUC-ROC: 0.964 |
 |   Dice:    0.665  |
 +------------------+
```

---

# 2. DATASET INbreast - CHIFFRES EXACTS

## 2.1 Source des Donnees

```
 BASE DE DONNEES INbreast Release 1.0
 ======================================

 Format d'origine : DICOM 16-bit + annotations XML (Apple plist)
 Hopital          : Centro Hospitalar de S. Joao, Porto, Portugal

 +---------------------------------------------+
 |              INbreast Original               |
 |                                              |
 |   Total images DICOM     :  410              |
 |   Total masques XML      :  343              |
 |                                              |
 |   Filtre : Micros = 1 (microcalcifications)  |
 |                                              |
 |   Images avec MC         :  313              |
 |   Images sans MC         :   97  (exclues)   |
 +---------------------------------------------+
```

## 2.2 Repartition des Donnees (Patient-Wise Split)

```
 SPLIT PAR PATIENT (GroupShuffleSplit, seed=42)
 ================================================

 +-----------------------+----------+----------+
 |                       |   ALL    |   TEST   |
 |                       | (CV)     | (Eval.)  |
 +-----------------------+----------+----------+
 | Nombre d'images       |   281    |    32    |
 | Pourcentage           |  89.8%   |  10.2%   |
 | Total pixels MC       | 1392942  |  29680   |
 | Moyenne pixels MC/img |  4957    |   928    |
 +-----------------------+----------+----------+

 IMPORTANT : Le split est fait par PATIENT (pas par image)
             pour eviter toute fuite de donnees (data leakage)
```

## 2.3 Distribution BI-RADS

```
 DISTRIBUTION BI-RADS PAR SPLIT
 ================================

 BI-RADS :  1=Normal  2=Benin  3=Prob.Benin  4=Suspect  5=Malin  6=Confirme

 Split ALL (281 images) :            Split TEST (32 images) :

 BI-RADS 2  : |||||||||||||| 195     BI-RADS 2  : ||||||||  21
 BI-RADS 3  : ||             11      BI-RADS 3  : |          1
 BI-RADS 4a : ||             10      BI-RADS 4a : |          1
 BI-RADS 4b : |               6      BI-RADS 4b :            0
 BI-RADS 4c : |||            14      BI-RADS 4c :            0
 BI-RADS 5  : |||||||        40      BI-RADS 5  : |||        8
 BI-RADS 6  : |               5      BI-RADS 6  : |          1
              ___________  ____                    _______  ___
              Total    =   281                     Total =   32
```

## 2.4 Statistiques des Pixels MC (Microcalcifications)

```
 DISTRIBUTION DES PIXELS MC PAR IMAGE
 ======================================

 Minimum  :        29 pixels   (micro-point, ~5x6 px)
 Median   :       507 pixels
 Moyenne  :     4 545 pixels
 Maximum  :   152 667 pixels   (large cluster dense)

 Taille image typique : ~3000 x 4000 = 12 000 000 pixels
 Ratio MC / fond      : 4545 / 12 000 000 = 0.038%

 +----------------------------------------------------------+
 |  PROBLEME DE DESEQUILIBRE EXTREME                        |
 |                                                          |
 |  Fond (background) : 99.96%  ████████████████████████    |
 |  MC (positif)      :  0.04%  ▏                           |
 |                                                          |
 |  --> Necessite une fonction de perte specialisee         |
 |      (topk_CE avec OHEM, voir Section 6)                |
 +----------------------------------------------------------+
```

---

# 3. PRE-TRAITEMENT DES IMAGES

## 3.1 Pipeline de Pre-traitement

```
 ETAPES DE PRE-TRAITEMENT (preprocessing.py)
 =============================================

 Image DICOM 16-bit         Masque XML plist
       |                          |
       v                          v
 +---------------+         +---------------+
 | Normalisation |         | Parse XML     |
 | 16-bit -> 8-bit|        | Type 19: dots |
 | [0,255] uint8  |        | (cercle r=3)  |
 +---------------+         | Type 15+:     |
       |                   | polygones     |
       v                   +---------------+
 +---------------+                |
 | Flip vers     |                |
 | GAUCHE si     |<---------------+
 | necessaire    |  (aligner image + masque)
 +---------------+
       |
       v
 +---------------------------+
 | Suppression d'artefacts   |
 | - Bord droit  (90%)       |
 | - Bord bas    (90%)       |
 | - Bord haut   (90%)       |
 | - Bande gauche (si bruit) |
 +---------------------------+
       |
       v
 +---------------------------+
 | Plus grande composante    |
 | connexe                   |
 | Kernel: (23,23) MORPH_RECT|
 +---------------------------+
       |
       v
 +---------------------------+
 | Recadrage (crop)          |
 | Supprimer bordures noires |
 +---------------------------+
       |
       v
 +---------------------------+
 | CLAHE                     |
 | clipLimit  = 2.0          |
 | tileGrid   = (8, 8)      |
 +---------------------------+
       |
       v
   Image traitee
   (313 images PNG 8-bit)
```

## 3.2 Parametres CLAHE

```
 CLAHE (Contrast Limited Adaptive Histogram Equalization)
 =========================================================

 +----------------------------------+-------------------+
 | Parametre                        | Valeur            |
 +----------------------------------+-------------------+
 | clipLimit (limite de contraste)  | 2.0               |
 | tileGridSize (taille de grille) | (8, 8) pixels     |
 | Type d'entree                   | uint8 [0, 255]    |
 +----------------------------------+-------------------+

 Objectif : Ameliorer le contraste local pour rendre
            les microcalcifications plus visibles
            sans amplifier le bruit global.

 Avant CLAHE              Apres CLAHE
 +----------------+       +----------------+
 |   Contraste    |       |   Contraste    |
 |   faible,      |  -->  |   ameliore,    |
 |   MC invisibles|       |   MC visibles  |
 +----------------+       +----------------+
```

---

# 4. EXTRACTION DE PATCHES

## 4.1 Processus d'Extraction

```
 EXTRACTION DE PATCHES 256x256 (patch_extraction_seg.py)
 =========================================================

 Image pre-traitee (~3000 x 4000)
       |
       v
 +-----------------------------------+
 | 1. Padding a multiple de 256      |
 |    H' = H + (256 - H%256) % 256   |
 |    W' = W + (256 - W%256) % 256   |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 2. Fenetre glissante              |
 |    Taille   : 256 x 256 pixels    |
 |    Pas      : 256 (sans overlap)  |
 |    ~180 patches par image         |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 3. Filtrage : GARDER UNIQUEMENT   |
 |    les patches POSITIFS           |
 |    (mask.sum() > 0)               |
 |                                   |
 |    Patches negatifs : SUPPRIMES   |
 |    (99.96% du fond = inutile)     |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 4. Resultat final                 |
 +-----------------------------------+

 +=============================================+
 |  CHIFFRES EXACTS DES PATCHES               |
 |                                             |
 |  Split ALL (entr./valid. 10-fold) :         |
 |     Images  : 2 417 patches positifs        |
 |     Masques : 2 417 masques binaires        |
 |                                             |
 |  Split TEST (evaluation finale) :           |
 |     Images  : 184 patches positifs          |
 |     Masques : 184 masques binaires          |
 |                                             |
 |  TOTAL      : 2 601 patches positifs        |
 +=============================================+

 Ratio d'extraction :
   ~313 images x ~180 patches/image = ~56 000 patches potentiels
   Patches positifs gardes           =  2 601
   Taux de retention                 =  ~4.6%
```

## 4.2 Structure des Patches

```
 STRUCTURE DES PATCHES GENERES
 ===============================

 Patch Image (256x256)         Patch Masque (256x256)
 +------------------+          +------------------+
 |                  |          |     (noir=0)     |
 |  Niveaux de gris |          |                  |
 |  [0.0 - 1.0]    |          |   **  (blanc=1)  |
 |  float32         |          |   *** = MC       |
 |  1 canal         |          |   **             |
 |                  |          |     (noir=0)     |
 +------------------+          +------------------+
   Shape: (1, 256, 256)          Shape: (1, 256, 256)

 Tous les patches ont mask.sum() > 0
 = CHAQUE patch contient au moins 1 pixel MC
```

---

# 5. ARCHITECTURE U-Net

## 5.1 Schema Detaille du Modele

```
 ARCHITECTURE U-Net (src/models/unet.py)
 =========================================
 Parametres totaux : 31 037 633 (~31M)

 ENTREE: (Batch, 1, 256, 256)  -- niveaux de gris
           |
 ==========|========= ENCODEUR (Contraction) ==========
           |
           v
 +-------------------+
 | EncoderBlock 1    |    Conv(1->64) + BN + ReLU   x2
 | 64 canaux         |--> skip_1 (B,64,256,256)
 | MaxPool(2)        |
 +-------------------+
           | (B,64,128,128)
           v
 +-------------------+
 | EncoderBlock 2    |    Conv(64->128) + BN + ReLU x2
 | 128 canaux        |--> skip_2 (B,128,128,128)
 | MaxPool(2)        |
 +-------------------+
           | (B,128,64,64)
           v
 +-------------------+
 | EncoderBlock 3    |    Conv(128->256) + BN + ReLU x2
 | 256 canaux        |--> skip_3 (B,256,64,64)
 | MaxPool(2)        |
 +-------------------+
           | (B,256,32,32)
           v
 +-------------------+
 | EncoderBlock 4    |    Conv(256->512) + BN + ReLU x2
 | 512 canaux        |--> skip_4 (B,512,32,32)
 | MaxPool(2)        |
 +-------------------+
           | (B,512,16,16)
           v
 ======== GOULOT D'ETRANGLEMENT (Bottleneck) =========

 +-------------------+
 | ConvBlock         |    Conv(512->1024) + BN + ReLU x2
 | 1024 canaux       |
 +-------------------+
           | (B,1024,16,16)
           v
 ========= DECODEUR (Expansion) =======================

 +-------------------+
 | DecoderBlock 4    |    ConvTranspose2d(1024->512, k=2, s=2)
 | + skip_4 concat   |    Concat skip_4 --> (B,1024,32,32)
 | 512 canaux        |    ConvBlock(1024->512)
 +-------------------+
           | (B,512,32,32)
           v
 +-------------------+
 | DecoderBlock 3    |    ConvTranspose2d(512->256, k=2, s=2)
 | + skip_3 concat   |    Concat skip_3 --> (B,512,64,64)
 | 256 canaux        |    ConvBlock(512->256)
 +-------------------+
           | (B,256,64,64)
           v
 +-------------------+
 | DecoderBlock 2    |    ConvTranspose2d(256->128, k=2, s=2)
 | + skip_2 concat   |    Concat skip_2 --> (B,256,128,128)
 | 128 canaux        |    ConvBlock(256->128)
 +-------------------+
           | (B,128,128,128)
           v
 +-------------------+
 | DecoderBlock 1    |    ConvTranspose2d(128->64, k=2, s=2)
 | + skip_1 concat   |    Concat skip_1 --> (B,128,256,256)
 | 64 canaux         |    ConvBlock(128->64)
 +-------------------+
           | (B,64,256,256)
           v
 +-------------------+
 | Couche de sortie  |    Conv2d(64->1, kernel=1x1)
 | 1 canal (logits)  |    PAS de sigmoid (applique dans la perte)
 +-------------------+
           |
           v
 SORTIE: (Batch, 1, 256, 256)  -- logits bruts
```

## 5.2 Detail du ConvBlock

```
 ConvBlock(in_c, out_c)
 =======================

 Entree (B, in_c, H, W)
    |
    v
 Conv2d(in_c, out_c, kernel=3x3, padding=1, bias=False)
    |
    v
 BatchNorm2d(out_c)
    |
    v
 ReLU(inplace=True)
    |
    v
 Conv2d(out_c, out_c, kernel=3x3, padding=1, bias=False)
    |
    v
 BatchNorm2d(out_c)
    |
    v
 ReLU(inplace=True)
    |
    v
 Sortie (B, out_c, H, W)

 Note: bias=False car BatchNorm absorbe le biais
```

## 5.3 Tableau des Dimensions

```
 DIMENSIONS A CHAQUE ETAPE (Batch=8, Input=256x256)
 =====================================================

 +--------------------+---------+-----------+----------+
 | Couche             | Canaux  | H x W     | Params   |
 +--------------------+---------+-----------+----------+
 | Input              |    1    | 256 x 256 |     -    |
 | Encoder 1          |   64    | 256 x 256 |  38 272  |
 | Pool 1             |   64    | 128 x 128 |     -    |
 | Encoder 2          |  128    | 128 x 128 | 221 440  |
 | Pool 2             |  128    |  64 x  64 |     -    |
 | Encoder 3          |  256    |  64 x  64 | 885 248  |
 | Pool 3             |  256    |  32 x  32 |     -    |
 | Encoder 4          |  512    |  32 x  32 |3 539 968 |
 | Pool 4             |  512    |  16 x  16 |     -    |
 | Bottleneck         | 1024    |  16 x  16 |14 158 848|
 | Decoder 4          |  512    |  32 x  32 |7 080 448 |
 | Decoder 3          |  256    |  64 x  64 |1 770 496 |
 | Decoder 2          |  128    | 128 x 128 |  442 624 |
 | Decoder 1          |   64    | 256 x 256 |  110 784 |
 | Output (Conv 1x1)  |    1    | 256 x 256 |      65  |
 +--------------------+---------+-----------+----------+
 | TOTAL              |         |           |31 037 633|
 +--------------------+---------+-----------+----------+
```

---

# 6. FONCTION DE PERTE : topk_CE (OHEM)

## 6.1 Principe

```
 topk_CE : Online Hard Example Mining (OHEM)
 =============================================

 Probleme : Le desequilibre extreme (0.04% MC vs 99.96% fond)
            fait que la BCE standard est dominee par les pixels
            faciles du fond --> le modele predit tout comme fond.

 Solution : topk_CE garde TOUS les pixels positifs (MC)
            + seulement les TOP-K pixels negatifs les PLUS DIFFICILES

 +----------------------------------------------------------+
 |                                                          |
 |  Pour chaque image du batch :                            |
 |                                                          |
 |  1. Calculer BCE par pixel (reduction="none")            |
 |                                                          |
 |  2. Pixels positifs (MC) :                               |
 |     --> GARDER TOUS (n_pos pixels)                       |
 |                                                          |
 |  3. Pixels negatifs (fond) :                             |
 |     --> Trier par perte decroissante                     |
 |     --> Garder TOP n_keep = min(3 x n_pos, n_neg)       |
 |     --> Ce sont les negatifs les plus "confus"           |
 |                                                          |
 |  4. Si n_pos = 0 : garder top 100 negatifs durs         |
 |                                                          |
 |  5. Moyenne de tous les pixels selectionnes              |
 |                                                          |
 +----------------------------------------------------------+
```

## 6.2 Schema du Fonctionnement

```
 FONCTIONNEMENT topk_CE (neg_ratio = 3)
 ========================================

 Masque GT (256x256)           Carte de perte BCE (256x256)
 +------------------+          +------------------+
 |      0 0 0 0     |          | 0.01 0.02 0.15   |
 |      0 1 1 0     |   BCE    | 0.03 0.89 0.92   |
 |      0 1 0 0     |  ----->  | 0.05 0.78 0.20   |
 |      0 0 0 0     |          | 0.01 0.03 0.88   |
 +------------------+          +------------------+

 Etape 1: Separer positifs et negatifs
 +------------------------------------------+
 | Pixels positifs (GT=1):                  |
 |   n_pos = 3 pixels                       |
 |   Pertes: [0.89, 0.92, 0.78]            |
 |   --> TOUS gardes                        |
 +------------------------------------------+
 | Pixels negatifs (GT=0):                  |
 |   n_neg = 13 pixels                      |
 |   Pertes: [0.01, 0.02, 0.15, 0.03,      |
 |            0.05, 0.20, 0.01, 0.03,       |
 |            0.88, ...]                     |
 +------------------------------------------+

 Etape 2: Selectionner les negatifs durs
 +------------------------------------------+
 | n_keep = min(3 x 3, 13) = 9             |
 | Top-9 negatifs par perte decroissante:   |
 |   [0.88, 0.20, 0.15, 0.05, 0.03,        |
 |    0.03, 0.02, 0.01, 0.01]              |
 +------------------------------------------+

 Etape 3: Perte finale
 +------------------------------------------+
 | Perte = mean([0.89, 0.92, 0.78,         |
 |               0.88, 0.20, 0.15, ...])    |
 +------------------------------------------+
```

## 6.3 Parametres de la Perte

```
 +---------------------------+-------------------+
 | Parametre                 | Valeur            |
 +---------------------------+-------------------+
 | Base loss                 | BCEWithLogitsLoss |
 | reduction                 | "none" (par pixel)|
 | neg_ratio                 | 3                 |
 | Fallback si n_pos = 0     | top 100 negatifs  |
 | Activation dans la perte  | Sigmoid interne   |
 +---------------------------+-------------------+
```

---

# 7. METRIQUES D'EVALUATION

## 7.1 Formules

```
 METRIQUES DE SEGMENTATION (seuil = 0.5)
 ==========================================

 Prediction binaire : pred = sigmoid(logits) > 0.5

                          2 x |pred ∩ GT| + 1
 Dice Coefficient  = ─────────────────────────────
                       |pred| + |GT| + 1

                          |pred ∩ GT| + 1
 IoU (Jaccard)     = ─────────────────────────────
                       |pred ∪ GT| + 1

                               TP + 1
 Sensibilite       = ─────────────────────────
                          TP + FN + 1

 Ou :
   TP = Vrais Positifs (pixels MC correctement detectes)
   FN = Faux Negatifs  (pixels MC manques)
   smooth = 1.0 (evite division par zero)

 Relations :
   Dice = 2 x IoU / (1 + IoU)
   IoU  = Dice / (2 - Dice)
```

## 7.2 AUC-ROC

```
 AUC-ROC (Area Under ROC Curve)
 ================================

 Calculee au niveau PIXEL sur tout le jeu de test :
   - Probabilite predite (continue, 0 a 1) vs label binaire (0 ou 1)
   - Balaye tous les seuils possibles
   - Calcule TPR (sensibilite) et FPR (1-specificite) a chaque seuil

 Interpretation :
   AUC = 0.50 : modele aleatoire
   AUC = 0.80 : bon
   AUC = 0.90 : excellent
   AUC = 0.96 : tres excellent (notre resultat)
   AUC = 1.00 : parfait
```

---

# 8. ENTRAINEMENT : 10-FOLD CROSS-VALIDATION

## 8.1 Schema de la Validation Croisee

```
 10-FOLD CROSS-VALIDATION (KFold, shuffle=True, seed=42)
 =========================================================

 Donnees totales : 2 417 patches positifs (split ALL)
 Methode         : KFold(n_splits=10, shuffle=True, random_state=42)
 Principe        : 9 folds pour TRAIN, 1 fold pour VALIDATION (rotation)
```

```
 SCHEMA DE ROTATION DES FOLDS
 ==============================

 2 417 patches divises en 10 groupes (~242 chacun)

         Groupe: | G1  | G2  | G3  | G4  | G5  | G6  | G7  | G8  | G9  | G10 |
                 | 242 | 242 | 242 | 242 | 242 | 241 | 241 | 241 | 241 | 242 |
 ________________|_____|_____|_____|_____|_____|_____|_____|_____|_____|_____|
 Fold  1 :       | VAL | TR  | TR  | TR  | TR  | TR  | TR  | TR  | TR  | TR  |
 Fold  2 :       | TR  | VAL | TR  | TR  | TR  | TR  | TR  | TR  | TR  | TR  |
 Fold  3 :       | TR  | TR  | VAL | TR  | TR  | TR  | TR  | TR  | TR  | TR  |
 Fold  4 :       | TR  | TR  | TR  | VAL | TR  | TR  | TR  | TR  | TR  | TR  |
 Fold  5 :       | TR  | TR  | TR  | TR  | VAL | TR  | TR  | TR  | TR  | TR  |
 Fold  6 :       | TR  | TR  | TR  | TR  | TR  | VAL | TR  | TR  | TR  | TR  |
 Fold  7 :       | TR  | TR  | TR  | TR  | TR  | TR  | VAL | TR  | TR  | TR  |
 Fold  8 :       | TR  | TR  | TR  | TR  | TR  | TR  | TR  | VAL | TR  | TR  |
 Fold  9 :       | TR  | TR  | TR  | TR  | TR  | TR  | TR  | TR  | VAL | TR  |
 Fold 10 :       | TR  | TR  | TR  | TR  | TR  | TR  | TR  | TR  | TR  | VAL |
                   TR = Train (augmente)    VAL = Validation (PAS augmente)
```

```
 DETAIL PAR FOLD : DONNEES TRAIN / VALIDATION / AUGMENTATION
 ==============================================================

 +------+-------+--------+----------+------------+-----------+
 | Fold | Train | Valid. | Augment. | Variantes  | Samples   |
 |      | (brut)| (brut) | (train)  | possibles  | sur 300   |
 |      |       |        |          | (x16)      | epoques   |
 +------+-------+--------+----------+------------+-----------+
 |  1   | 2 175 |   242  |   OUI    |  34 800    |  652 500  |
 |  2   | 2 175 |   242  |   OUI    |  34 800    |  652 500  |
 |  3   | 2 175 |   242  |   OUI    |  34 800    |  652 500  |
 |  4   | 2 175 |   242  |   OUI    |  34 800    |  652 500  |
 |  5   | 2 175 |   242  |   OUI    |  34 800    |  652 500  |
 |  6   | 2 176 |   241  |   OUI    |  34 816    |  652 800  |
 |  7   | 2 176 |   241  |   OUI    |  34 816    |  652 800  |
 |  8   | 2 176 |   241  |   OUI    |  34 816    |  652 800  |
 |  9   | 2 176 |   241  |   OUI    |  34 816    |  652 800  |
 | 10   | 2 175 |   242  |   OUI    |  34 800    |  652 500  |
 +------+-------+--------+----------+------------+-----------+
 |TOTAL |       |        |          |            | 6 525 200 |
 +------+-------+--------+----------+------------+-----------+

 Calcul :
   Train brut    = 2 417 - ~242 = ~2 175 patches par fold
   Variantes x16 = 2 175 x 16  = 34 800 combinaisons uniques
   Sur 300 ep.   = 2 175 x 300 = 652 500 samples/fold
   (chaque epoque tire 1 variante aleatoire par patch)

 VALIDATION : 242 patches x 1 (PAS d'augmentation) = 242
              --> metriques stables et reproductibles
```

```
 FLUX COMPLET DES DONNEES
 ==========================

 INbreast (410 DICOM)
       |
       | Filtre Micros=1
       v
 313 images MC
       |
       | Split par patient (seed=42)
       v
 +-------------------+     +------------------+
 | ALL : 281 images  |     | TEST : 32 images |
 +-------------------+     +------------------+
       |                          |
       | Extraction patches       | Extraction patches
       | 256x256 positifs         | 256x256 positifs
       v                          v
 +-------------------+     +------------------+
 | 2 417 patches     |     | 184 patches      |
 +-------------------+     +------------------+
       |                          |
       | 10-Fold CV               | Evaluation finale
       v                          | (apres entrainement)
 +-------------------+            |
 | Par fold :        |            |
 |  Train: ~2 175    |            |
 |  + augment x16    |            |
 |  = 34 800 var.    |            |
 |  x 300 epoques    |            |
 |  = 652 500 samples|            |
 |                    |            |
 |  Valid: ~242       |            |
 |  (PAS augmente)   |            |
 +-------------------+            |
       |                          |
       | Meilleur fold (best IoU) |
       v                          v
 +-------------------------------------------+
 | unet_best.pth  --> Evaluer sur 184 TEST   |
 | Resultat final : AUC-ROC = 0.9642         |
 +-------------------------------------------+
```

## 8.2 Hyperparametres d'Entrainement

```
 HYPERPARAMETRES COMPLETS (train_unet.py)
 ==========================================

 +----------------------------------+-------------------+
 | Parametre                        | Valeur            |
 +----------------------------------+-------------------+
 |                                                      |
 | --- ARCHITECTURE ---                                 |
 | Modele                           | U-Net standard    |
 | Canaux d'entree                  | 1 (niv. de gris)  |
 | Canaux de sortie                 | 1 (masque binaire)|
 | Canal de base                    | 64                |
 | Parametres totaux                | 31 037 633        |
 |                                                      |
 | --- OPTIMISATION ---                                 |
 | Optimiseur                       | SGD               |
 | Learning rate                    | 0.001             |
 | Momentum                         | 0.99              |
 | Scheduler                        | MultiStepLR       |
 | Milestone (baisse du LR)         | epoch 150         |
 | Gamma (facteur de reduction)     | 0.1               |
 | LR apres milestone               | 0.0001            |
 |                                                      |
 | --- ENTRAINEMENT ---                                 |
 | Epoques par fold                 | 300               |
 | Batch size                       | 8                 |
 | Nombre de folds                  | 10                |
 | Workers (chargement donnees)     | 4                 |
 | Random seed                      | 42                |
 | Total epoques (10 folds)         | 3 000             |
 |                                                      |
 | --- PERTE ---                                        |
 | Fonction de perte                | topk_CE (OHEM)    |
 | neg_ratio                        | 3                 |
 |                                                      |
 | --- AUGMENTATION (train only) ---                    |
 | Flip horizontal                  | p = 0.5           |
 | Flip vertical                    | p = 0.5           |
 | Rotation                         | {0, 90, 180, 270} |
 |                                                      |
 | --- SAUVEGARDE ---                                   |
 | Critere de sauvegarde            | Meilleur IoU val  |
 | Debug visuel                     | chaque 10 epoques |
 | Checkpoint par fold              | unet_fold{1-10}   |
 +----------------------------------+-------------------+
```

## 8.3 Augmentation des Donnees (On-the-Fly)

```
 AUGMENTATION EN TEMPS REEL (seg_dataset.py)
 =============================================

 L'augmentation est appliquee ALEATOIREMENT a chaque epoque
 pendant le chargement des donnees (PAS sauvegardee sur disque).

 +----------------------------------------------------------+
 |  Patch original (256x256)                                |
 |  +----------+                                            |
 |  |          |                                            |
 |  |  Image   |   Transformations aleatoires :             |
 |  |          |                                            |
 |  +----------+                                            |
 |       |                                                  |
 |       +---> Flip Horizontal ?  (p = 0.5)  --> x2         |
 |       |                                                  |
 |       +---> Flip Vertical ?    (p = 0.5)  --> x2         |
 |       |                                                  |
 |       +---> Rotation ?  {0, 90, 180, 270} --> x4         |
 |                                                          |
 |  Combinaisons possibles : 2 x 2 x 4 = 16 variantes     |
 +----------------------------------------------------------+

 IMPORTANT : L'augmentation est appliquee IDENTIQUEMENT
             sur l'image ET le masque (meme seed aleatoire)
             pour garder l'alignement spatial.

 Exemple pour UN patch :
 +----------+  +----------+  +----------+  +----------+
 | Original |  | Flip H   |  | Rot 90   |  | Flip V + |
 |          |  |          |  |          |  | Rot 270  |
 |  ** .    |  |    . **  |  |    .     |  |     .    |
 |  *  .    |  |    .  *  |  |   **     |  |    **    |
 |     .    |  |    .     |  |   *      |  |     *    |
 +----------+  +----------+  +----------+  +----------+
   1 / 16        2 / 16        5 / 16       12 / 16
```

```
 VOLUME EFFECTIF DE DONNEES D'ENTRAINEMENT
 ============================================

 +-----------------------------------------------+----------+
 | Donnee                                        | Nombre   |
 +-----------------------------------------------+----------+
 | Patches sur disque (split ALL)                | 2 417    |
 | Patches train par fold (~90%)                 | ~2 175   |
 | Patches validation par fold (~10%)            | ~242     |
 +-----------------------------------------------+----------+
 | Variantes possibles par patch                 | 16       |
 | Pool effectif train (2 175 x 16)              | 34 800   |
 +-----------------------------------------------+----------+
 | Epoques par fold                              | 300      |
 | Samples vus par fold (2 175 x 300)            | 652 500  |
 | Samples vus TOTAL (10 folds x 652 500)        | 6 525 000|
 +-----------------------------------------------+----------+

 +----------------------------------------------------------+
 |                                                          |
 |  TRAIN  : augmentation ACTIVEE  (augment=True)           |
 |           --> Chaque epoque voit des variantes            |
 |              differentes du meme patch                    |
 |                                                          |
 |  VALID  : augmentation DESACTIVEE (augment=False)        |
 |           --> Evaluation sur les patches originaux        |
 |              pour des metriques stables et reproductibles |
 |                                                          |
 |  TEST   : augmentation DESACTIVEE (augment=False)        |
 |           --> 184 patches originaux non modifies          |
 |                                                          |
 +----------------------------------------------------------+
```

```
 POURQUOI L'AUGMENTATION ON-THE-FLY ?
 =======================================

 Methode OFFLINE (pas utilisee) :       Methode ON-THE-FLY (utilisee) :
 +---------------------------+          +---------------------------+
 | Generer 16 copies/patch   |          | 1 copie sur disque       |
 | AVANT l'entrainement      |          | Transformer aleatoirement|
 | Stockage: 2175 x 16       |          | A CHAQUE epoque          |
 |         = 34 800 fichiers |          | = 2 175 fichiers         |
 | Disque : ~2.2 Go          |          | Disque : ~150 Mo         |
 | Variabilite : FIXE        |          | Variabilite : INFINIE    |
 +---------------------------+          +---------------------------+
                                              ↑
                                        NOTRE CHOIX
                                        (plus efficace,
                                         plus de diversite)
```

## 8.4 Schema du Learning Rate

```
 EVOLUTION DU LEARNING RATE
 ============================

 LR
 ^
 |
 0.001 |████████████████████████████
 |                            |
 |                            |  MultiStepLR
 |                            |  gamma = 0.1
 |                            |
 0.0001|                            ██████████████████████████
 |                            :
 +----+----+----+----+----+----+----+----+----+-->  Epoques
 0   50  100  150  200  250  300

              milestone=150
              (LR x 0.1)
```

## 8.5 Sanity Check (Verification)

```
 SANITY CHECK AVANT ENTRAINEMENT
 =================================

 But : Verifier que le pipeline de donnees fonctionne

 1. Prendre 10 patches positifs
 2. Surentrainer le modele sur ces 10 patches
 3. Apres 100 pas (SGD lr=0.01, momentum=0.99)
 4. Objectif : IoU > 0.5

 Si IoU > 0.5 --> PASSED (pipeline OK)
 Si IoU < 0.5 --> FAILED (erreur dans les donnees)
```

---

# 9. INFERENCE (Segmentation sur Image Complete)

## 9.1 Pipeline d'Inference

```
 PIPELINE D'INFERENCE (src/inference/segment.py)
 =================================================

 Image mammographique complete (~3000 x 4000)
       |
       v
 +-----------------------------------+
 | 1. Charger en niveaux de gris     |
 |    Normaliser /255 -> [0,1]       |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 2. Padding a multiple de 256      |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 3. Fenetre glissante 256x256      |
 |    Pas = 256 (sans chevauchement) |
 |    Forward pass par patch         |
 |    Collecter logits -> carte      |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 4. Sigmoid -> carte de probabilite|
 |    Seuil -> masque binaire        |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 5. Composantes connexes (8-conn.) |
 |    Filtrer par :                   |
 |      - aire min : 50 px           |
 |      - aire max : 5000 px         |
 |      - ratio d'aspect max : 3.0   |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | 6. Clustering des bounding boxes  |
 |    Union-Find (distance < 100 px) |
 |    Padding autour cluster : 36 px |
 +-----------------------------------+
       |
       v
 +-----------------------------------+
 | SORTIES :                         |
 |   - Masque binaire (.png)         |
 |   - Overlay avec bbox (.png)      |
 |   - Rapport JSON (.json)          |
 +-----------------------------------+
```

## 9.2 Parametres d'Inference

```
 +----------------------------------+-------------------+
 | Parametre                        | Valeur            |
 +----------------------------------+-------------------+
 | Taille du patch                  | 256 x 256         |
 | Pas de la fenetre                | 256 (sans overlap)|
 | Seuil optimal                    | 0.45              |
 | Seuil par defaut                 | 0.50              |
 | Aire minimum (composante)        | 50 pixels         |
 | Aire maximum (composante)        | 5 000 pixels      |
 | Ratio d'aspect maximum           | 3.0               |
 | Distance de clustering           | 100 pixels        |
 | Padding autour du cluster        | 36 pixels         |
 | Connectivite                     | 8 (diagonales)    |
 +----------------------------------+-------------------+
```

## 9.3 TTA (Test-Time Augmentation)

```
 TTA : 4 PASSES D'INFERENCE
 ============================

 Image originale ──> Forward ──> prob_1
      |
      ├── Flip H ──────> Forward ──> inv. flip H ──> prob_2
      |
      ├── Flip V ──────> Forward ──> inv. flip V ──> prob_3
      |
      └── Rotation 180°──> Forward ──> inv. rot ──> prob_4

                    prob_final = (prob_1 + prob_2 + prob_3 + prob_4) / 4

 Avantage : +0.004 IoU (amelioration modeste)
 Cout     : x4 temps d'inference (~555 ms -> ~2220 ms par patch)
```

## 9.4 Exemple de Rapport JSON

```json
 {
   "image": "data/processed/inbreast/AllPng/20587148_...png",
   "clusters_found": 2,
   "raw_mc_count": 2,
   "threshold": 0.5,
   "tta_used": false,
   "inference_ms": 22107.5,
   "clusters": [
     {
       "cluster_id": 1,
       "location": [559, 1020, 643, 1104],
       "total_area_px": 316,
       "centroid": [601.0, 1062.0],
       "width": 84,
       "height": 84
     },
     {
       "cluster_id": 2,
       "location": [364, 915, 438, 989],
       "total_area_px": 68,
       "centroid": [401.0, 952.0],
       "width": 74,
       "height": 74
     }
   ]
 }
```

---

# 10. RESULTATS FINAUX

## 10.1 Resultats de la Validation Croisee (10-Fold)

```
 RESULTATS 10-FOLD CROSS-VALIDATION
 =====================================

 +------------------+----------+----------+
 | Metrique         | Moyenne  | Ecart-T. |
 +------------------+----------+----------+
 | IoU              | 0.5694   | +/- 0.2870|
 | Dice             | 0.6654   | +/- 0.2955|
 | Sensibilite      | 0.7015   | +/- 0.3082|
 | AUC-ROC          | 0.9642   |     -     |
 | AUC-PR           | 0.8009   |     -     |
 +------------------+----------+----------+
```

## 10.2 Resultats sur le Jeu de Test (184 patches)

```
 RESULTATS SUR LE JEU DE TEST INDEPENDANT
 ==========================================

 +------------------+----------+
 | Metrique         | Valeur   |
 +------------------+----------+
 | IoU moyen        | 0.5521   |
 | Dice moyen       | 0.6507   |
 | Sensibilite moy. | 0.6848   |
 | Temps inference  | ~555 ms  |
 |    par patch     |          |
 +------------------+----------+

 Test effectue sur 184 patches provenant
 de 32 images de patients non vus pendant
 l'entrainement.
```

## 10.3 Comparaison avec DeepMiCa (2023)

```
 COMPARAISON AVEC L'ETAT DE L'ART
 ===================================

                        Notre modele    DeepMiCa 2023
                        ────────────    ─────────────
 AUC-ROC              :    0.9642          0.95
 Difference           :   +0.0142         (ref.)
 Resultat             :    SUPERIEUR

 +-------------------------------------------+
 |                                           |
 |  Notre AUC-ROC = 0.9642                   |
 |  ████████████████████████████████████████  |
 |                                           |
 |  DeepMiCa 2023 = 0.9500                  |
 |  █████████████████████████████████████    |
 |                                           |
 |  Amelioration : +1.42%                   |
 |                                           |
 +-------------------------------------------+
```

## 10.4 Interpretation de la Variance

```
 POURQUOI LA VARIANCE EST ELEVEE ?
 ===================================

 IoU = 0.5694 +/- 0.2870  (ecart-type eleve)

 Raison : Les annotations INbreast contiennent des :
   - Points isolees (dots, ~29 pixels)  --> IoU tres faible
   - Clusters denses (~150 000 pixels)  --> IoU tres eleve

 +---------------------------------------------+
 |  Distribution typique des IoU par patch :    |
 |                                              |
 |  0.0 ██                  (patches quasi-vides)|
 |  0.1 ██                                      |
 |  0.2 ███                                     |
 |  0.3 ████                                    |
 |  0.4 █████                                   |
 |  0.5 ███████                                 |
 |  0.6 ████████                                |
 |  0.7 ██████████                              |
 |  0.8 ████████████                            |
 |  0.9 ██████████████████ (clusters bien detectes)|
 |  1.0 ██████                                  |
 +---------------------------------------------+

 L'AUC-ROC (0.9642) est la metrique la plus fiable car
 elle est independante du seuil et du desequilibre.
```

---

# 11. STRUCTURE DES FICHIERS DU PROJET

```
 ARBORESCENCE DES FICHIERS (phase segmentation)
 =================================================

 mammo-cad/
 |
 |-- data/
 |   |-- interim/inbreast/
 |   |   |-- AllPng/           410 images PNG 8-bit (DICOM converties)
 |   |   |-- Masks/            343 masques binaires (XML parsees)
 |   |   +-- inbreast_mc.csv   313 lignes (Micros=1 seulement)
 |   |
 |   |-- processed/inbreast/
 |   |   |-- AllPng/           313 images pre-traitees (CLAHE)
 |   |   +-- Masks/            313 masques alignes
 |   |
 |   +-- patches/segmentation/
 |       |-- all/
 |       |   |-- images/       2 417 patches 256x256 (pour 10-fold CV)
 |       |   +-- masks/        2 417 masques correspondants
 |       +-- test/
 |           |-- images/       184 patches 256x256 (evaluation finale)
 |           +-- masks/        184 masques correspondants
 |
 |-- src/
 |   |-- inbreast/
 |   |   |-- dicom_to_png.py       Conversion DICOM -> PNG
 |   |   |-- parse_xml_masks.py    XML plist -> masques binaires
 |   |   +-- build_csv.py          Filtre MC, split par patient
 |   |
 |   |-- preprocessing/
 |   |   +-- preprocessing.py      CLAHE, flip, artefacts, crop
 |   |
 |   |-- patches/
 |   |   +-- patch_extraction_seg.py  Extraction 256x256
 |   |
 |   |-- dataset/
 |   |   +-- seg_dataset.py        SegDataset + augmentations
 |   |
 |   |-- models/
 |   |   |-- unet.py               Architecture U-Net
 |   |   +-- losses.py             topk_CE + Dice + IoU + Sensibilite
 |   |
 |   |-- training/
 |   |   +-- train_unet.py         10-fold CV, 300 epoques/fold
 |   |
 |   +-- inference/
 |       +-- segment.py            Fenetre glissante + TTA + bbox
 |
 |-- checkpoints/
 |   |-- unet_best.pth             Meilleur modele (248 MB)
 |   |-- unet_fold{1-10}.pth       Checkpoints par fold
 |   +-- debug_fold*_ep*.png       Visualisations (300 images)
 |
 +-- outputs/
     +-- segment_inference/
         |-- *_mask.png             Masques de segmentation
         |-- *_overlay.png          Overlays avec bounding boxes
         |-- *_report.json          Rapports JSON par image
         |-- test_results.csv       184 lignes de metriques
         |-- best_predictions.png   Meilleures predictions
         |-- worst_predictions.png  Pires predictions
         |-- threshold_sweep_roc_pr.png  Courbes ROC/PR
         +-- tta_comparison.png     Comparaison avec/sans TTA
```

---

# 12. RESUME DES CHIFFRES CLES

```
 ╔═══════════════════════════════════════════════════════════╗
 ║           RESUME - SEGMENTATION MC                       ║
 ╠═══════════════════════════════════════════════════════════╣
 ║                                                          ║
 ║  DATASET                                                 ║
 ║  -------                                                 ║
 ║  Source           : INbreast Release 1.0                  ║
 ║  Images DICOM     : 410                                  ║
 ║  Images avec MC   : 313   (filtre Micros=1)              ║
 ║  Split train(CV)  : 281   (89.8%)                        ║
 ║  Split test       :  32   (10.2%)                        ║
 ║  Patches CV       : 2 417 (256x256, positifs)            ║
 ║  Patches test     :   184 (256x256, positifs)            ║
 ║  Total patches    : 2 601                                ║
 ║  Pixels MC/image  : 29 - 152 667 (median: 507)          ║
 ║  Total pixels MC  : 1 422 622                            ║
 ║                                                          ║
 ║  MODELE                                                  ║
 ║  ------                                                  ║
 ║  Architecture     : U-Net standard                       ║
 ║  Parametres       : 31 037 633 (~31M)                    ║
 ║  Canal de base    : 64                                   ║
 ║  Profondeur       : 4 niveaux encodeur/decodeur          ║
 ║  Goulot           : 1024 canaux                          ║
 ║  Poids modele     : 248 MB (.pth)                        ║
 ║                                                          ║
 ║  ENTRAINEMENT                                            ║
 ║  -------------                                           ║
 ║  Validation       : 10-Fold Cross-Validation             ║
 ║  Epoques/fold     : 300                                  ║
 ║  Total epoques    : 3 000                                ║
 ║  Batch size       : 8                                    ║
 ║  Optimiseur       : SGD (lr=0.001, momentum=0.99)        ║
 ║  Scheduler        : MultiStepLR (ep150, gamma=0.1)       ║
 ║  Perte            : topk_CE (neg_ratio=3)                ║
 ║  Augmentations    : HFlip, VFlip, Rot{90,180,270}        ║
 ║  Augmentation type: On-the-fly (temps reel)              ║
 ║  Variantes/patch  : 16 (2x2x4 combinaisons)             ║
 ║  Samples/fold     : 652 500 (2175 x 300 epoques)        ║
 ║  Samples total    : 6 525 000 (10 folds)                 ║
 ║  GPU              : NVIDIA RTX 4090                      ║
 ║                                                          ║
 ║  RESULTATS                                               ║
 ║  ---------                                               ║
 ║  AUC-ROC          : 0.9642  (DeepMiCa: 0.95)            ║
 ║  AUC-PR           : 0.8009                               ║
 ║  IoU (CV)         : 0.5694 +/- 0.2870                    ║
 ║  Dice (CV)        : 0.6654 +/- 0.2955                    ║
 ║  Sensibilite (CV) : 0.7015 +/- 0.3082                    ║
 ║  IoU (test)       : 0.5521                               ║
 ║  Dice (test)      : 0.6507                               ║
 ║  Sensibilite (test): 0.6848                              ║
 ║  Inference/patch  : ~555 ms                              ║
 ║                                                          ║
 ╚═══════════════════════════════════════════════════════════╝
```

---

# 13. IMAGES DE REFERENCE (fichiers disponibles)

```
 VISUALISATIONS GENEREES (disponibles dans le projet)
 ======================================================

 Resultats d'inference :
   outputs/segment_inference/best_predictions.png
   outputs/segment_inference/worst_predictions.png
   outputs/segment_inference/threshold_sweep_roc_pr.png
   outputs/segment_inference/tta_comparison.png
   outputs/segment_inference/metric_distributions.png
   outputs/segment_inference/cluster_crops.png
   outputs/segment_inference/region_crops.png
   outputs/segment_inference/single_inference.png
   outputs/segment_inference/*_overlay.png  (3 exemples)
   outputs/segment_inference/*_mask.png     (3 exemples)

 Debug d'entrainement :
   checkpoints/debug_fold{1-10}_ep{10-300}.png  (300 images)
   checkpoints/debug_test_final.png

 Pour inclure dans le rapport :
   1. best_predictions.png    --> Exemples de bonnes detections
   2. worst_predictions.png   --> Cas difficiles
   3. threshold_sweep_roc_pr.png --> Courbes ROC et PR
   4. single_inference.png    --> Exemple complet d'inference
   5. debug_test_final.png    --> Predictions vs verite terrain
```
