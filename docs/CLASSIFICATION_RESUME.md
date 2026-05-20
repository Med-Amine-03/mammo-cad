# Classification — Strategie & Resultats

## EfficientNet-B3 sur CBIS-DDSM

---

# 1. STRATEGIE & AMELIORATIONS

## 1.1 Modele et Donnees

```
 Modele      : EfficientNet-B3 (11.9M parametres, ImageNet pretrain)
 Dataset     : CBIS-DDSM (DICOM 16-bit lossless, 3 567 cas)
 Input       : Patches 224x224 (crops ROI + CLAHE + ImageNet norm)
 Output      : Probabilite maligne [0, 1]
 Split       : 2 260 train / 520 val / 787 test (par patient, seed=42)
```

## 1.2 Entrainement 2-Stage

```
 STAGE 1 : Head + BatchNorm seulement
 ──────────────────────────────────────
   Params entraines : 262K / 11.9M (2.2%)
   Backbone         : GELE
   LR               : 1e-4 (AdamW, wd=1e-4)
   Epochs           : 60 (early stop patience=20)
   Scheduler        : ReduceLROnPlateau (factor=0.5, patience=7)

 STAGE 2 : Fine-tuning blocks 5-8 + SWA
 ──────────────────────────────────────
   Params entraines : 6.5M / 11.9M (54.7%)
   Blocks 0-4       : GELES
   Blocks 5-8       : DEGEL + fine-tune
   LR               : Backbone 1e-5 | Head 1e-4
   Warmup           : 5 epochs (1e-8 → 1e-5)
   Epochs           : 150 (early stop patience=35)
   SWA              : 20 epochs LR=1e-6 (moyenne des poids)
```

## 1.3 Changements / Echanges (pourquoi ces choix)

```
 ┌────────────────────────┬──────────────────────────────────────┐
 │  Echange               │  Justification                       │
 ├────────────────────────┼──────────────────────────────────────┤
 │  JPEG → DICOM 16-bit   │  Texture des MC preservee            │
 │                        │  Val AUC 0.722 → 0.806 (+0.084)      │
 ├────────────────────────┼──────────────────────────────────────┤
 │  CosineLR → Plateau    │  LR stable, pas d'oscillations       │
 │                        │  Reduit LR sur plateau seulement     │
 ├────────────────────────┼──────────────────────────────────────┤
 │  Ajout pos_weight=1.43 │  Compense 60% benin / 40% malin      │
 │                        │  Spec 0.17 → 0.76                    │
 ├────────────────────────┼──────────────────────────────────────┤
 │  BN gele → BN degele   │  Stats ImageNet ≠ stats mammo        │
 │                        │  Adaptation au domaine medical       │
 ├────────────────────────┼──────────────────────────────────────┤
 │  Warmup 5 epochs       │  Evite destruction features en S2    │
 ├────────────────────────┼──────────────────────────────────────┤
 │  Ajout MixUp (a=0.2)   │  Frontieres de decision lisses       │
 ├────────────────────────┼──────────────────────────────────────┤
 │  Label Smoothing 0.05  │  Labels {0,1} → {0.025, 0.975}       │
 │                        │  Evite sur-confiance                 │
 ├────────────────────────┼──────────────────────────────────────┤
 │  Ajout SWA             │  Moyenne des poids = plateau plat    │
 │                        │  Meilleure generalisation            │
 ├────────────────────────┼──────────────────────────────────────┤
 │  TTA-1 → TTA-16        │  16 vues (8 geo + 8 echelle)         │
 │                        │  +0.010 AUC sans re-entrainement     │
 ├────────────────────────┼──────────────────────────────────────┤
 │  Seuil 0.5 → 0.535     │  F1-optimal sur validation           │
 └────────────────────────┴──────────────────────────────────────┘
```

## 1.4 Strategies Abandonnees

```
 ✗ DenseNet-121 torchxrayvision  → AUC 0.607 (domain shift thorax)
 ✗ Mask-based precise cropping   → AUC -0.080 (crops trop homogenes)
```

## 1.5 Augmentations Appliquees

```
 11 augmentations on-the-fly :
   flip H/V, rotation ±15°, affine, scale jitter 0.85-1.15x,
   translation, elastic, CLAHE, blur, noise, RandomResizedCrop
 + MixUp (alpha=0.2, p=0.5)
 + Normalisation ImageNet (critique !)
```

---

# 2. RESULTATS

## 2.1 Metriques Globales (Test Set, n=787)

```
 ┌────────────────────────────────────────────────────────┐
 │  Metrique     │  Valeur   │  IC 95%                   │
 ├───────────────┼───────────┼───────────────────────────┤
 │  AUC-ROC      │  0.7812   │  [0.7483 — 0.8130]        │
 │  AUC-PR (AP)  │  0.7183   │  [0.6662 — 0.7678]        │
 │  Accuracy     │  71.03%   │  (559/787)                │
 │  Sensitivity  │  63.14%   │  [0.5793 — 0.6871]        │
 │  Specificity  │  76.21%   │  [0.7244 — 0.7996]        │
 │  PPV          │  63.55%   │                           │
 │  NPV          │  75.89%   │                           │
 │  F1 Score     │  0.6334   │  [0.5878 — 0.6775]        │
 │  MCC          │  0.3940   │                           │
 │  Brier Score  │  0.1958   │                           │
 └────────────────────────────────────────────────────────┘

 Checkpoint   : best_0.7717/efficientnet_stage2.pth
 Val AUC      : 0.8056
 Test AUC     : 0.7812
 Val-Test Gap : 2.43% (bonne generalisation)
 Seuil        : 0.535 (F1-optimal)
 TTA          : 16 vues
```

## 2.2 Matrice de Confusion

```
                       PREDICTION
                   BENIN         MALIN       Total
 ┌────────┬────────────────┬────────────────┬───────┐
 │ BENIN  │   TN = 362     │   FP = 113     │  475  │
 │        │   (76.2%)      │   (23.8%)      │       │
 ├────────┼────────────────┼────────────────┼───────┤
 │ MALIN  │   FN = 115     │   TP = 197     │  312  │
 │        │   (36.9%)      │   (63.1%)      │       │
 └────────┴────────────────┴────────────────┴───────┘
   Total       477              310            787

   559 correct / 787 total = 71.03%
```

## 2.3 Par Type de Lesion

```
 CALCIFICATIONS (n=382)          MASSES (n=405)
 ─────────────────────           ─────────────────────
 AUC         : 0.7808            AUC         : 0.7836
 Accuracy    : 70.9%             Accuracy    : 71.1%
 Sensitivity : 60.6%             Sensitivity : 65.6%
 Specificity : 78.0%             Specificity : 74.6%
 F1          : 0.6288            F1          : 0.6378

 → AUC presque identique entre calc et mass
 → Masses mieux detectees (+5% sensitivity)
 → Calc avec moins de FP (+3.4% specificity)
```

## 2.4 Evolution des Versions

```
 v0 : 0.477 AUC  ❌  (8 bugs critiques)
 v1 : 0.722 AUC  ✓  (+0.245, corrections bugs)
 v2 : 0.731 AUC  ✓  (+0.009, DICOM)
 v3 : 0.793 AUC  ✓  (+0.062, Stage 2 + TTA)
 v4 : 0.754 AUC  ❌  (mask crop, abandonne)
 v5 : 0.781 AUC  ✓  FINAL (+0.020, regularisation)

 Amelioration totale : +63.7% (de 0.477 a 0.781)
```

## 2.5 Analyse des Erreurs

```
 FAUX POSITIFS (n=113) — Benin predit Malin
   Calc: 50 | Mass: 63
   Prob moyenne : 0.617 (proche seuil 0.535)
   → Cas borderlines

 FAUX NEGATIFS (n=115) — Malin predit Benin (DANGEREUX)
   Calc: 61 | Mass: 54
   Prob moyenne : 0.433 (loin du seuil)
   → Le modele est confiant mais il se trompe
```

## 2.6 Positionnement Litterature

```
 Plage publiee : AUC 0.78 — 0.84
 Notre modele  : AUC 0.7812  [0.748 — 0.813]
 Status        : Dans la plage publiee
                 IC haut approche le top (0.813)
```

---

*Modele : EfficientNet-B3 | Dataset : CBIS-DDSM | Seuil : 0.535 | TTA-16*
*Checkpoint : checkpoints/best_0.7717/efficientnet_stage2.pth*
