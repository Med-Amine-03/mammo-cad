# Chapitre 3 — Handoff pour nouvelle session

**Dernière mise à jour** : 2026-05-10
**État** : 3.4.1 rédigée — prochaine étape **3.4.2**

---

## 1. Contexte général

- **Auteur** : Amine Othmani, PFE FSM / LATIS, encadrant M. Karim Kalti.
- **Date soutenance estimée** : non confirmée (rapport en cours).
- **Sujet** : Mammo-CAD — détection + classification des microcalcifications.
- **Ch1** (cadre clinique + état de l'art) : **accepté par Kalti**.
- **Ch2** (segmentation U-Net sur INbreast, AUC 0,96) : **rédigé, pas encore soumis**.
- **Ch3** (classification + déploiement) : **EN COURS** — voir Section 3 ci-dessous.

---

## 2. Préférences de l'utilisateur (à respecter impérativement)

### 2.1 Style de rédaction
- **Français académique pur** (jamais d'anglais sans `\textit{}`).
- **« Nous »** à la première personne du pluriel — **jamais** commencer une phrase par « Nous ».
- **Style narratif** comme le Ch2 (pas technique froid). Phrases qui partent du clinique avant la technique.
- **Pas de redondance** entre paragraphes — l'utilisateur le vérifie systématiquement.
- **Termes-clés** en `\textbf{}`, **termes anglais** en `\textit{}`, **espaces fines** `~` avant `:` `;` `?` `!` et avant les numéros.
- **Équations** : utiliser `\begin{equation}...\label{...}\end{equation}` suivi de `\myequation{Nom de l'équation}` sur ligne séparée (macro custom du template).

### 2.2 Jury-friendliness — **RÈGLE CRITIQUE**
- **L'utilisateur DOIT pouvoir défendre chaque terme** en 2 phrases simples.
- **Mode B activé** : pour chaque section nouvelle, retirer le jargon de niveau 🔴 (Squeeze-and-Excitation, fan-out, MONOCHROME1, BCEWithLogitsLoss exact, etc.).
- **Toujours fournir** après chaque section : un mini-tableau « Si le jury demande X → tu réponds Y ».
- **Ne JAMAIS inventer** de chiffres d'ablation (+0,084 AUC, etc.) qui ne sont pas dans le code/expériences réels.

### 2.3 Vérifications systématiques
- **Avant toute affirmation technique**, vérifier dans le code (`src/`) que c'est vrai.
- Si une info vient d'un PDF externe sans correspondance dans le code → **demander confirmation** avant de l'écrire.
- Ne pas mettre les **valeurs exactes** des hyperparamètres dans le texte si elles sont déjà dans un tableau.
- **Pas de paramètres détaillés** (α=15, σ=4 pour les distorsions, valeurs ImageNet exactes…) — détails impossibles à défendre.

---

## 3. État du Chapitre 3

### 3.1 Plan final validé

```
3.1  Introduction                                       ✅ RÉDIGÉE
3.2  Architecture EfficientNet-B3
     3.2.1  Compound scaling et choix architecture     ✅ RÉDIGÉE (allégée)
            [fig : efficientnet_schema.png ✅ générée]
     3.2.2  Tête de classification personnalisée       ✅ RÉDIGÉE (allégée)

3.3  Stratégie d'entraînement en deux étapes
     3.3.1  Données et augmentation                    ✅ RÉDIGÉE (allégée)
            [fig : preprocessing_pipeline.png ✅ générée]
            [tab : 3.1 split CBIS-DDSM patient-wise]
     3.3.2  Étape 1 — entraînement de la tête          ✅ RÉDIGÉE
            [fig : stage1_curves.png ✅ existante (en anglais)]
            [tab : 3.2 hyperparamètres stage 1]
     3.3.3  Étape 2 — dégel partiel + SWA              ✅ RÉDIGÉE
            [fig : stage2_curves.png ✅ existante (en anglais)]
            [tab : 3.3 hyperparamètres stage 2]
     3.3.4  Techniques de régularisation               ❌ SUPPRIMÉE (redondante)

3.4  Évaluation sur CBIS-DDSM
     3.4.1  Protocole d'évaluation                     ✅ RÉDIGÉE
     3.4.2  Résultats quantitatifs                     ⏳ EN COURS — PROCHAINE
            [fig : cls_roc_cm_final.png — À GÉNÉRER]
     3.4.3  Impact du TTA                              ⏳ À FAIRE
            [fig : fig3_tta_comparison.png ✅ existante]
     3.4.4  Sélection du seuil de décision             ⏳ À FAIRE
            [fig : fig2_f1_vs_threshold.png ✅ existante]
     3.4.5  Analyse par type de lésion (calc/masse)    ⏳ À FAIRE
            [fig : fig4_per_type_lesion.png ✅ existante]

3.5  Analyse des cas difficiles
     3.5.1  Distribution des probabilités              ⏳ À FAIRE
            [fig : fig1_prob_distribution.png ✅ existante]
     3.5.2  Profil des FP et FN                        ⏳ À FAIRE

3.6  Adaptation de domaine : INbreast
     3.6.1  Dérive de domaine CBIS-DDSM → INbreast     ⏳ À FAIRE
     3.6.2  Protocole de fine-tuning                   ⏳ À FAIRE
     3.6.3  Résultats et analyse BI-RADS               ⏳ À FAIRE

3.7  Pipeline intégré et déploiement
     3.7.1  Technologies et environnement              ⏳ À FAIRE
            [tab : Composant / Outil+Version / Rôle]
     3.7.2  Pipeline de bout en bout (6 étapes)        ⏳ À FAIRE
     3.7.3  API REST FastAPI                           ⏳ À FAIRE
            [fig : api_schema.png — À GÉNÉRER]
     3.7.4  Interface utilisateur Next.js              ⏳ À FAIRE
            [screenshots à fournir par utilisateur]

3.8  Conclusion                                        ⏳ À FAIRE
```

### 3.2 Phrase de transition AJOUTÉE en 3.1 (entre paragraphes 2 et 3)

> *Bien que ces deux étapes s'inscrivent dans un même pipeline, elles répondent à des questions de nature différente — l'une géométrique, l'autre diagnostique — et imposent en conséquence des choix méthodologiques distincts, que nous justifions à mesure qu'ils sont introduits.*

Ce paragraphe prépare le jury à accepter la différence de démarche entre Ch2 (segmentation) et Ch3 (classification).

---

## 4. Faits chiffrés vérifiés (à utiliser)

### 4.1 Données CBIS-DDSM
- Total : **3 567 cas** issus de **1 566 patientes uniques**
- Calcifications : **1 871 cas** | Masses : **1 696 cas**
- Total patches extraits : **4 484** (multi-ROI possible) — dimension **224×224**
- Distribution classes : **2 111 bénins (59,2 %)** vs **1 456 malins (40,8 %)**
- Split par patient (seed=42) :
  | Partition    | Patients | Cas    | Bénins | Malins | Ratio B/M |
  |--------------|----------|--------|--------|--------|-----------|
  | Entraînement | 973      | 2 260  | 1 329  | 931    | 1,43      |
  | Validation   | 244      | 520    | 307    | 213    | 1,44      |
  | Test         | 349      | 787    | 475    | 312    | 1,52      |

### 4.2 Architecture EfficientNet-B3
- Paramètres totaux : **11,1 M**
- Stage 1 : tête uniquement, ~**400 K** paramètres entraînables (~4 %)
- Stage 2 : tête + features[5,6,7,8], ~**5,1 M** entraînables (~46 %)
- Tête : Linear(1536→256) → ReLU → Dropout(p=0,5) → Linear(256→1)

### 4.3 Hyperparamètres Stage 1 (vérifiés dans `train_classifier.py`)
| Paramètre | Valeur |
|---|---|
| epochs | 60 |
| lr | 1e-4 |
| weight_decay | 1e-4 |
| batch | 32 |
| grad_clip | 1.0 |
| label_smooth | ε=0,05 |
| mixup_alpha | α=0,2 |
| patience_lr | 7 (ReduceLROnPlateau) |
| patience_stop | 20 (early stopping) |
| min_lr | 1e-7 |

### 4.4 Hyperparamètres Stage 2 (vérifiés)
| Paramètre | Valeur |
|---|---|
| epochs | 150 (max) |
| lr_bb | 1e-5 |
| lr_head | 1e-4 |
| warmup_epochs | 5 |
| batch | 32 |
| grad_clip | 0.5 |
| label_smooth | 0,05 |
| mixup_alpha | 0,2 |
| patience_lr | 7 |
| patience_stop | 35 (FIX-8) |
| swa_epochs | 20 |
| swa_lr | 1e-6 |
| Arrêt anticipé effectif | ~115 époques (observé) |

### 4.5 Sampling & loss
- **WeightedRandomSampler** par patient : poids = 1/n_patches(patient), `num_samples=n_pats`, `replacement=False` → **chaque patient = 1 patch / époque**
- **pos_weight** = N_bénin / N_malin = **1,43** (plafonné à 2,5, cap = POS_WEIGHT_CAP)
- **NE PAS DIRE** : « WeightedRandomSampler 5× sur les malins » → FAUX (info du PDF erronée, voir code)

### 4.6 Évaluation CBIS-DDSM (résultats officiels — checkpoint `best_0.7717`)
- **AUC = 0,7812** (avec TTA×16)
- **TTA×1 = 0,7627** | **TTA×8 = 0,7738** | **TTA×16 = 0,7812**
- Seuil optimal τ = **0,535** (max F1 sur validation)
- Sensibilité = **0,6314** | Spécificité = **0,7621**
- F1 = **0,6334** | Accuracy = **71,0 %** | MCC = **0,2476** | AUC-PR = **0,7183**
- TP=197, TN=362, FP=113, FN=115
- FN prob moyenne = 0,433 | FP prob moyenne = 0,617
- Calcifications : AUC = 0,7808 (n=382)
- Masses : AUC = 0,7836 (n=405)

### 4.7 Fine-tuning INbreast
- AUC zero-shot CBIS-DDSM → INbreast : **0,4117** (chute de domaine massive)
- AUC après fine-tuning : **0,62**
- Test set INbreast : 50 images isolées (30B + 20M, seed=42)
- 50 époques, SWA 10ep, lr_bb=5e-6, lr_head=5e-5, warmup=3ep
- Mammographe CBIS = GE Senographe, INbreast = Siemens Mammomat
- Performance par BI-RADS après FT : 2→65%, 3→75%, 4a→33%, 4b→100%, 5→47%

---

## 5. Bibliographie

**Fichier consolidé** : [docs/bibliographie_ch1_ch3.tex](bibliographie_ch1_ch3.tex)

- Total : **38 entrées** (11 Ch1 + 7 Ch2 + 20 Ch3) en style `\bibitem{}` manuel cohérent.
- Clés à réutiliser dans Ch3 :
  - `\cite{moreira2012inbreast}` (INbreast)
  - `\cite{cbisddsm2017}` (CBIS-DDSM, attention pas `lee2017cbisddsm`)
  - `\cite{he2015kaiming}` (init Kaiming, attention pas `he2015delving`)
- À vérifier par l'utilisateur : `shia2025mammography` (volume + page exacts non confirmés)

---

## 6. Liste des abréviations

**Total : 53 abréviations** (43 d'origine Ch1+Ch2 + 10 ajoutées Ch3).

Nouvelles ajoutées : AdamW, CUDA, GAP, HTTP, ImageNet, MBConv, PyTorch, SE, UUID, VRAM.

---

## 7. Figures générées et localisation

| Figure | Script | Statut | Cible LaTeX |
|---|---|---|---|
| `efficientnet_schema.png` | `scripts/generate_efficientnet_schema.py` | ✅ générée | `images/efficientnet_schema.png` |
| `preprocessing_pipeline.png` | `scripts/generate_preprocessing_pipeline.py` | ✅ générée | `images/preprocessing_pipeline.png` |
| `patches_gallery.png` | `scripts/generate_patches_gallery.py` | ❌ **NON utilisée** (rejetée par utilisateur) | — |
| `stage1_curves.png` | (origine entraînement) | ✅ existe en anglais dans `checkpoints/best_0.7717/` | à copier vers `images/` |
| `stage2_curves.png` | idem | ✅ existe en anglais | à copier vers `images/` |
| `cls_roc_cm_final.png` | **À CRÉER** | ❌ | sera dans 3.4.2 |
| `fig1_prob_distribution.png` | déjà dans `outputs/visual_analysis/` | ✅ | à copier vers `images/` |
| `fig2_f1_vs_threshold.png` | idem | ✅ | à copier vers `images/` |
| `fig3_tta_comparison.png` | idem | ✅ | à copier vers `images/` |
| `fig4_per_type_lesion.png` | idem | ✅ | à copier vers `images/` |
| `api_schema.png` | **À CRÉER** | ❌ | sera dans 3.7.3 |

⚠️ **Note importante** : les checkpoints `.pth` ne contiennent **pas** l'historique des époques (juste le best snapshot), donc impossible de régénérer les courbes stage1/stage2 en français sans relancer l'entraînement. → Les figures restent en anglais (acceptable PFE), captions en français explicites.

---

## 8. Code — points-clés vérifiés

### Fichiers consultés et validés
- `src/models/efficientnet.py` : architecture + tête + freeze/unfreeze
- `src/training/train_classifier.py` : STAGE1, STAGE2, sampler, loss, optimizer
- `src/inference/classify.py` : TTA×16 (8 géométriques + 8 multi-échelle 260→224)
- `src/inference/pipeline.py` : `_crop_bbox(target=224)` redimensionne les crops U-Net pour le classificateur
- `src/cbis_ddsm/extract_crops.py` : DICOM extraction + MONOCHROME1 inversion (ligne 71-72)
- `src/dataset/cls_dataset.py` : pos_weight, augmentations (5 familles)
- `src/patches/patch_extraction_cls.py` : organisation patches par split/label

### Configs vérifiées
- `POS_WEIGHT_CAP = 2.5` ([train_classifier.py:46](src/training/train_classifier.py:46))
- `SCHEDULER_LOSS_CLIP = 2.0` (FIX-7, évite que les spikes de val_loss déclenchent decay)
- Stage 2 unfreeze : `features[5,6,7,8]` (3 derniers MBConv stages + conv 1×1 finale)

---

## 9. Décisions méthodologiques prises

| # | Décision | Raison |
|---|---|---|
| 1 | **Mode B** activé (allègement jury-friendly) | L'utilisateur veut défendre chaque terme |
| 2 | Phrase de transition en 3.1 | Justifier la différence de démarche Ch2 vs Ch3 |
| 3 | Suppression de 3.3.4 (régularisation) | Redondant avec 3.3.2 et 3.3.3 |
| 4 | Garder figures en anglais | Impossible à régénérer sans historique |
| 5 | Figure `patches_gallery.png` rejetée | Pas d'apport pédagogique suffisant |
| 6 | **Pas** d'ablations chiffrées (+0,084 AUC etc.) | Pas réalisées expérimentalement |
| 7 | « MONOCHROME1 » remplacé par paraphrase | Jargon DICOM indéfendable |
| 8 | Paramètres exacts des augmentations retirés | α=15, σ=4 indéfendables |
| 9 | « Squeeze-and-Excitation », « Swish » retirés | Trop techniques |
| 10 | « 200 000 vues » placée en fin de 3.3.1 | Logique : après description sampler+augmentation |

---

## 10. Prochaine action — Section 3.4.2

**À rédiger** : Résultats quantitatifs sur CBIS-DDSM.

**Doit contenir** :
- Tableau de métriques finales (AUC=0,7812 ; Sens=0,6314 ; Spec=0,7621 ; F1=0,6334 ; Acc=71,0% ; MCC=0,2476)
- Matrice de confusion 2×2 : TP=197, TN=362, FP=113, FN=115
- Discussion clinique (sensibilité limitée, importance du dépistage etc.)
- Comparaison à la littérature (AUC 0,75–0,85 standard pour CBIS-DDSM)

**Figure à générer** : `cls_roc_cm_final.png`
- Courbe ROC (gauche) avec AUC=0,7812 et seuil τ=0,535 marqué
- Matrice de confusion (droite) avec les 4 quadrants
- Style cohérent avec les autres figures (matplotlib, dpi=180)
- **Données** : à charger depuis `checkpoints/best_0.7717/efficientnet_stage2_swa.pth` (val_probs/val_labels sont dans le checkpoint, mais on veut le TEST → relancer évaluation OU utiliser fichier de résultats existant)

**Source potentielle des prédictions test** : `outputs/evaluation/final_results_tta.txt` (à vérifier).

---

## 11. Conventions LaTeX du projet (à respecter)

```latex
% Chapitre header
\chapter{Titre}
\label{ch:ch3}
\begin{spacing}{1.2}
\minitoc
\thispagestyle{MyStyle}
\end{spacing}
\newpage

% Séparateur sections
% ═══════════════════════════════════════════
\section{Titre}
\label{sec:ch3-xxx}
% ═══════════════════════════════════════════

% Séparateur sous-sections
% ═══════════════════════════════════════════
\subsection{Titre}
\label{subsec:ch3-xxx}
% ═══════════════════════════════════════════

% Équation nommée (macro custom)
\begin{equation}
    formula
    \label{eq:xxx}
\end{equation}
\myequation{Nom de l'équation}

% Figure
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{images/xxx.png}
    \caption[xxx]{Caption longue en français}
    \label{fig:ch3-xxx}
\end{figure}

% Tableau (booktabs)
\begin{table}[H]
    \centering
    \caption{Caption}
    \label{tab:ch3-xxx}
    \begin{tabular}{ll}
        \toprule
        En-tête & Valeur \\
        \midrule
        ...
        \bottomrule
    \end{tabular}
\end{table}
```

---

## 12. Pour reprendre la session

1. **Lire ce fichier en entier** avant tout.
2. **Charger en mémoire** : 4.1 → 4.7 (chiffres vérifiés).
3. **Ne pas re-rédiger** ce qui est déjà ✅ — l'utilisateur les a validés.
4. **Continuer à 3.4.2** : générer `cls_roc_cm_final.png`, rédiger la sous-section.
5. **À chaque section** : fournir le bloc LaTeX + tableau « Si le jury demande X → Y ».
6. **Vérifier dans le code** avant toute affirmation technique.
7. **Demander confirmation** avant d'ajouter des chiffres qui ne sont pas dans le code.

---

## 13. Fichiers liés

- [docs/bibliographie_ch1_ch3.tex](bibliographie_ch1_ch3.tex) — bibliographie complète
- `outputs/figures/efficientnet_schema.png` — Figure 3.2.1
- `outputs/figures/preprocessing_pipeline.png` — Figure 3.3.1
- `checkpoints/best_0.7717/stage1_curves.png` — Figure 3.3.2
- `checkpoints/best_0.7717/stage2_curves.png` — Figure 3.3.3
- `checkpoints/best_0.7717/efficientnet_stage2_swa.pth` — modèle final (~0,7812 AUC test)

---

**Fin du handoff. Reprendre directement à 3.4.2.**
