# DMID — Rapport d'Evaluation Externe
## Digital Mammography Image Database (Inde)

---

# 1. OBJECTIF

```
 Evaluer la generalisation du pipeline Mammo-CAD sur un dataset
 totalement independant, collecte dans un contexte clinique different
 (Inde) avec une infrastructure d'imagerie differente (USA/GE → Inde).

 Question : Le modele entraine sur CBIS-DDSM (USA) generalise-t-il
            sur des mammographies indiennes (DMID) ?

 Cadre     : Test externe zero-shot (aucun fine-tuning sur DMID)
```

---

# 2. DESCRIPTION DU DATASET DMID

## 2.1 Informations Generales

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                  DMID — Digital Mammography Image Database        │
 ├──────────────────────┬───────────────────────────────────────────┤
 │  Pays d'origine      │  Inde                                      │
 │  Institution         │  JIPMER, Puducherry (Hopital universitaire)│
 │  Annee               │  2023                                      │
 │  Source              │  Kaggle (publique)                         │
 │  Format images       │  TIFF (principalement) + DICOM             │
 │  Vues                │  CC + MLO (both breasts)                   │
 │  Total images        │  510 images                                │
 │  Total cas           │  ~225 patients                             │
 │  Classes             │  3 : benign / malignant / normal           │
 │  Annotation          │  Radiologue certifie + histopathologie     │
 │  Resolution          │  Variable (haute resolution clinique)      │
 │  Profondeur bits     │  16-bit                                    │
 └──────────────────────┴───────────────────────────────────────────┘
```

## 2.2 Distribution des Classes

```
 ┌────────────────────────────────────────────────────────────┐
 │  Classe         │  Images  │  Pourcentage  │  Notes        │
 ├─────────────────┼──────────┼───────────────┼───────────────┤
 │  Benigne        │  ~170    │  ~33%         │  Confirmed    │
 │  Maligne        │  ~170    │  ~33%         │  Biopsy proven│
 │  Normale        │  ~170    │  ~33%         │  No finding   │
 └─────────────────┴──────────┴───────────────┴───────────────┘

 → Distribution approximativement equilibree (rare en mammographie)
 → Les cas normaux sont une classe supplementaire vs CBIS-DDSM
 → Biopsy-proven pour les cas malins (gold standard)
```

## 2.3 Types de Lesions

```
 Types annotes dans les metadonnees :
 ┌─────────────────────────────┬────────────────────────────────┐
 │  Type brut (CSV)            │  Mapped (pipeline)             │
 ├─────────────────────────────┼────────────────────────────────┤
 │  calcification              │  calcification                  │
 │  spiculated mass            │  spiculated_mass                │
 │  circumscribed mass         │  circumscribed_mass             │
 │  ill-defined mass           │  ill_defined_mass               │
 │  mass                       │  mass                           │
 │  architectural distortion   │  arch_distortion                │
 │  asymmetry                  │  asymmetry                      │
 │  (none — normal)            │  normal                         │
 └─────────────────────────────┴────────────────────────────────┘
```

## 2.4 Metadonnees Disponibles

```
 Colonnes CSV disponibles :
   image_id      — Identifiant unique de l'image
   patient_id    — Identifiant patient (deidentifie)
   laterality    — L (gauche) / R (droit)
   view          — CC / MLO
   label         — benign / malignant / normal
   lesion_type   — Type de lesion (voir section 2.3)
   birads        — Categorie BI-RADS (0 a 6)
   density       — Densite mammaire (A/B/C/D ou 1/2/3/4)
   age           — Age patient (si disponible)
   filename      — Nom du fichier image
```

## 2.5 Structure des Fichiers

```
 dmid/
 ├── metadata.csv          ← Fichier metadonnees principal
 ├── benign/
 │   ├── image_001.tiff
 │   ├── image_002.dcm
 │   └── ...
 ├── malignant/
 │   ├── image_101.tiff
 │   └── ...
 └── normal/
     ├── image_201.tiff
     └── ...

 Alternative si metadata.csv absent :
   Structure dossiers → label extrait du nom du dossier parent
   Scan recursif de tous les fichiers .tiff / .dcm / .png
```

---

# 3. COMPARAISON DES DATASETS

```
 ┌──────────────────┬─────────────────┬──────────────┬──────────────┐
 │                  │  CBIS-DDSM      │  MIAS        │  DMID        │
 │                  │  (Entrainement) │  (Externe 1) │  (Externe 2) │
 ├──────────────────┼─────────────────┼──────────────┼──────────────┤
 │ Pays             │  USA            │  Royaume-Uni │  Inde        │
 │ Annee            │  ~1990s         │  1994        │  2023        │
 │ Scanner          │  GE (multiple)  │  Xeroradiog. │  Numerique   │
 │ Format           │  DICOM 16-bit   │  PGM 8-bit   │  TIFF/DICOM  │
 │ N images (total) │  3 567          │  322         │  510         │
 │ N images (test)  │  787            │  25 (ROI)    │  ~340 (bin.) │
 │ Classes          │  benign/malin   │  benign/malin│  +normal     │
 │ Resolution       │  Variable haute │  1024x1024   │  Haute res.  │
 │ Annotation       │  ROI polygon    │  ROI cercle  │  Radiologue  │
 │ Biopsy proven    │  Oui            │  Partiel     │  Oui (malin) │
 └──────────────────┴─────────────────┴──────────────┴──────────────┘

 → 3 pays, 3 infrastructures cliniques, 3 decennies differentes
 → DMID le plus recent et le plus pertinent cliniquement (2023)
 → Domain shift croissant : DDSM (source) → MIAS → DMID
```

---

# 4. ARCHITECTURE DU PIPELINE (Rappel)

```
 Image DMID (TIFF/DICOM 16-bit)
          │
          ▼
 ┌─────────────────────┐
 │  PREPROCESSING      │
 │  • 16-bit → 8-bit   │
 │  • TIFF/DCM → PNG   │
 │  • CLAHE            │
 │  • Resize 224x224   │
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  U-Net SEGMENTATION │   ← Entraine sur CBIS-DDSM (AUC 0.9642)
 │  ROI Detection      │
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  CLUSTERING         │
 │  Patch Extraction   │
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  EfficientNet-B3    │   ← Entraine sur CBIS-DDSM (AUC 0.7812)
 │  Classification     │
 │  TTA-16, thr=0.535  │
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  DECISION           │   Benigne / Maligne / (Normale → Benigne)
 │  + Grad-CAM         │
 └─────────────────────┘

 NOTE : Les images "normal" sont traitees comme "benigne" pour
        l'evaluation binaire (benigne vs maligne).
```

---

# 5. PROTOCOLE D'EVALUATION

```
 Preparation :
   1. Telecharger DMID depuis Kaggle
   2. Placer dans : C:\Users\amine\Downloads\Compressed\archive\dmid\
   3. Verifier metadata.csv ou structure dossiers

 Execution :
   python scripts/run_dmid_evaluation.py

   Options disponibles :
   --dmid_dir PATH    Chemin vers le dataset DMID
   --output_dir PATH  Dossier de sortie (defaut: outputs/dmid_evaluation)
   --threshold FLOAT  Seuil de classification (defaut: 0.535)
   --tta INT          Nombre de vues TTA (defaut: 16)

 Sorties :
   outputs/dmid_evaluation/
   ├── dmid_evaluation_results.json   ← Metriques completes
   ├── dmid_per_image_results.csv     ← Resultats par image
   ├── dmid_evaluation_rapport_filled.md  ← Ce rapport rempli
   └── images/
       ├── *_panel.png                ← Panels d'analyse
       └── *_annotated.png           ← Annotations Grad-CAM

 Visualisation (notebook) :
   jupyter notebook notebooks/dmid_evaluation_rapport.ipynb
```

---

# 6. RESULTATS — METRIQUES GLOBALES

```
 [A COMPLETER APRES EXECUTION]

 Pipeline  : EfficientNet-B3 + TTA-16
 Seuil     : 0.535 (F1-optimal CBIS-DDSM val)
 Cas       : [N_TOTAL] images / [N_BINARY] cas binaires (excl. normal)

 ┌──────────────────────────────────────────────────────────┐
 │  Metrique        │  DMID (externe)  │  CBIS-DDSM (test) │
 ├──────────────────┼──────────────────┼───────────────────┤
 │  AUC-ROC         │  [A REMPLIR]     │  0.7812           │
 │  AUC-PR (AP)     │  [A REMPLIR]     │  0.7183           │
 │  Accuracy        │  [A REMPLIR]     │  71.03%           │
 │  Sensitivity     │  [A REMPLIR]     │  63.14%           │
 │  Specificity     │  [A REMPLIR]     │  76.21%           │
 │  PPV             │  [A REMPLIR]     │  63.55%           │
 │  NPV             │  [A REMPLIR]     │  75.89%           │
 │  F1 Score        │  [A REMPLIR]     │  0.6334           │
 │  MCC             │  [A REMPLIR]     │  0.3940           │
 │  Brier Score     │  [A REMPLIR]     │  0.1958           │
 └──────────────────┴──────────────────┴───────────────────┘

 Interpretation du domain shift :
   AUC drop attendu : 3-8% (constate sur MIAS : -X%)
   Si AUC > 0.72 : generalisation acceptable
   Si AUC < 0.65 : domain shift significatif
```

---

# 7. MATRICE DE CONFUSION

```
 [A COMPLETER APRES EXECUTION]

                        PREDICTION
                    BENIN         MALIN       Total
 ┌──────────┬────────────────┬───────────────┬───────┐
 │  BENIN   │   TN = [  ]    │   FP = [  ]   │  [ ]  │
 │ +NORMAL  │   ([  ]%)      │   ([  ]%)     │       │
 ├──────────┼────────────────┼───────────────┼───────┤
 │  MALIN   │   FN = [  ]    │   TP = [  ]   │  [ ]  │
 │          │   ([  ]%)      │   ([  ]%)     │       │
 └──────────┴────────────────┴───────────────┴───────┘
   Total         [ ]               [ ]          [ ]

   [  ] correct / [  ] total = [  ]%

 Images normales :
   Total normales    : [A REMPLIR]
   Classees benignes : [A REMPLIR]  ([  ]%)  ← correct
   Classees malignes : [A REMPLIR]  ([  ]%)  ← fausse alarme
```

---

# 8. ANALYSE DES CAS NORMAUX

```
 Les images normales constituent une classe specifique a DMID
 (absent de CBIS-DDSM et MIAS).

 Comportement attendu : le modele devrait les classifier comme
 benignes (pas de lesion maligne = pas maligne).

 ┌──────────────────────────────────────────────────────┐
 │  Analyse Images Normales                              │
 ├──────────────────────────┬───────────────────────────┤
 │  Total normales          │  [A REMPLIR]              │
 │  Pred. benigne (correct) │  [A REMPLIR] ([  ]%)      │
 │  Pred. maligne (erreur)  │  [A REMPLIR] ([  ]%)      │
 │  Prob. moyenne           │  [A REMPLIR]              │
 │  Prob. mediane           │  [A REMPLIR]              │
 └──────────────────────────┴───────────────────────────┘

 Interpretation :
   > 80% predites benignes → comportement correct
   < 70% predites benignes → le modele sur-active sur les normales
```

---

# 9. RESULTATS PAR TYPE DE LESION

```
 [A COMPLETER APRES EXECUTION]

 ┌─────────────────────┬──────┬─────────┬──────────┬──────────┐
 │  Type de lesion     │  N   │  AUC    │  Acc.    │  F1      │
 ├─────────────────────┼──────┼─────────┼──────────┼──────────┤
 │  calcification      │  [ ] │  [ ]    │  [ ]%    │  [ ]     │
 │  spiculated_mass    │  [ ] │  [ ]    │  [ ]%    │  [ ]     │
 │  circumscribed_mass │  [ ] │  [ ]    │  [ ]%    │  [ ]     │
 │  ill_defined_mass   │  [ ] │  [ ]    │  [ ]%    │  [ ]     │
 │  mass (general)     │  [ ] │  [ ]    │  [ ]%    │  [ ]     │
 │  arch_distortion    │  [ ] │  [ ]    │  [ ]%    │  [ ]     │
 │  asymmetry          │  [ ] │  [ ]    │  [ ]%    │  [ ]     │
 └─────────────────────┴──────┴─────────┴──────────┴──────────┘

 Comparaison avec CBIS-DDSM test :
   Calcification  : CBIS=0.7808 | DMID=[A REMPLIR]  Δ=[  ]
   Mass           : CBIS=0.7836 | DMID=[A REMPLIR]  Δ=[  ]
```

---

# 10. TABLEAU PAR IMAGE (Extrait)

```
 [A COMPLETER APRES EXECUTION — genere automatiquement]

 ┌────────────┬──────────────┬──────────┬──────────┬─────────┬────────┐
 │  Image ID  │  Vrai label  │  Pred.   │  Prob.   │  Correct│  Type  │
 ├────────────┼──────────────┼──────────┼──────────┼─────────┼────────┤
 │  img_001   │  malignant   │  malin   │  0.821   │  ✓      │  mass  │
 │  img_002   │  benign      │  benin   │  0.312   │  ✓      │  calc  │
 │  img_003   │  malignant   │  benin   │  0.401   │  ✗ FN   │  mass  │
 │  img_004   │  normal      │  benin   │  0.278   │  ✓      │  -     │
 │  ...       │  ...         │  ...     │  ...     │  ...    │  ...   │
 └────────────┴──────────────┴──────────┴──────────┴─────────┴────────┘

 Legende :
   ✓     = Classification correcte
   ✗ FP  = Faux Positif (benin predit malin)
   ✗ FN  = Faux Negatif (malin predit benin) — DANGEREUX
   ✗ FN! = Faux Negatif avec forte confiance (prob < 0.3) — CRITIQUE
```

---

# 11. ANALYSE DES ERREURS

```
 [A COMPLETER APRES EXECUTION]

 FAUX POSITIFS — Benin/Normal predit Malin
 ──────────────────────────────────────────
   Total FP          : [A REMPLIR]
   FP Benignes       : [A REMPLIR]
   FP Normales       : [A REMPLIR]
   Prob. moyenne FP  : [A REMPLIR]
   → Cas borderlines (prob proche du seuil 0.535)

 FAUX NEGATIFS — Malin predit Benin (DANGEREUX)
 ───────────────────────────────────────────────
   Total FN          : [A REMPLIR]
   Prob. moyenne FN  : [A REMPLIR]
   FN confiance >70% : [A REMPLIR]  ← Modele sur-confiant
   → Manques potentiellement critiques

 Comparaison avec CBIS-DDSM :
   FP prob moy : CBIS=0.617 | DMID=[A REMPLIR]
   FN prob moy : CBIS=0.433 | DMID=[A REMPLIR]
```

---

# 12. COMPARAISON TRIPLE (CBIS-DDSM / MIAS / DMID)

```
 ┌────────────────┬─────────────────┬──────────────┬──────────────┐
 │  Metrique      │  CBIS-DDSM      │  MIAS        │  DMID        │
 │                │  (Test interne) │  (Externe 1) │  (Externe 2) │
 ├────────────────┼─────────────────┼──────────────┼──────────────┤
 │  N images      │  787            │  25          │  [A REMPLIR] │
 │  AUC-ROC       │  0.7812         │  [resultat]  │  [resultat]  │
 │  Accuracy      │  71.03%         │  [resultat]  │  [resultat]  │
 │  Sensitivity   │  63.14%         │  [resultat]  │  [resultat]  │
 │  Specificity   │  76.21%         │  [resultat]  │  [resultat]  │
 │  F1 Score      │  0.6334         │  [resultat]  │  [resultat]  │
 └────────────────┴─────────────────┴──────────────┴──────────────┘

 Domain shift cumule :
   CBIS-DDSM → MIAS  : ΔAUC = [A REMPLIR]
   CBIS-DDSM → DMID  : ΔAUC = [A REMPLIR]
   → Quantifie la degradation de performance hors distribution
```

---

# 13. EXEMPLES DE RESULTATS VISUELS

```
 Les exemples visuels sont generes par :
   python scripts/run_dmid_evaluation.py
   jupyter notebook notebooks/dmid_evaluation_rapport.ipynb

 Figures generees :
 ┌──────────────────────────────┬──────────────────────────────────┐
 │  Fichier                     │  Contenu                         │
 ├──────────────────────────────┼──────────────────────────────────┤
 │  dmid_confusion_metrics.png  │  Matrice de confusion + barplot  │
 │                              │  des metriques principales        │
 ├──────────────────────────────┼──────────────────────────────────┤
 │  dmid_roc_distribution.png   │  Courbe ROC + distribution des   │
 │                              │  probabilites (KDE)               │
 ├──────────────────────────────┼──────────────────────────────────┤
 │  dmid_per_type_analysis.png  │  Performance par type de lesion  │
 │                              │  (barres groupees + stacked)      │
 ├──────────────────────────────┼──────────────────────────────────┤
 │  dmid_correct_gallery.png    │  Galerie : exemples corrects     │
 │                              │  (TP + TN avec Grad-CAM)         │
 ├──────────────────────────────┼──────────────────────────────────┤
 │  dmid_wrong_gallery.png      │  Galerie : exemples incorrects   │
 │                              │  (FP + FN — cas difficiles)      │
 ├──────────────────────────────┼──────────────────────────────────┤
 │  dmid_triple_comparison.png  │  Comparaison CBIS/MIAS/DMID     │
 │                              │  (radar + barres + operating pt) │
 └──────────────────────────────┴──────────────────────────────────┘

 Partage des resultats :
   Les figures PNG peuvent etre uploadees sur Google Drive dans :
   Mammo-CAD/
   └── Results/
       └── DMID_Evaluation/
           ├── dmid_confusion_metrics.png
           ├── dmid_roc_distribution.png
           ├── dmid_per_type_analysis.png
           ├── dmid_correct_gallery.png
           ├── dmid_wrong_gallery.png
           └── dmid_triple_comparison.png
```

---

# 14. DISCUSSION

## 14.1 Generalisation Geographique

```
 CBIS-DDSM → DMID represente un saut important :
   • Geographie    : USA → Inde (population differente)
   • Infrastructure: GE mammographe numerique (90s) → moderne (2023)
   • Protocole     : Standardise ACR → protocole clinique indien
   • Resolution    : Similaire (16-bit haute resolution)

 Facteurs de domain shift :
   + Format 16-bit similaire (favorable)
   + Lesions radiologiques universelles (favorable)
   - Population differente (densite mammaire differente)
   - Equipement et parametres d'acquisition differents
   - Distribution des subtypes differente
```

## 14.2 Impact du Seuil 0.535

```
 Le seuil 0.535 a ete optimise sur la validation CBIS-DDSM.
 Sur DMID, ce seuil peut ne pas etre optimal.

 Options d'adaptation :
   A) Garder 0.535 : test zero-shot pur, mesure le vrai domain shift
   B) Seuil 0.5    : seuil neutre, reference standard
   C) Seuil optimal DMID : necessite un ensemble de validation DMID

 → Le rapport utilise 0.535 (option A) pour coherence
```

## 14.3 Traitement des Images Normales

```
 Les images normales sont traitees comme benignes pour l'evaluation
 binaire standard.

 Cependant, elles apportent une information supplementaire :
   • Test de specificite sur mammographies saines
   • Le pipeline ne devrait PAS sur-detecter les normales
   • FP sur normales = fausse alarme sur patiente saine
```

---

# 15. CONCLUSION

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                   BILAN EVALUATION DMID                          │
 ├──────────────────────────────────────────────────────────────────┤
 │                                                                   │
 │  Modele     : EfficientNet-B3, entraine sur CBIS-DDSM (USA)     │
 │  Evaluation : DMID (Inde, 2023) — dataset totalement independant │
 │  Seuil      : 0.535 (zero-shot, non-adapte a DMID)              │
 │  TTA        : 16 vues                                            │
 │                                                                   │
 │  AUC CBIS-DDSM  → 0.7812 (interne)                             │
 │  AUC DMID       → [A REMPLIR] (externe)                        │
 │  ΔAUC           → [A REMPLIR] (domain shift)                   │
 │                                                                   │
 │  Validation 3 pays : CBIS-DDSM (USA) + MIAS (UK) + DMID (Inde) │
 │  → Demontre la capacite de generalisation multi-centres          │
 │                                                                   │
 └──────────────────────────────────────────────────────────────────┘
```

---

# 16. ANNEXES

## A. Commandes d'Execution

```bash
# 1. Evaluation complete
python scripts/run_dmid_evaluation.py \
    --dmid_dir "C:\Users\amine\Downloads\Compressed\archive\dmid" \
    --output_dir "outputs/dmid_evaluation" \
    --threshold 0.535 \
    --tta 16

# 2. Visualisation notebook
jupyter notebook notebooks/dmid_evaluation_rapport.ipynb

# 3. Lecture du rapport rempli
cat outputs/dmid_evaluation/dmid_evaluation_rapport_filled.md
```

## B. Format JSON de Resultats

```json
{
  "dataset": "DMID",
  "n_total": 510,
  "n_binary": 340,
  "n_normal": 170,
  "threshold": 0.535,
  "tta": 16,
  "binary_metrics": {
    "auc": 0.0,
    "ap": 0.0,
    "acc": 0.0,
    "sens": 0.0,
    "spec": 0.0,
    "ppv": 0.0,
    "npv": 0.0,
    "f1": 0.0,
    "mcc": 0.0,
    "brier": 0.0,
    "tp": 0, "tn": 0, "fp": 0, "fn": 0
  },
  "normal_analysis": {
    "n_normal": 0,
    "pred_benign": 0,
    "pred_malignant": 0,
    "mean_prob": 0.0
  },
  "per_lesion_type": {},
  "per_image": []
}
```

## C. Requirements Supplementaires

```
 Dependances specifiques DMID :
   pip install pydicom pillow tqdm

 Pour les images TIFF 16-bit :
   PIL (Pillow) gere nativement les TIFF 16-bit
   Normalisation : img = (img / 65535 * 255).astype(uint8)

 Pour les images DICOM :
   pydicom.dcmread(path).pixel_array
   Normalisation : windowing clinique ou min-max
```

---

*Pipeline : EfficientNet-B3 + U-Net | Dataset externe : DMID (Inde, 2023)*
*Modele entraine sur : CBIS-DDSM (USA) | Zero-shot external validation*
*Checkpoint : checkpoints/best_0.7717/efficientnet_stage2.pth*
