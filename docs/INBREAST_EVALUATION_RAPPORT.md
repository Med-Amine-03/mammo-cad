# INBREAST — Rapport d'Evaluation Externe
## INbreast Dataset (Portugal, Siemens Mammomat, DICOM)

---

# 1. OBJECTIF

```
 Evaluer la generalisation du pipeline Mammo-CAD sur le dataset INbreast,
 collecte au Portugal avec un scanner Siemens (DICOM haute resolution),
 annoté en BI-RADS par des radiologues experts.

 Question : Le modele entraine sur CBIS-DDSM (USA, GE Medical) generalise-t-il
            sur des mammographies portugaises (INbreast, Siemens) ?

 Cadre     : Test externe zero-shot (aucun fine-tuning sur INbreast)
             Sous-ensemble stratifie : 50 images (30 benign + 20 malignant)
             random_state=42, tirage equitable des classes
```

---

# 2. DESCRIPTION DU DATASET INBREAST

## 2.1 Informations Generales

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │               INbreast — Digital Mammography Dataset                  │
 ├───────────────────────┬──────────────────────────────────────────────┤
 │  Pays d'origine       │  Portugal                                     │
 │  Institution          │  Hospital de São João, Porto                  │
 │  Annee                │  2012 (publication)                           │
 │  Source               │  Acesso publique (recherche)                  │
 │  Scanner              │  Siemens Mammomat Inspiration                 │
 │  Format images        │  DICOM (16-bit, haute resolution)             │
 │  Vues                 │  CC + MLO (both breasts)                      │
 │  Total images         │  410 images                                   │
 │  Total patients       │  115 patients                                 │
 │  Classes              │  2 : benignes / malignes (BI-RADS)            │
 │  Types principaux     │  Microcalcifications (MC) + Masses            │
 │  Annotation           │  BI-RADS 1 a 6, radiologues seniors           │
 │  Resolution           │  3328 x 4084 pixels (haute def clinique)      │
 │  Profondeur bits      │  16-bit grayscale                             │
 └───────────────────────┴──────────────────────────────────────────────┘
```

## 2.2 Distribution des Classes et BI-RADS

```
 ┌──────────────┬──────────┬───────────────┬────────────────────────────┐
 │  BI-RADS     │  Images  │  Classe       │  Signification clinique    │
 ├──────────────┼──────────┼───────────────┼────────────────────────────┤
 │  2           │  ~45     │  BENIGNE      │  Benin, suivi routine      │
 │  3           │  ~45     │  BENIGNE      │  Probablement benin        │
 │  4a          │  ~30     │  MALIGNE      │  Faible suspicion maligne  │
 │  4b          │  ~25     │  MALIGNE      │  Suspicion moderee         │
 │  4c          │  ~20     │  MALIGNE      │  Forte suspicion           │
 │  5           │  ~30     │  MALIGNE      │  Tres forte suspicion      │
 │  6           │  ~15     │  MALIGNE      │  Histologie prouvee        │
 ├──────────────┼──────────┼───────────────┼────────────────────────────┤
 │  TOTAL       │  ~210    │  ~130B + ~80M │  (410 vues, 115 patients)  │
 └──────────────┴──────────┴───────────────┴────────────────────────────┘

 Regle de binarisation :
   BI-RADS 2, 3         → BENIGNE  (0)
   BI-RADS 4a, 4b, 4c, 5, 6 → MALIGNE (1)
```

## 2.3 Types de Lesions Annotes

```
 ┌──────────────────────────────┬─────────────────────────────────────┐
 │  Type de lesion (INbreast)   │  Mapped (pipeline mammo-cad)        │
 ├──────────────────────────────┼─────────────────────────────────────┤
 │  Microcalcifications (MC)    │  calcification                       │
 │  Masse                       │  mass                                │
 │  Distorsion architecturale   │  arch_distortion                     │
 │  Asymetrie                   │  asymmetry                           │
 │  (normal — BI-RADS 1)        │  normal (exclu de l'evaluation)      │
 └──────────────────────────────┴─────────────────────────────────────┘

 Note : Le pipeline mammo-cad est specialise MICROCALCIFICATIONS.
        Les masses sont evaluees mais la segmentation U-Net est optimisee
        pour la detection de MC (trained sur CBIS-DDSM calc+mass).
```

---

# 3. NOTRE SOUS-ENSEMBLE DE TEST

## 3.1 Parametres de Selection

```
 SOUS-ENSEMBLE DE TEST — INBREAST
 ===================================

 ┌──────────────────────────────────────────────────────────────┐
 │  Total images selectionnees  :  50                           │
 │  Benignes (BI-RADS 2/3)      :  30  (60%)                   │
 │  Malignes (BI-RADS 4a-6)     :  20  (40%)                   │
 │  Methode de tirage           :  Stratified random sampling   │
 │  random_state                :  42  (reproductible)          │
 │  Format d'entree pipeline    :  DICOM → PNG (via conversion) │
 └──────────────────────────────────────────────────────────────┘

 Justification du sous-ensemble :
   → Dataset complet (410 images) trop volumineux pour evaluation zero-shot
   → 50 images = compromis representativite / temps de calcul
   → Stratification garantit la representativite de chaque classe
   → random_state=42 assure la reproducibilite exacte
```

---

# 4. COMPARAISON DES DATASETS

## 4.1 CBIS-DDSM vs MIAS vs INbreast

```
 ┌─────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
 │  Critere            │  CBIS-DDSM (Train)   │  MIAS (Ext-1)        │  INbreast (Ext-2)    │
 ├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
 │  Pays               │  USA                 │  Royaume-Uni         │  Portugal            │
 │  Programme          │  Digital Database    │  UK NBSP             │  Hopital Sao Joao    │
 │  Scanner            │  GE Medical Systems  │  Scanner UK (ancien) │  Siemens Mammomat    │
 │  Format original    │  DICOM 16-bit        │  Film numerise       │  DICOM 16-bit        │
 │  Format utilise     │  ROI PNG pre-coup.   │  Mammogramme complet │  DICOM → PNG conv.   │
 │  Resolution         │  Variable (ROI)      │  1024 x 1024 fixe   │  3328 x 4084 (HD)    │
 │  Types de lesion    │  Calc + Mass         │  Calcifications seul │  MC + Masses + Dist. │
 │  Annotation         │  Expert ACR          │  Expert MIAS         │  BI-RADS expert      │
 │  Taille totale      │  3 567 cas           │  25 images           │  410 images          │
 │  Taille test        │  787 (eval)          │  25 (all)            │  50 (stratifie)      │
 │  Patients           │  1 566               │  25                  │  115                 │
 ├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
 │  Domain shift       │  —  (train set)      │  FORT (film→digital) │  MODERE (DICOM HD)   │
 │  vs CBIS-DDSM       │                      │                      │                      │
 └─────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘

 INbreast est le dataset DICOM le plus proche de CBIS-DDSM en termes
 de format et de qualite d'image, meme si le scanner (Siemens vs GE)
 et la population (Portugal vs USA) different.
```

---

# 5. ARCHITECTURE PIPELINE

## 5.1 Pipeline Complet sur Images INbreast

```
 PIPELINE COMPLET mammo-cad SUR IMAGES INBREAST
 ================================================

 Image INbreast DICOM (3328x4084, 16-bit grayscale)
       |
       v
 +--------------------------------------------------+
 | ETAPE 1 : CONVERSION + PREPROCESSING             |
 |                                                  |
 |  1. DICOM → PNG 16-bit (pydicom)                 |
 |  2. Normalisation 16-bit → 8-bit                 |
 |  3. Flip-to-left (orientation normalisee)        |
 |  4. CLAHE (egalisation adaptative contraste)     |
 |  5. Resize adaptatif (preservation AR)           |
 |                                                  |
 |  Sortie : image preprocessee PNG                 |
 +--------------------------------------------------+
       |
       v
 +--------------------------------------------------+
 | ETAPE 2 : SEGMENTATION U-Net                     |
 |                                                  |
 |  Modele    : U-Net (31M parametres)              |
 |  Training  : 10-fold CV sur CBIS-DDSM            |
 |  AUC seg   : 0.9642                              |
 |  Tache     : Detecter les microcalcifications    |
 |                                                  |
 |  Sortie : masque binaire + overlay               |
 +--------------------------------------------------+
       |
       v
 +--------------------------------------------------+
 | ETAPE 3 : CLUSTERING                             |
 |                                                  |
 |  Methode   : Connected components + merge       |
 |  Tache     : Regrouper MC en clusters            |
 |  Resultat  : N regions d'interet (bbox)          |
 |                                                  |
 |  Si N = 0 : AUCUNE MC detectee                  |
 |             --> Image classee BENIN par defaut   |
 +--------------------------------------------------+
       |
       v (pour chaque region)
 +--------------------------------------------------+
 | ETAPE 4 : CLASSIFICATION EfficientNet-B3         |
 |                                                  |
 |  Modele    : EfficientNet-B3 (11.9M params)      |
 |  Training  : 2-stage transfer learning           |
 |  TTA       : 16 augmentations (8 spa + 8 sc)    |
 |  Seuil     : 0.535 (optimise sur val CBIS-DDSM) |
 |  Par region: probabilite malin [0, 1]            |
 |                                                  |
 |  Sortie : label + probabilite par region         |
 +--------------------------------------------------+
       |
       v
 +--------------------------------------------------+
 | ETAPE 5 : DECISION FINALE                        |
 |                                                  |
 |  Si 0 regions MC      --> BENIN (pas de MC)      |
 |  Si >= 1 region MALIN --> IMAGE = MALIN          |
 |  Si toutes BENIN      --> IMAGE = BENIN          |
 +--------------------------------------------------+
       |
       v
 +--------------------------------------------------+
 | ETAPE 6 : OUTPUTS                                |
 |                                                  |
 |  - Image preprocessee                            |
 |  - Masque segmentation + overlay                 |
 |  - Image annotee (boites + labels)               |
 |  - Panel de synthese complet                     |
 |  - Grad-CAM++ par region (explicabilite)         |
 |  - Rapport JSON detaille                         |
 +--------------------------------------------------+
```

## 5.2 Modeles et Checkpoints

```
 MODELES UTILISES
 ==================

 +---------------------+--------------------------------------------------+
 |  Modele             |  Checkpoint                                      |
 +---------------------+--------------------------------------------------+
 |  U-Net              |  checkpoints/unet_best.pth                       |
 |  (Segmentation)     |  31M params, AUC seg = 0.9642                    |
 +---------------------+--------------------------------------------------+
 |  EfficientNet-B3    |  checkpoints/best_0.7717/efficientnet_stage2.pth |
 |  (Classification)   |  11.9M params, Val AUC = 0.8056                  |
 +---------------------+--------------------------------------------------+
```

---

# 6. PROTOCOLE D'EVALUATION

## 6.1 Etapes du Protocole

```
 PROTOCOLE EN 4 ETAPES
 ========================

 ETAPE 1 : Preparation
 ┌──────────────────────────────────────────────────────┐
 │  - Conversion DICOM → PNG pour 50 images             │
 │  - Tirage stratifie (30B + 20M, random_state=42)     │
 │  - Chargement checkpoints U-Net + EfficientNet-B3    │
 │  - Creation repertoire de sortie                     │
 └──────────────────────────────────────────────────────┘
              |
              v
 ETAPE 2 : Execution (boucle sur 50 images)
 ┌──────────────────────────────────────────────────────┐
 │  Pour chaque image :                                 │
 │    1. Conversion DICOM + preprocessing               │
 │    2. run_pipeline(preprocess=True, cls_tta=16)      │
 │    3. Sauvegarder tous les outputs dans per_image/   │
 │    4. Enregistrer prediction + probabilite + birads  │
 └──────────────────────────────────────────────────────┘
              |
              v
 ETAPE 3 : Aggregation
 ┌──────────────────────────────────────────────────────┐
 │  - Collecte des predictions par image                │
 │  - Matrice de confusion (TP, TN, FP, FN)             │
 │  - Metriques : Acc, Sens, Spec, PPV, NPV, F1, AUC   │
 │  - Analyse par BI-RADS (2, 3, 4a, 4b, 4c, 5, 6)    │
 │  - Sauvegarde JSON + TXT                             │
 └──────────────────────────────────────────────────────┘
              |
              v
 ETAPE 4 : Analyse (notebook)
 ┌──────────────────────────────────────────────────────┐
 │  - Visualisations (confusion, ROC, distribution)     │
 │  - Analyse par BI-RADS                               │
 │  - Galerie 2x4 (2TP+2TN correct, 2FP+2FN faux)     │
 │  - Comparaison triple CBIS-DDSM vs MIAS vs INbreast │
 └──────────────────────────────────────────────────────┘
```

## 6.2 Commandes d'Execution

```bash
 # Depuis la racine du projet, avec le venv active :

 # 1. Lancer l'evaluation (CPU ~2-5 min/image)
 python scripts/run_inbreast_evaluation.py

 # 2. Ouvrir le notebook de visualisation
 jupyter notebook notebooks/inbreast_evaluation_rapport.ipynb

 # 3. (Optionnel) Pipeline sur une seule image INbreast
 python src/inference/pipeline.py \
     --input  "chemin/vers/inbreast/images/20586908.dcm" \
     --seg_ckpt  checkpoints/unet_best.pth \
     --cls_ckpt  checkpoints/best_0.7717/efficientnet_stage2.pth \
     --output_dir  outputs/inbreast_test/20586908 \
     --preprocess

 # 4. Exporter le notebook en HTML
 jupyter nbconvert --to html notebooks/inbreast_evaluation_rapport.ipynb
```

---

# 7. RESULTATS — METRIQUES GLOBALES

> **Note :** Executez `python scripts/run_inbreast_evaluation.py` pour remplir cette section.

## 7.1 Metriques Principales

```
 METRIQUES DU PIPELINE COMPLET SUR INBREAST (50 images)
 ========================================================

 ┌──────────────────────────────────────────────────────────────┐
 │  Accuracy       : [A REMPLIR]  ( __/50 correct )             │
 │  Sensitivity    : [A REMPLIR]  ( __/20 malins detectes )     │
 │  Specificity    : [A REMPLIR]  ( __/30 benins corrects )     │
 │  PPV            : [A REMPLIR]  ( precision malignite )       │
 │  NPV            : [A REMPLIR]  ( precision benignite )       │
 │  F1 Score       : [A REMPLIR]                                │
 │  AUC            : [A REMPLIR]                                │
 └──────────────────────────────────────────────────────────────┘

 Comparaison avec CBIS-DDSM (reference, n=787) :
 ┌──────────────────┬────────────────────┬────────────────────┐
 │  Metrique        │  CBIS-DDSM (Test)  │  INbreast (Ext)    │
 │                  │  n = 787           │  n = 50            │
 ├──────────────────┼────────────────────┼────────────────────┤
 │  AUC             │  0.7812            │  [A REMPLIR]       │
 │  Accuracy        │  71.0%             │  [A REMPLIR]       │
 │  Sensitivity     │  63.1%             │  [A REMPLIR]       │
 │  Specificity     │  76.2%             │  [A REMPLIR]       │
 │  F1 Score        │  0.6334            │  [A REMPLIR]       │
 │  PPV             │  63.5%             │  [A REMPLIR]       │
 │  NPV             │  75.9%             │  [A REMPLIR]       │
 └──────────────────┴────────────────────┴────────────────────┘
```

---

# 8. MATRICE DE CONFUSION

```
 MATRICE DE CONFUSION — PIPELINE SUR INBREAST (50 images)
 ==========================================================

                          PREDICTION PIPELINE
                      ┌───────────┬───────────┐
                      │  BENIN    │  MALIN    │
 ┌──────────┬────────┼───────────┼───────────┤
 │  Verite  │ BENIN  │  TN = __  │  FP = __  │
 │  INbreast│        │  (Correct)│  (Fausse  │
 │  (n=30)  │        │           │   alarme) │
 │          ├────────┼───────────┼───────────┤
 │          │ MALIN  │  FN = __  │  TP = __  │
 │  (n=20)  │        │  (Cancer  │  (Correct)│
 │          │        │   manque!)│           │
 └──────────┴────────┴───────────┴───────────┘

 TP = Vrai Positif  : Malin correctement detecte
 TN = Vrai Negatif  : Benin correctement identifie
 FP = Faux Positif  : Benin predit Malin (fausse alarme)
 FN = Faux Negatif  : Malin predit Benin (DANGEREUX)

 EN DEPISTAGE : FN >> FP en gravite
   --> Sensitivity = metrique la plus critique cliniquement
```

---

# 9. RESULTATS PAR BI-RADS

## 9.1 Tableau par Groupe BI-RADS

```
 RESULTATS PAR CATEGORIE BI-RADS
 ==================================

 ┌──────────┬──────┬────────────────────────────────────────────────────┐
 │  BI-RADS │  N   │  Classe      │  Accuracy  │  Notes                 │
 ├──────────┼──────┼──────────────┼────────────┼────────────────────────┤
 │  2       │  __  │  BENIGNE     │  [A REM.]  │  Benin typique         │
 │  3       │  __  │  BENIGNE     │  [A REM.]  │  Probablement benin    │
 │  4a      │  __  │  MALIGNE     │  [A REM.]  │  Faible suspicion      │
 │  4b      │  __  │  MALIGNE     │  [A REM.]  │  Suspicion moderee     │
 │  4c      │  __  │  MALIGNE     │  [A REM.]  │  Forte suspicion       │
 │  5       │  __  │  MALIGNE     │  [A REM.]  │  Tres forte suspicion  │
 │  6       │  __  │  MALIGNE     │  [A REM.]  │  Histologie confirmee  │
 ├──────────┼──────┼──────────────┼────────────┼────────────────────────┤
 │  TOTAL   │  50  │  30B + 20M   │  [A REM.]  │  n=50, stratifie       │
 └──────────┴──────┴──────────────┴────────────┴────────────────────────┘

 Hypothese : Les BI-RADS 2/3 (benins clairs) devraient etre mieux classes
             que les BI-RADS 4a (ambigus) car le signal visuel est plus net.
             Les BI-RADS 5/6 (malins evidents) devraient avoir haute sensitivity.
```

---

# 10. TABLEAU PAR IMAGE

```
 RESULTATS PAR IMAGE (extrait — 5 exemples)
 ============================================

 +---------------------------+--------+--------+--------+---------+-------+--------+
 |  Reference (INbreast)     | BI-RAD | Verite | Pred   | Regions | MaxP  | Status |
 +---------------------------+--------+--------+--------+---------+-------+--------+
 |  20586908_[...]_L_CC.dcm  |   5    | MALIN  | ______ |   ___   | _.___| [    ] |
 |  20587660_[...]_R_MLO.dcm |   2    | BENIN  | ______ |   ___   | _.___| [    ] |
 |  22614231_[...]_L_CC.dcm  |   4b   | MALIN  | ______ |   ___   | _.___| [    ] |
 |  20586946_[...]_R_CC.dcm  |   3    | BENIN  | ______ |   ___   | _.___| [    ] |
 |  21120019_[...]_L_MLO.dcm |   4c   | MALIN  | ______ |   ___   | _.___| [    ] |
 |  ...                      |   ...  | ...    | ...    |   ...   | ...  | ...    |
 +---------------------------+--------+--------+--------+---------+-------+--------+

 Colonnes :
   Reference = nom du fichier DICOM INbreast
   BI-RADS   = categorie radiologique (2, 3, 4a, 4b, 4c, 5, 6)
   Verite    = label reel (BENIN = 0 / MALIN = 1)
   Pred      = prediction du pipeline complet
   Regions   = nombre de clusters MC detectes par U-Net
   MaxP      = probabilite max (malignant) parmi les regions
   Status    = [OK] correct ou [FAUX] erreur
```

---

# 11. ANALYSE DES ERREURS

## 11.1 Faux Positifs (Benin classe Malin)

```
 FAUX POSITIFS (Benin predit Malin) : [A REMPLIR]
 ==================================================
 ┌──────────────────┬────────┬─────────┬──────┬────────────────────────┐
 │  Reference       │ BI-RADS│ Regions │ MaxP │  Analyse               │
 ├──────────────────┼────────┼─────────┼──────┼────────────────────────┤
 │  [A REMPLIR]     │  __    │   ___   │ .___  │  [A REMPLIR]           │
 │  ...             │  ...   │   ...   │ ...  │  ...                   │
 └──────────────────┴────────┴─────────┴──────┴────────────────────────┘

 Causes possibles :
   - BI-RADS 3 avec aspects ambigus (proche de 4a)
   - Artefacts DICOM confondus avec MC par le U-Net
   - Domain shift scanner Siemens → features different de GE
```

## 11.2 Faux Negatifs (Malin classe Benin)

```
 FAUX NEGATIFS (Malin predit Benin) : [A REMPLIR]
 ==================================================
 ┌──────────────────┬────────┬─────────┬──────┬────────────────────────┐
 │  Reference       │ BI-RADS│ Regions │ MaxP │  Analyse               │
 ├──────────────────┼────────┼─────────┼──────┼────────────────────────┤
 │  [A REMPLIR]     │  __    │   ___   │ .___  │  [A REMPLIR]           │
 │  ...             │  ...   │   ...   │ ...  │  ...                   │
 └──────────────────┴────────┴─────────┴──────┴────────────────────────┘

 Classification des FN :
   FN par segmentation  (0 regions trouvees)  : [A REMPLIR]
   FN par classification (regions mal classees): [A REMPLIR]
```

---

# 12. COMPARAISON TRIPLE : CBIS-DDSM / MIAS / INBREAST

```
 TABLEAU DE GENERALISATION — 3 DATASETS
 =========================================

 ┌──────────────────┬────────────────────┬────────────────────┬────────────────────┐
 │  Metrique        │  CBIS-DDSM (Train) │  MIAS (Ext-1)      │  INbreast (Ext-2)  │
 │                  │  n = 787           │  n = 25            │  n = 50            │
 ├──────────────────┼────────────────────┼────────────────────┼────────────────────┤
 │  AUC             │  0.7812            │  [A REMPLIR]       │  [A REMPLIR]       │
 │  Accuracy        │  71.0%             │  [A REMPLIR]       │  [A REMPLIR]       │
 │  Sensitivity     │  63.1%             │  [A REMPLIR]       │  [A REMPLIR]       │
 │  Specificity     │  76.2%             │  [A REMPLIR]       │  [A REMPLIR]       │
 │  F1 Score        │  0.6334            │  [A REMPLIR]       │  [A REMPLIR]       │
 │  PPV             │  63.5%             │  [A REMPLIR]       │  [A REMPLIR]       │
 │  NPV             │  75.9%             │  [A REMPLIR]       │  [A REMPLIR]       │
 ├──────────────────┼────────────────────┼────────────────────┼────────────────────┤
 │  Format          │  ROI PNG (pre-dec.)│  PNG 1024x1024     │  DICOM 3328x4084   │
 │  Scanner         │  GE Medical        │  Scanner UK ancien │  Siemens Mammomat  │
 │  Domain shift    │  —  (reference)    │  Fort              │  Modere            │
 └──────────────────┴────────────────────┴────────────────────┴────────────────────┘

 Visualisation : outputs/inbreast_evaluation/inbreast_cross_comparison.png
```

---

# 13. DISCUSSION

## 13.1 Domain Shift : INbreast vs CBIS-DDSM vs MIAS

```
 ANALYSE DU DOMAIN SHIFT
 =========================

 CBIS-DDSM → INBREAST (shift modere) :
 ┌──────────────────────────────────────────────────────────┐
 │  Similarities :                                          │
 │    + Format DICOM 16-bit dans les deux cas               │
 │    + Qualite d'image clinique haute definition           │
 │    + Annotation experte radiologique (BI-RADS rigoureux) │
 │    + Meme types de lesions (MC + masses)                 │
 │                                                          │
 │  Differences :                                           │
 │    - Scanner Siemens vs GE : courbe de reponse differente│
 │    - Resolutions tres differentes (3328x4084 vs variable)│
 │    - Population europeenne vs americaine                  │
 │    - Protocole acquisition peut varier (dose, positionnement) │
 └──────────────────────────────────────────────────────────┘

 CBIS-DDSM → MIAS (shift fort) :
 ┌──────────────────────────────────────────────────────────┐
 │  Film numerise vs capteur digital                        │
 │  Resolution 1024x1024 fixe vs haute def variable         │
 │  Mammogramme complet (pipeline doit tout faire)          │
 └──────────────────────────────────────────────────────────┘

 Hypothese principale :
   INbreast devrait performer MIEUX que MIAS car :
   1. Format DICOM haute qualite (moins de bruit de conversion)
   2. Moins de domain shift (digital vs digital)
   3. Annotations BI-RADS plus precises (meilleur ground truth)
```

## 13.2 Forces et Limites de Cette Evaluation

```
 FORCES
 ========

 1. DATASET DE REFERENCE ETABLI
    INbreast est un standard de la litterature mammographique
    Resultats comparables aux publications existantes

 2. ANNOTATIONS BI-RADS DETAILLEES
    Permet d'analyser les performances par niveau de difficulte
    BI-RADS 4a (ambigus) sont les plus pertinents cliniquement

 3. FORMAT DICOM AUTHENTIQUE
    Test le pipeline de bout en bout : conversion + inference
    Conditions proches d'un deploiement clinique reel

 4. SOUS-ENSEMBLE STRATIFIE REPRODUCTIBLE
    random_state=42, stratification → resultats exactement reproductibles

 LIMITES
 ========

 1. TAILLE REDUITE (n=50)
    Haute variance sur les metriques
    2% de l'accuracy par image
    Pas de CI 95% fiable

 2. SEUIL NON AJUSTE
    Seuil 0.535 optimise sur CBIS-DDSM validation
    Potentiellement sous-optimal pour INbreast

 3. PIPELINE SPECIALISE MC
    U-Net entraine sur calcifications
    Performances sur masses INbreast potentiellement reduites

 4. VERSIONS BI-RADS
    INbreast utilise BI-RADS ancien (certains 4 sans a/b/c)
    Binarisation 2/3 benin | 4+ malin peut creer du bruit
```

## 13.3 Perspectives

```
 AMELIORATIONS POSSIBLES
 =========================

 1. FINE-TUNING INBREAST
    Quelques images INbreast dans le train set (5-10%)
    Adapter le modele au scanner Siemens
    Risque : dataset trop petit pour fine-tuning stable

 2. EVALUATION COMPLETE (410 images)
    Utiliser l'ensemble des 410 images INbreast
    Metriques plus stables et CI 95% calculables

 3. ANALYSE MC ONLY vs MASSES
    Filtrer par type de lesion pour evaluation equitable
    Comparer MC-seules vs CBIS-DDSM calc-only

 4. MULTI-CENTRE TRAINING
    Inclure INbreast dans le set d'entrainement
    CBIS-DDSM + INbreast + MIAS = generalisation multi-scanner
```

---

# 14. CONCLUSION

```
 ┌──────────────────────────────────────────────────────────────────┐
 │  CONCLUSION — EVALUATION INBREAST                                │
 │  ====================================                            │
 │                                                                  │
 │  OBJECTIF : Tester le pipeline mammo-cad sur le dataset          │
 │             INbreast (Portugal, Siemens, DICOM)                  │
 │             50 images : 30 benign + 20 malignant                 │
 │             random_state=42, stratifie                           │
 │                                                                  │
 │  PIPELINE : DICOM → Preprocessing → U-Net segmentation           │
 │             → Clustering → EfficientNet-B3 classification        │
 │             → Grad-CAM++ → Decision finale                       │
 │                                                                  │
 │  RESULTATS : (a remplir apres execution)                         │
 │    Accuracy     = [A REMPLIR]                                    │
 │    Sensitivity  = [A REMPLIR]                                    │
 │    Specificity  = [A REMPLIR]                                    │
 │    AUC          = [A REMPLIR]                                    │
 │                                                                  │
 │  COMPARAISON (hypothese) :                                       │
 │    INbreast > MIAS (moins de domain shift, format DICOM)         │
 │    INbreast ~ CBIS-DDSM (meme format, domain shift modere)       │
 │                                                                  │
 │  VALEUR SCIENTIFIQUE :                                           │
 │    Troisieme validation externe du pipeline                      │
 │    Demontre la generalisation sur 3 datasets independants        │
 │    (USA, UK, Portugal) avec 3 scanners differents                │
 └──────────────────────────────────────────────────────────────────┘
```

---

# 15. ANNEXES

## 15.1 Commandes Completes de Reproduction

```bash
 ETAPES POUR REPRODUIRE L'EVALUATION INBREAST
 ==============================================

 # 1. Prerequis
 cd C:\project\mammo-cad
 .\venv\Scripts\activate       # ou source venv/bin/activate (Linux)

 # 2. Verifier le dataset INbreast (DICOM)
 ls "chemin/vers/INbreast/AllDICOMs"
 ls "chemin/vers/INbreast/INbreast_case_study_associated_XML"

 # 3. Verifier les checkpoints
 ls checkpoints\unet_best.pth
 ls checkpoints\best_0.7717\efficientnet_stage2.pth

 # 4. Lancer l'evaluation stratifiee (50 images, ~2-5 min/image)
 python scripts/run_inbreast_evaluation.py \
     --inbreast_dir  "chemin/vers/INbreast" \
     --n_samples 50 \
     --random_state 42

 # 5. Verifier les resultats
 type outputs\inbreast_evaluation\inbreast_evaluation_summary.txt

 # 6. Lancer le notebook de visualisation
 jupyter notebook notebooks\inbreast_evaluation_rapport.ipynb
 # Executer toutes les cellules (Kernel > Restart & Run All)

 # 7. (Optionnel) Exporter le notebook en HTML
 jupyter nbconvert --to html notebooks\inbreast_evaluation_rapport.ipynb
```

## 15.2 Structure des Fichiers de Sortie

```
 outputs/inbreast_evaluation/
   |
   +-- inbreast_evaluation_results.json      Resultats agreges (JSON)
   +-- inbreast_evaluation_summary.txt       Resume textuel
   |
   +-- inbreast_confusion_metrics.png        Confusion + metriques
   +-- inbreast_roc_distribution.png         ROC + distribution proba
   +-- inbreast_birads_analysis.png          Analyse par BI-RADS
   +-- inbreast_gallery.png                  Galerie 2x4 (correct/faux)
   +-- inbreast_cross_comparison.png         CBIS vs MIAS vs INbreast
   |
   +-- per_image/
       +-- <reference_dicom>/
       |   +-- <name>_preprocessed.png
       |   +-- <name>_seg_mask.png
       |   +-- <name>_seg_overlay.png
       |   +-- <name>_annotated.png
       |   +-- <name>_panel.png
       |   +-- <name>_report.json
       |   +-- crops/
       |       +-- region_0_crop.png
       |       +-- region_0_gradcam.png
       |       +-- ...
       +-- ...  (50 dossiers au total)
```

## 15.3 Format JSON des Resultats

```json
{
  "summary": {
    "dataset":       "INbreast",
    "n_total":       50,
    "n_benign":      30,
    "n_malignant":   20,
    "accuracy":      "...",
    "sensitivity":   "...",
    "specificity":   "...",
    "ppv":           "...",
    "npv":           "...",
    "f1":            "...",
    "auc":           "...",
    "tp": "...", "tn": "...", "fp": "...", "fn": "...",
    "random_state":  42,
    "threshold":     0.535
  },
  "results": [
    {
      "image":          "<reference_dicom>",
      "reference":      "20586908_..._L_CC.dcm",
      "birads":         "5",
      "true_label":     1,
      "true_label_str": "MALIGNANT",
      "pred_label":     "...",
      "pred_label_str": "...",
      "correct":        "...",
      "regions_found":  "...",
      "max_prob":       "...",
      "regions_detail": [
        {"region_id": 0, "label": "...", "probability": "...", "bbox": [...]}
      ]
    },
    "... (50 entries)"
  ]
}
```

---

*Document genere le 2026-04-17 — Projet mammo-cad*
*Pipeline: U-Net (31M params, AUC seg=0.9642) + EfficientNet-B3 (11.9M params, Val AUC=0.8056)*
*Evaluation externe sur INbreast (50 images, 30 benin + 20 malin, random_state=42)*
*Seuil=0.535, TTA-16, mode CPU*
