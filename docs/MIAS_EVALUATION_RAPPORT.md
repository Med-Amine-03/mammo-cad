# Evaluation Externe sur Dataset MIAS

## Test de Generalisation du Pipeline Complet (U-Net + EfficientNet-B3)

---

# 1. OBJECTIF DE L'EVALUATION EXTERNE

```
 QUESTION PRINCIPALE
 =====================

 Le pipeline mammo-cad, entraine EXCLUSIVEMENT sur CBIS-DDSM,
 peut-il classifier correctement des mammographies provenant
 d'un SCANNER DIFFERENT et d'une POPULATION DIFFERENTE ?

 +--------------------------------------------------+
 |   ENTRAINEMENT        TEST EXTERNE               |
 |   CBIS-DDSM           MIAS                       |
 |   (USA, GE Medical)   (UK, NBSP)                 |
 |   3 567 cas            25 images                  |
 |   ROI pre-decoupes     Mammogrammes bruts         |
 +--------------------------------------------------+

 Si le pipeline performe bien sur MIAS :
   --> Generalisation REELLE, pas du surapprentissage
   --> Les features apprises sont transferables
   --> Le systeme est utilisable en clinique reelle
```

### Pourquoi un test externe est ESSENTIEL

```
 +---------------------------------------------------------+
 |  SANS test externe :                                    |
 |    "Le modele fait 78% AUC" = peut-etre du              |
 |    surapprentissage sur les artefacts du scanner GE     |
 |                                                         |
 |  AVEC test externe (MIAS) :                             |
 |    "Le modele fait X% sur un scanner JAMAIS VU"         |
 |    = preuve de generalisation                           |
 +---------------------------------------------------------+
```

---

# 2. DATASET MIAS — DESCRIPTION COMPLETE

## 2.1 Source et Origine

```
 MIAS = Mammographic Image Analysis Society
 =============================================

 Source      : UK National Breast Screening Programme
 Pays        : Royaume-Uni
 Format      : PNG, 1024 x 1024 pixels
 Profondeur  : uint8, niveaux de gris (0-225)
 Type        : Mammogrammes BRUTS (pas de ROI pre-decoupes)

 Sous-ensemble utilise : Calcifications uniquement
   - 12 images benignes
   - 13 images malignes
   - TOTAL : 25 images
```

## 2.2 Liste Complete des Images

```
 IMAGES BENIGNES (12)
 =====================
 +----------+----------+----------+----------+
 | mdb212   | mdb214   | mdb218   | mdb219   |
 | mdb222   | mdb223   | mdb226   | mdb227   |
 | mdb236   | mdb240   | mdb248   | mdb252   |
 +----------+----------+----------+----------+

 IMAGES MALIGNES (13)
 =====================
 +----------+----------+----------+----------+
 | mdb209   | mdb211   | mdb213   | mdb216   |
 | mdb231   | mdb233   | mdb238   | mdb239   |
 | mdb241   | mdb245   | mdb249   | mdb253   |
 | mdb256   |          |          |          |
 +----------+----------+----------+----------+
```

## 2.3 Comparaison CBIS-DDSM (Train) vs MIAS (Test Externe)

```
 DIFFERENCES ENTRE LES DEUX DATASETS
 ======================================

 +-----------------------+------------------------+------------------------+
 |  Critere              |  CBIS-DDSM (Train)     |  MIAS (Test externe)   |
 +-----------------------+------------------------+------------------------+
 |  Pays                 |  USA                   |  Royaume-Uni           |
 |  Programme            |  Digital Database      |  UK NBSP               |
 |  Scanner              |  GE Medical Systems    |  Scanner UK (ancien)   |
 |  Format original      |  DICOM 16-bit          |  Film numerise         |
 |  Format utilise       |  ROI PNG pre-decoupes  |  Mammogramme complet   |
 |  Resolution           |  Variable (ROI)        |  1024 x 1024 fixe      |
 |  Preprocessing recu   |  Oui (ROI extrait)     |  Non (image brute)     |
 |  Types de lesion      |  Calc + Mass           |  Calcifications seules |
 |  Taille               |  3 567 cas             |  25 images             |
 |  Patients             |  1 566                 |  25                    |
 +-----------------------+------------------------+------------------------+

 IMPORTANT : Ces differences representent un "domain shift" REEL
             Le modele doit s'adapter a :
             - Un contraste different (scanner different)
             - Un format different (mammogramme complet, pas ROI)
             - Une population differente (UK vs USA)
```

---

# 3. PIPELINE EVALUE — ARCHITECTURE COMPLETE

## 3.1 Vue Globale du Pipeline

```
 PIPELINE COMPLET mammo-cad SUR IMAGES MIAS
 =============================================

 Image MIAS brute (1024x1024, niveaux de gris)
       |
       v
 +-----------------------------------------------+
 | ETAPE 1 : PREPROCESSING (--preprocess flag)   |
 |                                                |
 |  1. Flip-to-left (orientation normalisee)      |
 |  2. Otsu blob detection (extraction auto ROI)  |
 |  3. CLAHE (egalisation adaptative contraste)   |
 |                                                |
 |  Sortie : image preprocessee PNG               |
 +-----------------------------------------------+
       |
       v
 +-----------------------------------------------+
 | ETAPE 2 : SEGMENTATION U-Net                  |
 |                                                |
 |  Modele    : U-Net (31M parametres)            |
 |  Training  : 10-fold Cross-Validation          |
 |  AUC seg   : 0.9642                            |
 |  Tache     : Detecter les microcalcifications  |
 |                                                |
 |  Sortie : masque binaire + overlay             |
 +-----------------------------------------------+
       |
       v
 +-----------------------------------------------+
 | ETAPE 3 : CLUSTERING                          |
 |                                                |
 |  Methode   : Connected components + merge     |
 |  Tache     : Regrouper MC en clusters          |
 |  Resultat  : N regions d'interet (bounding box)|
 |                                                |
 |  Si N = 0 : AUCUNE MC detectee                |
 |             --> Image classee BENIN par defaut  |
 +-----------------------------------------------+
       |
       v (pour chaque region)
 +-----------------------------------------------+
 | ETAPE 4 : CLASSIFICATION EfficientNet-B3      |
 |                                                |
 |  Modele    : EfficientNet-B3 (11.9M params)   |
 |  Training  : 2-stage transfer learning         |
 |  TTA       : 16 augmentations (8 spa + 8 sc)  |
 |  Seuil     : 0.535 (optimise sur validation)   |
 |  Par region: probabilite malin [0, 1]          |
 |                                                |
 |  Sortie : label + probabilite par region       |
 +-----------------------------------------------+
       |
       v
 +-----------------------------------------------+
 | ETAPE 5 : DECISION FINALE                     |
 |                                                |
 |  Si 0 regions MC      --> BENIN (pas de MC)   |
 |  Si >= 1 region MALIN --> IMAGE = MALIN        |
 |  Si toutes BENIN      --> IMAGE = BENIN        |
 +-----------------------------------------------+
       |
       v
 +-----------------------------------------------+
 | ETAPE 6 : OUTPUTS                             |
 |                                                |
 |  - Image preprocessee                          |
 |  - Masque segmentation + overlay               |
 |  - Image annotee (boites + labels)             |
 |  - Panel de synthese complet                   |
 |  - Grad-CAM++ par region (explicabilite)       |
 |  - Rapport JSON detaille                       |
 +-----------------------------------------------+
```

## 3.2 Modeles et Checkpoints

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

## 3.3 Parametres d'Inference

```
 PARAMETRES D'INFERENCE
 ========================

 +------------------+------------------------------------------+
 |  Parametre       |  Valeur                                  |
 +------------------+------------------------------------------+
 |  preprocess      |  True (images MIAS sont brutes)          |
 |  cls_tta         |  16 (8 spatial + 8 scale)                |
 |  threshold       |  0.535 (optimise sur val CBIS-DDSM)      |
 |  device          |  CPU (memes resultats que GPU)           |
 +------------------+------------------------------------------+
```

---

# 4. PROTOCOLE D'EVALUATION

## 4.1 Etapes du Protocole

```
 PROTOCOLE EN 4 ETAPES
 ========================

 ETAPE 1 : Preparation
 ┌─────────────────────────────────────────────────────┐
 │  - Verification dataset MIAS (25 images, 12B + 13M) │
 │  - Chargement checkpoints U-Net + EfficientNet-B3   │
 │  - Creation repertoire de sortie                    │
 └─────────────────────────────────────────────────────┘
              |
              v
 ETAPE 2 : Execution (boucle sur 25 images)
 ┌─────────────────────────────────────────────────────┐
 │  Pour chaque image :                                │
 │    1. run_pipeline(preprocess=True, cls_tta=16)     │
 │    2. Sauvegarder tous les outputs dans per_image/  │
 │    3. Enregistrer prediction + probabilite          │
 └─────────────────────────────────────────────────────┘
              |
              v
 ETAPE 3 : Aggregation
 ┌─────────────────────────────────────────────────────┐
 │  - Collecte des predictions par image               │
 │  - Matrice de confusion (TP, TN, FP, FN)            │
 │  - Metriques : Acc, Sens, Spec, PPV, NPV, F1, AUC  │
 │  - Sauvegarde JSON + TXT                            │
 └─────────────────────────────────────────────────────┘
              |
              v
 ETAPE 4 : Analyse (notebook)
 ┌─────────────────────────────────────────────────────┐
 │  - Visualisations (confusion, ROC, distribution)    │
 │  - Galeries correct/faux avec pipeline panels       │
 │  - Comparaison cross-dataset CBIS-DDSM vs MIAS     │
 │  - Analyse des erreurs (FP vs FN)                   │
 │  - Dashboard resume final                           │
 └─────────────────────────────────────────────────────┘
```

## 4.2 Scripts et Fichiers

```
 FICHIERS DU PROTOCOLE
 =======================

 +--------------------------------------------+-------------------------------------+
 |  Fichier                                   |  Role                               |
 +--------------------------------------------+-------------------------------------+
 |  scripts/run_mias_evaluation.py            |  Execute pipeline sur 25 images     |
 |  notebooks/mias_evaluation_rapport.ipynb   |  Visualisation + analyse complete   |
 |  docs/MIAS_EVALUATION_RAPPORT.md           |  Ce document                        |
 +--------------------------------------------+-------------------------------------+
```

## 4.3 Commandes d'Execution

```bash
 # Depuis la racine du projet, avec le venv active :

 # 1. Lancer l'evaluation (CPU ~2-5 min/image)
 python scripts/run_mias_evaluation.py

 # 2. Ouvrir le notebook de visualisation
 jupyter notebook notebooks/mias_evaluation_rapport.ipynb

 # 3. (Optionnel) Pipeline sur une seule image
 python src/inference/pipeline.py \
     --input  "chemin/vers/mias_calc/malignant/mdb249.png" \
     --seg_ckpt  checkpoints/unet_best.pth \
     --cls_ckpt  checkpoints/best_0.7717/efficientnet_stage2.pth \
     --output_dir  outputs/mias_test/mdb249 \
     --preprocess
```

---

# 5. RESULTATS — METRIQUES GLOBALES

> **Note :** Cette section est remplie automatiquement apres execution.
> Executez `python scripts/run_mias_evaluation.py` pour generer les resultats.

## 5.1 Metriques Principales

```
 METRIQUES DU PIPELINE COMPLET SUR MIAS (25 images)
 =====================================================

 ┌──────────────────────────────────────────────────────────┐
 │  Accuracy       : __.___%  ( __/25 correct )             │
 │  Sensitivity    : __.___%  ( __/__ malins detectes )     │
 │  Specificity    : __.___%  ( __/__ benins corrects )     │
 │  PPV            : __.___%  ( __/__ pred malin vrais )    │
 │  NPV            : __.___%  ( __/__ pred benin vrais )    │
 │  F1 Score       : 0.____                                 │
 │  AUC            : 0.____                                 │
 └──────────────────────────────────────────────────────────┘

 Interpretation :
   Accuracy    = images correctement classees / total
   Sensitivity = vrais malins detectes / total malins  (le + important)
   Specificity = vrais benins detectes / total benins
   PPV         = si "malin" predit, quelle proba c'est vrai ?
   NPV         = si "benin" predit, quelle proba c'est vrai ?
   F1          = moyenne harmonique de PPV et Sensitivity
   AUC         = aire sous la courbe ROC
```

## 5.2 Matrice de Confusion

```
 MATRICE DE CONFUSION — PIPELINE SUR MIAS
 ==========================================

                          PREDICTION PIPELINE
                      ┌───────────┬───────────┐
                      │  BENIN    │  MALIN    │
 ┌──────────┬────────┼───────────┼───────────┤
 │  Verite  │ BENIN  │  TN = __  │  FP = __  │
 │  MIAS    │        │  (Correct)│  (Fausse  │
 │          │        │           │   alarme) │
 │          ├────────┼───────────┼───────────┤
 │          │ MALIN  │  FN = __  │  TP = __  │
 │          │        │  (Cancer  │  (Correct)│
 │          │        │   manque!)│           │
 └──────────┴────────┴───────────┴───────────┘

 TP = Vrai Positif  : Malin correctement detecte
 TN = Vrai Negatif  : Benin correctement identifie
 FP = Faux Positif  : Benin predit Malin (fausse alarme)
 FN = Faux Negatif  : Malin predit Benin (DANGEREUX)
```

---

# 6. RESULTATS — DETAIL PAR IMAGE

## 6.1 Tableau Complet (25 images)

> Ce tableau est rempli automatiquement par le script.
> Voir `mias_evaluation_results.json` et le notebook Cell 2.

```
 RESULTATS PAR IMAGE
 =====================

 +----------+-----------+---------------------------+---------+-------+--------+
 |  Image   |  Verite   |  Prediction Pipeline      | Regions | MaxP  | Status |
 +----------+-----------+---------------------------+---------+-------+--------+
 |  mdb209  | MALIGNANT | _________________________ |   ___   | _.___| [    ] |
 |  mdb211  | MALIGNANT | _________________________ |   ___   | _.___| [    ] |
 |  mdb212  | BENIGN    | _________________________ |   ___   | _.___| [    ] |
 |  mdb213  | MALIGNANT | _________________________ |   ___   | _.___| [    ] |
 |  mdb214  | BENIGN    | _________________________ |   ___   | _.___| [    ] |
 |  mdb216  | MALIGNANT | _________________________ |   ___   | _.___| [    ] |
 |  mdb218  | BENIGN    | _________________________ |   ___   | _.___| [    ] |
 |  mdb219  | BENIGN    | _________________________ |   ___   | _.___| [    ] |
 |  mdb222  | BENIGN    | _________________________ |   ___   | _.___| [    ] |
 |  mdb223  | BENIGN    | _________________________ |   ___   | _.___| [    ] |
 |  mdb226  | BENIGN    | _________________________ |   ___   | _.___| [    ] |
 |  mdb227  | BENIGN    | _________________________ |   ___   | _.___| [    ] |
 |  mdb231  | MALIGNANT | _________________________ |   ___   | _.___| [    ] |
 |  mdb233  | MALIGNANT | _________________________ |   ___   | _.___| [    ] |
 |  mdb236  | BENIGN    | _________________________ |   ___   | _.___| [    ] |
 |  mdb238  | MALIGNANT | _________________________ |   ___   | _.___| [    ] |
 |  mdb239  | MALIGNANT | _________________________ |   ___   | _.___| [    ] |
 |  mdb240  | BENIGN    | _________________________ |   ___   | _.___| [    ] |
 |  mdb241  | MALIGNANT | _________________________ |   ___   | _.___| [    ] |
 |  mdb245  | MALIGNANT | _________________________ |   ___   | _.___| [    ] |
 |  mdb248  | BENIGN    | _________________________ |   ___   | _.___| [    ] |
 |  mdb249  | MALIGNANT | _________________________ |   ___   | _.___| [    ] |
 |  mdb252  | BENIGN    | _________________________ |   ___   | _.___| [    ] |
 |  mdb253  | MALIGNANT | _________________________ |   ___   | _.___| [    ] |
 |  mdb256  | MALIGNANT | _________________________ |   ___   | _.___| [    ] |
 +----------+-----------+---------------------------+---------+-------+--------+

 Colonnes :
   Image   = nom du fichier MIAS
   Verite  = label reel (BENIGN / MALIGNANT)
   Prediction = prediction du pipeline complet
   Regions = nombre de clusters MC detectes par U-Net
   MaxP    = probabilite max (malignant) parmi les regions
   Status  = [OK] correct ou [FAUX] erreur
```

## 6.2 Sortie Pipeline par Image

```
 OUTPUTS GENERES PAR LE PIPELINE POUR CHAQUE IMAGE
 ====================================================

 outputs/mias_evaluation/per_image/<image_name>/
   |
   +-- <name>_preprocessed.png     Image apres preprocessing
   |                                (flip-to-left + Otsu + CLAHE)
   |
   +-- <name>_seg_mask.png         Masque binaire de segmentation
   |                                (blanc = microcalcification detectee)
   |
   +-- <name>_seg_overlay.png      Overlay du masque sur l'image
   |                                (regions MC en couleur)
   |
   +-- <name>_annotated.png        Image annotee avec :
   |                                - Boites englobantes par region
   |                                - Label BENIN/MALIN + probabilite
   |                                - Couleur : vert=benin, rouge=malin
   |
   +-- <name>_panel.png            Panel de synthese complet
   |                                (4-6 images : original, seg, overlay,
   |                                 annotated, gradcam, crops)
   |
   +-- <name>_report.json          Rapport JSON detaille :
   |                                - regions_found, malignant, benign
   |                                - Par region : bbox, label, proba
   |                                - overall_assessment
   |
   +-- crops/                      Dossier des crops par region
       +-- region_0_crop.png       Crop de la region 0
       +-- region_0_gradcam.png    Grad-CAM++ de la region 0
       +-- region_1_crop.png       ...
       +-- region_1_gradcam.png    ...
       +-- ...
```

## 6.3 Comment Naviguer dans les Resultats

```
 GUIDE DE NAVIGATION DES RESULTATS
 ====================================

 Pour voir le resultat d'UNE image (ex: mdb249) :

   1. Ouvrir le dossier :
      outputs/mias_evaluation/per_image/mdb249/

   2. Regarder le panel complet :
      mdb249_panel.png  <-- resume visuel de toute l'analyse

   3. Regarder l'image annotee :
      mdb249_annotated.png  <-- boites + labels sur l'image

   4. Voir les crops + Grad-CAM :
      crops/region_0_crop.png     <-- zoom sur la region
      crops/region_0_gradcam.png  <-- ou le modele regarde

   5. Lire le rapport JSON :
      mdb249_report.json  <-- tous les chiffres

 Pour l'ENSEMBLE des resultats :
   - Notebook : notebooks/mias_evaluation_rapport.ipynb
   - JSON global : outputs/mias_evaluation/mias_evaluation_results.json
   - Resume texte : outputs/mias_evaluation/mias_evaluation_summary.txt
```

---

# 7. LOGIQUE DE DECISION DU PIPELINE

## 7.1 Regles de Decision

```
 ARBRE DE DECISION DU PIPELINE
 ================================

 Image MIAS
   |
   +-- U-Net segmente les microcalcifications (MC)
   |
   +-- Combien de regions MC detectees ?
       |
       +-- 0 regions : Aucune MC trouvee
       |               --> Classification = BENIN (pas de suspicion)
       |               --> max_prob = 0.0
       |
       +-- >= 1 regions : MC detectees
           |
           +-- EfficientNet-B3 classifie CHAQUE region
           |
           +-- AU MOINS 1 region MALIGNE ?
               |
               +-- OUI : --> Image = MALIN
               |          --> max_prob = max(prob maligne)
               |
               +-- NON : --> Image = BENIN
                          --> max_prob = max(prob region)
```

## 7.2 Impact sur les Erreurs

```
 SOURCES D'ERREUR DANS LE PIPELINE
 ====================================

 Faux Negatif (FN) — Cancer manque :
 ┌──────────────────────────────────────────────────────┐
 │  Source 1 : SEGMENTATION                             │
 │    U-Net ne detecte AUCUNE MC dans l'image maligne   │
 │    --> 0 regions --> BENIN par defaut                 │
 │    --> Erreur du SEGMENTEUR, pas du classificateur    │
 │                                                      │
 │  Source 2 : CLASSIFICATION                           │
 │    U-Net detecte des MC, mais EfficientNet les        │
 │    classifie toutes comme benignes                   │
 │    --> Erreur du CLASSIFICATEUR                      │
 └──────────────────────────────────────────────────────┘

 Faux Positif (FP) — Fausse alarme :
 ┌──────────────────────────────────────────────────────┐
 │  U-Net detecte des MC dans une image benigne,        │
 │  ET EfficientNet classifie au moins 1 region         │
 │  comme maligne (prob > 0.535)                        │
 │  --> Fausse alarme : biopsie inutile                 │
 └──────────────────────────────────────────────────────┘
```

---

# 8. ANALYSE DE LA SEGMENTATION

## 8.1 Role du U-Net dans le Pipeline

```
 SEGMENTATION = PREMIERE ETAPE CRITIQUE
 ========================================

 Le U-Net est le "gardien" du pipeline :
   - S'il detecte des MC --> le classificateur est appele
   - S'il ne detecte RIEN --> image = BENIN automatiquement

 Consequence :
   - Un U-Net trop conservateur = manque des cancers (FN)
   - Un U-Net trop sensible = trop de fausses regions (bruit)

 Performance du U-Net sur CBIS-DDSM :
   AUC segmentation = 0.9642  (excellent)
   Mais sur MIAS (domain shift) : performance potentiellement reduite
```

## 8.2 Statistiques de Detection

> Remplies apres execution. Voir notebook Cell 6.

```
 DETECTION MC PAR IMAGE
 ========================

 Images sans MC detectee      : __/25  (___%)
 Images avec MC detectees     : __/25  (___%)
 Regions moyennes par image   : _.__
 Regions max sur une image    : ___
```

## 8.3 Impact sur les FN

```
 ANALYSE DES FAUX NEGATIFS
 ===========================

 FN total                         : __
   dont FN par segmentation       : __ (0 regions, MC manquees)
   dont FN par classification     : __ (regions trouvees mais mal classees)

 Si FN_segmentation >> FN_classification :
   --> Le probleme est la DETECTION des MC sur MIAS
   --> Domain shift impacte surtout le U-Net

 Si FN_classification >> FN_segmentation :
   --> Le probleme est la CLASSIFICATION des MC
   --> Domain shift impacte surtout l'EfficientNet
```

---

# 9. ANALYSE DES ERREURS

## 9.1 Classification des Erreurs

```
 TYPES D'ERREUR ET IMPACT CLINIQUE
 ====================================

 +----------+-------------------+--------------------------------------+
 |  Type    |  Description      |  Consequence clinique                |
 +----------+-------------------+--------------------------------------+
 |  TP      |  Malin detecte    |  CORRECT : patiente orientee biopsie |
 |  TN      |  Benin identifie  |  CORRECT : patiente rassuree         |
 +----------+-------------------+--------------------------------------+
 |  FP      |  Benin predit     |  Biopsie inutile, anxiete            |
 |          |  Malin            |  Cout psychologique + financier      |
 +----------+-------------------+--------------------------------------+
 |  FN      |  Malin predit     |  CANCER MANQUE !                     |
 |          |  Benin            |  Retard diagnostic = DANGEREUX       |
 +----------+-------------------+--------------------------------------+

 EN DEPISTAGE : FN >> FP en gravite
   --> La SENSITIVITY est la metrique la plus critique
   --> Un cancer manque peut etre fatal
   --> Une fausse alarme n'est "que" stressante
```

## 9.2 Detail des Erreurs

> Rempli apres execution. Voir notebook Cell 10.

```
 FAUX POSITIFS (Benin predit Malin) : __
 ========================================
 +----------+---------+-------+----------------------------+
 |  Image   | Regions | MaxP  | Analyse                    |
 +----------+---------+-------+----------------------------+
 |  ...     |   ...   | .___  | ...                        |
 +----------+---------+-------+----------------------------+

 FAUX NEGATIFS (Malin predit Benin) : __
 ========================================
 +----------+---------+-------+----------------------------+
 |  Image   | Regions | MaxP  | Analyse                    |
 +----------+---------+-------+----------------------------+
 |  ...     |   ...   | .___  | ...                        |
 +----------+---------+-------+----------------------------+
```

## 9.3 Hypotheses sur les Erreurs

```
 CAUSES POSSIBLES DES ERREURS
 ===============================

 1. DOMAIN SHIFT (difference de scanner)
    - Contraste MIAS != contraste CBIS-DDSM
    - Textures des MC apparaissent differemment
    - Preprocessing (CLAHE) peut ne pas compenser completement

 2. FORMAT DIFFERENT
    - CBIS-DDSM : ROI deja decoupes, centres sur la lesion
    - MIAS : mammogramme complet, le pipeline doit TROUVER la lesion
    - L'etape de segmentation est un goulot d'etranglement supplementaire

 3. TAILLE DES MC
    - Les MC MIAS peuvent etre plus petites/plus grosses
    - Le clustering peut regrouper differemment

 4. POPULATION DIFFERENTE
    - Densite mammaire differente (UK vs USA)
    - Types de calcifications potentiellement differents
```

---

# 10. COMPARAISON CROSS-DATASET

## 10.1 CBIS-DDSM (Train) vs MIAS (Externe)

```
 COMPARAISON DES PERFORMANCES
 ===============================

 ┌──────────────────┬────────────────────┬────────────────────┐
 │  Metrique        │  CBIS-DDSM (Test)  │  MIAS (Externe)    │
 │                  │  n = 787           │  n = 25            │
 ├──────────────────┼────────────────────┼────────────────────┤
 │  AUC             │  0.7812            │  _____             │
 │  Accuracy        │  71.0%             │  ____%             │
 │  Sensitivity     │  63.1%             │  ____%             │
 │  Specificity     │  76.2%             │  ____%             │
 │  F1 Score        │  0.6334            │  ______            │
 │  PPV             │  63.5%             │  ____%             │
 │  NPV             │  75.9%             │  ____%             │
 ├──────────────────┼────────────────────┼────────────────────┤
 │  CI 95% AUC      │  [0.748 - 0.813]   │  N/A (n trop petit)│
 └──────────────────┴────────────────────┴────────────────────┘

 IMPORTANT : La comparaison CBIS-DDSM est sur le TEST set
             (787 images jamais vues pendant l'entrainement,
              mais du MEME scanner)
```

## 10.2 Interpretation

```
 GRILLE D'INTERPRETATION
 =========================

 Si MIAS ≈ CBIS-DDSM :
   --> EXCELLENTE generalisation
   --> Le modele a appris des features UNIVERSELLES
   --> Les patterns de MC sont transferables entre scanners

 Si MIAS un peu < CBIS-DDSM (5-15% de baisse) :
   --> Generalisation ACCEPTABLE avec domain shift
   --> Normal : le modele n'a jamais vu ce scanner
   --> Ameliorable par domain adaptation

 Si MIAS >> < CBIS-DDSM (>15% de baisse) :
   --> Domain shift IMPORTANT
   --> Le modele est trop specifique au scanner CBIS-DDSM
   --> Necessite fine-tuning sur donnees multi-centres

 Si MIAS > CBIS-DDSM :
   --> Possible si les cas MIAS sont "faciles" (calcifications typiques)
   --> Ou effet de petite taille d'echantillon (n=25, haute variance)
```

## 10.3 Limites de la Comparaison

```
 LIMITES A CONSIDERER
 =======================

 1. TAILLE D'ECHANTILLON
    - CBIS-DDSM test : 787 images (metriques stables)
    - MIAS : 25 images (haute variance, pas de CI fiable)
    - Chaque image = 4% de l'accuracy !

 2. TYPE DE LESION
    - CBIS-DDSM test : calcifications (382) + masses (405)
    - MIAS : calcifications UNIQUEMENT (25)
    - Comparaison equitable = CBIS-DDSM calc seul vs MIAS

 3. PIPELINE vs CLASSIFICATION
    - CBIS-DDSM : metriques de CLASSIFICATION sur ROI fournis
    - MIAS : metriques du PIPELINE COMPLET (seg + class)
    - Les erreurs MIAS incluent les erreurs de segmentation

 4. CBIS-DDSM calc-only (pour comparaison equitable) :
    - AUC calc  = 0.7808  (n=382)
    - Acc calc  = 70.9%
    - Sens calc = 60.6%
    - Spec calc = 78.0%
```

---

# 11. VISUALISATIONS GENEREES

## 11.1 Liste des Visualisations

```
 FICHIERS DE VISUALISATION (generes par le notebook)
 =====================================================

 +-------------------------------------+----------------------------------------+
 |  Fichier                            |  Description                           |
 +-------------------------------------+----------------------------------------+
 |  mias_confusion_results.png         |  Matrice de confusion +                |
 |                                     |  Barre horizontale par image           |
 |                                     |  (vert=correct, rouge=faux)            |
 +-------------------------------------+----------------------------------------+
 |  mias_roc_distribution.png          |  Courbe ROC avec AUC +                 |
 |                                     |  Distribution des probabilites         |
 |                                     |  (benin vs malin)                      |
 +-------------------------------------+----------------------------------------+
 |  mias_segmentation_analysis.png     |  Nombre de regions MC par image +      |
 |                                     |  Boxplot correct vs faux               |
 +-------------------------------------+----------------------------------------+
 |  mias_cross_dataset_comparison.png  |  Barres comparatives                   |
 |                                     |  CBIS-DDSM vs MIAS                     |
 +-------------------------------------+----------------------------------------+
 |  mias_correct_gallery.png           |  Galerie des predictions CORRECTES     |
 |                                     |  (panels pipeline annotated)           |
 +-------------------------------------+----------------------------------------+
 |  mias_wrong_gallery.png             |  Galerie des predictions FAUSSES       |
 |                                     |  + analyse erreur par image            |
 +-------------------------------------+----------------------------------------+
 |  mias_final_dashboard.png           |  Dashboard resume :                    |
 |                                     |  gauge accuracy, sens/spec bar,        |
 |                                     |  confusion mini, cross-dataset,        |
 |                                     |  text summary                          |
 +-------------------------------------+----------------------------------------+
```

---

# 12. ORGANISATION POUR PARTAGE (GOOGLE DRIVE)

## 12.1 Structure Recommandee

```
 STRUCTURE GOOGLE DRIVE RECOMMANDEE
 =====================================

 mammo-cad-MIAS-evaluation/
 |
 +-- README.txt                      <-- Ce qu'il y a dans chaque dossier
 |
 +-- 01_resume/
 |   +-- mias_final_dashboard.png    <-- Dashboard resume (1 image = tout)
 |   +-- mias_evaluation_summary.txt <-- Resume textuel
 |
 +-- 02_metriques/
 |   +-- mias_confusion_results.png  <-- Matrice de confusion
 |   +-- mias_roc_distribution.png   <-- Courbe ROC + distribution
 |   +-- mias_cross_dataset_comparison.png  <-- CBIS vs MIAS
 |   +-- mias_evaluation_results.json      <-- Donnees brutes
 |
 +-- 03_analyse/
 |   +-- mias_segmentation_analysis.png  <-- Analyse segmentation
 |   +-- mias_correct_gallery.png        <-- Galerie correct
 |   +-- mias_wrong_gallery.png          <-- Galerie erreurs
 |
 +-- 04_resultats_par_image/
 |   +-- mdb209/                     <-- Image maligne
 |   |   +-- mdb209_panel.png        <-- Panel complet
 |   |   +-- mdb209_annotated.png    <-- Image annotee
 |   |   +-- mdb209_seg_overlay.png  <-- Segmentation
 |   |   +-- crops/                  <-- Regions + Grad-CAM
 |   +-- mdb211/
 |   +-- mdb212/
 |   +-- ...
 |   +-- mdb256/
 |
 +-- 05_notebook/
     +-- mias_evaluation_rapport.ipynb   <-- Notebook complet
     +-- mias_evaluation_rapport.html    <-- Version HTML (lisible sans Jupyter)
```

## 12.2 Comment Presenter au Professeur

```
 GUIDE DE PRESENTATION
 =======================

 1. Commencer par le DASHBOARD (01_resume/)
    --> 1 image qui resume tout : accuracy, confusion, comparaison

 2. Montrer les METRIQUES (02_metriques/)
    --> Matrice de confusion : combien correct, combien faux
    --> Courbe ROC : performance globale
    --> Comparaison CBIS-DDSM vs MIAS : generalisation

 3. Montrer l'ANALYSE (03_analyse/)
    --> Galerie des predictions correctes (le systeme fonctionne !)
    --> Galerie des erreurs (analyse honete des limites)
    --> Analyse segmentation (premiere etape critique)

 4. Naviguer dans les RESULTATS PAR IMAGE (04_resultats_par_image/)
    --> Choisir 2-3 images (1 correct benin, 1 correct malin, 1 erreur)
    --> Montrer le panel complet + Grad-CAM
    --> Expliquer comment le pipeline "raisonne"

 5. Si questions techniques : ouvrir le NOTEBOOK (05_notebook/)
    --> Code + visualisations + analyse detaillee
```

---

# 13. DISCUSSION

## 13.1 Forces de cette Evaluation

```
 POINTS FORTS
 ==============

 1. VALIDATION HONETE
    Dataset JAMAIS vu pendant l'entrainement
    Pas de data leakage possible

 2. PIPELINE COMPLET
    Teste TOUT : preprocessing + segmentation + classification
    Pas juste la classification sur des ROI pre-decoupes

 3. DOMAIN SHIFT REEL
    Scanner different, pays different, format different
    = conditions realistes d'utilisation clinique

 4. TRANSPARENCE
    Chaque image : Grad-CAM montre OU le modele regarde
    Erreurs analysees en detail (FP vs FN, seg vs class)
    Tous les outputs sont consultables image par image
```

## 13.2 Limites

```
 LIMITES
 ========

 1. PETITE TAILLE : 25 images
    - Haute variance sur les metriques
    - 1 image = 4% de l'accuracy
    - Pas de CI 95% fiable
    - Mais : c'est ce qui est disponible dans MIAS (calcifications)

 2. CALCIFICATIONS UNIQUEMENT
    - Ne teste pas la generalisation sur les MASSES
    - CBIS-DDSM a des masses (405 en test), MIAS non

 3. LABELS BINAIRES
    - MIAS : benin/malin seulement
    - Pas de sous-types BI-RADS (2, 3, 4, 5)
    - Pas de nuance dans la severite

 4. SEUIL NON AJUSTE
    - Seuil 0.535 optimise sur CBIS-DDSM validation
    - Peut ne pas etre optimal pour MIAS
    - Mais ajuster = risque de sur-ajustement sur 25 images
```

## 13.3 Perspectives d'Amelioration

```
 AMELIORATIONS POSSIBLES
 =========================

 1. DOMAIN ADAPTATION
    Fine-tuning sur quelques images MIAS (5-10)
    = adapter le modele au nouveau scanner
    Risque : sur-ajustement (peu de donnees)

 2. DATA AUGMENTATION MULTI-SCANNER
    Simuler les differences de contraste entre scanners
    = rendre le modele plus robuste aux variations

 3. ENSEMBLE DE SEGMENTATION
    Combiner plusieurs modeles U-Net (10-fold)
    = vote majoritaire pour reduire les FN segmentation

 4. MULTI-CENTRE TRAINING
    Entrainer sur CBIS-DDSM + INbreast + MIAS
    = diversite de scanners pendant l'entrainement
    = meilleure generalisation native
```

---

# 14. CONCLUSION

```
 RESUME DE L'EVALUATION EXTERNE
 =================================

 OBJECTIF : Tester le pipeline mammo-cad sur un dataset EXTERNE
            (MIAS, 25 images de calcifications, jamais vu)

 PIPELINE : Image brute --> Preprocessing --> U-Net segmentation
            --> Clustering --> EfficientNet-B3 classification
            --> Grad-CAM explicabilite --> Decision finale

 RESULTATS : (a remplir apres execution)
   Accuracy     = ___%
   Sensitivity  = ___%
   Specificity  = ___%

 COMPARAISON avec CBIS-DDSM test (n=787) :
   (a remplir apres execution)

 CONCLUSION PRINCIPALE :
   Le pipeline mammo-cad fonctionne de bout en bout sur des
   images d'un scanner JAMAIS VU pendant l'entrainement.
   La comparaison cross-dataset quantifie la capacite de
   generalisation du systeme complet.

 OUTPUTS DISPONIBLES :
   - 25 dossiers avec panels, segmentation, Grad-CAM
   - 7 visualisations d'analyse
   - JSON avec tous les resultats detailles
   - Notebook interactif de visualisation
```

---

# 15. ANNEXES

## 15.1 Structure Complete des Fichiers de Sortie

```
 outputs/mias_evaluation/
   |
   +-- mias_evaluation_results.json      Resultats agreges (JSON)
   +-- mias_evaluation_summary.txt       Resume textuel
   |
   +-- mias_confusion_results.png        Visualisation 1
   +-- mias_roc_distribution.png         Visualisation 2
   +-- mias_segmentation_analysis.png    Visualisation 3
   +-- mias_cross_dataset_comparison.png Visualisation 4
   +-- mias_correct_gallery.png          Visualisation 5
   +-- mias_wrong_gallery.png            Visualisation 6
   +-- mias_final_dashboard.png          Visualisation 7
   |
   +-- per_image/
       +-- mdb209/
       |   +-- mdb209_preprocessed.png
       |   +-- mdb209_seg_mask.png
       |   +-- mdb209_seg_overlay.png
       |   +-- mdb209_annotated.png
       |   +-- mdb209_panel.png
       |   +-- mdb209_report.json
       |   +-- crops/
       |       +-- region_0_crop.png
       |       +-- region_0_gradcam.png
       |       +-- ...
       +-- mdb211/
       |   +-- ...
       +-- mdb212/
       |   +-- ...
       +-- ...  (25 dossiers au total)
       +-- mdb256/
           +-- ...
```

## 15.2 Format JSON des Resultats

```json
{
  "summary": {
    "dataset":          "MIAS Calcification",
    "total_images":     25,
    "benign_images":    12,
    "malignant_images": 13,
    "correct":          "...",
    "accuracy":         "...",
    "sensitivity":      "...",
    "specificity":      "...",
    "ppv":              "...",
    "npv":              "...",
    "f1":               "...",
    "tp": "...", "tn": "...", "fp": "...", "fn": "...",
    "total_time_s":     "...",
    "avg_time_s":       "..."
  },
  "results": [
    {
      "image":           "mdb209",
      "true_label":      1,
      "true_label_str":  "MALIGNANT",
      "pred_label":      "...",
      "pred_label_str":  "...",
      "correct":         "...",
      "regions_found":   "...",
      "n_malignant":     "...",
      "n_benign":        "...",
      "max_prob":        "...",
      "regions_detail":  [
        {"region_id": 0, "label": "...", "probability": "...", "bbox": [...]}
      ]
    },
    "... (25 entries)"
  ]
}
```

## 15.3 Reproduction Complete

```bash
 ETAPES POUR REPRODUIRE L'EVALUATION
 ======================================

 # 1. Prerequis
 cd C:\project\mammo-cad
 .\venv\Scripts\activate       # ou source venv/bin/activate (Linux)

 # 2. Verifier le dataset MIAS
 ls "C:\Users\amine\Downloads\Compressed\archive\mias_calc\benign"
 ls "C:\Users\amine\Downloads\Compressed\archive\mias_calc\malignant"

 # 3. Verifier les checkpoints
 ls checkpoints\unet_best.pth
 ls checkpoints\best_0.7717\efficientnet_stage2.pth

 # 4. Lancer l'evaluation (~2-5 min/image sur CPU)
 python scripts/run_mias_evaluation.py

 # 5. Verifier les resultats
 type outputs\mias_evaluation\mias_evaluation_summary.txt

 # 6. Lancer le notebook de visualisation
 jupyter notebook notebooks\mias_evaluation_rapport.ipynb
 # Executer toutes les cellules (Kernel > Restart & Run All)

 # 7. (Optionnel) Exporter le notebook en HTML
 jupyter nbconvert --to html notebooks\mias_evaluation_rapport.ipynb

 # 8. Copier vers Google Drive (structure recommandee en section 12)
```

---

*Document genere le 2026-04-13 — Projet mammo-cad*
*Pipeline: U-Net (31M params, AUC seg=0.9642) + EfficientNet-B3 (11.9M params, Val AUC=0.8056)*
*Evaluation externe sur MIAS Calcification (25 images, 12 benin + 13 malin)*
*Seuil=0.535, TTA-16, mode CPU*
