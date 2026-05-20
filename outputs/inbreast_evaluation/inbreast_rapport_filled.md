# Evaluation Externe sur Dataset INbreast — RESULTATS

## Pipeline mammo-cad : U-Net + EfficientNet-B3
## Document genere automatiquement apres execution

---

# 1. RESUME EXECUTIF

```
 RESULTATS DU PIPELINE COMPLET SUR INbreast (sous-ensemble MC)
 ==============================================================

 +---------------------------------------------------------------+
 |  Accuracy       :  60.0%  (  30/ 50 correct )                |
 |  Sensitivity    :  50.0%  (  10/ 20 malins detectes )          |
 |  Specificity    :  66.7%  (  20/ 30 benins corrects )          |
 |  PPV            :  50.0%  (  10/ 20 pred malin vrais )          |
 |  NPV            :  66.7%  (  20/ 30 pred benin vrais )          |
 |  F1 Score       : 0.5000                                      |
 |  AUC            : 0.4117                                      |
 +---------------------------------------------------------------+

 Matrice de Confusion :
                          PREDICTION PIPELINE
                      +-----------+-----------+
                      |  BENIN    |  MALIN    |
 +----------+--------+-----------+-----------+
 |  Verite  | BENIN  |  TN =  20 |  FP =  10 |
 | INbreast | MALIN  |  FN =  10 |  TP =  10 |
 +----------+--------+-----------+-----------+

 Sous-ensemble : 50 images (30 benin + 20 malin, random_state=42)
 Temps total   : 35860.3s  (717.2s/image)
```

---

# 2. DATASET INbreast

```
 INbreast Microcalcification Dataset
 =====================================

 Source      : Centro Hospitalar de Sao Joao, Porto, Portugal
 Scanner     : Siemens Mammomat Inspiration
 Annotation  : BI-RADS scale (1–6)
 Subset      : MC cases only (313 images)
 Sous-ensemble utilise : 30 benin + 20 malin = 50 images
 Preprocessing : Deja effectue (AllPng/) — preprocess=False
 Parametres pipeline : seg_threshold=0.45, min_area=50, cluster_dist=60
```

---

# 3. RESULTATS PAR IMAGE

```
 +-------------+-----------+------------------------+--------+-----+-------+--------+
 |  Reference  | Verite    | Prediction Pipeline    | BIRADS | Reg | MaxP  | Status |
 +-------------+-----------+------------------------+--------+-----+-------+--------+
 |  51049628     | BENIGN    | BENIGN                 |   2    |   1 | 0.435 | [  OK] |
 |  50993616     | BENIGN    | BENIGN (minority mal)  |   2    |   3 | 0.525 | [  OK] |
 |  20588562     | MALIGNANT | BENIGN (no MC detected |   5    |   0 | 0.000 | [FAUX] |
 |  51049682     | BENIGN    | BENIGN                 |   2    |   3 | 0.448 | [  OK] |
 |  24058660     | BENIGN    | MALIGNANT              |   2    |   3 | 0.563 | [FAUX] |
 |  51049462     | BENIGN    | MALIGNANT              |   2    |   3 | 0.698 | [FAUX] |
 |  26933830     | BENIGN    | MALIGNANT              |   2    |   2 | 0.464 | [FAUX] |
 |  22579847     | BENIGN    | BENIGN                 |   2    |   1 | 0.433 | [  OK] |
 |  50999008     | MALIGNANT | MALIGNANT              |   5    |   1 | 0.595 | [  OK] |
 |  53580804     | MALIGNANT | BENIGN (minority mal)  |   5    |   4 | 0.468 | [FAUX] |
 |  30011824     | MALIGNANT | MALIGNANT              |   4b   |   6 | 0.635 | [  OK] |
 |  50997796     | BENIGN    | MALIGNANT              |   3    |   7 | 0.675 | [FAUX] |
 |  20587200     | BENIGN    | MALIGNANT              |   2    |   2 | 0.532 | [FAUX] |
 |  20587372     | BENIGN    | BENIGN                 |   2    |   3 | 0.359 | [  OK] |
 |  50997769     | BENIGN    | BENIGN (minority mal)  |   3    |   8 | 0.614 | [  OK] |
 |  20587784     | BENIGN    | BENIGN                 |   3    |   3 | 0.424 | [  OK] |
 |  22579730     | MALIGNANT | MALIGNANT              |   5    |   2 | 0.507 | [  OK] |
 |  50998467     | BENIGN    | BENIGN                 |   2    |   2 | 0.412 | [  OK] |
 |  20587492     | BENIGN    | BENIGN                 |   2    |   3 | 0.443 | [  OK] |
 |  22427840     | MALIGNANT | MALIGNANT              |   5    |   2 | 0.499 | [  OK] |
 |  51049134     | BENIGN    | BENIGN (minority mal)  |   2    |   5 | 0.605 | [  OK] |
 |  50999432     | MALIGNANT | BENIGN (no MC detected |   5    |   0 | 0.000 | [FAUX] |
 |  24055725     | BENIGN    | BENIGN (minority mal)  |   2    |   5 | 0.575 | [  OK] |
 |  20587544     | BENIGN    | MALIGNANT              |   2    |   2 | 0.575 | [FAUX] |
 |  50998981     | MALIGNANT | MALIGNANT              |   5    |   2 | 0.608 | [  OK] |
 |  53582791     | BENIGN    | BENIGN (minority mal)  |   2    |  20 | 0.693 | [  OK] |
 |  24058738     | MALIGNANT | BENIGN (minority mal)  |   4a   |   4 | 0.519 | [FAUX] |
 |  53582818     | BENIGN    | BENIGN (minority mal)  |   2    |  16 | 0.544 | [  OK] |
 |  24058686     | MALIGNANT | MALIGNANT              |   4a   |   3 | 0.641 | [  OK] |
 |  22614266     | MALIGNANT | BENIGN                 |   5    |   1 | 0.399 | [FAUX] |
 |  53580885     | BENIGN    | MALIGNANT              |   2    |   5 | 0.641 | [FAUX] |
 |  22670855     | MALIGNANT | BENIGN                 |   5    |   3 | 0.324 | [FAUX] |
 |  24065761     | MALIGNANT | MALIGNANT              |   5    |   1 | 0.534 | [  OK] |
 |  51049053     | BENIGN    | BENIGN                 |   2    |   1 | 0.287 | [  OK] |
 |  26933801     | MALIGNANT | MALIGNANT              |   4b   |   6 | 0.585 | [  OK] |
 |  22614353     | BENIGN    | BENIGN (minority mal)  |   2    |   3 | 0.461 | [  OK] |
 |  50999246     | BENIGN    | BENIGN                 |   2    |   1 | 0.353 | [  OK] |
 |  22670465     | MALIGNANT | MALIGNANT              |   5    |   7 | 0.601 | [  OK] |
 |  24055752     | BENIGN    | MALIGNANT              |   2    |  11 | 0.726 | [FAUX] |
 |  22580706     | MALIGNANT | MALIGNANT              |   5    |   2 | 0.559 | [  OK] |
 |  27829188     | MALIGNANT | BENIGN (minority mal)  |   5    |   5 | 0.456 | [FAUX] |
 |  51049489     | BENIGN    | MALIGNANT              |   2    |   2 | 0.502 | [FAUX] |
 |  22614074     | MALIGNANT | BENIGN                 |   5    |   1 | 0.255 | [FAUX] |
 |  50993787     | BENIGN    | BENIGN (minority mal)  |   2    |  48 | 0.767 | [  OK] |
 |  50994589     | BENIGN    | BENIGN (minority mal)  |   2    |   6 | 0.568 | [  OK] |
 |  53586987     | BENIGN    | BENIGN                 |   2    |   2 | 0.420 | [  OK] |
 |  22427705     | MALIGNANT | BENIGN (minority mal)  |   5    |   3 | 0.607 | [FAUX] |
 |  50998580     | BENIGN    | MALIGNANT              |   2    |   1 | 0.623 | [FAUX] |
 |  22670809     | MALIGNANT | BENIGN                 |   4a   |   2 | 0.373 | [FAUX] |
 |  50997742     | BENIGN    | BENIGN (minority mal)  |   3    |   6 | 0.814 | [  OK] |
 +-------------+-----------+------------------------+--------+-----+-------+--------+

 TOTAL : 30/50 correct = 60.0%
```

---

# 4. RESULTATS PAR BI-RADS

```
 CLASSIFICATION PAR CATEGORIE BI-RADS
 ======================================

 +----------+----------+-----+-----------+-------------+-------------+
 |  BI-RADS |  Type    |  N  | Accuracy  | Sensitivity | Specificity |
 +----------+----------+-----+-----------+-------------+-------------+
 |    2     |  benin   |  26 | 0.654    | 0.000       | 0.654       |
 |    3     |  benin   |   4 | 0.750    | 0.000       | 0.750       |
 |    4a    |  malin   |   3 | 0.333    | 0.333       | 0.000       |
 |    4b    |  malin   |   2 | 1.000    | 1.000       | 0.000       |
 |    5     |  malin   |  15 | 0.467    | 0.467       | 0.000       |
 +----------+----------+-----+-----------+-------------+-------------+

 Notes :
   BI-RADS 1/2/3 → BENIN   (label = 0)
   BI-RADS 4a/4b/4c/5/6 → MALIN  (label = 1)
```

---

# 5. MATRICE DE CONFUSION DETAILLEE

```
                          PREDICTION PIPELINE
                      +-----------+-----------+
                      |  BENIN    |  MALIN    |  Total
 +----------+--------+-----------+-----------+--------+
 |  Verite  | BENIN  |  TN =  20 |  FP =  10 |    30  |
 | INbreast |        | (67%)      | (33%)      |        |
 |          +--------+-----------+-----------+--------+
 |          | MALIN  |  FN =  10 |  TP =  10 |    20  |
 |          |        | (50%)      | (50%)      |        |
 +----------+--------+-----------+-----------+--------+
   Total              |      30 |      20 |    50  |
                      +-----------+-----------+--------+
```

---

# 6. ANALYSE DE LA SEGMENTATION

```
 DETECTION DES MICROCALCIFICATIONS PAR U-Net
 =============================================

 Images sans MC detectee      :   2/50  (4%)
 Images avec MC detectees     :  48/50  (96%)
 Regions moyennes par image   : 4.7
 Regions max sur une image    : 48

 IMPACT SUR LES FAUX NEGATIFS :
   FN total                   : 10
   FN par segmentation        : 2  (0 regions, MC manquees par U-Net)
   FN par classification      : 8  (MC trouvees mais mal classees)
```

---

# 7. ANALYSE DES ERREURS

```
 FAUX POSITIFS (Benin predit Malin) : 10
 ============================================
 +-------------+------+-----+-------+----------------------------------------+
 |  Image      | BIRADS| Reg | MaxP  | Analyse                                |
 +-------------+------+-----+-------+----------------------------------------+
 |  24058660     |    2 |   3 | 0.563 | Classification trop agressive       |
 |  51049462     |    2 |   3 | 0.698 | Classification trop agressive       |
 |  26933830     |    2 |   2 | 0.464 | Classification trop agressive       |
 |  50997796     |    3 |   7 | 0.675 | Classification trop agressive       |
 |  20587200     |    2 |   2 | 0.532 | Classification trop agressive       |
 |  20587544     |    2 |   2 | 0.575 | Classification trop agressive       |
 |  53580885     |    2 |   5 | 0.641 | Classification trop agressive       |
 |  24055752     |    2 |  11 | 0.726 | Classification trop agressive       |
 |  51049489     |    2 |   2 | 0.502 | Classification trop agressive       |
 |  50998580     |    2 |   1 | 0.623 | Classification trop agressive       |
 +-------------+------+-----+-------+----------------------------------------+

 FAUX NEGATIFS (Malin predit Benin) : 10
 ============================================
 +-------------+------+-----+-------+----------------------------------------+
 |  Image      | BIRADS| Reg | MaxP  | Analyse                                |
 +-------------+------+-----+-------+----------------------------------------+
 |  20588562     |    5 |   0 | 0.000 | Seg: 0 regions MC detectees           |
 |  53580804     |    5 |   4 | 0.468 | Cls: regions classees BENIN            |
 |  50999432     |    5 |   0 | 0.000 | Seg: 0 regions MC detectees           |
 |  24058738     |   4a |   4 | 0.519 | Cls: regions classees BENIN            |
 |  22614266     |    5 |   1 | 0.399 | Cls: regions classees BENIN            |
 |  22670855     |    5 |   3 | 0.324 | Cls: regions classees BENIN            |
 |  27829188     |    5 |   5 | 0.456 | Cls: regions classees BENIN            |
 |  22614074     |    5 |   1 | 0.255 | Cls: regions classees BENIN            |
 |  22427705     |    5 |   3 | 0.607 | Cls: regions classees BENIN            |
 |  22670809     |   4a |   2 | 0.373 | Cls: regions classees BENIN            |
 +-------------+------+-----+-------+----------------------------------------+
```

---

# 8. COMPARAISON CROSS-DATASET

```
 CBIS-DDSM (Train) vs DMID (Externe) vs INbreast (Externe)
 ===========================================================

 +------------------+--------------------+--------------------+--------------------+
 |  Metrique        |  CBIS-DDSM (Test)  |  DMID (Externe)    |  INbreast (Ext.)   |
 |                  |  n = 787           |  n = varies        |  n = 50             |
 +------------------+--------------------+--------------------+--------------------+
 |  AUC             |  0.7812            |  voir evaluation   |  0.4117            |
 |  Accuracy        |  71.0%             |  voir evaluation   |  60.0%            |
 |  Sensitivity     |  63.1%             |  voir evaluation   |  50.0%            |
 |  Specificity     |  76.2%             |  voir evaluation   |  66.7%            |
 |  F1 Score        |  0.6334            |  voir evaluation   |  0.5000            |
 |  PPV             |  63.5%             |  voir evaluation   |  50.0%            |
 |  NPV             |  75.9%             |  voir evaluation   |  66.7%            |
 +------------------+--------------------+--------------------+--------------------+

 INbreast : Portugal, Siemens Mammomat — dataset EXTERNE (jamais vu en training).
```

---

# 9. CONCLUSION

```
 Le pipeline mammo-cad a ete teste sur 50 images INbreast
 (dataset EXTERNE, jamais vu pendant l'entrainement, Portugal).

 Evaluation binaire : 30/50 images correctes (60.0%)
 Sensitivity : 50.0%  (detection des malins)
 Specificity : 66.7%  (identification des benins)
 AUC         : 0.4117

 Parametres : preprocess=False (images deja traitees), seg_threshold=0.45,
              min_area=50, cluster_dist=60, TTA=16
 Decision   : vote majoritaire (>= 50% regions malignes → MALIN)
```

---

*Document genere automatiquement le 2026-04-18 par run_inbreast_evaluation.py*
*Pipeline: U-Net + EfficientNet-B3 | seg_threshold=0.45 | TTA-16 | CPU*
