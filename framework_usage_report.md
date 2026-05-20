# medical-image-std Integration Report
**Version used:** 0.6.2  
**Project:** Mammo-CAD — Microcalcification Detection Pipeline

---

## What We Used

### 1. `BreastMaskAlgorithm`
- **File:** `src/inference/pipeline.py` — `preprocess_mammogram()`
- **Purpose:** Breast region extraction (Otsu threshold + largest connected component) before U-Net segmentation.
- **Replaces:** Custom OpenCV implementation with the standardized framework version.

### 2. `FebdsAlgorithm` (method: `"dog"`)
- **File:** `src/inference/pipeline.py` — fallback branch when U-Net detects 0 regions
- **Purpose:** When the U-Net finds no microcalcification clusters, FEBDS enhancement is applied to extract candidate regions, which are then classified by EfficientNet-B3.
- **Impact:** Recovered 2 missed malignant cases (BIRADS-5) that had 0 detections → AUC improved from 0.41 to 0.62 on INbreast cross-domain evaluation.

---

## Issues / Notes

- `FebdsAlgorithm(method="fft")` produces a very large number of candidate regions on full-resolution mammograms (140+ regions on some images). `"dog"` was used instead as it gives more controlled output.
- `BreastMaskAlgorithm` requires the input tensor to be `float32` — passing `uint8` raises no error but produces an empty mask. We convert explicitly before calling.

---

## Integration Code (minimal)

```python
from medical_image.algorithms.breast_mask import BreastMaskAlgorithm
from medical_image.algorithms.FEBDS import FebdsAlgorithm
from medical_image.data.in_memory_image import InMemoryImage
import torch

# Breast masking
img_t = InMemoryImage(array=torch.from_numpy(img.astype("float32")))
out_t = img_t.clone()
BreastMaskAlgorithm(mask_only=True, device="cpu")(img_t, out_t)

# FEBDS fallback
img_f = InMemoryImage(array=torch.from_numpy(img.astype("float32") / 255.0))
out_f = img_f.clone()
FebdsAlgorithm(method="dog", device="cpu")(img_f, out_f)
febds_mask = (out_f.pixel_data.numpy() * 255).astype("uint8")
```
