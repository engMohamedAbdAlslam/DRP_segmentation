# DRP Segmentation — Diabetic Retinopathy Screening Prototype

[![Open NB01 In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engMohamedAbdAlslam/DRP_segmentation/blob/copilot%2Fdevelop-preprocessing-pipeline/notebooks/01_disease_segmentation.ipynb)

An AI-assisted diabetic retinopathy (DR) screening prototype that detects and grades DR from fundus images, highlights suspicious retinal regions, and provides interpretable outputs for clinical decision support.

---

## Project Structure

```
DRP_segmentation/
├── notebooks/
│   ├── 01_disease_segmentation.ipynb       # DR grading preprocessing — APTOS 2019
│   ├── 02_vessel_segmentation.ipynb        # Vessel segmentation preprocessing — DRIVE
│   └── 03_binary_classification_filter.ipynb  # Binary DR filter — DR Resized
├── src/
│   └── engine/
│       └── image_preprocessing.py          # Core preprocessing pipeline
├── data/
│   ├── raw/                                # Downloaded datasets (gitignored)
│   └── processed/                          # Preprocessed .npz files (gitignored)
└── README.md
```

---

## Notebooks Overview

| # | Notebook | Dataset | Task | Images |
|---|----------|---------|------|--------|
| 01 | `01_disease_segmentation.ipynb` | **APTOS 2019** | DR Grading (0–4) + Preprocessing | 3,662 |
| 02 | `02_vessel_segmentation.ipynb` | **DRIVE** | Vessel Segmentation + Masks | 40 |
| 03 | `03_binary_classification_filter.ipynb` | **DR Resized** | Binary DR vs No-DR Filter | ~35k |

---

## Preprocessing Pipeline

All notebooks share the same core pipeline (`src/engine/image_preprocessing.py`):

1. **Load** — Read fundus image (PNG/JPEG/TIFF)
2. **Resize** — Standardize to 512×512
3. **CLAHE** — Contrast Limited Adaptive Histogram Equalization for vessel/lesion enhancement
4. **Normalize** — Zero-one scaling [0, 1]
5. **Save** — Export as compressed `.npz` with image array + metadata

---

## Quick Start — Google Colab

### Requirements
Before running any notebook, add your Kaggle credentials to **Colab Secrets** (🔑 icon):

| Secret Name | Value |
|-------------|-------|
| `KAGGLE_TOKEN` | Full content of your `kaggle.json` |

Get your token from: [kaggle.com/settings → API → Create New Token](https://www.kaggle.com/settings)

### Run Order

```
1. Open notebook in Colab (use badge above)
2. Runtime → Run all  (Ctrl+F9)
3. Run Cell 11 separately to save .npz files to Google Drive
```

---

## Datasets

| Dataset | Source | DR Grading | Lesion Masks | License |
|---------|--------|------------|--------------|---------|
| [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection) | Kaggle | ✅ 0–4 | ❌ | CC BY 4.0 |
| [DRIVE](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction) | Kaggle | ❌ | ✅ Vessel masks | Research |
| [DR Resized](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) | Kaggle | ✅ 0–4 | ❌ | CC BY 4.0 |

---

## Evaluation Metrics

- **DR Grading:** Quadratic Weighted Kappa (QWK), AUC-ROC, Accuracy per grade
- **Segmentation:** Dice coefficient, IoU, Sensitivity, Specificity
- **Classification:** AUC-ROC, F1-score, Sensitivity, Specificity
- **Explainability:** Grad-CAM heatmaps for lesion localization

---

## Next Steps

- [ ] Notebook 04 — DR Grading Model (EfficientNet-B4 on APTOS)
- [ ] Notebook 05 — Vessel Segmentation Model (U-Net on DRIVE)
- [ ] Notebook 06 — Grad-CAM Explainability
- [ ] Notebook 07 — Clinical Interface (Gradio/Streamlit)

---

## Branch

Active development branch: `copilot/develop-preprocessing-pipeline`
