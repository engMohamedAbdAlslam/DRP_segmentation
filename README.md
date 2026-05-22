# Diabetic Retinopathy (DRP) Segmentation Prototype

This repository contains a modular retinal image analysis prototype for diabetic retinopathy screening. The codebase separates preprocessing, model training, and reporting to support reproducibility and clinical review.

## Preprocessing Module

Location: `src/engine/image_preprocessing.py`

Core contract:
- **Input**: fundus image path or `numpy.ndarray` (H, W, 3), optional mask path or array.
- **Output**: normalized image tensor, optional resized mask, and metadata describing each preprocessing step.

Implemented steps:
1. Foreground crop to remove dark borders.
2. Illumination correction using CLAHE.
3. Resize to a fixed target size.
4. Normalize using ImageNet statistics or zero-one scaling.

### Example (Python)

```python
from engine import PreprocessConfig, preprocess_fundus_image, save_preprocessed

config = PreprocessConfig(target_size=(512, 512), normalization="imagenet")
result = preprocess_fundus_image("data/raw/images/sample.png", config=config)
save_preprocessed(result, "data/processed/sample.npz")
```

### CLI

```bash
python src/engine/image_preprocessing.py --image data/raw/images/sample.png --output data/processed/sample.npz
```

## Minimal Dataset Layout

```
data/
  raw/
    images/        # Original fundus images
    masks/         # Optional lesion or vessel masks
  processed/
    images/        # Preprocessed outputs (.npz)
```

## Next Integration Targets

- Segmentation training pipelines can consume the `.npz` outputs and metadata.
- Classification and explainability modules should reuse the preprocessing API for consistent inputs.
