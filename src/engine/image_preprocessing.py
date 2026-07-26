from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

MIN_CROP_SIZE = 10

ImageInput = Union[str, Path, np.ndarray]
MaskInput = Optional[Union[str, Path, np.ndarray]]


@dataclass(frozen=True)
class PreprocessConfig:
    target_size: Tuple[int, int] = (512, 512)
    crop_border: bool = True
    illumination_correction: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    normalization: str = "imagenet"
    channel_first: bool = False
    foreground_threshold: int = 10


@dataclass(frozen=True)
class PreprocessMetadata:
    original_shape: Tuple[int, int, int]
    cropped_shape: Tuple[int, int, int]
    crop_bbox: Optional[Tuple[int, int, int, int]]
    resized_shape: Tuple[int, int, int]
    normalization: str
    config: Dict[str, Any]


@dataclass(frozen=True)
class PreprocessResult:
    image: np.ndarray
    mask: Optional[np.ndarray]
    metadata: PreprocessMetadata


def preprocess_fundus_image(
    image: ImageInput,
    mask: MaskInput = None,
    config: Optional[PreprocessConfig] = None,
) -> PreprocessResult:
    """Preprocess a fundus image into a standardized tensor and metadata.

    Args:
        image: File path or RGB array with shape (H, W, 3).
        mask: Optional file path or mask array aligned to the image.
        config: Preprocess configuration; defaults are used when None.

    Returns:
        PreprocessResult containing the normalized image, optional mask, and metadata.
    """
    if config is None:
        config = PreprocessConfig()

    image_bgr = _load_image(image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_shape = image_rgb.shape

    crop_bbox = None
    if config.crop_border:
        crop_bbox = _compute_crop_bbox(image_rgb, config.foreground_threshold)
        if crop_bbox is not None:
            image_rgb = _apply_crop(image_rgb, crop_bbox)

    if config.illumination_correction:
        image_rgb = _apply_clahe(image_rgb, config)

    mask_array = None
    if mask is not None:
        mask_array = _load_mask(mask)
        if crop_bbox is not None:
            mask_array = _apply_crop(mask_array, crop_bbox)

    cropped_shape = image_rgb.shape
    image_rgb = _resize_image(image_rgb, config.target_size, cv2.INTER_AREA)
    if mask_array is not None:
        mask_array = _resize_image(mask_array, config.target_size, cv2.INTER_NEAREST)

    image_rgb = _normalize_image(image_rgb, config.normalization)

    if config.channel_first:
        image_rgb = np.transpose(image_rgb, (2, 0, 1))
        if mask_array is not None and mask_array.ndim == 3:
            mask_array = np.transpose(mask_array, (2, 0, 1))

    metadata = PreprocessMetadata(
        original_shape=original_shape,
        cropped_shape=cropped_shape,
        crop_bbox=crop_bbox,
        resized_shape=image_rgb.shape,
        normalization=config.normalization,
        config=asdict(config),
    )
    return PreprocessResult(image=image_rgb, mask=mask_array, metadata=metadata)


def save_preprocessed(result: PreprocessResult, output_path: Union[str, Path]) -> None:
    """Save a preprocessing result to a compressed .npz file.

    The output includes the image array, optional mask, and JSON metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "image": result.image,
        "metadata": json.dumps(asdict(result.metadata)),
    }
    if result.mask is not None:
        payload["mask"] = result.mask
    np.savez_compressed(output_path, **payload)


def _load_image(image: ImageInput) -> np.ndarray:
    """Load an image from disk or validate an RGB numpy array input.

    For array inputs, the expected format is RGB. The returned array is BGR uint8 for
    OpenCV compatibility.
    """
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image path not found: {image_path}")
        image_array = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_array is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image_array
    if not isinstance(image, np.ndarray):
        raise TypeError("Image input must be a path or numpy array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image array must have shape (H, W, 3).")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _load_mask(mask: ImageInput) -> np.ndarray:
    """Load a mask from disk or normalize a numpy array mask input."""
    if isinstance(mask, (str, Path)):
        mask_path = Path(mask)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask path not found: {mask_path}")
        mask_array = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_array is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        return mask_array
    if not isinstance(mask, np.ndarray):
        raise TypeError("Mask input must be a path or numpy array.")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask.astype(np.uint8)


def _compute_crop_bbox(image: np.ndarray, threshold: int) -> Optional[Tuple[int, int, int, int]]:
    """Compute the foreground bounding box using a grayscale threshold.

    Args:
        image: RGB image array.
        threshold: Grayscale threshold to separate foreground from background.

    Returns:
        Bounding box as (x_min, y_min, x_max, y_max) or None if no foreground.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    foreground = gray > threshold
    if not np.any(foreground):
        return None
    coords = np.argwhere(foreground)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    if (y_max - y_min) < MIN_CROP_SIZE or (x_max - x_min) < MIN_CROP_SIZE:
        return None
    return int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1


def _apply_crop(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop an image or mask using a bounding box (x_min, y_min, x_max, y_max)."""
    x_min, y_min, x_max, y_max = bbox
    return image[y_min:y_max, x_min:x_max]


def _apply_clahe(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Apply CLAHE to the L channel in LAB space for illumination correction."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _resize_image(image: np.ndarray, target_size: Tuple[int, int], interpolation: int) -> np.ndarray:
    """Resize an image to (width, height) using the specified interpolation."""
    target_width, target_height = target_size
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def _normalize_image(image: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize an RGB image using zero-one or ImageNet statistics."""
    image = image.astype(np.float32) / 255.0
    if normalization == "zero_one":
        return image
    if normalization == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (image - mean) / std
    raise ValueError(f"Unsupported normalization mode: {normalization}")


def _build_cli_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for preprocessing a single image."""
    parser = argparse.ArgumentParser(description="Preprocess a fundus image for diabetic retinopathy analysis.")
    parser.add_argument("--image", required=True, help="Path to the input fundus image.")
    parser.add_argument("--mask", default=None, help="Optional path to an input mask.")
    parser.add_argument("--output", required=True, help="Output .npz file to store preprocessed arrays.")
    parser.add_argument("--size", default="512,512", help="Target size as width,height.")
    parser.add_argument(
        "--normalization",
        default="imagenet",
        choices=["imagenet", "zero_one"],
        help="Normalization strategy.",
    )
    return parser


def _parse_size(value: str) -> Tuple[int, int]:
    """Parse a comma-separated width,height string into an integer tuple."""
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("Size must be provided as width,height.")
    width, height = (int(part.strip()) for part in parts)
    return width, height


def _run_cli() -> None:
    """Execute the preprocessing command-line interface."""
    parser = _build_cli_parser()
    args = parser.parse_args()
    config = PreprocessConfig(
        target_size=_parse_size(args.size),
        normalization=args.normalization,
    )
    result = preprocess_fundus_image(image=args.image, mask=args.mask, config=config)
    save_preprocessed(result, args.output)


if __name__ == "__main__":
    _run_cli()
