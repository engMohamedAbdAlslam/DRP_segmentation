"""Compatibility shim for legacy preprocessing imports.

Deprecated: import from engine.image_preprocessing instead.
"""

from .image_preprocessing import PreprocessConfig, PreprocessMetadata, PreprocessResult, preprocess_fundus_image, save_preprocessed

__all__ = [
    "PreprocessConfig",
    "PreprocessMetadata",
    "PreprocessResult",
    "preprocess_fundus_image",
    "save_preprocessed",
]
