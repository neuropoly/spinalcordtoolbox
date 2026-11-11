"""
sc-crop — Spinal cord detection and cropping.

Public API:
    detect(img_path, ...)              → bbox           — detect SC bbox
    crop(img, bbox)                    → cropped NIfTI  — crop any volume (image or label)
    uncrop(img, bbox)                  → full-space NIfTI
    detect_and_crop(img_path, ...)     → (crop_nii, bbox)
    download()                         → pre-download models (auto on first use)
    ensure_model()                     → path to the detector ONNX file
    ensure_cls_model()                 → path to cls_model.onnx
    load_config()                      → loaded config dict

Quality control:
    check_label_crop(label, bbox)      → dict
    save_bbox_nifti(bbox, ref_nii, path) → None
    check_seg_truncation(seg_nii, bbox)  → list[str]
    CropReport                           — accumulate and save per-volume QC

Release metadata:
    write_crop_metadata(path, **pad_kwargs) → None — write crop_metadata.json for a model
        release (e.g. for SpinalCordToolbox's `sct_deepseg`); also available as the
        `sc_crop write-metadata` CLI subcommand.

Version constants:
    __version__       → sc-crop package version
    __model_version__ → detection model version
    MODEL_VERSION     → alias for __model_version__
"""

from .crop import crop, detect, detect_and_crop, load_config, uncrop
from .qc import check_label_crop, CropReport, save_bbox_nifti, check_seg_truncation
from .download import download, ensure_model, ensure_cls_model, _MODEL_TAG
from .metadata import write_crop_metadata

__version__ = "0.12.1"
__model_version__ = _MODEL_TAG
MODEL_VERSION = _MODEL_TAG

__all__ = [
    "crop",
    "detect",
    "detect_and_crop",
    "load_config",
    "uncrop",
    "check_label_crop",
    "CropReport",
    "save_bbox_nifti",
    "check_seg_truncation",
    "download",
    "ensure_model",
    "ensure_cls_model",
    "write_crop_metadata",
]
