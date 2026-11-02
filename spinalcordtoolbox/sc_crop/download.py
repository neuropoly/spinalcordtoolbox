"""
Model download for sc_crop.

Models are cached in ~/.cache/sc_crop/<model_version>/. SHA256 is verified on every load.
The model version (_MODEL_TAG) is decoupled from the package version.

To reclaim disk space from old versions:
    rm -rf ~/.cache/sc_crop/v0.0.9
"""

import hashlib
import urllib.request
from pathlib import Path

_ASSETS = {
    "det_model.onnx": {
        "url": "https://github.com/ivadomed/sc-crop/releases/download/v0.0.11/det_model.onnx",
        "sha256": "052abe794f3878422e2acc2df4e4942f995aab3788ad513bbcc92b66b16ee9c0",
    },
    "cls_model.onnx": {
        "url": "https://github.com/ivadomed/sc-crop/releases/download/v0.0.11/cls_model.onnx",
        "sha256": "25fc911e501ae944dbfb66546e36d32f26021e35506c2495b0f71d4ff6bf3d6b",
    },
    "det_model.pt": {
        "url": "https://github.com/ivadomed/sc-crop/releases/download/v0.0.11/det_model.pt",
        "sha256": "ec667d683bf7766305c32346eac14c09079d522bb43d1dd925d7c613a8461417",
    },
    "cls_model.pt": {
        "url": "https://github.com/ivadomed/sc-crop/releases/download/v0.0.11/cls_model.pt",
        "sha256": "6f2c66153904e644835fec5e71342904f11cd527ad9310a0dde54c122b1af482",
    },
}

_MODEL_TAG = "v0.0.11"
_CACHE_DIR = Path.home() / ".cache" / "sc_crop" / _MODEL_TAG


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_file(name: str) -> Path:
    info = _ASSETS[name]
    dest = _CACHE_DIR / name
    if dest.exists() and _sha256(dest) == info["sha256"]:
        return dest
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sc_crop {name} …", flush=True)
    urllib.request.urlretrieve(info["url"], dest)
    actual = _sha256(dest)
    if actual != info["sha256"]:
        dest.unlink()
        raise RuntimeError(
            f"SHA256 mismatch for {name}.\n"
            f"  expected : {info['sha256']}\n"
            f"  got      : {actual}\n"
            "The download may be corrupted — please retry."
        )
    return dest


def ensure_model() -> Path:
    """Return path to the detector ONNX file, downloading and verifying if needed."""
    return _ensure_file("det_model.onnx")


def ensure_cls_model() -> Path:
    """Return path to cls_model.onnx, downloading and verifying if needed."""
    return _ensure_file("cls_model.onnx")


def download() -> None:
    """Pre-download model files. Optional — auto on first use."""
    for name in ("det_model.onnx", "cls_model.onnx"):
        _ensure_file(name)
    print(f"sc_crop models ready in {_CACHE_DIR}")
