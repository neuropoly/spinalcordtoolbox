"""
Core logic for spinal cord detection and bounding box computation.

Public API (pure — no file I/O):
    detect(img_path, ...)              → bbox dict     — detect SC bbox
    crop(img, bbox)                     → NIfTI        — crop any volume with the same bbox
    detect_and_crop(img_path, ...)     → (NIfTI, bbox) — convenience: detect + crop in one call
    uncrop(img, bbox)     → NIfTI        — restore any cropped volume to original space

The context dict returned by detect() contains:
    xmin, xmax, ymin, ymax, zmin, zmax  — inclusive bbox in native voxel space
    original_axcodes                    — e.g. "RAS", "LPI"
  Private keys (for CLI / advanced use):
    _original_img   — loaded NIfTI in native orientation (for uncrop)
    _img_las        — LAS-reoriented NIfTI (for --las crop)
    _bbox_pad_las   — BBox3D in LAS space (for --las crop)
    _original_ornt  — nibabel orientation array of the original image

Regularization (--regularization):
  cls       (default): runs a classifier on detection components from most superior
                       to most inferior; stops at first component with ≥1 positive slice;
                       keeps all detections from that component's min_z downward.
  graphtrim:           removes superior outlier detections by checking the 2 topmost
                       SI edges; discards slices above the first broken edge.
  none:                no regularization, raw detection output.

Normalisation (--norm-scope) — no default is hardcoded here; the value always comes
from config.yaml unless explicitly overridden, so it always matches training:
  volume:     percentile 0.5/99.5 over every voxel of the whole resampled volume,
              computed once, applied to every slice.
  slice_all:  percentile 0.5/99.5 per slice on ALL voxels (background included).
  slice:      percentile 0.5/99.5 per slice on foreground voxels only.
  Each must match preprocess.py's norm_scope of the same name exactly (training repo).

Padding API (9 parameters, priority: individual > symmetric > default):
  Individual (per face):  pad_superior, pad_inferior, pad_left, pad_right, pad_anterior, pad_posterior
  Symmetric: pad_si  (= superior + inferior),
                          pad_rl  (= left + right),
                          pad_ap  (= anterior + posterior)
  Defaults: superior=40mm, inferior=100mm, left=right=15mm, anterior=15mm, posterior=22mm.

Usage:
    from sc_crop import detect, crop
    import nibabel as nib

    bbox        = detect("t2.nii.gz")
    crop_img   = crop(nib.load("t2.nii.gz"),       bbox)
    crop_label = crop(nib.load("t2_label.nii.gz"), bbox)

    bbox = detect("t2.nii.gz", pad_superior=50, pad_inferior=80)
    bbox = detect("t2.nii.gz", pad_si=30)                    # symmetric SI
    bbox = detect("t2.nii.gz", pad_si=30, pad_inferior=60)   # symmetric + override
    bbox = detect("t2.nii.gz", device="cuda")  # GPU inference
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.resources
from pathlib import Path
import yaml

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, ornt_transform
from nibabel.processing import resample_to_output

import spinalcordtoolbox.sc_crop


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load config.yaml bundled with the package."""
    config_path = importlib.resources.files(spinalcordtoolbox.sc_crop) / "config.yaml"
    return yaml.safe_load(config_path.read_text())


# ─── BBox3D: single source of truth for voxel bboxes in LAS index space ──────

_DEFAULT_PAD_SUPERIOR = 40.0
_DEFAULT_PAD_INFERIOR = 100.0   # cords extend far inferiorly (lumbar); covers anisotropic sagittal cases
_DEFAULT_PAD_RL = 15.0    # +5mm over the detected bbox to cover lateral GT overshoot (≤3.2mm observed)
_DEFAULT_PAD_ANTERIOR = 15.0
_DEFAULT_PAD_POSTERIOR = 22.0    # cords (esp. PSIR) overshoot posteriorly; covers the ≤6.2mm observed


def _resolve_padding(
    pad_si=None, pad_superior=None, pad_inferior=None,
    pad_rl=None, pad_left=None,     pad_right=None,
    pad_ap=None, pad_anterior=None, pad_posterior=None,
) -> tuple:
    """Resolve 9 padding inputs to 6 face scalars (mm).

    Priority per face: individual > symmetric > default.
    Returns (left, right, anterior, posterior, superior, inferior).
    """
    sup = pad_superior if pad_superior is not None else (pad_si if pad_si is not None else _DEFAULT_PAD_SUPERIOR)
    inf = pad_inferior if pad_inferior is not None else (pad_si if pad_si is not None else _DEFAULT_PAD_INFERIOR)
    left = pad_left if pad_left is not None else (pad_rl if pad_rl is not None else _DEFAULT_PAD_RL)
    right = pad_right if pad_right is not None else (pad_rl if pad_rl is not None else _DEFAULT_PAD_RL)
    ant = pad_anterior if pad_anterior is not None else (pad_ap if pad_ap is not None else _DEFAULT_PAD_ANTERIOR)
    post = pad_posterior if pad_posterior is not None else (pad_ap if pad_ap is not None else _DEFAULT_PAD_POSTERIOR)
    return left, right, ant, post, sup, inf


@dataclass(frozen=True)
class BBox3D:
    """Voxel bbox in LAS index space: (rl1, rl2, ap1, ap2, z1, z2)."""
    rl1: int
    rl2: int
    ap1: int
    ap2: int
    z1: int
    z2: int

    def pad(self,
            left: float, right: float,
            anterior: float, posterior: float,
            superior: float, inferior: float,
            zooms: tuple[float, float, float],
            shape: tuple[int, int, int]) -> "BBox3D":
        """Return a new BBox3D padded in mm (per face), clamped to image bounds.

        All six arguments are scalars in mm. Use _resolve_padding() to produce them
        from the public 9-parameter API.

        LAS index convention (nibabel axcodes "LAS" — each axis index increases
        toward L / A / S, so index 0 is the opposite face):
            rl1 = Right side,    rl2 = Left side
            ap1 = Posterior side, ap2 = Anterior side
            z1  = Inferior side,  z2  = Superior side
        Each low-index face is padded by its own anatomical margin, and each
        high-index face by the complementary one.
        """
        rl_mm, ap_mm, si_mm = zooms
        RL, AP, Z = shape
        return BBox3D(
            rl1=max(0,  self.rl1 - int(np.ceil(right / rl_mm))),
            rl2=min(RL, self.rl2 + int(np.ceil(left / rl_mm))),
            ap1=max(0,  self.ap1 - int(np.ceil(posterior / ap_mm))),
            ap2=min(AP, self.ap2 + int(np.ceil(anterior / ap_mm))),
            z1=max(0,   self.z1 - int(np.ceil(inferior / si_mm))),
            z2=min(Z,   self.z2 + int(np.ceil(superior / si_mm))),
        )

    def crop(self, img: nib.Nifti1Image, translate: bool = True) -> nib.Nifti1Image:
        """Crop a NIfTI. With translate=True, updates affine so the crop sits at the
        correct world position (required for FSLeyes overlay).

        Generic over orientation: bbox indices must match img's voxel orientation.
        """
        data = img.get_fdata(dtype=np.float32)
        cropped = data[self.rl1:self.rl2, self.ap1:self.ap2, self.z1:self.z2]
        affine = img.affine.copy()
        if translate:
            affine[:3, 3] = img.affine[:3, :3] @ np.array([self.rl1, self.ap1, self.z1]) \
                           + img.affine[:3, 3]
        return nib.Nifti1Image(cropped, affine)

    def reorient(self,
                 src_shape: tuple[int, int, int],
                 src_ornt: np.ndarray,
                 dst_ornt: np.ndarray) -> tuple["BBox3D", tuple[int, int, int]]:
        """Reorient bbox voxel indices from src to dst orientation.

        Returns (new_bbox, new_shape). nibabel convention:
        ornt_transform(src, dst)[src_ax] = [dst_ax, flip].
        """
        src_ranges = [(self.rl1, self.rl2), (self.ap1, self.ap2), (self.z1, self.z2)]
        T = ornt_transform(src_ornt, dst_ornt)

        dst_ranges: list[tuple[int, int] | None] = [None, None, None]
        dst_shape:  list[int | None] = [None, None, None]

        for src_ax, (dst_ax, flip) in enumerate(T):
            dst_ax = int(dst_ax)
            n = int(src_shape[src_ax])
            lo, hi = src_ranges[src_ax]
            dst_ranges[dst_ax] = (lo, hi) if flip == 1 else (n - hi, n - lo)
            dst_shape[dst_ax] = n

        (a1, a2), (b1, b2), (c1, c2) = dst_ranges  # type: ignore[misc]
        return BBox3D(a1, a2, b1, b2, c1, c2), tuple(dst_shape)  # type: ignore[return-value]


# ─── Orientation helpers ──────────────────────────────────────────────────────

def reorient_to_las(img: nib.Nifti1Image) -> nib.Nifti1Image:
    current = nib.io_orientation(img.affine)
    target = axcodes2ornt(("L", "A", "S"))
    return img.as_reoriented(ornt_transform(current, target))


# ─── Resampling ───────────────────────────────────────────────────────────────

def resample_for_inference(img_las: nib.Nifti1Image,
                           si_res: float,
                           inplane_res: float | None) -> nib.Nifti1Image:
    """Resample LAS image to match training preprocessing resolution (order=1)."""
    rl_mm, ap_mm, si_mm = [float(v) for v in img_las.header.get_zooms()[:3]]
    target_rl = inplane_res if inplane_res is not None else rl_mm
    target_ap = inplane_res if inplane_res is not None else ap_mm
    if (abs(target_rl - rl_mm) < 0.01
            and abs(target_ap - ap_mm) < 0.01
            and abs(si_res - si_mm) < 0.01):
        return img_las
    return resample_to_output(img_las, voxel_sizes=(target_rl, target_ap, si_res), order=1)


# ─── Slice extraction ─────────────────────────────────────────────────────────

def normalize_to_uint8(arr: np.ndarray,
                       lo: float | None = None,
                       hi: float | None = None) -> np.ndarray:
    """Normalize arr to uint8.

    If lo/hi are None, compute percentiles 0.5/99.5 from non-background pixels (slice-level).
    Pass pre-computed lo/hi for volume-level normalisation.

    Background detection:
      MRI : background = 0            → threshold = 0    (keep arr > 0)
      CT  : background = air ≈ −1000 HU → threshold = −200 (keep arr > −200)
    CT is auto-detected when arr.min() < −100 (only Hounsfield units go that negative).
    """
    if lo is None or hi is None:
        flat = arr.ravel()
        threshold = -200 if float(flat.min()) < -100 else 0
        nz = flat[flat > threshold]
        if not len(nz):
            return np.zeros_like(arr, dtype=np.uint8)
        lo, hi = np.percentile(nz, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def _volume_percentiles(data: np.ndarray) -> tuple[float, float]:
    """Percentile 0.5/99.5 over every voxel of the whole volume (background included),
    computed once. Must match preprocess.py norm_scope="volume" exactly — see
    process_pair() in that file (training repo). Do not change without updating that too.
    """
    lo, hi = np.percentile(data, [0.5, 99.5])
    return float(lo), float(hi)


def _normalize_volume(data: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalize entire volume to uint8 in one vectorized pass.

    Equivalent to calling normalize_to_uint8 per-slice but ~2× faster because
    clip/scale operates on the full 3D array in one numpy call.
    """
    if hi <= lo:
        return np.zeros_like(data, dtype=np.uint8)
    return ((np.clip(data, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def build_slices(data: np.ndarray, channels: int, norm_scope: str) -> tuple[list, list]:
    """Build all axial slices Superior→Inferior, matching preprocess.py convention.

    norm_scope must be one of the 3 values below — nothing else is implemented, and
    any other string raises rather than silently picking one of these:
      "volume"    — percentile 0.5/99.5 over every voxel of the whole volume, computed
                    once, volume normalized in a single vectorized pass (~2× faster).
      "slice_all" — percentile 0.5/99.5 per slice on ALL voxels (background included).
      "slice"     — percentile 0.5/99.5 per slice on foreground voxels only
                    (>0 for MRI, >-200 for CT).
    Each must match preprocess.py's norm_scope of the same name exactly (training repo).

    3ch: R=Superior neighbour (las_idx+1), G=current, B=Inferior neighbour (las_idx-1).
    Border channels are black (zeros).

    Returns (slices, las_idxs); las_idx 0 = Inferior, Z-1 = Superior.
    """
    RL, AP, Z = data.shape
    black = np.zeros((AP, RL), dtype=np.uint8)

    if norm_scope == "volume":
        lo, hi = _volume_percentiles(data)
        data_u8 = _normalize_volume(data, lo, hi)

        def _get(idx):
            if idx < 0 or idx >= Z:
                return black
            return data_u8[:, :, idx].T[::-1, ::-1]
    elif norm_scope == "slice_all":
        def _get(idx):
            if idx < 0 or idx >= Z:
                return black
            arr = data[:, :, idx].T[::-1, ::-1]
            lo, hi = np.percentile(arr, [0.5, 99.5])
            return normalize_to_uint8(arr, lo, hi)
    elif norm_scope == "slice":
        def _get(idx):
            if idx < 0 or idx >= Z:
                return black
            return normalize_to_uint8(data[:, :, idx]).T[::-1, ::-1]
    else:
        raise ValueError(f"unsupported norm_scope: {norm_scope!r} (expected volume, slice_all, or slice)")

    slices, las_idxs = [], []
    for las_idx in range(Z - 1, -1, -1):   # Superior → Inferior
        cur = _get(las_idx)
        if channels == 3:
            slices.append(np.stack([_get(las_idx + 1), cur, _get(las_idx - 1)], axis=2))
        else:
            slices.append(cur)
        las_idxs.append(las_idx)

    return slices, las_idxs


# ─── YOLO inference ───────────────────────────────────────────────────────────

def infer_slices(model, slices: list, las_idxs: list, conf_thresh: float,
                 device: str | None = None, imgsz: int = 320) -> dict:
    """Run YOLO detection inference on pre-built slices.

    Returns {las_idx: (cx, cy, w, h)} in slice-image normalised coords [0,1].
    imgsz must match training imgsz (read from config.yaml) — required for ONNX
    models which do not embed imgsz in their metadata unlike .pt.
    """
    kw = {"conf": conf_thresh, "imgsz": imgsz, "verbose": False}
    if device:
        kw["device"] = device
    results = model.predict(slices, **kw)
    preds = {}
    for las_idx, res in zip(las_idxs, results):
        if res.boxes is None or len(res.boxes) == 0:
            continue
        best = int(res.boxes.conf.argmax())
        cx, cy, w, h = res.boxes.xywhn[best].tolist()
        preds[las_idx] = (cx, cy, w, h)
    return preds


# ─── Regularization helpers ───────────────────────────────────────────────────

def _si_connected_components(preds: dict) -> list:
    """Group consecutive z-indices of preds into components (ascending z = superior first)."""
    if not preds:
        return []
    zs = sorted(preds)
    comp = [zs[0]]
    comps: list = []
    for z in zs[1:]:
        if z == comp[-1] + 1:
            comp.append(z)
        else:
            comps.append(comp)
            comp = [z]
    comps.append(comp)
    return comps


_CLS_SC_IDX = 1  # index of "sc" class in classifier output


def _letterbox_cls(sl: np.ndarray, imgsz: int) -> np.ndarray:
    """Square letterbox for the classifier — matches LetterboxClsTransform from training.

    Returns float32 NCHW tensor [1, 3, imgsz, imgsz] ready for onnxruntime.
    """
    import cv2
    rgb = sl if sl.ndim == 3 else np.stack([sl] * 3, axis=2)
    h, w = rgb.shape[:2]
    r = imgsz / max(h, w)
    nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w):
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph, pw = imgsz - nh, imgsz - nw
    padded = cv2.copyMakeBorder(rgb, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0


def cls_comp_filter(preds: dict, slices: list, las_idxs: list,
                    cls_sess, cls_conf: float, imgsz: int = 320) -> dict:
    """Keep first cls-validated SI component + all preds below it.

    Iterates components from most superior, runs onnxruntime classifier slice by
    slice, stops as soon as one component has ≥1 positive slice (conf ≥ cls_conf).
    Returns all preds with z ≥ min_z of the validated component.
    Fallback: returns all preds if no component is validated.
    """
    comps = _si_connected_components(preds)
    slice_map = {idx: sl for idx, sl in zip(las_idxs, slices)}

    for comp in comps:
        for z in comp:
            if z not in slice_map:
                continue
            out = cls_sess.run(None, {"images": _letterbox_cls(slice_map[z], imgsz)})[0][0]
            if float(out[_CLS_SC_IDX]) >= cls_conf:
                return {z_: b for z_, b in preds.items() if z_ >= min(comp)}
    return preds


def _graphreg_edge_broken(preds: dict, z_i: int, z_j: int,
                          H: int, W: int,
                          ap_mm: float, rl_mm: float, si_mm: float) -> bool:
    hop = z_j - z_i
    if hop * si_mm >= 40.0:
        return True
    cx_i, cy_i, w_i, h_i = preds[z_i]
    cx_j, cy_j, w_j, h_j = preds[z_j]
    return (abs((cx_j + w_j / 2) - (cx_i + w_i / 2)) * W * rl_mm > 15.0 * hop or
            abs((cx_j - w_j / 2) - (cx_i - w_i / 2)) * W * rl_mm > 15.0 * hop or
            abs((cy_j - h_j / 2) - (cy_i - h_i / 2)) * H * ap_mm > 25.0 * hop or
            abs((cy_j + h_j / 2) - (cy_i + h_i / 2)) * H * ap_mm > 25.0 * hop)


def graphtrim_superior_filter(preds: dict, H: int, W: int,
                              ap_mm: float, rl_mm: float, si_mm: float) -> dict:
    """Remove superior outlier detections by checking only the 2 topmost SI edges.

    If the edge between the 1st↔2nd or 2nd↔3rd most superior slice is broken
    (SI gap ≥ 40mm or face shift exceeds per-hop threshold), slices above the
    first broken edge are discarded.
    """
    if len(preds) <= 2:
        return preds
    zs = sorted(preds)
    cut = 0
    for i in range(min(2, len(zs) - 1)):
        if _graphreg_edge_broken(preds, zs[i], zs[i + 1], H, W, ap_mm, rl_mm, si_mm):
            cut = i + 1
    return preds if cut == 0 else {z: preds[z] for z in zs[cut:]}


# ─── Per-slice → LAS bbox aggregation ─────────────────────────────────────────

def aggregate_bbox_3d(preds: dict,
                      RL_nat: int, AP_nat: int, Z_nat: int,
                      si_zoom: float) -> BBox3D:
    """Aggregate per-slice detections → BBox3D in native LAS voxels.

    Slices were presented as (AP, RL) with T[::-1, ::-1], so to map back to native LAS:
      rl_c = cx * RL_nat  (no flip — kept as in original logic)
      ap_c = (1 - cy) * AP_nat
    Normalised in-plane coords are FOV-preserving → valid in native space directly.

    si_zoom = si_mm_nat / si_res; z_nat = round(las_idx_inf / si_zoom).
    """
    rl1s, rl2s, ap1s, ap2s, zs = [], [], [], [], []
    for las_idx_inf, (cx, cy, w, h) in preds.items():
        z_nat = min(Z_nat - 1, round(las_idx_inf / si_zoom))
        rl_c = cx * RL_nat
        ap_c = (1.0 - cy) * AP_nat
        rl_half = w / 2 * RL_nat
        ap_half = h / 2 * AP_nat
        rl1s.append(max(0,      int(rl_c - rl_half)))
        rl2s.append(min(RL_nat, int(rl_c + rl_half)))
        ap1s.append(max(0,      int(ap_c - ap_half)))
        ap2s.append(min(AP_nat, int(ap_c + ap_half)))
        zs.append(z_nat)
    return BBox3D(min(rl1s), max(rl2s), min(ap1s), max(ap2s), min(zs), max(zs))


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def _stem(input_path: str) -> tuple[Path, str]:
    inp = Path(input_path)
    stem = inp.name.replace(".nii.gz", "").replace(".nii", "")
    return inp.parent, stem


_YELLOW = "\033[33m"
_RESET = "\033[0m"


def _warn_overwrite(path: Path) -> None:
    if path.exists():
        print(f"{_YELLOW}Warning : {path} already exists and will be overwritten{_RESET}")


def _write_bbox_txt(path: Path, bbox: BBox3D) -> None:
    """Write inclusive voxel indices compatible with SCT's ImageCropper.get_bbox_from_minmax()."""
    _warn_overwrite(path)
    with open(path, "w") as f:
        f.write("# Bounding box in native voxel space (inclusive indices)\n")
        f.write("# xmin xmax ymin ymax zmin zmax\n")
        f.write(f"{bbox.rl1} {bbox.rl2 - 1} {bbox.ap1} {bbox.ap2 - 1} {bbox.z1} {bbox.z2 - 1}\n")


def _read_bbox_txt(path: Path) -> tuple[int, int, int, int, int, int]:
    """Read inclusive voxel indices from a bbox txt file. Returns (xmin, xmax, ymin, ymax, zmin, zmax)."""
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        xmin, xmax, ymin, ymax, zmin, zmax = map(int, line.split())
        return xmin, xmax, ymin, ymax, zmin, zmax
    raise ValueError(f"No coordinate line found in {path}")


# ─── Detection pipeline ───────────────────────────────────────────────────────

def detect(img_path: "str | Path | nib.Nifti1Image",
           config: dict | None = None,
           model_path: str | None = None,
           pad_superior: float | None = None,
           pad_inferior: float | None = None,
           pad_left: float | None = None,
           pad_right: float | None = None,
           pad_anterior: float | None = None,
           pad_posterior: float | None = None,
           pad_si: float | None = None,
           pad_rl: float | None = None,
           pad_ap: float | None = None,
           conf: float | None = None,
           regularization: str | None = None,
           cls_conf: float | None = None,
           device: str | None = None,
           norm_scope: str | None = None) -> dict:
    """Detect the spinal cord bounding box. Pure — no files written.

    This is the primary entry point for inference pipelines. Call once, then
    pass the returned context to crop() for any number of volumes (image, labels…).

    Args:
        img_path:       Path to the input NIfTI image (str or Path), or a pre-loaded
                        nib.Nifti1Image (any orientation, any contrast).
        config:         Config dict (loaded from config.yaml if None).
        model_path:     Path to model file (auto-downloaded if None).
        pad_superior:   Superior padding mm (default 40). Individual > symmetric > default.
        pad_inferior:   Inferior padding mm (default 100).
        pad_left:       Left padding mm (default 15).
        pad_right:      Right padding mm (default 15).
        pad_anterior:   Anterior padding mm (default 15).
        pad_posterior:  Posterior padding mm (default 22).
        pad_si:         Symmetric SI — overridden by pad_superior/inferior.
        pad_rl:         Symmetric RL — overridden by pad_left/right.
        pad_ap:         Symmetric AP — overridden by pad_anterior/posterior.
        conf:           Detection confidence threshold (default: from config.yaml).
        regularization: "cls" (default), "graphtrim", or "none".
        cls_conf:       CLS classifier confidence threshold (default: 0.5).
        device:         "cpu", "cuda", "mps" — passed to ultralytics YOLO.predict().
        norm_scope:     "volume", "slice_all", or "slice" (default: from config.yaml —
                        must match training, see build_slices() for what each computes).

    Returns:
        bbox dict with:
          xmin, xmax, ymin, ymax, zmin, zmax — inclusive bbox in native voxel space
          original_axcodes                   — e.g. "RAS", "LPI"
          _original_img, _img_las, _bbox_pad_las, _original_ornt  (private keys)

    Example::

        from sc_crop import detect, crop
        import nibabel as nib

        bbox        = detect("t2.nii.gz")
        crop_img   = crop(nib.load("t2.nii.gz"),       bbox)
        crop_label = crop(nib.load("t2_label.nii.gz"), bbox)

        # Custom padding:
        bbox = detect("t2.nii.gz", pad_superior=50, pad_inferior=80)
        bbox = detect("t2.nii.gz", pad_si=30)                    # symmetric SI
        bbox = detect("t2.nii.gz", pad_si=30, pad_inferior=60)   # symmetric + override
    """
    config = config if config is not None else load_config()
    si_res = config["si_res"]
    inplane_res = config.get("inplane_res")
    channels = config.get("channels", 3)
    conf = conf if conf is not None else config.get("conf", 0.1)
    regularization = regularization if regularization is not None else config.get("regularization", "cls")
    cls_conf = cls_conf if cls_conf is not None else config.get("cls_conf", 0.5)
    # Must match the normalisation the shipped detector was trained with — an explicit
    # kwarg/CLI flag can still override it, but the silent default always defers to
    # config.yaml, never to a value hardcoded here.
    norm_scope = norm_scope if norm_scope is not None else config["norm_scope"]
    assert norm_scope in ("volume", "slice_all", "slice"), \
        f"unsupported norm_scope: {norm_scope!r} (expected volume, slice_all, or slice)"
    imgsz = config.get("imgsz", 320)

    pad_left, pad_right, pad_anterior, pad_posterior, pad_superior, pad_inferior = _resolve_padding(
        pad_si=pad_si, pad_superior=pad_superior, pad_inferior=pad_inferior,
        pad_rl=pad_rl, pad_left=pad_left,         pad_right=pad_right,
        pad_ap=pad_ap, pad_anterior=pad_anterior, pad_posterior=pad_posterior,
    )

    if isinstance(img_path, nib.Nifti1Image):
        img = img_path
        img_name = getattr(img.file_map.get("image"), "filename", None)
        img_name = Path(img_name).name if img_name else "NIfTI"
    else:
        img = nib.load(img_path)
        img_name = Path(img_path).name
    original_ornt = nib.io_orientation(img.affine)
    original_axcodes = "".join(str(a) for a in nib.aff2axcodes(img.affine))
    img_las = reorient_to_las(img)
    las_ornt = axcodes2ornt(("L", "A", "S"))
    zooms = tuple(float(v) for v in img_las.header.get_zooms()[:3])
    shape = img_las.shape

    print(f"Input   : {img_name}  shape={img.shape}  ornt={original_axcodes}")

    from .download import ensure_cls_model, ensure_model
    from ultralytics import YOLO
    import onnxruntime as ort
    model_file = Path(model_path) if model_path else ensure_model()
    det_model = YOLO(str(model_file), task="detect")
    cls_sess = ort.InferenceSession(str(ensure_cls_model())) if regularization == "cls" else None

    si_zoom = zooms[2] / si_res
    img_inf = resample_for_inference(img_las, si_res, inplane_res)
    data_inf = img_inf.get_fdata(dtype=np.float32)

    slices, las_idxs = build_slices(data_inf, channels, norm_scope)

    preds = infer_slices(det_model, slices, las_idxs, conf, device, imgsz)
    print(f"Detected: {len(preds)}/{data_inf.shape[2]} slices")

    if regularization == "cls":
        preds = cls_comp_filter(preds, slices, las_idxs, cls_sess, cls_conf, imgsz)
        print(f"Cls reg : {len(preds)} slices kept (conf≥{cls_conf})")
    elif regularization == "graphtrim":
        inf_zooms = tuple(float(v) for v in img_inf.header.get_zooms()[:3])
        H_inf, W_inf = data_inf.shape[1], data_inf.shape[0]
        preds = graphtrim_superior_filter(preds, H_inf, W_inf,
                                          inf_zooms[1], inf_zooms[0], inf_zooms[2])
        print(f"Graphtrim: {len(preds)} slices kept")

    if not preds:
        raise RuntimeError("No spinal cord detected — check the volume or lower --conf")

    bbox = aggregate_bbox_3d(preds, shape[0], shape[1], shape[2], si_zoom)
    bbox_pad = bbox.pad(pad_left, pad_right, pad_anterior, pad_posterior,
                        pad_superior, pad_inferior, zooms, shape)
    bbox_pad_orig, _ = bbox_pad.reorient(shape, las_ornt, original_ornt)

    xmin, xmax = bbox_pad_orig.rl1, bbox_pad_orig.rl2 - 1
    ymin, ymax = bbox_pad_orig.ap1, bbox_pad_orig.ap2 - 1
    zmin, zmax = bbox_pad_orig.z1,  bbox_pad_orig.z2 - 1
    print(f"BBox    : xmin={xmin} xmax={xmax}  ymin={ymin} ymax={ymax}  zmin={zmin} zmax={zmax}")

    return {
        "original_axcodes": original_axcodes,
        "xmin": xmin, "xmax": xmax,
        "ymin": ymin, "ymax": ymax,
        "zmin": zmin, "zmax": zmax,
        # private keys — used by CLI and uncrop
        "_original_img":  img,
        "_img_las":       img_las,
        "_bbox_pad_las":  bbox_pad,
        "_original_ornt": original_ornt,
    }


# ─── High-level inference helpers ─────────────────────────────────────────────


def crop(img: "str | Path | nib.Nifti1Image", bbox: dict,
         translate: bool = True) -> "nib.Nifti1Image":
    """Crop a NIfTI image to the bbox detected by detect().

    Works for any volume in the same space as the image passed to detect() —
    use it for both the image and its label(s).

    Args:
        img:       NIfTI image to crop — str/Path (loaded automatically) or
                   a pre-loaded nib.Nifti1Image.
        bbox:      Context dict returned by detect() or detect_and_crop().
        translate: If True (default), update the affine so the crop sits at the
                   correct world position (required for FSLeyes overlay).

    Returns:
        nib.Nifti1Image cropped to the detected bbox.

    Example::

        from sc_crop import detect, crop

        img  = nib.load("t2.nii.gz")
        bbox = detect(img)                    # pass NIfTI directly
        crop_img   = crop(img,           bbox)
        crop_label = crop("t2_seg.nii.gz", bbox)  # or pass a path
    """
    if not isinstance(img, nib.Nifti1Image):
        img = nib.load(img)
    xmin, xmax = bbox["xmin"], bbox["xmax"]
    ymin, ymax = bbox["ymin"], bbox["ymax"]
    zmin, zmax = bbox["zmin"], bbox["zmax"]

    data = np.asarray(img.dataobj)
    affine = img.affine.copy()
    if translate:
        affine[:3, 3] = (img.affine @ np.array([xmin, ymin, zmin, 1.0]))[:3]
    return nib.Nifti1Image(data[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1], affine, img.header)


def detect_and_crop(img_path, **kwargs) -> tuple:
    """Convenience wrapper: detect + crop the image in one call.

    Equivalent to::

        bbox = detect(img_path, **kwargs)
        return crop(nib.load(img_path), bbox), bbox

    Use detect() + crop() directly when you need to crop multiple volumes
    (image + label) with the same bbox.

    Returns:
        (crop_nii, bbox) where crop_nii is the cropped image and bbox is passed
        to crop() or uncrop().
    """
    bbox = detect(img_path, **kwargs)
    return crop(bbox["_original_img"], bbox), bbox


def uncrop(seg_nii, bbox) -> "nib.Nifti1Image":
    """Place any cropped volume back into the full original image space.

    The volume must be in the same orientation as the original image.
    If your model reoriented the crop (e.g., to RPI), reorient back
    before calling this function (see detect_and_crop() example).

    Args:
        seg_nii: NIfTI volume in cropped space (original orientation).
        bbox:     Context dict returned by detect() or detect_and_crop().

    Returns:
        nib.Nifti1Image with segmentation padded to the full original image space,
        using the original affine and header.
    """
    original_img = bbox["_original_img"]
    xmin, xmax = bbox["xmin"], bbox["xmax"]
    ymin, ymax = bbox["ymin"], bbox["ymax"]
    zmin, zmax = bbox["zmin"], bbox["zmax"]

    dtype = seg_nii.get_data_dtype()
    full = np.zeros(original_img.shape[:3], dtype=dtype)
    seg_arr = np.asarray(seg_nii.dataobj).astype(dtype)
    full[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] = seg_arr

    return nib.Nifti1Image(full, original_img.affine, original_img.header)
