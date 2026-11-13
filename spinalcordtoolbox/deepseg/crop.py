"""
sc-crop: crop an image around the spinal cord (or the spinal canal) before deep learning inference.

The structure is detected on each axial slice by a YOLO model (https://github.com/ivadomed/sc-crop). The
preprocessing must match the one used to build the training slices (`training/data/build_yolo_dataset.py`):
average 4D images over time, reorient to LAS, resample to 1 x 1 x 10 mm, normalize to uint8 (0.5-99.5
percentiles of the volume), then cut grayscale axial slices (row 0 = Anterior, column 0 = patient Right).

The bbox dict returned by `detect()` contains xmin, xmax, ymin, ymax, zmin, zmax (inclusive voxel indices of the
original image) and `_original_img` (used by `uncrop()`).

Author: Pierre-Louis Benveniste
"""

import itertools
import os
import shutil

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from nibabel.processing import resample_from_to

from spinalcordtoolbox import download
from spinalcordtoolbox.utils.sys import __deepseg_dir__

MODEL_URL = "https://github.com/ivadomed/sc-crop/releases/download/r20261113/model_detect_canal_sc.pt"
MODEL_PATH = os.path.join(__deepseg_dir__, "model_detect_canal_sc_r20261113", "model_detect_canal_sc.pt")
# For each structure: (class index in the YOLO model, confidence threshold chosen on the validation set, name)
STRUCTURES = {"sc": (0, 0.26, "spinal cord"), "canal": (1, 0.29, "spinal canal")}
VOXEL_SIZE = np.array([1.0, 1.0, 10.0])  # LAS (RL, AP, SI), in mm
IMAGE_SIZE = 320


def get_model_path():
    """Return the path to the YOLO detector, downloading it if needed."""
    if not os.path.isfile(MODEL_PATH):
        path_tmp, _ = download.download_data(MODEL_URL)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        shutil.move(path_tmp, MODEL_PATH)
        shutil.rmtree(os.path.dirname(path_tmp), ignore_errors=True)
    return MODEL_PATH


def resample_image(img):
    """Return `img` averaged over time (if 4D), reoriented to LAS, resampled to VOXEL_SIZE and normalized to uint8."""
    if img.ndim == 4:
        img = nib.Nifti1Image(img.get_fdata(dtype=np.float32).mean(axis=-1), img.affine)
    img = img.as_reoriented(ornt_transform(io_orientation(img.affine), axcodes2ornt("LAS")))
    zooms = np.array(img.header.get_zooms()[:3])
    shape = tuple(np.maximum(np.round(np.array(img.shape[:3]) * zooms / VOXEL_SIZE), 1).astype(int))
    affine = img.affine.copy()
    affine[:3, :3] *= VOXEL_SIZE / zooms
    affine[:3, 3] += img.affine[:3, :3] @ ((VOXEL_SIZE / zooms - 1) / 2)  # same extent as the original grid
    data = resample_from_to(img, (shape, affine), order=1).get_fdata(dtype=np.float32)
    lo, hi = np.percentile(data, (0.5, 99.5))
    data = ((np.clip(data, lo, hi) - lo) / max(hi - lo, 1e-8) * 255).astype(np.uint8)
    return nib.Nifti1Image(data, affine)


def detect(img, pad_superior, pad_inferior, pad_left, pad_right, pad_anterior, pad_posterior, structure="sc"):
    """
    Detect the box around the spinal cord (structure="sc") or the spinal canal (structure="canal"), padded on
    each face by the given margins (mm). Raise RuntimeError if the structure is not detected.
    """
    from ultralytics import YOLO
    cls, conf, name = STRUCTURES[structure]

    resampled = resample_image(img)
    data = np.asarray(resampled.dataobj)
    n_x, n_y, n_z = data.shape
    # Axial slices from superior to inferior, row 0 = Anterior, column 0 = patient Right, 3 identical channels
    slices = list(np.repeat(data.transpose(2, 1, 0)[::-1, ::-1, :, None], 3, axis=-1))
    results = YOLO(get_model_path(), task="detect").predict(source=slices, imgsz=IMAGE_SIZE, conf=conf,
                                                             batch=len(slices), verbose=False)
    # Most confident box of the structure on each slice: (slice index, (x1, y1, x2, y2) in pixels)
    detected = []
    for i, r in enumerate(results):
        boxes = r.boxes[r.boxes.cls == cls]
        if len(boxes):
            detected.append((i, boxes.xyxy[boxes.conf.argmax()].tolist()))
    if not detected:
        raise RuntimeError(f"No {name} detected: the image could not be cropped.")

    # 3D box around all the slice boxes, in voxel coordinates of the resampled image (pixel/slice i -> voxel
    # x = u - 0.5, y = n_y - v - 0.5, z = n_z - 1 - i), padded on the R/P/I (lower) and L/A/S (upper) faces
    boxes = np.array([box for _, box in detected])
    first, last = detected[0][0], detected[-1][0]
    lower = np.array([boxes[:, 0].min() - 0.5, n_y - boxes[:, 3].max() - 0.5, n_z - 1 - last - 0.5])
    upper = np.array([boxes[:, 2].max() - 0.5, n_y - boxes[:, 1].min() - 0.5, n_z - 1 - first + 0.5])
    lower -= np.array([pad_right, pad_posterior, pad_inferior]) / VOXEL_SIZE
    upper += np.array([pad_left, pad_anterior, pad_superior]) / VOXEL_SIZE

    # Voxels of the original image whose center is inside the box (1e-3: rounding errors of the affines)
    corners = np.array(list(itertools.product(*zip(lower, upper))))
    voxels = nib.affines.apply_affine(np.linalg.inv(img.affine) @ resampled.affine, corners)
    shape = np.array(img.shape[:3])
    start = np.clip(np.ceil(voxels.min(axis=0) - 1e-3), 0, shape - 1).astype(int).tolist()
    stop = np.clip(np.floor(voxels.max(axis=0) + 1e-3), 0, shape - 1).astype(int).tolist()
    return {"xmin": start[0], "xmax": stop[0], "ymin": start[1], "ymax": stop[1], "zmin": start[2], "zmax": stop[2],
            "_original_img": img}


def _box(bbox):
    return slice(bbox["xmin"], bbox["xmax"] + 1), slice(bbox["ymin"], bbox["ymax"] + 1), slice(bbox["zmin"], bbox["zmax"] + 1)


def crop(img, bbox):
    """Crop `img` to the bbox, updating the affine so the crop keeps its world position."""
    affine = img.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(img.affine, [bbox["xmin"], bbox["ymin"], bbox["zmin"]])
    return nib.Nifti1Image(np.asarray(img.dataobj)[_box(bbox)], affine, img.header)


def uncrop(img, bbox):
    """Put a cropped volume back in the space of the original image (zeros outside the bbox)."""
    original = bbox["_original_img"]
    full = np.zeros(original.shape[:3], dtype=img.get_data_dtype())
    full[_box(bbox)] = np.asarray(img.dataobj)
    return nib.Nifti1Image(full, original.affine, original.header)


def save_bbox_nifti(bbox, ref_img, fname_out):
    """Save the bbox as a binary mask (1 inside the box) on the grid of `ref_img`."""
    data = np.zeros(ref_img.shape[:3], dtype=np.uint8)
    data[_box(bbox)] = 1
    nib.save(nib.Nifti1Image(data, ref_img.affine, ref_img.header), str(fname_out))


def check_seg_truncation(seg_img, bbox):
    """Return the bbox faces (e.g. ["zmax", "zmin"]) touched by the segmentation, excluding the image edges."""
    seg = np.asanyarray(seg_img.dataobj)
    return [key for ax, (lo, hi) in enumerate([("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax")])
            for key, interior in ((hi, bbox[hi] < seg.shape[ax] - 1), (lo, bbox[lo] > 0))
            if interior and np.any(np.take(seg, bbox[key], axis=ax))]
