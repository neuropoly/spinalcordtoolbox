# This file is vendored from ivadomed 2.9.10 (https://github.com/ivadomed/ivadomed),
# originally `ivadomed/loader/utils.py`, licensed under the MIT License.
# See the LICENSE.md file in this directory for the full license text.
#
# Modifications for SCT:
#   - Removed training and dataset-splitting helpers that SCT doesn't use.
#   - Fixed flake8 warnings, without changing behavior.

from __future__ import annotations
import collections.abc
import re
import sys
import gc
from tempfile import mkdtemp

import numpy as np
import torch
from loguru import logger
import nibabel as nib
import random
import typing
if typing.TYPE_CHECKING:
    from typing import Optional

__numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

TRANSFORM_PARAMS = ['elastic', 'rotation', 'scale', 'offset', 'crop_params', 'reverse',
                    'translation', 'gaussian_noise']

# Ordered list of supported file extensions
# TODO: Implement support of the following OMETIFF formats (#739):
# [".ome.tif", ".ome.tiff", ".ome.tf2", ".ome.tf8", ".ome.btf"]
# They are included in the list to avoid a ".ome.tif" or ".ome.tiff" following the ".tif" or ".tiff" pipeline
EXT_LST = [".nii", ".nii.gz", ".ome.tif", ".ome.tiff", ".ome.tf2", ".ome.tf8", ".ome.btf", ".tif",
           ".tiff", ".png", ".jpg", ".jpeg"]


def imed_collate(batch: dict) -> dict | list | str | torch.Tensor:
    """Collates data to create batches

    Args:
        batch (dict): Contains input and gt data with their corresponding metadata.

    Returns:
        list or dict or str or tensor: Collated data.
    """
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        stacked = torch.stack(batch, 0)
        return stacked
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return __numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], collections.abc.Mapping):
        return {key: imed_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.abc.Sequence):
        return [imed_collate(samples) for samples in batch]

    return batch


def filter_roi(roi_data: np.ndarray, nb_nonzero_thr: int) -> bool:
    """Filter slices from dataset using ROI data.

    This function filters slices (roi_data) where the number of non-zero voxels within the
    ROI slice (e.g. centerline, SC segmentation) is inferior or equal to a given threshold
    (nb_nonzero_thr).

    Args:
        roi_data (nd.array): ROI slice.
        nb_nonzero_thr (int): Threshold.

    Returns:
        bool: True if the slice needs to be filtered, False otherwise.
    """
    # Discard slices with less nonzero voxels than nb_nonzero_thr
    return not np.any(roi_data) or np.count_nonzero(roi_data) <= nb_nonzero_thr


def orient_img_hwd(data: np.ndarray, slice_axis: int) -> np.ndarray:
    """Orient a given RAS image to height, width, depth according to slice axis.

    Args:
        data (ndarray): RAS oriented data.
        slice_axis (int): Indicates the axis used for the 2D slice extraction:
            Sagittal: 0, Coronal: 1, Axial: 2.

    Returns:
        ndarray: Array oriented with the following dimensions: (height, width, depth).
    """
    if slice_axis == 0:
        return data.transpose(2, 1, 0)
    elif slice_axis == 1:
        return data.transpose(2, 0, 1)
    elif slice_axis == 2:
        return data


def orient_img_ras(data: np.ndarray, slice_axis: int) -> np.ndarray:
    """Orient a given array with dimensions (height, width, depth) to RAS orientation.

    Args:
        data (ndarray): Data with following dimensions (Height, Width, Depth).
        slice_axis (int): Indicates the axis used for the 2D slice extraction:
            Sagittal: 0, Coronal: 1, Axial: 2.

    Returns:
        ndarray: Array oriented in RAS.
    """

    if slice_axis == 0:
        return data.transpose(2, 1, 0) if len(data.shape) == 3 else data.transpose(0, 3, 2, 1)
    elif slice_axis == 1:
        return data.transpose(1, 2, 0) if len(data.shape) == 3 else data.transpose(0, 2, 3, 1)
    elif slice_axis == 2:
        return data


def orient_shapes_hwd(data: list | tuple, slice_axis: int) -> np.ndarray:
    """Swap dimensions according to match the height, width, depth orientation.

    Args:
        data (list or tuple): Shape or numbers associated with each image dimension
            (e.g. image resolution).
        slice_axis (int): Indicates the axis used for the 2D slice extraction:
            Sagittal: 0, Coronal: 1, Axial: 2.

    Returns:
        ndarray: Reoriented vector.
    """
    if slice_axis == 0:
        return np.array(data)[[2, 1, 0]]
    elif slice_axis == 1:
        return np.array(data)[[2, 0, 1]]
    elif slice_axis == 2:
        return np.array(data)


def update_metadata(metadata_src_lst: list, metadata_dest_lst: list) -> list:
    """Update metadata keys with a reference metadata.

    A given list of metadata keys will be changed and given the values of the reference metadata.

    Args:
        metadata_src_lst (list): List of source metadata used as reference for the
            destination metadata.
        metadata_dest_lst (list): List of metadate that needs to be updated.

    Returns:
        list: updated metadata list.
    """
    if metadata_src_lst and metadata_dest_lst:
        if not isinstance(metadata_dest_lst[0], list):  # annotation from one rater only
            metadata_dest_lst[0]._update(metadata_src_lst[0], TRANSFORM_PARAMS)
        else:  # annotations from several raters
            for idx, _ in enumerate(metadata_dest_lst[0]):
                metadata_dest_lst[0][idx]._update(metadata_src_lst[0], TRANSFORM_PARAMS)
    return metadata_dest_lst


def reorient_image(arr: np.ndarray, slice_axis: int, nib_ref: nib, nib_ref_canonical: nib) -> np.ndarray:
    """Reorient an image to match a reference image orientation.

    It reorients a array to a given orientation and convert it to a nibabel object using the
    reference nibabel header.

    Args:
        arr (ndarray): Input array, array to re orient.
        slice_axis (int): Indicates the axis used for the 2D slice extraction:
            Sagittal: 0, Coronal: 1, Axial: 2.
        nib_ref (nibabel): Reference nibabel object, whose header is used.
        nib_ref_canonical (nibabel): `nib_ref` that has been reoriented to canonical orientation (RAS).
    """
    # Orient image in RAS according to slice axis
    arr_ras = orient_img_ras(arr, slice_axis)

    # https://gitship.com/neuroscience/nibabel/blob/master/nibabel/orientations.py
    ref_orientation = nib.orientations.io_orientation(nib_ref.affine)
    ras_orientation = nib.orientations.io_orientation(nib_ref_canonical.affine)
    # Return the orientation that transforms from ras to ref_orientation
    trans_orient = nib.orientations.ornt_transform(ras_orientation, ref_orientation)
    # apply transformation
    return nib.orientations.apply_orientation(arr_ras, trans_orient)


def get_file_extension(filename: str) -> Optional[str]:
    """ Get file extension if it is supported
    Args:
        filename (str): Path of the file.

    Returns:
        str: File extension
    """
    # Find the first match from the list of supported file extensions
    extension = next((ext for ext in EXT_LST if filename.lower().endswith(ext)), None)
    return extension


def update_filename_to_nifti(filename: str) -> str:
    """
    Update filename extension to 'nii.gz' if not a NifTI file.

    This function is used to help make non-NifTI files (e.g. PNG/TIF/JPG)
    compatible with NifTI-only pipelines. The expectation is that a NifTI
    version of the file has been created alongside the original file, which
    allows the extension to be cleanly swapped for a `.nii.gz` extension.

    Args:
        filename (str): Path of original file.

    Returns:
        str: Path of the corresponding NifTI file.
    """
    extension = get_file_extension(filename)
    if "nii" not in extension:
        filename = filename.replace(extension, ".nii.gz")
    return filename


def dropout_input(seg_pair: dict) -> dict:
    """Applies input-level dropout: zero to all channels minus one will be randomly set to zeros. This function verifies
    if some channels are already empty. Always at least one input channel will be kept.

    Args:
        seg_pair (dict): Batch containing torch tensors (input and gt) and metadata.

    Return:
        seg_pair (dict): Batch containing torch tensors (input and gt) and metadata with channel(s) dropped.
    """
    n_channels = seg_pair['input'].size(0)
    # Verify if the input is multichannel
    if n_channels > 1:
        # Verify if some channels are already empty
        n_unique_values = [len(torch.unique(input_data)) > 1 for input_data in seg_pair['input']]
        idx_empty = np.where(np.invert(n_unique_values))[0]

        # Select how many channels will be dropped between 0 and n_channels - 1 (keep at least one input)
        n_dropped = random.randint(0, n_channels - 1)

        if n_dropped > len(idx_empty):
            # Remove empty channel to the number of channels to drop
            n_dropped = n_dropped - len(idx_empty)
            # Select which channels will be dropped
            idx_dropped = []
            while len(idx_dropped) != n_dropped:
                idx = random.randint(0, n_channels - 1)
                # Don't include the empty channel in the dropped channels
                if idx not in idx_empty:
                    idx_dropped.append(idx)
        else:
            idx_dropped = idx_empty

        seg_pair['input'][idx_dropped] = torch.zeros_like(seg_pair['input'][idx_dropped])

    else:
        logger.warning("\n Impossible to apply input-level dropout since input is not multi-channel.")

    return seg_pair


def create_temp_directory() -> str:
    """Creates a temporary directory and returns its path.
    This temporary directory is only deleted when explicitly requested.

    Returns:
        str: Path of the temporary directory.
    """
    import datetime
    time_stamp = datetime.datetime.now().isoformat().replace(":", "")
    temp_folder_location = mkdtemp(prefix="ivadomed_", suffix=f"_{time_stamp}")
    return temp_folder_location


def get_obj_size(obj) -> int:
    """
    Returns the size of an object in bytes. Used to gauge whether storing object in memory vs write to disk.

    Source: https://stackoverflow.com/a/53705610

    Args:
        obj:

    Returns:

    """
    marked = {id(obj)}
    obj_q = [obj]
    object_size = 0

    while obj_q:
        object_size += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return object_size
