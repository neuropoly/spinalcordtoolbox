"""
Postprocessing on generated segmentation: removing outliers, filling holes, etc.

Copyright (c) 2022 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import logging
import numpy as np
from scipy.ndimage import label, binary_fill_holes

from spinalcordtoolbox.process_seg import compute_shape


logger = logging.getLogger(__name__)


def _fill_z_holes(zz_lst, data, z_spaccing):
    data_interpol = np.copy(data)

    for z_hole_start, z_hole_end in list(_list2range(zz_lst)):
        z_ref_start, z_ref_end = z_hole_start - 1, z_hole_end
        slice_ref_start, slice_ref_end = data[:, :, z_ref_start], data[:, :, z_ref_end]

        hole_cur_lst = list(range(z_hole_start, z_hole_end))
        lenght_hole = len(hole_cur_lst) + 1
        phys_lenght_hole = lenght_hole * z_spaccing

        denom_interpolation = (lenght_hole + 1)

        if phys_lenght_hole < 10:
            logger.warning(f'Filling a hole in the segmentation around z_slice #:{z_ref_start}')

            for idx_z, z_hole_cur in enumerate(hole_cur_lst):
                num_interpolation = (lenght_hole - idx_z - 1) * slice_ref_start  # Contribution of the bottom ref slice
                num_interpolation += (idx_z + 1) * slice_ref_end  # Contribution of the top ref slice

                slice_interpolation = num_interpolation * 1. / denom_interpolation
                slice_interpolation = (slice_interpolation > 0).astype(int)

                data_interpol[:, :, z_hole_cur] = slice_interpolation

    return data_interpol


def _list2range(lst):
    tmplst = lst[:]
    tmplst.sort()
    start = tmplst[0]

    currentrange = [start, start + 1]

    for item in tmplst[1:]:
        if currentrange[1] == item:  # contiguous
            currentrange[1] += 1
        else:  # new range start
            yield list(currentrange)
            currentrange = [item, item + 1]

    yield list(currentrange)  # last range


def _remove_blobs(data):
    """Remove false positive blobs, likely occuring in brain sections."""
    labeled_obj, num_obj = label(data)
    if num_obj > 1:  # If there is more than one connected object
        bigger_obj = (labeled_obj == (np.bincount(labeled_obj.flat)[1:].argmax() + 1))

        data2clean = np.copy(data)

        # remove blobs only above the bigger connected object
        z_max = np.max(np.where(bigger_obj)[2])
        data2clean[:, :, :z_max + 1] = 0

        labeled_obj2clean, num_obj2clean = label(data2clean)
        if num_obj2clean:  # If there is connected object above the biffer connected one
            for obj_id in range(1, num_obj2clean + 1):
                # if the blob has a volume < 10% of the bigger connected object, then remove it
                if np.sum(labeled_obj2clean == obj_id) < 0.1 * np.sum(bigger_obj):
                    logger.warning(f'Removing small objects above slice #{z_max}')
                    data[np.where(labeled_obj2clean == obj_id)] = 0

    return data


def _remove_extrem_holes(z_lst, end_z, start_z=0):
    """Remove extrem holes from the list of holes so that we will not interpolate on the extrem slices."""
    if start_z in z_lst:
        while start_z in z_lst:
            z_lst = z_lst[1:]
            start_z += 1
        if len(z_lst):
            z_lst.pop(0)

    if end_z in z_lst:
        while end_z in z_lst:
            z_lst = z_lst[:-1]
            end_z -= 1

    return z_lst


def _remove_isolated_voxels_on_the_edge(im_seg, n_slices=5):
    """
    Remove isolated voxels on the edge if the CSA of the edge slice is smaller than half the median of adjacent slices.

    :param im_seg:
    :param n_slices: Number of adjacent slices to consider. If not enough slices, this test will be bypassed.
    :return:
    """
    # Compute shape info across segmentation
    metrics, _ = compute_shape(im_seg, angle_correction=False)
    # Extract CSA and get min/max index, corresponding to the top/bottom edges of the segmentation
    ind_nonnan = np.where(np.isnan(metrics['area'].data) == False)[0]  # noqa: E712
    ind_min, ind_max = ind_nonnan[0], ind_nonnan[-1]
    # Check if the CSA at the edge is inferior to half of the median across adjacent slices...
    # ... for the top slice
    if metrics['area'].data[ind_min] < np.median(metrics['area'].data[ind_min:n_slices])/2:
        im_seg.data[:, :, ind_min] = 0
        logger.warning(f'Found isolated voxels on slice {ind_min}, Removing them')
    # ... for the bottom slice
    if metrics['area'].data[ind_max] < np.median(metrics['area'].data[ind_max-n_slices+1:ind_max+1])/2:
        im_seg.data[:, :, ind_max] = 0
        logger.warning(f'Found isolated voxels on slice {ind_min}, Removing them')
    return im_seg


def post_processing_volume_wise(im_seg):
    """
    Post processing function to clean the input segmentation: fill holes, remove edge outlier, etc.
    Note: This function is compatible with soft segmentation (i.e. float between 0-1).
    """
    data_bin = (im_seg.data > 0).astype(int)  # will binarize soft segmentation

    # Remove blobs
    data_bin = _remove_blobs(data_bin)

    # Fill z_holes, i.e. interpolate for z_slice not segmented
    zz_zeros = [zz for zz in range(im_seg.dim[2]) if 1 not in list(np.unique(data_bin[:, :, zz]))]
    zz_holes = _remove_extrem_holes(zz_zeros, im_seg.dim[2] - 1, 0)
    data_pp = _fill_z_holes(zz_holes, data_bin, im_seg.dim[6]) if len(zz_holes) else data_bin

    im_seg.data[np.where(data_pp == 0)] = 0  # to be compatible with soft segmentation

    # Set isolated voxels at edge slices to zero
    im_seg = _remove_isolated_voxels_on_the_edge(im_seg)

    return im_seg


def keep_largest_object(prediction, x_cOm, y_cOm):
    """
    Keep the largest connected object from the input array (2D or 3D).
    See the following commit message for more details:
    https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4546/commits/213062a80222730171592453713c5e78931dae1e
    Note: This function only works for binary segmentation.

    When x_c0m/y_c0m are None, this function is identical to:
    https://github.com/ivadomed/ivadomed/blob/1fccf77239985fc3be99161f9eb18c9470d65206/ivadomed/postprocessing.py#L99-L116

    :param prediction: ndarray: Input binary segmentation. It could be 2D or 3D.
    :param x_cOm: int: X center of mass of the segmentation for the previous 2d slice
    :param y_cOm: int: Y center of mass of the segmentation for the previous 2d slice
    :return: prediction: ndarray: Output binary segmentation. 2D or 3D depending on the input.
    """
    # Find number of closed objects using skimage "label"
    labeled_obj, num_obj = label(prediction)
    # If more than one object is found, keep the largest one
    if num_obj > 1:
        # If the center of mass is not provided (e.g. is first slice, or segmentation is empty), keep the largest object
        if x_cOm is None or np.isnan(x_cOm):
            prediction[np.where(labeled_obj != (np.bincount(labeled_obj.flat)[1:].argmax() + 1))] = 0
        # If the center of mass is provided,
        else:
            idx_z_minus_1 = np.bincount(labeled_obj.flat)[1:].argmax() + 1
            for idx in range(1, num_obj + 1):
                z_idx = labeled_obj == idx
                if z_idx[int(x_cOm), int(y_cOm)]:
                    idx_z_minus_1 = idx
            prediction[np.where(labeled_obj != idx_z_minus_1)] = 0
    return prediction


def fill_holes(predictions, structure=None):
    """Fill holes in the predictions using a given structuring element.
    Note: This function only works for binary segmentation.

    Taken from:
    https://github.com/ivadomed/ivadomed/blob/master/ivadomed/postprocessing.py#L143

    Args:
        predictions (ndarray or nibabel object): Input binary segmentation. Image could be 2D or 3D.
        structure (tuple of integers): Structuring element, number of ints equals
            number of dimensions in the input array.

    Returns:
        ndrray or nibabel (same object as the input). Output type is int.
    """
    assert np.array_equal(predictions, predictions.astype(bool)), (
        "fill_holes expects a binary segmentation as input. "
        "Use `-thr` to binarize the prediction before using fill_holes."
    )
    if structure is None:
        structure = [3] * len(predictions.shape)
    assert len(structure) == len(predictions.shape)
    return binary_fill_holes(predictions, structure=np.ones(structure)).astype(int)
