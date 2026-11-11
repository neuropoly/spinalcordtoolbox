# This file is vendored from ivadomed 2.9.10 (https://github.com/ivadomed/ivadomed),
# originally `ivadomed/transforms.py`, licensed under the MIT License.
# See the LICENSE.md file in this directory for the full license text.
#
# Modifications for SCT:
#   - Removed `RandomBiasField`, a training-only transform that needs `torchio`.
#   - Fixed flake8 warnings, without changing behavior.
#   - Imports point to `spinalcordtoolbox.ivadomed` instead of `ivadomed`.

import copy
import functools
import math
import numbers
import random

from typing import Tuple

import numpy as np
import torch
from loguru import logger
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter, map_coordinates, affine_transform, label, center_of_mass, binary_dilation, \
    binary_fill_holes, binary_closing
from skimage.exposure import equalize_adapthist
from torchvision import transforms as torchvision_transforms


from spinalcordtoolbox.ivadomed.loader import utils as imed_loader_utils
from spinalcordtoolbox.ivadomed.keywords import TransformationKW, MetadataKW


def multichannel_capable(wrapped):
    """Decorator to make a given function compatible multichannel images.

    Args:
        wrapped: Given function.

    Returns:
        Functions' return.
    """

    @functools.wraps(wrapped)
    def wrapper(self, sample, metadata):
        if isinstance(sample, list):
            list_data, list_metadata = [], []
            for s_cur, m_cur in zip(sample, metadata):
                if len(list_metadata) > 0:
                    if not isinstance(list_metadata[-1], list):
                        imed_loader_utils.update_metadata([list_metadata[-1]], [m_cur])
                    else:
                        imed_loader_utils.update_metadata(list_metadata[-1], [m_cur])
                # Run function for each sample of the list
                data_cur, metadata_cur = wrapped(self, s_cur, m_cur)
                list_data.append(data_cur)
                list_metadata.append(metadata_cur)
            return list_data, list_metadata
        # If sample is None, then return a pair (None, None)
        if sample is None:
            return None, None
        else:
            return wrapped(self, sample, metadata)

    return wrapper


def two_dim_compatible(wrapped):
    """Decorator to make a given function compatible 2D or 3D images.

    Args:
        wrapped: Given function.

    Returns:
        Functions' return.
    """

    @functools.wraps(wrapped)
    def wrapper(self, sample, metadata):
        # Check if sample is 2D
        if len(sample.shape) == 2:
            # Add one dimension
            sample = np.expand_dims(sample, axis=-1)
            # Run transform
            result_sample, result_metadata = wrapped(self, sample, metadata)
            # Remove last dimension
            return np.squeeze(result_sample, axis=-1), result_metadata
        else:
            return wrapped(self, sample, metadata)

    return wrapper


class ImedTransform(object):
    """Base class for transforamtions."""

    def __call__(self, sample, metadata=None):
        raise NotImplementedError("You need to implement the transform() method.")


class Compose(object):
    """Composes transforms together.

    Composes transforms together and split between images, GT and ROI.

    self.transform is a dict:
        - keys: "im", "gt" and "roi"
        - values torchvision_transform.Compose objects.

    Attributes:
        dict_transforms (dict): Dictionary where the keys are the transform names
            and the value their parameters.
        requires_undo (bool): If True, does not include transforms which do not have an undo_transform
            implemented yet.

    Args:
        transform (dict): Keys are "im", "gt", "roi" and values are torchvision_transforms.Compose of the
            transformations of interest.
    """

    def __init__(self, dict_transforms, requires_undo=False):
        list_tr_im, list_tr_gt, list_tr_roi = [], [], []
        for transform in dict_transforms.keys():
            parameters = dict_transforms[transform]

            # Get list of data type
            if "applied_to" in parameters:
                list_applied_to = parameters["applied_to"]
            else:
                list_applied_to = ["im", "gt", "roi"]

            # call transform
            if transform in globals():
                if transform == "NumpyToTensor":
                    continue
                params_cur = {k: parameters[k] for k in parameters if k != "applied_to" and k != "preprocessing"}
                transform_obj = globals()[transform](**params_cur)
            else:
                raise ValueError('ERROR: {} transform is not available. '
                                 'Please check its compatibility with your model json file.'.format(transform))

            # check if undo_transform method is implemented
            if requires_undo:
                if not hasattr(transform_obj, 'undo_transform'):
                    logger.info('{} transform not included since no undo_transform available for it.'.format(transform))
                    continue

            if "im" in list_applied_to:
                list_tr_im.append(transform_obj)
            if "roi" in list_applied_to:
                list_tr_roi.append(transform_obj)
            if "gt" in list_applied_to:
                list_tr_gt.append(transform_obj)

        self.transform = {
            "im": torchvision_transforms.Compose(list_tr_im),
            "gt": torchvision_transforms.Compose(list_tr_gt),
            "roi": torchvision_transforms.Compose(list_tr_roi)}

    def __call__(self, sample, metadata, data_type='im', preprocessing=False):
        if self.transform[data_type] is None or len(metadata) == 0:
            # In case self.transform[data_type] is None
            return None, None
        else:
            for tr in self.transform[data_type].transforms:
                sample, metadata = tr(sample, metadata)

            if not preprocessing:
                numpy_to_tensor = NumpyToTensor()
                sample, metadata = numpy_to_tensor(sample, metadata)
            return sample, metadata


class UndoCompose(object):
    """Undo the Compose transformations.

    Call the undo transformations in the inverse order than the "do transformations".

    Attributes:
        compose (torchvision_transforms.Compose):

    Args:
        transforms (torchvision_transforms.Compose):
    """

    def __init__(self, compose):
        self.transforms = compose

    def __call__(self, sample, metadata, data_type='gt'):
        if self.transforms.transform[data_type] is None:
            # In case self.transforms.transform[data_type] is None
            return None, None
        else:
            numpy_to_tensor = NumpyToTensor()
            sample, metadata = numpy_to_tensor.undo_transform(sample, metadata)
            for tr in self.transforms.transform[data_type].transforms[::-1]:
                sample, metadata = tr.undo_transform(sample, metadata)
            return sample, metadata


class UndoTransform(object):
    """Call undo transformation.

    Attributes:
        transform (ImedTransform):

    Args:
        transform (ImedTransform):
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return self.transform.undo_transform(sample)


class NumpyToTensor(ImedTransform):
    """Converts nd array to tensor object."""

    def undo_transform(self, sample, metadata=None):
        """Converts Tensor to nd array."""
        return list(sample.numpy()), metadata

    def __call__(self, sample, metadata=None):
        """Converts nd array to Tensor."""
        sample = np.array(sample)
        # Use np.ascontiguousarray to avoid axes permutations issues
        arr_contig = np.ascontiguousarray(sample, dtype=sample.dtype)
        return torch.from_numpy(arr_contig), metadata


class Resample(ImedTransform):
    """
    Resample image to a given resolution.

    Args:
        hspace (float): Resolution along the first axis, in mm.
        wspace (float): Resolution along the second axis, in mm.
        dspace (float): Resolution along the third axis, in mm.
        interpolation_order (int): Order of spline interpolation. Set to 0 for label data. Default=2.
    """

    def __init__(self, hspace, wspace, dspace=1.):
        self.hspace = hspace
        self.wspace = wspace
        self.dspace = dspace

    @multichannel_capable
    @two_dim_compatible
    def undo_transform(self, sample, metadata=None):
        """Resample to original resolution."""
        assert MetadataKW.DATA_SHAPE in metadata
        is_2d = sample.shape[-1] == 1

        # Get params
        original_shape = metadata[MetadataKW.PRE_RESAMPLE_SHAPE]
        current_shape = sample.shape
        params_undo = [x / y for x, y in zip(original_shape, current_shape)]
        if is_2d:
            params_undo[-1] = 1.0

        # Undo resampling
        data_out = zoom(sample,
                        zoom=params_undo,
                        order=1 if metadata[MetadataKW.DATA_TYPE] == 'gt' else 2)

        # Data type
        data_out = data_out.astype(sample.dtype)

        return data_out, metadata

    @multichannel_capable
    @multichannel_capable  # for multiple raters during training/preprocessing
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        """Resample to a given resolution, in millimeters."""
        # Get params
        # Voxel dimension in mm
        is_2d = sample.shape[-1] == 1
        metadata[MetadataKW.PRE_RESAMPLE_SHAPE] = sample.shape
        # metadata is not a dictionary!
        zooms = list(metadata[MetadataKW.ZOOMS])

        if len(zooms) == 2:
            zooms += [1.0]

        hfactor = zooms[0] / self.hspace
        wfactor = zooms[1] / self.wspace
        dfactor = zooms[2] / self.dspace
        params_resample = (hfactor, wfactor, dfactor) if not is_2d else (hfactor, wfactor, 1.0)

        # Run resampling
        data_out = zoom(sample,
                        zoom=params_resample,
                        order=1 if metadata[MetadataKW.DATA_TYPE] == 'gt' else 2)

        # Data type
        data_out = data_out.astype(sample.dtype)

        return data_out, metadata


class NormalizeInstance(ImedTransform):
    """Normalize a tensor or an array image with mean and standard deviation estimated from the sample itself."""

    @multichannel_capable
    def undo_transform(self, sample, metadata=None):
        # Nothing
        return sample, metadata

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        # if sample uniform: do mean-subtraction
        if sample.std() < 1e-5:
            data_out = (sample - sample.mean())
        # else: normalize sample
        else:
            data_out = (sample - sample.mean()) / sample.std()
        return data_out, metadata


class CroppableArray(np.ndarray):
    """Zero padding slice past end of array in numpy.

    Adapted From: https://stackoverflow.com/a/41155020/13306686
    """

    def __getitem__(self, item):
        all_in_slices = []
        pad = []
        for dim in range(self.ndim):
            # If the slice has no length then it's a single argument.
            # If it's just an integer then we just return, this is
            # needed for the representation to work properly
            # If it's not then create a list containing None-slices
            # for dim>=1 and continue down the loop
            try:
                len(item)
            except TypeError:
                if isinstance(item, int):
                    return super().__getitem__(item)
                newitem = [slice(None)] * self.ndim
                newitem[0] = item
                item = newitem
            # We're out of items, just append noop slices
            if dim >= len(item):
                all_in_slices.append(slice(0, self.shape[dim]))
                pad.append((0, 0))
            # We're dealing with an integer (no padding even if it's
            # out of bounds)
            if isinstance(item[dim], int):
                all_in_slices.append(slice(item[dim], item[dim] + 1))
                pad.append((0, 0))
            # Dealing with a slice, here it get's complicated, we need
            # to correctly deal with None start/stop as well as with
            # out-of-bound values and correct padding
            elif isinstance(item[dim], slice):
                # Placeholders for values
                start, stop = 0, self.shape[dim]
                this_pad = [0, 0]
                if item[dim].start is None:
                    start = 0
                else:
                    if item[dim].start < 0:
                        this_pad[0] = -item[dim].start
                        start = 0
                    else:
                        start = item[dim].start
                if item[dim].stop is None:
                    stop = self.shape[dim]
                else:
                    if item[dim].stop > self.shape[dim]:
                        this_pad[1] = item[dim].stop - self.shape[dim]
                        stop = self.shape[dim]
                    else:
                        stop = item[dim].stop
                all_in_slices.append(slice(start, stop))
                pad.append(tuple(this_pad))

        # Let numpy deal with slicing
        ret = super().__getitem__(tuple(all_in_slices))
        # and padding
        ret = np.pad(ret, tuple(pad), mode='constant', constant_values=0)

        return ret


class Crop(ImedTransform):
    """Crop data.

    Args:
        size (tuple of int): Size of the output sample. Tuple of size 2 if dealing with 2D samples, 3 with 3D samples.

    Attributes:
        size (tuple of int): Size of the output sample. Tuple of size 3.
    """

    def __init__(self, size):
        self.size = size if len(size) == 3 else size + [0]

    @staticmethod
    def _adjust_padding(npad, sample):
        npad_out_tuple = []
        for idx_dim, tuple_pad in enumerate(npad):
            pad_start, pad_end = tuple_pad
            if pad_start < 0 or pad_end < 0:
                # Move axis of interest
                sample_reorient = np.swapaxes(sample, 0, idx_dim)
                # Adjust pad and crop
                if pad_start < 0 and pad_end < 0:
                    sample_crop = sample_reorient[abs(pad_start):pad_end, ]
                    pad_end, pad_start = 0, 0
                elif pad_start < 0:
                    sample_crop = sample_reorient[abs(pad_start):, ]
                    pad_start = 0
                else:  # i.e. pad_end < 0:
                    sample_crop = sample_reorient[:pad_end, ]
                    pad_end = 0
                # Reorient
                sample = np.swapaxes(sample_crop, 0, idx_dim)

            npad_out_tuple.append((pad_start, pad_end))

        return npad_out_tuple, sample

    @multichannel_capable
    @multichannel_capable  # for multiple raters during training/preprocessing
    def __call__(self, sample, metadata):
        # Get params
        is_2d = sample.shape[-1] == 1
        th, tw, td = self.size
        fh, fw, fd, h, w, d = metadata[MetadataKW.CROP_PARAMS].get(self.__class__.__name__)

        # Crop data
        # Note we use here CroppableArray in order to deal with "out of boundaries" crop
        # e.g. if fh is negative or fh+th out of bounds, then it will pad
        if is_2d:
            data_out = sample.view(CroppableArray)[fh:fh + th, fw:fw + tw, :]
        else:
            data_out = sample.view(CroppableArray)[fh:fh + th, fw:fw + tw, fd:fd + td]

        return data_out, metadata

    @multichannel_capable
    @two_dim_compatible
    def undo_transform(self, sample, metadata=None):
        # Get crop params
        is_2d = sample.shape[-1] == 1
        th, tw, td = self.size
        fh, fw, fd, h, w, d = metadata[MetadataKW.CROP_PARAMS].get(self.__class__.__name__)

        # Compute params to undo transform
        pad_left = fw
        pad_right = w - pad_left - tw
        pad_top = fh
        pad_bottom = h - pad_top - th
        pad_front = fd if not is_2d else 0
        pad_back = d - pad_front - td if not is_2d else 0
        npad = [(pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)]

        # Check and adjust npad if needed, i.e. if crop out of boundaries
        npad_adj, sample_adj = self._adjust_padding(npad, sample.copy())

        # Apply padding
        data_out = np.pad(sample_adj,
                          npad_adj,
                          mode='constant',
                          constant_values=0).astype(sample.dtype)

        return data_out, metadata


class CenterCrop(Crop):
    """Make a centered crop of a specified size."""

    @multichannel_capable
    @multichannel_capable  # for multiple raters during training/preprocessing
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # Crop parameters
        th, tw, td = self.size
        h, w, d = sample.shape
        fh = int(round((h - th) / 2.))
        fw = int(round((w - tw) / 2.))
        fd = int(round((d - td) / 2.))
        params = (fh, fw, fd, h, w, d)
        metadata[MetadataKW.CROP_PARAMS][self.__class__.__name__] = params

        # Call base method
        return super().__call__(sample, metadata)


class ROICrop(Crop):
    """Make a crop of a specified size around a Region of Interest (ROI)."""

    @multichannel_capable
    @multichannel_capable  # for multiple raters during training/preprocessing
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # If crop_params are not in metadata,
        # then we are here dealing with ROI data to determine crop params
        if self.__class__.__name__ not in metadata[MetadataKW.CROP_PARAMS]:
            # Compute center of mass of the ROI
            h_roi, w_roi, d_roi = center_of_mass(sample.astype(int))
            h_roi, w_roi, d_roi = int(round(h_roi)), int(round(w_roi)), int(round(d_roi))
            th, tw, td = self.size
            th_half, tw_half, td_half = int(round(th / 2.)), int(round(tw / 2.)), int(round(td / 2.))

            # compute top left corner of the crop area
            fh = h_roi - th_half
            fw = w_roi - tw_half
            fd = d_roi - td_half

            # Crop params
            h, w, d = sample.shape
            params = (fh, fw, fd, h, w, d)
            metadata[MetadataKW.CROP_PARAMS][self.__class__.__name__] = params

        # Call base method
        return super().__call__(sample, metadata)


class DilateGT(ImedTransform):
    """Randomly dilate a ground-truth tensor.

    .. image:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/technical_features/dilate-gt.png
        :width: 500px
        :align: center

    Args:
        dilation_factor (float): Controls the number of dilation iterations. For each individual lesion, the number of
        dilation iterations is computed as follows:
            nb_it = int(round(dilation_factor * sqrt(lesion_area)))
        If dilation_factor <= 0, then no dilation will be performed.
    """

    def __init__(self, dilation_factor):
        self.dil_factor = dilation_factor

    @staticmethod
    def dilate_lesion(arr_bin, arr_soft, label_values):
        for lb in label_values:
            # binary dilation with 1 iteration
            arr_dilated = binary_dilation(arr_bin, iterations=1)

            # isolate new voxels, i.e. the ones from the dilation
            new_voxels = np.logical_xor(arr_dilated, arr_bin).astype(int)

            # assign a soft value (]0, 1[) to the new voxels
            soft_new_voxels = lb * new_voxels

            # add the new voxels to the input mask
            arr_soft += soft_new_voxels
            arr_bin = (arr_soft > 0).astype(int)

        return arr_bin, arr_soft

    def dilate_arr(self, arr, dil_factor):
        # identify each object
        arr_labeled, lb_nb = label(arr.astype(int))

        # loop across each object
        arr_bin_lst, arr_soft_lst = [], []
        for obj_idx in range(1, lb_nb + 1):
            arr_bin_obj = (arr_labeled == obj_idx).astype(int)
            arr_soft_obj = np.copy(arr_bin_obj).astype(float)
            # compute the number of dilation iterations depending on the size of the lesion
            nb_it = int(round(dil_factor * math.sqrt(arr_bin_obj.sum())))
            # values of the voxels added to the input mask
            soft_label_values = [x / (nb_it + 1) for x in range(nb_it, 0, -1)]
            # dilate lesion
            arr_bin_dil, arr_soft_dil = self.dilate_lesion(arr_bin_obj, arr_soft_obj, soft_label_values)
            arr_bin_lst.append(arr_bin_dil)
            arr_soft_lst.append(arr_soft_dil)

        # sum dilated objects
        arr_bin_idx = np.sum(np.array(arr_bin_lst), axis=0)
        arr_soft_idx = np.sum(np.array(arr_soft_lst), axis=0)
        # clip values in case dilated voxels overlap
        arr_bin_clip, arr_soft_clip = np.clip(arr_bin_idx, 0, 1), np.clip(arr_soft_idx, 0.0, 1.0)

        return arr_soft_clip.astype(float), arr_bin_clip.astype(int)

    @staticmethod
    def random_holes(arr_in, arr_soft, arr_bin):
        arr_soft_out = np.copy(arr_soft)

        # coordinates of the new voxels, i.e. the ones from the dilation
        new_voxels_xx, new_voxels_yy, new_voxels_zz = np.where(np.logical_xor(arr_bin, arr_in))
        nb_new_voxels = new_voxels_xx.shape[0]

        # ratio of voxels added to the input mask from the dilated mask
        new_voxel_ratio = random.random()
        # randomly select new voxel indexes to remove
        idx_to_remove = random.sample(range(nb_new_voxels),
                                      int(round(nb_new_voxels * (1 - new_voxel_ratio))))

        # set to zero the here-above randomly selected new voxels
        arr_soft_out[new_voxels_xx[idx_to_remove],
                     new_voxels_yy[idx_to_remove],
                     new_voxels_zz[idx_to_remove]] = 0.0
        arr_bin_out = (arr_soft_out > 0).astype(int)

        return arr_soft_out, arr_bin_out

    @staticmethod
    def post_processing(arr_in, arr_soft, arr_bin, arr_dil):
        # remove new object that are not connected to the input mask
        arr_labeled, lb_nb = label(arr_bin)

        connected_to_in = arr_labeled * arr_in
        for lb in range(1, lb_nb + 1):
            if np.sum(connected_to_in == lb) == 0:
                arr_soft[arr_labeled == lb] = 0

        struct = np.ones((3, 3, 1) if arr_soft.shape[2] == 1 else (3, 3, 3))
        # binary closing
        arr_bin_closed = binary_closing((arr_soft > 0).astype(int), structure=struct)
        # fill binary holes
        arr_bin_filled = binary_fill_holes(arr_bin_closed)

        # recover the soft-value assigned to the filled-holes
        arr_soft_out = arr_bin_filled * arr_dil

        return arr_soft_out

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # binarize for processing
        gt_data_np = (sample > 0.5).astype(np.int_)

        if self.dil_factor > 0 and np.sum(sample):
            # dilation
            gt_dil, gt_dil_bin = self.dilate_arr(gt_data_np, self.dil_factor)

            # random holes in dilated area
            # gt_holes, gt_holes_bin = self.random_holes(gt_data_np, gt_dil, gt_dil_bin)

            # post-processing
            # gt_pp = self.post_processing(gt_data_np, gt_holes, gt_holes_bin, gt_dil)

            # return gt_pp.astype(np.float32), metadata
            return gt_dil.astype(np.float32), metadata

        else:
            return sample, metadata


class BoundingBoxCrop(Crop):
    """Crops image according to given bounding box."""

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata):
        assert MetadataKW.BOUNDING_BOX in metadata
        x_min, x_max, y_min, y_max, z_min, z_max = metadata[MetadataKW.BOUNDING_BOX]
        x, y, z = sample.shape
        metadata[MetadataKW.CROP_PARAMS][self.__class__.__name__] = (x_min, y_min, z_min, x, y, z)

        # Call base method
        return super().__call__(sample, metadata)


class RandomAffine(ImedTransform):
    """Apply Random Affine transformation.

    Args:
        degrees (float): Positive float or list (or tuple) of length two. Angles in degrees. If only a float is
            provided, then rotation angle is selected within the range [-degrees, degrees]. Otherwise, the list / tuple
            defines this range.
        translate (list of float): List of floats between 0 and 1, of length 2 or 3 depending on the sample shape (2D
            or 3D). These floats defines the maximum range of translation along each axis.
        scale (list of float): List of floats between 0 and 1, of length 2 or 3 depending on the sample shape (2D
            or 3D). These floats defines the maximum range of scaling along each axis.

    Attributes:
        degrees (tuple of floats):
        translate (list of float):
        scale (list of float):
    """

    def __init__(self, degrees=0, translate=None, scale=None):
        # Rotation
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        # Scale
        if scale is not None:
            assert isinstance(scale, (tuple, list)) and (len(scale) == 2 or len(scale) == 3), \
                "scale should be a list or tuple and it must be of length 2 or 3."
            for s in scale:
                if not (0.0 <= s <= 1.0):
                    raise ValueError("scale values should be between 0 and 1")
            if len(scale) == 2:
                scale.append(0.0)
            self.scale = scale
        else:
            self.scale = [0., 0., 0.]

        # Translation
        if translate is not None:
            assert isinstance(translate, (tuple, list)) and (len(translate) == 2 or len(translate) == 3), \
                "translate should be a list or tuple and it must be of length 2 or 3."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
            if len(translate) == 2:
                translate.append(0.0)
        self.translate = translate

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # Rotation
        # If angle and metadata have been already defined for this sample, then use them
        if MetadataKW.ROTATION in metadata:
            angle, axes = metadata[MetadataKW.ROTATION]
        # Otherwise, get random ones
        else:
            # Get the random angle
            angle = math.radians(np.random.uniform(self.degrees[0], self.degrees[1]))
            # Get the two axes that define the plane of rotation
            axes = list(random.sample(range(3 if sample.shape[2] > 1 else 2), 2))
            axes.sort()
            # Save params
            metadata[MetadataKW.ROTATION] = [angle, axes]

        # Scale
        if MetadataKW.SCALE in metadata:
            scale_x, scale_y, scale_z = metadata[MetadataKW.SCALE]
        else:
            scale_x = random.uniform(1 - self.scale[0], 1 + self.scale[0])
            scale_y = random.uniform(1 - self.scale[1], 1 + self.scale[1])
            scale_z = random.uniform(1 - self.scale[2], 1 + self.scale[2])
            metadata[MetadataKW.SCALE] = [scale_x, scale_y, scale_z]

        # Get params
        if MetadataKW.TRANSLATION in metadata:
            translations = metadata[MetadataKW.TRANSLATION]
        else:
            self.data_shape = sample.shape

            if self.translate is not None:
                max_dx = self.translate[0] * self.data_shape[0]
                max_dy = self.translate[1] * self.data_shape[1]
                max_dz = self.translate[2] * self.data_shape[2]
                translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                                np.round(np.random.uniform(-max_dy, max_dy)),
                                np.round(np.random.uniform(-max_dz, max_dz)))
            else:
                translations = (0, 0, 0)

            metadata[MetadataKW.TRANSLATION] = translations

        # Do rotation
        shape = 0.5 * np.array(sample.shape)
        if axes == [0, 1]:
            rotate = np.array([[math.cos(angle), -math.sin(angle), 0],
                               [math.sin(angle), math.cos(angle), 0],
                               [0, 0, 1]])
        elif axes == [0, 2]:
            rotate = np.array([[math.cos(angle), 0, math.sin(angle)],
                               [0, 1, 0],
                               [-math.sin(angle), 0, math.cos(angle)]])
        elif axes == [1, 2]:
            rotate = np.array([[1, 0, 0],
                               [0, math.cos(angle), -math.sin(angle)],
                               [0, math.sin(angle), math.cos(angle)]])
        else:
            raise ValueError("Unknown axes value")

        scale = np.array([[1 / scale_x, 0, 0], [0, 1 / scale_y, 0], [0, 0, 1 / scale_z]])
        if MetadataKW.UNDO in metadata and metadata[MetadataKW.UNDO]:
            transforms = scale.dot(rotate)
        else:
            transforms = rotate.dot(scale)

        offset = shape - shape.dot(transforms) + translations

        data_out = affine_transform(sample, transforms.T, order=1, offset=offset,
                                    output_shape=sample.shape).astype(sample.dtype)

        return data_out, metadata

    @multichannel_capable
    @two_dim_compatible
    def undo_transform(self, sample, metadata=None):
        assert MetadataKW.ROTATION in metadata
        assert MetadataKW.SCALE in metadata
        assert MetadataKW.TRANSLATION in metadata
        # Opposite rotation, same axes
        angle, axes = - metadata[MetadataKW.ROTATION][0], metadata[MetadataKW.ROTATION][1]
        scale = 1 / np.array(metadata[MetadataKW.SCALE])
        translation = - np.array(metadata[MetadataKW.TRANSLATION])

        # Undo rotation
        dict_params = {MetadataKW.ROTATION: [angle, axes], MetadataKW.SCALE: scale,
                       MetadataKW.TRANSLATION: [0, 0, 0], MetadataKW.UNDO: True}

        data_out, _ = self.__call__(sample, dict_params)

        data_out = affine_transform(data_out, np.identity(3), order=1, offset=translation,
                                    output_shape=sample.shape).astype(sample.dtype)

        return data_out, metadata


class RandomReverse(ImedTransform):
    """Make a randomized symmetric inversion of the different values of each dimensions."""

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        if MetadataKW.REVERSE in metadata:
            flip_axes = metadata[MetadataKW.REVERSE]
        else:
            # Flip axis booleans
            flip_axes = [np.random.randint(2) == 1 for _ in [0, 1, 2]]
            # Save in metadata
            metadata[MetadataKW.REVERSE] = flip_axes

        # Run flip
        for idx_axis, flip_bool in enumerate(flip_axes):
            if flip_bool:
                sample = np.flip(sample, axis=idx_axis).copy()

        return sample, metadata

    @multichannel_capable
    @two_dim_compatible
    def undo_transform(self, sample, metadata=None):
        assert MetadataKW.REVERSE in metadata
        return self.__call__(sample, metadata)


class RandomShiftIntensity(ImedTransform):
    """Add a random intensity offset.

    Args:
        shift_range (tuple of floats): Tuple of length two. Specifies the range where the offset that is applied is
            randomly selected from.
        prob (float): Between 0 and 1. Probability of occurence of this transformation.
    """

    def __init__(self, shift_range, prob=0.1):
        self.shift_range = shift_range
        self.prob = prob

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        if np.random.random() < self.prob:
            # Get random offset
            offset = np.random.uniform(self.shift_range[0], self.shift_range[1])
        else:
            offset = 0.0

        # Update metadata
        metadata[MetadataKW.OFFSET] = offset
        # Shift intensity
        data = (sample + offset).astype(sample.dtype)
        return data, metadata

    @multichannel_capable
    def undo_transform(self, sample, metadata=None):
        assert MetadataKW.OFFSET in metadata
        # Get offset
        offset = metadata[MetadataKW.OFFSET]
        # Substract offset
        data = (sample - offset).astype(sample.dtype)
        return data, metadata


class ElasticTransform(ImedTransform):
    """Applies elastic transformation.

    .. seealso::
        Simard, Patrice Y., David Steinkraus, and John C. Platt. "Best practices for convolutional neural networks
        applied to visual document analysis." Icdar. Vol. 3. No. 2003. 2003.

    Args:
        alpha_range (tuple of floats): Deformation coefficient. Length equals 2.
        sigma_range (tuple of floats): Standard deviation. Length equals 2.
    """

    def __init__(self, alpha_range, sigma_range, p=0.1):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.p = p

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        # if params already defined, i.e. sample is GT
        if MetadataKW.ELASTIC in metadata:
            alpha, sigma = metadata[MetadataKW.ELASTIC]

        elif np.random.random() < self.p:
            # Get params
            alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

            # Save params
            metadata[MetadataKW.ELASTIC] = [alpha, sigma]

        else:
            metadata[MetadataKW.ELASTIC] = [None, None]

        if any(metadata[MetadataKW.ELASTIC]):
            # Get shape
            shape = sample.shape

            # Compute random deformation
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                 sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                 sigma, mode="constant", cval=0) * alpha
            dz = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                 sigma, mode="constant", cval=0) * alpha
            if shape[2] == 1:
                dz = 0  # No deformation along the last dimension
            x, y, z = np.meshgrid(np.arange(shape[0]),
                                  np.arange(shape[1]),
                                  np.arange(shape[2]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), \
                np.reshape(y + dy, (-1, 1)), \
                np.reshape(z + dz, (-1, 1))

            # Apply deformation
            data_out = map_coordinates(sample, indices, order=1, mode='reflect')
            # Keep input shape
            data_out = data_out.reshape(shape)
            # Keep data type
            data_out = data_out.astype(sample.dtype)

            return data_out, metadata

        else:
            return sample, metadata


class AdditiveGaussianNoise(ImedTransform):
    """Adds Gaussian Noise to images.

    Args:
        mean (float): Gaussian noise mean.
        std (float): Gaussian noise standard deviation.
    """

    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        if MetadataKW.GAUSSIAN_NOISE in metadata:
            noise = metadata[MetadataKW.GAUSSIAN_NOISE]
        else:
            # Get random noise
            noise = np.random.normal(self.mean, self.std, sample.shape)
            noise = noise.astype(np.float32)

        # Apply noise
        data_out = sample + noise

        return data_out.astype(sample.dtype), metadata


class Clahe(ImedTransform):
    """ Applies Contrast Limited Adaptive Histogram Equalization for enhancing the local image contrast.

    .. seealso::
       Zuiderveld, Karel. "Contrast limited adaptive histogram equalization." Graphics gems (1994): 474-485.

    Default values are based on:

    .. seealso::
       Zheng, Qiao, et al. "3-D consistent and robust segmentation of cardiac images by deep learning with spatial
       propagation." IEEE transactions on medical imaging 37.9 (2018): 2137-2148.

    Args:
        clip_limit (float): Clipping limit, normalized between 0 and 1.
        kernel_size (tuple of int): Defines the shape of contextual regions used in the algorithm. Length equals image
        dimension (ie 2 or 3 for 2D or 3D, respectively).
    """

    def __init__(self, clip_limit=3.0, kernel_size=(8, 8)):
        self.clip_limit = clip_limit
        self.kernel_size = kernel_size

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        assert len(self.kernel_size) == len(sample.shape)
        # Run equalization
        data_out = equalize_adapthist(sample,
                                      kernel_size=self.kernel_size,
                                      clip_limit=self.clip_limit).astype(sample.dtype)

        return data_out, metadata


class HistogramClipping(ImedTransform):
    """Performs intensity clipping based on percentiles.

    Args:
        min_percentile (float): Between 0 and 100. Lower clipping limit.
        max_percentile (float): Between 0 and 100. Higher clipping limit.
    """

    def __init__(self, min_percentile=5.0, max_percentile=95.0):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    @multichannel_capable
    def __call__(self, sample, metadata=None):
        data = np.copy(sample)
        # Run clipping
        percentile1 = np.percentile(sample, self.min_percentile)
        percentile2 = np.percentile(sample, self.max_percentile)
        data[sample <= percentile1] = percentile1
        data[sample >= percentile2] = percentile2
        return data, metadata


class RandomGamma(ImedTransform):
    """Randomly changes the contrast of an image by gamma exponential

    Args:
        log_gamma_range (tuple of floats): Log gamma range for changing contrast. Length equals 2.
        p (float): Probability of performing the gamma contrast
    """

    def __init__(self, log_gamma_range, p=0.5):
        self.log_gamma_range = log_gamma_range
        self.p = p

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        if np.random.random() < self.p:
            # Get params
            gamma = np.exp(np.random.uniform(self.log_gamma_range[0], self.log_gamma_range[1]))

            # Save params
            metadata[MetadataKW.GAMMA] = [gamma]

        else:
            metadata[MetadataKW.GAMMA] = [None]

        if any(metadata[MetadataKW.GAMMA]):
            # Suppress the overflow case (due to exponentiation)
            with np.errstate(over='ignore'):
                # Apply gamma contrast
                data_out = np.sign(sample) * (np.abs(sample) ** gamma)

                # Keep data type
                data_out = data_out.astype(sample.dtype)

                # Clip +/- inf values to the max/min quantization of the native dtype
                data_out = np.nan_to_num(data_out)

            return data_out, metadata

        else:
            return sample, metadata


class RandomBlur(ImedTransform):
    """Applies a random blur to the image

        Args:
            sigma_range (tuple of floats): Standard deviation range for the gaussian filter
            p (float): Probability of performing blur
        """

    def __init__(self, sigma_range, p=0.5):
        self.sigma_range = sigma_range
        self.p = p

    @multichannel_capable
    @two_dim_compatible
    def __call__(self, sample, metadata=None):
        if np.random.random() < self.p:
            # Get params
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

            # Save params
            metadata[MetadataKW.BLUR] = [sigma]

        else:
            metadata[MetadataKW.BLUR] = [None]

        if any(metadata[MetadataKW.BLUR]):
            # Apply random blur
            data_out = gaussian_filter(sample, sigma)

            # Keep data type
            data_out = data_out.astype(sample.dtype)

            return data_out, metadata

        else:
            return sample, metadata


def get_subdatasets_transforms(transform_params):
    """Get transformation parameters for each subdataset: training, validation and testing.

    Args:
        transform_params (dict):

    Returns:
        dict, dict, dict: Training, Validation and Testing transformations.
    """
    transform_params = copy.deepcopy(transform_params)
    train, valid, test = {}, {}, {}
    subdataset_default = ["training", "validation", "testing"]
    # Loop across transformations
    for transform_name in transform_params:
        subdataset_list = ["training", "validation", "testing"]
        # Only consider subdatasets listed in dataset_type
        if "dataset_type" in transform_params[transform_name]:
            subdataset_list = transform_params[transform_name]["dataset_type"]
        # Add current transformation to the relevant subdataset transformation dictionaries
        for subds_name, subds_dict in zip(subdataset_default, [train, valid, test]):
            if subds_name in subdataset_list:
                subds_dict[transform_name] = transform_params[transform_name]
                if "dataset_type" in subds_dict[transform_name]:
                    del subds_dict[transform_name]["dataset_type"]
    return train, valid, test


def get_preprocessing_transforms(transforms):
    """Checks the transformations parameters and selects the transformations which are done during preprocessing only.

    Args:
        transforms (dict): Transformation dictionary.

    Returns:
        dict: Preprocessing transforms.
    """
    original_transforms = copy.deepcopy(transforms)
    preprocessing_transforms = copy.deepcopy(transforms)
    for idx, tr in enumerate(original_transforms):
        if tr == TransformationKW.RESAMPLE or tr == TransformationKW.CENTERCROP or tr == TransformationKW.ROICROP:
            del transforms[tr]
        else:
            del preprocessing_transforms[tr]

    return preprocessing_transforms


def apply_preprocessing_transforms(transforms, seg_pair, roi_pair=None) -> Tuple[dict, dict]:
    """
    Applies preprocessing transforms to segmentation pair (input, gt and metadata).

    Args:
        transforms (Compose): Preprocessing transforms.
        seg_pair (dict): Segmentation pair containing input and gt.
        roi_pair (dict): Segementation pair containing input and roi.

    Returns:
        tuple: Segmentation pair and roi pair.
    """
    if transforms is None:
        return (seg_pair, roi_pair)

    metadata_input = seg_pair['input_metadata']
    if roi_pair is not None:
        stack_roi, metadata_roi = transforms(sample=roi_pair["gt"],
                                             metadata=roi_pair['gt_metadata'],
                                             data_type="roi",
                                             preprocessing=True)
        metadata_input = imed_loader_utils.update_metadata(metadata_roi, metadata_input)
    # Run transforms on images
    stack_input, metadata_input = transforms(sample=seg_pair["input"],
                                             metadata=metadata_input,
                                             data_type="im",
                                             preprocessing=True)
    # Run transforms on images
    metadata_gt = imed_loader_utils.update_metadata(metadata_input, seg_pair['gt_metadata'])
    stack_gt, metadata_gt = transforms(sample=seg_pair["gt"],
                                       metadata=metadata_gt,
                                       data_type="gt",
                                       preprocessing=True)

    seg_pair = {
        'input': stack_input,
        'gt': stack_gt,
        MetadataKW.INPUT_METADATA: metadata_input,
        MetadataKW.GT_METADATA: metadata_gt
    }

    if roi_pair is not None and len(roi_pair['gt']):
        roi_pair = {
            'input': stack_input,
            'gt': stack_roi,
            MetadataKW.INPUT_METADATA: metadata_input,
            MetadataKW.GT_METADATA: metadata_roi
        }
    return (seg_pair, roi_pair)


def prepare_transforms(transform_dict, requires_undo=True):
    """
    This function separates the preprocessing transforms from the others and generates the undo transforms related.

    Args:
        transform_dict (dict): Dictionary containing the transforms and there parameters.
        requires_undo (bool): Boolean indicating if transforms can be undone.

    Returns:
        list, UndoCompose: transform lst containing the preprocessing transforms and regular transforms, UndoCompose
            object containing the transform to undo.
    """
    training_undo_transform = None
    if requires_undo:
        training_undo_transform = UndoCompose(Compose(transform_dict.copy()))
    preprocessing_transforms = get_preprocessing_transforms(transform_dict)
    prepro_transforms = Compose(preprocessing_transforms, requires_undo=requires_undo)
    transforms = Compose(transform_dict, requires_undo=requires_undo)
    tranform_lst = [prepro_transforms if len(preprocessing_transforms) else None, transforms]
    return tranform_lst, training_undo_transform
