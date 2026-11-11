# This file is vendored from ivadomed 2.9.10 (https://github.com/ivadomed/ivadomed),
# originally `ivadomed/postprocessing.py`, licensed under the MIT License.
# See the LICENSE.md file in this directory for the full license text.
#
# Modifications for SCT: none.

# Deals with postprocessing on generated segmentation.

import functools

import nibabel as nib
import numpy as np
from loguru import logger
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes
from skimage.feature import peak_local_max
from pathlib import Path


def nifti_capable(wrapped):
    """Decorator to make a given function compatible with input being Nifti objects.

    Args:
        wrapped: Given function.

    Returns:
        Functions' return.
    """

    @functools.wraps(wrapped)
    def wrapper(data, *args, **kwargs):
        if isinstance(data, nib.Nifti1Image):
            return nib.Nifti1Image(
                dataobj=wrapper(np.copy(np.asanyarray(data.dataobj)), *args, **kwargs),
                affine=data.header.get_best_affine(),
                header=data.header.copy()
            )
        return wrapped(data, *args, **kwargs)

    return wrapper


def binarize_with_low_threshold(wrapped):
    """Decorator to set low values (< 0.001) to 0.

    Args:
        wrapped: Given function.

    Returns:
        Functions' return.
    """

    @functools.wraps(wrapped)
    def wrapper(data, *args, **kwargs):
        if not np.array_equal(data, data.astype(bool)):
            return mask_predictions(data, wrapper(threshold_predictions(data, thr=0.001), *args, **kwargs))
        return wrapped(data, *args, **kwargs)

    return wrapper


def multilabel_capable(wrapped):
    """Decorator to make a given function compatible multilabel images.

    Args:
        wrapped: Given function.

    Returns:
        Functions' return.
    """

    @functools.wraps(wrapped)
    def wrapper(data, *args, **kwargs):
        if len(data.shape) == 4:
            label_list = []
            for i in range(data.shape[-1]):
                out_data = wrapped(data[..., i], *args, **kwargs)
                label_list.append(out_data)
            return np.array(label_list).transpose((1, 2, 3, 0))
        return wrapped(data, *args, **kwargs)

    return wrapper


@nifti_capable
def threshold_predictions(predictions, thr=0.5):
    """Threshold a soft (i.e. not binary) array of predictions given a threshold value, and returns
    a binary array.

    Args:
        predictions (ndarray or nibabel object): Image to binarize.
        thr (float): Threshold value: voxels with a value < to thr are assigned 0 as value, 1
            otherwise.

    Returns:
        ndarray: ndarray or nibabel (same object as the input) containing only zeros or ones. Output type is int.
    """
    thresholded_preds = np.copy(predictions)[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds.astype(int)


@nifti_capable
@binarize_with_low_threshold
def keep_largest_object(predictions):
    """Keep the largest connected object from the input array (2D or 3D).

    Args:
        predictions (ndarray or nibabel object): Input segmentation. Image could be 2D or 3D.

    Returns:
        ndarray or nibabel (same object as the input).
    """
    # Find number of closed objects using skimage "label"
    labeled_obj, num_obj = label(np.copy(predictions))
    # If more than one object is found, keep the largest one
    if num_obj > 1:
        # Keep the largest object
        predictions[np.where(labeled_obj != (np.bincount(labeled_obj.flat)[1:].argmax() + 1))] = 0
    return predictions


@nifti_capable
def keep_largest_object_per_slice(predictions, axis=2):
    """Keep the largest connected object for each 2D slice, along a specified axis.

    Args:
        predictions (ndarray or nibabel object): Input segmentation. Image could be 2D or 3D.
        axis (int): 2D slices are extracted along this axis.

    Returns:
        ndarray or nibabel (same object as the input).
    """
    # Split the 3D input array as a list of slice along axis
    list_preds_in = np.split(predictions, predictions.shape[axis], axis=axis)
    # Init list of processed slices
    list_preds_out = []
    # Loop across the slices along the given axis
    for idx in range(len(list_preds_in)):
        slice_processed = keep_largest_object(np.squeeze(list_preds_in[idx], axis=axis))
        list_preds_out.append(slice_processed)
    return np.stack(list_preds_out, axis=axis)


@nifti_capable
@multilabel_capable
def fill_holes(predictions, structure=(3, 3, 3)):
    """Fill holes in the predictions using a given structuring element.
    Note: This function only works for binary segmentation.

    Args:
        predictions (ndarray or nibabel object): Input binary segmentation. Image could be 2D or 3D.
        structure (tuple of integers): Structuring element, number of ints equals
            number of dimensions in the input array.

    Returns:
        ndrray or nibabel (same object as the input). Output type is int.
    """
    assert np.array_equal(predictions, predictions.astype(bool))
    assert len(structure) == len(predictions.shape)
    return binary_fill_holes(predictions, structure=np.ones(structure)).astype(int)


@nifti_capable
def mask_predictions(predictions, mask_binary):
    """Mask predictions using a binary mask: sets everything outside the mask to zero.

    Args:
        predictions (ndarray or nibabel object): Input binary segmentation. Image could be 2D or 3D.
        mask_binary (ndarray): Numpy array with the same shape as predictions, containing only zeros or ones.

    Returns:
        ndarray or nibabel (same object as the input).
    """
    assert predictions.shape == mask_binary.shape
    assert np.array_equal(mask_binary, mask_binary.astype(bool))
    return predictions * mask_binary


def coordinate_from_heatmap(nifti_image, thresh=0.3):
    """
    Retrieve coordinates of local maxima in a soft segmentation.
    Args:
        nifti_image (nibabel object): nifti image of the soft segmentation.
        thresh (float): Relative threshold for local maxima, i.e., after normalizing
        the min and max between 0 and 1, respectively.

    Returns:
        list: A list of computed coordinates found by local maximum. each element will be a list composed of
        [x, y, z]
    """

    image = np.array(nifti_image.dataobj)
    coordinates_tmp = peak_local_max(image, min_distance=5, threshold_rel=thresh)
    return coordinates_tmp


def label_file_from_coordinates(nifti_image, coord_list):
    """
    Creates a nifti object with single-voxel labels. Each label has a value of 1. The nifti object as the same
    orientation as the input.
    Args:
        nifti_image (nibabel object): Path to the image which affine matrix will be used to generate a new image with
        labels.
        coord_list (list): list of coordinates. Each element is [x, y, z]. Orientation should be the same as the image

    Returns:
        nib_pred: A nifti object containing the singe-voxel label of value 1. The matrix will be the same size as
        `nifti_image`.

    """

    imsh = list(np.array(nifti_image.dataobj).shape)
    # create an empty 3d object.
    label_array = np.zeros(tuple(imsh))

    for j in range(len(coord_list)):
        label_array[coord_list[j][0], coord_list[j][1], coord_list[j][2]] = 1

    nib_pred = nib.Nifti1Image(
        dataobj=label_array,
        affine=nifti_image.header.get_best_affine(),
    )

    return nib_pred


def remove_small_objects(data, bin_structure, size_min):
    """Removes all unconnected objects smaller than the minimum specified size.

    Args:
        data (ndarray): Input data.
        bin_structure (ndarray): Structuring element that defines feature connections.
        size_min (int): Minimal object size to keep in input data.

    Returns:
        ndarray: Array with small objects.
    """

    data_label, n = label(data, structure=bin_structure)

    for idx in range(1, n + 1):
        data_idx = (data_label == idx).astype(int)
        n_nonzero = np.count_nonzero(data_idx)

        if n_nonzero < size_min:
            data[data_label == idx] = 0

    return data


class Postprocessing(object):
    """Postprocessing steps manager

    Args:
        postprocessing_params (dict): Indicates postprocessing steps (in the right order)
        data_pred (ndarray): Prediction from the model.
        dim_lst (list): Dimensions of a voxel in mm.
        filename_prefix (str): Path to prediction file without suffix.

    Attributes:
        postprocessing_params (dict): Indicates postprocessing steps (in the right order)
        data_pred (ndarray): Prediction from the model.
        px (float): Resolution (mm) along the first axis.
        py (float): Resolution (mm) along the second axis.
        pz (float): Resolution (mm) along the third axis.
        filename_prefix (str): Path to prediction file without suffix.
        n_classes (int): Number of classes.
        bin_struct (ndarray): Binary structure.

    """

    def __init__(self, postprocessing_params, data_pred, dim_lst, filename_prefix):
        self.postprocessing_dict = postprocessing_params
        self.data_pred = data_pred
        self.filename_prefix = filename_prefix
        self.px, self.py, self.pz = dim_lst
        h, w, d, self.n_classes = self.data_pred.shape
        self.bin_struct = generate_binary_structure(3, 2)

    def apply(self):
        """Parse postprocessing parameters and apply postprocessing steps to data.
        """
        for postprocessing in self.postprocessing_dict:
            getattr(self, postprocessing)(**self.postprocessing_dict[postprocessing])
        return self.data_pred

    def binarize_prediction(self, thr):
        """Binarize output.
        """
        if thr >= 0:
            self.data_pred = threshold_predictions(self.data_pred, thr)

    def binarize_maxpooling(self):
        """Binarize by setting to 1 the voxel having the max prediction across all classes.
        """
        # Generate background class
        background = np.ones(self.data_pred[..., 0].shape)
        n_class = self.data_pred.shape[-1]
        for c in range(n_class):
            background -= self.data_pred[..., c]

        # Concatenate background class
        pred_with_background = np.concatenate((background[..., None], self.data_pred), axis=-1)

        # Find class with max pred
        class_pred = np.argmax(pred_with_background, axis=-1)
        self.data_pred = np.zeros_like(self.data_pred)
        for c in range(n_class):
            self.data_pred[..., c] = class_pred == c + 1

    def uncertainty(self, thr, suffix):
        """Removes the most uncertain predictions.

        Args:
            thr (float): Uncertainty threshold.
            suffix (str): Suffix of uncertainty filename.

        """
        if thr >= 0:
            uncertainty_path = self.filename_prefix + suffix
            if Path(uncertainty_path).exists():
                data_uncertainty = nib.load(uncertainty_path).get_fdata()
                if suffix == "_unc-iou.nii.gz" or suffix == "_soft.nii.gz":
                    self.data_pred = mask_predictions(self.data_pred, data_uncertainty > thr)
                else:
                    self.data_pred = mask_predictions(self.data_pred, data_uncertainty < thr)
            else:
                raise ValueError('No uncertainty file found.')

    def remove_small(self, unit, thr):
        """Remove small objects

        Args:
            unit (str): Indicates the units of the objects: "mm3" or "vox"
            thr (int or list): Minimal object size to keep in input data.

        """
        if isinstance(thr, list) and (self.n_classes != len(thr)):
            raise ValueError("Length mismatch for remove small object postprocessing step: threshold length of {} "
                             "while the number of predicted class is {}.".format(len(thr), self.n_classes))

        # Convert thr to list
        if isinstance(thr, int):
            thr = [thr] * self.n_classes

        if unit == 'vox':
            size_min = thr
        elif unit == 'mm3':
            size_min = np.round(thr / (self.px * self.py * self.pz))
        else:
            logger.error('Please choose a different unit for removeSmall. Choices: vox or mm3')
            exit()

        for idx in range(self.n_classes):
            self.data_pred[..., idx] = remove_small_objects(data=self.data_pred[..., idx],
                                                            bin_structure=self.bin_struct,
                                                            size_min=size_min[idx])

    def fill_holes(self):
        """Fill holes in the predictions
        """
        # Function fill_holes requires a binary input
        self.data_pred = threshold_predictions(self.data_pred)
        self.data_pred = fill_holes(self.data_pred)

    def keep_largest(self):
        """Keep largest object in prediction
        """
        self.data_pred = keep_largest_object(self.data_pred)

    def remove_noise(self, thr):
        """Remove prediction values under the given threshold

        Args:
            thr (float): Threshold under which predictions are set to 0.

       """
        if thr >= 0:
            mask = self.data_pred > thr
            self.data_pred = mask_predictions(self.data_pred, mask)
