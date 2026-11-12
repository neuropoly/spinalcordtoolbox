"""
API for functions used in the sct_deepseg_gm CLI script

Reference paper:
    Perone, C. S., Calabrese, E., & Cohen-Adad, J. (2017).
    Spinal cord gray matter segmentation using deep dilated convolutions.
    URL: https://doi.org/10.48550/arXiv.1710.01269

Copyright (c) 2022 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import os


import numpy as np

from spinalcordtoolbox import resampling
from spinalcordtoolbox.utils.sys import __data_dir__
from spinalcordtoolbox.deepseg_.onnx import onnx_inference
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import LazyLoader

nib = LazyLoader("nib", globals(), "nibabel")

# Models
# Tuple of (model, metadata)
MODELS = {
    'challenge': ('challenge_model.onnx', 'challenge_model.json'),
    'large': ('large_model.onnx', 'large_model.json'),
}

INPUT_SIZE = 200
BATCH_SIZE = 4


class DataResource(object):
    """This class is responsible for resource file
    management (such as loding models)."""

    def __init__(self, dirname):
        """Initialize the resource with the directory
        name context.

        :param dirname: the root directory name.
        """
        self.data_root = os.path.join(__data_dir__, dirname)

    def get_file_path(self, filename):
        """Get the absolute file path based on the
        data root directory.

        :param filename: the filename.
        """
        return os.path.join(self.data_root, filename)


class ShapeTransform(object):
    def __init__(self, original_shape, transformed_shape):
        self.original_shape = original_shape
        self.transformed_shape = transformed_shape
        self.pad_mode = "symmetric"

    def apply(self, image):
        """Transform the input image into the desired dimensions."""
        self.pad_mode = "symmetric"  # Symmetry-pad if padding a smaller image to 200x200
        return self.transform(image, old_dim=self.original_shape, new_dim=self.transformed_shape)

    def undo(self, image):
        """Return the transformed image back into its original dimensions."""
        self.pad_mode = "constant"  # Zero-pad if un-cropping an image to its original, larger dimensions
        return self.transform(image, old_dim=self.transformed_shape, new_dim=self.original_shape)

    def transform(self, image, old_dim, new_dim):
        """Conditionally crop or pad the height and width of the image to match the requested dimensions."""
        if image.shape != old_dim:
            raise ValueError(f"Input shape ({image.shape}) does not match expected shape for transform ({old_dim}).")

        height, height_old = new_dim[0], old_dim[0]
        if height - height_old > 0:
            image = self.pad(image, height_old, height, axis=0)
        else:
            image = self.crop(image, height_old, height, axis=0)

        width, width_old = new_dim[1], old_dim[1]
        if width - width_old > 0:
            image = self.pad(image, width_old, width, axis=1)
        else:
            image = self.crop(image, width_old, width, axis=1)

        return image

    def pad(self, image, old_dim, new_dim, axis=0):
        """Add zeros to a single image axis to match a requested dimension."""
        if not new_dim > old_dim:
            raise ValueError(f"Can't pad image. New dimension ({new_dim}) must be > starting dimension ({old_dim}).")
        pad_1 = new_dim // 2 - (old_dim // 2)
        pad_2 = new_dim - (pad_1 + old_dim)
        if axis == 0:
            return np.pad(image, ((pad_1, pad_2), (0, 0)), mode=self.pad_mode)
        else:
            return np.pad(image, ((0, 0), (pad_1, pad_2)), mode=self.pad_mode)

    def crop(self, image, old_dim, new_dim, axis=0):
        """Center crop a single image axis to match the requested dimension."""
        if not new_dim <= old_dim:
            raise ValueError(f"Can't crop image. New dimension ({new_dim}) must be <= starting dimension ({old_dim}).")
        start = old_dim // 2 - (new_dim // 2)
        if axis == 0:
            return image[start:start+new_dim, :]
        else:
            return image[:, start:start+new_dim]


class StandardizationTransform(object):
    """This transformation will standardize the volume
    according to the specified mean/std.dev.
    """

    def __init__(self, mean, std):
        """Constructor for the normalization transformation.

        :param mean: the mean parameter
        :param std: the standar deviation parameter
        """
        self.mean = mean
        self.std = std

    def __call__(self, volume):
        """This method will enable the function call for the
        class object.

        :param volume: the volume to be normalized.
        """
        volume -= self.mean
        volume /= self.std
        return volume


class VolumeStandardizationTransform(object):
    """This transformation will standardize the volume with
    the parameters estimated from the volume itself.
    """

    def __call__(self, volume):
        """This method will enable the function call for the
        class object.

        :param volume: the volume to be normalized.
        """
        volume_mean = volume.mean()
        volume_std = volume.std()

        volume -= volume_mean
        volume /= volume_std

        return volume


def threshold_predictions(predictions, thr=0.999):
    """This method will threshold predictions.

    :param thr: the threshold (if None, no threshold will
                be applied).
    :return: thresholded predictions
    """
    if thr is None:
        return predictions[:]

    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds <= thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds > thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


def segment_volume(ninput_volume, model_name,
                   threshold=0.999, use_tta=False):
    """Segment a nifti volume.

    :param ninput_volume: the input volume.
    :param model_name: the name of the model to use.
    :param threshold: threshold to be applied in predictions.
    :param use_tta: whether TTA (test-time augmentation)
                    should be used or not.
    :return: segmented slices.
    """
    gmseg_model_challenge = DataResource('deepseg_gm_models')
    model_path, metadata_path = MODELS[model_name]
    model_abs_path = gmseg_model_challenge.get_file_path(model_path)

    # Padding/cropping data to match the input dimensions of the model
    volume_data = np.asanyarray(ninput_volume.dataobj)
    axial_slices = []
    transforms = []
    for slice_num in range(volume_data.shape[2]):
        data = volume_data[..., slice_num]
        transform = ShapeTransform(original_shape=data.shape,
                                   transformed_shape=(INPUT_SIZE, INPUT_SIZE))
        axial_slices.append(transform.apply(data))
        transforms.append(transform)

    axial_slices = np.asarray(axial_slices, dtype=np.float32)
    axial_slices = np.expand_dims(axial_slices, axis=3)

    normalization = VolumeStandardizationTransform()
    axial_slices = normalization(axial_slices)

    if use_tta:
        pred_sampled = []
        for i in range(8):
            sampled_value = np.random.uniform(high=2.0)
            sampled_axial_slices = axial_slices + sampled_value
            preds = onnx_inference(model_abs_path, sampled_axial_slices)[0]
            pred_sampled.append(preds)

        preds = onnx_inference(model_abs_path, axial_slices)[0]
        pred_sampled.append(preds)
        pred_sampled = np.asarray(pred_sampled)
        pred_sampled = np.mean(pred_sampled, axis=0)
        preds = threshold_predictions(pred_sampled, threshold)
    else:
        preds = onnx_inference(model_abs_path, axial_slices)[0]
        preds = threshold_predictions(preds, threshold)

    pred_slices = []

    # Reversing the cropping/passing to preserve original dimensions
    for slice_num in range(preds.shape[0]):
        pred_slice = preds[slice_num][..., 0]
        pred_slice = transforms[slice_num].undo(pred_slice)
        pred_slices.append(pred_slice)

    pred_slices = np.asarray(pred_slices, dtype=np.uint8)
    pred_slices = np.transpose(pred_slices, (1, 2, 0))

    return pred_slices


def segment_file(input_filename, output_filename,
                 model_name, threshold, verbosity,
                 use_tta):
    """Segment a volume file.

    :param input_filename: the input filename.
    :param output_filename: the output filename.
    :param model_name: the name of model to use.
    :param threshold: threshold to apply in predictions (if None,
                      no threshold will be applied)
    :param verbosity: the verbosity level.
    :param use_tta: whether it should use TTA (test-time augmentation)
                    or not.
    :return: the output filename.
    """
    nii_original = nib.load(input_filename)
    pixdim = nii_original.header["pixdim"][3]
    target_resample = [0.25, 0.25, pixdim]

    nii_resampled = resampling.resample_nib(
        nii_original, new_size=target_resample, new_size_type='mm', interpolation='linear')
    pred_slices = segment_volume(nii_resampled, model_name, threshold,
                                 use_tta)

    original_res = [
        nii_original.header["pixdim"][1],
        nii_original.header["pixdim"][2],
        nii_original.header["pixdim"][3]]

    volume_affine = nii_resampled.affine
    volume_header = nii_resampled.header
    nii_segmentation = nib.Nifti1Image(pred_slices, volume_affine,
                                       volume_header)
    nii_resampled_original = resampling.resample_nib(
        nii_segmentation, new_size=original_res, new_size_type='mm', interpolation='linear')
    res_data = np.asanyarray(nii_resampled_original.dataobj)

    # Threshold after resampling, only if specified
    if threshold is not None:
        res_data = threshold_predictions(res_data, 0.5)

    img = Image(param=res_data, hdr=nii_resampled_original.header)
    img.save(output_filename)
    return output_filename
