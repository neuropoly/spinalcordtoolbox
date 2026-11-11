# This file is vendored from ivadomed 2.9.10 (https://github.com/ivadomed/ivadomed),
# originally `ivadomed/loader/segmentation_pair.py`, licensed under the MIT License.
# See the LICENSE.md file in this directory for the full license text.
#
# Modifications for SCT:
#   - Fixed flake8 warnings, without changing behavior.
#   - Imports point to `spinalcordtoolbox.ivadomed` instead of `ivadomed`.

from __future__ import annotations
from pathlib import Path
import imageio
import nibabel as nib
import numpy as np

from spinalcordtoolbox.ivadomed.loader import utils as imed_loader_utils
from spinalcordtoolbox.ivadomed.loader.sample_meta_data import SampleMetadata
from spinalcordtoolbox.ivadomed import postprocessing as imed_postpro
from spinalcordtoolbox.ivadomed.keywords import MetadataKW
import typing
if typing.TYPE_CHECKING:
    from typing import List
    import nibabel.nifti1


class SegmentationPair(object):
    """This class is used to build segmentation datasets. It represents
    a pair of of two data volumes (the input data and the ground truth data).

    Args:
        input_filenames (list of str): The input filename list (supported by nibabel). For single channel, the list will
            contain 1 input filename.
        gt_filenames (list of str): The ground-truth filenames list.
        metadata (list): Metadata list with each item corresponding to an image (contrast) in input_filenames.
            For single channel, the list will contain metadata related to one image.
        cache (bool): If the data should be cached in memory or not.
        slice_axis (int): Indicates the axis used to extract 2D slices from 3D NifTI files:
            "axial": 2, "sagittal": 0, "coronal": 1. 2D PNG/TIF/JPG files use default "axial": 2.
        prepro_transforms (dict): Output of get_preprocessing_transforms.

    Attributes:
        input_filenames (list): List of input filenames.
        gt_filenames (list): List of ground truth filenames.
        metadata (dict): Dictionary containing metadata of input and gt.
        cache (bool): If the data should be cached in memory or not.
        slice_axis (int): Indicates the axis used to extract 2D slices from 3D NifTI files:
            "axial": 2, "sagittal": 0, "coronal": 1. 2D PNG/TIF/JPG files use default "axial": 2.
        prepro_transforms (dict): Transforms to be applied before training.
        input_handle (list): List of input NifTI data as 'nibabel.nifti1.Nifti1Image' object
        gt_handle (list): List of gt (ground truth) NifTI data as 'nibabel.nifti1.Nifti1Image' object
    """

    def __init__(self, input_filenames: List[str], gt_filenames: List[str], metadata: list = None, slice_axis: int = 2,
                 cache: bool = True, prepro_transforms: dict = None, soft_gt: bool = False) -> None:

        self.input_filenames = input_filenames
        self.gt_filenames = gt_filenames
        self.metadata = metadata
        self.cache = cache
        self.slice_axis = slice_axis
        self.soft_gt = soft_gt
        self.prepro_transforms = prepro_transforms
        # list of the images
        self.input_handle = []

        # loop over the filenames (list)
        for input_file in self.input_filenames:
            input_img = self.read_file(input_file)
            self.input_handle.append(input_img)
            if len(input_img.shape) > 3:
                raise RuntimeError("4-dimensional volumes not supported.")

        # list of GT for multiclass segmentation
        self.gt_handle = []

        # Labeled data (ie not inference time)
        if self.gt_filenames is not None:
            if not isinstance(self.gt_filenames, list):
                self.gt_filenames = [self.gt_filenames]
            for gt in self.gt_filenames:
                if gt is not None:
                    if isinstance(gt, str):  # this tissue has annotation from only one rater
                        self.gt_handle.append(self.read_file(gt, is_gt=True))
                    else:  # this tissue has annotation from several raters
                        self.gt_handle.append([self.read_file(gt_rater, is_gt=True) for gt_rater in gt])
                else:
                    self.gt_handle.append(None)

        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()

        if self.gt_filenames is not None and self.gt_filenames[0] is not None:
            if not np.allclose(input_shape, gt_shape):
                raise RuntimeError('Input and ground truth with different dimensions.')

        for idx, handle in enumerate(self.input_handle):
            self.input_handle[idx] = nib.as_closest_canonical(handle)

        # Labeled data (ie not inference time)
        if self.gt_filenames is not None:
            for idx, gt in enumerate(self.gt_handle):
                if gt is not None:
                    if not isinstance(gt, list):  # this tissue has annotation from only one rater
                        self.gt_handle[idx] = nib.as_closest_canonical(gt)
                    else:  # this tissue has annotation from several raters
                        self.gt_handle[idx] = [nib.as_closest_canonical(gt_rater) for gt_rater in gt]

        # If binary classification, then extract labels from GT mask

        if self.metadata:
            self.metadata = []
            for data, input_filename in zip(metadata, input_filenames):
                data[MetadataKW.INPUT_FILENAMES] = input_filename
                data[MetadataKW.GT_FILENAMES] = gt_filenames
                self.metadata.append(data)

    def get_pair_shapes(self) -> (tuple, tuple):
        """Return the tuple (input, ground truth) representing both the input and ground truth shapes."""
        input_shape = []
        for handle in self.input_handle:
            shape = imed_loader_utils.orient_shapes_hwd(handle.header.get_data_shape(), self.slice_axis)
            input_shape.append(tuple(shape))

            if not len(set(input_shape)):
                raise RuntimeError('Inputs have different dimensions.')

        gt_shape = []

        for gt in self.gt_handle:
            if gt is not None:
                if not isinstance(gt, list):  # this tissue has annotation from only one rater
                    gt = [gt]
                for gt_rater in gt:
                    shape = imed_loader_utils.orient_shapes_hwd(gt_rater.header.get_data_shape(), self.slice_axis)
                    gt_shape.append(tuple(shape))

                if not len(set(gt_shape)):
                    raise RuntimeError('Labels have different dimensions.')

        return input_shape[0], gt_shape[0] if len(gt_shape) else None

    def get_pair_data(self) -> (list, list):
        """Return the tuple (input, ground truth) with the data content in numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'

        input_data = []
        for handle in self.input_handle:
            hwd_oriented = imed_loader_utils.orient_img_hwd(handle.get_fdata(cache_mode, dtype=np.float32), self.slice_axis)
            input_data.append(hwd_oriented)

        gt_data = []
        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        for gt in self.gt_handle:
            if gt is not None:
                if not isinstance(gt, list):  # this tissue has annotation from only one rater
                    hwd_oriented = imed_loader_utils.orient_img_hwd(gt.get_fdata(cache_mode, dtype=np.float32), self.slice_axis)
                    gt_data.append(hwd_oriented)
                else:  # this tissue has annotation from several raters
                    hwd_oriented_list = [
                        imed_loader_utils.orient_img_hwd(gt_rater.get_fdata(cache_mode, dtype=np.float32),
                                                         self.slice_axis) for gt_rater in gt]
                    gt_data.append([hwd_oriented for hwd_oriented in hwd_oriented_list])
            else:
                gt_data.append(
                    np.zeros(imed_loader_utils.orient_shapes_hwd(self.input_handle[0].shape, self.slice_axis),
                             dtype=np.float32).astype(np.uint8))

        return input_data, gt_data

    def get_pair_metadata(self, slice_index: int = 0, coord: tuple | list = None) -> dict:
        """Return dictionary containing input and gt metadata.

        Args:
            slice_index (int): Index of 2D slice if 2D model is used, else 0.
            coord (tuple or list): Coordinates of subvolume in volume if 3D model is used, else None.

        Returns:
            dict: Input and gt metadata.
        """
        gt_meta_dict = []
        for idx_class, gt in enumerate(self.gt_handle):
            if gt is not None:
                if not isinstance(gt, list):  # this tissue has annotation from only one rater
                    gt_meta_dict.append(SampleMetadata({
                        MetadataKW.ZOOMS: imed_loader_utils.orient_shapes_hwd(gt.header.get_zooms(), self.slice_axis),
                        MetadataKW.DATA_SHAPE: imed_loader_utils.orient_shapes_hwd(gt.header.get_data_shape(), self.slice_axis),
                        MetadataKW.GT_FILENAMES: self.metadata[0].get(MetadataKW.GT_FILENAMES),
                        MetadataKW.BOUNDING_BOX: self.metadata[0].get(MetadataKW.BOUNDING_BOX),
                        MetadataKW.DATA_TYPE: 'gt',
                        MetadataKW.CROP_PARAMS: {}
                    }))
                else:
                    gt_meta_dict.append([SampleMetadata({
                        MetadataKW.ZOOMS: imed_loader_utils.orient_shapes_hwd(gt_rater.header.get_zooms(), self.slice_axis),
                        MetadataKW.DATA_SHAPE: imed_loader_utils.orient_shapes_hwd(gt_rater.header.get_data_shape(), self.slice_axis),
                        MetadataKW.GT_FILENAMES: self.metadata[0].get(MetadataKW.GT_FILENAMES)[idx_class][idx_rater],
                        MetadataKW.BOUNDING_BOX: self.metadata[0].get(MetadataKW.BOUNDING_BOX),
                        MetadataKW.DATA_TYPE: 'gt',
                        MetadataKW.CROP_PARAMS: {}
                    }) for idx_rater, gt_rater in enumerate(gt)])

            else:
                # Temporarily append null metadata to null gt
                gt_meta_dict.append(None)

        # Replace null metadata with metadata from other existing classes of the same subject
        for idx, gt_metadata in enumerate(gt_meta_dict):
            if gt_metadata is None:
                gt_meta_dict[idx] = list(filter(None, gt_meta_dict))[0]

        input_meta_dict = []
        for handle in self.input_handle:
            input_meta_dict.append(SampleMetadata({
                MetadataKW.ZOOMS: imed_loader_utils.orient_shapes_hwd(handle.header.get_zooms(), self.slice_axis),
                MetadataKW.DATA_SHAPE: imed_loader_utils.orient_shapes_hwd(handle.header.get_data_shape(), self.slice_axis),
                MetadataKW.DATA_TYPE: 'im',
                MetadataKW.CROP_PARAMS: {}
            }))

        dreturn = {
            MetadataKW.INPUT_METADATA: input_meta_dict,
            MetadataKW.GT_METADATA: gt_meta_dict,
        }

        for idx, metadata in enumerate(self.metadata):  # loop across channels
            metadata[MetadataKW.SLICE_INDEX] = slice_index
            metadata[MetadataKW.COORD] = coord
            self.metadata[idx] = metadata
            for metadata_key in metadata.keys():  # loop across input metadata
                dreturn[MetadataKW.INPUT_METADATA][idx][metadata_key] = metadata[metadata_key]

        return dreturn

    def get_pair_slice(self, slice_index: int, gt_type: str = "segmentation") -> dict:
        """Return the specified slice from (input, ground truth).

        Args:
            slice_index (int): Slice number.
            gt_type (str): Choice between segmentation or classification, returns mask (array) or label (int) resp.
                for the ground truth.
        """

        metadata = self.get_pair_metadata(slice_index)
        input_dataobj, gt_dataobj = self.get_pair_data()

        if self.slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        input_slices = []
        # Loop over contrasts
        for data_object in input_dataobj:
            input_slices.append(np.asarray(data_object[..., slice_index],
                                           dtype=np.float32))

        # Handle the case for unlabeled data
        if self.gt_handle is None:
            gt_slices = None
        else:
            gt_slices = []
            for gt_obj in gt_dataobj:
                if gt_type == "segmentation":
                    if not isinstance(gt_obj, list):  # annotation from only one rater
                        gt_slices.append(np.asarray(gt_obj[..., slice_index],
                                                    dtype=np.float32))
                    else:  # annotations from several raters
                        gt_slices.append([np.asarray(gt_obj_rater[..., slice_index],
                                                     dtype=np.float32) for gt_obj_rater in gt_obj])
                else:
                    if not isinstance(gt_obj, list):  # annotation from only one rater
                        gt_slices.append(np.asarray(int(np.any(gt_obj[..., slice_index]))))
                    else:  # annotations from several raters
                        gt_slices.append([np.asarray(int(np.any(gt_obj_rater[..., slice_index])))
                                          for gt_obj_rater in gt_obj])
        dreturn = {
            "input": input_slices,
            "gt": gt_slices,
            MetadataKW.INPUT_METADATA: metadata.get(MetadataKW.INPUT_METADATA),
            MetadataKW.GT_METADATA: metadata.get(MetadataKW.GT_METADATA),
        }

        return dreturn

    def read_file(self, filename: str, is_gt: bool = False) -> nibabel.nifti1.Nifti1Image:
        """Read file according to file extension and returns 'nibabel.nifti1.Nifti1Image' object.

        Args:
            filename (str): Subject filename.
            is_gt (bool): Indicate if the file is a ground-truth.

        Returns:
            'nibabel.nifti1.Nifti1Image' object
        """
        extension = imed_loader_utils.get_file_extension(filename)
        # TODO: remove "ome" from condition when implementing OMETIFF support (#739)
        if (not extension) or ("ome" in extension):
            raise RuntimeError(f"The input file extension '{extension}' of '{Path(filename).name}' is not "
                               f"supported. ivadomed supports the following "
                               f"file extensions: '.nii', '.nii.gz', '.png', '.tif', '.tiff', '.jpg' and '.jpeg'.")

        if "nii" in extension:
            # For '.nii' and '.nii.gz' extentions
            img = nib.load(filename)
        else:
            img = self.convert_file_to_nifti(filename, extension, is_gt)
        return img

    def convert_file_to_nifti(self, filename: str, extension: str, is_gt: bool = False) -> nibabel.nifti1.Nifti1Image:
        """
        Convert a non-NifTI image into a 'nibabel.nifti1.Nifti1Image' object and save to a file.
        This method is especially relevant for making microscopy data compatible with NifTI-only
        pipelines.

        TODO: (#739) implement OMETIFF behavior (if "ome" in extension)

        Args:
            filename (str): Subject filename.
            extension (str): File extension.
            is_gt (bool): Indicate if the file is a ground-truth.

        Returns:
            'nibabel.nifti1.Nifti1Image' object
        """
        # For '.png', '.tif', '.tiff', '.jpg' and 'jpeg' extentions
        # Read image as 8 bit grayscale in numpy array (behavior TBD in ivadomed for RGB, RBGA or higher bit depth)
        try:
            imageio.v3.improps(filename)  # new in v3 - standardized metadata
            _img = imageio.v3.imread(filename)
        except Exception:
            # brute force fall back to support backward compatibility
            if '.tif' in extension:
                img = np.expand_dims(
                    imageio.v3.imread(filename, plugin='TIFF-PIL'),
                    axis=-1).astype(np.unint8)

                if len(img.shape) > 3:
                    img = np.expand_dims(
                        imageio.v3.imread(filename, plugin='TIFF-PIL', as_gray=True),
                        axis=-1).astype(np.unint8)
            else:
                # NOTE: The following yield same result but the first one is preferred
                #       to support default backward compatibility:
                # 1. _img = imageio.v3.imread(filename, plugin='PNG-PIL', as_gray=True)
                # 2. _img = imageio.v3.imread(filename, plugin='PNG-PIL', pilmode='F')
                # 3. _img = imageio.v3.imread(filename, plugin='pillow', pilmode='F')

                img = np.expand_dims(
                    imageio.v3.imread(filename, plugin='PNG-PIL', as_gray=True),
                    axis=-1).astype(np.unint8)
        else:
            # c.f for more details: https://github.com/ivadomed/ivadomed/pull/1297#discussion_r1267563980 and
            # https://github.com/ivadomed/ivadomed/pull/1297#discussion_r1267563980

            # TIFF is a "wild" format. A few assumptions greatly simplify the code below:
            # 1. the image is interleaved/channel-last (not planar)
            # 2. the colorspace is one of: binary, gray, RGB, RGBA (not aliasing ones like YUV or CMYK)
            # 3. the image is flat (not a volume or time-series)
            # Note: All of these are trivially true for JPEG and PNG due to limitations of these formats.

            # make grayscale (treats binary as 1-bit grayscale)
            colorspace_idx = 2
            if _img.ndim <= colorspace_idx:  # binary or gray
                pass  # nothing to do
            elif _img.shape[colorspace_idx] == 2:  # gray with alpha channel
                _img = _img[:, :, 0]
            elif _img.shape[colorspace_idx] == 3:  # RGB
                _img = np.sum(_img * (.299, .587, .114), axis=-1)
            else:  # RGBA
                # discards alpha
                _img = np.sum(_img * (.299, .587, .114, 0), axis=-1)
            if len(_img.shape) < 3:
                _img = np.expand_dims(_img, axis=-1)

            img = imageio.core.image_as_uint(_img, bitdepth=8)

        # Binarize ground-truth values (0-255) to 0 and 1 in uint8 with threshold 0.5
        if is_gt:
            img = imed_postpro.threshold_predictions(img / 255, thr=0.5).astype(np.uint8)

        # Convert numpy array to Nifti1Image object with 4x4 identity affine matrix
        img = nib.Nifti1Image(img, affine=np.eye(4))

        # Get PixelSize in millimeters in order (PixelSizeY, PixelSizeX, PixelSizeZ), where X is the width,
        # Y the height and Z the depth of the image.
        ps_in_mm = self.get_microscopy_pixelsize(filename)

        # Set "pixdim" (zooms) in Nifti1Image object header
        img.header.set_zooms(ps_in_mm)

        # If it doesn't already exist, save NifTI file in path_data alongside PNG/TIF/JPG file
        fname_out = imed_loader_utils.update_filename_to_nifti(filename)
        if not Path(fname_out).exists():
            nib.save(img, fname_out)

        return img

    def get_microscopy_pixelsize(self, filename: str) -> tuple:
        """
        Get the microscopy pixel size from the metadata and convert to millimeters.

        The implementation of this method is compliant with BIDS version 1.7.0:
        * "pixdim" (zooms) for Nifti1Image object is extracted as follows:
            * For train, test and segment commands, PixelSize is taken from the metadata in BIDS JSON sidecar file.
            * For inference with the segment_volume function, PixelSize and PixelSizeUnits must be provided in the
              'options' argument.
        * The function supports the PixelSize definition of BIDS 1.7.0 as a list of 2-numbers
          [PixelSizeX, PixelSizeY] or 3-numbers [PixelSizeX, PixelSizeY, PixelSizeZ] for 2D and 3D
          respectively, where X is the width, Y the height and Z the depth of the image.
        * The function supports the PixelSizeUnits definition of BIDS 1.7.0 as "mm", "um" or "nm".

        Returns:
            ndrray: Pixel size in millimeters (ps_in_mm) in order (PixelSizeY, PixelSizeX, PixelSizeZ),
            where Y is the height, X the width and Z the depth of the image.
        """

        # Get pixel size units from json metadata and set conversion factor from pixel size units to mm
        if MetadataKW.PIXEL_SIZE_UNITS in self.metadata[0]:
            pixel_size_units = self.metadata[0][MetadataKW.PIXEL_SIZE_UNITS]
            if pixel_size_units == "mm":
                conversion_factor = 1          # Conversion factor from mm to mm
            elif pixel_size_units == "um":
                conversion_factor = 0.001      # Conversion factor from um to mm
            elif pixel_size_units == "nm":
                conversion_factor = 0.000001   # Conversion factor from nm to mm
            else:
                raise RuntimeError(f"The PixelSizeUnits '{pixel_size_units}' of '{Path(filename).stem}' is not "
                                   f"supported. ivadomed supports the following PixelSizeUnits: 'mm', 'um' and 'nm'.")
        else:
            raise RuntimeError("'PixelSizeUnits' is missing from metadata")

        # Set accepted array length for 'PixelSize' metadata
        array_length = [2, 3]

        # Get pixel size from json metadata and convert to mm
        if MetadataKW.PIXEL_SIZE in self.metadata[0]:
            pixel_size = self.metadata[0][MetadataKW.PIXEL_SIZE]

            if len(pixel_size) in array_length:
                # PixelSize array in order [PixelSizeX, PixelSizeY] or [PixelSizeX, PixelSizeY, PixelSizeZ]
                pixel_size = np.asarray(pixel_size)

                # Note: pixdim[3] (PixelSizeZ) must be non-zero in Nifti objects even if there is only one slice.
                # When PixelSizeZ is not present or 0, we assign the same PixelSize as PixelSizeX
                pixel_size = np.resize(pixel_size, 3)
                if pixel_size[2] == 0:
                    pixel_size[2] = pixel_size[0]

                # Swap PixelSizeX and PixelSizeY resulting in an array in order [PixelSizeY, PixelSizeX, PixelSizeZ]
                # to match NIfTI pixdim[1,2,3] in [Height, Width, Depth] orientation with axial slice axis.
                pixel_size[[1, 0]] = pixel_size[[0, 1]]

            else:
                raise RuntimeError("'PixelSize' metadata type is not supported. Format must be a 2D"
                                   " [PixelSizeX, PixelSizeY] array or 3D [PixelSizeX, PixelSizeY, PixelSizeZ] array"
                                   " where X is the width, Y the height and Z the depth of the image.")
            ps_in_mm = tuple(pixel_size * conversion_factor)
        else:
            raise RuntimeError("'PixelSize' is missing from metadata")

        return ps_in_mm
