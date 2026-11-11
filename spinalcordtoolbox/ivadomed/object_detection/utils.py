# This file is vendored from ivadomed 2.9.10 (https://github.com/ivadomed/ivadomed),
# originally `ivadomed/object_detection/utils.py`, licensed under the MIT License.
# See the LICENSE.md file in this directory for the full license text.
#
# Modifications for SCT:
#   - Imports point to `spinalcordtoolbox.ivadomed` instead of `ivadomed`.

import json
import statistics

import nibabel as nib
import numpy as np
from loguru import logger
from scipy import ndimage
from pathlib import Path

from spinalcordtoolbox.ivadomed import postprocessing as imed_postpro
from spinalcordtoolbox.ivadomed import transforms as imed_transforms
from spinalcordtoolbox.ivadomed.loader import utils as imed_loader_utils
from spinalcordtoolbox.ivadomed.keywords import ObjectDetectionParamsKW, MetadataKW


def get_bounding_boxes(mask):
    """Generates a 3D bounding box around a given mask.
    Args:
        mask (Numpy array): Mask of the ROI.

    Returns:
        list: Bounding box coordinate (x_min, x_max, y_min, y_max, z_min, z_max).

    """

    # Label the different objects in the mask
    labeled_mask, _ = ndimage.label(mask)
    object_labels = np.unique(labeled_mask)
    bounding_boxes = []
    for label in object_labels[1:]:
        single_object = labeled_mask == label
        coords = np.where(single_object)
        dimensions = []
        for i in range(len(coords)):
            dimensions.append(int(coords[i].min()))
            dimensions.append(int(coords[i].max()))
        bounding_boxes.append(dimensions)

    return bounding_boxes


def adjust_bb_size(bounding_box, factor, resample=False):
    """Modifies the bounding box dimensions according to a given factor.

    Args:
        bounding_box (list or tuple): Coordinates of bounding box (x_min, x_max, y_min, y_max, z_min, z_max).
        factor (list or tuple): Multiplicative factor for each dimension (list or tuple of length 3).
        resample (bool):  Boolean indicating if this resize is for resampling.

    Returns:
        list: New coordinates (x_min, x_max, y_min, y_max, z_min, z_max).
    """
    coord = []
    for i in range(len(bounding_box) // 2):
        d_min, d_max = bounding_box[2 * i: (2 * i) + 2]
        if resample:
            d_min, d_max = d_min * factor[i], d_max * factor[i]
            dim_len = d_max - d_min
        else:
            dim_len = (d_max - d_min) * factor[i]

        # new min and max coordinates
        min_coord = d_min - (dim_len - (d_max - d_min)) // 2
        coord.append(int(round(max(min_coord, 0))))
        coord.append(int(coord[-1] + dim_len))

    return coord


def resize_to_multiple(shape, multiple, length):
    """Modify a given shape so each dimension is a multiple of a given number. This is used to avoid dimension mismatch
    with patch training. The return shape is always larger then the initial shape (no cropping).

    Args:
        shape (tuple or list): Initial shape to be modified.
        multiple (tuple or list): Multiple for each dimension.
        length (tuple or list): Patch length.

    Returns:
        list: New dimensions.
    """
    new_dim = []
    for dim_len, m, l in zip(shape, multiple, length):
        padding = (m - (dim_len - l) % m) if (m - (dim_len - l) % m) != m else 0
        new_dim.append(dim_len + padding)
    return new_dim


def generate_bounding_box_file(subject_path_list, model_path, path_output, gpu_id=0, slice_axis=0, contrast_lst=None,
                               keep_largest_only=True, safety_factor=None):
    """Creates json file containing the bounding box dimension for each images. The file has the following format:
    {"path/to/img.nii.gz": [[x1_min, x1_max, y1_min, y1_max, z1_min, z1_max],
    [x2_min, x2_max, y2_min, y2_max, z2_min, z2_max]]}
    where each list represents the coordinates of an object on the image (2 instance of a given object in this example).

    Args:
        subject_path_list (list): List of all subjects in all the BIDS directories.
        model_path (string): Path to object detection model.
        path_output (string): Output path.
        gpu_id (int): If available, GPU number.
        slice_axis (int): Slice axis (0: sagittal, 1: coronal, 2: axial).
        contrast_lst (list): Contrasts.
        keep_largest_only (bool): Boolean representing if only the largest object of the prediction is kept.
        safety_factor (list or tuple): Factors to multiply each dimension of the bounding box.

    Returns:
        dict: Dictionary containing bounding boxes related to their image.

    """
    from spinalcordtoolbox.ivadomed import inference as imed_inference
    bounding_box_dict = {}
    if safety_factor is None:
        safety_factor = [1.0, 1.0, 1.0]
    for subject_path in subject_path_list:
        object_mask, _ = imed_inference.segment_volume(model_path, [subject_path], gpu_id=gpu_id)
        object_mask = object_mask[0]
        if keep_largest_only:
            object_mask = imed_postpro.keep_largest_object(object_mask)
        mask_path = Path(path_output, "detection_mask")
        if not mask_path.exists():
            mask_path.mkdir(parents=True)
        nib.save(object_mask, Path(mask_path, subject_path.split("/")[-1]))
        ras_orientation = nib.as_closest_canonical(object_mask)
        hwd_orientation = imed_loader_utils.orient_img_hwd(ras_orientation.get_fdata(), slice_axis)
        bounding_boxes = get_bounding_boxes(hwd_orientation)
        bounding_box_dict[subject_path] = [adjust_bb_size(bb, safety_factor) for bb in bounding_boxes]

    file_path = Path(path_output, 'bounding_boxes.json')
    with file_path.open(mode="w") as fp:
        json.dump(bounding_box_dict, fp, indent=4)
    return bounding_box_dict


def resample_bounding_box(metadata, transform):
    """Resample bounding box.

    Args:
        metadata (dict): Dictionary containing the metadata to be modified with the resampled coordinates.
        transform (Compose): Transformations possibly containing the resample params.
    """
    for idx, transfo in enumerate(transform.transform["im"].transforms):
        if "Resample" == transfo.__class__.__name__:
            hspace, wspace, dspace = (transfo.hspace, transfo.wspace, transfo.dspace)
            hfactor = metadata[MetadataKW.INPUT_METADATA][0][MetadataKW.ZOOMS][0] / hspace
            wfactor = metadata[MetadataKW.INPUT_METADATA][0][MetadataKW.ZOOMS][1] / wspace
            dfactor = metadata[MetadataKW.INPUT_METADATA][0][MetadataKW.ZOOMS][2] / dspace
            factor = (hfactor, wfactor, dfactor)
            coord = adjust_bb_size(metadata[MetadataKW.INPUT_METADATA][0][MetadataKW.BOUNDING_BOX], factor, resample=True)

            for i in range(len(metadata[MetadataKW.INPUT_METADATA])):
                metadata[MetadataKW.INPUT_METADATA][i][MetadataKW.BOUNDING_BOX] = coord

            for i in range(len(metadata[MetadataKW.GT_METADATA])):
                metadata[MetadataKW.GT_METADATA][i][MetadataKW.BOUNDING_BOX] = coord
            break


def adjust_transforms(transforms, seg_pair, length=None, stride=None):
    """This function adapts the transforms by adding the BoundingBoxCrop transform according the specific parameters of
    an image. The dimensions of the crop are also adapted to fit the length and stride parameters if the 3D loader is
    used.

    Args:
        transforms (Compose): Prepreocessing transforms.
        seg_pair (dict): Segmentation pair (input, gt and metadata).
        length (list or tuple): Patch size of the 3D loader.
        stride (list or tuple): Stride value of the 3D loader.

    Returns:
        Compose: Modified transforms.
    """
    resample_idx = [-1, -1, -1]
    if transforms is None:
        transforms = imed_transforms.Compose({})
    for i, img_type in enumerate(transforms.transform):
        for idx, transfo in enumerate(transforms.transform[img_type].transforms):
            if "BoundingBoxCrop" == transfo.__class__.__name__:
                transforms.transform[img_type].transforms.pop(idx)
            if "Resample" == transfo.__class__.__name__:
                resample_idx[i] = idx

    resample_bounding_box(seg_pair, transforms)
    index_shape = []
    for i, img_type in enumerate(transforms.transform):
        x_min, x_max, y_min, y_max, z_min, z_max = seg_pair['input_metadata'][0]['bounding_box']
        size = [x_max - x_min, y_max - y_min, z_max - z_min]

        if length is not None and stride is not None:
            for idx, dim in enumerate(size):
                if dim < length[idx]:
                    size[idx] = length[idx]
            # Adjust size according to stride to avoid dimension mismatch
            size = resize_to_multiple(size, stride, length)
        index_shape.append(tuple(size))
        transform_obj = imed_transforms.BoundingBoxCrop(size=size)
        transforms.transform[img_type].transforms.insert(resample_idx[i] + 1, transform_obj)

    for metadata in seg_pair['input_metadata']:
        assert len(set(index_shape)) == 1
        metadata[MetadataKW.INDEX_SHAPE] = index_shape[0]
    return transforms


def adjust_undo_transforms(transforms, seg_pair, index=0):
    """This function adapts the undo transforms by adding the BoundingBoxCrop to undo transform according the specific
    parameters of an image.

    Args:
        transforms (Compose): Transforms.
        seg_pair (dict): Segmentation pair (input, gt and metadata).
        index (int): Batch index of the seg_pair.
    """
    for img_type in transforms.transform:
        resample_idx = -1
        for idx, transfo in enumerate(transforms.transform[img_type].transforms):
            if "Resample" == transfo.__class__.__name__:
                resample_idx = idx
            if "BoundingBoxCrop" == transfo.__class__.__name__:
                transforms.transform[img_type].transforms.pop(idx)
        if "bounding_box" in seg_pair['input_metadata'][index][0]:
            size = list(seg_pair['input_metadata'][index][0]['index_shape'])
            transform_obj = imed_transforms.BoundingBoxCrop(size=size)
            transforms.transform[img_type].transforms.insert(resample_idx + 1, transform_obj)


def load_bounding_boxes(object_detection_params, subject_path_list, slice_axis, constrast_lst):
    """Verifies if bounding_box.json exists in the output path, if so loads the data, else generates the file if a
    valid detection model path exists.

    Args:
        object_detection_params (dict): Object detection parameters.
        subject_path_list (list): List of all subjects in all the BIDS directories.
        slice_axis (int): Slice axis (0: sagittal, 1: coronal, 2: axial).
        constrast_lst (list): Contrasts.

    Returns:
        dict: bounding boxes for every subject in BIDS directory
    """
    # Load or generate bounding boxes and save them in json file
    bounding_box_dict = {}
    if object_detection_params is None or object_detection_params[ObjectDetectionParamsKW.OBJECT_DETECTION_PATH] is None:
        return bounding_box_dict

    bounding_box_path = Path(object_detection_params.get(ObjectDetectionParamsKW.PATH_OUTPUT), 'bounding_boxes.json')
    if bounding_box_path.exists():
        with bounding_box_path.open(mode='r') as fp:
            bounding_box_dict = json.load(fp)
    elif object_detection_params[ObjectDetectionParamsKW.OBJECT_DETECTION_PATH] is not None and \
            Path(object_detection_params.get(ObjectDetectionParamsKW.OBJECT_DETECTION_PATH)).exists():
        bounding_box_dict = generate_bounding_box_file(subject_path_list,
                                                       object_detection_params[ObjectDetectionParamsKW.OBJECT_DETECTION_PATH],
                                                       object_detection_params[ObjectDetectionParamsKW.PATH_OUTPUT],
                                                       object_detection_params[ObjectDetectionParamsKW.GPU_IDS],
                                                       slice_axis,
                                                       constrast_lst,
                                                       safety_factor=object_detection_params[ObjectDetectionParamsKW.SAFETY_FACTOR])
    elif object_detection_params[ObjectDetectionParamsKW.OBJECT_DETECTION_PATH] is not None:
        raise RuntimeError("Path to object detection model doesn't exist")

    return bounding_box_dict


def verify_metadata(metadata, has_bounding_box):
    """Validates across all metadata that the 'bounding_box' param is present.

    Args:
        metadata (dict): Image metadata.
        has_bounding_box (bool): If 'bounding_box' is present across all metadata, True, else False.

    Returns:
        bool: Boolean indicating if 'bounding_box' is present across all metadata.
    """
    index_has_bounding_box = all([MetadataKW.BOUNDING_BOX in metadata[MetadataKW.INPUT_METADATA][i]
                                  for i in range(len(metadata[MetadataKW.INPUT_METADATA]))])
    for gt_metadata in metadata[MetadataKW.GT_METADATA]:
        if gt_metadata is not None:
            index_has_bounding_box &= MetadataKW.BOUNDING_BOX in gt_metadata

    has_bounding_box &= index_has_bounding_box
    return has_bounding_box


def bounding_box_prior(fname_mask, metadata, slice_axis, safety_factor=None):
    """
    Computes prior steps to a model requiring bounding box crop. This includes loading a mask of the ROI, orienting the
    given mask into the following dimensions: (height, width, depth), extracting the bounding boxes and storing the
    information in the metadata.

    Args:
        fname_mask (str): Filename containing the mask of the ROI
        metadata (dict): Dictionary containing the image metadata
        slice_axis (int): Slice axis (0: sagittal, 1: coronal, 2: axial)
        safety_factor (list or tuple): Factors to multiply each dimension of the bounding box.

    """
    nib_prior = nib.load(fname_mask)
    # orient image into HWD convention
    nib_ras = nib.as_closest_canonical(nib_prior)
    np_mask = nib_ras.get_fdata()[..., 0] if len(nib_ras.get_fdata().shape) == 4 else nib_ras.get_fdata()
    np_mask = imed_loader_utils.orient_img_hwd(np_mask, slice_axis)
    # Extract the bounding box from the list
    bounding_box = get_bounding_boxes(np_mask)[0]
    if safety_factor:
        bounding_box = adjust_bb_size(bounding_box, safety_factor)
    metadata[MetadataKW.BOUNDING_BOX] = bounding_box


def compute_bb_statistics(bounding_box_path):
    """Measures min, max and average, height, width, depth and volume of bounding boxes from a json file

    Args:
        bounding_box_path (string): Path to json file.
    """
    with Path(bounding_box_path).open(mode='r') as fp:
        bounding_box_dict = json.load(fp)

    h, w, d, v = [], [], [], []
    for box in bounding_box_dict:
        x_min, x_max, y_min, y_max, z_min, z_max = bounding_box_dict[box]
        h.append(x_max - x_min)
        w.append(y_max - y_min)
        d.append(z_max - z_min)
        v.append((x_max - x_min) * (y_max - y_min) * (z_max - z_min))

    logger.info('Mean height: {} +/- {}, min: {}, max: {}'.format(statistics.mean(h), statistics.stdev(h), min(h), max(h)))
    logger.info('Mean width: {} +/- {}, min: {}, max: {}'.format(statistics.mean(w), statistics.stdev(w), min(w), max(w)))
    logger.info('Mean depth: {} +/- {}, min: {}, max: {}'.format(statistics.mean(d), statistics.stdev(d), min(d), max(d)))
    logger.info('Mean volume: {} +/- {}, min: {}, max: {}'.format(statistics.mean(v), statistics.stdev(v), min(v), max(v)))
