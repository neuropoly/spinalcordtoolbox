# This file is vendored from ivadomed 2.9.10 (https://github.com/ivadomed/ivadomed),
# originally `ivadomed/inference.py`, licensed under the MIT License.
# See the LICENSE.md file in this directory for the full license text.
#
# Modifications for SCT:
#   - Removed FiLMedUnet/HeMISUnet support, which needs ivadomed's training code. Model configs that enable either one now
#     raise `NotImplementedError`.
#   - Fixed flake8 warnings, without changing behavior.
#   - Imports point to `spinalcordtoolbox.ivadomed` instead of `ivadomed`.
#   - `torch.load` is replaced by `pickle_compat.torch_load`, so that ivadomed-pickled `.pt` models load with the vendored
#     classes.

import nibabel as nib
import numpy as np
import onnxruntime
import torch
import imageio
from typing import List

from loguru import logger
from torch.utils.data import DataLoader
from torch import tensor

from spinalcordtoolbox.ivadomed.loader.mri3d_subvolume_segmentation_dataset import MRI3DSubVolumeSegmentationDataset
from spinalcordtoolbox.ivadomed.loader.mri2d_segmentation_dataset import MRI2DSegmentationDataset
from spinalcordtoolbox.ivadomed.transforms import UndoCompose
from spinalcordtoolbox.ivadomed import config_manager as imed_config_manager
from spinalcordtoolbox.ivadomed import models as imed_models
from spinalcordtoolbox.ivadomed import postprocessing as imed_postpro
from spinalcordtoolbox.ivadomed import transforms as imed_transforms
from spinalcordtoolbox.ivadomed.loader import utils as imed_loader_utils
from spinalcordtoolbox.ivadomed.loader.slice_filter import SliceFilter
from spinalcordtoolbox.ivadomed.object_detection import utils as imed_obj_detect
from spinalcordtoolbox.ivadomed import utils as imed_utils
from spinalcordtoolbox.ivadomed.pickle_compat import torch_load
from spinalcordtoolbox.ivadomed.keywords import ConfigKW, ModelParamsKW, ObjectDetectionParamsKW, TransformationKW, LoaderParamsKW, \
    ROIParamsKW, SliceFilterParamsKW, TrainingParamsKW, MetadataKW, OptionKW


def onnx_inference(model_path: str, inputs: tensor) -> tensor:
    """Run ONNX inference

    Args:
        model_path (str): Path to the ONNX model.
        inputs (Tensor): Batch of input image.

    Returns:
        Tensor: Network output.
    """
    inputs = np.array(inputs.cpu())
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: inputs}
    ort_outs = ort_session.run(None, ort_inputs)
    return torch.tensor(ort_outs[0])


def get_preds(context: dict, fname_model: str, model_params: dict, cuda_available: bool, device: torch.device, batch: dict) -> tensor:
    """Returns the predictions from the given model.

    Args:
        context (dict): configuration dict.
        fname_model (str): name of file containing model.
        model_params (dict): dictionary containing model parameters.
        cuda_available (bool): True if cuda is available.
        device (torch.device): Device used for prediction.
        batch (dict): dictionary containing input, gt and metadata

    Returns:
        tensor: predictions from the model.
    """

    with torch.no_grad():

        # Load the Input
        img = imed_utils.cuda(batch['input'], cuda_available=cuda_available)

        # Load the PyTorch model and evaluate if model files exist.
        if fname_model.lower().endswith('.pt'):
            logger.debug(f"PyTorch model detected at: {fname_model}")
            logger.debug(f"Loading model from: {fname_model}")
            model = torch_load(fname_model, map_location=device)
            # Inference time
            logger.debug(f"Evaluating model: {fname_model}")
            model.eval()
            preds = model(img)
        # Otherwise, Onnex Inference (PyTorch can't load .onnx)
        else:
            logger.debug(f"Likely ONNX model detected at: {fname_model}")
            logger.debug("Conduct ONNX model inference... ")
            preds = onnx_inference(fname_model, img)

        logger.debug("Sending predictions to CPU")
        # Move prediction to CPU
        preds = preds.cpu()

    return preds


def pred_to_nib(data_lst: List[np.ndarray], z_lst: List[int], fname_ref: str, fname_out: str, slice_axis: int,
                debug: bool = False, kernel_dim: str = '2d', bin_thr: float = 0.5, discard_noise: bool = True,
                postprocessing: dict = None) -> nib.Nifti1Image:
    """Save the network predictions as nibabel object.

    Based on the header of `fname_ref` image, it creates a nibabel object from the Network predictions (`data_lst`).

    Args:
        data_lst (list of np arrays): Predictions, either 2D slices either 3D patches.
        z_lst (list of ints): Slice indexes to reconstruct a 3D volume for 2D slices.
        fname_ref (str): Filename of the input image: its header is copied to the output nibabel object.
        fname_out (str): If not None, then the generated nibabel object is saved with this filename.
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.
        debug (bool): If True, extended verbosity and intermediate outputs.
        kernel_dim (str): Indicates whether the predictions were done on 2D or 3D patches. Choices: '2d', '3d'.
        bin_thr (float): If positive, then the segmentation is binarized with this given threshold. Otherwise, a soft
            segmentation is output.
        discard_noise (bool): If True, predictions that are lower than 0.01 are set to zero.
        postprocessing (dict): Contains postprocessing steps to be applied.

    Returns:
        nibabel.Nifti1Image: NiBabel object containing the Network prediction.
    """

    # Check fname_ref extention and update path if not NifTI
    fname_ref = imed_loader_utils.update_filename_to_nifti(fname_ref)

    # Load reference nibabel object
    nib_ref = nib.load(fname_ref)
    nib_ref_can = nib.as_closest_canonical(nib_ref)

    if kernel_dim == '2d':
        # complete missing z with zeros
        tmp_lst = []
        for z in range(nib_ref_can.header.get_data_shape()[slice_axis]):
            if z not in z_lst:
                tmp_lst.append(np.zeros(data_lst[0].shape))
            else:
                tmp_lst.append(data_lst[z_lst.index(z)])

        if debug:
            logger.debug(f"Len {len(tmp_lst)}")
            for arr in tmp_lst:
                logger.debug(f"Shape element lst {arr.shape}")

        # create data and stack on depth dimension
        arr_pred_ref_space = np.stack(tmp_lst, axis=-1)

    else:
        arr_pred_ref_space = data_lst[0]

    n_channel = arr_pred_ref_space.shape[0]
    oriented_volumes = []
    if len(arr_pred_ref_space.shape) == 4:
        for i in range(n_channel):
            oriented_volumes.append(
                imed_loader_utils.reorient_image(arr_pred_ref_space[i, ], slice_axis, nib_ref, nib_ref_can))
        # transpose to locate the channel dimension at the end to properly see image on viewer
        arr_pred_ref_space = np.asarray(oriented_volumes).transpose((1, 2, 3, 0))
    else:
        arr_pred_ref_space = imed_loader_utils.reorient_image(arr_pred_ref_space, slice_axis, nib_ref, nib_ref_can)

    if bin_thr >= 0:
        arr_pred_ref_space = imed_postpro.threshold_predictions(arr_pred_ref_space, thr=bin_thr)
    elif discard_noise:  # discard noise
        arr_pred_ref_space[arr_pred_ref_space <= 1e-2] = 0

    # create nibabel object
    if postprocessing:
        fname_prefix = fname_out.split("_pred.nii.gz")[0] if fname_out is not None else None
        postpro = imed_postpro.Postprocessing(postprocessing,
                                              arr_pred_ref_space,
                                              nib_ref.header['pixdim'][1:4],
                                              fname_prefix)
        arr_pred_ref_space = postpro.apply()

    # Here we prefer to copy the header (rather than just the affine matrix), in order to preserve the qform_code.
    # See: https://github.com/ivadomed/ivadomed/issues/711
    nib_pred = nib.Nifti1Image(
        dataobj=arr_pred_ref_space,
        affine=nib_ref.header.get_best_affine(),
        header=nib_ref.header.copy()
    )
    # save as NifTI file
    if fname_out is not None:
        nib.save(nib_pred, fname_out)

    return nib_pred


def pred_to_png(pred_list: list, target_list: list, subj_path: str, suffix: str = '', max_value: int = 1):
    """Save the network predictions as PNG files with suffix "_target_pred".

    Args:
        pred_list (list of np arrays): list of 2D predictions.
        target_list (list of str): list of target suffixes.
        subj_path (str): Path of the subject filename in output folder without extension
            (e.g. "path_output/pred_masks/sub-01_sample-01_SEM").
        suffix (str): additional suffix to append to the filename (e.g. "_pred.png")
        max_value (int): Maximum mask value of the float mask to use during the conversion to uint8.
    """
    for pred, target in zip(pred_list, target_list):
        filename = subj_path + target + suffix
        data = pred.get_fdata()
        _img = (data * 255 / max_value).astype(np.uint8).squeeze()
        imageio.v3.imwrite(filename, _img, extension='.png')


def process_transformations(context: dict, fname_roi: str, fname_prior: str, metadata: dict, slice_axis: int,
                            fname_images: list) -> dict:
    """Sets the transformation based on context parameters. When ROI is not provided center-cropping is applied.

    If there is an object_detection_path, then we modify the metadata to store transformation data.

    Args:
        context (dict): configuration dictionary.
        fname_roi (str): filename containing region for cropping image prior to segmentation.
        fname_prior (str): prior image filename.
        metadata (dict): metadata used in setting bounding box when we have object_detection_params.
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.
        fname_images (list): list of image filenames (e.g. .nii.gz) to segment.

    Returns:
        dict: metadata.
    """
    if fname_roi is None and TransformationKW.ROICROP in context[ConfigKW.TRANSFORMATION].keys():
        logger.warning(
            "fname_roi has not been specified, then a cropping around the center of the image is "
            "performed instead of a cropping around a Region of Interest.")

        # Convert transformation configuration into dict.
        for (key, value) in context[ConfigKW.TRANSFORMATION].items():
            if key != TransformationKW.ROICROP:
                context[ConfigKW.TRANSFORMATION] = dict(key, value)
            else:
                context[ConfigKW.TRANSFORMATION] = dict(TransformationKW.CENTERCROP, value)

    if ConfigKW.OBJECT_DETECTION_PARAMS in context and \
            context[ConfigKW.OBJECT_DETECTION_PARAMS][ObjectDetectionParamsKW.OBJECT_DETECTION_PATH] is not None:
        imed_obj_detect.bounding_box_prior(fname_prior, metadata, slice_axis,
                                           context[ConfigKW.OBJECT_DETECTION_PARAMS][ObjectDetectionParamsKW.SAFETY_FACTOR])
        metadata = [metadata] * len(fname_images)

    return metadata


def set_option(options: dict, postpro: dict, context: dict, key: str):
    """Generalized function that sets postprocessing option based on given list of options.

    When given key already exists in options, we initialize the key value for the postprocessing dictionary
    Otherwise, when the key is already found in the postprocessing attritute of the context, we remove it

    Args:
        options (dict): Contains postprocessing steps information.
        postpro (dict): Postprocessing settings.
        context (dict): Configuration dict.
        key (str): The key of the postprocessing option we wish to set.

    Returns:
        dict: postprocessing settings.
    """
    if options[key]:
        postpro[key] = {}
    # Remove key in context if value set to 0
    elif key in context[ConfigKW.POSTPROCESSING]:
        del context[ConfigKW.POSTPROCESSING][key]


def set_postprocessing_options(options: dict, context: dict):
    """Updates the postprocessing options based on existing settings found in options.

    Args:
        options (dict): Contains postprocessing steps information.
        context (dict): Configuration dict.
    """
    postpro = {}

    if OptionKW.BINARIZE_PREDICTION in options and options[OptionKW.BINARIZE_PREDICTION] is not None:
        postpro[OptionKW.BINARIZE_PREDICTION] = {"thr": options[OptionKW.BINARIZE_PREDICTION]}

    if OptionKW.BINARIZE_MAXPOOLING in options and options.get(OptionKW.BINARIZE_MAXPOOLING) is not None:
        set_option(options, postpro, context, OptionKW.BINARIZE_MAXPOOLING)

    if OptionKW.KEEP_LARGEST in options and options.get(OptionKW.KEEP_LARGEST) is not None:
        set_option(options, postpro, context, OptionKW.KEEP_LARGEST)

    if OptionKW.FILL_HOLES in options and options.get(OptionKW.FILL_HOLES) is not None:
        set_option(options, postpro, context, OptionKW.FILL_HOLES)

    if OptionKW.REMOVE_SMALL in options and options.get(OptionKW.REMOVE_SMALL) and \
            ('mm' in options[OptionKW.REMOVE_SMALL][-1] or 'vox' in options[OptionKW.REMOVE_SMALL][-1]):
        unit = 'mm3' if 'mm3' in options[OptionKW.REMOVE_SMALL][-1] else 'vox'
        thr = [int(t.replace(unit, "")) for t in options[OptionKW.REMOVE_SMALL]]
        postpro[OptionKW.REMOVE_SMALL] = {"unit": unit, "thr": thr}

    context[ConfigKW.POSTPROCESSING].update(postpro)


def segment_volume(folder_model: str, fname_images: list, gpu_id: int = 0, options: dict = None):
    """Segment an image.

    Segment an image (`fname_image`) using a pre-trained model (`folder_model`). If provided, a region of interest
    (`fname_roi`) is used to crop the image prior to segment it.

    Args:
        folder_model (str): foldername which contains
            (1) the model ('folder_model/folder_model.pt') to use
            (2) its configuration file ('folder_model/folder_model.json') used for the training,
            see https://github.com/neuropoly/ivadomed/wiki/configuration-file
        fname_images (list): list of image filenames (e.g. .nii.gz) to segment. Multichannel models require multiple
            images to segment, e.i., len(fname_images) > 1.
        gpu_id (int): Number representing gpu number if available. Currently does NOT support multiple GPU segmentation.
        options (dict): This can optionally contain any of the following key-value pairs:

            * 'binarize_prediction': (float) Binarize segmentation with specified threshold. \
                Predictions below the threshold become 0, and predictions above or equal to \
                threshold become 1. Set to -1 for no thresholding (i.e., soft segmentation).
            * 'binarize_maxpooling': (bool) Binarize by setting to 1 the voxel having the maximum prediction across \
                all classes. Useful for multiclass models.
            * 'fill_holes': (bool) Fill small holes in the segmentation.
            * 'keep_largest': (bool) Keep the largest connected-object for each class from the output segmentation.
            * 'remove_small': (list of str) Minimal object size to keep with unit (mm3 or vox). A single value can be provided \
                              or one value per prediction class. Single value example: ["1mm3"], ["5vox"]. Multiple values \
                              example: ["10", "20", "10vox"] (remove objects smaller than 10 voxels for class 1 and 3, \
                              and smaller than 20 voxels for class 2).
            * 'pixel_size': (list of float) List of microscopy pixel size in micrometers. \
                            Length equals 2 [PixelSizeX, PixelSizeY] for 2D or 3 [PixelSizeX, PixelSizeY, PixelSizeZ] for 3D, \
                            where X is the width, Y the height and Z the depth of the image.
            * 'pixel_size_units': (str) Units of pixel size (Must be either "mm", "um" or "nm")
            * 'no_patch': (bool) 2D patches are not used while segmenting with models trained with patches. \
                          The "no_patch" option supersedes the "overlap_2D" option. \
                          This option may not be suitable with large images depending on computer RAM capacity.
            * 'overlap_2D': (list of int) List of overlaps in pixels for 2D patching. Length equals 2 [OverlapX, OverlapY], \
                            where X is the width and Y the height of the image.
            * 'metadata': (str) Film metadata.
            * 'fname_prior': (str) An image filename (e.g., .nii.gz) containing processing information \
                (e.g., spinal cord segmentation, spinal location or MS lesion classification, spinal cord centerline), \
                used to crop the image prior to segment it if provided. \
                The segmentation is not performed on the slices that are empty in this image.

    Returns:
        list, list: List of nibabel objects containing the soft segmentation(s), one per prediction class, \
            List of target suffix associated with each prediction in `pred_list`

    """

    # Define device
    cuda_available, device = imed_utils.define_device(gpu_id)

    # Check if model folder exists and get filenames to be stored as string
    fname_model: str
    fname_model_metadata: str
    fname_model, fname_model_metadata = imed_models.get_model_filenames(folder_model)

    # Load model training config
    context = imed_config_manager.ConfigurationManager(fname_model_metadata).get_config()

    postpro_list = ['binarize_prediction', 'binarize_maxpooling', 'keep_largest', ' fill_holes',
                    'remove_small']
    if options is not None and any(pp in options for pp in postpro_list):
        set_postprocessing_options(options, context)

    # LOADER
    loader_params = context[ConfigKW.LOADER_PARAMETERS]
    slice_axis = imed_utils.AXIS_DCT[loader_params[LoaderParamsKW.SLICE_AXIS]]
    metadata = {}
    fname_roi = None

    if (options is not None) and (OptionKW.FNAME_PRIOR in options):
        fname_prior = options.get(OptionKW.FNAME_PRIOR)
    else:
        fname_prior = None

    if fname_prior is not None:
        if LoaderParamsKW.ROI_PARAMS in loader_params and loader_params[LoaderParamsKW.ROI_PARAMS][ROIParamsKW.SUFFIX] is not None:
            fname_roi = fname_prior
        # TRANSFORMATIONS
        metadata = process_transformations(context, fname_roi, fname_prior, metadata, slice_axis, fname_images)

    # Compose transforms
    _, _, transform_test_params = imed_transforms.get_subdatasets_transforms(context[ConfigKW.TRANSFORMATION])

    tranform_lst, undo_transforms = imed_transforms.prepare_transforms(transform_test_params)

    # Force filter_empty_mask to False if fname_roi = None
    if fname_roi is None and SliceFilterParamsKW.FILTER_EMPTY_MASK in loader_params[LoaderParamsKW.SLICE_FILTER_PARAMS]:
        logger.warning("fname_roi has not been specified, then the entire volume is processed.")
        loader_params[LoaderParamsKW.SLICE_FILTER_PARAMS][SliceFilterParamsKW.FILTER_EMPTY_MASK] = False

    kernel_3D = bool(ConfigKW.MODIFIED_3D_UNET in context and context[ConfigKW.MODIFIED_3D_UNET][ModelParamsKW.APPLIED]) or \
        not context[ConfigKW.DEFAULT_MODEL][ModelParamsKW.IS_2D]

    if (options is not None) and (OptionKW.NO_PATCH in options) and kernel_3D:
        logger.warning("The 'no-patch' option is provided but is not available for 3D models. "
                       "'no-patch' is ignored.")
    if (options is not None) and (OptionKW.OVERLAP_2D in options) and kernel_3D:
        logger.warning("The 'overlap-2d' option is provided but is not available for 3D models. "
                       "'overlap-2d' is ignored.")

    # Assign length_2D and stride_2D for 2D patching
    length_2D = context[ConfigKW.DEFAULT_MODEL][ModelParamsKW.LENGTH_2D] if \
        ModelParamsKW.LENGTH_2D in context[ConfigKW.DEFAULT_MODEL] else []
    stride_2D = context[ConfigKW.DEFAULT_MODEL][ModelParamsKW.STRIDE_2D] if \
        ModelParamsKW.STRIDE_2D in context[ConfigKW.DEFAULT_MODEL] else []

    is_2d_patch = bool(length_2D)
    if (options is not None) and (OptionKW.NO_PATCH in options) and not kernel_3D:
        if is_2d_patch:
            is_2d_patch = not options.get(OptionKW.NO_PATCH)
            length_2D = []
            stride_2D = []
        else:
            logger.warning(f"The 'no-patch' option is provided but the model has no 'length_2D' and "
                           f"'stride_2D' parameters in its configuration file "
                           f"'{fname_model_metadata.split('/')[-1]}'. 2D patching is ignored, the segmentation "
                           f"is done on the entire image without patches.")
        if OptionKW.OVERLAP_2D in options:
            logger.warning("The 'no-patch' option is provided along with the 'overlap-2D' option. "
                           "2D patching is ignored, the segmentation is done on the entire image without patches.")
    else:
        if (options is not None) and (OptionKW.OVERLAP_2D in options) and not kernel_3D:
            if (length_2D and stride_2D):
                overlap_2D = options.get(OptionKW.OVERLAP_2D)
                # Swap OverlapX and OverlapY resulting in an array in order [OverlapY, OverlapX]
                # to match length_2D and stride_2D in [Height, Width] orientation.
                overlap_2D[1], overlap_2D[0] = overlap_2D[0], overlap_2D[1]
                # Adjust stride_2D with overlap_2D
                stride_2D = [x1 - x2 for (x1, x2) in zip(length_2D, overlap_2D)]
            else:
                logger.warning(f"The 'overlap-2d' option is provided but the model has no 'length_2D' and "
                               f"'stride_2D' parameters in its configuration file "
                               f"'{fname_model_metadata.split('/')[-1]}'. 2D patching is ignored, the segmentation "
                               f"is done on the entire image without patches.")

    # Add microscopy pixel size and pixel size units from options to metadata for filenames_pairs
    if (options is not None) and (OptionKW.PIXEL_SIZE in options):
        metadata[MetadataKW.PIXEL_SIZE] = options.get(OptionKW.PIXEL_SIZE)
    if (options is not None) and (OptionKW.PIXEL_SIZE_UNITS in options):
        metadata[MetadataKW.PIXEL_SIZE_UNITS] = options.get(OptionKW.PIXEL_SIZE_UNITS)

    filename_pairs = [(fname_images, None, fname_roi, metadata if isinstance(metadata, list) else [metadata])]

    if kernel_3D:
        ds = MRI3DSubVolumeSegmentationDataset(filename_pairs,
                                               transform=tranform_lst,
                                               length=context[ConfigKW.MODIFIED_3D_UNET][ModelParamsKW.LENGTH_3D],
                                               stride=context[ConfigKW.MODIFIED_3D_UNET][ModelParamsKW.STRIDE_3D],
                                               slice_axis=slice_axis)
        logger.info(f"Loaded {len(ds)} {loader_params[LoaderParamsKW.SLICE_AXIS]} volumes of shape "
                    f"{context[ConfigKW.MODIFIED_3D_UNET][ModelParamsKW.LENGTH_3D]}.")
    else:
        ds = MRI2DSegmentationDataset(filename_pairs,
                                      length=length_2D,
                                      stride=stride_2D,
                                      slice_axis=slice_axis,
                                      nibabel_cache=True,
                                      transform=tranform_lst,
                                      slice_filter_fn=SliceFilter(
                                          **loader_params[LoaderParamsKW.SLICE_FILTER_PARAMS]))
        ds.load_filenames()
        if is_2d_patch:
            logger.info(f"Loaded {len(ds)} {loader_params[LoaderParamsKW.SLICE_AXIS]} patches of shape {length_2D}.")
        else:
            logger.info(f"Loaded {len(ds)} {loader_params[LoaderParamsKW.SLICE_AXIS]} slices.")

    # FiLMedUnet/HeMISUnet models need subject metadata at inference time, which requires ivadomed's training
    # code. None of SCT's ivadomed models use them, so they are not supported here.
    for model_name in (ConfigKW.FILMED_UNET, ConfigKW.HEMIS_UNET):
        if model_name in context and context[model_name].get(ModelParamsKW.APPLIED):
            raise NotImplementedError(f"'{model_name}' models are not supported by SCT's vendored copy of ivadomed.")
    model_params = {}

    # Data Loader
    data_loader = DataLoader(ds, batch_size=context[ConfigKW.TRAINING_PARAMETERS][TrainingParamsKW.BATCH_SIZE],
                             shuffle=False, pin_memory=True,
                             collate_fn=imed_loader_utils.imed_collate,
                             num_workers=0)

    # Loop across batches
    preds_list, slice_idx_list = [], []
    last_sample_bool, weight_matrix, volume, image = False, None, None, None
    for i_batch, batch in enumerate(data_loader):
        preds = get_preds(context, fname_model, model_params, cuda_available, device, batch)

        # Set datatype to gt since prediction should be processed the same way as gt
        for b in batch[MetadataKW.INPUT_METADATA]:
            for modality in b:
                modality['data_type'] = 'gt'

        # Reconstruct 3D object
        pred_list, target_list, last_sample_bool, weight_matrix, volume, image = reconstruct_3d_object(
            context, batch, undo_transforms, preds, preds_list, kernel_3D, is_2d_patch, slice_axis,
            slice_idx_list, data_loader, fname_images, i_batch, last_sample_bool, weight_matrix,
            volume, image
        )

    return pred_list, target_list


def split_classes(nib_prediction):
    """Split a 4D nibabel multi-class segmentation file in multiple 3D nibabel binary segmentation files.

    Args:
        nib_prediction (nibabelObject): 4D nibabel object.
    Returns:
        list of nibabelObject.
     """
    pred = nib_prediction.get_fdata()
    pred_list = []
    for c in range(pred.shape[-1]):
        class_pred = nib.Nifti1Image(pred[..., c].astype('float32'), nib_prediction.header.get_best_affine(),
                                     nib_prediction.header.copy())
        pred_list.append(class_pred)
    return pred_list


def reconstruct_3d_object(context: dict, batch: dict, undo_transforms: UndoCompose, preds: torch.tensor,
                          preds_list: list, kernel_3D: bool, is_2d_patch: bool, slice_axis: int, slice_idx_list: list,
                          data_loader: DataLoader, fname_images: list, i_batch: int, last_sample_bool: bool,
                          weight_matrix: tensor, volume: tensor, image: tensor):
    """Reconstructs the 3D object from the current batch, and returns the list of predictions and targets.

    Args:
        context (dict): configuration dict.
        batch (dict): Dictionary containing input, gt and metadata
        undo_transforms (UndoCompose): Undo transforms so prediction match original image resolution and shape
        preds (tensor): Subvolume predictions
        preds_list (list of tensor): list of subvolume predictions.
        kernel_3D (bool): true when using 3D kernel.
        is_2d_patch (bool): Indicates if 2d patching is used.
        slice_axis (int): Indicates the axis used for the 2D slice extraction: Sagittal: 0, Coronal: 1, Axial: 2.
        slice_idx_list (list of int): list of indices for the axis slices.
        data_loader (DataLoader): DataLoader object containing batches using in object construction.
        fname_images (list): list of image filenames (e.g. .nii.gz) to segment.
        i_batch (int): index of current batch.
        last_sample_bool (bool) : flag to indicate whether this is the last sample in the 3D volume
        weight_matrix (tensor): the weight matrix
        volume (tensor): the volume tensor that is being partially reconstructed through the loop
        image (tensor): the image tensor that is being partially reconstructed through the loop

    Returns:
        pred_list (list): list of predictions
        target_list (list): list of targets
        last_sample_bool (bool): flag to indicate whether this is the last sample in the 3D volume
        weight_matrix (tensor): the weight matrix. Must be returned as passing tensor by reference is NOT reliable.
        volume (tensor): the volume tensor that is being partially reconstructed through the loop. Must be returned \
            as passing tensor by reference is NOT reliable.
        image (tensor): the vimage tensor that is being partially reconstructed through the loop. Must be returned \
            as passing tensor by reference is NOT reliable.
    """
    pred_list = []
    target_list = []
    for i_slice in range(len(preds)):
        if "bounding_box" in batch[MetadataKW.INPUT_METADATA][i_slice][0]:
            imed_obj_detect.adjust_undo_transforms(undo_transforms.transforms, batch, i_slice)

        batch[MetadataKW.GT_METADATA] = [[metadata[0]] * preds.shape[1] for metadata in batch[MetadataKW.INPUT_METADATA]]
        if kernel_3D:
            preds_undo, metadata, last_sample_bool, volume, weight_matrix = \
                volume_reconstruction(batch, preds, undo_transforms, i_slice, volume, weight_matrix)
            if last_sample_bool:
                preds_list = [np.array(preds_undo)]
        else:
            if is_2d_patch:
                # undo transformations for patch and reconstruct slice
                preds_i_undo, metadata_idx, last_patch_bool, image, weight_matrix = \
                    image_reconstruction(batch, preds, undo_transforms, i_slice, image, weight_matrix)
                # If last patch of the slice
                if last_patch_bool:
                    # Add new segmented slice to preds_list
                    preds_list.append(np.array(preds_i_undo))
                    # Store the slice index of preds_i_undo in the original 3D image
                    slice_idx_list.append(int(batch[MetadataKW.INPUT_METADATA][i_slice][0]['slice_index']))
            else:
                # undo transformations for slice
                preds_i_undo, metadata_idx = undo_transforms(preds[i_slice],
                                                             batch[MetadataKW.GT_METADATA][i_slice],
                                                             data_type='gt')
                # Add new segmented slice to preds_list
                preds_list.append(np.array(preds_i_undo))
                # Store the slice index of preds_i_undo in the original 3D image
                slice_idx_list.append(int(batch[MetadataKW.INPUT_METADATA][i_slice][0]['slice_index']))

        # If last batch and last sample of this batch, then reconstruct 3D object
        if (i_batch == len(data_loader) - 1 and i_slice == len(batch['gt']) - 1) or last_sample_bool:
            pred_nib = pred_to_nib(data_lst=preds_list,
                                   fname_ref=fname_images[0],
                                   fname_out=None,
                                   z_lst=slice_idx_list,
                                   slice_axis=slice_axis,
                                   kernel_dim='3d' if kernel_3D else '2d',
                                   debug=False,
                                   bin_thr=-1,
                                   postprocessing=context[ConfigKW.POSTPROCESSING])

            pred_list = split_classes(pred_nib)
            target_list = context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.TARGET_SUFFIX]

    return pred_list, target_list, last_sample_bool, weight_matrix, volume, image


def volume_reconstruction(batch: dict, pred: tensor, undo_transforms: UndoCompose, smp_idx: int,
                          volume: tensor = None, weight_matrix: tensor = None):
    """
    Reconstructs volume prediction from subvolumes used during training
    Args:
        batch (dict): Dictionary containing input, gt and metadata
        pred (tensor): Subvolume prediction
        undo_transforms (UndoCompose): Undo transforms so prediction match original image resolution and shap
        smp_idx (int): Batch index
        volume (tensor): Reconstructed volume
        weight_matrix (tensor): Weights containing the number of predictions for each voxel

    Returns:
        pred_undo (tensor): undone subvolume,
        metadata (dict): metadata,
        last_sample_bool (bool): boolean representing if its the last sample of the volume
        volume (tensor): representing the volume reconstructed
        weight_matrix (tensor): weight matrix
    """
    pred_undo, metadata = None, None
    x_min, x_max, y_min, y_max, z_min, z_max = batch[MetadataKW.INPUT_METADATA][smp_idx][0]['coord']
    num_pred = pred[smp_idx].shape[0]

    # A boolean flag indicate whether the current volume is the VERY first subvolume of the entire 3D volume/space.
    # Formed by check if x_min/y_min/z_min are all NOT zero.
    first_sample: bool = (x_min == 0 and y_min == 0 and z_min == 0)

    # Get the Dimension
    x, y, z = batch[MetadataKW.INPUT_METADATA][smp_idx][0]['index_shape']

    # If this is the first sample, instantiate a ZERO tensor based on the dimension
    if first_sample:
        volume = torch.zeros((num_pred, x, y, z))
        weight_matrix = torch.zeros((num_pred, x, y, z))

    last_sample_bool = x_max == x and y_max == y and z_max == z

    # Average predictions
    volume[:, x_min:x_max, y_min:y_max, z_min:z_max] += pred[smp_idx]
    weight_matrix[:, x_min:x_max, y_min:y_max, z_min:z_max] += 1

    if last_sample_bool:
        volume /= weight_matrix
        pred_undo, metadata = undo_transforms(volume,
                                              batch[MetadataKW.GT_METADATA][smp_idx],
                                              data_type='gt')
    return pred_undo, metadata, last_sample_bool, volume, weight_matrix


def image_reconstruction(batch: dict, pred: tensor, undo_transforms: UndoCompose, smp_idx: int,
                         image: tensor = None, weight_matrix: tensor = None):
    """
    Reconstructs image prediction from patches used during training
    Args:
        batch (dict): Dictionary containing input, gt and metadata
        pred (tensor): Patch prediction
        undo_transforms (UndoCompose): Undo transforms so prediction match original image resolution and shape
        smp_idx (int): Batch index
        image (tensor): Reconstructed image
        weight_matrix (tensor): Weights containing the number of predictions for each pixel

    Returns:
        pred_undo (tensor): undone image
        metadata (dict): metadata
        last_patch_bool (bool): boolean representing if its the last patch of the image
        image (tensor): representing the image reconstructed
        weight_matrix (tensor): weight matrix
    """
    pred_undo, metadata = None, None
    x_min, x_max, y_min, y_max = batch[MetadataKW.INPUT_METADATA][smp_idx][0]['coord']
    num_pred = pred[smp_idx].shape[0]

    # A boolean flag indicate whether the current patch is the VERY first patch of the entire 2D image.
    # Formed by check if x_min/y_min are all NOT zero
    first_patch: bool = (x_min == 0 and y_min == 0)

    # Get the Dimension
    x, y = batch[MetadataKW.INPUT_METADATA][smp_idx][0]['index_shape']

    # If this is the first sample, instantiate a ZERO tensor based on the dimension
    if first_patch:
        image = torch.zeros((num_pred, x, y))
        weight_matrix = torch.zeros((num_pred, x, y))

    last_patch_bool = x_max == x and y_max == y

    # Average predictions
    image[:, x_min:x_max, y_min:y_max] += pred[smp_idx]
    weight_matrix[:, x_min:x_max, y_min:y_max] += 1
    if last_patch_bool:
        image /= weight_matrix
        pred_undo, metadata = undo_transforms(image, batch[MetadataKW.GT_METADATA][smp_idx], data_type='gt')

    return pred_undo, metadata, last_patch_bool, image, weight_matrix
