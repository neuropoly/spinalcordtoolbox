"""
Ensemble inference

Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import logging
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
import glob
import multiprocessing as mp

from spinalcordtoolbox.ivadomed import inference as imed_inference
from totalspineseg.inference import inference as tss_inference
import numpy as np
import torch
from monai.transforms import SaveImage
from monai.inferers import sliding_window_inference

from spinalcordtoolbox.utils.fs import tmp_create, extract_fname
from spinalcordtoolbox.image import Image, get_orientation, add_suffix
from spinalcordtoolbox.math import binarize, remove_small_objects
from spinalcordtoolbox.deepseg_.postprocessing import keep_largest_object, fill_holes

import spinalcordtoolbox.deepseg.models as ds_models
import spinalcordtoolbox.deepseg.monai as ds_monai
import spinalcordtoolbox.deepseg.nnunet as ds_nnunet
import spinalcordtoolbox.sc_crop as sc_crop

from spinalcordtoolbox.utils.sys import LazyLoader, stylize

nib = LazyLoader("nib", globals(), "nibabel")

logger = logging.getLogger(__name__)


def prediction_from_probability_map(prob_map):
    """
    Keep only the voxels where the foreground probability
    beats the background probability, similar to nnunetv2.
    """
    if prob_map.shape[0] != 2:
        raise ValueError(f"Expected 2-channel probability map with channels "
                         f"in first axis, got {prob_map.shape}.")
    background, foreground = prob_map[0], prob_map[1]
    return np.where(foreground > background, foreground, 0)


def segment_and_average_volumes(model_paths, input_filenames, options, use_gpu=False):
    """
        Run `ivadomed.inference.segment_volume()` once per model, then average the outputs.

        :param model_paths: A list of folder paths. The folders must contain:
            (1) the model ('folder_model/folder_model.pt') to use
            (2) its configuration file ('folder_model/folder_model.json') used for the training,
            see https://github.com/neuropoly/ivadomed/wiki/configuration-file
        :param input_filenames: list of image filenames (e.g. .nii.gz) to segment. Multichannel models require multiple
            images to segment, i.e.., len(fname_images) > 1.
        :param options: A dictionary containing optional configuration settings, as specified by the
            ivadomed.inference.segment_volume function.
        :param use_gpu: bool. Whether to try to perform inference using CUDA. (NB: Only a single GPU will be used.)

        :return: list, list: List of Image objects containing the soft segmentation(s), one per prediction class,
            List of target suffix associated with each prediction
    """
    if not isinstance(model_paths, list):
        raise TypeError("'model_paths' must be a list of model path strings.")
    if not len(model_paths) > 0:
        raise ValueError("'model_paths' must contain one or more model path strings.")

    # Fetch the name of the model (to be used in logging)
    name_model = Path(model_paths[0]).parts[-1]
    logger.info(f"\nRunning inference for model '{name_model}'...")

    # Perform inference once per model
    nii_lsts, target_lsts = [], []
    for path_model in model_paths:
        if len(model_paths) > 1:  # We have an ensemble, so output messages to distinguish between seeds
            name_seed = Path(path_model).parts[-2]
            logger.info(f"\nUsing '{name_seed}'...")
        # NB: ivadomed turns on GPU inference via specifying a single GPU ID. This isn't the recommended way to do
        #     things, since it prevents us from running multi-GPU jobs. I think ivadomed did things this way to limit
        #     which GPU is used, but we can already accomplish this using the more universal 'CUDA_VISIBLE_DEVICES'
        #     environment variable). For now, the best we can do on our end is to select the first GPU from the list
        #     of available GPUs.
        gpu_id = 0 if use_gpu else None  # NB: If e.g. 'CUDA_VISIBLE_DEVICES=2,3,4', then 0 will refer to GPU 2.
        nii_lst, target_lst = imed_inference.segment_volume(path_model, input_filenames, gpu_id=gpu_id, options=options)
        nii_lsts.append(nii_lst)
        target_lsts.append(target_lst)

    # If we have a single model, skip averaging
    if len(model_paths) == 1:
        nii_lst = nii_lsts[0]
    # Otherwise, we have a model ensemble, so average the image data
    else:
        logger.info(f"\nAveraging outputs across the ensemble for '{name_model}'...")
        nii_lst = []
        # NB: `nii_lsts` is a list of lists, with each sublist being the *per-model* predictions. Example:
        #         [
        #             [m1_prediction_1.nii.gz, m1_prediction_2.nii.gz, ...],  # model 1 predictions
        #             [m2_prediction_1.nii.gz, m2_prediction_2.nii.gz, ...]   # model 2 predictions
        #         ]
        # So, we want to take: the average of "prediction_1", the average of "prediction_2", etc.
        # To do this, we unpack + zip `nii_lists`, so that "prediction_N" files are grouped as "predictions".
        for predictions in zip(*nii_lsts):
            # Average the data for each output in the ensemble
            data_stack = np.stack([pred.get_fdata() for pred in predictions], axis=0)
            data_mean = np.mean(data_stack, axis=0)
            # Take the first image's header to reuse for the averaged image
            nii_header = predictions[0].header
            # Create a new Nifti1Image containing the averaged output
            nii_lst.append(nib.Nifti1Image(data_mean, header=nii_header, affine=nii_header.get_best_affine()))

    # The 'targets' should be identical for each model, so just take the first
    target_lst = target_lsts[0]

    # Convert the output Nifti1Images into SCT images, to avoid scaling issues
    # (see https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4346)
    im_lst = [Image(np.asanyarray(nii.dataobj), hdr=nii.header) for nii in nii_lst]

    return im_lst, target_lst


def segment_non_ivadomed(path_model, model_type, input_filenames, threshold, keep_largest, fill_holes_in_pred,
                         remove_small, use_gpu=False, remove_temp_files=True, extra_network_kwargs=None, extra_inference_kwargs=None):
    # MONAI and NNUnet have similar structure, and so we use nnunet+inference functions with the same signature
    # NB: For TotalSpineSeg, we don't need to create the network ourselves
    if os.path.basename(path_model) in ds_models.TASKS['spine']['models']:
        def create_net(pm, _): return pm  # just echo `path_model` back as 'net'
        inference = segment_totalspineseg
    elif model_type == "monai":
        create_net = ds_monai.create_nnunet_from_plans
        inference = segment_monai
    else:
        assert model_type == "nnunet"
        create_net = ds_nnunet.create_nnunet_from_plans
        inference = segment_nnunet

    device = torch.device("cuda" if use_gpu else "cpu")

    # load model from checkpoint
    net = create_net(path_model, device, **extra_network_kwargs)

    im_lst, target_lst = [], []
    for fname_in in input_filenames:
        tmpdir = tmp_create(basename="sct_deepseg")
        # model may be multiclass, so the `inference` func should output a list of fnames and targets
        fnames_out, targets = inference(path_img=fname_in, tmpdir=tmpdir, predictor=net, device=device, **extra_inference_kwargs)
        for fname_out, target in zip(fnames_out, targets):
            im_out = Image(fname_out)
            # Apply postprocessing (replicates existing functionality from ivadomed package)
            # 1. Binarize predictions based on the threshold value
            # NOTE: Any post-processing is done only after thresholding (as post-processing operations
            # expect binary masks)
            if threshold:  # Default: None, but `-thr 0` will also turn off binarization
                im_out.data = binarize(im_out.data, threshold)
            # 2. Keep the largest connected object
            if keep_largest != 0:
                im_out.data[im_out.data < 0.001] = 0  # Replicates ivadomed's `@binarize_with_low_threshold`
                im_out.data = keep_largest_object(im_out.data, x_cOm=None, y_cOm=None)
            # 3. Fill holes
            if fill_holes_in_pred != 0:
                im_out.data = fill_holes(im_out.data)
            # 4. Remove small objects
            if remove_small is not None:
                unit = 'mm3' if 'mm3' in remove_small[-1] else 'vox'
                thr = [int(t.replace(unit, "")) for t in remove_small]
                im_out.data = remove_small_objects(im_out.data, im_out.dim[4:7], unit, thr)
            im_lst.append(im_out)
            target_lst.append(target)
        if remove_temp_files:
            shutil.rmtree(tmpdir)

    return im_lst, target_lst


def segment_monai(path_img, tmpdir, predictor, device: torch.device):
    """
    Script to run inference on a MONAI-based model for contrast-agnostic soft segmentation of the spinal cord.

    Author: Naga Karthik
    Original script: https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/blob/e65478099d026f865b7f1d7d0082e6e9a507a744/monai/run_inference_single_image.py
    """
    # Copy the file to the temporary directory using shutil.copyfile
    path_img_tmp = os.path.join(tmpdir, os.path.basename(path_img))
    shutil.copyfile(path_img, path_img_tmp)
    logger.info(f'Copied {path_img} to {path_img_tmp}')

    # define inference patch size and center crop size
    crop_size = (64, 192, -1)
    inference_roi_size = (64, 192, 320)
    # NOTE: this is hard-coded to "edge" based on extensive experiments comparing "edge" vs "zero" padding
    # at test-time.
    pad_mode = "edge"

    # define the dataset and dataloader
    test_loader, test_post_pred = ds_monai.prepare_data(path_img_tmp, crop_size=crop_size, padding=pad_mode)
    [batch] = test_loader  # we expected there to only be one batch (with one image)

    # Run MONAI prediction
    print('Starting inference...')
    start = time.time()

    # run inference
    with torch.no_grad():
        test_input = batch["image"].to(device)
        batch["pred"] = sliding_window_inference(test_input, inference_roi_size, mode="gaussian",
                                                 sw_batch_size=4, predictor=predictor, overlap=0.5, progress=False,
                                                 sw_device=device)
        pred = ds_monai.postprocessing(batch, test_post_pred)

        end = time.time()
        print('Inference done.')
        total_time = end - start
        print(f'Total inference time: {int(total_time // 60)} minute(s) {int(round(total_time % 60))} seconds')

        # this takes about 0.25s on average on a CPU
        # image saver class
        _, fname, ext = extract_fname(path_img)
        postfix = "seg"
        target = f"_{postfix}"
        pred_saver = SaveImage(
            output_dir=tmpdir, output_postfix=postfix, output_ext=ext,
            separate_folder=False, print_log=False)
        # save the prediction
        fname_out = os.path.join(tmpdir, f"{fname}_{postfix}{ext}")
        logger.info(f"Saving results to: {fname_out}")
        pred_saver(pred)

    return [fname_out], [target]


def average_nnunet_predictions(pred, probabilities=False):
    """
    Compute an average prediction for multi-fold output of `nnunetv2`.

    The averaging process will differ depending on whether `pred` contains probability maps or per-fold logits.
    """
    # If we saved the probabilities, `pred` is a tuple of (binary pred, prob map)
    if probabilities:
        _, prob_maps = pred
        # In this case, pred is: pred[0] = list of softmax values per fold for class 0, pred[1] = list of softmax values per fold for class 1
        ensembled_pred = np.zeros_like(prob_maps[0][1])
        for prob_map in prob_maps:  # loop over folds
            # The shape of the prob_map is (num_classes, z, y, x), so we keep only the non-background class (1)
            # Similar to nnunet, the value of the soft max is the value of the probability map at class 1 if the value is greater than that of class 0
            fold_pred = prediction_from_probability_map(prob_map)
            ensembled_pred += fold_pred
        # Divide by number of folds to get the average
        pred = ensembled_pred / len(prob_maps)
        # We do not binarize the output, since we want the soft segmentation
    else:
        # We sum the elements of the list to get the ensembled output
        ensembled_pred = np.sum(pred, axis=0)
        # Divide by number of folds to get the average
        pred = ensembled_pred / len(pred)
        # Binarize the ensembled output at a low threshold to get the final segmentation
        pred = binarize(pred, 0.5)
    return pred


def segment_nnunet(path_img, tmpdir, predictor, device: torch.device, ensemble=False, soft_seg=False,
                   crop=False, crop_pad=None, box_overrides=None, out_fname=None):
    """
    This script is used to run inference on a single subject using a nnUNetV2 model.
    For soft segmentation (of MS lesions, the spinal cord, etc.), set `soft_seg=True`.
    If `crop=True`, the spinal cord is detected and the image cropped before inference (sc-crop), then the
    prediction is restored to the full image space.

    Author: Jan Valosek, Naga Karthik
    Original script: https://github.com/ivadomed/model_seg_sci/blob/4184bc22ef7317b3de5f85dee28449d6f381c984/packaging/run_inference_single_subject.py

    TODO: Find a less brittle way to specify model-based parameters such as model orientation, suffix, etc.
    """
    # NB: `device` should already be set when the `predictor` is initialized. We only have the `device` parameter here
    #     to match the function signature of `segment_monai`, which *does* require `device` at inference time.
    if device != predictor.device:
        logger.warning(f"Param `device` (value: {device}) is ignored in favor of `predictor.device` (value: "
                       f"{predictor.device}). To change the device, please modify the initialization of the predictor.")

    # Copy the file to the temporary directory using shutil.copyfile
    path_img_tmp = os.path.join(tmpdir, os.path.basename(path_img))
    shutil.copyfile(path_img, path_img_tmp)
    logger.info(f'Copied {path_img} to {path_img_tmp}')

    # sc-crop: detect the SC and crop before inference; bbox is used to uncrop after.
    # Cropping is done here in the original orientation (sc-crop is orientation-agnostic),
    # before the reorientation step below (which preserves the orientation after cropping).
    if crop:
        img_nii = nib.load(path_img_tmp)
        box_overrides = box_overrides or {}
        box_faces = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        if all(face in box_overrides for face in box_faces):
            # All 6 faces given manually: skip detection entirely rather than running it just
            # to immediately overwrite every value it would have produced.
            bbox = {face: box_overrides[face] for face in box_faces}
            bbox["original_axcodes"] = "".join(nib.aff2axcodes(img_nii.affine))
            bbox["_original_img"] = img_nii
        else:
            try:
                bbox = sc_crop.detect(img_nii, **(crop_pad or {}))
            except RuntimeError:
                # Detection found no cord at all, so there's no bbox to patch with box_overrides.
                # Fall back to the full image instead, letting box_overrides (if any) restrict
                # only the face(s) the user actually specified.
                if not box_overrides:
                    raise
                shape = img_nii.shape
                bbox = {
                    "xmin": 0, "xmax": shape[0] - 1,
                    "ymin": 0, "ymax": shape[1] - 1,
                    "zmin": 0, "zmax": shape[2] - 1,
                    "original_axcodes": "".join(nib.aff2axcodes(img_nii.affine)),
                    "_original_img": img_nii,
                }
            bbox.update(box_overrides)
        if not (bbox["xmax"] >= bbox["xmin"] and bbox["ymax"] >= bbox["ymin"] and bbox["zmax"] >= bbox["zmin"]):
            raise ValueError("sc-crop: invalid bounding box — check that -box-* values are not inverted.")
        fname_cropbox = add_suffix(out_fname if out_fname else path_img, "_cropbox")
        # If -o points to a not-yet-existing subdirectory, create it -- sct_deepseg.py does the
        # same for the segmentation output, but only after this function returns, which is too
        # late for the cropbox file (saved here, earlier in the pipeline).
        path_out = os.path.dirname(fname_cropbox)
        if path_out and not os.path.exists(path_out):
            os.makedirs(path_out)
        sc_crop.save_bbox_nifti(bbox, img_nii, fname_cropbox)
        nib.save(sc_crop.crop(img_nii, bbox), path_img_tmp)

    # Get the original orientation of the image, for example LPI
    orig_orientation = get_orientation(Image(path_img_tmp))

    # Get the orientation used by the model
    # Check if predictor.dataset_json['image_orientation'] key exists, if so, read the orientation from there
    # NOTE: the 'image_orientation' key-value pair needs to be manually added to the dataset.json file
    if 'image_orientation' in predictor.dataset_json:
        model_orientation = predictor.dataset_json['image_orientation']
        print(f"Orientation (based on dataset.json): {model_orientation}")
    else:
        if "RegionBasedLesionSeg" in predictor.plans_manager.dataset_name:
            model_orientation = "AIL"
        else:
            model_orientation = "LPI"

    # Reorient the image to model orientation if not already
    img_in = Image(path_img_tmp)
    if orig_orientation != model_orientation:
        logger.info(f'Changing orientation of the input to the model orientation ({model_orientation})...')
        img_in.change_orientation(model_orientation)

    # Create directory for nnUNet prediction
    tmpdir_nnunet = os.path.join(tmpdir, 'nnUNet_prediction')
    fname_prediction = os.path.join(tmpdir_nnunet, os.path.basename(add_suffix(path_img_tmp, "_pred")))
    os.mkdir(tmpdir_nnunet)

    # Run nnUNet prediction
    print('Starting inference...')
    start = time.time()

    # NOTE: nnUNet loads `.nii.gz` images using SimpleITK. When working with SimpleITK images, the axes are [x,y,z].
    #       But, during training, when nnUNet fetches a numpy array from the SimpleITK image, the axes get swapped
    #       ([z,y,x]). Nibabel (the image processing library SCT uses internally) _doesn't_ have this axis-swapping
    #       behavior. So, when SCT fetches the numpy array, we have to swap axes [x] and [z] to mimic nnUNet's internal
    #       behavior. See also: https://github.com/MIC-DKFZ/nnUNet/issues/1951
    data = img_in.data.transpose([2, 1, 0])
    # We also need to add an axis and convert to float32 to match nnUNet's input expectations
    # (This would automatically be done by nnUNet if we were to predict from image files, rather than a npy array.)
    data = np.expand_dims(data, axis=0).astype(np.float32)
    pred = predictor.predict_single_npy_array(
        input_image=data,
        # The spacings also have to be reversed to match nnUNet's conventions.
        image_properties={'spacing': img_in.dim[6:3:-1]},
        # Save the probability maps if specified
        save_or_return_probabilities=soft_seg,
        # If using a model ensemble, return the logits per fold so we can average them ourselves
        return_logits_per_fold=True if ensemble else False
    )
    # For the lesion_ms model, `pred` is a list of np.arrays, one per fold and needs averaging
    if ensemble:
        pred = average_nnunet_predictions(pred, probabilities=soft_seg)
    elif soft_seg:
        # `save_or_return_probabilities=True` without ensembling (i.e. a single, non-multi-fold model like
        # the contrast-agnostic SC model) returns a `(segmentation, probabilities)` tuple. Keep the value only
        # where the foreground probability beats the background one, matching the soft-value logic used by
        # `average_nnunet_predictions` for per-fold lesion_ms predictions.
        _, prob_map = pred
        pred = prediction_from_probability_map(prob_map)
    # Lastly, we undo the transpose to return the image from [z,y,x] (SimpleITK) to [x,y,z] (nibabel)
    pred = pred.transpose([2, 1, 0])
    img_out = img_in.copy()
    img_out.data = pred

    end = time.time()
    print('Inference done.')
    total_time = end - start
    print(f'Total inference time: {int(total_time // 60)} minute(s) {int(round(total_time % 60))} seconds')

    logger.info('Reorienting the prediction back to original orientation...')
    # Reorient the image back to original orientation
    if orig_orientation != model_orientation:
        img_out.change_orientation(orig_orientation)
        logger.info(f'Reorientation to original orientation {orig_orientation} done.')

    # sc-crop: restore the prediction to the full image space.
    if crop:
        seg_dtype = np.uint8 if not soft_seg else np.asanyarray(img_out.data).dtype
        seg_full = sc_crop.uncrop(nib.Nifti1Image(np.asanyarray(img_out.data).astype(seg_dtype), img_out.affine), bbox)
        truncated = sc_crop.check_seg_truncation(seg_full, bbox)
        if truncated:
            suggestions = "  ".join(f"-box-{key} <value>" for key in truncated)
            logger.warning(stylize(
                f"\nWARNING: the segmentation reaches the crop box at: {', '.join(truncated)}.\n"
                f"The spinal cord is likely truncated by the crop.", ["Red", "Bold"]))
            logger.warning(stylize(
                f"Open *_cropbox.nii.gz in FSLeyes, read off the voxel index of the truncated "
                f"face(s), and re-run with: {suggestions}\n", "Green"))
        img_out = Image(np.asanyarray(seg_full.dataobj), hdr=seg_full.header)

    labels = {k: v for k, v in predictor.dataset_json['labels'].items() if k != 'background'}
    # for the canal model, keep only the largest object
    if 'canal_seg' in labels.keys():
        logger.info('Keeping only the largest component.')
        img_out.data = keep_largest_object(img_out.data, x_cOm=None, y_cOm=None)
    # for rootlets model (which has labels 'lvl1', 'lvl2', etc.), save the image directly without splitting
    is_rootlet_model = all((label == f"lvl{i}") for i, label in enumerate(labels.keys(), start=1))
    if is_rootlet_model:
        targets = ["_rootlets"]
        outputs = [img_out]
    # for spinal cord segmentation models with only 1 output label, save the image directly with specific "_seg" suffix
    # this is added to preserve the typical expected output for SC models (`sct_propseg`, `sct_deepseg_sc`, etc.)
    # see also: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4805
    elif sorted(labels.keys()) == ['sc']:
        targets = ["_seg"]
        outputs = [img_out]
    # in the case of a multiclass soft-segmentation model (e.g. lesion_ms), we don't want the per-label
    # binarization done afterwards
    elif soft_seg:
        targets = ["_softseg"]
        outputs = [img_out]
    # for the other multiclass models (SCI lesion/SC, mouse GM/WM, etc.), save 1 image per label
    else:
        targets, outputs = [], []
        for label, label_values in labels.items():
            # Binarize data array by matching array values with label values
            # e.g. if `label_values == [1, 2]`, then for the data array, 0 -> False and 1,2 -> True
            # This handles nested labels, e.g. when the SC includes both lesion labels (2) and cord label (1)
            # Numpy syntax reference: https://stackoverflow.com/a/20528566
            label_values = label_values if isinstance(label_values, list) else [label_values]      # convert to list
            bool_array = np.logical_or.reduce([img_out.data == int(val) for val in label_values])  # 'OR' each label val
            img_bin = Image(bool_array.astype(np.uint8), hdr=img_out.hdr)
            target = f"_{label}_seg"
            targets.append(target)
            outputs.append(img_bin)

    # Save each image using the suffixes determined above
    fnames_out = []
    for target, img_out in zip(targets, outputs):
        fname_out = add_suffix(fname_prediction, target)
        logger.info(f"Saving results to: {fname_out}")
        img_out.save(fname_out)
        fnames_out.append(fname_out)

    return fnames_out, targets


def _reset_tqdm_mp_lock():
    """
    Discard tqdm's cached multiprocessing lock so that the next progress bar creates a new one.

    Since Python 3.11, passing a lock created in a 'fork' context (default on Linux) to a 'spawn' worker (used by
    totalspineseg) will raise an error with the message:

       "A SemLock created in a fork context is being shared with a process in a spawn context".

    Importantly, tqdm creates one `multiprocessing.RLock` per process, caches it on its classes, and passes it to
    every worker started by `tqdm.contrib.concurrent.process_map`. This means that if we ever create a 'fork' tqdm progress
    bar and then later in the same process we do totalspineseg inference (e.g. running our full test suite), then we'll
    run into the aforementioned exception. So, we need a way to reset the lock type so that we can recreate it later.
    """
    from tqdm import tqdm
    from tqdm.std import TqdmDefaultWriteLock  # noqa: Not exposed via __all__, so we're accessing TQDM internals here
    if 'mp_lock' in vars(TqdmDefaultWriteLock):
        del TqdmDefaultWriteLock.mp_lock
    # Iterate over `tqdm` and all of its subclasses to check for cached copies of the multiprocessing lock
    classes = [tqdm]
    while classes:
        cls = classes.pop()
        if '_lock' in vars(cls):
            delattr(cls, '_lock')
        classes.extend(cls.__subclasses__())


@contextmanager
def _spawn_start_method():
    """
    Temporarily use the 'spawn' multiprocessing start method (e.g. for totalspineseg).

    This context manager fixes https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4690#discussion_r1854731025.
    tl;dr: totalspineseg runs each of its processing steps in a process pool. With 'fork' (the default on Linux),
    these workers can deadlock once the parent process has started threads (e.g. after running PyTorch/nnUNet inference),
    which shows up as `sct_deepseg` stalling indefinitely partway through totalspineseg.

    Notes:
        - This context manager replaces our old approach from that PR where we set the 'spawn' method globally
          whenever we invoked TSS. This avoids leaking that change to any further steps within the Python process.
        - The calls to `_reset_tqdm_mp_lock()` are needed to prevent re-using a 'fork' tqdm lock in a 'spawn' process
        - FIXME: This would not be needed if totalspineseg ran its steps in-process when `max_workers=1`.
    """
    previous_method = mp.get_start_method(allow_none=True)  # None if the start method hasn't been fixed yet
    mp.set_start_method("spawn", force=True)
    _reset_tqdm_mp_lock()
    try:
        yield
    finally:
        mp.set_start_method(previous_method, force=True)  # `None` restores the platform default
        _reset_tqdm_mp_lock()


def segment_totalspineseg(path_img, tmpdir, predictor, device, label_vert=False):
    # for totalspineseg, the 'predictor' is just the model path
    path_model = predictor
    # fetch the release subdirectory from the model path
    installed_releases = sorted(
        os.path.basename(release_path) for release_path in
        glob.glob(os.path.join(path_model, 'nnUNet', 'results', 'r*'), recursive=True)
    )
    # There should always be a release subdirectory, hence the 'assert'
    assert installed_releases, f"No 'nnUNet/results/rYYYYMMDD' subdirectory found in {path_model}"

    # Copy the file to the temporary directory using shutil.copyfile
    path_img_tmp = os.path.join(tmpdir, os.path.basename(path_img))
    shutil.copyfile(path_img, path_img_tmp)
    logger.info(f'Copied {path_img} to {path_img_tmp}')

    # Create directory for nnUNet prediction
    tmpdir_nnunet = os.path.join(tmpdir, 'nnUNet_prediction')
    os.mkdir(tmpdir_nnunet)

    # totalspineseg uses multiprocessing internally, so here we try to
    # set multiprocessing method to 'spawn' to avoid deadlock issues
    # - 'spawn' is already default on Windows/macOS, but 'fork' is default on Linux
    # - 'spawn' is reliable though slower, but 'fork' causes intermittent stalling
    with _spawn_start_method():
        tss_inference(
            input_path=path_img_tmp,
            output_path=tmpdir_nnunet,
            data_path=path_model,
            # totalspineseg requires explicitly specifying the release subdirectory
            default_release=installed_releases[-1],  # use the most recent release
            # totalspineseg expects the device type, not torch.device
            device=device,
            # Try to address stalling due to the use of concurrent.futures in totalspineseg
            max_workers=1,
            max_workers_nnunet=1,
            # Optional argument to choose which models to run
            step1_only=bool(not label_vert),
            keep_only=['step1_levels', 'step1_output'] if not label_vert else ['step1_levels', 'step2_output'],
        )

    name_remapping = {
        "step1_levels": "totalspineseg_discs",
        "step1_output": "totalspineseg_all",
        "step2_output": "totalspineseg_all",
    }
    fnames_out, targets = [], []
    expected_outputs = [
        "step1_levels",
        "step2_output" if label_vert else "step1_output",
    ]
    for output_dirname in expected_outputs:
        fnames_out.append(os.path.join(tmpdir_nnunet, output_dirname, os.path.basename(path_img)))
        targets.append(f"_{name_remapping[output_dirname]}")

    return fnames_out, targets
