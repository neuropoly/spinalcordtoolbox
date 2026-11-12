"""
API for functions used in the sct_deepseg_lesion CLI script

Copyright (c) 2022 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import os
import logging
import numpy as np

from scipy.interpolate import interp1d

from spinalcordtoolbox.image import Image, add_suffix, zeros_like, empty_like
from spinalcordtoolbox.deepseg_.onnx import onnx_inference
from spinalcordtoolbox.deepseg_.sc import find_centerline, crop_image_around_centerline, uncrop_image, _normalize_data
from spinalcordtoolbox import resampling
from spinalcordtoolbox.utils.fs import TempFolder
from spinalcordtoolbox.utils.sys import sct_dir_local_path, LazyLoader

nib = LazyLoader("nib", globals(), "nibabel")

logger = logging.getLogger(__name__)

BATCH_SIZE = 4
MODEL_LST = ['t2', 't2_ax', 't2s']


def apply_intensity_normalization_model(img, landmarks_lst):
    """Description: apply the learned intensity landmarks to the input image."""
    percent_decile_lst = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    vals = list(img)
    landmarks_lst_cur = np.percentile(vals, q=percent_decile_lst)

    # create linear mapping models for the percentile segments to the learned standard intensity space
    linear_mapping = interp1d(landmarks_lst_cur, landmarks_lst, bounds_error=False)

    # transform the input image intensity values
    output = linear_mapping(img)

    # treat image intensity values outside of the cut-off percentiles range separately
    below_mapping = exp_model(landmarks_lst_cur[:2], landmarks_lst[:2], landmarks_lst[0])
    output[img < landmarks_lst_cur[0]] = below_mapping(img[img < landmarks_lst_cur[0]])

    above_mapping = exp_model(landmarks_lst_cur[-3:-1], landmarks_lst[-3:-1], landmarks_lst[-1])
    output[img > landmarks_lst_cur[-1]] = above_mapping(img[img > landmarks_lst_cur[-1]])

    return output.astype(np.float32)


def exp_model(xs, ys, s2):
    """
    FIXME doc
    """
    x1, x2 = xs
    y1, y2 = ys
    m = (y2 - y1) / (x2 - x1)
    b = y1 - (m * x1)
    mu90 = x2

    # y2 = alpha + beta * exp(gamma * x)
    alpha = s2

    omega = m * mu90 - s2 + b
    beta = omega * np.exp(-m * mu90 * 1.0 / omega)

    gamma = m * 1.0 / omega

    return lambda x: alpha + beta * np.exp(gamma * x)


def apply_intensity_normalization(img, contrast):
    """Standardize the intensity range."""
    data2norm = img.data.astype(np.float32)

    dct_norm = {'t2': [0.000000, 136.832187, 312.158435, 448.968030, 568.657779, 696.671586, 859.221138, 1074.463414, 1373.289174, 1811.522669, 2611.000000],
                't2_ax': [0.000000, 112.195357, 291.611185, 446.727066, 581.103970, 702.979079, 833.318257, 1011.856313, 1268.801813, 1687.137075, 2611.000000],
                't2s': [0.000000, 123.246969, 226.422561, 338.361023, 532.341924, 788.693675, 1096.975553, 1407.979466, 1716.524530, 2079.788451, 2611.000000]}

    img_normalized = empty_like(img)
    img_normalized.data = apply_intensity_normalization_model(data2norm, dct_norm[contrast])
    return img_normalized


def segment_3d(model_fname, contrast_type, im):
    """Perform segmentation with 3D convolutions."""
    dct_patch_3d = {'t2': {'size': (48, 48, 48), 'mean': 871.309, 'std': 557.916},
                    't2_ax': {'size': (48, 48, 48), 'mean': 835.592, 'std': 528.386},
                    't2s': {'size': (48, 48, 48), 'mean': 1011.31, 'std': 678.985}}

    out_data = np.zeros(im.data.shape)

    # segment the spinal cord
    z_patch_size = dct_patch_3d[contrast_type]['size'][2]
    z_step_keep = list(range(0, im.data.shape[2], z_patch_size))
    for zz in z_step_keep:
        if zz == z_step_keep[-1]:  # deal with instances where the im.data.shape[2] % patch_size_z != 0
            patch_im = np.zeros(dct_patch_3d[contrast_type]['size'])
            z_patch_extracted = im.data.shape[2] - zz
            patch_im[:, :, :z_patch_extracted] = im.data[:, :, zz:]
        else:
            z_patch_extracted = z_patch_size
            patch_im = im.data[:, :, zz:z_patch_size + zz]

        if np.any(patch_im):  # Check if the patch is (not) empty, which could occur after a brain detection.
            patch_norm = _normalize_data(patch_im, dct_patch_3d[contrast_type]['mean'], dct_patch_3d[contrast_type]['std'])
            x = np.expand_dims(np.expand_dims(patch_norm, 0), 0).astype(np.float32)
            onnx_pred = onnx_inference(model_fname, x)
            pred_seg_th = (onnx_pred[0] > 0.1).astype(int)[0, 0, :, :, :]
            if zz == z_step_keep[-1]:
                out_data[:, :, zz:] = pred_seg_th[:, :, :z_patch_extracted]
            else:
                out_data[:, :, zz:z_patch_size + zz] = pred_seg_th

    out = zeros_like(im, dtype=np.uint8)
    out.data = out_data

    return out.copy()


def deep_segmentation_MSlesion(im_image, contrast_type, ctr_algo='svm', ctr_file=None, brain_bool=True,
                               remove_temp_files=1, verbose=1):
    """
    Segment lesions from MRI data.

    :param im_image: Image() object containing the lesions to segment
    :param contrast_type: Constrast of the image. Need to use one supported by the CNN models.
    :param ctr_algo: Algo to find the centerline. See sct_get_centerline
    :param ctr_file: Centerline or segmentation (optional)
    :param brain_bool: If brain if present or not in the image.
    :param remove_temp_files:
    :return:
    """

    # create temporary folder with intermediate results
    tmp_folder = TempFolder(basename="deepseg-lesion", verbose=verbose)
    tmp_folder_path = tmp_folder.get_path()
    if ctr_algo == 'file':  # if the ctr_file is provided
        tmp_folder.copy_from(ctr_file)
        file_ctr = os.path.basename(ctr_file)
    else:
        file_ctr = None
    tmp_folder.chdir()
    fname_in = im_image.absolutepath

    # re-orient image to RPI
    logger.info("Reorient the image to RPI, if necessary...")
    original_orientation = im_image.orientation
    # fname_orient = 'image_in_RPI.nii'
    im_image.change_orientation('RPI')

    input_resolution = im_image.dim[4:7]

    # Resample image to 0.5mm in plane
    im_image_res = \
        resampling.resample_nib(im_image, new_size=[0.5, 0.5, im_image.dim[6]], new_size_type='mm', interpolation='linear')

    fname_orient = 'image_in_RPI_res.nii'
    im_image_res.save(fname_orient)

    # find the spinal cord centerline - execute OptiC binary
    logger.info("\nFinding the spinal cord centerline...")
    contrast_type_ctr = contrast_type.split('_')[0]
    _, im_ctl, im_labels_viewer = find_centerline(algo=ctr_algo,
                                                  image_fname=fname_orient,
                                                  contrast_type=contrast_type_ctr,
                                                  brain_bool=brain_bool,
                                                  folder_output=tmp_folder_path,
                                                  remove_temp_files=remove_temp_files,
                                                  centerline_fname=file_ctr)
    if ctr_algo == 'file':
        im_ctl = \
            resampling.resample_nib(im_ctl, new_size=[0.5, 0.5, im_image.dim[6]], new_size_type='mm', interpolation='linear')

    # Save the centerline image file
    fname_res_ctr = add_suffix(fname_orient, '_ctr')
    im_ctl.save(fname_res_ctr)

    # crop image around the spinal cord centerline
    logger.info("\nCropping the image around the spinal cord...")
    crop_size = 48
    X_CROP_LST, Y_CROP_LST, Z_CROP_LST, im_crop_nii = crop_image_around_centerline(im_in=im_image_res,
                                                                                   ctr_in=im_ctl,
                                                                                   crop_size=crop_size)
    del im_ctl

    # normalize the intensity of the images
    logger.info("Normalizing the intensity...")
    im_norm_in = apply_intensity_normalization(img=im_crop_nii, contrast=contrast_type)
    del im_crop_nii

    # resample to 0.5mm isotropic
    fname_norm = add_suffix(fname_orient, '_norm')
    im_norm_in.save(fname_norm)
    fname_res3d = add_suffix(fname_norm, '_resampled3d')
    resampling.resample_file(fname_norm, fname_res3d, '0.5x0.5x0.5', 'mm', 'linear',
                             verbose=0)

    # segment data using 3D convolutions
    logger.info("\nSegmenting the MS lesions using deep learning on 3D patches...")
    segmentation_model_fname = sct_dir_local_path('data', 'deepseg_lesion_models',
                                                  f'{contrast_type}_lesion.onnx')
    fname_seg_crop_res = add_suffix(fname_res3d, '_lesionseg')
    im_res3d = Image(fname_res3d)
    seg_im = segment_3d(model_fname=segmentation_model_fname,
                        contrast_type=contrast_type,
                        im=im_res3d.copy())
    seg_im.save(fname_seg_crop_res)
    del im_res3d, seg_im

    # resample to the initial pz resolution
    fname_seg_res2d = add_suffix(fname_seg_crop_res, '_resampled2d')
    initial_2d_resolution = 'x'.join(['0.5', '0.5', str(input_resolution[2])])
    resampling.resample_file(fname_seg_crop_res, fname_seg_res2d, initial_2d_resolution,
                             'mm', 'linear', verbose=0)
    seg_crop = Image(fname_seg_res2d)

    # reconstruct the segmentation from the crop data
    logger.info("\nReassembling the image...")
    seg_uncrop_nii = uncrop_image(ref_in=im_image_res, data_crop=seg_crop.copy().data, x_crop_lst=X_CROP_LST,
                                  y_crop_lst=Y_CROP_LST, z_crop_lst=Z_CROP_LST)
    fname_seg_res_RPI = add_suffix(fname_in, '_res_RPI_seg')
    seg_uncrop_nii.save(fname_seg_res_RPI)
    del seg_crop

    # resample to initial resolution
    logger.info("Resampling the segmentation to the original image resolution...")
    initial_resolution = 'x'.join([str(input_resolution[0]), str(input_resolution[1]), str(input_resolution[2])])
    fname_seg_RPI = add_suffix(fname_in, '_RPI_seg')
    resampling.resample_file(fname_seg_res_RPI, fname_seg_RPI, initial_resolution,
                             'mm', 'linear', verbose=0)
    seg_initres_nii = Image(fname_seg_RPI)

    if ctr_algo == 'viewer':  # resample and reorient the viewer labels
        im_labels_viewer_nib = nib.Nifti1Image(im_labels_viewer.data, im_labels_viewer.hdr.get_best_affine())
        im_viewer_r_nib = resampling.resample_nib(im_labels_viewer_nib, new_size=input_resolution, new_size_type='mm',
                                                  interpolation='linear')
        im_viewer = Image(np.asanyarray(im_viewer_r_nib.dataobj), hdr=im_viewer_r_nib.header, orientation='RPI',
                          dim=im_viewer_r_nib.header.get_data_shape()).change_orientation(original_orientation)

    else:
        im_viewer = None

    if verbose == 2:
        resampling.resample_file(fname_res_ctr, fname_res_ctr, initial_resolution,
                                 'mm', 'linear', verbose=0)
        im_image_res_ctr_downsamp = Image(fname_res_ctr).change_orientation(original_orientation)
    else:
        im_image_res_ctr_downsamp = None

    # binarize the resampled image to remove interpolation effects
    logger.info("\nBinarizing the segmentation to avoid interpolation effects...")
    thr = 0.1
    seg_initres_nii.data[np.where(seg_initres_nii.data >= thr)] = 1
    seg_initres_nii.data[np.where(seg_initres_nii.data < thr)] = 0

    # change data type
    seg_initres_nii.change_type(np.uint8)

    # reorient to initial orientation
    logger.info("\nReorienting the segmentation to the original image orientation...")
    tmp_folder.chdir_undo()

    # remove temporary files
    if remove_temp_files:
        logger.info("\nRemove temporary files...")
        tmp_folder.cleanup()

    # reorient to initial orientation
    return seg_initres_nii.change_orientation(original_orientation), im_viewer, im_image_res_ctr_downsamp
