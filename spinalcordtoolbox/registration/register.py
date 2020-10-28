#!/usr/bin/env python
#########################################################################################
# Spinal Cord Registration module
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2020 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
#
# License: see the LICENSE.TXT
#########################################################################################

import logging
import os # FIXME
import shutil
from math import asin, cos, sin, acos

import numpy as np
from scipy import ndimage
from nibabel import load, Nifti1Image, save
from scipy.signal import argrelmax, medfilt
from sklearn.decomposition import PCA
from scipy.io import loadmat

import spinalcordtoolbox.image as image
from spinalcordtoolbox.math import laplacian
from spinalcordtoolbox.registration.landmarks import register_landmarks
from spinalcordtoolbox.utils import sct_progress_bar, copy_helper, run_proc, tmp_create

# TODO [AJ]
# introduce potential cleanup functions in case exceptions occur and
# filesystem is left with temp artefacts everywhere?

logger = logging.getLogger(__name__)


class Paramreg(object):
    def __init__(self, step=None, type=None, algo='syn', metric='MeanSquares', samplingStrategy='None',
                 samplingPercentage='0.2', iter='10', shrink='1', smooth='0', gradStep='0.5', deformation='1x1x0',
                 init='', filter_size=5, poly='5', slicewise='0', laplacian='0', dof='Tx_Ty_Tz_Rx_Ry_Rz',
                 smoothWarpXY='2', pca_eigenratio_th='1.6', rot_method='pca'):
        """
        Class to define registration method.

        :param step: int: Step number (starts at 1, except for type=label which corresponds to step=0).
        :param type: {im, seg, imseg, label} Type of data used for registration. Use type=label only at step=0.
        :param algo:
        :param metric:
        :param samplingStrategy: {'Regular', 'Random', 'None'}
        :param samplingPercentage: [0, 1]
        :param iter:
        :param shrink:
        :param smooth:
        :param gradStep:
        :param deformation:
        :param init:
        :param filter_size: int: Size of the Gaussian kernel when filtering the cord rotation estimate across z.
        :param poly:
        :param slicewise: {'0', '1'}: Slice-by-slice 2d transformation.
        :param laplacian:
        :param dof:
        :param smoothWarpXY:
        :param pca_eigenratio_th:
        :param rot_method: {'pca', 'hog', 'pcahog'}: Rotation method to be used with algo=centermassrot.
            pca: approximate cord segmentation by an ellipse and finds it orientation using PCA's
            eigenvectors; hog: finds the orientation using the symmetry of the image; pcahog: tries method pca and if it
            fails, uses method hog. If using hog or pcahog, type should be set to 'imseg'."
        """
        self.step = step
        self.type = type
        self.algo = algo
        self.metric = metric
        self.samplingStrategy = samplingStrategy
        self.samplingPercentage = samplingPercentage
        self.iter = iter
        self.shrink = shrink
        self.smooth = smooth
        self.laplacian = laplacian
        self.gradStep = gradStep
        self.deformation = deformation
        self.slicewise = slicewise
        self.init = init
        self.poly = poly  # only for algo=slicereg
        self.filter_size = filter_size  # only for algo=centermassrot
        self.dof = dof  # only for type=label
        self.smoothWarpXY = smoothWarpXY  # only for algo=columnwise
        self.pca_eigenratio_th = pca_eigenratio_th  # only for algo=centermassrot
        self.rot_method = rot_method  # only for algo=centermassrot
        self.rot_src = None  # this variable is used to set the angle of the cord on the src image if it is known
        self.rot_dest = None  # same as above for the destination image (e.g., if template, should be set to 0)

        # list of possible values for self.type
        self.type_list = ['im', 'seg', 'imseg', 'label']

    # update constructor with user's parameters
    def update(self, paramreg_user):
        list_objects = paramreg_user.split(',')
        for object in list_objects:
            if len(object) < 2:
                raise ValueError("Invalid use of -param! Check usage (usage changed from previous version)")
            obj = object.split('=')
            setattr(self, obj[0], obj[1])


class ParamregMultiStep:
    """
    Class to aggregate multiple Paramreg() classes into a dictionary. The method addStep() is used to build this class.
    """
    def __init__(self, listParam=[]):
        self.steps = dict()
        for stepParam in listParam:
            if isinstance(stepParam, Paramreg):
                self.steps[stepParam.step] = stepParam
            else:
                self.addStep(stepParam)

    def addStep(self, stepParam):
        """
        Checks if the step is already present.
        If it exists: update it.
        If not: add it.
        """
        param_reg = Paramreg()
        param_reg.update(stepParam)
        if param_reg.step is None:
            raise ValueError("Parameters must contain 'step'!")
        else:
            if param_reg.step in self.steps:
                self.steps[param_reg.step].update(stepParam)
            else:
                self.steps[param_reg.step] = param_reg
        if int(param_reg.step) != 0 and param_reg.type not in param_reg.type_list:
            raise ValueError("Parameters must contain a type, either 'im' or 'seg'")


def register_step_ants_slice_regularized_registration(src, dest, step, metricSize, fname_mask='', verbose=1):
    """
    """
    # Find the min (and max) z-slice index below which (and above which) slices only have voxels below a given
    # threshold.
    list_fname = [src, dest]
    if fname_mask:
        list_fname.append(fname_mask)
        mask_options = ['-x', fname_mask]
    else:
        mask_options = []

    zmin_global, zmax_global = 0, 99999  # this is assuming that typical image has less slice than 99999

    for fname in list_fname:
        im = image.Image(fname)
        zmin, zmax = image.find_zmin_zmax(im, threshold=0.1)
        if zmin > zmin_global:
            zmin_global = zmin
        if zmax < zmax_global:
            zmax_global = zmax

    # crop images (see issue #293)
    src_crop = image.add_suffix(src, '_crop')
    image.spatial_crop(image.Image(src), dict(((2, (zmin_global, zmax_global)),))).save(src_crop)
    dest_crop = image.add_suffix(dest, '_crop')
    image.spatial_crop(image.Image(dest), dict(((2, (zmin_global, zmax_global)),))).save(dest_crop)

    # update variables
    src = src_crop
    dest = dest_crop
    scr_regStep = image.add_suffix(src, '_regStep' + str(step.step))

    # estimate transfo
    cmd = ['isct_antsSliceRegularizedRegistration',
           '-t', 'Translation[' + step.gradStep + ']',
           '-m', step.metric + '['
           + ','.join([dest, src, '1', metricSize, step.samplingStrategy, step.samplingPercentage]) + ']',
           '-p', step.poly,
           '-i', step.iter,
           '-f', step.shrink,
           '-s', step.smooth,
           '-v', '1',  # verbose (verbose=2 does not exist, so we force it to 1)
           '-o', '[step' + str(step.step) + ',' + scr_regStep + ']',  # here the warp name is stage10 because
           # antsSliceReg add "Warp"
           ] + mask_options

    warp_forward_out = 'step' + str(step.step) + 'Warp.nii.gz'
    warp_inverse_out = 'step' + str(step.step) + 'InverseWarp.nii.gz'

    # run command
    status, output = run_proc(cmd, verbose, is_sct_binary=True)

    return warp_forward_out, warp_inverse_out


def register_step_ants_registration(src, dest, step, masking, ants_registration_params, padding, metricSize, verbose=1):
    """
    """
    # Pad the destination image (because ants doesn't deform the extremities)
    # N.B. no need to pad if iter = 0
    if not step.iter == '0':
        dest_pad = image.add_suffix(dest, '_pad')
        run_proc(['sct_image', '-i', dest, '-o', dest_pad, '-pad', '0,0,' + str(padding)])
        dest = dest_pad

    # apply Laplacian filter
    if not step.laplacian == '0':
        logger.info(f"\nApply Laplacian filter")

        sigmas = [step.laplacian, step.laplacian, 0]

        src_img = image.Image(src)
        src_out = src_img.copy()
        src_out.data = laplacian(src_out.data, sigmas)
        src_out.save(path=image.add_suffix(src, '_laplacian'))

        dest_img = image.Image(dest)
        dest_out = dest_img.copy()
        dest_out.data = laplacian(dest_out.data, sigmas)
        dest_out.save(path=image.add_suffix(dest, '_laplacian'))

        src = image.add_suffix(src, '_laplacian')
        dest = image.add_suffix(dest, '_laplacian')

    # Estimate transformation
    logger.info(f"\nEstimate transformation")
    scr_regStep = image.add_suffix(src, '_regStep' + str(step.step))

    cmd = ['isct_antsRegistration',
           '--dimensionality', '3',
           '--transform', step.algo + '[' + step.gradStep
           + ants_registration_params[step.algo.lower()] + ']',
           '--metric', step.metric + '[' + dest + ',' + src + ',1,' + metricSize + ']',
           '--convergence', step.iter,
           '--shrink-factors', step.shrink,
           '--smoothing-sigmas', step.smooth + 'mm',
           '--restrict-deformation', step.deformation,
           '--output', '[step' + str(step.step) + ',' + scr_regStep + ']',
           '--interpolation', 'BSpline[3]',
           '--verbose', '1',
           ] + masking

    # add init translation
    if step.init:
        init_dict = {'geometric': '0', 'centermass': '1', 'origin': '2'}
        cmd += ['-r', '[' + dest + ',' + src + ',' + init_dict[step.init] + ']']

    # run command
    status, output = run_proc(cmd, verbose, is_sct_binary=True)

    # get appropriate file name for transformation
    if step.algo in ['rigid', 'affine', 'translation']:
        warp_forward_out = 'step' + str(step.step) + '0GenericAffine.mat'
        warp_inverse_out = '-step' + str(step.step) + '0GenericAffine.mat'
    else:
        warp_forward_out = 'step' + str(step.step) + '0Warp.nii.gz'
        warp_inverse_out = 'step' + str(step.step) + '0InverseWarp.nii.gz'

    return warp_forward_out, warp_inverse_out


def register_step_slicewise_ants(src, dest, step, ants_registration_params, fname_mask, remove_temp_files, verbose=1):
    """
    """
    # if shrink!=1, force it to be 1 (otherwise, it generates a wrong 3d warping field). TODO: fix that!
    if not step.shrink == '1':
        logger.warning(f"\nWhen using slicewise with SyN or BSplineSyN, shrink factor needs to be one. Forcing shrink=1")
        step.shrink = '1'

    warp_forward_out = 'step' + str(step.step) + 'Warp.nii.gz'
    warp_inverse_out = 'step' + str(step.step) + 'InverseWarp.nii.gz'

    register_slicewise(
     fname_src=src,
     fname_dest=dest,
     paramreg=step,
     fname_mask=fname_mask,
     warp_forward_out=warp_forward_out,
     warp_inverse_out=warp_inverse_out,
     ants_registration_params=ants_registration_params,
     remove_temp_files=remove_temp_files,
     verbose=verbose
    )

    return warp_forward_out, warp_inverse_out


def register_step_slicewise(src, dest, step, ants_registration_params, remove_temp_files, verbose=1):
    """
    """
    # smooth data
    if not step.smooth == '0':
        logger.warning(f"\nAlgo {step.algo} will ignore the parameter smoothing.\n")

    warp_forward_out = 'step' + str(step.step) + 'Warp.nii.gz'
    warp_inverse_out = 'step' + str(step.step) + 'InverseWarp.nii.gz'

    register_slicewise(
     fname_src=src,
     fname_dest=dest,
     paramreg=step,
     fname_mask='',
     warp_forward_out=warp_forward_out,
     warp_inverse_out=warp_inverse_out,
     ants_registration_params=ants_registration_params,
     remove_temp_files=remove_temp_files,
     verbose=verbose
    )

    return warp_forward_out, warp_inverse_out


def register_step_label(src, dest, step, verbose=1):
    """
    """
    warp_forward_out = 'step' + step.step + '0GenericAffine.txt'
    warp_inverse_out = '-step' + step.step + '0GenericAffine.txt'

    register_landmarks(src,
                       dest,
                       step.dof,
                       fname_affine=warp_forward_out,
                       verbose=verbose)

    return warp_forward_out, warp_inverse_out


def register_slicewise(fname_src, fname_dest, paramreg=None, fname_mask='', warp_forward_out='step0Warp.nii.gz',
                       warp_inverse_out='step0InverseWarp.nii.gz', ants_registration_params=None,
                       path_qc='./', remove_temp_files=0, verbose=0):
    """
    Main function that calls various methods for slicewise registration.

    :param fname_src: Str or List: If List, first element is image, second element is segmentation.
    :param fname_dest: Str or List: If List, first element is image, second element is segmentation.
    :param paramreg: Class Paramreg()
    :param fname_mask:
    :param warp_forward_out:
    :param warp_inverse_out:
    :param ants_registration_params:
    :param path_qc:
    :param remove_temp_files:
    :param verbose:
    :return:
    """

    # create temporary folder
    path_tmp = tmp_create(basename="register")

    # copy data to temp folder
    logger.info(f"\nCopy input data to temp folder...")
    if isinstance(fname_src, list):
        # TODO: swap 0 and 1 (to be consistent with the child function below)
        src_img = image.convert(image.Image(fname_src[0]))
        src_img.save(os.path.join(path_tmp, "src.nii"))

        src_seg = image.convert(image.Image(fname_src[1]))
        src_seg.save(os.path.join(path_tmp, "src_seg.nii"))

        dest_img = image.convert(image.Image(fname_dest[0]))
        dest_img.save(os.path.join(path_tmp, "dest.nii"))

        dest_seg = image.convert(image.Image(fname_dest[1]))
        dest_seg.save(os.path.join(path_tmp, "dest_seg.nii"))
    else:
        src_img = image.convert(image.Image(fname_src))
        src_img.save(os.path.join(path_tmp, "src.nii"))

        dest_image = image.convert(image.Image(fname_dest))
        dest_image.save(os.path.join(path_tmp, "dest.nii"))

    if fname_mask != '':
        image.convert(fname_mask, os.path.join(path_tmp, "mask.nii.gz"))

    # go to temporary folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Calculate displacement
    if paramreg.algo in ['centermass', 'centermassrot']:
        # translation of center of mass between source and destination in voxel space
        if paramreg.algo in 'centermass':
            rot_method = 'none'
        else:
            rot_method = paramreg.rot_method
        if rot_method in ['hog', 'pcahog']:
            src_input = ['src_seg.nii', 'src.nii']
            dest_input = ['dest_seg.nii', 'dest.nii']
        else:
            src_input = ['src.nii']
            dest_input = ['dest.nii']
        register2d_centermassrot(
            src_input, dest_input, paramreg=paramreg, fname_warp=warp_forward_out, fname_warp_inv=warp_inverse_out,
            rot_method=rot_method, filter_size=paramreg.filter_size, path_qc=path_qc, verbose=verbose,
            pca_eigenratio_th=float(paramreg.pca_eigenratio_th), )

    elif paramreg.algo == 'columnwise':
        # scaling R-L, then column-wise center of mass alignment and scaling
        register2d_columnwise('src.nii',
                              'dest.nii',
                              fname_warp=warp_forward_out,
                              fname_warp_inv=warp_inverse_out,
                              verbose=verbose,
                              path_qc=path_qc,
                              smoothWarpXY=int(paramreg.smoothWarpXY),
                              )

    # ANTs registration
    else:
        # convert SCT flags into ANTs-compatible flags
        algo_dic = {'translation': 'Translation', 'rigid': 'Rigid', 'affine': 'Affine', 'syn': 'SyN', 'bsplinesyn': 'BSplineSyN', 'centermass': 'centermass'}
        paramreg.algo = algo_dic[paramreg.algo]
        # run slicewise registration
        register2d('src.nii',
                   'dest.nii',
                   fname_mask=fname_mask,
                   fname_warp=warp_forward_out,
                   fname_warp_inv=warp_inverse_out,
                   paramreg=paramreg,
                   ants_registration_params=ants_registration_params,
                   verbose=verbose,
                   )

    logger.info(f"\nMove warping fields...")
    copy_helper(warp_forward_out, curdir)
    copy_helper(warp_inverse_out, curdir)

    # go back
    os.chdir(curdir)

    if remove_temp_files:
        logger.info(f"rm -rf {path_tmp}")
        shutil.rmtree(path_tmp)


def register2d_centermassrot(fname_src, fname_dest, paramreg=None, fname_warp='warp_forward.nii.gz',
                             fname_warp_inv='warp_inverse.nii.gz', rot_method='pca', filter_size=0, path_qc='./',
                             verbose=0, pca_eigenratio_th=1.6, th_max_angle=40):
    """
    Rotate the source image to match the orientation of the destination image, using the first and second eigenvector
    of the PCA. This function should be used on segmentations (not images).
    This works for 2D and 3D images.  If 3D, it splits the image and performs the rotation slice-by-slice.

    :param fname_src: List: Name of moving image. If rot=0 or 1, only the first element is used (should be a
        segmentation). If rot=2 or 3, the first element is a segmentation and the second is an image.
    :param fname_dest: List: Name of fixed image. If rot=0 or 1, only the first element is used (should be a
        segmentation). If rot=2 or 3, the first element is a segmentation and the second is an image.
    :param paramreg: Class Paramreg()
    :param fname_warp: name of output 3d forward warping field
    :param fname_warp_inv: name of output 3d inverse warping field
    :param rot_method: {'none', 'pca', 'hog', 'pcahog'}. Depending on the rotation method, input might be segmentation
        only or segmentation and image.
    :param filter_size: size of the gaussian filter for regularization along z for rotation angle (type: float).
        0: no regularization
    :param path_qc:
    :param verbose:
    :param pca_eigenratio_th: threshold for the ratio between the first and second eigenvector of the estimated ellipse
        for the PCA rotation detection method. If below this threshold, the estimation will be discarded (poorly robust)
    :param th_max_angle: threshold of the absolute value of the estimated rotation using the PCA method, above
        which the estimation will be discarded (unlikely to happen genuinely and hence considered outlier)
    :return:
    """
    # TODO: no need to split the src or dest if it is the template (we know its centerline and orientation already)

    if verbose == 2:
        import matplotlib
        matplotlib.use('Agg')  # prevent display figure
        import matplotlib.pyplot as plt

    # Get image dimensions and retrieve nz
    logger.info(f"\nGet image dimensions of destination image...")
    nx, ny, nz, nt, px, py, pz, pt = image.Image(fname_dest[0]).dim

    logger.info(f"  matrix size: {str(nx)} x {str(ny)} x {str(nz)}")
    logger.info(f"  voxel size: {str(px)}mm x {str(py)}mm x {str(nz)}mm")

    # Split source volume along z
    logger.info(f"\nSplit input segmentation...")
    im_src = image.Image(fname_src[0])
    split_source_list = image.split_img_data(im_src, 2)
    for im in split_source_list:
        im.save()

    # Split destination volume along z
    logger.info(f"\nSplit destination segmentation...")
    im_dest = image.Image(fname_dest[0])
    split_dest_list = image.split_img_data(im_dest, 2)
    for im in split_dest_list:
        im.save()

    data_src = im_src.data
    data_dest = im_dest.data

    # if input data is 2D, reshape into pseudo 3D (only one slice)
    if len(data_src.shape) == 2:
        new_shape = list(data_src.shape)
        new_shape.append(1)
        new_shape = tuple(new_shape)
        data_src = data_src.reshape(new_shape)
        data_dest = data_dest.reshape(new_shape)

    # Deal with cases where both an image and segmentation are input
    if len(fname_src) > 1:
        # Split source volume along z
        logger.info(f"\nSplit input image...")
        im_src_im = image.Image(fname_src[1])
        split_source_list = image.split_img_data(im_src_im, 2)
        for im in split_source_list:
            im.save()

        # Split destination volume along z
        logger.info(f"\nSplit destination image...")
        im_dest_im = image.Image(fname_dest[1])
        split_dest_list = image.split_img_data(im_dest_im, 2)
        for im in split_dest_list:
            im.save()

        data_src_im = im_src_im.data
        data_dest_im = im_dest_im.data

    # initialize displacement and rotation
    coord_src = [None] * nz
    pca_src = [None] * nz
    coord_dest = [None] * nz
    pca_dest = [None] * nz
    centermass_src = np.zeros([nz, 2])
    centermass_dest = np.zeros([nz, 2])
    # displacement_forward = np.zeros([nz, 2])
    # displacement_inverse = np.zeros([nz, 2])
    angle_src_dest = np.zeros(nz)
    z_nonzero = []
    th_max_angle *= np.pi / 180

    # Loop across slices
    for iz in sct_progress_bar(range(0, nz), unit='iter', unit_scale=False, desc="Estimate cord angle for each slice",
                               ascii=False, ncols=100):
        try:
            # compute PCA and get center or mass based on segmentation
            coord_src[iz], pca_src[iz], centermass_src[iz, :] = compute_pca(data_src[:, :, iz])
            coord_dest[iz], pca_dest[iz], centermass_dest[iz, :] = compute_pca(data_dest[:, :, iz])

            # detect rotation using the HOG method
            if rot_method in ['hog', 'pcahog']:
                angle_src_hog, conf_score_src = find_angle_hog(data_src_im[:, :, iz], centermass_src[iz, :],
                                                               px, py, angle_range=th_max_angle)
                angle_dest_hog, conf_score_dest = find_angle_hog(data_dest_im[:, :, iz], centermass_dest[ iz, : ],
                                                                 px, py, angle_range=th_max_angle)
                # In case no maxima is found (it should never happen)
                if (angle_src_hog is None) or (angle_dest_hog is None):
                    logger.warning(f"Slice #{str(iz)} not angle found in dest or src. It will be ignored.")
                    continue
                if rot_method == 'hog':
                    angle_src = -angle_src_hog  # flip sign to be consistent with PCA output
                    angle_dest = angle_dest_hog

            # Detect rotation using the PCA or PCA-HOG method
            if rot_method in ['pca', 'pcahog']:
                eigenv_src = pca_src[iz].components_.T[0][0], pca_src[iz].components_.T[1][0]
                eigenv_dest = pca_dest[iz].components_.T[0][0], pca_dest[iz].components_.T[1][0]
                # Make sure first element is always positive (to prevent sign flipping)
                if eigenv_src[0] <= 0:
                    eigenv_src = tuple([i * (-1) for i in eigenv_src])
                if eigenv_dest[0] <= 0:
                    eigenv_dest = tuple([i * (-1) for i in eigenv_dest])
                angle_src = angle_between(eigenv_src, [1, 0])
                angle_dest = angle_between([1, 0], eigenv_dest)
                # compute ratio between axis of PCA
                pca_eigenratio_src = pca_src[iz].explained_variance_ratio_[0] / pca_src[iz].explained_variance_ratio_[1]
                pca_eigenratio_dest = pca_dest[iz].explained_variance_ratio_[0] / pca_dest[iz].explained_variance_ratio_[1]
                # angle is set to 0 if either ratio between axis is too low or outside angle range
                if pca_eigenratio_src < pca_eigenratio_th or angle_src > th_max_angle or angle_src < -th_max_angle:
                    if rot_method == 'pca':
                        angle_src = 0
                    elif rot_method == 'pcahog':
                        logger.info("Switched to method 'hog' for slice: {}".format(iz))
                        angle_src = -angle_src_hog  # flip sign to be consistent with PCA output
                if pca_eigenratio_dest < pca_eigenratio_th or angle_dest > th_max_angle or angle_dest < -th_max_angle:
                    if rot_method == 'pca':
                        angle_dest = 0
                    elif rot_method == 'pcahog':
                        logger.info("Switched to method 'hog' for slice: {}".format(iz))
                        angle_dest = angle_dest_hog

            if not rot_method == 'none':
                # bypass estimation is source or destination angle is known a priori
                if paramreg.rot_src is not None:
                    angle_src = paramreg.rot_src
                if paramreg.rot_dest is not None:
                    angle_dest = paramreg.rot_dest
                # the angle between (src, dest) is the angle between (src, origin) + angle between (origin, dest)
                angle_src_dest[iz] = angle_src + angle_dest

            # append to list of z_nonzero
            z_nonzero.append(iz)

        # if one of the slice is empty, ignore it
        except ValueError:
            logger.warning(f"Slice #{str(iz)} is empty. It will be ignored.")

    # regularize rotation
    if not filter_size == 0 and (rot_method in ['pca', 'hog', 'pcahog']):
        # Filtering the angles by gaussian filter
        angle_src_dest_regularized = ndimage.filters.gaussian_filter1d(angle_src_dest[z_nonzero], filter_size)
        if verbose == 2:
            plt.plot(180 * angle_src_dest[z_nonzero] / np.pi, 'ob')
            plt.plot(180 * angle_src_dest_regularized / np.pi, 'r', linewidth=2)
            plt.grid()
            plt.xlabel('z')
            plt.ylabel('Angle (deg)')
            plt.title("Regularized cord angle estimation (filter_size: {})".format(filter_size))
            plt.savefig(os.path.join(path_qc, 'register2d_centermassrot_regularize_rotation.png'))
            plt.close()
        # update variable
        angle_src_dest[z_nonzero] = angle_src_dest_regularized

    warp_x = np.zeros(data_dest.shape)
    warp_y = np.zeros(data_dest.shape)
    warp_inv_x = np.zeros(data_src.shape)
    warp_inv_y = np.zeros(data_src.shape)

    # construct 3D warping matrix
    for iz in sct_progress_bar(z_nonzero, unit='iter', unit_scale=False, desc="Build 3D deformation field",
                               ascii=False, ncols=100):
        # get indices of x and y coordinates
        row, col = np.indices((nx, ny))
        # build 2xn array of coordinates in pixel space
        coord_init_pix = np.array([row.ravel(), col.ravel(), np.array(np.ones(len(row.ravel())) * iz)]).T
        # convert coordinates to physical space
        coord_init_phy = np.array(im_src.transfo_pix2phys(coord_init_pix))
        # get centermass coordinates in physical space
        centermass_src_phy = im_src.transfo_pix2phys([[centermass_src[iz, :].T[0], centermass_src[iz, :].T[1], iz]])[0]
        centermass_dest_phy = im_src.transfo_pix2phys([[centermass_dest[iz, :].T[0], centermass_dest[iz, :].T[1], iz]])[0]
        # build rotation matrix
        R = np.matrix(((cos(angle_src_dest[iz]), sin(angle_src_dest[iz])), (-sin(angle_src_dest[iz]), cos(angle_src_dest[iz]))))
        # build 3D rotation matrix
        R3d = np.eye(3)
        R3d[0:2, 0:2] = R
        # apply forward transformation (in physical space)
        coord_forward_phy = np.array(np.dot((coord_init_phy - np.transpose(centermass_dest_phy)), R3d) + np.transpose(centermass_src_phy))
        # apply inverse transformation (in physical space)
        coord_inverse_phy = np.array(np.dot((coord_init_phy - np.transpose(centermass_src_phy)), R3d.T) + np.transpose(centermass_dest_phy))
        # display rotations
        if verbose == 2 and not angle_src_dest[iz] == 0 and not rot_method == 'hog':
            # compute new coordinates
            coord_src_rot = coord_src[iz] * R
            coord_dest_rot = coord_dest[iz] * R.T
            # generate figure
            plt.figure(figsize=(9, 9))
            # plt.ion()  # enables interactive mode (allows keyboard interruption)
            for isub in [221, 222, 223, 224]:
                # plt.figure
                plt.subplot(isub)
                # ax = matplotlib.pyplot.axis()
                try:
                    if isub == 221:
                        plt.scatter(coord_src[iz][:, 0], coord_src[iz][:, 1], s=5, marker='o', zorder=10, color='steelblue',
                                    alpha=0.5)
                        pcaaxis = pca_src[iz].components_.T
                        pca_eigenratio = pca_src[iz].explained_variance_ratio_
                        plt.title('src')
                    elif isub == 222:
                        plt.scatter([coord_src_rot[i, 0] for i in range(len(coord_src_rot))], [coord_src_rot[i, 1] for i in range(len(coord_src_rot))], s=5, marker='o', zorder=10, color='steelblue', alpha=0.5)
                        pcaaxis = pca_dest[iz].components_.T
                        pca_eigenratio = pca_dest[iz].explained_variance_ratio_
                        plt.title('src_rot')
                    elif isub == 223:
                        plt.scatter(coord_dest[iz][:, 0], coord_dest[iz][:, 1], s=5, marker='o', zorder=10, color='red',
                                    alpha=0.5)
                        pcaaxis = pca_dest[iz].components_.T
                        pca_eigenratio = pca_dest[iz].explained_variance_ratio_
                        plt.title('dest')
                    elif isub == 224:
                        plt.scatter([coord_dest_rot[i, 0] for i in range(len(coord_dest_rot))], [coord_dest_rot[i, 1] for i in range(len(coord_dest_rot))], s=5, marker='o', zorder=10, color='red', alpha=0.5)
                        pcaaxis = pca_src[iz].components_.T
                        pca_eigenratio = pca_src[iz].explained_variance_ratio_
                        plt.title('dest_rot')
                    plt.text(-2.5, -2, 'eigenvectors:', horizontalalignment='left', verticalalignment='bottom')
                    plt.text(-2.5, -2.8, str(pcaaxis), horizontalalignment='left', verticalalignment='bottom')
                    plt.text(-2.5, 2.5, 'eigenval_ratio:', horizontalalignment='left', verticalalignment='bottom')
                    plt.text(-2.5, 2, str(pca_eigenratio), horizontalalignment='left', verticalalignment='bottom')
                    plt.plot([0, pcaaxis[0, 0]], [0, pcaaxis[1, 0]], linewidth=2, color='red')
                    plt.plot([0, pcaaxis[0, 1]], [0, pcaaxis[1, 1]], linewidth=2, color='orange')
                    plt.axis([-3, 3, -3, 3])
                    plt.gca().set_aspect('equal', adjustable='box')
                except Exception as e:
                    raise Exception

            plt.savefig(os.path.join(path_qc, 'register2d_centermassrot_pca_z' + str(iz) + '.png'))
            plt.close()

        # construct 3D warping matrix
        warp_x[:, :, iz] = np.array([coord_forward_phy[i, 0] - coord_init_phy[i, 0] for i in range(nx * ny)]).reshape((nx, ny))
        warp_y[:, :, iz] = np.array([coord_forward_phy[i, 1] - coord_init_phy[i, 1] for i in range(nx * ny)]).reshape((nx, ny))
        warp_inv_x[:, :, iz] = np.array([coord_inverse_phy[i, 0] - coord_init_phy[i, 0] for i in range(nx * ny)]).reshape((nx, ny))
        warp_inv_y[:, :, iz] = np.array([coord_inverse_phy[i, 1] - coord_init_phy[i, 1] for i in range(nx * ny)]).reshape((nx, ny))

    # Generate forward warping field (defined in destination space)
    generate_warping_field(fname_dest[0], warp_x, warp_y, fname_warp, verbose)
    generate_warping_field(fname_src[0], warp_inv_x, warp_inv_y, fname_warp_inv, verbose)


def register2d_columnwise(fname_src, fname_dest, fname_warp='warp_forward.nii.gz', fname_warp_inv='warp_inverse.nii.gz', verbose=0, path_qc='./', smoothWarpXY=1):
    """
    Column-wise non-linear registration of segmentations. Based on an idea from Allan Martin.
    - Assumes src/dest are segmentations (not necessarily binary), and already registered by center of mass
    - Assumes src/dest are in RPI orientation.
    - Split along Z, then for each slice:
    - scale in R-L direction to match src/dest
    - loop across R-L columns and register by (i) matching center of mass and (ii) scaling.
    :param fname_src:
    :param fname_dest:
    :param fname_warp:
    :param fname_warp_inv:
    :param verbose:
    :return:
    """

    # initialization
    th_nonzero = 0.5  # values below are considered zero

    # for display stuff
    if verbose == 2:
        import matplotlib
        matplotlib.use('Agg')  # prevent display figure
        import matplotlib.pyplot as plt

    # Get image dimensions and retrieve nz
    logger.info(f"\nGet image dimensions of destination image...")
    nx, ny, nz, nt, px, py, pz, pt = image.Image(fname_dest).dim

    logger.info(f"  matrix size: {str(nx)} x {str(ny)} x {str(nz)}")
    logger.info(f"  voxel size: {str(px)}mm x {str(py)}mm x {str(nz)}mm")

    # Split source volume along z
    logger.info(f"\nSplit input volume...")
    im_src = image.Image('src.nii')
    split_source_list = image.split_img_data(im_src, 2)
    for im in split_source_list:
        im.save()

    # Split destination volume along z
    logger.info(f"\nSplit destination volume...")
    im_dest = image.Image('dest.nii')
    split_dest_list = image.split_img_data(im_dest, 2)
    for im in split_dest_list:
        im.save()

    # open image
    data_src = im_src.data
    data_dest = im_dest.data

    if len(data_src.shape) == 2:
        # reshape 2D data into pseudo 3D (only one slice)
        new_shape = list(data_src.shape)
        new_shape.append(1)
        new_shape = tuple(new_shape)
        data_src = data_src.reshape(new_shape)
        data_dest = data_dest.reshape(new_shape)

    # initialize forward warping field (defined in destination space)
    warp_x = np.zeros(data_dest.shape)
    warp_y = np.zeros(data_dest.shape)

    # initialize inverse warping field (defined in source space)
    warp_inv_x = np.zeros(data_src.shape)
    warp_inv_y = np.zeros(data_src.shape)

    # Loop across slices
    logger.info(f"\nEstimate columnwise transformation...")
    for iz in range(0, nz):
        logger.info(f"{str(iz)}/{str(nz)}..")

        # PREPARE COORDINATES
        # ============================================================
        # get indices of x and y coordinates
        row, col = np.indices((nx, ny))
        # build 2xn array of coordinates in pixel space
        # ordering of indices is as follows:
        # coord_init_pix[:, 0] = 0, 0, 0, ..., 1, 1, 1..., nx, nx, nx
        # coord_init_pix[:, 1] = 0, 1, 2, ..., 0, 1, 2..., 0, 1, 2
        coord_init_pix = np.array([row.ravel(), col.ravel(), np.array(np.ones(len(row.ravel())) * iz)]).T
        # convert coordinates to physical space
        coord_init_phy = np.array(im_src.transfo_pix2phys(coord_init_pix))
        # get 2d data from the selected slice
        src2d = data_src[:, :, iz]
        dest2d = data_dest[:, :, iz]
        # julien 20161105
        #<<<
        # threshold at 0.5
        src2d[src2d < th_nonzero] = 0
        dest2d[dest2d < th_nonzero] = 0
        # get non-zero coordinates, and transpose to obtain nx2 dimensions
        coord_src2d = np.array(np.where(src2d > 0)).T
        coord_dest2d = np.array(np.where(dest2d > 0)).T
        # here we use 0.5 as threshold for non-zero value
        # coord_src2d = np.array(np.where(src2d > th_nonzero)).T
        # coord_dest2d = np.array(np.where(dest2d > th_nonzero)).T
        #>>>

        # SCALING R-L (X dimension)
        # ============================================================
        # sum data across Y to obtain 1D signal: src_y and dest_y
        src1d = np.sum(src2d, 1)
        dest1d = np.sum(dest2d, 1)
        # make sure there are non-zero data in src or dest
        if np.any(src1d > th_nonzero) and np.any(dest1d > th_nonzero):
            # retrieve min/max of non-zeros elements (edge of the segmentation)
            # julien 20161105
            # <<<
            src1d_min, src1d_max = min(np.where(src1d != 0)[0]), max(np.where(src1d != 0)[0])
            dest1d_min, dest1d_max = min(np.where(dest1d != 0)[0]), max(np.where(dest1d != 0)[0])
            # for i in range(len(src1d)):
            #     if src1d[i] > 0.5:
            #         found index above 0.5, exit loop
                    # break
            # get indices (in continuous space) at half-maximum of upward and downward slope
            # src1d_min, src1d_max = find_index_halfmax(src1d)
            # dest1d_min, dest1d_max = find_index_halfmax(dest1d)
            # >>>
            # 1D matching between src_y and dest_y
            mean_dest_x = (dest1d_max + dest1d_min) / 2
            mean_src_x = (src1d_max + src1d_min) / 2
            # compute x-scaling factor
            Sx = (dest1d_max - dest1d_min + 1) / float(src1d_max - src1d_min + 1)
            # apply transformation to coordinates
            coord_src2d_scaleX = np.copy(coord_src2d)  # need to use np.copy to avoid copying pointer
            coord_src2d_scaleX[:, 0] = (coord_src2d[:, 0] - mean_src_x) * Sx + mean_dest_x
            coord_init_pix_scaleX = np.copy(coord_init_pix)
            coord_init_pix_scaleX[:, 0] = (coord_init_pix[:, 0] - mean_src_x) * Sx + mean_dest_x
            coord_init_pix_scaleXinv = np.copy(coord_init_pix)
            coord_init_pix_scaleXinv[:, 0] = (coord_init_pix[:, 0] - mean_dest_x) / float(Sx) + mean_src_x
            # apply transformation to image
            from skimage.transform import warp
            row_scaleXinv = np.reshape(coord_init_pix_scaleXinv[:, 0], [nx, ny])
            src2d_scaleX = warp(src2d, np.array([row_scaleXinv, col]), order=1)

            # ============================================================
            # COLUMN-WISE REGISTRATION (Y dimension for each Xi)
            # ============================================================
            coord_init_pix_scaleY = np.copy(coord_init_pix)  # need to use np.copy to avoid copying pointer
            coord_init_pix_scaleYinv = np.copy(coord_init_pix)  # need to use np.copy to avoid copying pointer
            # coord_src2d_scaleXY = np.copy(coord_src2d_scaleX)  # need to use np.copy to avoid copying pointer
            # loop across columns (X dimension)
            for ix in range(nx):
                # retrieve 1D signal along Y
                src1d = src2d_scaleX[ix, :]
                dest1d = dest2d[ix, :]
                # make sure there are non-zero data in src or dest
                if np.any(src1d > th_nonzero) and np.any(dest1d > th_nonzero):
                    # retrieve min/max of non-zeros elements (edge of the segmentation)
                    # src1d_min, src1d_max = min(np.nonzero(src1d)[0]), max(np.nonzero(src1d)[0])
                    # dest1d_min, dest1d_max = min(np.nonzero(dest1d)[0]), max(np.nonzero(dest1d)[0])
                    # 1D matching between src_y and dest_y
                    # Ty = (dest1d_max + dest1d_min)/2 - (src1d_max + src1d_min)/2
                    # Sy = (dest1d_max - dest1d_min) / float(src1d_max - src1d_min)
                    # apply translation and scaling to coordinates in column
                    # get indices (in continuous space) at half-maximum of upward and downward slope
                    # src1d_min, src1d_max = find_index_halfmax(src1d)
                    # dest1d_min, dest1d_max = find_index_halfmax(dest1d)
                    src1d_min, src1d_max = np.min(np.where(src1d > th_nonzero)), np.max(np.where(src1d > th_nonzero))
                    dest1d_min, dest1d_max = np.min(np.where(dest1d > th_nonzero)), np.max(np.where(dest1d > th_nonzero))
                    # 1D matching between src_y and dest_y
                    mean_dest_y = (dest1d_max + dest1d_min) / 2
                    mean_src_y = (src1d_max + src1d_min) / 2
                    # Tx = (dest1d_max + dest1d_min)/2 - (src1d_max + src1d_min)/2
                    Sy = (dest1d_max - dest1d_min + 1) / float(src1d_max - src1d_min + 1)
                    # apply forward transformation (in pixel space)
                    # below: only for debugging purpose
                    # coord_src2d_scaleX = np.copy(coord_src2d)  # need to use np.copy to avoid copying pointer
                    # coord_src2d_scaleX[:, 0] = (coord_src2d[:, 0] - mean_src) * Sx + mean_dest
                    # coord_init_pix_scaleY = np.copy(coord_init_pix)  # need to use np.copy to avoid copying pointer
                    # coord_init_pix_scaleY[:, 0] = (coord_init_pix[:, 0] - mean_src ) * Sx + mean_dest
                    range_x = list(range(ix * ny, ix * ny + nx))
                    coord_init_pix_scaleY[range_x, 1] = (coord_init_pix[range_x, 1] - mean_src_y) * Sy + mean_dest_y
                    coord_init_pix_scaleYinv[range_x, 1] = (coord_init_pix[range_x, 1] - mean_dest_y) / float(Sy) + mean_src_y
            # apply transformation to image
            col_scaleYinv = np.reshape(coord_init_pix_scaleYinv[:, 1], [nx, ny])
            src2d_scaleXY = warp(src2d, np.array([row_scaleXinv, col_scaleYinv]), order=1)
            # regularize Y warping fields
            from skimage.filters import gaussian
            col_scaleY = np.reshape(coord_init_pix_scaleY[:, 1], [nx, ny])
            col_scaleYsmooth = gaussian(col_scaleY, smoothWarpXY)
            col_scaleYinvsmooth = gaussian(col_scaleYinv, smoothWarpXY)
            # apply smoothed transformation to image
            src2d_scaleXYsmooth = warp(src2d, np.array([row_scaleXinv, col_scaleYinvsmooth]), order=1)
            # reshape warping field as 1d
            coord_init_pix_scaleY[:, 1] = col_scaleYsmooth.ravel()
            coord_init_pix_scaleYinv[:, 1] = col_scaleYinvsmooth.ravel()
            # display
            if verbose == 2:
                # FIG 1
                plt.figure(figsize=(15, 3))
                # plot #1
                ax = plt.subplot(141)
                plt.imshow(np.swapaxes(src2d, 1, 0), cmap=plt.cm.gray, interpolation='none')
                plt.hold(True)  # add other layer
                plt.imshow(np.swapaxes(dest2d, 1, 0), cmap=plt.cm.copper, interpolation='none', alpha=0.5)
                plt.title('src')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(mean_dest_x - 15, mean_dest_x + 15)
                plt.ylim(mean_dest_y - 15, mean_dest_y + 15)
                ax.grid(True, color='w')
                # plot #2
                ax = plt.subplot(142)
                plt.imshow(np.swapaxes(src2d_scaleX, 1, 0), cmap=plt.cm.gray, interpolation='none')
                plt.hold(True)  # add other layer
                plt.imshow(np.swapaxes(dest2d, 1, 0), cmap=plt.cm.copper, interpolation='none', alpha=0.5)
                plt.title('src_scaleX')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(mean_dest_x - 15, mean_dest_x + 15)
                plt.ylim(mean_dest_y - 15, mean_dest_y + 15)
                ax.grid(True, color='w')
                # plot #3
                ax = plt.subplot(143)
                plt.imshow(np.swapaxes(src2d_scaleXY, 1, 0), cmap=plt.cm.gray, interpolation='none')
                plt.hold(True)  # add other layer
                plt.imshow(np.swapaxes(dest2d, 1, 0), cmap=plt.cm.copper, interpolation='none', alpha=0.5)
                plt.title('src_scaleXY')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(mean_dest_x - 15, mean_dest_x + 15)
                plt.ylim(mean_dest_y - 15, mean_dest_y + 15)
                ax.grid(True, color='w')
                # plot #4
                ax = plt.subplot(144)
                plt.imshow(np.swapaxes(src2d_scaleXYsmooth, 1, 0), cmap=plt.cm.gray, interpolation='none')
                plt.hold(True)  # add other layer
                plt.imshow(np.swapaxes(dest2d, 1, 0), cmap=plt.cm.copper, interpolation='none', alpha=0.5)
                plt.title('src_scaleXYsmooth (s=' + str(smoothWarpXY) + ')')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(mean_dest_x - 15, mean_dest_x + 15)
                plt.ylim(mean_dest_y - 15, mean_dest_y + 15)
                ax.grid(True, color='w')
                # save figure
                plt.savefig(os.path.join(path_qc, 'register2d_columnwise_image_z' + str(iz) + '.png'))
                plt.close()

            # ============================================================
            # CALCULATE TRANSFORMATIONS
            # ============================================================
            # calculate forward transformation (in physical space)
            coord_init_phy_scaleX = np.array(im_dest.transfo_pix2phys(coord_init_pix_scaleX))
            coord_init_phy_scaleY = np.array(im_dest.transfo_pix2phys(coord_init_pix_scaleY))
            # calculate inverse transformation (in physical space)
            coord_init_phy_scaleXinv = np.array(im_src.transfo_pix2phys(coord_init_pix_scaleXinv))
            coord_init_phy_scaleYinv = np.array(im_src.transfo_pix2phys(coord_init_pix_scaleYinv))
            # compute displacement per pixel in destination space (for forward warping field)
            warp_x[:, :, iz] = np.array([coord_init_phy_scaleXinv[i, 0] - coord_init_phy[i, 0] for i in range(nx * ny)]).reshape((nx, ny))
            warp_y[:, :, iz] = np.array([coord_init_phy_scaleYinv[i, 1] - coord_init_phy[i, 1] for i in range(nx * ny)]).reshape((nx, ny))
            # compute displacement per pixel in source space (for inverse warping field)
            warp_inv_x[:, :, iz] = np.array([coord_init_phy_scaleX[i, 0] - coord_init_phy[i, 0] for i in range(nx * ny)]).reshape((nx, ny))
            warp_inv_y[:, :, iz] = np.array([coord_init_phy_scaleY[i, 1] - coord_init_phy[i, 1] for i in range(nx * ny)]).reshape((nx, ny))

    # Generate forward warping field (defined in destination space)
    generate_warping_field(fname_dest, warp_x, warp_y, fname_warp, verbose)
    # Generate inverse warping field (defined in source space)
    generate_warping_field(fname_src, warp_inv_x, warp_inv_y, fname_warp_inv, verbose)


def register2d(fname_src, fname_dest, fname_mask='', fname_warp='warp_forward.nii.gz',
               fname_warp_inv='warp_inverse.nii.gz',
               paramreg=Paramreg(step='0', type='im', algo='Translation', metric='MI', iter='5', shrink='1', smooth='0',
                   gradStep='0.5'),
               ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '',
                                         'translation': '', 'bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                         'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'},
               verbose=0):
    """
    Slice-by-slice registration of two images.

    :param fname_src: name of moving image (type: string)
    :param fname_dest: name of fixed image (type: string)
    :param fname_mask: name of mask file (type: string) (parameter -x of antsRegistration)
    :param fname_warp: name of output 3d forward warping field
    :param fname_warp_inv: name of output 3d inverse warping field
    :param paramreg: Class Paramreg()
    :param ants_registration_params: dict: specific algorithm's parameters for antsRegistration
    :param verbose:
    :return:
        if algo==translation:
            x_displacement: list of translation along x axis for each slice (type: list)
            y_displacement: list of translation along y axis for each slice (type: list)
        if algo==rigid:
            x_displacement: list of translation along x axis for each slice (type: list)
            y_displacement: list of translation along y axis for each slice (type: list)
            theta_rotation: list of rotation angle in radian (and in ITK's coordinate system) for each slice (type: list)
        if algo==affine or algo==syn or algo==bsplinesyn:
            creation of two 3D warping fields (forward and inverse) that are the concatenations of the slice-by-slice
            warps.
    """
    # set metricSize
    # TODO: create internal function get_metricSize()
    if paramreg.metric == 'MI':
        metricSize = '32'  # corresponds to number of bins
    else:
        metricSize = '4'  # corresponds to radius (for CC, MeanSquares...)

    # Get image dimensions and retrieve nz
    logger.info(f"\nGet image dimensions of destination image...")
    nx, ny, nz, nt, px, py, pz, pt = image.Image(fname_dest).dim

    logger.info(f"  matrix size: {str(nx)} x {str(ny)} x {str(nz)}")
    logger.info(f"  voxel size: {str(px)}mm x {str(py)}mm x {str(nz)}mm")

    # Split input volume along z
    logger.info(f"\nSplit input volume...")

    im_src = image.Image(fname_src)
    split_source_list = image.split_img_data(im_src, 2)
    for im in split_source_list:
        im.save()

    # Split destination volume along z
    logger.info(f"\nSplit destination volume...")
    im_dest = image.Image(fname_dest)
    split_dest_list = image.split_img_data(im_dest, 2)
    for im in split_dest_list:
        im.save()

    # Split mask volume along z
    if fname_mask != '':
        logger.info(f"\nSplit mask volume...")
        im_mask = image.Image('mask.nii.gz')
        split_mask_list = image.split_img_data(im_mask, 2)
        for im in split_mask_list:
            im.save()

    # initialization
    if paramreg.algo in ['Translation']:
        x_displacement = [0 for i in range(nz)]
        y_displacement = [0 for i in range(nz)]
        theta_rotation = [0 for i in range(nz)]
    if paramreg.algo in ['Rigid', 'Affine', 'BSplineSyN', 'SyN']:
        list_warp = []
        list_warp_inv = []

    # loop across slices
    for i in range(nz):
        # set masking
        logger.info(f"Registering slice {str(i)}/{str(nz-1)}...")
        num = numerotation(i)
        prefix_warp2d = 'warp2d_' + num
        # if mask is used, prepare command for ANTs
        if fname_mask != '':
            masking = ['-x', 'mask_Z' + num + '.nii.gz']
        else:
            masking = []
        # main command for registration
        # TODO fixup isct_ants* parsers
        cmd = ['isct_antsRegistration',
         '--dimensionality', '2',
         '--transform', paramreg.algo + '[' + str(paramreg.gradStep) + ants_registration_params[paramreg.algo.lower()] + ']',
         '--metric', paramreg.metric + '[dest_Z' + num + '.nii' + ',src_Z' + num + '.nii' + ',1,' + metricSize + ']',  #[fixedImage,movingImage,metricWeight +nb_of_bins (MI) or radius (other)
         '--convergence', str(paramreg.iter),
         '--shrink-factors', str(paramreg.shrink),
         '--smoothing-sigmas', str(paramreg.smooth) + 'mm',
         '--output', '[' + prefix_warp2d + ',src_Z' + num + '_reg.nii]',    #--> file.mat (contains Tx,Ty, theta)
         '--interpolation', 'BSpline[3]',
         '--verbose', '1',
        ] + masking
        # add init translation
        if not paramreg.init == '':
            init_dict = {'geometric': '0', 'centermass': '1', 'origin': '2'}
            cmd += ['-r', '[dest_Z' + num + '.nii' + ',src_Z' + num + '.nii,' + init_dict[paramreg.init] + ']']

        try:
            # run registration
            run_proc(cmd, is_sct_binary=True)

            if paramreg.algo in ['Translation']:
                file_mat = prefix_warp2d + '0GenericAffine.mat'
                matfile = loadmat(file_mat, struct_as_record=True)
                array_transfo = matfile['AffineTransform_double_2_2']
                x_displacement[i] = array_transfo[4][0]  # Tx in ITK'S coordinate system
                y_displacement[i] = array_transfo[5][0]  # Ty  in ITK'S and fslview's coordinate systems
                theta_rotation[i] = asin(array_transfo[2])  # angle of rotation theta in ITK'S coordinate system (minus theta for fslview)

            if paramreg.algo in ['Rigid', 'Affine', 'BSplineSyN', 'SyN']:
                # List names of 2d warping fields for subsequent merge along Z
                file_warp2d = prefix_warp2d + '0Warp.nii.gz'
                file_warp2d_inv = prefix_warp2d + '0InverseWarp.nii.gz'
                list_warp.append(file_warp2d)
                list_warp_inv.append(file_warp2d_inv)

            if paramreg.algo in ['Rigid', 'Affine']:
                # Generating null 2d warping field (for subsequent concatenation with affine transformation)
                # TODO fixup isct_ants* parsers
                run_proc(['isct_antsRegistration',
                 '-d', '2',
                 '-t', 'SyN[1,1,1]',
                 '-c', '0',
                 '-m', 'MI[dest_Z' + num + '.nii,src_Z' + num + '.nii,1,32]',
                 '-o', 'warp2d_null',
                 '-f', '1',
                 '-s', '0',
                ], is_sct_binary=True)
                # --> outputs: warp2d_null0Warp.nii.gz, warp2d_null0InverseWarp.nii.gz
                file_mat = prefix_warp2d + '0GenericAffine.mat'
                # Concatenating mat transfo and null 2d warping field to obtain 2d warping field of affine transformation
                run_proc(['isct_ComposeMultiTransform', '2', file_warp2d, '-R', 'dest_Z' + num + '.nii', 'warp2d_null0Warp.nii.gz', file_mat], is_sct_binary=True)
                run_proc(['isct_ComposeMultiTransform', '2', file_warp2d_inv, '-R', 'src_Z' + num + '.nii', 'warp2d_null0InverseWarp.nii.gz', '-i', file_mat], is_sct_binary=True)

        # if an exception occurs with ants, take the last value for the transformation
        # TODO: DO WE NEED TO DO THAT??? (julien 2016-03-01)
        except Exception as e:
            # TODO [AJ] is it desired to completely ignore exception??
            logger.error(f"Exception occurred. \n {e}")

    # Merge warping field along z
    logger.info(f"\nMerge warping fields along z...")

    if paramreg.algo in ['Translation']:
        # convert to array
        x_disp_a = np.asarray(x_displacement)
        y_disp_a = np.asarray(y_displacement)
        theta_rot_a = np.asarray(theta_rotation)
        # Generate warping field
        generate_warping_field(fname_dest, x_disp_a, y_disp_a, fname_warp=fname_warp)  #name_warp= 'step'+str(paramreg.step)
        # Inverse warping field
        generate_warping_field(fname_src, -x_disp_a, -y_disp_a, fname_warp=fname_warp_inv)

    if paramreg.algo in ['Rigid', 'Affine', 'BSplineSyN', 'SyN']:
        # concatenate 2d warping fields along z
        image.concat_warp2d(list_warp, fname_warp, fname_dest)
        image.concat_warp2d(list_warp_inv, fname_warp_inv, fname_src)


def numerotation(nb):
    """Indexation of number for matching fslsplit's index.

    Given a slice number, this function returns the corresponding number in fslsplit indexation system.

    param nb: the number of the slice (type: int)
    return nb_output: the number of the slice for fslsplit (type: string)
    """
    if nb < 0 or nb > 9999:
        raise ValueError("Number must be between 0 and 9999")
    elif -1 < nb < 10:
        nb_output = '000' + str(nb)
    elif 9 < nb < 100:
        nb_output = '00' + str(nb)
    elif 99 < nb < 1000:
        nb_output = '0' + str(nb)
    elif 999 < nb < 10000:
        nb_output = str(nb)
    return nb_output


def generate_warping_field(fname_dest, warp_x, warp_y, fname_warp='warping_field.nii.gz', verbose=1):
    """
    Generate an ITK warping field
    :param fname_dest:
    :param warp_x:
    :param warp_y:
    :param fname_warp:
    :param verbose:
    :return:
    """
    logger.info(f"\nGenerate warping field...")

    # Get image dimensions
    nx, ny, nz, nt, px, py, pz, pt = image.Image(fname_dest).dim

    # initialize
    data_warp = np.zeros((nx, ny, nz, 1, 3))

    # fill matrix
    data_warp[:, :, :, 0, 0] = -warp_x  # need to invert due to ITK conventions
    data_warp[:, :, :, 0, 1] = -warp_y  # need to invert due to ITK conventions

    # save warping field
    im_dest = load(fname_dest)
    hdr_dest = im_dest.get_header()
    hdr_warp = hdr_dest.copy()
    hdr_warp.set_intent('vector', (), '')
    hdr_warp.set_data_dtype('float32')
    img = Nifti1Image(data_warp, None, hdr_warp)
    save(img, fname_warp)
    logger.info(f" --> {fname_warp}")


def angle_between(a, b):
    """
    Compute angle in radian between a and b. Throws an exception if a or b has zero magnitude.

    :param a: Coordinates of first point
    :param b: Coordinates of second point
    :return: angle in rads
    """
    arccosInput = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
    arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
    sign_angle = np.sign(np.cross(a, b))
    return sign_angle * acos(arccosInput)


def compute_pca(data2d):
    """
    Compute PCA using sklearn

    :param data2d: 2d array. PCA will be computed on non-zeros values.
    :return: coordsrc: 2d array: centered non-zero coordinates\
        pca: object: PCA result.\
        centermass: 2x1 array: 2d coordinates of the center of mass
    """
    # round it and make it int (otherwise end up with values like 10-7)
    data2d = data2d.round().astype(int)

    # get non-zero coordinates, and transpose to obtain nx2 dimensions
    coordsrc = np.array(data2d.nonzero()).T

    # get center of mass
    centermass = coordsrc.mean(0)

    # center data
    coordsrc = coordsrc - centermass

    # normalize data
    coordsrc /= coordsrc.std()

    # Performs PCA
    pca = PCA(n_components=2, copy=False, whiten=False)
    pca.fit(coordsrc)

    return coordsrc, pca, centermass


def find_index_halfmax(data1d):
    """
    Find the two indices at half maximum for a bell-type curve (non-parametric). Uses center of mass calculation.

    :param data1d:
    :return: xmin, xmax
    """
    # normalize data between 0 and 1
    data1d = data1d / float(np.max(data1d))

    # loop across elements and stops when found 0.5
    for i in range(len(data1d)):
        if data1d[i] > 0.5:
            break

    # compute center of mass to get coordinate at 0.5
    xmin = i - 1 + (0.5 - data1d[i - 1]) / float(data1d[i] - data1d[i - 1])

    # continue for the descending slope
    for i in range(i, len(data1d)):
        if data1d[i] < 0.5:
            break

    # compute center of mass to get coordinate at 0.5
    xmax = i - 1 + (0.5 - data1d[i - 1]) / float(data1d[i] - data1d[i - 1])

    return xmin, xmax


def find_angle_hog(image, centermass, px, py, angle_range=10):
    """
    Finds the angle of an image based on the method described by Sun, "Symmetry Detection Using Gradient Information."
    Pattern Recognition Letters 16, no. 9 (September 1, 1995): 987–96, and improved by N. Pinon

    :param: image : 2D numpy array to find symmetry axis on
    :param: centermass: tuple of floats indicating the center of mass of the image
    :param: px, py, dimensions of the pixels in the x and y direction
    :param: angle_range : float or None, in deg, the angle will be search in the range [-angle_range, angle_range], if None angle angle might be returned
    :return: angle found and confidence score
    """

    # param that can actually be tweeked to influence method performance :
    sigma = 10  # influence how far away pixels will vote for the orientation, if high far away pixels vote will count more, if low only closest pixels will participate
    nb_bin = 360  # number of angle bins for the histogram, can be more or less than 360, if high, a higher precision might be achieved but there is the risk of
    kmedian_size = 5

    # Normalization of sigma relative to pixdim :
    sigmax = sigma / px
    sigmay = sigma / py
    if nb_bin % 2 != 0:  # necessary to have even number of bins
        nb_bin = nb_bin - 1
    if angle_range is None:
        angle_range = 90

    # Constructing mask based on center of mass that will influence the weighting of the orientation histogram
    nx, ny = image.shape
    xx, yy = np.mgrid[:nx, :ny]
    seg_weighted_mask = np.exp(
        -(((xx - centermass[0]) ** 2) / (2 * (sigmax ** 2)) + ((yy - centermass[1]) ** 2) / (2 * (sigmay ** 2))))

    # Acquiring the orientation histogram :
    grad_orient_histo = gradient_orientation_histogram(image, nb_bin=nb_bin, seg_weighted_mask=seg_weighted_mask)

    # Bins of the histogram :
    repr_hist = np.linspace(-(np.pi - 2 * np.pi / nb_bin), (np.pi - 2 * np.pi / nb_bin), nb_bin - 1)

    # Smoothing of the histogram, necessary to avoid digitization effects that will favor angles 0, 45, 90, -45, -90:
    grad_orient_histo_smooth = circular_filter_1d(grad_orient_histo, kmedian_size, kernel='median')  # fft than square than ifft to calculate convolution

    # Computing the circular autoconvolution of the histogram to obtain the axis of symmetry of the histogram :
    grad_orient_histo_conv = circular_conv(grad_orient_histo_smooth, grad_orient_histo_smooth)

    # Restraining angle search to the angle range :
    index_restrain = int(np.ceil(np.true_divide(angle_range, 180) * nb_bin))
    center = (nb_bin - 1) // 2
    grad_orient_histo_conv_restrained = grad_orient_histo_conv[center - index_restrain + 1:center + index_restrain + 1]

    # Finding the symmetry axis by searching for the maximum in the autoconvolution of the histogram :
    index_angle_found = np.argmax(grad_orient_histo_conv_restrained) + (nb_bin // 2 - index_restrain)
    angle_found = repr_hist[index_angle_found] / 2
    angle_found_score = np.amax(grad_orient_histo_conv_restrained)

    # Finding other maxima to compute confidence score
    arg_maxs = argrelmax(grad_orient_histo_conv_restrained, order=kmedian_size, mode='wrap')[0]

    # Confidence score is the ratio of the 2 first maxima :
    if len(arg_maxs) > 1:
        conf_score = angle_found_score / grad_orient_histo_conv_restrained[arg_maxs[1]]
    else:
        conf_score = angle_found_score / np.mean(grad_orient_histo_conv)  # if no other maxima  in the region ratio of the maximum to the mean

    return angle_found, conf_score


def gradient_orientation_histogram(image, nb_bin, seg_weighted_mask=None):
    """
    This function takes an image as an input and return its orientation histogram

    :param image: the image to compute the orientation histogram from, a 2D numpy array
    :param nb_bin: the number of bins of the histogram, an int, for instance 360 for bins 1 degree large (can be more or less than 360)
    :param seg_weighted_mask: optional, mask weighting the histogram count, base on segmentation, 2D numpy array between 0 and 1
    :return grad_orient_histo: the histogram of the orientations of the image, a 1D numpy array of length nb_bin"""

    h_kernel = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]]) / 4.0
    v_kernel = h_kernel.T

    # Normalization by median, to resolve scaling problems
    median = np.median(image)
    if median != 0:
        image = image / median

    # x and y gradients of the image
    gradx = ndimage.convolve(image, v_kernel)
    grady = ndimage.convolve(image, h_kernel)

    # orientation gradient
    orient = np.arctan2(grady, gradx)  # results are in the range -pi pi

    # weight by gradient magnitude :  this step seems dumb, it alters the angles
    grad_mag = ((np.abs(gradx.astype(object)) ** 2 + np.abs(grady.astype(object)) ** 2) ** 0.5)  # weird data type manipulation, cannot explain why it failed without it
    if np.max(grad_mag) != 0:
        grad_mag = grad_mag / np.max(grad_mag)  # to have map between 0 and 1 (and keep consistency with the seg_weihting map if provided)

    if seg_weighted_mask is not None:
        weighting_map = np.multiply(seg_weighted_mask, grad_mag)  # include weightning by segmentation
    else:
        weighting_map = grad_mag

    # compute histogram :
    grad_orient_histo = np.histogram(np.concatenate(orient), bins=nb_bin - 1, range=(-(np.pi - np.pi / nb_bin), (np.pi - np.pi / nb_bin)),
                                     weights=np.concatenate(weighting_map))

    return grad_orient_histo[0].astype(float)  # return only the values of the bins, not the bins (we know them)


def circular_conv(signal1, signal2):
    """
    Takes two 1D numpy array and perform a circular convolution with them

    :param signal1: 1D numpy array
    :param signal2: 1D numpy array, same length as signal1
    :return: signal_conv : 1D numpy array, result of circular convolution of signal1 and signal2"""

    if signal1.shape != signal2.shape:
        raise Exception("The two signals for circular convolution do not have the same shape")

    signal2_extended = np.concatenate((signal2, signal2, signal2))  # replicate signal at both ends

    signal_conv_extended = np.convolve(signal1, signal2_extended, mode="same")  # median filtering

    signal_conv = signal_conv_extended[len(signal1):2*len(signal1)]  # truncate back the signal

    return signal_conv


def circular_filter_1d(signal, window_size, kernel='gaussian'):
    """
    This function filters circularly the signal inputted with a median filter of inputted size, in this context\
    circularly means that the signal is wrapped around and then filtered

    :param signal: 1D numpy array
    :param window_size: size of the kernel, an int
    :return: signal_smoothed: 1D numpy array, same size as signal"""

    signal_extended = np.concatenate((signal, signal, signal))  # replicate signal at both ends
    if kernel == 'gaussian':
        signal_extended_smooth = ndimage.gaussian_filter(signal_extended, window_size)  # gaussian
    elif kernel == 'median':
        signal_extended_smooth = medfilt(signal_extended, window_size)  # median filtering
    else:
        raise Exception("Unknow type of kernel")

    signal_smoothed = signal_extended_smooth[len(signal):2*len(signal)]  # truncate back the signal

    return signal_smoothed
