"""
Tools for motion correction (moco)

Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

# TODO: Inform user if soft mask is used
# TODO: no need to pass absolute image path-- makes it difficult to read
# TODO: check the status of spline()
# TODO: check the status of combine_matrix()
# TODO: params for ANTS: CC/MI, shrink fact, nb_it
# TODO: ants: explore optin  --float  for faster computation

from copy import deepcopy
import sys
import os
from shutil import copyfile
import glob
import math
import time
import functools
import operator
import csv
import textwrap

import numpy as np
import scipy.interpolate

from spinalcordtoolbox.math import binarize
from spinalcordtoolbox.image import Image, add_suffix, generate_output_file, convert
from spinalcordtoolbox.utils.shell import get_interpolation
from spinalcordtoolbox.utils.sys import sct_progress_bar, run_proc, printv
from spinalcordtoolbox.utils.fs import tmp_create, extract_fname, rmtree, copy

# FIXME don't import from scripts in API
from spinalcordtoolbox.scripts import sct_dmri_separate_b0_and_dwi
from spinalcordtoolbox.scripts.sct_image import split_data, concat_data, multicomponent_split
from spinalcordtoolbox.scripts import sct_apply_transfo


class ParamMoco:
    """
    Class with a bunch of moco-specific parameters
    """
    # The constructor

    def __init__(self, is_diffusion=None, fname_data='', fname_bvecs='', fname_bvals='', fname_mask='', path_out='',
                 group_size=1, remove_temp_files=1, verbose=1, poly='2', smooth='1', gradStep='1', iterations='10',
                 metric='MeanSquares', sampling='None', interp='spline', num_target='0',
                 bval_min=100, iterAvg=1, output_motion_param=True, fname_ref=''):
        """

        :param is_diffusion: Bool: If True, data will be treated as diffusion-MRI data (process slightly differs)
        :param fname_data: str: Input data file name (e.g. dmri.nii.gz)
        :param fname_bvecs: str: b-vector file name (e.g. bvecs.txt)
        :param fname_bvals: str: b-value file name (e.g. bvals.txt)
        :param fname_mask: str: Mask file name (e.g. mask.nii.gz)
        :param path_out: str: Output folder
        :param group_size: int: Number of images averaged for 'dwi' method.
        :param remove_temp_files: int: If 1, temporary files are removed.
        :param verbose: int: Verbosity level (0, 1, 2)
        :param poly: str: Degree of polynomial function used for regularization along Z. For no regularization set to 0.
        :param smooth: str: Smoothing sigma in mm # TODO: make it int
        :param gradStep: float: Searching step used by registration algorithm. The higher the more deformation is allowed.
        :param iterations: int: Number of iterations for the registration algorithm (default is 10)
        :param metric: {MeanSquares, MI, CC}: metric to use for registration
        :param sampling: str: Sampling rate used for registration metric; 'None' means use 'dense sampling'
        :param interp: str: Interpolation method for the final image ('nn', 'linear', 'spline')
        :param num_target: str: Number of target image (default is '0', which means the first image)
        :param bval_min: int: Minimum b-value threshold (default is 100, which means that b-values below this threshold
                              are considered as b=0)
                              - Note: Useful in case user does not have min bvalues at 0 (e.g. where csf disapeared).
                              - Note: This value is passed to `sct_dmri_separate_b0_and_dwi.identify_b0()`
        :param iterAvg: int: Whether or not to average registered volumes with target image (default is 1)
        :param output_motion_param: bool: If True, the motion parameters are outputted (default is True)
        :param fname_ref: str: Reference volume for motion correction, for example the mean fMRI volume.
        """
        # This parameter is set depending on whether `sct_dmri_moco` or `sct_fmri_moco` is called
        self.is_diffusion = is_diffusion

        # Parameters controlled by specific `sct_dmri_moco`/`sct_fmri_moco` arguments (e.g. `-i`, `-m`, `-g`, etc.)
        self.fname_data = fname_data
        self.fname_mask = fname_mask
        self.fname_ref = fname_ref
        self.path_out = path_out
        self.group_size = group_size
        self.interp = interp
        self.verbose = verbose
        self.remove_temp_files = remove_temp_files

        # Parameters controlled by the specific `sct_dmri_moco` arguments
        self.fname_bvecs = fname_bvecs
        self.fname_bvals = fname_bvals
        self.bval_min = bval_min

        # Advanced parameters defined by `-param` for both `sct_dmri_moco` and `sct_fmri_moco`
        self.poly = poly
        self.smooth = smooth
        self.metric = metric
        self.iter = iterations
        self.gradStep = gradStep
        self.sampling = sampling
        self.num_target = num_target
        self.iterAvg = iterAvg

        # Params that are always True
        self.output_motion_param = output_motion_param  # TODO: Why would we set this to False?

        # Unused parameters
        self.spline_fitting = 0  # TODO: This currently raises a NotImplementedError, so don't expose it
        self.plot_graph = 0      # TODO: This is only used for `spline_fitting` which raises a NotImplementedError
        self.min_norm = 0.001    # TODO: This param is currently not used anywhere
        self.swapXY = 0          # TODO: This param is currently not used anywhere
        self.debug = 0           # TODO: This param is currently not used anywhere

        # Parameters that are used internally during the moco procedure and shouldn't be determined by the user
        self.is_sagittal = None  # set automatically based on orientation
        self.suffix_mat = None   # set automatically based on `is_sagittal`
        self.suffix = '_moco'    # general suffix for all output files
        self.mat_moco = ''       # output folder for intermediate `.mat` files
        self.mat_final = ''      # output folder for final `.mat` files
        self.todo = ''           # the moco step to perform next

    # update constructor with user's parameters (`-param`)
    # `-param` should have the type "list_type(',', str)", meaning the argument string will already be split by `,`
    def update(self, param_user):
        for param in param_user:
            # check for user errors
            substrings = param.split('=')
            if len(substrings) != 2:
                raise ValueError(f"Invalid parameter format: '{param}'. Expected format is 'param_name=value'.")
            param_name, value = substrings
            if not hasattr(self, param_name):
                raise ValueError(f"Unknown parameter '{param_name}'.")
            # set the validated parameter value
            setattr(self, param_name, value)


def copy_mat_files(nt, list_file_mat, index, folder_out, param):
    """
    Copy mat file from the grouped folder to the final folder (will be used by all individual ungrouped volumes)

    :param nt: int: Total number of volumes in native 4d data
    :param list_file_mat: list of list: File name of transformations
    :param index: list: Index to associate a given matrix file with a 3d volume (from the 4d native data)
    :param param: Param class
    :param folder_out: str: Output folder
    :return: None
    """
    # create final mat folder
    try:
        os.makedirs(folder_out)
    except FileExistsError:
        pass
    # Loop across registration matrices and copy to mat_final folder
    # First loop is accross z. If axial orientation, there is only one z (i.e., len(file_mat)=1)
    for iz in range(len(list_file_mat)):
        # Second loop is across ALL volumes of the input dmri dataset (corresponds to its 4th dimension: time)
        for it in range(nt):
            # Check if this index corresponds to a volume listed in the index list
            if it in index:
                file_mat = list_file_mat[iz][index.index(it)]
                fsrc = os.path.join(file_mat + param.suffix_mat)
                # Build final transfo file name
                file_mat_final = f'{os.path.basename(file_mat)[:-9]}{iz:04}T{it:04}'
                fdest = os.path.join(folder_out, file_mat_final + param.suffix_mat)
                copyfile(fsrc, fdest)


def moco_wrapper(param):
    """
    Wrapper that performs motion correction.

    :param param: ParamMoco class
    :return: fname_moco
    """
    file_data = 'data.nii'  # corresponds to the full input data (e.g. dmri or fmri)
    file_data_dirname, file_data_basename, file_data_ext = extract_fname(file_data)
    file_b0 = 'b0.nii'
    file_datasub = 'datasub.nii'  # corresponds to the full input data minus the b=0 scans (if param.is_diffusion=True)
    file_datasubgroup = 'datasub-groups.nii'  # concatenation of the average of each file_datasub
    file_mask = 'mask.nii'
    file_ref = 'reference.nii'
    file_moco_params_csv = 'moco_params.tsv'
    file_moco_params_x = 'moco_params_x.nii.gz'
    file_moco_params_y = 'moco_params_y.nii.gz'
    ext_data = '.nii.gz'  # workaround "too many open files" by slurping the data
    # TODO: check if .nii can be used
    mat_final = 'mat_final'
    # ext_mat = 'Warp.nii.gz'  # warping field

    # Start timer
    start_time = time.time()

    printv(textwrap.dedent(f"""
        Input parameters:
        ----------------------------------------------------
        Input file:            {param.fname_data}
        Output folder:         {os.path.abspath(param.path_out)}
        Mask:                  {param.fname_mask if param.fname_mask != '' else 'None'}
        Reference image:       {param.fname_ref if param.fname_ref != '' else 'None'}
        bvals (dmri only):     {param.fname_bvals}
        bvecs (dmri only):     {param.fname_bvecs}
    """), param.verbose)

    printv(textwrap.dedent(f"""
        Motion correction parameters:
        ----------------------------------------------------
        Group size:            {param.group_size}
        Polynomial order:      {param.poly}
        Smoothing (mm):        {param.smooth}
        Metric:                {param.metric}
        Iterations:            {param.iter}
        Gradient step:         {param.gradStep}
        Sampling:              {param.sampling}
        Target:                {param.num_target if param.fname_ref == '' else 'N/A (reference image provided)'}
        Iterative averaging:   {param.iterAvg}
        Interpolation:         {param.interp}
    """), param.verbose)

    # Create tmp folder
    path_tmp = tmp_create(basename="moco-wrapper")

    # Copying input data to tmp folder
    printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    im_data = convert(Image(param.fname_data))
    im_data.save(os.path.join(path_tmp, file_data), mutable=True, verbose=param.verbose)
    if param.fname_mask != '':
        im_mask = convert(Image(param.fname_mask))
        im_mask.save(os.path.join(path_tmp, file_mask), mutable=True, verbose=param.verbose)
        # Update field in param (because used later in another function, and param class will be passed)
        param.fname_mask = file_mask
    if param.fname_ref != '':
        im_ref = convert(Image(param.fname_ref))
        im_ref.save(os.path.join(path_tmp, file_ref), mutable=True, verbose=param.verbose)
        # Update field in param (because used later in another function, and param class will be passed)
        param.fname_ref = file_ref
    if param.fname_bvals != '':
        _, _, ext_bvals = extract_fname(param.fname_bvals)
        file_bvals = f"bvals.{ext_bvals}"  # Use hardcoded name to avoid potential duplicate files when copying
        copyfile(param.fname_bvals, os.path.join(path_tmp, file_bvals))
        param.fname_bvals = file_bvals
    if param.fname_bvecs != '':
        _, _, ext_bvecs = extract_fname(param.fname_bvecs)
        file_bvecs = f"bvecs.{ext_bvecs}"  # Use hardcoded name to avoid potential duplicate files when copying
        copyfile(param.fname_bvecs, os.path.join(path_tmp, file_bvecs))
        param.fname_bvecs = file_bvecs

    # Build absolute output path and go to tmp folder
    curdir = os.getcwd()
    path_out_abs = os.path.abspath(param.path_out)
    os.chdir(path_tmp)

    # Get dimensions of data
    printv('\nGet dimensions of data...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    printv(f'  {nx} x {ny} x {nz}', param.verbose)

    # Get orientation
    printv(f'\nData orientation: {im_data.orientation}', param.verbose)
    if im_data.orientation[2] in 'LR':
        param.is_sagittal = True
        printv('  Treated as sagittal')
    elif im_data.orientation[2] in 'IS':
        param.is_sagittal = False
        printv('  Treated as axial')
    else:
        param.is_sagittal = False
        printv('WARNING: Orientation seems to be neither axial nor sagittal. Treated as axial.')

    printv("\nSet suffix of transformation file name, which depends on the orientation:")
    if param.is_sagittal:
        param.suffix_mat = '0GenericAffine.mat'
        printv(f"Orientation is sagittal, suffix is '{param.suffix_mat}'. The image is split across the R-L direction, and the "
               "estimated transformation is a 2D affine transfo.")
    else:
        param.suffix_mat = 'Warp.nii.gz'
        printv(f"Orientation is axial, suffix is '{param.suffix_mat}'. The estimated transformation is a 3D warping field, which is "
               "composed of a stack of 2D Tx-Ty transformations")

    # Adjust group size in case of sagittal scan
    if param.is_sagittal and param.group_size != 1:
        printv('For sagittal data group_size should be one for more robustness. Forcing group_size=1.', 1,
               'warning')
        param.group_size = 1

    if param.is_diffusion:
        # Identify b=0 and DWI images
        index_b0, index_dwi, nb_b0, nb_dwi = \
            sct_dmri_separate_b0_and_dwi.identify_b0(param.fname_bvecs, param.fname_bvals, param.bval_min,
                                                     param.verbose)

        # check if dmri and bvecs are the same size
        if nt != (nb_b0 + nb_dwi):
            printv(f"\nERROR in {os.path.basename(__file__)}: Size of data ({nt}) and size of bvecs ({nb_b0 + nb_dwi}) "
                   f"are not the same. Check your bvecs file.\n", 1, 'error')

    # ==================================================================================================================
    # Prepare data (mean/groups...)
    # ==================================================================================================================

    # Split into T dimension
    printv('\nSplit along T dimension...', param.verbose)
    im_data_split_list = split_data(im_data, 3)
    for im in im_data_split_list:
        x_dirname, x_basename, x_ext = extract_fname(im.absolutepath)
        im.absolutepath = os.path.join(x_dirname, f'{x_basename}.nii.gz')
        im.save()

    if param.is_diffusion:
        # Merge and average b=0 images
        printv('\nMerge and average b=0 data...', param.verbose)
        im_b0_list = []
        for it in range(nb_b0):
            im_b0_list.append(im_data_split_list[index_b0[it]])
        im_b0 = concat_data(im_b0_list, 3).save(file_b0, verbose=0)
        # Average across time
        im_b0.mean(dim=3).save(add_suffix(file_b0, '_mean'))

        n_moco = nb_dwi  # set number of data to perform moco on (using grouping)
        index_moco = index_dwi

    # If not a diffusion scan, we will motion-correct all volumes
    else:
        n_moco = nt
        index_moco = list(range(0, nt))

    nb_groups = int(math.floor(n_moco / param.group_size))

    # Generate groups indexes
    group_indexes = []
    for iGroup in range(nb_groups):
        group_indexes.append(index_moco[(iGroup * param.group_size):((iGroup + 1) * param.group_size)])

    # add the remaining images to a new last group (in case the total number of image is not divisible by group_size)
    nb_remaining = n_moco % param.group_size  # number of remaining images
    if nb_remaining > 0:
        nb_groups += 1
        group_indexes.append(index_moco[len(index_moco) - nb_remaining:len(index_moco)])

    _, file_dwi_basename, file_dwi_ext = extract_fname(file_datasub)
    # Group data
    list_file_group = []
    for iGroup in sct_progress_bar(range(nb_groups), unit='iter', unit_scale=False, desc="Merge within groups",
                                   ncols=80):
        # get index
        index_moco_i = group_indexes[iGroup]
        n_moco_i = len(index_moco_i)
        # concatenate images across time, within this group
        file_dwi_merge_i = f'{file_dwi_basename}_{iGroup}{ext_data}'
        im_dwi_list = []
        for it in range(n_moco_i):
            im_dwi_list.append(im_data_split_list[index_moco_i[it]])
        im_dwi_out = concat_data(im_dwi_list, 3).save(file_dwi_merge_i, verbose=0)
        # Average across time
        list_file_group.append(f'{file_dwi_basename}_{iGroup}_mean{ext_data}')
        im_dwi_out_mean = im_dwi_out.mean(dim=3)
        im_dwi_out_mean.hdr.set_data_dtype(im_dwi_out_mean.data.dtype)  # avoid issues with mismatched header dtype
        im_dwi_out_mean.save(list_file_group[-1])

    # Merge across groups
    printv('\nMerge across groups...', param.verbose)
    # file_dwi_groups_means_merge = 'dwi_averaged_groups'
    fname_dw_list = []
    for iGroup in range(nb_groups):
        fname_dw_list.append(list_file_group[iGroup])
    im_dw_list = [Image(fname) for fname in fname_dw_list]
    concat_data(im_dw_list, 3).save(file_datasubgroup, verbose=0)

    # Cleanup
    del im, im_data_split_list

    # ==================================================================================================================
    # Estimate moco
    # ==================================================================================================================

    # Initialize another class instance that will be passed on to the moco() function
    param_moco = deepcopy(param)

    if param.is_diffusion:
        # Estimate moco on b0 groups
        printv('\n-------------------------------------------------------------------------------', param.verbose)
        printv('  Estimating motion on b=0 images...', param.verbose)
        printv('-------------------------------------------------------------------------------', param.verbose)
        param_moco.file_data = 'b0.nii'
        # Identify target image
        if index_moco[0] != 0:
            # If first DWI is not the first volume (most common), then there is a least one b=0 image before. In that
            # case select it as the target image for registration of all b=0
            param_moco.file_target = os.path.join(file_data_dirname,
                                                  f'{file_data_basename}_T{index_b0[index_moco[0] - 1]:04}{ext_data}')
        else:
            # If first DWI is the first volume, then the target b=0 is the first b=0 from the index_b0.
            param_moco.file_target = os.path.join(file_data_dirname,
                                                  f'{file_data_basename}_T{index_b0[0]:04}{ext_data}')
        # Run moco
        param_moco.path_out = ''
        param_moco.todo = 'estimate_and_apply'
        param_moco.mat_moco = 'mat_b0groups'
        file_mat_b0, _ = moco(param_moco)

    # Estimate moco across groups
    printv('\n-------------------------------------------------------------------------------', param.verbose)
    printv('  Estimating motion across groups...', param.verbose)
    printv('-------------------------------------------------------------------------------', param.verbose)
    param_moco.file_data = file_datasubgroup
    if param.fname_ref != '':
        param_moco.file_target = file_ref
    else:
        if param.num_target.isdigit():
            num_target = int(param.num_target)
            if num_target < 0 or num_target >= nb_groups:
                printv(f'\nERROR: Target image number is out of range. It should be between 0 and {nb_groups - 1}.\n',
                       1, 'error')
                sys.exit(2)
        else:
            printv('\nERROR: Target image number is not an integer.\n', 1, 'error')
            sys.exit(2)
        param_moco.file_target = list_file_group[num_target]
    param_moco.path_out = ''
    param_moco.todo = 'estimate_and_apply'
    param_moco.mat_moco = 'mat_groups'
    file_mat_datasub_group, _ = moco(param_moco)

    # Spline Regularization along T
    if param.spline_fitting:
        # TODO: fix this scenario (haven't touched that code for a while-- it is probably buggy)
        raise NotImplementedError()
        # spline(mat_final, nt, nz, param.verbose, np.array(index_b0), param.plot_graph)

    # ==================================================================================================================
    # Apply moco
    # ==================================================================================================================

    # If group_size>1, assign transformation to each individual ungrouped 3d volume
    if param.group_size > 1:
        file_mat_datasub = []
        for iz in range(len(file_mat_datasub_group)):
            # duplicate by factor group_size the transformation file for each it
            #  example: [mat.Z0000T0001Warp.nii] --> [mat.Z0000T0001Warp.nii, mat.Z0000T0001Warp.nii] for group_size=2
            file_mat_datasub.append(
                functools.reduce(operator.iconcat, [[i] * param.group_size for i in file_mat_datasub_group[iz]], []))
    else:
        file_mat_datasub = file_mat_datasub_group

    # Copy transformations to mat_final folder and rename them appropriately
    copy_mat_files(nt, file_mat_datasub, index_moco, mat_final, param)
    if param.is_diffusion:
        copy_mat_files(nt, file_mat_b0, index_b0, mat_final, param)

    # Apply moco on all dmri data
    printv('\n-------------------------------------------------------------------------------', param.verbose)
    printv('  Apply moco', param.verbose)
    printv('-------------------------------------------------------------------------------', param.verbose)
    param_moco.file_data = file_data
    param_moco.file_target = list_file_group[0]  # reference for reslicing into proper coordinate system
    param_moco.path_out = ''  # TODO not used in moco()
    param_moco.mat_moco = mat_final
    param_moco.todo = 'apply'
    file_mat_data, im_moco = moco(param_moco)

    # copy geometric information from header
    # NB: this is required because WarpImageMultiTransform in 2D mode wrongly sets pixdim(3) to "1".
    im_moco.header = im_data.header
    im_moco.save(verbose=0)

    # Average across time
    if param.is_diffusion:
        # generate b0_moco_mean and dwi_moco_mean
        args = ['-i', im_moco.absolutepath, '-bvec', param.fname_bvecs, '-a', '1', '-v', '0']
        if not param.fname_bvals == '':
            # if bvals file is provided
            args.extend(['-bval', param.fname_bvals])
        fname_b0, fname_b0_mean, fname_dwi, fname_dwi_mean = sct_dmri_separate_b0_and_dwi.main(argv=args)
    else:
        fname_moco_mean = add_suffix(im_moco.absolutepath, '_mean')
        im_moco.mean(dim=3).save(fname_moco_mean)

    # Extract and output the motion parameters (doesn't work for sagittal orientation)
    printv('Extract motion parameters...')
    if param.output_motion_param:
        if param.is_sagittal:
            printv('Motion parameters cannot be generated for sagittal images.', 1, 'warning')
        else:
            files_warp_X, files_warp_Y = [], []
            moco_param = []
            for fname_warp in file_mat_data[0]:
                # Cropping the image to keep only one voxel in the XY plane
                im_warp = Image(fname_warp + param.suffix_mat)
                im_warp.data = np.expand_dims(np.expand_dims(im_warp.data[0, 0, :, :, :], axis=0), axis=0)

                # These three lines allow to generate one file instead of two, containing X, Y and Z moco parameters
                # fname_warp_crop = f'{fname_warp}_crop_{ext_mat}'
                # files_warp.append(fname_warp_crop)
                # im_warp.save(fname_warp_crop)

                # Separating the three components and saving X and Y only (Z is equal to 0 by default).
                im_warp_XYZ = multicomponent_split(im_warp)

                fname_warp_crop_X = f'{fname_warp}_crop_X_{param.suffix_mat}'
                im_warp_XYZ[0].save(fname_warp_crop_X)
                files_warp_X.append(fname_warp_crop_X)

                fname_warp_crop_Y = f'{fname_warp}_crop_Y_{param.suffix_mat}'
                im_warp_XYZ[1].save(fname_warp_crop_Y)
                files_warp_Y.append(fname_warp_crop_Y)

                # Calculating the slice-wise average moco estimate to provide a QC file
                # We're at a fixed time point, and we have a list of (X, Y) translation vectors, one for each Z slice
                # This is the average of the translation lengths
                X = im_warp_XYZ[0].data
                Y = im_warp_XYZ[1].data
                moco_param.append([np.mean(np.sqrt(X*X + Y*Y))])

            # These two lines allow to generate one file instead of two, containing X, Y and Z moco parameters
            # im_warp = [Image(fname) for fname in files_warp]
            # im_warp_concat = concat_data(im_warp, dim=3)
            # im_warp_concat.save('fmri_moco_params.nii')

            # Concatenating the moco parameters into a time series for X and Y components.
            im_warp_X = [Image(fname) for fname in files_warp_X]
            im_warp_concat = concat_data(im_warp_X, dim=3)
            im_warp_concat.save(file_moco_params_x)

            im_warp_Y = [Image(fname) for fname in files_warp_Y]
            im_warp_concat = concat_data(im_warp_Y, dim=3)
            im_warp_concat.save(file_moco_params_y)

            # Writing a TSV file with the slicewise average estimate of the moco parameters. Useful for QC
            with open(file_moco_params_csv, 'wt', newline='') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerow(['mean(sqrt(X^2 + Y^2))'])
                tsv_writer.writerows(moco_param)

    # Generate output files
    printv('\nGenerate output files...', param.verbose)
    fname_moco = os.path.join(path_out_abs, add_suffix(os.path.basename(param.fname_data), param.suffix))
    generate_output_file(im_moco.absolutepath, fname_moco)
    if param.is_diffusion:
        generate_output_file(fname_b0_mean, add_suffix(fname_moco, '_b0_mean'))
        generate_output_file(fname_dwi_mean, add_suffix(fname_moco, '_dwi_mean'))
    else:
        generate_output_file(fname_moco_mean, add_suffix(fname_moco, '_mean'))
    if os.path.exists(file_moco_params_csv):
        generate_output_file(file_moco_params_x, os.path.join(path_out_abs, file_moco_params_x),
                             squeeze_data=False)
        generate_output_file(file_moco_params_y, os.path.join(path_out_abs, file_moco_params_y),
                             squeeze_data=False)
        generate_output_file(file_moco_params_csv, os.path.join(path_out_abs, file_moco_params_csv))

    # Delete temporary files
    if param.remove_temp_files == 1:
        printv('\nDelete temporary files...', param.verbose)
        rmtree(path_tmp, verbose=param.verbose)

    # come back to working directory
    os.chdir(curdir)

    # display elapsed time
    elapsed_time = time.time() - start_time
    printv(f'\nElapsed time: {int(np.round(elapsed_time))}s', param.verbose)

    fname_moco = os.path.join(param.path_out, add_suffix(os.path.basename(param.fname_data), param.suffix))

    return fname_moco


def moco(param):
    """
    Main function that performs motion correction.

    :param param:
    :return:
    """
    # retrieve parameters
    file_data = param.file_data
    file_target = param.file_target
    folder_mat = param.mat_moco  # output folder of mat file
    todo = param.todo
    suffix = param.suffix
    verbose = param.verbose

    printv(f'Motion correction wrapper: {todo}', param.verbose)

    try:
        os.makedirs(folder_mat)
    except FileExistsError:
        pass

    # Get size of data
    printv('\nData dimensions:', verbose)
    im_data = Image(param.file_data)
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    printv(f'  {nx} x {ny} x {nz} x {nt}', verbose)

    # copy file_target to a temporary file
    printv('\nCopy file_target to a temporary file...', verbose)
    im_target = convert(Image(param.file_target))
    im_target.save("target.nii.gz", mutable=True, verbose=0)

    file_mask = param.fname_mask
    if file_mask != '':
        im_mask = Image(file_mask)
        im_mask.data = binarize(im_mask.data, bin_thr=0.5)
        file_mask = add_suffix(im_mask.absolutepath, "_bin")
        im_mask.save(file_mask)

    # If scan is sagittal, split src and target along Z (slice)
    if param.is_sagittal:
        dim_sag = 2  # TODO: find it
        # z-split data (time series)
        im_z_list = split_data(im_data, dim=dim_sag, squeeze_data=False)
        file_data_splitZ = []
        for im_z in im_z_list:
            im_z.save(verbose=0)
            file_data_splitZ.append(im_z.absolutepath)
        # z-split target
        im_targetz_list = split_data(Image(file_target), dim=dim_sag, squeeze_data=False)
        file_target_splitZ = []
        for im_targetz in im_targetz_list:
            im_targetz.save(verbose=0)
            file_target_splitZ.append(im_targetz.absolutepath)
        # z-split mask (if exists)
        if file_mask != '':
            im_maskz_list = split_data(Image(file_mask), dim=dim_sag, squeeze_data=False)
            file_mask_splitZ = []
            for im_maskz in im_maskz_list:
                im_maskz.save(verbose=0)
                file_mask_splitZ.append(im_maskz.absolutepath)
        # initialize file list for output matrices
        file_mat = np.empty((nz, nt), dtype=object)

    # axial orientation
    else:
        file_data_splitZ = [file_data]  # TODO: make it absolute like above
        file_target_splitZ = [file_target]  # TODO: make it absolute like above
        # initialize file list for output matrices
        file_mat = np.empty((1, nt), dtype=object)

        # deal with mask
        if file_mask != '':
            im_mask = convert(Image(file_mask), squeeze_data=False)
            im_mask.save(file_mask, mutable=True, verbose=0)
            im_maskz_list = [Image(file_mask)]  # use a list with single element

    # Loop across file list, where each file is either a 2D volume (if sagittal) or a 3D volume (otherwise)
    # file_mat = tuple([[[] for i in range(nt)] for i in range(nz)])

    file_data_splitZ_moco = []
    printv('\nRegister. Loop across Z (note: there is only one Z if orientation is axial)')
    for file in file_data_splitZ:
        iz = file_data_splitZ.index(file)
        # Split data along T dimension
        # printv('\nSplit data along T dimension.', verbose)
        im_z = Image(file)
        list_im_zt = split_data(im_z, dim=3)
        file_data_splitZ_splitT = []
        for im_zt in list_im_zt:
            im_zt.save(verbose=0)
            file_data_splitZ_splitT.append(im_zt.absolutepath)
        # file_data_splitT = f'{file_data}_T'

        # Motion correction: initialization
        index = np.arange(nt)
        file_data_splitZ_splitT_moco = []
        failed_transfo = [0 for i in range(nt)]

        # Motion correction: Loop across T
        for indice_index in sct_progress_bar(range(nt), unit='iter', unit_scale=False,
                                             desc=f"Z={iz}/{len(file_data_splitZ) - 1}", ncols=80):

            # create indices and display stuff
            it = index[indice_index]
            file_mat[iz][it] = os.path.join(folder_mat, f"mat.Z{iz:04}T{it:04}")
            file_data_splitZ_splitT_moco.append(add_suffix(file_data_splitZ_splitT[it], '_moco'))
            # deal with masking (except in the 'apply' case, where masking is irrelevant)
            input_mask = None
            if file_mask != '' and param.todo != 'apply':
                input_mask = im_maskz_list[iz]

            # run 3D registration
            failed_transfo[it] = register(param, file_data_splitZ_splitT[it], file_target_splitZ[iz], file_mat[iz][it],
                                          file_data_splitZ_splitT_moco[it], im_mask=input_mask)

            # average registered volume with target image
            # N.B. use weighted averaging: (target * nb_it + moco) / (nb_it + 1)
            if param.iterAvg and indice_index < 10 and failed_transfo[it] == 0 and not param.todo == 'apply':
                im_targetz = Image(file_target_splitZ[iz])
                data_targetz = im_targetz.data
                data_mocoz = Image(file_data_splitZ_splitT_moco[it]).data
                data_targetz = (data_targetz * (indice_index + 1) + data_mocoz) / (indice_index + 2)
                im_targetz.data = data_targetz
                im_targetz.save(verbose=0)

        # Replace failed transformation with the closest good one
        fT = [i for i, j in enumerate(failed_transfo) if j == 1]
        gT = [i for i, j in enumerate(failed_transfo) if j == 0]
        for it in range(len(fT)):
            abs_dist = [np.abs(gT[i] - fT[it]) for i in range(len(gT))]
            if not abs_dist == []:
                index_good = abs_dist.index(min(abs_dist))
                printv(f'  transfo #{fT[it]} --> use transfo #{gT[index_good]}', verbose)
                # copy transformation
                copy(f'{file_mat[iz][gT[index_good]]}Warp.nii.gz', f'{file_mat[iz][fT[it]]}Warp.nii.gz')
                # apply transformation
                sct_apply_transfo.main(argv=['-i', file_data_splitZ_splitT[fT[it]],
                                             '-d', file_target,
                                             '-w', f'{file_mat[iz][fT[it]]}Warp.nii.gz',
                                             '-o', file_data_splitZ_splitT_moco[fT[it]],
                                             '-x', param.interp,
                                             '-v', '0'])
            else:
                # exit program if no transformation exists.
                printv(f'\nERROR in {os.path.basename(__file__)}: No good transformation exist. Exit program.\n', verbose, 'error')
                sys.exit(2)

        # Merge data along T
        file_data_splitZ_moco.append(add_suffix(file, suffix))
        if todo != 'estimate':
            im_data_splitZ_splitT_moco = [Image(fname, mmap=False) for fname in file_data_splitZ_splitT_moco]
            im_out = concat_data(im_data_splitZ_splitT_moco, 3)
            im_out.absolutepath = file_data_splitZ_moco[iz]
            im_out.save(verbose=0)

    # If sagittal, merge along Z
    if param.is_sagittal:
        # TODO: im_out.dim is incorrect: Z value is one
        im_data_splitZ_moco = [Image(fname) for fname in file_data_splitZ_moco]
        im_out = concat_data(im_data_splitZ_moco, 2)
        dirname, basename, ext = extract_fname(file_data)
        path_out = os.path.join(dirname, basename + suffix + ext)
        im_out.absolutepath = path_out
        im_out.save(verbose=0)

    return file_mat, im_out


def register(param, file_src, file_dest, file_mat, file_out, im_mask=None):
    """
    Register two images by estimating slice-wise Tx and Ty transformations, which are regularized along Z. This function
    uses ANTs' isct_antsSliceRegularizedRegistration.

    :param param:
    :param file_src:
    :param file_dest:
    :param file_mat:
    :param file_out:
    :param im_mask: Image of mask, could be 2D or 3D
    :return:
    """

    # TODO: deal with mask

    # initialization
    failed_transfo = 0  # by default, failed matrix is 0 (i.e., no failure)
    do_registration = True

    # get metric radius (if MeanSquares, CC) or nb bins (if MI)
    if param.metric == 'MI':
        metric_radius = '16'
    else:
        metric_radius = '4'
    file_out_concat = file_out

    kw = dict()
    im_data = Image(file_src)  # TODO: pass argument to use antsReg instead of opening Image each time

    # register file_src to file_dest
    if param.todo == 'estimate' or param.todo == 'estimate_and_apply':
        # If orientation is sagittal, use antsRegistration in 2D mode
        # Note: the parameter --restrict-deformation is irrelevant with affine transfo

        if param.sampling == 'None':
            # 'None' sampling means 'fully dense' sampling
            # see https://github.com/ANTsX/ANTs/wiki/antsRegistration-reproducibility-issues
            sampling = param.sampling
        else:
            # param.sampling should be a float in [0,1], and means the
            # samplingPercentage that chooses a subset of points to
            # estimate from. We always use 'Regular' (evenly-spaced)
            # mode, though antsRegistration offers 'Random' as well.
            # Be aware: even 'Regular' is not fully deterministic:
            # > Regular includes a random perturbation on the grid sampling
            # - https://github.com/ANTsX/ANTs/issues/976#issuecomment-602313884
            sampling = f'Regular,{param.sampling}'

        if im_data.orientation[2] in 'LR':
            cmd = ['isct_antsRegistration',
                   '-d', '2',
                   '--transform', f'Affine[{param.gradStep}]',
                   '--metric', f'{param.metric}[{file_dest},{file_src},1,{metric_radius},{sampling}]',
                   '--convergence', param.iter,
                   '--shrink-factors', '1',
                   '--smoothing-sigmas', param.smooth,
                   '--verbose', '1',
                   '--output', f'[{file_mat},{file_out_concat}]']
            cmd.extend(get_interpolation('isct_antsRegistration', param.interp))
            if im_mask is not None:
                # if user specified a mask, make sure there are non-null voxels in the image before running the registration
                if np.count_nonzero(im_mask.data):
                    cmd.extend(['--masks', im_mask.absolutepath])
                else:
                    # Mask only contains zeros. Copying the image instead of estimating registration.
                    copy(file_src, file_out_concat, verbose=0)
                    do_registration = False
                    # TODO: create affine mat file with identity, in case used by -g 2
        # 3D mode
        else:
            cmd = ['isct_antsSliceRegularizedRegistration',
                   '--polydegree', param.poly,
                   '--transform', f'Translation[{param.gradStep}]',
                   '--metric', f'{param.metric}[{file_dest},{file_src},1,{metric_radius},{sampling}]',
                   '--iterations', param.iter,
                   '--shrinkFactors', '1',
                   '--smoothingSigmas', param.smooth,
                   '--verbose', '1',
                   '--output', f'[{file_mat},{file_out_concat}]']
            cmd.extend(get_interpolation('isct_antsSliceRegularizedRegistration', param.interp))
            if im_mask is not None:
                cmd.extend(['--mask', im_mask.absolutepath])
        # run command
        if do_registration:
            kw.update(dict(is_sct_binary=True))
            # reducing the number of CPU used for moco (see issue #201 and #2642)
            env = {**os.environ, **{"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1"}}
            status, output = run_proc(cmd, verbose=1 if param.verbose == 2 else 0, env=env, **kw)

    elif param.todo == 'apply':
        sct_apply_transfo.main(argv=['-i', file_src,
                                     '-d', file_dest,
                                     '-w', file_mat + param.suffix_mat,
                                     '-o', file_out_concat,
                                     '-x', param.interp,
                                     '-v', '0'])

    # check if output file exists
    # Note (from JCA): In the past, i've tried to catch non-zero output from ANTs function (via the 'status' variable),
    # but in some OSs, the function can fail while outputing zero. So as a pragmatic approach, I decided to go with
    # the "output file checking" approach, which is 100% sensitive.
    if not os.path.isfile(file_out_concat):
        # printv(output, verbose, 'error')
        printv(f'WARNING in {os.path.basename(__file__)}: No output. Maybe related to improper calculation of '
               'mutual information. Either the mask you provided is '
               'too small, or the subject moved a lot. If you see too '
               'many messages like this try with a bigger mask. '
               'Using previous transformation for this volume (if it '
               'exists).', param.verbose, 'warning')
        failed_transfo = 1

    # If sagittal, copy header (because ANTs screws it) and add singleton in 3rd dimension (for z-concatenation)
    if im_data.orientation[2] in 'LR' and do_registration:
        im_out = Image(file_out_concat)
        im_out.header = im_data.header
        im_out.data = np.expand_dims(im_out.data, 2)
        im_out.save(file_out, verbose=0)

    # return status of failure
    return failed_transfo


def spline(folder_mat, nt, nz, verbose, index_b0=[], graph=0):

    printv('\n\n\n------------------------------------------------------------------------------', verbose)
    printv('Spline Regularization along T: Smoothing Patient Motion...', verbose)

    file_mat = [[[] for i in range(nz)] for i in range(nt)]
    for it in range(nt):
        for iz in range(nz):
            file_mat[it][iz] = os.path.join(folder_mat, f'mat.T{it}_Z{iz}.txt')

    # Copying the existing Matrices to another folder
    old_mat = os.path.join(folder_mat, "old")
    try:
        os.makedirs(old_mat)
    except FileExistsError:
        pass

    # TODO
    for mat in glob.glob(os.path.join(folder_mat, '*.txt')):
        copy(mat, old_mat)

    printv('\nloading matrices...', verbose)
    X = [[[] for i in range(nt)] for i in range(nz)]
    Y = [[[] for i in range(nt)] for i in range(nz)]
    X_smooth = [[[] for i in range(nt)] for i in range(nz)]
    Y_smooth = [[[] for i in range(nt)] for i in range(nz)]
    for iz in range(nz):
        for it in range(nt):
            file = open(file_mat[it][iz])
            Matrix = np.loadtxt(file)
            file.close()

            X[iz][it] = Matrix[0, 3]
            Y[iz][it] = Matrix[1, 3]

    # Generate motion splines
    printv('\nGenerate motion splines...', verbose)
    T = np.arange(nt)
    if graph:
        import pylab as pl

    for iz in range(nz):

        spline = scipy.interpolate.UnivariateSpline(T, X[iz][:], w=None, bbox=[None, None], k=3, s=None)
        X_smooth[iz][:] = spline(T)

        if graph:
            pl.plot(T, X_smooth[iz][:], label='spline_smoothing')
            pl.plot(T, X[iz][:], marker='*', linestyle='None', label='original_val')
            if len(index_b0) != 0:
                T_b0 = [T[i_b0] for i_b0 in index_b0]
                X_b0 = [X[iz][i_b0] for i_b0 in index_b0]
                pl.plot(T_b0, X_b0, marker='D', linestyle='None', color='k', label='b=0')
            pl.title('X')
            pl.grid()
            pl.legend()
            pl.show()

        spline = scipy.interpolate.UnivariateSpline(T, Y[iz][:], w=None, bbox=[None, None], k=3, s=None)
        Y_smooth[iz][:] = spline(T)

        if graph:
            pl.plot(T, Y_smooth[iz][:], label='spline_smoothing')
            pl.plot(T, Y[iz][:], marker='*', linestyle='None', label='original_val')
            if len(index_b0) != 0:
                T_b0 = [T[i_b0] for i_b0 in index_b0]
                Y_b0 = [Y[iz][i_b0] for i_b0 in index_b0]
                pl.plot(T_b0, Y_b0, marker='D', linestyle='None', color='k', label='b=0')
            pl.title('Y')
            pl.grid()
            pl.legend()
            pl.show()

    # Storing the final Matrices
    printv('\nStoring the final Matrices...', verbose)
    for iz in range(nz):
        for it in range(nt):
            file = open(file_mat[it][iz])
            Matrix = np.loadtxt(file)
            file.close()

            Matrix[0, 3] = X_smooth[iz][it]
            Matrix[1, 3] = Y_smooth[iz][it]

            file = open(file_mat[it][iz], 'w')
            np.savetxt(file_mat[it][iz], Matrix, fmt="%s", delimiter='  ', newline='\n')
            file.close()

    printv('\n...Done. Patient motion has been smoothed', verbose)
    printv('------------------------------------------------------------------------------\n', verbose)
