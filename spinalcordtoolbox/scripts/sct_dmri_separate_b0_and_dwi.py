#!/usr/bin/env python
#
# Separate b=0 and DW images from diffusion dataset
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
import math
import time
import os
from typing import Sequence

import numpy as np

from spinalcordtoolbox.image import Image, generate_output_file, convert
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.fs import tmp_create, copy, extract_fname, rmtree

from spinalcordtoolbox.scripts.sct_image import split_data, concat_data


class Param:
    def __init__(self):
        self.debug = 0
        self.average = 1
        self.remove_temp_files = 1
        self.verbose = 1
        self.bval_min = 100  # in case user does not have min bvalues at 0, set threshold.


def get_parser():
    param_default = Param()
    parser = SCTArgumentParser(
        description="Separate b=0 and DW images from diffusion dataset. The output files will have a suffix "
                    "(_b0 and _dwi) appended to the input file name."
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        help="Diffusion data. Example: `dmri.nii.gz`"
    )
    mandatory.add_argument(
        '-bvec',
        metavar=Metavar.file,
        help="Bvecs file. Example: `bvecs.txt`"
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-a',
        type=int,
        choices=(0, 1),
        default=param_default.average,
        help="Average b=0 and DWI data. 0 = no, 1 = yes"
    )
    optional.add_argument(
        '-bval',
        metavar=Metavar.file,
        default="",
        help='bvals file. Used to identify low b-values (in case different from 0). Example: `bvals.txt`',
    )
    optional.add_argument(
        '-bvalmin',
        type=float,
        metavar=Metavar.float,
        help='B-value threshold (in s/mm2) below which data is considered as b=0. Example: `50.0`',
    )
    optional.add_argument(
        '-ofolder',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        default='.',
        help='Output folder.',
    )

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_tempfile_args()

    return parser


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    # initialize parameters
    param = Param()

    fname_data = arguments.i
    fname_bvecs = arguments.bvec
    average = arguments.a
    remove_temp_files = arguments.r
    path_out = arguments.ofolder

    fname_bvals = arguments.bval
    if arguments.bvalmin:
        param.bval_min = arguments.bvalmin

    # Initialization
    start_time = time.time()

    # printv(arguments)
    printv('\nInput parameters:', verbose)
    printv(f'  input file ............{fname_data}', verbose)
    printv(f'  bvecs file ............{fname_bvecs}', verbose)
    printv(f'  bvals file ............{fname_bvals}', verbose)
    printv(f'  average ...............{average}', verbose)

    # Get full path
    fname_data = os.path.abspath(fname_data)
    fname_bvecs = os.path.abspath(fname_bvecs)
    if fname_bvals:
        fname_bvals = os.path.abspath(fname_bvals)

    # Extract path, file and extension
    path_data, file_data, ext_data = extract_fname(fname_data)

    # create temporary folder
    path_tmp = tmp_create(basename="dmri-separate-b0-and-dwi")

    # copy files into tmp folder and convert to nifti
    printv('\nCopy files into temporary folder...', verbose)
    ext = '.nii'
    dmri_name = 'dmri'
    b0_name = f'{file_data}_b0'
    b0_mean_name = f'{b0_name}_mean'
    dwi_name = f'{file_data}_dwi'
    dwi_mean_name = f'{dwi_name}_mean'
    im_dmri = convert(Image(fname_data))
    im_dmri.save(os.path.join(path_tmp, dmri_name + ext), mutable=True, verbose=verbose)
    copy(fname_bvecs, os.path.join(path_tmp, "bvecs"), verbose=verbose)

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Get size of data
    printv('\nGet dimensions data...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = im_dmri.dim
    printv(f'.. {nx} x {ny} x {nz} x {nt}', verbose)

    # Identify b=0 and DWI images
    printv(fname_bvals)
    index_b0, index_dwi, nb_b0, nb_dwi = identify_b0(fname_bvecs, fname_bvals, param.bval_min, verbose)

    # Split into T dimension
    printv('\nSplit along T dimension...', verbose)
    im_dmri_split_list = split_data(im_dmri, 3)
    for im_d in im_dmri_split_list:
        im_d.save()

    # Merge b=0 images
    printv('\nMerge b=0...', verbose)
    fname_in_list_b0 = []
    for it in range(nb_b0):
        fname_in_list_b0.append(f'{dmri_name}_T{index_b0[it]:04}{ext}')
    im_in_list_b0 = [Image(fname) for fname in fname_in_list_b0]
    concat_data(im_in_list_b0, 3).save(b0_name + ext)

    # Average b=0 images
    if average:
        printv('\nAverage b=0...', verbose)
        img = Image(b0_name + ext)
        out = img.copy()
        dim_idx = 3
        if len(np.shape(img.data)) < dim_idx + 1:
            raise ValueError("Expecting image with 4 dimensions!")
        out.data = np.mean(out.data, dim_idx)
        out.save(path=b0_mean_name + ext)

    # Merge DWI
    fname_in_list_dwi = []
    for it in range(nb_dwi):
        fname_in_list_dwi.append(f'{dmri_name}_T{index_dwi[it]:04}{ext}')
    im_in_list_dwi = [Image(fname) for fname in fname_in_list_dwi]
    concat_data(im_in_list_dwi, 3).save(dwi_name + ext)

    # Average DWI images
    if average:
        printv('\nAverage DWI...', verbose)
        img = Image(dwi_name + ext)
        out = img.copy()
        dim_idx = 3
        if len(np.shape(img.data)) < dim_idx + 1:
            raise ValueError("Expecting image with 4 dimensions!")
        out.data = np.mean(out.data, dim_idx)
        out.save(path=dwi_mean_name + ext)

    # come back
    os.chdir(curdir)

    # Generate output files
    fname_b0 = os.path.abspath(os.path.join(path_out, b0_name + ext_data))
    fname_dwi = os.path.abspath(os.path.join(path_out, dwi_name + ext_data))
    fname_b0_mean = os.path.abspath(os.path.join(path_out, b0_mean_name + ext_data))
    fname_dwi_mean = os.path.abspath(os.path.join(path_out, dwi_mean_name + ext_data))
    printv('\nGenerate output files...', verbose)
    generate_output_file(os.path.join(path_tmp, b0_name + ext), fname_b0, verbose=verbose)
    generate_output_file(os.path.join(path_tmp, dwi_name + ext), fname_dwi, verbose=verbose)
    if average:
        generate_output_file(os.path.join(path_tmp, b0_mean_name + ext), fname_b0_mean, verbose=verbose)
        generate_output_file(os.path.join(path_tmp, dwi_mean_name + ext), fname_dwi_mean, verbose=verbose)

    # Remove temporary files
    if remove_temp_files == 1:
        printv('\nRemove temporary files...', verbose)
        rmtree(path_tmp, verbose=verbose)

    # display elapsed time
    elapsed_time = time.time() - start_time
    printv(f'\nFinished! Elapsed time: {int(np.round(elapsed_time))}s', verbose)

    return fname_b0, fname_b0_mean, fname_dwi, fname_dwi_mean


# ==========================================================================================
# identify b=0 and DW images
# ==========================================================================================
def identify_b0(fname_bvecs, fname_bvals, bval_min, verbose):

    # Identify b=0 and DWI images
    printv('\nIdentify b=0 and DWI images...', verbose)
    index_b0 = []
    index_dwi = []

    # if bval is not provided
    if not fname_bvals:
        # Open bvecs file
        bvecs = []
        with open(fname_bvecs) as f:
            for line in f:
                bvecs_new = [x for x in map(float, line.split())]
                bvecs.append(bvecs_new)

        # Check if bvecs file is nx3
        if not len(bvecs[0][:]) == 3:
            printv('  WARNING: bvecs file is 3xn instead of nx3. Consider using sct_dmri_transpose_bvecs.', verbose, 'warning')
            printv('  Transpose bvecs...', verbose)
            # transpose bvecs
            bvecs = list(zip(*bvecs))

        # get number of lines
        nt = len(bvecs)

        # identify b=0 and dwi
        for it in range(0, nt):
            if math.sqrt(math.fsum([i**2 for i in bvecs[it]])) < 0.01:
                index_b0.append(it)
            else:
                index_dwi.append(it)

    # if bval is provided
    else:

        # Open bvals file
        from dipy.io import read_bvals_bvecs
        bvals, bvecs = read_bvals_bvecs(fname_bvals, fname_bvecs)

        # get number of lines
        nt = len(bvals)

        # Identify b=0 and DWI images
        printv('\nIdentify b=0 and DWI images...', verbose)
        for it in range(0, nt):
            if bvals[it] < bval_min:
                index_b0.append(it)
            else:
                index_dwi.append(it)

    # check if no b=0 images were detected
    if index_b0 == []:
        printv('ERROR: no b=0 images detected. Maybe you are using non-null low bvals? in that case use flag -bvalmin. Exit program.', 1, 'error')
        sys.exit(2)

    # display stuff
    nb_b0 = len(index_b0)
    nb_dwi = len(index_dwi)
    printv(f'  Number of b=0: {nb_b0} {index_b0}', verbose)
    printv(f'  Number of DWI: {nb_dwi} {index_dwi}', verbose)

    # return
    return index_b0, index_dwi, nb_b0, nb_dwi


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
