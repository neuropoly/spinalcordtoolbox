#!/usr/bin/env python
#######################################################################################################################
#
# Merge images. See details in function "merge_images".
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Dominique Eden, Sara Dupont
# Modified: 2017-03-17
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

# TODO: parameter "almost_zero" might case problem if merging data with very low values (e.g. MD from diffusion)

import sys
import os

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_global_loglevel
from spinalcordtoolbox.utils.fs import tmp_create, rmtree
from spinalcordtoolbox.math import binarize

from spinalcordtoolbox.scripts import sct_apply_transfo


class Param:
    def __init__(self):
        self.fname_out = 'merged_images.nii.gz'
        self.interp = 'linear'
        self.rm_tmp = True
        self.verbose = 1
        self.almost_zero = 0.00000001


# PARSER
# ==========================================================================================
def get_parser():
    # Initialize the parser

    parser = SCTArgumentParser(
        description='Merge images to the same space'
    )
    mandatory = parser.add_argument_group("MANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        metavar=Metavar.file,
        nargs="+",
        help="Input images",
        required=True)
    mandatory.add_argument(
        "-d",
        metavar=Metavar.file,
        help="Destination image",
        required=True)
    mandatory.add_argument(
        "-w",
        nargs="+",
        metavar=Metavar.file,
        help="List of warping fields from input images to destination image",
        required=True)
    optional = parser.add_argument_group("OPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-x",
        metavar=Metavar.str,
        help="Interpolation for warping the input images to the destination image. Default is linear",
        required=False,
        default=Param().interp)
    optional.add_argument(
        "-o",
        metavar=Metavar.file,
        help="Output image",
        required=False,
        default=Param().fname_out)

    misc = parser.add_argument_group('MISC')
    misc.add_argument(
        "-r",
        type=bool,
        help='Remove temporary files.',
        required=False,
        default=Param().rm_tmp,
        choices=(0, 1))
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


def merge_images(list_fname_src, fname_dest, list_fname_warp, param):
    """
    Merge multiple source images onto destination space. All images are warped to the destination space and then added.
    To deal with overlap during merging (e.g. one voxel in destination image is shared with two input images), the
    resulting voxel is divided by the sum of the partial volume of each image. For example, if src(x,y,z)=1 is mapped to
    dest(i,j,k) with a partial volume of 0.5 (because destination voxel is bigger), then its value after linear interpolation
    will be 0.5. To account for partial volume, the resulting voxel will be: dest(i,j,k) = 0.5*0.5/0.5 = 0.5.
    Now, if two voxels overlap in the destination space, let's say: src(x,y,z)=1 and src2'(x',y',z')=1, then the
    resulting value will be: dest(i,j,k) = (0.5*0.5 + 0.5*0.5) / (0.5+0.5) = 0.5. So this function acts like a weighted
    average operator, only in destination voxels that share multiple source voxels.

    Parameters
    ----------
    list_fname_src
    fname_dest
    list_fname_warp
    param

    Returns
    -------

    """
    # create temporary folder
    path_tmp = tmp_create()

    # get dimensions of destination file
    nii_dest = Image(fname_dest)

    # initialize variables
    data = np.zeros([nii_dest.dim[0], nii_dest.dim[1], nii_dest.dim[2], len(list_fname_src)])
    partial_volume = np.zeros([nii_dest.dim[0], nii_dest.dim[1], nii_dest.dim[2], len(list_fname_src)])
    data_merge = np.zeros([nii_dest.dim[0], nii_dest.dim[1], nii_dest.dim[2]])

    # loop across files
    i_file = 0
    for fname_src in list_fname_src:

        # apply transformation src --> dest
        sct_apply_transfo.main(argv=[
            '-i', fname_src,
            '-d', fname_dest,
            '-w', list_fname_warp[i_file],
            '-x', param.interp,
            '-o', 'src_' + str(i_file) + '_template.nii.gz',
            '-v', str(param.verbose)])

        # create binary mask from input file by assigning one to all non-null voxels
        img = Image(fname_src)
        out = img.copy()
        out.data = binarize(out.data, param.almost_zero)
        out.save(path=f"src_{i_file}native_bin.nii.gz")

        # apply transformation to binary mask to compute partial volume
        sct_apply_transfo.main(argv=[
            '-i', 'src_' + str(i_file) + 'native_bin.nii.gz',
            '-d', fname_dest,
            '-w', list_fname_warp[i_file],
            '-x', param.interp,
            '-o', 'src_' + str(i_file) + '_template_partialVolume.nii.gz'])

        # open data
        data[:, :, :, i_file] = Image('src_' + str(i_file) + '_template.nii.gz').data
        partial_volume[:, :, :, i_file] = Image('src_' + str(i_file) + '_template_partialVolume.nii.gz').data
        i_file += 1

    # merge files using partial volume information (and convert nan resulting from division by zero to zeros)
    data_merge = np.divide(np.sum(data * partial_volume, axis=3), np.sum(partial_volume, axis=3))
    data_merge = np.nan_to_num(data_merge)

    # write result in file
    nii_dest.data = data_merge
    nii_dest.save(param.fname_out)

    # remove temporary folder
    if param.rm_tmp:
        rmtree(path_tmp)


# MAIN
# ==========================================================================================
def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)

    # create param objects
    param = Param()

    # set param arguments ad inputted by user
    list_fname_src = arguments.i
    fname_dest = arguments.d
    list_fname_warp = arguments.w
    param.fname_out = arguments.o

    # if arguments.ofolder is not None
    #     path_results = arguments.ofolder
    if arguments.x is not None:
        param.interp = arguments.x
    if arguments.r is not None:
        param.rm_tmp = arguments.r

    # check if list of input files and warping fields have same length
    assert len(list_fname_src) == len(list_fname_warp), "ERROR: list of files are not of the same length"

    # merge src images to destination image
    try:
        merge_images(list_fname_src, fname_dest, list_fname_warp, param)
    except Exception as e:
        printv(str(e), 1, 'error')

    display_viewer_syntax([fname_dest, os.path.abspath(param.fname_out)])


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

