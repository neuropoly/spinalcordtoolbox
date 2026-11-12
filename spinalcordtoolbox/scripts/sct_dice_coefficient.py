#!/usr/bin/env python
#
# This program is a wrapper for the isct_dice_coefficient binary
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
import os
from typing import Sequence

from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar
from spinalcordtoolbox.utils.sys import init_sct, run_proc, printv, set_loglevel
from spinalcordtoolbox.utils.fs import tmp_create, copy, extract_fname, rmtree
from spinalcordtoolbox.image import Image, add_suffix
from spinalcordtoolbox.math import binarize


def get_parser():
    parser = SCTArgumentParser(
        description='Compute the Dice Coefficient. '
                    'N.B.: indexing (in both time and space) starts with 0 not 1! Inputting `-1` for a '
                    'size will set it to the full image extent for that dimension.'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        help='First input image. Example: `t2_seg.nii.gz`',
    )
    mandatory.add_argument(
        '-d',
        help='Second input image. Example: `t2_manual_seg.nii.gz`',
        metavar=Metavar.file,
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-2d-slices',
        dest='_2d_slices',
        type=int,
        help='Compute DC on 2D slices in the specified dimension',
        choices=(0, 1, 2))
    optional.add_argument(
        '-b',
        metavar=Metavar.list,
        help='Bounding box with the coordinates of the origin and the size of the box as follow: '
             'x_origin,x_size,y_origin,y_size,z_origin,z_size. Example: 5,10,5,10,10,15')
    optional.add_argument(
        '-bmax',
        type=int,
        help='Use maximum bounding box of the images union to compute DC.',
        choices=(0, 1))
    optional.add_argument(
        '-bzmax',
        type=int,
        help='Use maximum bounding box of the images union in the "Z" direction to compute DC.',
        choices=(0, 1))
    optional.add_argument(
        '-bin',
        type=int,
        help='Binarize image before computing DC. (Put non-zero-voxels to 1)',
        choices=(0, 1))
    optional.add_argument(
        '-o',
        metavar=Metavar.str,
        help='Output file with DC results (.txt). Example: `dice_coeff.txt`')

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_tempfile_args()

    return parser


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    fname_input1 = arguments.i
    fname_input2 = arguments.d

    tmp_dir = tmp_create(basename="dice-coefficient")  # create tmp directory
    tmp_dir = os.path.abspath(tmp_dir)

    # copy input files to tmp directory
    fname_input1_tmp = f'tmp1_{"".join(extract_fname(fname_input1)[1:])}'
    fname_input2_tmp = f'tmp2_{"".join(extract_fname(fname_input2)[1:])}'
    copy(fname_input1, os.path.join(tmp_dir, fname_input1_tmp))
    copy(fname_input2, os.path.join(tmp_dir, fname_input2_tmp))
    fname_input1 = fname_input1_tmp
    fname_input2 = fname_input2_tmp

    curdir = os.getcwd()
    os.chdir(tmp_dir)  # go to tmp directory

    im_1 = Image(fname_input1)
    im_2 = Image(fname_input2)

    if arguments.bin is not None:
        im_1.data = binarize(im_1.data, 0)
        fname_input1_bin = add_suffix(fname_input1, '_bin')
        im_1.save(fname_input1_bin, mutable=True)

        im_2.data = binarize(im_2.data, 0)
        fname_input2_bin = add_suffix(fname_input2, '_bin')
        im_2.save(fname_input2_bin, mutable=True)

        # Use binarized images in subsequent steps
        fname_input1 = fname_input1_bin
        fname_input2 = fname_input2_bin

    # copy header of im_1 to im_2
    im_2.header = im_1.header
    im_2.save()

    cmd = ['isct_dice_coefficient', fname_input1, fname_input2]

    if arguments._2d_slices is not None:
        cmd.extend(['-2d-slices', arguments._2d_slices])
    if arguments.b is not None:
        cmd.append('-b')
        cmd.extend(arguments.b.split(','))
    if arguments.bmax == 1:
        cmd.append('-bmax')
    if arguments.bzmax == 1:
        cmd.append('-bzmax')
    if arguments.o is not None:
        path_output, fname_output, ext = extract_fname(arguments.o)
        cmd.extend(['-o', fname_output + ext])

    rm_tmp = bool(arguments.r)

    # # Computation of Dice coefficient using Python implementation.
    # # commented for now as it does not cover all the feature of isct_dice_coefficient
    # #from spinalcordtoolbox.image import Image, compute_dice
    # #dice = compute_dice(Image(fname_input1), Image(fname_input2), mode='3d', zboundaries=False)
    # #printv(f'Dice (python-based) = {dice}', verbose)

    status, output = run_proc(cmd, verbose, is_sct_binary=True)

    os.chdir(curdir)  # go back to original directory

    # copy output file into original directory
    if arguments.o is not None:
        copy(os.path.join(tmp_dir, fname_output + ext), os.path.join(path_output, fname_output + ext))

    # remove tmp_dir
    if rm_tmp:
        rmtree(tmp_dir)

    printv(output, verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
