#!/usr/bin/env python
#
# Function to segment the multiple sclerosis lesions using convolutional neural networks
#
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import os
import sys
from typing import Sequence
import textwrap

from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.fs import extract_fname


def get_parser():
    parser = SCTArgumentParser(
        description='MS lesion Segmentation using convolutional networks. Reference: Gros C et al. Automatic '
                    'segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional '
                    'neural networks. Neuroimage. 2018 Oct 6;184:901-915.'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        "-i",
        help='Input image. Example: `t2.nii.gz`',
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        "-c",
        help=textwrap.dedent("""
            Type of image contrast.

              - `t2`: T2w scan with isotropic or anisotropic resolution.
              - `t2_ax`: T2w scan with axial orientation and thick slices.
              - `t2s`: T2*w scan with axial orientation and thick slices.
        """),
        choices=('t2', 't2_ax', 't2s'),
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-centerline",
        help=textwrap.dedent("""
            Method used for extracting the centerline:

              - `svm`: Automatic detection using Support Vector Machine algorithm.
              - `cnn`: Automatic detection using Convolutional Neural Network.
              - `viewer`: Semi-automatic detection using manual selection of a few points with an interactive viewer followed by regularization.
              - `file`: Use an existing centerline (use with flag `-file_centerline`)
        """),
        choices=('svm', 'cnn', 'viewer', 'file'),
        default="svm")
    optional.add_argument(
        "-file_centerline",
        help='Input centerline file (to use with flag `-centerline` file). Example: `t2_centerline_manual.nii.gz`',
        metavar=Metavar.str,)
    optional.add_argument(
        "-brain",
        type=int,
        help='Indicate if the input image contains brain sections (to speed up segmentation). This flag is only '
             'effective with `-centerline cnn`.',
        choices=(0, 1),
        default=1)
    optional.add_argument(
        "-ofolder",
        help='Output folder.',
        action=ActionCreateFolder,
        metavar=Metavar.str,
        default=os.getcwd())

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_tempfile_args()

    return parser


def main(argv: Sequence[str]):
    """Main function."""
    # METHOD IS DEPRECATED, WARN THE USER AND UPDATE THEM ON WHAT TO USE INSTEAD
    from warnings import warn
    from spinalcordtoolbox.utils.sys import stylize
    from time import sleep
    warn(
        stylize(
            "`sct_deepseg_lesion` is deprecated, and will be removed in a future version of SCT. Please use "
            "`sct_deepseg ms_lesion` instead.", ["Red", "Bold"]
        ), DeprecationWarning
    )
    sleep(3)  # Give the user 3 seconds to read the message

    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    fname_image = arguments.i
    contrast_type = arguments.c

    ctr_algo = arguments.centerline

    brain_bool = bool(arguments.brain)
    if arguments.brain is None and contrast_type in ['t2s', 't2_ax']:
        brain_bool = False

    output_folder = arguments.ofolder

    if ctr_algo == 'file' and arguments.file_centerline is None:
        printv('Please use the flag -file_centerline to indicate the centerline filename.', 1, 'error')
        sys.exit(1)

    if arguments.file_centerline is not None:
        manual_centerline_fname = arguments.file_centerline
        ctr_algo = 'file'
    else:
        manual_centerline_fname = None

    remove_temp_files = arguments.r

    printv('\nMethod:'
           f'\n\tCenterline algorithm: {ctr_algo}'
           f'\n\tAssumes brain section included in the image: {brain_bool}\n')

    # Segment image
    from spinalcordtoolbox.image import Image
    from spinalcordtoolbox.deepseg_.lesion import deep_segmentation_MSlesion
    im_image = Image(fname_image)
    im_seg, im_labels_viewer, im_ctr = deep_segmentation_MSlesion(im_image, contrast_type, ctr_algo=ctr_algo, ctr_file=manual_centerline_fname,
                                                                  brain_bool=brain_bool, remove_temp_files=remove_temp_files, verbose=verbose)

    # Save segmentation
    fname_seg = os.path.abspath(os.path.join(
        output_folder, f'{extract_fname(fname_image)[1]}_lesionseg{extract_fname(fname_image)[2]}'))
    im_seg.save(fname_seg)

    if ctr_algo == 'viewer':
        # Save labels
        fname_labels = os.path.abspath(os.path.join(
            output_folder, f'{extract_fname(fname_image)[1]}_labels-centerline{extract_fname(fname_image)[2]}'))
        im_labels_viewer.save(fname_labels)

    if verbose == 2:
        # Save ctr
        fname_ctr = os.path.abspath(os.path.join(
            output_folder, f'{extract_fname(fname_image)[1]}_centerline{extract_fname(fname_image)[2]}'))
        im_ctr.save(fname_ctr)

    display_viewer_syntax([fname_image, fname_seg], im_types=['anat', 'seg'], opacities=['', '0.7'], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
