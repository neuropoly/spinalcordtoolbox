#!/usr/bin/env python
#
# Function to segment the spinal cord using convolutional neural networks
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
from spinalcordtoolbox.image import Image, check_dim
from spinalcordtoolbox.deepseg_.sc import deep_segmentation_spinalcord
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.types import EmptyArrayError


def get_parser():
    parser = SCTArgumentParser(
        description="Spinal Cord Segmentation using convolutional networks. Reference: Gros et al. Automatic "
                    "segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional "
                    "neural networks. Neuroimage. 2019 Jan 1;184:901-915."
    )
    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        "-i",
        metavar=Metavar.file,
        help='Input image. Example: `t1.nii.gz`',
    )
    mandatory.add_argument(
        "-c",
        help="Type of image contrast.",
        choices=('t1', 't2', 't2s', 'dwi'),
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
        metavar=Metavar.str,
        help='Input centerline file (to use with flag `-centerline` file). Example: `t2_centerline_manual.nii.gz`')
    optional.add_argument(
        "-thr",
        type=float,
        help="Binarization threshold (between `0` and `1`) to apply to the segmentation prediction. Set to `-1` for no "
             "binarization (i.e. soft segmentation output). The default threshold is specific to each contrast and was "
             "estimated using an optimization algorithm. More details at: "
             "https://github.com/sct-pipeline/deepseg-threshold.",
        metavar=Metavar.float)
    optional.add_argument(
        "-brain",
        type=int,
        help='Indicate if the input image contains brain sections (to speed up segmentation). Only use with '
             '`-centerline cnn`. (default: `1` for T1/T2 contrasts, `0` for T2*/DWI contrasts)',
        choices=(0, 1))
    optional.add_argument(
        "-kernel",
        help="Choice of kernel shape for the CNN. Segmentation with 3D kernels is slower than with 2D kernels.",
        choices=('2d', '3d'),
        default="2d")
    optional.add_argument(
        "-ofolder",
        metavar=Metavar.str,
        help='Output folder.',
        action=ActionCreateFolder,
        default=os.getcwd())
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help='Output filename. Example: `spinal_seg.nii.gz`'),
    optional.add_argument(
        '-qc',
        metavar=Metavar.str,
        help='The path where the quality control generated content will be saved')
    optional.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the dataset the process was run on',)
    optional.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the subject the process was run on',)
    optional.add_argument(
        '-qc-contrast',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the contrast the process was run on',)

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
    warn(stylize(
        "`sct_deepseg_sc` is deprecated, and will be removed in a future version of SCT. Please use "
        "`sct_deepseg spinalcord` instead.", ["Red", "Bold"]
        ), DeprecationWarning
    )
    sleep(3)  # Give the user 3 seconds to read the message

    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    fname_image = os.path.abspath(arguments.i)
    contrast_type = arguments.c

    ctr_algo = arguments.centerline

    if arguments.brain is None:
        if contrast_type in ['t2s', 'dwi']:
            brain_bool = False
        if contrast_type in ['t1', 't2']:
            brain_bool = True
    else:
        brain_bool = bool(arguments.brain)

    if bool(arguments.brain) and ctr_algo == 'svm':
        printv('Please only use the flag "-brain 1" with "-centerline cnn".', 1, 'warning')
        sys.exit(1)

    kernel_size = arguments.kernel
    if kernel_size == '3d' and contrast_type == 'dwi':
        kernel_size = '2d'
        printv('3D kernel model for dwi contrast is not available. 2D kernel model is used instead.',
               type="warning")

    if ctr_algo == 'file' and arguments.file_centerline is None:
        printv('Please use the flag -file_centerline to indicate the centerline filename.', 1, 'warning')
        sys.exit(1)

    if arguments.file_centerline is not None:
        manual_centerline_fname = arguments.file_centerline
        ctr_algo = 'file'
    else:
        manual_centerline_fname = None

    if arguments.o is not None:
        fname_out = arguments.o
    else:
        path, file_name, ext = extract_fname(fname_image)
        fname_out = f'{file_name}_seg{ext}'

    threshold = arguments.thr

    if threshold is not None:
        if threshold > 1.0 or (threshold < 0.0 and threshold != -1.0):
            raise SyntaxError("Threshold should be between 0 and 1, or equal to -1 (no threshold)")

    remove_temp_files = arguments.r

    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject
    qc_contrast = arguments.qc_contrast
    output_folder = arguments.ofolder

    # check if input image is 2D or 3D
    check_dim(fname_image, dim_lst=[2, 3])

    # Segment image

    im_image = Image(fname_image)
    # note: below we pass im_image.copy() otherwise the field absolutepath becomes None after execution of this function
    try:
        im_seg, im_image_RPI_upsamp, im_seg_RPI_upsamp = \
            deep_segmentation_spinalcord(im_image.copy(), contrast_type, ctr_algo=ctr_algo,
                                         ctr_file=manual_centerline_fname, brain_bool=brain_bool,
                                         kernel_size=kernel_size, threshold_seg=threshold,
                                         remove_temp_files=remove_temp_files, verbose=verbose)
    except EmptyArrayError as e:
        printv(f"Spinal cord could not be detected for {fname_image}\n"
               f"    {e.__class__.__name__}: '{e}'", 1, 'error')

    # Save segmentation
    fname_seg = os.path.abspath(os.path.join(output_folder, fname_out))
    im_seg.save(fname_seg)

    # Generate QC report
    if path_qc is not None:
        generate_qc(fname_image, fname_seg=fname_seg, args=argv, path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, contrast=qc_contrast, process='sct_deepseg_sc')
    display_viewer_syntax([fname_image, fname_seg], im_types=['anat', 'seg'], opacities=['', '0.7'], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
