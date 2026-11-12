#!/usr/bin/env python
#
# This command-line tool is the interface for the deepseg_gm API
# that implements the model for the Spinal Cord Gray Matter Segmentation.
#
# Reference paper:
#     Perone, C. S., Calabrese, E., & Cohen-Adad, J. (2017).
#     Spinal cord gray matter segmentation using deep dilated convolutions.
#     URL: https://doi.org/10.48550/arXiv.1710.01269
#
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE


import sys
import os
from typing import Sequence

from spinalcordtoolbox.utils.sys import init_sct, set_loglevel
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser, display_viewer_syntax
from spinalcordtoolbox.image import add_suffix
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.deepseg_.gm import segment_file


def get_parser():
    parser = SCTArgumentParser(
        description='Spinal Cord Gray Matter (GM) Segmentation using deep dilated convolutions. The contrast of the '
                    'input image must be similar to a T2*-weighted image: WM dark, GM bright and CSF bright. '
                    'Reference: Perone CS, Calabrese E, Cohen-Adad J. Spinal cord gray matter segmentation using deep '
                    'dilated convolutions. Sci Rep 2018;8(1):5966.'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        "-i",
        help="Image filename to segment (3D volume). Example: `t2s.nii.gz`.",
        metavar=Metavar.file
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-o",
        help="Output segmentation file name. Example: `sc_gm_seg.nii.gz`",
        metavar=Metavar.file)
    optional.add_argument(
        "-m",
        help="Model to use (large or challenge). "
             "The model 'large' will be slower but "
             "will yield better results. The model "
             "'challenge' was built using data from "
             "the following challenge: goo.gl/h4AVar.",
        choices=('large', 'challenge'),
        default='large')
    optional.add_argument(
        "-thr",
        type=float,
        help='Threshold to apply in the segmentation predictions, use 0 (zero) to disable it.',
        metavar=Metavar.float,
        default=0.999)
    optional.add_argument(
        "-t",
        help="Enable TTA (test-time augmentation). "
             "Better results, but takes more time and "
             "provides non-deterministic results.",
        metavar='')

    misc = parser.misc_arggroup
    misc.add_argument(
        '-qc',
        help="The path where the quality control generated content will be saved.",
        metavar=Metavar.str)
    misc.add_argument(
        '-qc-dataset',
        help='If provided, this string will be mentioned in the QC report as the dataset the process was run on',
        metavar=Metavar.str)
    misc.add_argument(
        '-qc-subject',
        help='If provided, this string will be mentioned in the QC report as the subject the process was run on',
        metavar=Metavar.str)
    misc.add_argument(
        '-qc-contrast',
        help='If provided, this string will be mentioned in the QC report as the contrast the process was run on',
        metavar=Metavar.str)

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    input_filename = arguments.i
    if arguments.o is not None:
        output_filename = arguments.o
    else:
        output_filename = add_suffix(input_filename, '_gmseg')

    use_tta = arguments.t
    model_name = arguments.m
    threshold = arguments.thr

    if threshold > 1.0 or threshold < 0.0:
        raise RuntimeError("Threshold should be between 0.0 and 1.0.")

    # Threshold zero means no thresholding
    if threshold == 0.0:
        threshold = None

    out_fname = segment_file(input_filename, output_filename,
                             model_name, threshold, int(verbose),
                             use_tta)

    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject
    qc_contrast = arguments.qc_contrast
    if path_qc is not None:
        generate_qc(fname_in1=input_filename, fname_seg=out_fname, args=argv, path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, contrast=qc_contrast, process='sct_deepseg_gm')

    display_viewer_syntax(
        [input_filename, format(out_fname)],
        im_types=['anat', 'seg'],
        opacities=['1', '0.7'],
        verbose=verbose,
    )


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
