#!/usr/bin/env python
#
# Extraction of the spinal cord centerline
#
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import os
import sys
from typing import Sequence
import textwrap

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline, _call_viewer_centerline
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.fs import extract_fname


def get_parser():
    parser = SCTArgumentParser(
        description=(
            "This function extracts the spinal cord centerline. Three methods are available: 'optic' (automatic), "
            "'viewer' (manual), and 'fitseg' (applied on segmented image). These functions output (i) a NIFTI file "
            "with labels corresponding to the discrete centerline, and (ii) a csv file containing the float (more "
            "precise) coordinates of the centerline in the RPI orientation. \n"
            "\n"
            "Reference: C Gros, B De Leener, et al. Automatic spinal cord localization, robust to MRI contrast using "
            "global curve optimization (2017). doi.org/10.1016/j.media.2017.12.001"
        )
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        help="Input image. Example: `t1.nii.gz`"
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-method",
        choices=['optic', 'viewer', 'fitseg'],
        default='optic',
        help=textwrap.dedent("""
            Method used for extracting the centerline.

              - `optic`: automatic spinal cord detection method
              - `viewer`: manual selection a few points followed by interpolation
              - `fitseg`: fit a regularized centerline on an already-existing cord segmentation. This method will interpolate if any slices are missing. Also, if  `-extrapolation 1` is specified, this method will extrapolate beyond the segmentation boundaries (i.e., every axial slice will exhibit a centerline pixel).
        """),  # noqa: E501 (line too long)
    )
    optional.add_argument(
        "-c",
        choices=['t1', 't2', 't2s', 'dwi'],
        help="Type of image contrast. Only relevant with `-method optic`."
    )
    optional.add_argument(
        "-centerline-algo",
        choices=['polyfit', 'bspline', 'linear', 'nurbs'],
        default='bspline',
        help="Algorithm for centerline fitting. Only relevant with `-method fitseg`."
    )
    optional.add_argument(
        "-centerline-smooth",
        metavar=Metavar.int,
        type=int,
        default=30,
        help="Degree of smoothing for centerline fitting. Only relevant with `-centerline-algo {bspline, linear}`."
    )
    optional.add_argument(
        "-centerline-soft",
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Binary or soft centerline. `0` = binarized, `1` = soft. Only relevant with `-method fitseg`."
    )
    optional.add_argument(
        "-space",
        metavar=Metavar.str,
        type=str,
        choices=["pix", "phys"],
        default="pix",
        help="The coordinate space to use for units when outputting the centerline coordinates to a .csv file."
             "'pix'=pixel dimensions, 'phys'=physical dimensions."
    )
    optional.add_argument(
        "-extrapolation",
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Extrapolate beyond the segmentation boundaries. `0` = no extrapolation, `1` = extrapolation. Only relevant with `-method fitseg`."
             "Note: `-extrapolation 1` works best with lower-order (linear, nurbs) centerline fitting algorithms"
    )
    optional.add_argument(
        "-o",
        metavar=Metavar.file,
        help="File name for the centerline output file. If file extension is not provided, "
             "`.nii.gz` will be used by default. If `-o` is not provided, then the output file will "
             "be the input with suffix `_centerline`. Example: `centerline_optic.nii.gz`"
    )
    optional.add_argument(
        "-gap",
        metavar=Metavar.float,
        type=float,
        default=20.0,
        help="Gap in mm between manually selected points. Only with method=viewer."
    )
    optional.add_argument(
        "-qc",
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="The path where the quality control generated content will be saved."
    )
    optional.add_argument(
        "-qc-dataset",
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the dataset the process was run on."
    )
    optional.add_argument(
        "-qc-subject",
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the subject the process was run on."
    )
    optional.add_argument(
        "-qc-contrast",
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the contrast the process was run on."
    )

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_tempfile_args()

    return parser


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    # Input filename
    fname_input_data = arguments.i
    fname_data = os.path.abspath(fname_input_data)

    # Method used
    method = arguments.method

    # Contrast type
    contrast_type = arguments.c

    # Contrast must be specified if method is optic
    if method == 'optic' and not contrast_type:
        printv("ERROR: -c is a mandatory argument when using '-method optic'.", type='error')

    # Soft centerline option can only be used with fitseg method
    if arguments.centerline_soft == 1 and method != 'fitseg':
        printv("ERROR: -centerline-soft can only be used with '-method fitseg'.", type='error')

    # Extrapolation option can only be used with fitseg method
    if arguments.extrapolation == 1 and method != 'fitseg':
        printv("ERROR: -extrapolation can only be used with '-method fitseg'.", type='error')

    # Gap between slices
    interslice_gap = arguments.gap

    param_centerline = ParamCenterline(
        algo_fitting=arguments.centerline_algo,
        smooth=arguments.centerline_smooth,
        minmax=(not arguments.extrapolation),  # NB: 'extrapolation=0' --> 'minmax=True' and vice versa
        soft=arguments.centerline_soft)

    # Output folder
    if arguments.o is not None:
        path_data, file_data, ext_data = extract_fname(arguments.o)
        if not ext_data:
            ext_data = '.nii.gz'
        file_output = os.path.join(path_data, file_data+ext_data)
    else:
        path_data, file_data, ext_data = extract_fname(fname_data)
        file_output = os.path.join(path_data, f'{file_data}_centerline.nii.gz')

    if method == 'viewer':
        # Manual labeling of cord centerline
        im_labels = _call_viewer_centerline(Image(fname_data), interslice_gap=interslice_gap)
    elif method == 'fitseg':
        im_labels = Image(fname_data)
    elif method == 'optic':
        # Automatic detection of cord centerline
        im_labels = Image(fname_data)
        param_centerline.algo_fitting = 'optic'
        param_centerline.contrast = contrast_type
    else:
        printv(f"ERROR: The selected method is not available: {method}. Please look at the help.", type='error')
        return

    # Extrapolate and regularize (or detect if optic) cord centerline
    im_centerline, arr_centerline, _, _ = get_centerline(im_labels,
                                                         param=param_centerline,
                                                         space=arguments.space,
                                                         verbose=verbose,
                                                         remove_temp_files=arguments.r)

    # save centerline as nifti (discrete) and csv (continuous) files
    im_centerline.save(file_output)
    # Replace `.nii.gz` extension with `.csv` extension
    path_output, filename_output, _ = extract_fname(file_output)
    np.savetxt(os.path.join(path_output, f"{filename_output}.csv"), arr_centerline.transpose(), delimiter=",")

    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject
    qc_contrast = arguments.qc_contrast

    # Generate QC report
    if path_qc is not None:
        generate_qc(fname_input_data, fname_seg=file_output, args=argv, path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, contrast=qc_contrast, process='sct_get_centerline')

    im_type_ctl = 'softseg' if arguments.centerline_soft else 'seg'
    display_viewer_syntax([fname_input_data, file_output], im_types=['anat', im_type_ctl], opacities=['', '0.7'], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
