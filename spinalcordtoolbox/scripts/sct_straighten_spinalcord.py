#!/usr/bin/env python
#
# This program takes as input an anatomic image and the centerline or segmentation of its spinal cord (that you can get
# using sct_get_centerline.py or sct_segmentation_propagation) and returns the anatomic image where the spinal
# cord was straightened.
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
import os
from typing import Sequence
import textwrap

from spinalcordtoolbox.straightening import SpinalCordStraightener
from spinalcordtoolbox.centerline.core import ParamCenterline
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.utils.shell import (SCTArgumentParser, Metavar, ActionCreateFolder, display_viewer_syntax,
                                           list_type)
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel


def get_parser():
    parser = SCTArgumentParser(
        description="This program takes as input an anatomic image and the spinal cord centerline (or "
                    "segmentation), and returns the an image of a straightened spinal cord. Reference: "
                    "De Leener B, Mangeat G, Dupont S, Martin AR, Callot V, Stikov N, Fehlings MG, "
                    "Cohen-Adad J. Topologically-preserving straightening of spinal cord MRI. J Magn "
                    "Reson Imaging. 2017 Oct;46(4):1209-1219"
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        "-i",
        metavar=Metavar.file,
        help='Input image with curved spinal cord. Example: `t2.nii.gz`')
    mandatory.add_argument(
        "-s",
        metavar=Metavar.file,
        help=textwrap.dedent("""
            Spinal cord centerline (or segmentation) of the input image. To obtain the centerline, you can use sct_get_centerline.
            To obtain the segmentation you can use `sct_propseg` or `sct_deepseg spinalcord`.
            Example: `centerline.nii.gz`
        """)
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-dest",
        metavar=Metavar.file,
        help="Spinal cord centerline (or segmentation) of a destination image (which could be "
             "straight or curved). An algorithm scales the length of the input centerline to match that of the "
             "destination centerline. If using `-ldisc-input` and `-ldisc-dest` with this parameter, "
             "instead of linear scaling, the source centerline will be non-linearly matched so "
             "that the inter-vertebral discs of the input image will match that of the "
             "destination image. This feature is particularly useful for registering to a "
             "template while accounting for disc alignment.")
    optional.add_argument(
        "-ldisc-input",
        metavar=Metavar.file,
        help="Labels located at the posterior edge of the intervertebral discs, for the input  "
        "image (`-i`). All disc covering the region of interest should be provided. Exmaple: if "
        "you are interested in levels C2 to C7, then you should provide disc labels 2,3,4,5,"
        "6,7). More details about label creation at "
        "https://spinalcordtoolbox.com/user_section/tutorials/vertebral-labeling.html. "
        "This option must be used with the `-ldisc-dest` parameter.")
    optional.add_argument(
        "-ldisc-dest",
        metavar=Metavar.file,
        help="Labels located at the posterior edge of the intervertebral discs, for the destination file (`-dest`). "
             "The same comments as in `-ldisc-input` apply. This option must be used with the `-ldisc-input` parameter.")
    optional.add_argument(
        "-disable-straight2curved",
        action='store_true',
        help="Disable straight to curved transformation computation, in case you do not need the "
             "output warping field straight-->curve (faster).")
    optional.add_argument(
        "-disable-curved2straight",
        action='store_true',
        help="Disable curved to straight transformation computation, in case you do not need the "
             "output warping field curve-->straight (faster).")
    optional.add_argument(
        "-speed-factor",
        metavar=Metavar.float,
        type=float,
        help='Acceleration factor for the calculation of the straightening warping field.'
             ' This speed factor enables an intermediate resampling to a lower resolution, which '
             'decreases the computational time at the cost of lower accuracy.'
             ' A speed factor of 2 means that the input image will be downsampled by a factor 2 '
             'before calculating the straightening warping field. For example, a 1x1x1 mm^3 image '
             'will be downsampled to 2x2x2 mm3, providing a speed factor of approximately 8.'
             ' Note that accelerating the straightening process reduces the precision of the '
             'algorithm, and induces undesirable edges effects. Default=1 (no downsampling).',
        default=1)
    optional.add_argument(
        "-xy-size",
        metavar=Metavar.float,
        type=float,
        help='Size of the output FOV in the RL/AP plane, in mm. The resolution of the destination '
             'image is the same as that of the source image (`-i`). Default: `35`.',
        default=35.0)
    optional.add_argument(
        "-o",
        metavar=Metavar.file,
        help='Straightened file. By default, the suffix "_straight" will be added to the input file name.',
        default='')
    optional.add_argument(
        "-ofolder",
        metavar=Metavar.folder,
        help="Output folder (all outputs will go there).",
        action=ActionCreateFolder,
        default='.')
    optional.add_argument(
        '-centerline-algo',
        help='Algorithm for centerline fitting. Default: nurbs.',
        choices=('bspline', 'linear', 'nurbs'),
        default='nurbs')
    optional.add_argument(
        '-centerline-smooth',
        metavar=Metavar.int,
        type=int,
        help='Degree of smoothing for centerline fitting. Only use with -centerline-algo {bspline, linear}. Default: `10`',
        default=10)

    optional.add_argument(
        "-param",
        metavar=Metavar.list,
        type=list_type(",", str),
        help=textwrap.dedent("""
            Parameters for spinal cord straightening. Separate arguments with ','.

              - `precision`: Float `[1, inf)` Precision factor of straightening, related to the number of slices. Increasing this parameter increases the precision along with increased computational time. Not taken into account with Hanning fitting method. Default=`2`
              - `threshold_distance`: Float `[0, inf)` Threshold at which voxels are not considered into displacement. Increase this threshold if the image is blackout around the spinal cord too much. Default=`10`
              - `accuracy_results`: `{0, 1}` Disable/Enable computation of accuracy results after straightening. Default=`0`
              - `template_orientation`: {0, 1}` Disable/Enable orientation of the straight image to be the same as the template. Default=`0`
              - `safe_zone: {0, 1}`: Disable/Enable enforcing the "safe zone". Turn this on to ensure that the warping fields will only be defined for areas covered by
              the spinal cord segmentation, which is useful when only a partial FOV is present (e.g. lumbar data). Default=`0`.
        """),  # noqa: E501 (line too long)
        )
    optional.add_argument(
        "-x",
        help="Final interpolation. Default: `spline`.",
        choices=("nn", "linear", "spline"),
        default="spline")
    optional.add_argument(
        '-qc',
        metavar=Metavar.str,
        help='The path where the quality control generated content will be saved')
    optional.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the dataset the '
             'process was run on')
    optional.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the subject the '
             'process was run on')
    optional.add_argument(
        '-qc-contrast',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the contrast the '
             'process was run on')

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_tempfile_args()

    return parser


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    """
    Main function
    :param argv:
    :return:
    """
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    input_filename = arguments.i
    centerline_file = arguments.s

    sc_straight = SpinalCordStraightener(input_filename, centerline_file)

    if arguments.dest is not None:
        sc_straight.use_straight_reference = True
        sc_straight.centerline_reference_filename = str(arguments.dest)

    if arguments.ldisc_input is not None:
        if not sc_straight.use_straight_reference:
            printv('Warning: discs position are not taken into account if reference is not provided.')
        else:
            sc_straight.discs_input_filename = str(arguments.ldisc_input)
            sc_straight.precision = 4.0
    if arguments.ldisc_dest is not None:
        if not sc_straight.use_straight_reference:
            printv('Warning: discs position are not taken into account if reference is not provided.')
        else:
            sc_straight.discs_ref_filename = str(arguments.ldisc_dest)
            sc_straight.precision = 4.0

    # Handling optional arguments
    sc_straight.remove_temp_files = arguments.r
    sc_straight.interpolation_warp = arguments.x
    sc_straight.output_filename = arguments.o
    sc_straight.path_output = arguments.ofolder
    path_qc = arguments.qc
    sc_straight.verbose = verbose

    # if arguments.cpu_nb is not None:
    #     sc_straight.cpu_number = arguments.cpu-nb)
    if arguments.disable_straight2curved:
        sc_straight.straight2curved = False
    if arguments.disable_curved2straight:
        sc_straight.curved2straight = False

    if arguments.speed_factor:
        sc_straight.speed_factor = arguments.speed_factor

    if arguments.xy_size:
        sc_straight.xy_size = arguments.xy_size

    sc_straight.param_centerline = ParamCenterline(
        algo_fitting=arguments.centerline_algo,
        smooth=arguments.centerline_smooth)
    if arguments.param is not None:
        params_user = arguments.param
        # update registration parameters
        for param in params_user:
            param_split = param.split('=')
            if param_split[0] == 'precision':
                sc_straight.precision = float(param_split[1])
            if param_split[0] == 'threshold_distance':
                sc_straight.threshold_distance = float(param_split[1])
            if param_split[0] == 'accuracy_results':
                sc_straight.accuracy_results = int(param_split[1])
            if param_split[0] == 'template_orientation':
                sc_straight.template_orientation = int(param_split[1])
            if param_split[0] == 'safe_zone':
                sc_straight.safe_zone = int(param_split[1])

    fname_straight = sc_straight.straighten()

    printv(f"\nFinished! Elapsed time: {sc_straight.elapsed_time} s", verbose)

    # Generate QC report
    if path_qc is not None:
        path_qc = os.path.abspath(path_qc)
        qc_dataset = arguments.qc_dataset
        qc_subject = arguments.qc_subject
        qc_contrast = arguments.qc_contrast
        generate_qc(fname_straight, args=argv, path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, contrast=qc_contrast,
                    process=os.path.basename(__file__).removesuffix(".py"))

    display_viewer_syntax([fname_straight], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
