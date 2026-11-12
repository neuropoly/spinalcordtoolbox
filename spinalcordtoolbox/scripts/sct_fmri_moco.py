#!/usr/bin/env python
#
# Motion correction of fMRI data.
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
import os
from typing import Sequence
import textwrap

from spinalcordtoolbox.moco.moco import ParamMoco, moco_wrapper
from spinalcordtoolbox.utils.sys import init_sct, set_loglevel
from spinalcordtoolbox.utils.shell import (SCTArgumentParser, Metavar, ActionCreateFolder, display_viewer_syntax,
                                           list_type, positive_int_type)
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.moco.dl.inference import moco_dl, check_dl_args


def get_parser():
    # initialize parameters
    # TODO: create a class ParamFmriMoco which inheritates from ParamMoco
    param_default = ParamMoco(group_size=1, metric='MeanSquares', smooth='0')

    # parser initialisation
    parser = SCTArgumentParser(
        description=textwrap.dedent("""
            Motion correction of fMRI data. Some robust features include:

              - group-wise (`-g`)
              - slice-wise regularized along z using polynomial function (`-param poly`). For more info about the method, type: `isct_antsSliceRegularizedRegistration`
              - masking (`-m`)
              - iterative averaging of target volume
              - Optional DL-based motion correction (`-dl`)

            The outputs of the motion correction process are:

              - the motion-corrected fMRI volumes
              - the time average of the corrected fMRI volumes
              - a time-series with 1 voxel in the XY plane, for the X and Y motion direction (two separate files), as required for FSL analysis.
              - a TSV file with one row for each time point, with the slice-wise average of the motion correction magnitude for that time point, that can be used for Quality Control.
        """),  # noqa: E501 (line too long)
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        help="Input data (4D). Example: `fmri.nii.gz`"
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-g',
        metavar=Metavar.int,
        type=positive_int_type,
        default=param_default.group_size,
        help='Group nvols successive fMRI volumes for more robustness. Values `2` or greater will create groups of '
             'that size, while a value of `1` will turn off grouping (i.e. per-volume motion correction).'
    )
    optional.add_argument(
        '-m',
        metavar=Metavar.file,
        default=param_default.fname_mask,
        help="Binary mask to limit voxels considered by the registration metric. You may also provide a softmask "
             "(nonbinary, [0, 1]), and it will be binarized at 0.5."
    )
    optional.add_argument(
        '-ref',
        metavar=Metavar.file,
        default=param_default.fname_ref,
        help="Reference volume for motion correction, for example the mean fMRI volume."
    )
    optional.add_argument(
        '-param',
        metavar=Metavar.list,
        type=list_type(',', str),
        help=f"Advanced parameters. Assign value with `=`; Separate arguments with `,`.\n"
             f"  - `poly` [int]: Degree of polynomial function used for regularization along Z. For no regularization "
             f"set to 0. Default={param_default.poly}.\n"
             f"  - `smooth` [mm]: Smoothing kernel. Default={param_default.smooth}.\n"
             f"  - `metric` {{MI, MeanSquares, CC}}: Metric used for registration. Default={param_default.metric}.\n"
             f"  - `iter` [int]: Number of iterations. Default={param_default.iter}.\n"
             f"  - `gradStep` [float]: Searching step used by registration algorithm. The higher the more deformation "
             f"allowed. Default={param_default.gradStep}.\n"
             f"  - `sampling` [None or 0-1]: Sampling rate used for registration metric. "
             f"Default={param_default.sampling}.\n"
             f"  - `num_target` [int]: Target volume or group (starting with 0). Not used if `-ref` is provided. Default={param_default.num_target}.\n"
             f"  - `iterAvg` [int]: Iterative averaging: Target volume is a weighted average of the "
             f"previously-registered volumes. Default={param_default.iterAvg}.\n"
    )
    optional.add_argument(
        '-ofolder',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        default='.',
        help="Output path."
    )
    optional.add_argument(
        '-x',
        choices=['nn', 'linear', 'spline'],
        default='linear',
        help="Final interpolation."
    )
    optional.add_argument(
        '-qc',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="The path where the quality control generated content will be saved. (Note: "
             "Both `-qc` and `-qc-seg` are required in order to generate a QC report.)"
    )
    optional.add_argument(
        '-qc-seg',
        metavar=Metavar.file,
        help="Segmentation of spinal cord to improve cropping in qc report. (Note: "
             "Both `-qc` and `-qc-seg` are required in order to generate a QC report.)"
    )
    optional.add_argument(
        '-qc-fps',
        metavar=Metavar.float,
        type=float,
        default=5,
        help="This float number is the number of frames per second for the output gif images."
    )
    optional.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the dataset the process was run on."
    )
    optional.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the subject the process was run on."
    )
    optional.add_argument(
        '-qc-contrast',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the contrast the process was run on."
    )
    optional.add_argument(
        '-dl',
        action='store_true',
        help="Use deep learning–based motion correction (DenseNet).\n"
             "  - Requires the `-m` argument (binary spinal cord mask). "
             "The binary mask (3D) defines the spinal cord region used by the model to estimate motion within the mask. "
             "It should be large enough to cover the full extent of the spinal cord in the image.\n"
             "DISCLAIMER: This preliminary method has not been thoroughly validated therefore we cannot guarantee it will work well on your data. "
             "We suggest you compare the performance with/without the -dl method, and pick the best for your data."
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

    # initialization
    param = ParamMoco(group_size=1, metric='MeanSquares', smooth='0')

    # Fetch user arguments
    param.fname_data = arguments.i
    param.path_out = arguments.ofolder
    param.remove_temp_files = arguments.r
    param.interp = arguments.x
    if arguments.g is not None:
        param.group_size = arguments.g
    param.fname_mask = arguments.m
    param.fname_ref = arguments.ref
    if arguments.param is not None:
        param.update(arguments.param)
    param.verbose = verbose

    path_qc = arguments.qc
    qc_fps = arguments.qc_fps
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject
    qc_contrast = arguments.qc_contrast
    qc_seg = arguments.qc_seg

    mutually_inclusive_args = (path_qc, qc_seg)
    is_qc_none, is_seg_none = [arg is None for arg in mutually_inclusive_args]
    if not (is_qc_none == is_seg_none):
        parser.error("Both '-qc' and '-qc-seg' are required in order to generate a QC report.")

    # Run moco
    if arguments.dl:
        try:
            check_dl_args(argv)
        except ValueError as e:
            raise parser.error(str(e))

        fname_output_image = moco_dl(
            fname_data=param.fname_data,
            fname_mask=param.fname_mask,
            path_out=param.path_out,
            fname_ref=param.fname_ref,
            mode="fmri"
        )
    else:
        # Run SCT-based motion correction
        fname_output_image = moco_wrapper(param)

    set_loglevel(verbose, caller_module_name=__name__)  # moco_wrapper changes verbose to 0, see issue #3341

    # QC report
    if path_qc is not None:
        generate_qc(fname_in1=fname_output_image, fname_in2=param.fname_data, fname_seg=qc_seg,
                    args=argv, path_qc=os.path.abspath(path_qc), fps=qc_fps, dataset=qc_dataset,
                    subject=qc_subject, contrast=qc_contrast, process='sct_fmri_moco')

    display_viewer_syntax([fname_output_image, param.fname_data], mode='ortho,ortho', verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
