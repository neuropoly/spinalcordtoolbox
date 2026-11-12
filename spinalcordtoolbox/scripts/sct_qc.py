#!/usr/bin/env python
#
# Generate QC report
#
# Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import json
import os
import re
import sys
from typing import Sequence
import textwrap

from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.reports import qc2
from spinalcordtoolbox.utils.sys import init_sct, list2cmdline, __sct_dir__, set_loglevel
from spinalcordtoolbox.utils.shell import SCTArgumentParser, printv

from spinalcordtoolbox.deepseg.models import TASKS


DEFAULT_LABELS = os.path.join(__sct_dir__, 'spinalcordtoolbox', 'reports', 'sct_label_vertebrae_regions.json')


def get_parser():
    parser = SCTArgumentParser(
        description='Generate Quality Control (QC) report following SCT processing.',
        epilog=textwrap.dedent("""
            Examples:

            - `sct_qc -i t2.nii.gz -s t2_seg.nii.gz -p sct_deepseg_sc`
            - `sct_qc -i t2.nii.gz -s t2_pmj.nii.gz -p sct_detect_pmj`
            - `sct_qc -i t2.nii.gz -s t2_seg_labeled.nii.gz -p sct_label_vertebrae`
            - `sct_qc -i t2.nii.gz -s t2_seg.nii.gz -p sct_deepseg_sc -qc-dataset mydata -qc-subject sub-45`
            - `sct_qc -i t2.nii.gz -s t2_seg.nii.gz -d t2_lesion.nii.gz -p sct_deepseg_lesion -plane axial`
        """),
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        metavar='IMAGE',
        help='Input image #1')
    processes = ('sct_deepseg', 'sct_deepseg_gm', 'sct_deepseg_lesion', 'sct_deepseg_sc', 'sct_detect_pmj',
                 'sct_dmri_moco', 'sct_fmri_compute_tsnr', 'sct_fmri_moco', 'sct_get_centerline',
                 'sct_image_stitch', 'sct_label_utils', 'sct_label_vertebrae', 'sct_propseg',
                 'sct_register_multimodal', 'sct_register_to_template', 'sct_warp_template')
    mandatory.add_argument(
        '-p',
        metavar='PROCESS',
        help="SCT function associated with the QC report to generate.\n"
             f"(choices: {', '.join(processes)})",
        choices=processes)

    optional = parser.optional_arggroup
    optional.add_argument(
        '-s',
        metavar='SEG',
        help='Input segmentation or label')
    optional.add_argument(
        '-s2',
        metavar='SEG2',
        help='Second input segmentation or label. Only relevant for `-p sct_deepseg`. Necessary for models that output '
             'two images, for example:\n'
             ' - 1. SC    + 2. lesion\n'
             ' - 1. SC    + 2. GM\n'
             ' - 1. discs + 2. full spine segmentation'
    )
    optional.add_argument(
        '-d',
        metavar='DEST',
        help='Input image #2 to overlay on image #1 (requires a segmentation), or output of another '
             'process. Example: `sct_straighten_spinalcord`')
    optional.add_argument(
        '-plane',
        help='Plane of the output QC. Only relevant for `-p sct_deepseg_lesion`.',
        choices=('axial', 'sagittal'))
    optional.add_argument(
        '-resample',
        help='Millimeter resolution to resample the image to. Set to 0 to turn off resampling. You can '
             'use this option to control the zoom of the QC report: higher values will result in '
             'smaller images, and lower values will result in larger images.',
        type=float)
    optional.add_argument(
        '-text-labels',
        help="If set to 0, text won't be drawn on top of labels. Only relevant for `-p sct_label_vertebrae`.",
        choices=(0, 1),
        default=1,
        type=int)
    optional.add_argument(
        '-custom-labels',
        metavar="JSON",
        help=f"Path to a JSON file containing custom region labels. Only relevant for `-p sct_label_vertebrae` or "
             f"`-p sct_deepseg -deepseg-task spine`.\n"
             f"   * For `-p spine`, the relevant label file will be inferred based on the model that was used to generate "
             f"the inputs. If the model type cannot be inferred, the step1 labels will be used by default.\n"
             f"   * For `-p sct_label_vertebrae`, the default labels will be used:\n"
             f"     {DEFAULT_LABELS})",
        )
    optional.add_argument(
        '-qc',
        metavar='QC',
        help='Path to save QC report.',
        default=os.path.join('.', 'qc'))
    optional.add_argument(
        '-qc-dataset',
        metavar='DATASET',
        help='If provided, this string will be mentioned in the QC report as the dataset the process '
             'was run on')
    optional.add_argument(
        '-qc-subject',
        metavar='SUBJECT',
        help='If provided, this string will be mentioned in the QC report as the subject the process '
             'was run on')
    optional.add_argument(
        '-qc-contrast',
        metavar='CONTRAST',
        help='If provided, this string will be mentioned in the QC report as the contrast the process '
             'was run on. If not provided, this will fall back to the name of the parent directory of '
             '`-i` (which may not be meaningful for all directory layouts).')
    optional.add_argument(
        "-qc-seg",
        metavar="IMAGE",
        help="Segmentation file to use for cropping the QC. This option is useful when you want to QC a region "
             "that is different from the output segmentation. For example, it might be useful to provide a "
             "dilated cord segmentation to expand the QC field of view. Only for use with `-p sct_deepseg`."
    )
    optional.add_argument(
        '-fps',
        metavar='float',
        type=float,
        default=5,
        help='The number of frames per second for output gif images. Only useful for sct_fmri_moco and '
             'sct_dmri_moco.')
    optional.add_argument(
        '-deepseg-task',
        help="The task used to generate the sct_deepseg output files.\n"
             "Only relevant for `-p sct_deepseg`.\n"
             "This option dictates which specific QC report will be generated (e.g. single-class segmentation, multi-class segmentation, "
             "whole-spine, etc.).\n"
             f"(choices: {', '.join(sorted(TASKS))})",
        metavar="TASK",
        choices=sorted(TASKS),
        default='spinalcord'
    )

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


def parse_label_type(json_file, verbose):
    """Parse `sct_deepseg spine` label type from its JSON sidecar file."""
    # Exit early if JSON sidecar doesn't exist
    label_type = "step1"  # default label type
    if not os.path.exists(json_file):
        printv(f"Could not find JSON sidecar file: {json_file}. Defaulting to label type: {label_type}", verbose=verbose)
        return label_type

    # Load the JSON sidecar file
    printv(f"Attempting to parse label type from JSON sidecar file: {json_file}", verbose=verbose)
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # 'GeneratedBy' is a list of entries (BIDS), so check every entry's 'Name' for '-label-vert' usage
    names = []
    if (
        isinstance(json_data, dict) and
        isinstance(generated_by := json_data.get('GeneratedBy'), list)
    ):
        names = [
            name
            for entry in generated_by
            if isinstance(entry, dict)
            and isinstance(name := entry.get('Name'), str)
            and "-label-vert" in name
        ]

    # Try to find evidence of steps type via the argument usage ("-label-vert 0" (step1) or "-label-vert 1" (step2))
    steps = {
        "step2" if match.group(1) == "1" else "step1"
        for name in names
        for match in re.finditer(r"-label-vert\s+([01])\b", name)
    }
    if len(steps) == 1:
        label_type = steps.pop()
        printv(f"Inferred label type: {label_type}", verbose=verbose)
    elif len(steps) > 1:
        printv(f"Conflicting '-label-vert' values in 'GeneratedBy' entries of JSON sidecar file: {json_file}. "
               f"Defaulting to label type: {label_type}", verbose=verbose, type='warning')
    else:
        printv(f"Could not find 'GeneratedBy' -> 'Name' -> '-label-vert' usage in JSON sidecar file: {json_file}. "
               f"Defaulting to label type: {label_type}", verbose=verbose, type='warning')
    return label_type


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    if arguments.p == 'sct_deepseg_lesion' and arguments.plane is None:
        parser.error('Please provide the plane of the output QC with `-plane`')

    # Load the label file depending on the provided arguments:
    #   - Whether the user provided a custom label file via `-custom-labels` (takes precedence)
    #   - The label type inferred from the `spine` JSON sidecar file (if present)
    if arguments.custom_labels is not None:
        label_file = arguments.custom_labels
    elif arguments.p == 'sct_deepseg' and arguments.deepseg_task == 'spine':
        json_file = arguments.s2.replace('.nii.gz', '.json')
        label_type = parse_label_type(json_file, verbose=verbose)  # falls back to default 'spine' label type (step1)
        label_file = os.path.join(__sct_dir__, 'spinalcordtoolbox', 'reports', f'totalspineseg_{label_type}_regions.json')
    else:  # sct_label_vertebrae
        label_file = DEFAULT_LABELS

    # Common arguments for QC reports
    # NB: For the qc2 reports, we override the command name to be `sct_qc` so that the QC report will always show
    # `sct_qc` as the command that was run, even if the user ran a different SCT command (e.g. `sct_deepseg`) and then
    # called `sct_qc` to generate a QC report for that command. (Alternatively, we could fake the command name to be
    # the original command (e.g. `sct_deepseg`) but that would require us to reverse engineer the corresponding
    # command from the provided arguments.)
    kwargs = dict(
        fname_input=arguments.i,
        fname_output=arguments.d,
        fname_seg=arguments.s,
        command="sct_qc",
        argv=argv,
        path_qc=arguments.qc,
        dataset=arguments.qc_dataset,
        subject=arguments.qc_subject,
        contrast=arguments.qc_contrast,
    )
    if arguments.resample is None:
        assert 'p_resample' not in kwargs  # Use the report's default
    elif arguments.resample == 0:
        kwargs['p_resample'] = None  # Explicitly turn it off
    else:
        kwargs['p_resample'] = arguments.resample

    if arguments.p in ['sct_register_multimodal', 'sct_register_to_template']:
        qc2.sct_register(**kwargs)
    elif arguments.p == 'sct_fmri_compute_tsnr':
        qc2.sct_fmri_compute_tsnr(**kwargs)
    elif arguments.p == 'sct_label_vertebrae':
        del kwargs['fname_output']  # not used by this report
        qc2.sct_label_vertebrae(
            draw_text=bool(arguments.text_labels),
            path_custom_labels=label_file,
            **kwargs
        )
    elif arguments.p == 'sct_label_utils':
        del kwargs['fname_output']  # not used by this report
        qc2.sct_label_utils(**kwargs)
    elif arguments.p == 'sct_deepseg':
        del kwargs['fname_output']  # not used by this report
        if arguments.deepseg_task == 'spine':
            # For the spine task, we need two outputs (fname_discs and fname_all) to generate 2 different QC reports.
            # Because these are passed via two separate arguments (-s and -s2), we can't just rely on the "common"
            # argument "fname_seg=arguments.s", so we delete it and specify the segmentation files explicitly.
            del kwargs['fname_seg']
            qc2.sct_label_utils(
                fname_seg=arguments.s,
                **kwargs
            )
            qc2.sct_label_vertebrae(
                fname_seg=arguments.s2,
                offset_text=False,
                path_custom_labels=label_file,
                **kwargs
            )

        else:
            # For all other deepseg tasks, we just have to trust that the user provided the correct segmentation file(s) for the task.
            # - The QC report will be generated based on the task name (present in argv) + the provided files.
            # - The responsibility for input validation lies with the QC report for the specific task
            qc2.sct_deepseg(
                fname_seg2=arguments.s2,
                species='mouse' if 'mouse' in arguments.deepseg_task else 'human',
                # Internal functions use capitalized strings ('Axial'/'Sagittal') + sct_deepseg assumes 'Axial' by default
                plane=arguments.plane.capitalize() if isinstance(arguments.plane, str) else 'Axial',
                fname_qc_seg=arguments.qc_seg,
                **kwargs
            )

    else:
        generate_qc(fname_in1=arguments.i,
                    fname_in2=arguments.d,
                    fname_seg=arguments.s,
                    # Internal functions use capitalized strings ('Axial'/'Sagittal')
                    plane=arguments.plane.capitalize() if isinstance(arguments.plane, str) else arguments.plane,
                    args=f'("sct_qc {list2cmdline(argv)}")',
                    path_qc=arguments.qc,
                    dataset=arguments.qc_dataset,
                    subject=arguments.qc_subject,
                    contrast=arguments.qc_contrast,
                    process=arguments.p,
                    fps=arguments.fps,
                    p_resample=arguments.resample,
                    )


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
