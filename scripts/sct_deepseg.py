#!/usr/bin/env python
# -*- coding: utf-8
"""
This command-line tool is the interface for the deepseg API that performs segmentation using deep learning from the
ivadomed package.
"""

# TODO: Add link to example image so users can decide wether their images look "close enough" to some of the proposed
#  models (e.g., mice, etc.).
# TODO: add test to make sure that all postprocessing flags match the core.DEFAULT dictionary items
# TODO: Fetch default value (and display) depending on the model that is used.
# TODO: accommodate multiclass segmentation

import argparse
import os
import sys
import logging

from ivadomed import inference as imed_inference
import nibabel as nib

import spinalcordtoolbox as sct
from spinalcordtoolbox import image
import spinalcordtoolbox.deepseg as deepseg
import spinalcordtoolbox.deepseg.models

from spinalcordtoolbox.utils.shell import SmartFormatter, Metavar, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Segment an anatomical structure or pathologies according to the specified deep learning model.",
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    input_output = parser.add_argument_group("\nINPUT/OUTPUT")
    input_output.add_argument(
        "-i",
        nargs="+",
        help="Image to segment. Can be multiple images (separated with space).",
        metavar=Metavar.file)
    input_output.add_argument(
        "-c",
        nargs="+",
        help="Type of image contrast. Indicates the order in which the images have been presented with -i. "
             "Optional if only one image is specified with -i. The contrasts should be separated by spaces "
             "(e.g., -c t1 t2).",
        choices=('t1', 't2', 't2star'),
        metavar=Metavar.file)
    input_output.add_argument(
        "-o",
        help="Output file name. In case of multi-class segmentation, class-specific suffixes will be added. By default,"
             "suffix '_seg' will be added and output extension will be .nii.gz.",
        metavar=Metavar.str)

    seg = parser.add_argument_group('\nTASKS')
    seg.add_argument(
        "-task",
        help="Task to perform. It could either be a pre-installed task, task that could be installed, or a custom task."
             " To list available tasks, run: sct_deepseg -list-tasks",
        metavar=Metavar.str)
    seg.add_argument(
        "-list-tasks",
        action='store_true',
        help="Display a list of tasks that can be achieved.")
    seg.add_argument(
        "-install-task",
        help="Install models that are required for specified task.",
        choices=list(deepseg.models.TASKS.keys()))

    misc = parser.add_argument_group('\nPARAMETERS')
    misc.add_argument(
        "-thr",
        type=float,
        help="Binarize segmentation with specified threshold. Set to 0 for no thresholding (i.e., soft segmentation). "
             "Default value is model-specific and was set during optimization "
             "(more info at https://github.com/sct-pipeline/deepseg-threshold).",
        metavar=Metavar.float,
        default=0.9)
    misc.add_argument(
        "-r",
        type=int,
        help="Remove temporary files.",
        choices=(0, 1),
        default=1)
    misc.add_argument(
        "-largest",
        type=int,
        help="Keep the largest connected-objects from the output segmentation. Specify the number of objects to keep."
             "To keep all objects, set to 0",
        default=0)
    misc.add_argument(
        "-fill-holes",
        type=int,
        help="Fill small holes in the segmentation.",
        choices=(0, 1),
        default=0)
    misc.add_argument(
        "-remove-small",
        type=str,
        help="Minimal object size to keep with unit (mm3 or vox). Example: 1mm3, 5vox.",
        default='0vox')

    misc = parser.add_argument_group('\nMISC')
    misc.add_argument(
        "-v",
        type=int,
        help="Verbose: 0 = no verbosity, 1 = verbose.",
        choices=(0, 1),
        default=1)
    misc.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")

    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv if argv else ['--help'])

    # Deal with task
    if args.list_tasks:
        deepseg.models.display_list_tasks()

    if args.install_task is not None:
        for name_model in deepseg.models.TASKS[args.install_task]['models']:
            deepseg.models.install_model(name_model)
        exit(0)

    # Deal with input/output
    for file in args.i:
        if not os.path.isfile(file):
            parser.error("This file does not exist: {}".format(file))

    # Check if at least a model or task has been specified
    if args.task is None:
        parser.error("You need to specify a task.")

    # Check if all input images are provided
    required_contrasts = deepseg.models.get_required_contrasts(args.task)
    if len(args.i) != len(required_contrasts):
        parser.error("{} input files found. Please provide all required input files for the task {}, i.e. contrasts: {}."
                     .format(args.task, ', '.join(required_contrasts)))

    # Check modality order
    if len(args.i) > 1 and args.c is None:
        parser.error("You need to specify the order in which you put the contrasts in the input images with flag -c.")

    # Get pipeline model names
    name_models = deepseg.models.TASKS[args.task]['models']

    # Run pipeline by iterating through the models
    fname_prior = None
    output_filenames = None
    for name_model in name_models:
        # Check if this is an official model
        if name_model in list(deepseg.models.MODELS.keys()):
            # If it is, check if it is installed
            path_model = deepseg.models.folder(name_model)
            if not deepseg.models.is_valid(path_model):
                printv("Model {} is not installed. Installing it now...".format(name_model))
                deepseg.models.install_model(name_model)
        # If it is not, check if this is a path to a valid model
        else:
            path_model = os.path.abspath(name_model)
            if not deepseg.models.is_valid(path_model):
                parser.error("The input model is invalid: {}".format(path_model))

        # Order input images
        if args.c is not None:
            input_filenames = []
            for required_contrasts in deepseg.models.MODELS[name_model]['contrasts']:
                for provided_contrasts, input_filename in zip(args.c, args.i):
                    if required_contrasts == provided_contrasts:
                        input_filenames.append(input_filename)
        else:
            input_filenames = args.i

        # Call segment_nifti
        options = {**vars(args), "fname_prior": fname_prior}
        nii_lst, target_lst = imed_inference.segment_volume(path_model, input_filenames, options=options)

        # Delete intermediate outputs
        if fname_prior and os.path.isfile(fname_prior) and args.r:
            logger.info("Remove temporary files...")
            os.remove(fname_prior)

        output_filenames = []
        # Save output seg
        for nii_seg, target in zip(nii_lst, target_lst):
            if 'o' in options and options['o'] is not None:
                fname_seg = options['o'].replace(".nii.gz", target + ".nii.gz") if len(target_lst) > 1 else options['o']
            else:
                fname_seg = ''.join([sct.image.splitext(input_filenames[0])[0], target + '.nii.gz'])

            # If output folder does not exist, create it
            path_out = os.path.dirname(fname_seg)
            if not (path_out == '' or os.path.exists(path_out)):
                os.makedirs(path_out)

            nib.save(nii_seg, fname_seg)
            output_filenames.append(fname_seg)

        # Use the result of the current model as additional input of the next model
        fname_prior = fname_seg

    for output_filename in output_filenames:
        display_viewer_syntax([args.i[0], output_filename], colormaps=['gray', 'red'], opacities=['', '0.7'])


if __name__ == '__main__':
    init_sct()
    main(sys.argv[1:])
