#!/usr/bin/env python
#
# Utilities for labels
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

# TODO: for vert-disc: make it faster! currently the module display-voxel is very long
#       (esp. when ran on PAM50). We can find an alternative approach by sweeping through
#       centerline voxels.
# TODO: label_disc: for top vertebrae, make label at the center of the cord (currently it's at the tip)
# TODO: check if use specified several processes.
# TODO: currently it seems like cross_radius is given in pixel instead of mm

import os
import sys
from typing import Sequence
import textwrap

import numpy as np

import spinalcordtoolbox.labels as sct_labels
from spinalcordtoolbox.image import Image, zeros_like, add_suffix
from spinalcordtoolbox.types import Coordinate
from spinalcordtoolbox.reports import qc2
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.shell import (ActionCreateFolder, Metavar, SCTArgumentParser,
                                           display_viewer_syntax, list_type, parse_num_list)


def get_parser():
    parser = SCTArgumentParser(
        description="Utility functions for label images."
    )

    req_group = parser.add_argument_group("REQUIRED I/O")
    req_group.add_argument(
        '-i',
        metavar=Metavar.file,
        help="Input image (Required) Example: `t2_labels.nii.gz`"
    )

    io_group = parser.add_argument_group("OPTIONAL I/O")
    io_group.add_argument(
        '-o',
        metavar=Metavar.file,
        nargs='+',
        default=None,
        help="Output image (default: labels.nii.gz). Note: Only some label utilities create an output image."
    )
    io_group.add_argument(
        '-ilabel',
        metavar=Metavar.file,
        help="File that contain labels that you want to correct. It is possible to add new points with this option. "
             "Use with `-create-viewer`. Example: `t2_labels_auto.nii.gz`"
    )

    functions = parser.add_argument_group("LABEL FUNCTIONS")
    func_group = functions.add_mutually_exclusive_group(required=True)
    func_group.add_argument(
        '-add',
        metavar=Metavar.int,
        type=int,
        help="Add value to all labels. Value can be negative."
    )
    func_group.add_argument(
        '-create',
        metavar=Metavar.list,
        type=list_type(':', Coordinate),
        help="Create labels in a new image. List labels as: x1,y1,z1,value1:x2,y2,z2,value2. "
             "Example: 12,34,32,1:12,35,33,2"
    )
    func_group.add_argument(
        '-create-add',
        metavar=Metavar.list,
        type=list_type(':', Coordinate),
        help="Same as `-create`, but add labels to the input image instead of creating a new image. "
             "Example: 12,34,32,1:12,35,33,2"
    )
    func_group.add_argument(
        '-create-seg',
        metavar=Metavar.list,
        type=list_type(':', list_type(',', int)),
        help=textwrap.dedent("""
            Create labels on a cord segmentation (or centerline) image defined by `-i`. Each label should be specified using the form `v1,v2` where `v1` is value of the slice index along the inferior-superior axis, and `v2` is the value of the label. Separate each label with `:`.

            Example: `-create-seg 5,1:14,2:23,3` adds three labels at the axial slices 5, 14, and 23 (starting from the most inferior slice).
        """),  # noqa: E501 (line too long)
    )
    func_group.add_argument(
        '-create-seg-mid',
        metavar=Metavar.int,
        type=int,
        help=textwrap.dedent("""
            Similar to `-create-seg`. This option takes a single label value, and will automatically select the mid-point slice in the inferior-superior direction (so there is no need for a slice index).

            This is useful for when you have centered the field of view of your data at a specific location. For example, if you already know that the C2-C3 disc is centered in the I-S direction, then you can enter `-create-seg-mid 3` for that label. This saves you the trouble of having to manually specify a slice index using `-create-seg`.
        """),  # noqa: E501 (line too long)
    )
    func_group.add_argument(
        '-create-viewer',
        metavar=Metavar.list,
        help="Manually label from a GUI a list of labels IDs. Provide a comma-separated list "
             "containing individual values and/or intervals. Example: `-create-viewer 1:4,6,8` "
             "will allow you to add labels [1,2,3,4,6,8] using the GUI."
    )
    func_group.add_argument(
        '-cubic-to-point',
        action="store_true",
        help="Compute the center-of-mass for each label value."
    )
    func_group.add_argument(
        '-project-centerline',
        metavar=Metavar.file,
        help="Project labels (e.g. disc labels) onto a spinal cord segmentation or centerline. "
             "Example: sct_label_utils -i spinalcord.nii.gz -project-centerline labels.nii.gz "
    )
    func_group.add_argument(
        '-display',
        action="store_true",
        help="Display all labels (i.e. non-zero values)."
    )
    func_group.add_argument(
        '-increment',
        action="store_true",
        help="Takes all non-zero values, sort them along the inverse z direction, and attributes the values "
             "1, 2, 3, etc."
    )
    func_group.add_argument(
        '-vert-body',
        metavar=Metavar.list,
        type=list_type(',', int),
        help=textwrap.dedent("""
            From vertebral labeling, create points that are centered at the mid-vertebral levels. Separate desired levels with `,`. Example: `3,8`

            To get all levels, enter 0.
        """),
    )
    func_group.add_argument(
        '-vert-continuous',
        action="store_true",
        help="Convert discrete vertebral labeling to continuous vertebral labeling.",
    )
    func_group.add_argument(
        '-MSE',
        metavar=Metavar.file,
        help="Compute Mean Square Error between labels from input and reference image. Specify reference image here."
    )
    func_group.add_argument(
        '-remove-reference',
        metavar=Metavar.file,
        help="Remove labels from input image (`-i`) that are not in reference image (specified here)."
    )
    func_group.add_argument(
        '-remove-sym',
        metavar=Metavar.file,
        help="Remove labels from input image (`-i`) and reference image (specified here) that don't match. This "
             "produces two output images; use `-o` to provide two space-separated output filenames."
    )
    func_group.add_argument(
        '-remove',
        metavar=Metavar.list,
        type=list_type(',', int),
        help="Remove labels of specific value (specified here) from reference image."
    )
    func_group.add_argument(
        '-keep',
        metavar=Metavar.list,
        type=list_type(',', int),
        help="Keep labels of specific value (specified here) from reference image."
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-msg',
        metavar=Metavar.str,
        help="Display a message to explain the labeling task. Use with -create-viewer"
    )
    optional.add_argument(
        '-qc',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="The path where the quality control generated content will be saved."
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

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    for i, arg in enumerate(argv):
        if arg == '-create-seg' and len(argv) > i+1 and '-1,' in argv[i+1]:
            raise DeprecationWarning("The use of '-1' for '-create-seg' has been deprecated. Please use "
                                     "'-create-seg-mid' instead.")
        if arg == '-disc':
            raise DeprecationWarning("The use of '-disc' has been deprecated. Please use `sct_label_vertebrae -discfile`.")

    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    input_filename = arguments.i

    # Validate the output filename(s) early, before any processing occurs.
    # `-remove-sym` produces two outputs (filtered `-i` and filtered reference); every other
    # function produces a single output.
    n_expected = 2 if arguments.remove_sym is not None else 1
    if arguments.o is None:
        if arguments.remove_sym is not None:
            arguments.o = [add_suffix(arguments.i, '_sym'), add_suffix(arguments.remove_sym, '_sym')]
        else:
            arguments.o = ['labels.nii.gz']
    elif len(arguments.o) != n_expected:
        parser.error(f"'-o' expects {n_expected} filename(s) but got {len(arguments.o)}")
    output_fnames = arguments.o

    img = Image(input_filename)
    dtype = None

    if arguments.add is not None:
        value = arguments.add
        out = sct_labels.add(img, value)
    elif arguments.create is not None:
        labels = arguments.create
        out = sct_labels.create_labels_empty(img, labels)
    elif arguments.create_add is not None:
        labels = arguments.create_add
        out = sct_labels.create_labels(img, labels)
    elif arguments.create_seg is not None:
        labels = arguments.create_seg
        out = sct_labels.create_labels_along_segmentation(img, labels)
    elif arguments.create_seg_mid is not None:
        labels = [(-1, arguments.create_seg_mid)]
        out = sct_labels.create_labels_along_segmentation(img, labels)
    elif arguments.cubic_to_point:
        out = sct_labels.cubic_to_point(img)
    elif arguments.display:
        display_voxel(img, verbose)
        return
    elif arguments.increment:
        out = sct_labels.increment_z_inverse(img)
    elif arguments.project_centerline is not None:
        ref = Image(arguments.project_centerline)
        try:
            out = sct_labels.project_centerline(img, ref)
        except sct_labels.ShapeMismatchError as e:
            printv(
                f"Input image: {input_filename} and referenced labels: {arguments.project_centerline} "
                f"should have the same shape according to the RPI orientation.\n"
                f"Referenced labels with shape {e.dims['ref']} cannot be projected on "
                f"input image with shape {e.dims['img']}",
                type='error')
    elif arguments.vert_body is not None:
        levels = arguments.vert_body
        if len(levels) == 1 and levels[0] == 0:
            levels = None  # all levels
        out = sct_labels.label_vertebrae(img, levels)
    elif arguments.vert_continuous:
        out = sct_labels.continuous_vertebral_levels(img)
        dtype = 'float32'
    elif arguments.MSE is not None:
        ref = Image(arguments.MSE)
        mse = sct_labels.compute_mean_squared_error(img, ref)
        printv(f"Computed MSE: {mse}")
        return
    elif arguments.remove_reference is not None:
        ref = Image(arguments.remove_reference)
        out = sct_labels.remove_missing_labels(img, ref)
    elif arguments.remove_sym is not None:
        # first pass use img as source
        ref = Image(arguments.remove_sym)
        out = sct_labels.remove_missing_labels(img, ref)

        # second pass use previous pass result as reference
        ref_out = sct_labels.remove_missing_labels(ref, out)
    elif arguments.remove is not None:
        labels = arguments.remove
        out = sct_labels.remove_labels_from_image(img, labels)
    elif arguments.keep is not None:
        labels = arguments.keep
        out = sct_labels.remove_other_labels_from_image(img, labels)
    elif arguments.create_viewer is not None:
        msg = "" if arguments.msg is None else f"{arguments.msg}\n"
        if arguments.ilabel is not None:
            input_labels_img = Image(arguments.ilabel)
            out = launch_manual_label_gui(img, input_labels_img, parse_num_list(arguments.create_viewer), msg)
        else:
            out = launch_sagittal_viewer(img, parse_num_list(arguments.create_viewer), msg)

    printv("Generating output files...")
    # `-remove-sym` filters both `-i` and the reference image, so it produces two
    # (input, result) pairs. Every other function produces a single pair.
    if arguments.remove_sym is not None:
        inputs, results = [input_filename, arguments.remove_sym], [out, ref_out]
    else:
        inputs, results = [input_filename], [out]

    for fname_input, fname_output, result in zip(inputs, output_fnames, results):
        result.save(path=fname_output, dtype=dtype)
        display_viewer_syntax([fname_input, fname_output], verbose=verbose)
        if arguments.qc is not None:
            qc2.sct_label_utils(
                fname_input=fname_input,
                fname_seg=fname_output,
                command='sct_label_utils',
                argv=argv,
                path_qc=os.path.abspath(arguments.qc),
                dataset=arguments.qc_dataset,
                subject=arguments.qc_subject,
                contrast=arguments.qc_contrast,
            )


def display_voxel(img: Image, verbose: int = 1) -> Sequence[Coordinate]:
    """
    Display all the labels that are contained in the input image.
    :param img: source image
    :param verbose: verbosity level
    """

    coordinates_input = img.getNonZeroCoordinates(sorting='value')
    useful_notation = ':'.join(str(coord) for coord in coordinates_input)

    for coord in coordinates_input:
        printv(f'Position=({coord.x},{coord.y},{coord.z}) -- Value= {coord.value}', verbose=verbose)

    printv('All labels (useful syntax):', verbose=verbose)
    printv(useful_notation, verbose=verbose)


def launch_sagittal_viewer(img: Image, labels: Sequence[int], msg: str, previous_points: Sequence[Coordinate] = None, output_img: Image = None) -> Image:
    from spinalcordtoolbox.gui import base
    from spinalcordtoolbox.gui.sagittal import launch_sagittal_dialog
    params = base.AnatomicalParams()
    params.vertebraes = labels
    params.input_file_name = img.absolutepath

    if output_img is not None:
        params.output_file_name = output_img.absolutepath
    else:
        params.output_file_name = img.absolutepath

    params.subtitle = msg

    if previous_points is not None:
        params.message_warn = 'Please select the label you want to add \nor correct in the list below before clicking \non the image'

    out = zeros_like(img, dtype='uint8')
    out.absolutepath = params.output_file_name
    launch_sagittal_dialog(img, out, params, previous_points)

    return out


def launch_manual_label_gui(img: Image, input_labels_img: Image, labels: Sequence[int], msg):
    # the input image is reoriented to 'SAL' when open by the GUI
    input_labels_img.change_orientation('SAL')
    mid = int(np.round(input_labels_img.data.shape[2] / 2))
    previous_points = input_labels_img.getNonZeroCoordinates()

    # boolean used to mark first element to initiate the list.
    first = True

    previous_label = None

    for i in range(len(previous_points)):
        if int(previous_points[i].value) in labels:
            pass
        else:
            labels.append(int(previous_points[i].value))
        if first:
            points = np.array([previous_points[i]. x, previous_points[i].y, previous_points[i].z, previous_points[i].value])
            points = np.reshape(points, (1, 4))
            previous_label = points
            first = False
        else:
            points = np.array([previous_points[i].x, previous_points[i].y, previous_points[i].z, previous_points[i].value])
            points = np.reshape(points, (1, 4))
            previous_label = np.append(previous_label, points, axis=0)
        labels.sort()

    # check if variable was created which means the file was not empty and contains some points asked in labels
    if previous_label is not None:
        # project onto mid sagittal plane
        for i in range(len(previous_label)):
            previous_label[i][2] = mid

    out = launch_sagittal_viewer(img, labels, msg, previous_points=previous_label)

    return out


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
