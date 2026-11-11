#!/usr/bin/env python
#
# Parser for PropSeg binary.
#
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

# TODO: remove temp files in case rescaled is not "1"

import os
import pathlib
import sys
import logging
from typing import Sequence
import textwrap

import numpy as np
from scipy.ndimage import label, center_of_mass

from spinalcordtoolbox.image import Image, add_suffix, zeros_like, convert
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, run_proc, printv, set_loglevel, LazyLoader
from spinalcordtoolbox.utils.fs import tmp_create, rmtree, extract_fname, mv, copy
from spinalcordtoolbox.centerline import optic
from spinalcordtoolbox.reports.qc import generate_qc

from spinalcordtoolbox.scripts import sct_image

nib = LazyLoader("nib", globals(), "nibabel")

logger = logging.getLogger(__name__)


def check_and_correct_segmentation(fname_segmentation, fname_centerline, folder_output='', threshold_distance=5.0,
                                   remove_temp_files=1, verbose=0):
    """
    This function takes the outputs of isct_propseg (centerline and segmentation) and check if the centerline of the
    segmentation is coherent with the centerline provided by the isct_propseg, especially on the edges (related
    to issue #1074).
    Args:
        fname_segmentation: filename of binary segmentation
        fname_centerline: filename of binary centerline
        threshold_distance: threshold, in mm, beyond which centerlines are not coherent
        verbose:

    Returns: None
    """
    printv('\nCheck consistency of segmentation...', verbose)
    # creating a temporary folder in which all temporary files will be placed and deleted afterwards
    path_tmp = tmp_create(basename="propseg-check-segmentation")
    im_seg = convert(Image(fname_segmentation))
    im_seg.save(os.path.join(path_tmp, "tmp.segmentation.nii.gz"), mutable=True, verbose=0)
    im_centerline = convert(Image(fname_centerline))
    im_centerline.save(os.path.join(path_tmp, "tmp.centerline.nii.gz"), mutable=True, verbose=0)

    # go to tmp folder (and store original info to use when converting back at the end)
    fname_seg_absolute = os.path.abspath(fname_segmentation)
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # convert input to RPI
    image_input_orientation = im_seg.orientation
    sct_image.main("-i tmp.segmentation.nii.gz -setorient RPI -o tmp.segmentation_RPI.nii.gz -v 0".split())
    sct_image.main("-i tmp.centerline.nii.gz -setorient RPI -o tmp.centerline_RPI.nii.gz -v 0".split())

    # go through segmentation image, and compare with centerline from propseg
    im_seg = Image('tmp.segmentation_RPI.nii.gz')
    im_centerline = Image('tmp.centerline_RPI.nii.gz')

    # Get size of data
    printv('\nGet data dimensions...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim

    # extraction of centerline provided by isct_propseg and computation of center of mass for each slice
    # the centerline is defined as the center of the tubular mesh outputed by propseg.
    centerline, key_centerline = {}, []
    for i in range(nz):
        slice = im_centerline.data[:, :, i]
        if np.any(slice):
            x_centerline, y_centerline = center_of_mass(slice)
            centerline[str(i)] = [x_centerline, y_centerline]
            key_centerline.append(i)

    minz_centerline = np.min(key_centerline)
    maxz_centerline = np.max(key_centerline)
    mid_slice = int((maxz_centerline - minz_centerline) / 2)

    # for each slice of the segmentation, check if only one object is present. If not, remove the slice from segmentation.
    # If only one object (the spinal cord) is present in the slice, check if its center of mass is close to the centerline of isct_propseg.
    slices_to_remove = [False] * nz  # flag that decides if the slice must be removed
    for i in range(minz_centerline, maxz_centerline + 1):
        # extraction of slice
        slice = im_seg.data[:, :, i]
        distance = -1
        label_objects, nb_labels = label(slice)  # count binary objects in the slice
        if nb_labels > 1:  # if there is more that one object in the slice, the slice is removed from the segmentation
            slices_to_remove[i] = True
        elif nb_labels == 1:  # check if the centerline is coherent with the one from isct_propseg
            x_centerline, y_centerline = center_of_mass(slice)
            slice_nearest_coord = min(key_centerline, key=lambda x: abs(x - i))
            coord_nearest_coord = centerline[str(slice_nearest_coord)]
            distance = np.sqrt(((x_centerline - coord_nearest_coord[0]) * px) ** 2 +
                               ((y_centerline - coord_nearest_coord[1]) * py) ** 2 +
                               ((i - slice_nearest_coord) * pz) ** 2)

            if distance >= threshold_distance:  # threshold must be adjusted, default is 5 mm
                slices_to_remove[i] = True

    # Check list of removal and keep one continuous centerline (improve this comment)
    # Method:
    # starting from mid-centerline (in both directions), the first True encountered is applied to all following slices
    slice_to_change = False
    for i in range(mid_slice, nz):
        if slice_to_change:
            slices_to_remove[i] = True
        elif slices_to_remove[i]:
            slice_to_change = True

    slice_to_change = False
    for i in range(mid_slice, 0, -1):
        if slice_to_change:
            slices_to_remove[i] = True
        elif slices_to_remove[i]:
            slice_to_change = True

    for i in range(0, nz):
        # remove the slice
        if slices_to_remove[i]:
            im_seg.data[:, :, i] *= 0

    # saving the image
    im_seg.save('tmp.segmentation_RPI_c.nii.gz')

    # replacing old segmentation with the corrected one
    sct_image.main('-i tmp.segmentation_RPI_c.nii.gz -setorient {} -o {} -v 0'.
                   format(image_input_orientation, fname_seg_absolute).split())

    os.chdir(curdir)

    # display information about how much of the segmentation has been corrected

    # remove temporary files
    if remove_temp_files:
        # printv("\nRemove temporary files...", verbose)
        rmtree(path_tmp)


def get_parser():
    # Initialize the parser
    parser = SCTArgumentParser(
        description=(
            "This program segments automatically the spinal cord on T1- and T2-weighted images, for any field of view. "
            "You must provide the type of contrast, the image as well as the output folder path. The segmentation "
            "follows the spinal cord centerline, which is provided by an automatic tool: Optic. The initialization of "
            "the segmentation is made on the median slice of the centerline, and can be adjusted using the `-init` "
            "parameter. The initial radius of the tubular mesh that will be propagated should be adapted to size of "
            "the spinal cord on the initial propagation slice. \n"
            "\n"
            "Primary output is the binary mask of the spinal cord segmentation. This method must provide VTK "
            "triangular mesh of the segmentation (option `-mesh`). Spinal cord centerline is available as a binary image "
            "(`-centerline-binary`) or a text file with coordinates in world referential (`-centerline-coord`).\n"
            "\n"
            "Cross-sectional areas along the spinal cord can be available (`-cross`). Several tips on segmentation "
            "correction can be found on the 'Correcting sct_propseg' page in the Tutorials section of the "
            "documentation.\n"
            "\n"
            "If the segmentation fails at some location (e.g. due to poor contrast between spinal cord and CSF), edit "
            "your anatomical image (e.g. with fslview) and manually enhance the contrast by adding bright values "
            "around the spinal cord for T2-weighted images (dark values for T1-weighted). Then, launch the "
            "segmentation again.\n"
            "\n"
            "References:\n"
            "  - [De Leener B, Kadoury S, Cohen-Adad J. Robust, accurate and fast automatic segmentation of the spinal "
            "cord. Neuroimage 98, 2014. pp 528-536. DOI: 10.1016/j.neuroimage.2014.04.051](https://doi.org/10.1016/j.neuroimage.2014.04.051)\n"
            "  - [De Leener B, Cohen-Adad J, Kadoury S. Automatic segmentation of the spinal cord and spinal canal "
            "coupled with vertebral labeling. IEEE Trans Med Imaging. 2015 Aug;34(8):1705-18.](https://doi.org/10.1109/tmi.2015.2437192)"
        )
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        help="Input image. Example: `ti.nii.gz`"
    )
    mandatory.add_argument(
        '-c',
        choices=['t1', 't2', 't2s', 'dwi'],
        help="Type of image contrast. If your contrast is not in the available options (t1, t2, t2s, dwi), use "
             "t1 (cord bright / CSF dark) or t2 (cord dark / CSF bright)"
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help='Output filename. Example: `spinal_seg.nii.gz`'
        )
    optional.add_argument(
        '-ofolder',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="Output folder."
    )
    optional.add_argument(
        '-down',
        metavar=Metavar.int,
        type=int,
        help="Down limit of the propagation. Default is 0."
    )
    optional.add_argument(
        '-up',
        metavar=Metavar.int,
        type=int,
        help="Up limit of the propagation. Default is the highest slice of the image."
    )
    optional.add_argument(
        '-mesh',
        action="store_true",
        help="Output: mesh of the spinal cord segmentation"
    )
    optional.add_argument(
        '-centerline-binary',
        action="store_true",
        help="Output: centerline as a binary image."
    )
    optional.add_argument(
        '-CSF',
        action="store_true",
        help="Output: CSF segmentation."
    )
    optional.add_argument(
        '-centerline-coord',
        action="store_true",
        help="Output: centerline in world coordinates."
    )
    optional.add_argument(
        '-cross',
        action="store_true",
        help="Output: cross-sectional areas."
    )
    optional.add_argument(
        '-init-tube',
        action="store_true",
        help="Output: initial tubular meshes."
    )
    optional.add_argument(
        '-low-resolution-mesh',
        action="store_true",
        help="Output: low-resolution mesh."
    )
    optional.add_argument(
        '-init-centerline',
        metavar=Metavar.file,
        help=textwrap.dedent("""
            Filename of centerline to use for the propagation. Use format `.txt` or `.nii`; see file structure in documentation.

            Replace filename by `viewer` to use interactive viewer for providing centerline. Example: `-init-centerline viewer`
        """),
    )
    optional.add_argument(
        '-init',
        metavar=Metavar.float,
        type=float,
        help="Axial slice where the propagation starts, default is middle axial slice."
    )
    optional.add_argument(
        '-init-mask',
        metavar=Metavar.file,
        help=textwrap.dedent("""
            Mask containing three center of the spinal cord, used to initiate the propagation.

            Replace filename by `viewer` to use interactive viewer for providing mask. Example: `-init-mask viewer`
        """),
    )
    optional.add_argument(
        '-mask-correction',
        metavar=Metavar.file,
        help="mask containing binary pixels at edges of the spinal cord on which the segmentation algorithm will be "
             "forced to register the surface. Can be used in case of poor/missing contrast between spinal cord and "
             "CSF or in the presence of artefacts/pathologies."
    )
    optional.add_argument(
        '-rescale',
        metavar=Metavar.float,
        type=float,
        default=1.0,
        help="Rescale the image (only the header, not the data) in order to enable segmentation on spinal cords with "
             "dimensions different than that of humans (e.g., mice, rats, elephants, etc.). For example, if the "
             "spinal cord is 2x smaller than that of human, then use `-rescale 2`"
    )
    optional.add_argument(
        '-radius',
        metavar=Metavar.float,
        type=float,
        help="Approximate radius (in mm) of the spinal cord. Default is 4."
    )
    optional.add_argument(
        '-nbiter',
        metavar=Metavar.int,
        type=int,
        help="Stop condition (affects only the Z propogation): number of iteration for the propagation for both "
             "direction. Default is 200."
    )
    optional.add_argument(
        '-max-area',
        metavar=Metavar.float,
        type=float,
        help="[mm^2], stop condition (affects only the Z propogation): maximum cross-sectional area. Default is 120."
    )
    optional.add_argument(
        '-max-deformation',
        metavar=Metavar.float,
        type=float,
        help="[mm], stop condition (affects only the Z propogation): maximum deformation per iteration. Default is "
             "2.5"
    )
    optional.add_argument(
        '-min-contrast',
        metavar=Metavar.float,
        type=float,
        help="[intensity value], stop condition (affects only the Z propogation): minimum local SC/CSF contrast, "
             "default is 50"
    )
    optional.add_argument(
        '-d',
        metavar=Metavar.float,
        type=float,
        help="trade-off between distance of most promising point (d is high) and feature strength (d is low), "
             "default depend on the contrast. Range of values from 0 to 50. 15-25 values show good results. Default "
             "is 10."
    )
    optional.add_argument(
        '-distance-search',
        metavar=Metavar.float,
        type=float,
        help="maximum distance of optimal points computation along the surface normals. Range of values from 0 to 30. "
             "Default is 15"
    )
    optional.add_argument(
        '-alpha',
        metavar=Metavar.float,
        type=float,
        help="Trade-off between internal (alpha is high) and external (alpha is low) forces. Range of values from 0 "
             "to 50. Default is 25."
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
    optional.add_argument(
        '-correct-seg',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable (1) or disable (0) the algorithm that checks and correct the output segmentation. More "
             "specifically, the algorithm checks if the segmentation is consistent with the centerline provided by "
             "isct_propseg."
    )

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_tempfile_args()

    return parser


def func_rescale_header(fname_data, rescale_factor, verbose=0):
    """
    Rescale the voxel dimension by modifying the NIFTI header qform. Write the output file in a temp folder.
    :param fname_data:
    :param rescale_factor:
    :return: fname_data_rescaled
    """
    img = nib.load(fname_data)
    # get qform
    qform = img.header.get_qform()
    # multiply by scaling factor
    qform[0:3, 0:3] *= rescale_factor
    # generate a new nifti file
    header_rescaled = img.header.copy()
    header_rescaled.set_qform(qform)
    # the data are the same-- only the header changes
    img_rescaled = Image(np.asanyarray(img.dataobj), hdr=header_rescaled)
    path_tmp = tmp_create(basename="propseg-rescale-header")
    fname_data_rescaled = os.path.join(path_tmp, os.path.basename(add_suffix(fname_data, "_rescaled")))
    img_rescaled.save(fname_data_rescaled)
    return fname_data_rescaled


def propseg(img_input, options_dict):
    """
    :param img_input: source image, to be segmented
    :param options_dict: arguments as dictionary
    :return: segmented Image
    """
    arguments = options_dict
    fname_input_data = img_input.absolutepath
    fname_data = os.path.abspath(fname_input_data)
    contrast_type = arguments.c
    contrast_type_conversion = {'t1': 't1', 't2': 't2', 't2s': 't2', 'dwi': 't1'}
    contrast_type_propseg = contrast_type_conversion[contrast_type]

    # Starting building the command
    cmd = ['isct_propseg', '-t', contrast_type_propseg]

    if arguments.o is not None:
        fname_out = arguments.o
    else:
        fname_out = os.path.basename(add_suffix(fname_data, "_seg"))

    if arguments.ofolder is not None:
        folder_output = os.path.abspath(arguments.ofolder)
    else:
        folder_output = str(pathlib.Path(fname_out).parent)
        fname_out = pathlib.Path(fname_out).name
    if not os.path.isdir(folder_output) and os.path.exists(folder_output):
        logger.error("output directory %s is not a valid directory", folder_output)
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    # We output to a temporary directory first, then copy the files to the real output directory later
    # This addresses https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3716.
    folder_output_tmp = tmp_create(basename="propseg")
    cmd.extend(['-o', folder_output_tmp])

    if arguments.down is not None:
        cmd.extend(["-down", str(arguments.down)])
    if arguments.up is not None:
        cmd.extend(["-up", str(arguments.up)])

    remove_temp_files = arguments.r

    verbose = int(arguments.v)
    # Update for propseg binary
    if verbose > 0:
        cmd.append("-verbose")

    # Output options
    if arguments.mesh:
        cmd.append("-mesh")
    if arguments.centerline_binary:
        cmd.append("-centerline-binary")
    if arguments.CSF:
        cmd.append("-CSF")
    if arguments.centerline_coord:
        cmd.append("-centerline-coord")
    if arguments.cross:
        cmd.append("-cross")
    if arguments.init_tube:
        cmd.append("-init-tube")
    if arguments.low_resolution_mesh:
        cmd.append("-low-resolution-mesh")
    # TODO: Not present. Why is this here? Was this renamed?
    # if arguments.detect_nii is not None:
    #     cmd.append("-detect-nii")
    # TODO: Not present. Why is this here? Was this renamed?
    # if arguments.detect_png is not None:
    #     cmd.append("-detect-png")

    # Helping options
    use_viewer = None
    use_optic = True  # enabled by default
    init_option = None
    rescale_header = arguments.rescale
    if arguments.init is not None:
        init_option = float(arguments.init)
        if init_option < 0:
            printv(f"Command-line usage error: {init_option} is not a valid value for '-init'", 1, 'error')
            sys.exit(1)
    if arguments.init_centerline is not None:
        if str(arguments.init_centerline) == "viewer":
            use_viewer = "centerline"
        elif str(arguments.init_centerline) == "hough":
            use_optic = False
        else:
            if rescale_header != 1.0:
                fname_labels_viewer = func_rescale_header(str(arguments.init_centerline), rescale_header, verbose=verbose)
            else:
                fname_labels_viewer = str(arguments.init_centerline)
            cmd.extend(["-init-centerline", fname_labels_viewer])
            use_optic = False
    if arguments.init_mask is not None:
        if str(arguments.init_mask) == "viewer":
            use_viewer = "mask"
        else:
            if rescale_header != 1.0:
                fname_labels_viewer = func_rescale_header(str(arguments.init_mask), rescale_header)
            else:
                fname_labels_viewer = str(arguments.init_mask)
            cmd.extend(["-init-mask", fname_labels_viewer])
            use_optic = False
    if arguments.mask_correction is not None:
        cmd.extend(["-mask-correction", str(arguments.mask_correction)])
    if arguments.radius is not None:
        cmd.extend(["-radius", str(arguments.radius)])
    # TODO: Not present. Why is this here? Was this renamed?
    # if arguments.detect_n is not None:
    #     cmd.extend(["-detect-n", str(arguments.detect_n)])
    # TODO: Not present. Why is this here? Was this renamed?
    # if arguments.detect_gap is not None:
    #     cmd.extend(["-detect-gap", str(arguments.detect_gap)])
    # TODO: Not present. Why is this here? Was this renamed?
    # if arguments.init_validation is not None:
    #     cmd.extend(["-init-validation"])
    if arguments.nbiter is not None:
        cmd.extend(["-nbiter", str(arguments.nbiter)])
    if arguments.max_area is not None:
        cmd.extend(["-max-area", str(arguments.max_area)])
    if arguments.max_deformation is not None:
        cmd.extend(["-max-deformation", str(arguments.max_deformation)])
    if arguments.min_contrast is not None:
        cmd.extend(["-min-contrast", str(arguments.min_contrast)])
    if arguments.d is not None:
        cmd.extend(["-d", str(arguments.d)])
    if arguments.distance_search is not None:
        cmd.extend(["-dsearch", str(arguments.distance_search)])
    if arguments.alpha is not None:
        cmd.extend(["-alpha", str(arguments.alpha)])

    # check if input image is in 3D. Otherwise itk image reader will cut the 4D image in 3D volumes and only take the first one.
    image_input = Image(fname_data)
    image_input_rpi = image_input.copy().change_orientation('RPI')
    nx, ny, nz, nt, px, py, pz, pt = image_input_rpi.dim
    if nt > 1:
        printv('ERROR: your input image needs to be 3D in order to be segmented.', 1, 'error')

    path_data, file_data, ext_data = extract_fname(fname_data)

    # rescale header (see issue #1406)
    if rescale_header != 1.0:
        fname_data_propseg = func_rescale_header(fname_data, rescale_header)
    else:
        fname_data_propseg = fname_data

    # add to command
    cmd.extend(['-i', fname_data_propseg])

    # if centerline or mask is asked using viewer
    if use_viewer:
        from spinalcordtoolbox.gui.base import AnatomicalParams
        from spinalcordtoolbox.gui.centerline import launch_centerline_dialog

        params = AnatomicalParams()
        if use_viewer == 'mask':
            params.num_points = 3
            params.interval_in_mm = 15  # superior-inferior interval between two consecutive labels
            params.starting_slice = 'midfovminusinterval'
        if use_viewer == 'centerline':
            # setting maximum number of points to a reasonable value
            params.num_points = 20
            params.interval_in_mm = 30
            params.starting_slice = 'top'
        im_data = Image(fname_data_propseg)

        im_mask_viewer = zeros_like(im_data)
        # im_mask_viewer.absolutepath = add_suffix(fname_data_propseg, '_labels_viewer')
        controller = launch_centerline_dialog(im_data, im_mask_viewer, params)
        fname_labels_viewer = add_suffix(fname_data_propseg, '_labels_viewer')

        if not controller.saved:
            printv('The viewer has been closed before entering all manual points. Please try again.', 1, 'error')
            sys.exit(1)
        # save labels
        controller.as_niftii(fname_labels_viewer)

        # add mask filename to parameters string
        if use_viewer == "centerline":
            cmd.extend(["-init-centerline", fname_labels_viewer])
        elif use_viewer == "mask":
            cmd.extend(["-init-mask", fname_labels_viewer])

    # If using OptiC
    elif use_optic:
        image_centerline = optic.detect_centerline(image_input, contrast_type, remove_temp_files, verbose)
        path_tmp = tmp_create(basename="propseg-centerline-optic")
        fname_centerline_optic = os.path.join(path_tmp, 'centerline_optic.nii.gz')
        image_centerline.save(fname_centerline_optic)
        cmd.extend(["-init-centerline", fname_centerline_optic])

    if init_option is not None:
        if init_option > 1:
            init_option /= (nz - 1)
        cmd.extend(['-init', str(init_option)])

    # enabling centerline extraction by default (needed by check_and_correct_segmentation() )
    cmd.append('-centerline-binary')

    # run propseg
    status, output = run_proc(cmd, verbose, raise_exception=False, is_sct_binary=True)

    # check status is not 0
    if not status == 0:
        printv('Automatic cord detection failed. Please initialize using `-init-centerline` or `-init-mask` (see help)',
               1, 'error')
        sys.exit(1)

    # rename output files and move them from the tmp directory to the output directory
    fname_seg_old = os.path.join(folder_output_tmp, add_suffix(os.path.basename(fname_data_propseg), "_seg"))
    fname_seg = os.path.join(folder_output, fname_out)
    mv(fname_seg_old, fname_seg)
    fname_centerline_old = os.path.join(folder_output_tmp, add_suffix(os.path.basename(fname_data_propseg), "_centerline"))
    fname_centerline = os.path.join(folder_output, os.path.basename(add_suffix(fname_data, "_centerline")))
    mv(fname_centerline_old, fname_centerline)

    # move over any additional files generated by sct_propseg (no renaming)
    for file in os.listdir(folder_output_tmp):
        mv(os.path.join(folder_output_tmp, file), os.path.join(folder_output, file))

    # if viewer was used, copy the labelled points to the output folder
    if use_viewer:
        fname_labels_viewer_new = os.path.join(folder_output, os.path.basename(add_suffix(fname_data, "_labels_viewer")))
        copy(fname_labels_viewer, fname_labels_viewer_new)
        # update variable (used later)
        fname_labels_viewer = fname_labels_viewer_new

    # check consistency of segmentation
    if arguments.correct_seg:
        check_and_correct_segmentation(fname_seg, fname_centerline, folder_output=folder_output, threshold_distance=3.0,
                                       remove_temp_files=remove_temp_files, verbose=verbose)

    # copy header from input to segmentation to make sure qform is the same
    printv("Copy header input --> output(s) to make sure qform is the same.", verbose)
    list_fname = [fname_seg, fname_centerline]
    if use_viewer:
        list_fname.append(fname_labels_viewer)
    for fname in list_fname:
        im = Image(fname)
        im.header = image_input.header
        im.save(dtype='int8', verbose=0)  # they are all binary masks hence fine to save as int8

    return Image(fname_seg)


def main(argv: Sequence[str]):
    parser = get_parser()
    if sys.platform.startswith("win32"):
        # This isn't *really* a parsing error, but it feels a little more official to display the help with this error
        parser.error("`sct_propseg` is not currently supported on native Windows installations. \n\n"
                     "For spinal cord segmentation, please migrate to the new and improved `sct_deepseg spinalcord` "
                     "tool, or consider using WSL to install SCT instead.\n\n"
                     "For further updates on `sct_propseg` Windows support, please visit:\n"
                     "https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3694")
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    fname_input_data = os.path.abspath(arguments.i)
    img_input = Image(fname_input_data)
    img_seg = propseg(img_input, arguments)
    fname_seg = img_seg.absolutepath
    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject
    qc_contrast = arguments.qc_contrast
    if path_qc is not None:
        generate_qc(fname_in1=fname_input_data, fname_seg=fname_seg, args=argv,
                    path_qc=os.path.abspath(path_qc), dataset=qc_dataset, subject=qc_subject,
                    contrast=qc_contrast, process='sct_propseg')
    display_viewer_syntax([fname_input_data, fname_seg], im_types=['anat', 'seg'], opacities=['', '1'], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
