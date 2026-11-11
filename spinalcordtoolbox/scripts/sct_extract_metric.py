#!/usr/bin/env python
#
# Extract metrics within spinal labels as defined by the white matter atlas and the template
# The folder atlas should have a .txt file that lists all tract files with labels.
#
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

# TODO: fetch vert level in atlas by default-- would be nice to output in csv
# TODO (not urgent): vertebral levels selection should only consider voxels of the selected levels in slices where
#  two different vertebral levels coexist (and not the whole slice)

import sys
import os
import argparse
from typing import Sequence
import textwrap

import numpy as np

from spinalcordtoolbox.metadata import read_label_file
from spinalcordtoolbox.aggregate_slicewise import check_labels, extract_metric, save_as_csv, Metric, LabelStruc
from spinalcordtoolbox.image import Image, add_suffix
from spinalcordtoolbox.centerline.core import get_centerline
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, list_type, parse_num_list, display_open
from spinalcordtoolbox.utils.sys import init_sct, printv, __data_dir__, set_loglevel
from spinalcordtoolbox.utils.fs import check_file_exist, extract_fname, get_absolute_path, TempFolder
from spinalcordtoolbox.scripts import sct_maths
from spinalcordtoolbox.metrics_to_PAM50 import build_pam50_agg_metric


class Param:
    def __init__(self):
        self.method = 'wa'
        self.path_label = os.path.join(__data_dir__, "PAM50", "atlas")
        self.verbose = 1
        self.vertebral_levels = ''
        self.slices_of_interest = ''  # 2-element list corresponding to zmin:zmax. example: '5:8'. For all slices, leave empty.
        self.average_all_labels = 0  # average all labels together after concatenation
        self.fname_output = 'extract_metric.csv'
        self.file_info_label = 'info_label.txt'
        self.perslice = None
        self.perlevel = None


class _ListLabelsAction(argparse.Action):
    """This class makes it possible to call the flag '-list-labels' without the need to input the required '-i'."""
    def __call__(self, parser, namespace, values, option_string=None):
        file_label = os.path.join(namespace.f, Param().file_info_label)
        check_file_exist(file_label, 0)
        with open(file_label, 'r') as default_info_label:
            label_references = default_info_label.read()
        txt_label = (
            f"List of labels in {file_label}:\n"
            f"--------------------------------------------------------------------------------------\n"
            f"{label_references}"
            f"--------------------------------------------------------------------------------------\n")
        print(txt_label)
        parser.exit()


def get_parser():

    param_default = Param()

    parser = SCTArgumentParser(
        description=(
            "This program extracts metrics (e.g., DTI or MTR) within labels. Labels could be a single file or "
            "a folder generated with 'sct_warp_template' containing multiple label files and a label "
            "description file (info_label.txt). The labels should be in the same space coordinates as the "
            "input image.\n"
            "\n"
            "The labels used by default are taken from the PAM50 template. To learn about the available PAM50 "
            "white/grey matter atlas labels and their corresponding ID values, please refer to: "
            "https://spinalcordtoolbox.com/overview/concepts/pam50.html#white-and-grey-matter-atlas-pam50-atlas\n"
            "\n"
            "To compute FA within labels 0, 2 and 3 within vertebral levels C2 to C7 using binary method:\n"
            "`sct_extract_metric -i dti_FA.nii.gz -l 0,2,3 -vert 2:7 -method bin`\n"
            "\n"
            "To compute average MTR in a region defined by a single label file (could be binary or 0-1 "
            "weighted mask) between slices 1 and 4:\n"
            "s`ct_extract_metric -i mtr.nii.gz -f "
            "my_mask.nii.gz -z 1:4 -method wa`")
    )
    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        help="Image file to extract metrics from. Example: `FA.nii.gz`"
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-f',
        metavar=Metavar.folder,
        default=os.path.join("label", "atlas"),
        help=(f"Single label file, or folder that contains WM tract labels. "
              f"Example: {param_default.path_label}")
    )
    optional.add_argument(
        '-l',
        metavar=Metavar.str,
        default='',
        help="Label IDs to extract the metric from. Default = all labels. Separate labels with `,`. To select a group "
             "of consecutive labels use `:`. Example: `1:3` is equivalent to `1,2,3`. Maximum Likelihood (or MAP) is "
             "computed using all tracts, but only values of the selected tracts are reported."
    )
    optional.add_argument(
        '-list-labels',
        action=_ListLabelsAction,
        nargs=0,
        help="List available labels. These labels are defined in the file `info_label.txt` located in the folder "
             "specified by the flag `-f`."
    )
    optional.add_argument(
        '-method',
        choices=['ml', 'map', 'wa', 'bin', 'median', 'max'],
        default=param_default.method,
        help=textwrap.dedent("""
            Method to extract metrics.

              - `ml`: maximum likelihood: This method is recommended for large labels and low noise. Also, this method should only be used with the PAM50 white/gray matter atlas, or with any custom atlas as long as the sum across all labels equals 1, in each voxel part of the atlas.
              - `map`: maximum a posteriori: Mean priors are estimated by maximum likelihood within three clusters (white matter, gray matter and CSF). Tract and noise variance are set with flag `-p`. This method should only be used with the PAM50 white/gray matter atlas, or with any custom atlas as long as the sum across all labels equals 1, in each voxel part of the atlas.
              - `wa`: weighted average
              - `bin`: binarize mask (threshold=0.5)
              - `median`: weighted median: This implementation of the median treats quantiles as a continuous (vs. discrete) function. For more details, see https://pypi.org/project/wquantiles/
              - `max`: for each z-slice of the input data, extract the max value for each slice of the input data.
        """),  # noqa 501 (line too long)
    )
    optional.add_argument(
        '-append',
        type=int,
        choices=(0, 1),
        default=0,
        help="Whether to append results as a new line in the output csv file instead of overwriting it. 0 = no, 1 = yes"
    )
    optional.add_argument(
        '-combine',
        type=int,
        choices=(0, 1),
        default=0,
        help="Whether to combine multiple labels into a single estimation. `0` = no, `1` = yes"
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        default=param_default.fname_output,
        help="File name of the output result file collecting the metric estimation results. Include the `.csv` "
             "file extension in the file name. Example: `extract_metric.csv`"
    )
    optional.add_argument(
        '-output-map',
        metavar=Metavar.file,
        default='',
        help="File name for an image consisting of the atlas labels multiplied by the estimated metric values "
             "yielding the metric value map, useful to assess the metric estimation and especially partial volume "
             "effects."
    )
    optional.add_argument(
        '-z',
        metavar=Metavar.str,
        default=param_default.slices_of_interest,
        help=textwrap.dedent("""
            Slice range to estimate the metric from. First slice is 0. Example: `5:23`

            You can also select specific slices using commas. Example: `0,2,3,5,12`

            Note: When using `-normalize-PAM50`, this argument still refers to slices in the native image space, since metrics are aggregated before being interpolated into PAM50 space.
        """),  # noqa: E501 (line too long)
    )
    optional.add_argument(
        '-perslice',
        type=int,
        choices=(0, 1),
        default=param_default.perslice,
        help=textwrap.dedent("""
            Whether to output one metric per slice instead of a single output metric. `0` = no, `1` = yes.

            Please note that when methods ml or map are used, outputting a single metric per slice and then averaging them all is not the same as outputting a single metric at once across all slices.
        """),  # noqa: E501 (line too long)
    )
    optional.add_argument(
        '-vert',
        metavar=Metavar.str,
        default=param_default.vertebral_levels,
        help="Vertebral levels to compute the metrics across. Example: `2:9` for C2 to T2. If you also specify a range "
             "of slices with flag `-z`, the intersection between the specified slices and vertebral levels will be "
             "considered."
    )
    optional.add_argument(
        '-vertfile',
        metavar=Metavar.file,
        default=os.path.join(".", "label", "template", "PAM50_levels.nii.gz"),
        help=textwrap.dedent("""
            Vertebral labeling file. Only use with flag `-vert`.

            The input Image and the vertebral labelling file must in the same voxel coordinate system and must match the dimensions between each other.
        """),
    )
    optional.add_argument(
        '-perlevel',
        type=int,
        metavar=Metavar.int,
        default=0,
        help=textwrap.dedent("""
            Whether to output one metric per vertebral level instead of a single output metric. `0` = no, `1` = yes.

            Please note that this flag needs to be used with the -vert option.
        """),
    )

    optional.add_argument(
        '-normalize-PAM50',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Set to 1 to interpolate the extracted metric values into PAM50 anatomical "
             "dimensions per slice. "
             "Requires `-vertfile` and `-perslice 1`. "
             "Inspired by: Valošek J, Bédard S et al. Imaging Neuroscience 2024. "
             "https://doi.org/10.1162/imag_a_00075"
    )

    advanced = parser.add_argument_group("FOR ADVANCED USERS")
    advanced.add_argument(
        '-param',
        metavar=Metavar.str,
        default='',
        help=textwrap.dedent("""
            Advanced parameters for the `map` method. All values must be provided, and separated with `,`.

              - First value: standard deviation of metrics across labels
              - Second value: standard deviation of the noise (assumed Gaussian)
        """),
    )
    advanced.add_argument(
        '-fix-label',
        metavar=Metavar.list,
        type=list_type(',', str),
        default='',
        help="When using ML or MAP estimations, if you do not want to estimate the metric in one label and fix its "
             "value to avoid effects on other labels, specify <label_ID>,<metric_value. Example: -fix-label 36,0 "
             "(Fix the CSF value)"
    )
    advanced.add_argument(
        '-norm-file',
        metavar=Metavar.file,
        default='',
        help='Filename of the label by which the user wants to normalize.'
    )
    advanced.add_argument(
        '-norm-method',
        choices=['sbs', 'whole'],
        default='',
        help=textwrap.dedent("""
            Method to use for normalization:

              - `sbs`: normalization slice-by-slice
              - `whole`: normalization by the metric value in the whole label for all slices.
        """),
    )
    advanced.add_argument(
        '-mask-weighted',
        metavar=Metavar.file,
        default='',
        help="Nifti mask to weight each voxel during ML or MAP estimation. Example: PAM50_wm.nii.gz"
    )
    advanced.add_argument(
        '-discard-neg-val',
        choices=('0', '1'),
        default='0',
        help='Whether to discard voxels with negative value when computing metrics statistics. 0 = no, 1 = yes'
    )

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_tempfile_args()

    return parser


def main(argv: Sequence[str]):
    # Ensure that the "-list-labels" argument is always parsed last. That way, if `-f` is passed, then `-list-labels`
    # will see the new location and look there. (https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3634)
    if "-list-labels" in argv:
        argv = [s for s in argv if s != "-list-labels"] + ["-list-labels"]

    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    param_default = Param()

    fname_data = get_absolute_path(arguments.i)
    path_label = arguments.f
    method = arguments.method
    fname_output = arguments.o
    append_csv = arguments.append
    combine_labels = arguments.combine
    labels_user = arguments.l
    slices = parse_num_list(arguments.z)
    levels = parse_num_list(arguments.vert)
    fname_vert_level = arguments.vertfile
    perslice = arguments.perslice
    perlevel = arguments.perlevel
    normalize_pam50 = arguments.normalize_PAM50

    # check if path_label is a file (e.g., single binary mask) instead of a folder (e.g., SCT atlas structure which
    # contains info_label.txt file)
    if os.path.isfile(path_label):
        # Label is a single file
        indiv_labels_ids = [0]
        indiv_labels_files = [path_label]
        combined_labels_ids = []
        label_struc = {0: LabelStruc(id=0,
                                     name=extract_fname(path_label)[1],
                                     filename=path_label)}
        # set path_label to empty string, because indiv_labels_files will replace it from now on
        path_label = ''
    elif os.path.isdir(path_label):
        # Labels is an SCT atlas folder structure
        # Parse labels according to the file info_label.txt
        # Note: the "combined_labels_*" is a list of single labels that are defined in the section defined by the keyword
        # "# Keyword=CombinedLabels" in info_label.txt.
        # TODO: redirect to appropriate Sphinx documentation
        # TODO: output Class instead of multiple variables.
        #   Example 1:
        #     label_struc[2].id = (2)
        #     label_struc[2].name = "left fasciculus cuneatus"
        #     label_struc[2].filename = "PAM50_atlas_02.nii.gz"
        #   Example 2:
        #     label_struc[51].id = (1, 2, 3, ..., 29)
        #     label_struc[51].name = "White Matter"
        #     label_struc[51].filename = ""  # no name because it is combined
        indiv_labels_ids, indiv_labels_names, indiv_labels_files, \
            combined_labels_ids, combined_labels_names, combined_labels_id_groups, map_clusters \
            = read_label_file(path_label, param_default.file_info_label)

        label_struc = {}
        # fill IDs for indiv labels
        for i_label in range(len(indiv_labels_ids)):
            label_struc[indiv_labels_ids[i_label]] = LabelStruc(id=indiv_labels_ids[i_label],
                                                                name=indiv_labels_names[i_label],
                                                                filename=indiv_labels_files[i_label],
                                                                map_cluster=[indiv_labels_ids[i_label] in map_cluster for
                                                                             map_cluster in map_clusters].index(True))
        # fill IDs for combined labels
        # TODO: problem for defining map_cluster: if labels overlap two regions, e.g. WM and GM (e.g. id=50),
        #  map_cluster will take value 0, which is wrong.
        for i_label in range(len(combined_labels_ids)):
            label_struc[combined_labels_ids[i_label]] = LabelStruc(id=combined_labels_id_groups[i_label],
                                                                   name=combined_labels_names[i_label],
                                                                   map_cluster=[indiv_labels_ids[i_label] in map_cluster for
                                                                                map_cluster in map_clusters].index(True))
    else:
        raise RuntimeError(f'{path_label} does not exist')

    # check syntax of labels asked by user
    labels_id_user = check_labels(indiv_labels_ids + combined_labels_ids, parse_num_list(labels_user))
    nb_labels = len(indiv_labels_files)

    # Load data and systematically reorient to RPI because we need the 3rd dimension to be z
    printv('\nLoad metric image...', verbose)
    input_im = Image(fname_data).change_orientation("RPI")

    data = Metric(data=input_im.data, label='')
    # Load labels
    labels_tmp = np.empty([nb_labels], dtype=object)
    for i_label in range(nb_labels):
        im_label = Image(os.path.join(path_label, indiv_labels_files[i_label])).change_orientation("RPI")
        labels_tmp[i_label] = np.expand_dims(im_label.data, 3)  # TODO: generalize to 2D input label
    labels = np.concatenate(labels_tmp[:], 3)  # labels: (x,y,z,label)
    # Load vertebral levels
    temp_folder = None
    if os.path.isfile(fname_vert_level):
        # Extract centerline of vertebral levels
        im_vertlevel = Image(fname_vert_level)
        # Binarize vertebral levels before getting centerline
        im_vertlevel_bin = im_vertlevel.copy()
        im_vertlevel_bin.data[im_vertlevel_bin.data > 0] = 1
        # Create temp path for outputs
        temp_folder = TempFolder(basename="optic-detect-centerline", verbose=verbose)
        path_temp = temp_folder.get_path()
        # Extract centerline from segmentation
        im_centerline, _, _, _ = get_centerline(im_vertlevel_bin)
        fname_ctl = os.path.join(path_temp, add_suffix(os.path.basename(fname_vert_level), '_ctl'))
        im_centerline.save(fname_ctl)
        fname_ctl_levels = os.path.join(path_temp, add_suffix(os.path.basename(fname_vert_level), '_ctl_levels'))
        # Mask the centerline with the vertebral levels
        sct_maths.main(argv=['-i', fname_ctl, '-mul', fname_vert_level, '-o', fname_ctl_levels])
        # Use levels on centerline instead
        fname_vert_level = fname_ctl_levels
    else:
        # The severity of a missing vertlevel file depends on if levels was passed
        message_type = 'error' if levels else 'warning'
        message = ("Cannot aggregate by vert level." if levels else
                   "Vert level information will not be displayed.")
        printv(f"Vertebral level file {fname_vert_level} does not exist. {message} "
               f"To use vertebral level information, you may need to run "
               f"`sct_warp_template` to generate the appropriate level file in your working directory.", type=message_type)
        fname_vert_level = None
        if normalize_pam50:
            parser.error("Option '-normalize-PAM50' requires a valid '-vertfile'.")
    # Get dimensions of data and labels
    nx, ny, nz = data.data.shape
    nx_atlas, ny_atlas, nz_atlas, nt_atlas = labels.shape

    # Check dimensions consistency between atlas and data
    if (nx, ny, nz) != (nx_atlas, ny_atlas, nz_atlas):
        printv('\nERROR: Metric data and labels DO NOT HAVE SAME DIMENSIONS.', 1, type='error')

    # Combine individual labels for estimation
    if combine_labels:
        if len(labels_id_user) == 1:
            # Trying to combine 1 label may result in https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4467
            printv("Ignoring `-combine` arg since only 1 label value was provided", 1, type='warning')
        else:
            # If user is trying to combine more than one label that is part of CombinedLabels, exit with error
            if len(labels_id_user) != 1 and any(element in combined_labels_ids for element in labels_id_user):
                printv('\nERROR: You are trying to combine multiple labels that are already combined (under '
                       'section "# Combined labels" the info_label.txt file. Instead, enter the all the labels that '
                       'you wish to combine from the list "# Keyword=IndivLabels".', 1, type='error')
            else:
                # Add entry with internal ID value (99) which corresponds to combined labels
                label_struc[99] = LabelStruc(id=labels_id_user, name=','.join([str(i) for i in labels_id_user]),
                                             map_cluster=None)
                labels_id_user = [99]

    if normalize_pam50 and not perslice:
        parser.error("Option '-normalize-PAM50' requires option '-perslice 1'.")
    if normalize_pam50 and perlevel:
        parser.error("Option '-normalize-PAM50' is not compatible with option '-perlevel 1'.")

    fname_vert_level_PAM50 = os.path.join(__data_dir__, 'PAM50', 'template', 'PAM50_levels.nii.gz')

    for id_label in labels_id_user:
        printv(f'Estimation for label: {label_struc[id_label].name}', verbose)

        agg_metric = extract_metric(data, labels=labels, slices=slices, levels=levels, perslice=perslice,
                                    perlevel=perlevel, fname_vert_level=fname_vert_level, method=method,
                                    label_struc=label_struc, id_label=id_label, indiv_labels_ids=indiv_labels_ids)
        if normalize_pam50:
            agg_metric = build_pam50_agg_metric(
                agg_metric_native=agg_metric,
                nz_native=nz,
                label_name=label_struc[id_label].name,
                method=method,
                fname_vert_level=fname_vert_level,
                fname_vert_level_PAM50=fname_vert_level_PAM50)

        save_as_csv(agg_metric, fname_output, fname_in=fname_data, append=append_csv)
        append_csv = True  # when looping across labels, need to append results in the same file
    if arguments.r and temp_folder is not None:
        printv("\nRemove temporary files...", verbose)
        temp_folder.cleanup()
    display_open(fname_output)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
