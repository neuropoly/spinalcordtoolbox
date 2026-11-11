#!/usr/bin/env python
#
# Perform various types of processing from the spinal cord segmentation (e.g. extract centerline, compute CSA, etc.).
# (extract_centerline) extract the spinal cord centerline from the segmentation. Output file is an image in the same
# space as the segmentation.
#
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

# TODO: the import of scipy.misc imsave was moved to the specific cases (orth and ellipse) in order to avoid issue #62. This has to be cleaned in the future.

import sys
import os
import logging
import argparse
from typing import Sequence
import textwrap
from warnings import warn
from spinalcordtoolbox.utils.sys import stylize
from time import sleep
import numpy as np

from spinalcordtoolbox.reports import qc2
from spinalcordtoolbox.aggregate_slicewise import aggregate_per_slice_or_level, save_as_csv, func_wa, func_std, \
    func_sum, merge_dict, normalize_csa
from spinalcordtoolbox.process_seg import compute_shape, get_missing_slices
from spinalcordtoolbox.scripts import sct_maths
from spinalcordtoolbox.csa_pmj import get_slices_for_pmj_distance
from spinalcordtoolbox.metrics_to_PAM50 import interpolate_metrics
from spinalcordtoolbox.centerline.core import ParamCenterline
from spinalcordtoolbox.image import add_suffix, splitext, Image
from spinalcordtoolbox.labels import project_centerline, label_regions_from_reference
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.utils.fs import get_absolute_path, copy, TempFolder
from spinalcordtoolbox.utils.sys import __sct_dir__, init_sct, sct_progress_bar, set_loglevel
from spinalcordtoolbox.utils.shell import (ActionCreateFolder, Metavar, SCTArgumentParser,
                                           display_open, parse_num_list)
from spinalcordtoolbox.utils.sys import __data_dir__, LazyLoader

pd = LazyLoader("pd", globals(), "pandas")
mpl_ticker = LazyLoader("mpl_ticker", globals(), "matplotlib.ticker")

logger = logging.getLogger(__name__)


class SeparateNormArgs(argparse.Action):
    """Separates predictors from their values and puts the results in a dict"""
    def __call__(self, parser, namespace, values, option_string=None):
        pred = values[::2]
        val = values[1::2]
        if len(pred) != len(val):
            parser.error("Values for normalization need to be specified for each predictor.")
        try:
            data_subject = {p: float(v) for p, v in zip(pred, val)}
        except ValueError as e:
            parser.error(f"Non-numeric value passed to '-normalize': {e}")
        setattr(namespace, self.dest, data_subject)


def get_parser(ascor=False):
    """
    :param bool: ascor: Whether parser is for `sct_compute_ascor` (a light wrapper for `sct_process_segmentation`
                        that skips some irrelevant args.)
    :return: Returns the parser with the command line documentation contained in it.
    """
    # Determine whether we are creating a parser for `sct_process_segmentation` (or alternatively `sct_compute_ascor`)
    is_sct_process_segmentation = (not ascor)
    # Some arguments aren't relevant for `sct_compute_ascor`, so we want to keep them by default, but skip them if `ascor=True`:
    #   - `-i`: `sct_compute_ascor` uses 2 different inputs (`-i-SC` and `-i-canal`)
    #   - `-normalize`: CSA normalization not relevant for aSCOR, since we are computing a ratio
    #   = `-qc`: QC reports are only available for `-pmj` (displaying extent etc.) which is less relevant for ascor calculation

    # Initialize the parser
    parser = SCTArgumentParser(
        description=(
            "Compute the following morphometric measures based on the spinal cord segmentation:\n"
            "  - area [mm^2]: Cross-sectional area, measured by counting pixels in each slice. Partial volume can be "
            "accounted for by inputting a mask comprising values within [0,1]. Can be normalized when specifying the flag `-normalize`\n"
            "  - angle_AP, angle_RL: Estimated angle between the cord centerline and the axial slice. This angle is "
            "used to correct for morphometric information.\n"
            "  - diameter_AP: Measured as average across 3 mm extent centred at cord mask center of mass.\n"
            "  - length_anterior: Anterior length of the spinal cord along the AP axis, measured as average across "
            "3 mm extent centred at cord mask center of mass, from the center of mass to the anterior cord edge.\n"
            "  - length_posterior: Posterior length of the spinal cord along the AP axis, measured as average across "
            "3 mm extent centred at cord mask center of mass, from the center of mass to the posterior cord edge.\n"
            "  - diameter_RL: Measured as the major axis of the ellipse fitted to the cord.\n"
            "  - eccentricity: Eccentricity of the ellipse that has the same second-moments as the spinal cord. "
            "The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis "
            "length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.\n"
            "  - orientation: angle (in degrees) between the AP axis of the spinal cord and the AP axis of the "
            "image\n"
            "  - solidity: CSA(spinal_cord) / CSA_convex(spinal_cord). The more ellipse-shaped the cord is (i.e. the "
            "closer the perimeter of the cord is to being fully convex), the closer the solidity ratio will be to 1. "
            "This metric is interesting for detecting concave regions (e.g., in case of strong compression).\n"
            "  - length: Length of the segmentation, computed by summing the slice thickness (corrected for the "
            "centerline angle at each slice) across the specified superior-inferior region.\n"
            "\n"
            "  - symmetry_dice:  Dice score between the left and right (RL) hemicords or anterior and posterior (AP) halves of the spinal cord.\n"
            "  - symmetry_hausdorff [mm]:  Hausdorff distance between the left and right (RL) hemicords or anterior and posterior (AP) halves of the spinal cord.\n"
            "  - symmetry_difference [mm^2]:  Absolute difference between the left and right (RL) hemicords or anterior and posterior (AP) halves of the spinal cord.\n"
            "  - area_quadrant:  Cross-sectional area of the spinal cord in each quadrant (anterior-left, anterior-right, posterior-left, posterior-right).\n"
            "IMPORTANT: There is a limit to the precision you can achieve for a given image resolution. SCT does not "
            "truncate spurious digits when performing angle correction, so please keep in mind that there may be "
            "non-significant digits in the computed values. You may wish to compare angle-corrected values with "
            "their corresponding uncorrected values to get a sense of the limits on precision.\n"
            "\n"
            "To select the region to compute metrics over, choose one of the following arguments:\n"
            "   1. `-z`: Select axial slices based on slice index.\n"
            "   2. `-pmj` + `-pmj-distance` + `-pmj-extent`: Select axial slices based on distance from pontomedullary "
            "junction.\n"
            "      (For options 1 and 2, you can also add '-perslice' to compute metrics for each axial slice, rather "
            "than averaging.)\n"
            "   3. `-vert` + `-vertfile`: Select a region based on vertebral labels instead of individual slices.\n"
            "      (For option 3, you can also add `-perlevel` to compute metrics for each vertebral level, rather "
            "than averaging.)\n"
            "\n"
            "References:\n"
            "  - `-pmj`/`-normalize`:\n"
            "    Bédard S, Cohen-Adad J. Automatic measure and normalization of spinal cord cross-sectional area using "
            "the pontomedullary junction. Frontiers in Neuroimaging 2022.\n"
            "    https://doi.org/10.3389/fnimg.2022.1031253\n"
            "  - `-normalize-PAM50`:\n"
            "    Valošek J, Bédard S, Keřkovský M, Rohan T, Cohen-Adad J. A database of the healthy human spinal cord "
            "morphometry in the PAM50 template space. Imaging Neuroscience 2024; 2 1–15.\n"
            "    https://doi.org/10.1162/imag_a_00075"
        )
    )

    mandatory = parser.mandatory_arggroup
    if is_sct_process_segmentation:
        mandatory.add_argument(
            '-i',
            metavar=Metavar.file,
            type=get_absolute_path,
            help="Mask to compute morphometrics from. Could be binary or weighted. E.g., spinal cord segmentation. "
                 "Example: seg.nii.gz"
        )
    optional = parser.optional_arggroup
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        default='csa.csv',
        help="Output file name (add extension)."
    )
    optional.add_argument(
        '-append',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Append results as a new line in the output csv file instead of overwriting it."
    )
    optional.add_argument(
        '-z',
        metavar=Metavar.str,
        type=parse_num_list,
        default='',
        help="Slice range to compute the metrics across. Example: `5:23`. "
             "Note: When using `-normalize-PAM50`, this argument refers to slices in the PAM50 template space, "
             "since metrics are interpolated into PAM50 space before being aggregated."
    )
    optional.add_argument(
        '-perslice',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Set to 1 to output one metric per slice instead of a single output metric. Please note that when "
             "methods ml or map is used, outputting a single metric per slice and then averaging them all is not the "
             "same as outputting a single metric at once across all slices."
    )
    optional.add_argument(
        '-vert',
        metavar=Metavar.str,
        type=parse_num_list,
        default='',
        help="Vertebral levels to compute the metrics across. Example: 2:9 for C2 to T2. If you also specify a range of "
             "slices with flag `-z`, the intersection between the specified slices and vertebral levels will be "
             "considered."
    )
    optional.add_argument(
        '-vertfile',
        metavar=Metavar.str,
        help=textwrap.dedent("""
            Vertebral labeling file generated by sct_label_vertebrae or sct_warp_template. Only use with flag `-vert`.

            The input and the vertebral labelling file must be in the same voxel coordinate system and must match the dimensions between each other.
            Example: ./label/template/PAM50_levels.nii.gz
            This flag will be deprecated in favor of -discfile in the future.
        """),
    )
    optional.add_argument(
        '-discfile',
        metavar=Metavar.str,
        help=textwrap.dedent("""
            File with single-voxel labels identifying the intervertebral discs generated with sct_deepseg spine or sct_label_utils.
            Used with `-vert` to aggregate metrics within vertebral levels. Disc labels will be projected onto the spinal
            cord to identify vertebral levels.
            Example: ./label/template/PAM50_label_disc.nii.gz
        """),
    )
    optional.add_argument(
        '-perlevel',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Set to 1 to output one metric per vertebral level instead of a single output metric. This flag needs "
             "to be used with flag `-vert`."
    )
    optional.add_argument(
        '-angle-corr',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=1,
        help="Angle correction for computing morphometric measures. When angle correction is used, the cord within "
             "the slice is stretched/expanded by a factor corresponding to the cosine of the angle between the "
             "centerline and the axial plane. If the cord is already quasi-orthogonal to the slab, you can set "
             "-angle-corr to 0."
    )
    optional.add_argument(
        '-centerline',
        metavar=Metavar.str,
        help=textwrap.dedent("""
        Mask image containing a SC seg or centerline. (Optional: If not provided, the centerline will be derived from `-i` instead.)

        The `-centerline` image is used with other flags:
          - `-angle-corr 1`: If provided, angles will be computed using the shape of the `-centerline` image (which will then be used to angle-correct the metrics for `-i`).
          - `-discfile`: If provided, the disc labels will be projected onto the `-centerline` image (which will then be used to aggregate the metrics for `-i`).

        You should specify this option if you want to override the centerline that would be derived from `-i`. For example:
          - You are calculating the CSA of an irregular segmentation (e.g. lesion mask) that would produce poor angle correction.
          - You are calculating the CSA of multiple segmentations for the same subject and want a consistent centerline for both.
        """)
    )
    optional.add_argument(
        '-centerline-algo',
        choices=['polyfit', 'bspline', 'linear', 'nurbs'],
        default='bspline',
        help="Algorithm for centerline fitting. Only relevant with `-angle-corr 1`."
    )
    optional.add_argument(
        '-centerline-smooth',
        metavar=Metavar.int,
        type=int,
        default=30,
        help="Degree of smoothing for centerline fitting. Only use with `-centerline-algo {bspline, linear}`."
    )
    optional.add_argument(
        '-centerline-exclude-missing',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        # For aSCOR computation, we don't need to display this argument since aSCOR will always turn this behavior on
        help=(argparse.SUPPRESS if ascor else
              "By default, an error will be thrown if `-centerline` does not completely cover the input segmentation. "
              "Setting this flag to 1 will bypass the error, by skipping segmentation slices that aren't covered by "
              "the centerline. Use with caution; sometimes it is preferable to manually correct the `-centerline` image "
              "instead (so that the `-centerline` image covers all of the segmentation slices).")
    )
    optional.add_argument(
        '-pmj',
        metavar=Metavar.file,
        type=get_absolute_path,
        help="Ponto-Medullary Junction (PMJ) label file. "
             "Example: `pmj.nii.gz`"
    )
    optional.add_argument(
        '-pmj-distance',
        type=float,
        metavar=Metavar.float,
        help="Distance (mm) from Ponto-Medullary Junction (PMJ) to the center of the mask used to compute morphometric "
             "measures. (To be used with flag `-pmj`.)"
    )
    optional.add_argument(
        '-pmj-extent',
        type=float,
        metavar=Metavar.float,
        default=20.0,
        help="Extent (in mm) for the mask used to compute morphometric measures. Each slice covered by the mask is "
             "included in the calculation. (To be used with flag `-pmj` and `-pmj-distance`.)"
    )
    optional.add_argument(
        '-anat',
        metavar=Metavar.file,
        default=None,
        type=get_absolute_path,
        help="Input image used to compute spinal cord orientation (using HOG method). It is required to compute symmetry and quadrants area metrics. "
             "Example: t2.nii.gz"
    )
    if is_sct_process_segmentation:
        optional.add_argument(
            '-normalize',
            metavar=Metavar.list,
            action=SeparateNormArgs,
            nargs="+",
            help="Normalize CSA values ('MEAN(area)').\n"
                 "Two models are available:\n"
                 "    1. sex, brain-volume, thalamus-volume.\n"
                 "    2. sex, brain-volume.\n"
                 "Specify each value for the subject after the corresponding predictor.\n"
                 "Example:\n    `-normalize sex 0 brain-volume 960606.0 thalamus-volume 13942.0` \n"
                 "*brain-volume and thalamus-volume are in mm^3. For sex, female: 0, male: 1.\n"
                 "\n"
                 "The models were generated using T1w brain images from 804 healthy (non-pathological) participants "
                 "ranging from 48 to 80 years old, taken from the UK Biobank dataset.\n"
                 "For more details on the subjects and methods used to create the models, go to: "
                 "https://github.com/sct-pipeline/ukbiobank-spinalcord-csa#readme \n"  # TODO add ref of the paper
                 "Given the risks and lack of consensus surrounding CSA normalization, we recommend thoroughly reviewing "
                 "the literature on this topic before applying this feature to your data.\n"
        )
    optional.add_argument(
        '-normalize-PAM50',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Set to 1 to bring the metrics in the PAM50 anatomical dimensions perslice. `-vertfile` and `-perslice` need to be specified."
    )
    if is_sct_process_segmentation:
        optional.add_argument(
            '-qc',
            metavar=Metavar.folder,
            type=os.path.abspath,
            action=ActionCreateFolder,
            help="The path where the quality control generated content will be saved."
                 " The QC report is only available for PMJ-based CSA (with flag `-pmj`)."
        )
        optional.add_argument(
            '-qc-image',
            metavar=Metavar.str,
            help="Input image to display in QC report. Typically, it would be the "
                 "source anatomical image used to generate the spinal cord "
                 "segmentation. This flag is mandatory if using flag `-qc`."
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
    parser.add_tempfile_args()

    return parser


def _make_figure(metric, fit_results):
    """
    Make a graph showing CSA and angles per slice.
    :param metric: Dictionary of metrics
    :param fit_results: class centerline.core.FitResults()
    :return: image object
    """
    import tempfile
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fname_img = f'{tempfile.NamedTemporaryFile().name}.png'
    z, csa, angle_ap, angle_rl = [], [], [], []
    for key, value in metric.items():
        z.append(key[0])
        csa.append(value['MEAN(area)'])
        angle_ap.append(value['MEAN(angle_AP)'])
        angle_rl.append(value['MEAN(angle_RL)'])

    z_ord = np.argsort(z)
    z, csa, angle_ap, angle_rl = (
        [np.array(x)[z_ord] for x in (z, csa, angle_ap, angle_rl)]
    )

    # Make figure
    fig = Figure(figsize=(8, 7), tight_layout=True)  # 640x700 pix
    FigureCanvas(fig)
    # If -angle-corr was set to 1, fit_results exists and centerline fitting results are displayed
    if fit_results is not None:
        ax = fig.add_subplot(311)
        ax.plot(z, csa, 'k')
        ax.plot(z, csa, 'k.')
        ax.grid(True)
        ax.set_ylabel('CSA [$mm^2$]')
        ax.set_xticklabels([])
        ax.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))

        ax = fig.add_subplot(312)
        ax.grid(True)
        ax.plot(z, angle_ap, 'b', label='_nolegend_')
        ax.plot(z, angle_ap, 'b.')
        ax.plot(z, angle_rl, 'r', label='_nolegend_')
        ax.plot(z, angle_rl, 'r.')
        ax.legend(['Rotation about AP axis', 'Rotation about RL axis'])
        ax.set_ylabel('Angle [$deg$]')
        ax.set_xticklabels([])
        ax.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))

        ax = fig.add_subplot(313)
        ax.grid(True)
        # find a way to condense the following lines
        zmean_list, xmean_list, xfit_list, ymean_list, yfit_list, zref_list = [], [], [], [], [], []
        for i, value in enumerate(fit_results.data.zref):
            if value in z:
                zmean_list.append(fit_results.data.zmean[i])
                xmean_list.append(fit_results.data.xmean[i])
                xfit_list.append(fit_results.data.xfit[i])
                ymean_list.append(fit_results.data.ymean[i])
                yfit_list.append(fit_results.data.yfit[i])
                zref_list.append(fit_results.data.zref[i])
        ax.plot(zmean_list, xmean_list, 'b.', label='_nolegend_')
        ax.plot(zref_list, xfit_list, 'b')
        ax.plot(zmean_list, ymean_list, 'r.', label='_nolegend_')
        ax.plot(zref_list, yfit_list, 'r')
        ax.legend(['Fitted (RL)', 'Fitted (AP)'])
        ax.set_ylabel('Centerline [$vox$]')
        ax.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))
    else:
        ax = fig.add_subplot(111)
        ax.plot(z, csa, 'k')
        ax.plot(z, csa, 'k.')
        ax.grid(True)
        ax.set_ylabel('CSA [$mm^2$]')

    ax.set_xlabel('Slice (Inferior-Superior direction)')
    fig.savefig(fname_img)

    return fname_img


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    # Initialization
    group_funcs = (('MEAN', func_wa), ('STD', func_std))  # functions to perform when aggregating metrics along S-I

    fname_segmentation = arguments.i
    fname_image = arguments.anat

    file_out = os.path.abspath(arguments.o)
    append = bool(arguments.append)
    levels = arguments.vert
    fname_vert_level = None
    normalize_pam50 = arguments.normalize_PAM50
    temp_folder = None
    # make sure we have a valid VertLevel file (used for aggregation + VertLevel column)
    if arguments.vertfile is not None and arguments.discfile is not None:
        parser.error("Both '-vertfile' and '-discfile' were specified. Please only specify one of these options.")
    elif arguments.discfile is not None:
        fname_vert_level = arguments.discfile
    elif arguments.vertfile is not None:
        fname_vert_level = arguments.vertfile
        warn(
            stylize(
                "`-vertfile flag` is deprecated, and will be removed in a future version of SCT. Please use "
                "`-discfile` instead (single-voxel labels identifying the intervertebral discs).", ["Red", "Bold"]
            ), DeprecationWarning
        )
        sleep(3)  # Give the user 3 seconds to read the message
    else:
        logger.info("No -vertfile/-discfile argument provided. Attempting to get VertLevel "
                    "information from local PAM50 warped template file (if it exists).")
        fname_vert_level = os.path.join('.', 'label', 'template', 'PAM50_levels.nii.gz')
    # Make sure that the vertfile exists before processing it
    if not os.path.isfile(fname_vert_level):
        logger.warning(f"Vertebral level file {fname_vert_level} does not exist. Vert level information will "
                       f"not be displayed. To use vertebral level information, specify either -vertfile or -discfile."
                       f"For generating the required files, please review sct_process_segmentation -h for these arguments.")
        # Discard the default '-vertfile', so that we don't attempt to find vertebral levels
        fname_vert_level = None
        # If `fname_vert_level` is invalid but vert levels are required, raise an error.
        if normalize_pam50:
            parser.error("Option '-normalize-PAM50' requires a valid vertebral level file ('-vertfile' or '-discfile').")
        elif levels:
            parser.error("Option '-vert' requires a valid vertebral level file ('-vertfile' or '-discfile').")

    # Vertfile exists, so pre-process it if it's a `-discfile`
    else:
        if arguments.discfile is not None:
            # Copy the input files to the tempdir
            temp_folder = TempFolder(basename="process-segmentation", verbose=verbose)
            path_tmp_seg = temp_folder.copy_from(fname_segmentation)
            # label the segmentation by default, mimicking the old `-vertfile` workflow
            path_tmp_ctl = path_tmp_seg
            # however, if a centerline file is provided by the user, label that instead
            # we do this to ensure that consistent labeling is used for across multiple calls to
            # sct_process_segmentation in tandem, for example for `sct_compute_ascor`
            if arguments.centerline is not None:
                path_tmp_ctl = temp_folder.copy_from(arguments.centerline)
            path_tmp_vert_level = temp_folder.copy_from(fname_vert_level)
            # Project discs labels onto centerline
            discs_projected = project_centerline(Image(path_tmp_ctl), Image(path_tmp_vert_level))
            discs_projected.save(add_suffix(path_tmp_vert_level, '_projected'), mutable=True)
            # Use the projected disc labels to extract a labeled centerline from the input segmentation
            ctl_projected = label_regions_from_reference(Image(path_tmp_seg), discs_projected, centerline=True)
            ctl_projected.save(add_suffix(path_tmp_vert_level, '_projected_centerline'), mutable=True)
            # If requested, save the projected centerline to the same directory as the input discfile
            if verbose == 2:
                copy(ctl_projected.absolutepath, os.path.dirname(os.path.abspath(fname_vert_level)))
            # Overwrite the input argument so that the labeled centerline (in the tmpdir) is used from now on
            fname_vert_level = ctl_projected.absolutepath

    perlevel = bool(arguments.perlevel)
    slices = arguments.z
    perslice = bool(arguments.perslice)
    angle_correction = bool(arguments.angle_corr)
    fname_pmj = arguments.pmj
    distance_pmj = arguments.pmj_distance
    extent_pmj = arguments.pmj_extent
    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject
    qc_contrast = arguments.qc_contrast

    fname_centerline = arguments.centerline
    param_centerline = ParamCenterline(
        algo_fitting=arguments.centerline_algo,
        smooth=arguments.centerline_smooth,
        minmax=True)
    # Exclude slices in segmentation where centerline is missing
    if angle_correction and fname_centerline:
        im_seg, im_ctl = Image(fname_segmentation), Image(fname_centerline)
        missing_slices = get_missing_slices(im=im_seg, im_mask=im_ctl)
        if missing_slices:
            # Default behavior: Raise an error
            if not arguments.centerline_exclude_missing:
                parser.error(f"The centerline image does not cover slice(s) '{missing_slices}' of the segmentation, "
                             f"meaning that angle correction cannot be computed for these slices. To bypass this error, "
                             f"you can:\n"
                             f"  a. Set the flag `-centerline-exclude-missing 1` to exclude these slices from the analysis \n"
                             f"  b. Set the flag `-angle-corr 0` to skip angle correction altogether for this analysis, or \n"
                             f"  c. Provide a centerline image that fully covers the segmentation")
            # Opt-in behavior: Set the missing slices to 0 in the segmentation
            else:
                logger.warning(f"The centerline image does not cover slice(s) '{missing_slices}' of the segmentation. "
                               f"These slices will be excluded from the analysis.")
                temp_folder = TempFolder(basename="process-segmentation-modified-seg", verbose=verbose)
                fname_segmentation = temp_folder.copy_from(fname_segmentation)
                orig_orientation = im_seg.orientation
                im_seg = im_seg.change_orientation('RPI')
                im_seg.data[:, :, missing_slices] = 0
                im_seg.change_orientation(orig_orientation)
                im_seg.save(fname_segmentation, verbose=0)  # suppress overwrite in tmpdir

    if normalize_pam50 and not perslice:
        parser.error("Option '-normalize-PAM50' requires option '-perslice 1'.")
    if distance_pmj is not None and fname_pmj is None:
        parser.error("Option '-pmj-distance' requires option '-pmj'.")
    if fname_pmj is not None and distance_pmj is None and not perslice:
        parser.error("Option '-pmj' requires option '-pmj-distance' or '-perslice 1'.")

    # update fields
    metrics_agg = {}

    metrics, fit_results = compute_shape(fname_segmentation,
                                         fname_image,
                                         angle_correction=angle_correction,
                                         centerline_path=fname_centerline,
                                         param_centerline=param_centerline,
                                         verbose=verbose,
                                         remove_temp_files=arguments.r)
    if normalize_pam50:
        fname_vert_level_PAM50 = os.path.join(__data_dir__, 'PAM50', 'template', 'PAM50_levels.nii.gz')
        metrics_native_space = metrics  # Save metrics in native space to use them for HOG angle QC
        metrics_PAM50_space = interpolate_metrics(metrics, fname_vert_level_PAM50, fname_vert_level)
        if not levels:  # If no levels -vert were specified by user
            if verbose == 2:
                # Get all available vertebral levels from PAM50 template to only include slices from available levels in .csv file
                levels = Image(fname_vert_level_PAM50).getNonZeroValues()
            else:
                # Get all available vertebral levels to only include slices from available levels in .csv file
                levels = Image(fname_vert_level).getNonZeroValues()
        metrics = metrics_PAM50_space  # Set metrics to the metrics in PAM50 space to use instead
        fname_vert_level = fname_vert_level_PAM50  # Set vertebral levels to PAM50
    if fname_pmj is not None:
        im_ctl, mask, slices, centerline, length_from_pmj = get_slices_for_pmj_distance(fname_segmentation, fname_pmj,
                                                                                        distance_pmj, extent_pmj,
                                                                                        param_centerline=param_centerline, perslice=perslice,
                                                                                        verbose=verbose)

        # Save array of the centerline in a .csv file if verbose == 2
        if verbose == 2:
            fname_ctl_csv, _ = splitext(add_suffix(arguments.i, '_centerline_extrapolated'))
            np.savetxt(f'{fname_ctl_csv}.csv', centerline, delimiter=",")
    else:
        length_from_pmj = None
    # Aggregate metrics
    for key in sct_progress_bar(metrics, unit='iter', unit_scale=False, desc="Aggregating metrics", ncols=80):
        if key == 'length':
            # For computing cord length, slice-wise length needs to be summed across slices
            metrics_agg[key] = aggregate_per_slice_or_level(metrics[key], slices=slices,
                                                            levels=levels,
                                                            distance_pmj=distance_pmj, perslice=perslice,
                                                            perlevel=perlevel, fname_vert_level=fname_vert_level,
                                                            group_funcs=(('SUM', func_sum),), length_pmj=length_from_pmj)
        else:
            # For other metrics, we compute the average and standard deviation across slices
            metrics_agg[key] = aggregate_per_slice_or_level(metrics[key], slices=slices,
                                                            levels=levels,
                                                            distance_pmj=distance_pmj, perslice=perslice,
                                                            perlevel=perlevel, fname_vert_level=fname_vert_level,
                                                            group_funcs=group_funcs, length_pmj=length_from_pmj)
    metrics_agg_merged = merge_dict(metrics_agg)
    # Normalize CSA values (MEAN(area))
    if arguments.normalize is not None:
        data_subject = pd.DataFrame([arguments.normalize])
        path_model = os.path.join(__sct_dir__, 'data', 'csa_normalization_models',
                                  f'{"_".join(sorted(data_subject.columns))}.csv')
        if not os.path.isfile(path_model):
            parser.error('Invalid choice of predictors in -normalize. Please specify sex and brain-volume or sex, brain-volume and thalamus-volume.')
        # Get normalization model
        # Models are generated with https://github.com/sct-pipeline/ukbiobank-spinalcord-csa/blob/master/pipeline_ukbiobank/cli/compute_stats.py
        # TODO update link with release tag.
        data_predictors = pd.read_csv(path_model, index_col=0)
        # Add interaction term
        data_subject['inter-BV_sex'] = data_subject['brain-volume']*data_subject['sex']
        for line in metrics_agg_merged.values():
            line['MEAN(area)'] = normalize_csa(line['MEAN(area)'], data_predictors, data_subject)

    save_as_csv(metrics_agg_merged, file_out, fname_in=fname_segmentation, append=append)

    # QC report (only for PMJ-based CSA)
    # TODO: refactor this with qc2. Replace arguments.qc_image with arguments.i
    if path_qc is not None:
        if fname_pmj is not None:
            if arguments.qc_image is not None:
                fname_mask_out = add_suffix(arguments.i, '_mask_csa')
                fname_ctl = add_suffix(arguments.i, '_centerline_extrapolated')
                fname_ctl_smooth = add_suffix(fname_ctl, '_smooth')
                if verbose != 2:
                    from spinalcordtoolbox.utils.fs import tmp_create
                    path_tmp = tmp_create(basename="pmj-qc")
                    fname_mask_out = os.path.join(path_tmp, fname_mask_out)
                    fname_ctl = os.path.join(path_tmp, fname_ctl)
                    fname_ctl_smooth = os.path.join(path_tmp, fname_ctl_smooth)
                # Save mask
                mask.save(fname_mask_out)
                # Save extrapolated centerline
                im_ctl.save(fname_ctl)
                # Generated centerline smoothed in RL direction for visualization (and QC report)
                sct_maths.main(['-i', fname_ctl, '-smooth', '10,1,1', '-o', fname_ctl_smooth, '-v', '0'])

                generate_qc(fname_in1=get_absolute_path(arguments.qc_image),
                            # NB: For this QC figure, the centerline has to be first in the list in order for the centerline
                            # to be properly layered underneath the PMJ + mask. However, Sagittal.get_center_spit
                            # is called during QC, and it uses `fname_seg[-1]` to center the slices. `fname_mask_out`
                            # doesn't work for this, so we have to repeat `fname_ctl_smooth` at the end of the list.
                            fname_seg=[fname_ctl_smooth, fname_pmj, fname_mask_out, fname_ctl_smooth],
                            args=argv,
                            path_qc=path_qc,
                            dataset=qc_dataset,
                            subject=qc_subject,
                            contrast=qc_contrast,
                            process='sct_process_segmentation')
            else:
                parser.error('-qc-image is required to display QC report.')
        else:
            logger.warning('QC report only available for -pmj or -anat flags. QC report not generated.')

    # Create QC report for the HOG angle
    if arguments.qc is not None:
        if fname_image is not None:
            qc2.sct_process_segmentation(
                fname_input=fname_image,
                fname_seg=fname_segmentation,
                metrics=metrics_native_space if normalize_pam50 else metrics,
                argv=argv,
                path_qc=arguments.qc,
                dataset=arguments.qc_dataset,
                subject=arguments.qc_subject,
                contrast=arguments.qc_contrast,
                angle_qc=(verbose == 2),  # only output HOG angle plot if `-v 2` is passed
                angle_type='angle_hog',
            )

    if verbose:
        display_open(file_out)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
