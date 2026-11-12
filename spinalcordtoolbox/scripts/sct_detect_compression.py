#!/usr/bin/env python
#
# Predict compression probability in a spinal cord MRI image using spinal cord shape metrics.
# The script process axial slices at the level of intervertebral discs C3/C4 (value: 4) to C6/C7 (value: 7).
# In other words, the script does *not* process all axial slices of the spinal cord segmentation.
#
# The script requires spinal cord segmentation (used to compute the shape metrics) and disc labels.
#
# More details in: https://doi.org/10.21037/qims-21-782
#
# Copyright (c) 2024 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import os
import sys
import math
from typing import Sequence
import textwrap

import numpy as np

from spinalcordtoolbox.process_seg import compute_shape
from spinalcordtoolbox.aggregate_slicewise import aggregate_per_slice_or_level, func_wa, func_std, merge_dict
from spinalcordtoolbox.centerline.core import ParamCenterline
from spinalcordtoolbox.template import get_slices_from_vertebral_levels
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.fs import get_absolute_path, extract_fname
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar
from spinalcordtoolbox.utils.sys import init_sct, set_loglevel, printv, LazyLoader

pd = LazyLoader("pd", globals(), "pandas")


# Currently, the model works only for discs C3/C4 (4) to C6/C7 (7)
SUPPORTED_DISCS = {4: 'C3/C4', 5: 'C4/C5', 6: 'C5/C6', 7: 'C6/C7'}
# Compression cut-offs --> compression probability
#   >0.451 high probability of compression --> yes
#   0.345-0.451 moderate probability of compression --> possible
#   <0.345 low probability of compression --> no
CUT_OFF_HIGH = 0.451
CUT_OFF_MODERATE = 0.345


def get_parser():
    parser = SCTArgumentParser(
        description=textwrap.dedent(f"""
            Predict compression probability in a spinal cord MRI image using spinal cord shape metrics.
            The script process axial slices at the level of intervertebral discs C3/C4 (value: 4) to C6/C7 (value: 7).

            Compression categories are determined based on the following cut-offs:

              - 'yes'     : p > {CUT_OFF_HIGH}
              - 'possible': {CUT_OFF_MODERATE} <= p <= {CUT_OFF_HIGH}
              - 'no'      : p < {CUT_OFF_MODERATE}

            These cut-off values and compression categories were determined by ROC analysis and the Youden’s index
            (the sum of sensitivity and specificity). More info in the reference below.

            Reference:

              - Horáková M, Horák T, Valošek J, Rohan T, Koriťáková E, Dostál M, Kočica J, Skutil T, Keřkovský M, Kadaňka Z Jr, Bednařík P, Svátková A, Hluštík P, Bednařík J. Semi-automated detection of cervical spinal cord compression with the Spinal Cord Toolbox. Quant Imaging Med Surg 2022; 12:2261–2279.
                  https://doi.org/10.21037/qims-21-782
        """),  # noqa: E501 (line too long)
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-s',
        metavar=Metavar.file,
        help="Segmentation of the spinal cord, which will be used to compute the shape metrics. "
             "Example: `t2s_seg.nii.gz`."
    )
    mandatory.add_argument(
        '-discfile',
        metavar=Metavar.file,
        help=textwrap.dedent("""
            File with disc labels. Each label must be a single voxel. Only label values 4, 5, 6, and 7 (C3/C4 to C6/C7) are supported for now; all other labels will be ignored.
            Labels can be located either at the posterior edge of the intervertebral discs, or at the orthogonal projection of each disc onto the spinal cord.
            Such a label file can be manually created using: `sct_label_utils -i IMAGE_REF -create-viewer 4:7` or
            obtained automatically using the `sct_label_vertebrae` function (the file with the `labeled_discs.nii.gz` suffix).
            Example: `t2s_discs.nii.gz`.
        """),  # noqa: E501 (line too long)
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-num-of-slices',
        metavar=Metavar.int,
        type=int,
        default=0,
        help=textwrap.dedent("""
            Number of additional axial slices above and below the intervertebral disc to process.
            Specifying `-num-of-slices 1` will process 3 slices total per disc (+/- 1), `2` will process 5 slices total (+/- 2), and so on.
        """),  # noqa: E501 (line too long)
    )
    optional.add_argument(
        '-angle-corr',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=1,
        help=textwrap.dedent("""
            Angle correction for computing morphometric measures. When angle correction is used, the cord within the slice is stretched/expanded by a factor corresponding to the cosine of the angle between the centerline and the axial plane. If the cord is already quasi-orthogonal to the slab, you can set `-angle-corr` to 0.
        """)  # noqa: E501 (line too long)
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help='Output CSV file name. If not provided, the suffix `compression_results` is added to the file name '
             'provided by the flag `-s`.'
    )

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


def predict_compression_probability(cr, csa, solidity, torsion, disc):
    """
    Predict compression probability using the spinal cord shape metrics for currently processed axial slice.
    See Equation 4 and Table 3 for details: https://doi.org/10.21037/qims-21-782
    :param cr: compression ratio (CR) in %
    :param csa: cross-sectional area (CSA) in mm2
    :param solidity: solidity in %
    :param torsion: torsion in degrees
    :param disc: intervertebral disc level (e.g., value=4 -> disc C3/C4)
    :return: probability
    """
    coefficients = {
        "constant": 57.501,
        "cr": -0.273,
        "csa": -0.102,
        "solidity": -0.408,
        "torsion": 2.168,
        "level_c6c7": -2.729,
    }
    model = (
        coefficients["constant"]
        + coefficients["cr"] * cr
        + coefficients["csa"] * csa
        + coefficients["solidity"] * solidity
        + (coefficients["torsion"] * torsion if torsion is not None else 0)     # edge-case when torsion is None
        + (coefficients["level_c6c7"] if disc == 7 else 0)
    )
    return math.exp(model) / (1 + math.exp(model))


def compute_shape_metrics(fname_seg, fname_disc, angle_correction, verbose):
    """
    Compute shape metrics per slice.
    Then, compute compression ratio and torsion.
    :param fname_seg: file with spinal cord segmentation
    :param fname_disc: file with disc labels
    :param angle_correction: bool: angle correction for computing morphometric measures
    :param verbose: verbosity
    :return: dictionary with aggregated metrics
    """
    if not np.any(Image(fname_seg).data):
        raise ValueError("Spinal cord segmentation file is empty.")

    metrics, fit_results = compute_shape(fname_seg,
                                         angle_correction=angle_correction,
                                         param_centerline=ParamCenterline(),
                                         verbose=verbose)

    # Compute the average and standard deviation across axial slices
    group_funcs = (('MEAN', func_wa), ('STD', func_std))
    metrics_agg = {}
    for key in ['area', 'diameter_AP', 'diameter_RL', 'solidity', 'orientation']:
        # Note: we do not need to calculate all the metrics, we need just:
        #   - area (will be CSA)
        #   - diameter_AP and diameter_RL (used to calculate compression ratio)
        #   - solidity
        #   - orientation (used to calculate torsion)
        # Note: we have to calculate metrics across all slices (perslice) to be able to compute orientation
        metrics_agg[key] = aggregate_per_slice_or_level(metrics[key],
                                                        perslice=True,
                                                        perlevel=False, fname_vert_level=fname_disc,
                                                        group_funcs=group_funcs
                                                        )
    metrics_agg_merged = merge_dict(metrics_agg)
    # Compute compression ratio
    metrics_agg_merged = compute_compression_ratio(metrics_agg_merged)
    # Compute torsion
    metrics_agg_merged = compute_torsion(metrics_agg_merged, verbose)

    return metrics_agg_merged


def compute_compression_ratio(metrics_agg_merged):
    """
    Compute compression ratio (CR) as 'diameter_AP' / 'diameter_RL'
    :param metrics_agg_merged: dictionary with aggregated metrics
    :return: dictionary with compression ratio added
    """
    for metrics in metrics_agg_merged.values():  # Loop across slices
        ap = metrics['MEAN(diameter_AP)']
        rl = metrics['MEAN(diameter_RL)']
        metrics['MEAN(compression_ratio)'] = None if ap is None or rl is None else ap / rl

    return metrics_agg_merged


def compute_torsion(metrics_agg_merged, verbose):
    """
    Compute torsion as the average of absolute differences in orientation taking into account 3 slices above and
    3 slices below. For details, see Equation 3 in https://doi.org/10.21037/qims-21-782.
    For a comparison between torsion computed using 1, 2, and 3 slices, see https://cdn.amegroups.cn/static/public/QIMS-21-782-supplementary.pdf
    NOTE: As the torsion is computed from 3 slices above and below, it cannot be computed for the 3 first and last
    3 slices --> `torsion = None` is used for the 3 first and 3 last slices.
    :param metrics_agg_merged: dictionary with aggregated metrics
    :param verbose: verbosity
    :return: dictionary with torsion added
    """
    slices = list(metrics_agg_merged.keys())[3:-3]

    for key in metrics_agg_merged.keys():  # Loop across slices
        metrics_agg_merged[key]['Torsion'] = None
        if key in slices:
            try:
                # Note: the key is a tuple (e.g. `(1,)`), thus key[0] converts to int, and `(s,)` converts back
                # Here we use pairs of adjacent slices, with each slice pair offset from the initial slice
                slice_pairs = [(key[0] + offset, key[0] + offset + 1) for offset in range(-3, 3)]
                metrics_agg_merged[key]['Torsion'] = 1 / 6 * sum([abs(metrics_agg_merged[(s1,)]['MEAN(orientation)'] -
                                                                      metrics_agg_merged[(s2,)]['MEAN(orientation)'])
                                                                  for s1, s2 in slice_pairs])
            except Exception as e:
                # TODO: the warning below is raise mainly in cases of no SC segmentation, for example above the C1 level
                #  --> consider processing only slices with SC segmentation
                printv(f"WARNING: Failed to compute torsion for slice {key[0]}. Use `-v 2` flag to see details",
                       verbose, type='warning')
                if verbose == 2:
                    printv(f"\t{e}", verbose)

    return metrics_agg_merged


def process_compression(metrics_agg_merged, disc_slices, num_of_slices):
    """
    Process shape metrics for axial slices at the level of each disc to predict the compression probability.
    :param metrics_agg_merged: dictionary with aggregated metrics
    :param disc_slices: dictionary with disc labels as keys and corresponding axial slice numbers as values
    :param num_of_slices: number of slices around (above and below) the intervertebral disc to process
    :return: dataframe with compression probability
    """
    compression_df = pd.DataFrame()
    # Loop across discs
    for disc, disc_slice in disc_slices.items():
        # Add one slice above and below to process multiple slices around the disc to compensate for potential
        # disc label shift in the superior-inferior (S-I) axis
        for slc in range(disc_slice - num_of_slices, disc_slice + num_of_slices + 1):
            if disc in SUPPORTED_DISCS.keys():
                # Note: [slice,] is used to convert int to tuple
                cr = metrics_agg_merged[slc,]['MEAN(compression_ratio)'] * 100    # to convert to %
                csa = metrics_agg_merged[slc,]['MEAN(area)']
                solidity = metrics_agg_merged[slc,]['MEAN(solidity)'] * 100     # to convert to %
                torsion = metrics_agg_merged[slc,]['Torsion']
                # Compute compression probability
                probability = predict_compression_probability(cr, csa, solidity, torsion, disc)
                compression_category = 'yes' if probability > CUT_OFF_HIGH else \
                    'possible' if probability >= CUT_OFF_MODERATE else 'no'
                compression_df = pd.concat([compression_df,
                                            pd.DataFrame([{'Disc': SUPPORTED_DISCS[disc], 'Axial slice #': slc,
                                                           'Compression probability': probability,
                                                           'Compression probability category': compression_category,
                                                           'Compression ratio (%)': cr, 'CSA (mm2)': csa,
                                                           'Solidity (%)': solidity, 'Torsion (degrees)': torsion}])],
                                           ignore_index=True)

    return compression_df


def get_disc_slices(fname_disc):
    """
    Get axial slice numbers corresponding to the intervertebral discs.
    :param fname_disc: file with disc labels
    :return: dictionary with disc labels as keys and corresponding axial slice numbers as values
    """
    im_disc = Image(fname_disc).change_orientation('RPI')
    discs = [int(disc) for disc in np.trim_zeros(np.unique(im_disc.data))]  # get unique disc labels (e.g., 3, 4, 5 etc)

    if not discs:
        raise ValueError("Disc file is empty.")
    # Check if discs contains any supported discs, if not, raise an error
    if not any(disc in SUPPORTED_DISCS.keys() for disc in discs):
        raise ValueError(f"No supported disc labels found in the disc file. Supported discs: {SUPPORTED_DISCS.keys()}")

    # Find axial slices corresponding to the discs
    # Note: [0] is used because get_slices_from_vertebral_levels returns list
    disc_slices = {disc: get_slices_from_vertebral_levels(im_disc, disc)[0] for disc in discs}

    return disc_slices


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    # Parse arguments
    fname_seg = get_absolute_path(arguments.s)
    fname_disc = get_absolute_path(arguments.discfile)
    angle_correction = bool(arguments.angle_corr)

    # Get discs corresponding to the compression
    disc_slices = get_disc_slices(fname_disc)

    # Compute shape metrics per slice
    metrics_agg_merged = compute_shape_metrics(fname_seg, fname_disc, angle_correction, verbose)

    # Process axial slices for each disc to compute compression probability
    compression_df = process_compression(metrics_agg_merged, disc_slices, arguments.num_of_slices)

    # Save the results to a CSV file
    if arguments.o is not None:
        fname_out = arguments.o
    else:
        path, file_name, ext = extract_fname(get_absolute_path(fname_seg))
        fname_out = os.path.join(path, f'{file_name}_compression_results.csv')
    compression_df.to_csv(fname_out, index=False)
    printv(f"Results saved to: {fname_out}", verbose)

    for index, row in compression_df.iterrows():
        printv(f"Disc {(row['Disc'])}, axial slice {int(row['Axial slice #'])} --> "
               f"compression probability: {row['Compression probability category']} "
               f"({row['Compression probability'] * 100:.2f}%)")


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
