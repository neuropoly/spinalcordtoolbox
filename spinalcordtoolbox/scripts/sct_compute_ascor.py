#!/usr/bin/env python
#
# Compute adapted Spinal Cord Occupation Ratio (aSCOR) from spinal cord and spinal canal masks.
# morphometrics.
#
# Copyright (c) 2025 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
import os
import textwrap
import logging
import itertools
import numpy as np
from typing import Sequence
from spinalcordtoolbox.utils.fs import get_absolute_path, TempFolder
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.shell import display_open, Metavar
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import LazyLoader
from spinalcordtoolbox.scripts import sct_process_segmentation

pd = LazyLoader("pd", globals(), "pandas")

logger = logging.getLogger(__name__)

INDEX_COLUMNS = ['Filename_sc', 'Filename_canal', 'Slice (I->S)', 'VertLevel', 'DistancePMJ', 'aSCOR']


# PARSER
# ==========================================================================================


def get_parser():
    """
    :return: Returns the parser with the command line documentation contained in it.
    """
    # Because `sct_compute_ascor` is a light wrapper for `sct_process_segmentation`, we want to re-use most of the arguments.
    parser = sct_process_segmentation.get_parser(ascor=True)
    parser.description = textwrap.dedent("""
        Compute adapted Spinal Cord Occupation Ratio (aSCOR) from spinal cord and spinal canal segmentation masks.
        The aSCOR is defined as the ratio between the cross-sectional area of the spinal cord and the cross-sectional area of the spinal canal
    """)  # noqa: E501 (line too long)

    # In `sct_process_segmentation`, we normally use the `-i` argument. But because we set `ascor=True` when calling
    # `get_parser()`, `-i` won't be created (to allow us to add `-i-SC` and `-i-canal` instead).
    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i-SC',
        metavar=Metavar.file,
        help=textwrap.dedent("""
            Spinal cord segmentation mask to compute morphometrics from. Example: `sub-001_T2w_seg.nii.gz`
            Spinal cord segmentation can be obtained with `sct_deepseg spinalcord`.
        """),  # noqa: E501 (line too long)
    )
    mandatory.add_argument(
        '-i-canal',
        metavar=Metavar.file,
        help=textwrap.dedent("""
            Spinal canal segmentation mask to compute morphometrics from. Example: `sub-001_T2w_canal_seg.nii.gz`
            Canal segmentation can be obtained with `sct_deepseg canal`.
        """),  # noqa: E501 (line too long)
    )

    return parser


def compute_ascor(csa_sc, csa_canal):
    """
    Computes the aSCOR (spinal cord area to canal area ratio) for each row in the provided CSV files and saves the results to a CSV file.

    :param str csa_sc: Path to the CSV file containing spinal cord area measurements.
    :param str csa_canal: Path to the CSV file containing spinal canal area measurements.
    :return pandas.DataFrame: DataFrame containing the aSCOR results."""
    df_sc = pd.read_csv(csa_sc)
    df_canal = pd.read_csv(csa_canal)
    df_merged = pd.merge(df_sc, df_canal, on=['Slice (I->S)', 'VertLevel', 'DistancePMJ'], suffixes=('_sc', '_canal'))
    df_merged['aSCOR'] = df_merged['MEAN(area)_sc'].div(df_merged['MEAN(area)_canal'], fill_value=0)
    df_merged['aSCOR'].replace(np.inf, np.nan)  # output nan instead of inf for /0
    printv(f"Computed aSCOR for {len(df_merged)} rows.", 1, 'normal')
    return df_merged[INDEX_COLUMNS]


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)    # values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG]

    # Load input and output filenames (while passing the remaining args to `sct_process_segmentation`)
    fname_sc_seg = get_absolute_path(arguments.i_SC)
    fname_canal_seg = get_absolute_path(arguments.i_canal)
    fname_out = arguments.o
    process_seg_argv = list(itertools.chain.from_iterable([
        [arg, value] for arg, value in zip(argv[::2], argv[1::2])
        if arg not in ['-i-SC', '-i-canal', '-o']
    ]))
    # Validate the inputs
    img_sc = Image(fname_sc_seg).change_orientation('RPI')
    img_canal = Image(fname_canal_seg).change_orientation('RPI')
    if not img_sc.data.shape == img_canal.data.shape:
        raise ValueError(f"Shape mismatch between spinal cord segmentation [{img_sc.data.shape}],"
                         f" and canal segmentation [{img_canal.data.shape}]). "
                         f"Please verify that your spinal cord and canal segmentations were done in the same space.")

    # If `-centerline` has not been passed, specify it ourselves (to ensure that a consistent centerline is used for both segmentations)
    if '-centerline' not in process_seg_argv:
        printv("No `-centerline` provided. A centerline will be computed from `-i-SC` and used for both SC and canal.", verbose, 'warning')
        process_seg_argv.extend(['-centerline', fname_sc_seg])

    # Override the default value of `-centerline-exclude-missing` to be `1` instead of `0`
    # Doing this limits the aSCOR computation to only the slices where both SC and canal segmentations are present.
    # This avoids a potential exception caused by the centerline (i.e. `-i-SC`) not fully covering the canal segmentation.
    if '-centerline-exclude-missing' not in process_seg_argv:
        process_seg_argv.extend(['-centerline-exclude-missing', '1'])

    # Run sct_process_segmentation twice: 1) SC seg 2) canal seg
    temp_folder = TempFolder(basename="process-segmentation", verbose=verbose)
    path_tmp = temp_folder.get_path()
    inputs = {'spinal cord': fname_sc_seg,                     'spinal canal': fname_canal_seg}
    output = {'spinal cord': os.path.join(path_tmp, "sc.csv"), 'spinal canal': os.path.join(path_tmp, "canal.csv")}
    for step_name in inputs.keys():
        printv(f"\nRunning sct_process_segmentation on {step_name} segmentation...", verbose, 'normal')
        process_seg_argv_full = ['-i', inputs[step_name], '-o', output[step_name], '-v', '0'] + process_seg_argv
        if verbose >= 2:
            printv(f'sct_process_segmentation {" ".join(process_seg_argv_full)}', verbose, 'code')
        sct_process_segmentation.main(process_seg_argv_full)

    # Compute aSCOR
    printv("\nComputing aSCOR...", verbose, 'normal')
    df_ascor = compute_ascor(csa_sc=output['spinal cord'], csa_canal=output['spinal canal'])
    # Save aSCOR to csv
    if arguments.append and os.path.exists(fname_out):
        dataframe_old = pd.read_csv(fname_out, index_col=INDEX_COLUMNS)
        df_ascor = pd.concat([dataframe_old.reset_index(), df_ascor], ignore_index=True)
    df_ascor.to_csv(fname_out, index=False)

    # Clean up temp
    if arguments.r and temp_folder is not None:
        logger.info("\nRemove temporary files...")
        temp_folder.cleanup()

    # Print final log message for user to open the output file
    display_open(os.path.abspath(fname_out))


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
