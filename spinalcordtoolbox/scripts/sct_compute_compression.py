#!/usr/bin/env python
#
# Compute maximum spinal cord compression (MSCC) or maximum canal compromise (MCC) using AP diameter or other
# morphometrics.
#
# Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

# TODO: maybe create an API or move some functions
import sys
import os
import textwrap
import numpy as np
import logging
from typing import Sequence
from spinalcordtoolbox.utils.fs import extract_fname, get_absolute_path
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import get_centerline, ParamCenterline
from spinalcordtoolbox.types import Centerline
from spinalcordtoolbox.utils.sys import __data_dir__, LazyLoader
from spinalcordtoolbox.scripts import sct_process_segmentation

pd = LazyLoader("pd", globals(), "pandas")

logger = logging.getLogger(__name__)


NEAR_ZERO_THRESHOLD = 1e-6
INDEX_COLUMNS = ['filename', 'compression_level', 'Slice (I->S)']


# PARSER
# ==========================================================================================
def get_parser():
    parser = SCTArgumentParser(
        description=textwrap.dedent("""
            Compute normalized morphometric metrics to assess:

              - spinal cord compression using MSCC (maximum spinal cord compression)
              - spinal canal stenosis using MCC (maximum canal compromise)

            Metrics are normalized using the non-compressed levels above and below the compression site (or injury site with lesion) using the following equation:

              ```
              ratio = (1 - mi/((ma+mb)/2))
              ```

            Where mi: metric at the compression level, ma: metric above the compression level, mb: metric below the compression level.

            Additionally, if the `-normalize-hc` flag is used, metrics are normalized using a database built from healthy control subjects. This database uses the PAM50 template as an anatomical reference system.

            References:
              - Sandrine Bédard, Jan Valošek, Maryam Seif, Armin Curt, Simon Schading, Nikolai Pfender, Patrick Freund, Markus Hupp, Julien Cohen-Adad. Normalizing Spinal Cord Compression Morphometric Measures: Application in Degenerative Cervical Myelopathy. medRxiv 2024.03.13.24304177
                https://doi.org/10.1101/2024.03.13.24304177
              - Miyanji F, Furlan JC, Aarabi B, Arnold PM, Fehlings MG. Acute cervical traumatic spinal cord injury: MR imaging findings correlated with neurologic outcome--prospective study with 100 consecutive patients. Radiology 2007;243[3]:820-827.
                https://doi.org/10.1148/radiol.2433060583
              - `-normalize-hc` flag:
                Valošek J, Bédard S, Keřkovský M, Rohan T, Cohen-Adad J. A database of the healthy human spinal cord morphometry in the PAM50 template space. Imaging Neuroscience 2024; 2 1–15.
                https://doi.org/10.1162/imag_a_00075
        """),  # noqa: E501 (line too long)
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        help=textwrap.dedent("""
            Spinal cord or spinal canal segmentation mask to compute morphometrics from. If spinal cord segmentation is provided, MSCC is computed. If spinal canal segmentation (spinal cord + CSF) is provided, MCC is computed. Example: `sub-001_T2w_seg.nii.gz`

            Note: If no normalization is wanted (i.e., if the `-normalize-hc` flag is not specified), metric ratio will take the average along the segmentation centerline.
        """),  # noqa: E501 (line too long)
    )
    mandatory.add_argument(
        '-l',
        metavar=Metavar.file,
        help=textwrap.dedent("""
            A NIfTI file that includes either i) compression labels or ii) lesion mask.
              i) compression labels = labels at the compression sites. Each compression site is denoted by a single voxel of value `1`. Example: `sub-001_T2w_compression_labels.nii.gz`. Use the `-mode compression` flag when providing this input.
              ii) lesion mask = binary mask of the lesion. Currently only a single lesion is supported. Example: `sub-001_T2w_lesion.nii.gz`. Use the `-mode lesion` flag when providing this input.

            Note: The '-i' and '-l' files must be in the same voxel coordinate system and must match the dimensions between each other.
        """),  # noqa: E501 (line too long)
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-vertfile',
        metavar=Metavar.file,
        help=textwrap.dedent("""
            Vertebral labeling file. Example: `sub-001_T2w_seg_labeled.nii.gz`

            Note: The '-i' and '-vertfile' files must be in the same voxel coordinate system and must match the dimensions between each other.
        """),
    )
    optional.add_argument(
        '-mode',
        type=str,
        choices=['compression', 'lesion'],
        default='compression',
        help=textwrap.dedent("""
            Choose between:
              - compression: use compression labels (provided by the '-l' arg).
              - lesion: use lesion mask (provided by the '-l' arg). In this case, the level of maximum injury is automatically determined as the axial slice within the lesion mask that has the minimum spinal cord AP diameter.
        """),   # noqa: E501 (line too long)
    )
    optional.add_argument(
        '-extent',
        type=float,
        metavar=Metavar.float,
        default=20.0,
        help='Extent (in mm) to average metrics of healthy levels in superior-inferior direction.'
    )
    optional.add_argument(
        '-distance',
        type=float,
        metavar=Metavar.float,
        default=10.0,
        help='Distance (in mm) in the superior-inferior direction from the compression to average healthy slices.'
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help='Output CSV file name. If not provided, the suffix `_compression_metrics` is added to the file name '
             'provided by the flag `-i`.'
    )

    normalize_hc = parser.add_argument_group(
        "NORMALIZATION ARGUMENTS"
    )
    # TODO: swap this over to a flag, rather than numeric
    normalize_hc.add_argument(
        '-normalize-hc',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        help=textwrap.dedent(
            """
            Set to 1 to normalize the metrics using a database of healthy controls. Default: 0.

            Note: This flag should not be set to 1 when computing the MCC (i.e. using spinal canal segmentation),
            and should only be used when computing the MSCC (i.e. using spinal cord segmentation).
            """
        ),
    )
    normalize_hc.add_argument(
        '-metric',
        help='Metric to normalize.',
        default='diameter_AP',
        choices=['diameter_AP', 'area', 'diameter_RL', 'eccentricity', 'solidity'],
    )
    normalize_hc.add_argument(
        '-sex',
        type=str,
        choices=['F', 'M'],
        help='Sex of healthy subject to use for the normalization. By default, both sexes are used.'
    )
    normalize_hc.add_argument(
        '-age',
        type=int,
        nargs=2,
        metavar="[0 100]",
        help='Age range of healthy subjects to use for the normalization. Example: `-age 60 80`. '
             'By default, all ages are considered.'
    )

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


# Functions for Step 1 (Load subject input files and get csv files)
# ==========================================================================================
def get_slice_thickness(img):
    """
    Get slice thickness from the input image.
    :param img: Image: source image
    :return float: slice thickness in mm.
    """
    return img.dim[6]


def get_compressed_slice(img, df_metrics, mode):
    """
    Get all the compression labels (voxels of value: '1') that are contained in the input image.
    :param img: Image: RPI-oriented source image
    :param df_metrics: pandas.DataFrame: dataframe with spinal cord shape metrics (output of sct_process_segmentation)
    :param mode: str: either 'compression' or 'lesion'
    :return list: list of slices number
    """
    # Get all coordinates
    coordinates = img.getNonZeroCoordinates(sorting='z')
    # Check it coordinates is empty
    if not coordinates:
        raise ValueError('No compression labels found.')

    # Get only unique axial slice numbers (z coordinate in RPI orientation)
    # 'set' is used to get unique z coordinates (as lesion can have multiple pixels within a single slice)
    slices_compressed = list(set([int(coordinate.z) for coordinate in coordinates]))

    # For compression labels, return the slices with compression labels
    if mode == 'compression':
        logger.debug(f'Compression labels coordinates: {coordinates}')
        return slices_compressed

    # For lesion mask, return the slice with the maximum injury
    if mode == 'lesion':
        # Identify the level of maximum injury, defined as the axial slice within the lesion (`slices_compressed`) that
        # has the minimum 'MEAN(diameter_AP)' value in `df_metrics`.
        # This metric is computed from either the spinal cord or spinal canal segmentation (`arguments.i`).
        df_filtered = df_metrics[df_metrics['Slice (I->S)'].isin(slices_compressed)]
        # NOTE: we use 'MEAN(diameter_AP)' for all metrics here as this definition was used in the original publication:
        #  https://doi.org/10.1097/00007632-199903150-00023
        min_idx = df_filtered['MEAN(diameter_AP)'].idxmin()
        slice_num = df_filtered.loc[min_idx, 'Slice (I->S)']  # this might not be necessary as the index is already the slice
        return [slice_num]

    raise ValueError(f"Invalid mode '{mode}'. Expected 'compression' or 'lesion'.")


def get_verterbral_level_from_slice(slices, df_metrics):
    """
    From slices, gets the corresponding vertebral level and creates a dict fo level and corresponding slice(s).
    :param slices: list: list of slices number.
    :param df_metrics: pandas.DataFrame: dataframe of metrics (output of sct_process_segmentation).
    :return level_slice_dict: dict:
    """
    idx = df_metrics['Slice (I->S)'].isin(slices).tolist()
    df_level_slice_compression = df_metrics.loc[idx, ['VertLevel', 'Slice (I->S)']]
    if df_level_slice_compression.empty:
        raise ValueError(f"Slice {slices} doesn't have a computed metric")
    # Check if level_compression['VertLevel'] is nan
    for _, row in df_level_slice_compression.iterrows():
        if np.isnan(row['VertLevel']):
            raise ValueError(f"Slice {int(row['Slice (I->S)'])} doesn't have computed vertebral level. "
                             f"Check vertebral labeling file.")
    level_slice_dict = {}
    # TODO adjust for multiple slices for one compresssion (that can have multiple levels too)
    # slices_combined = []
    # for slice in slices:
    #    slices_same_compression = [slice_1 for slice_1 in slices if np.abs((slice - slice_1)) == 1]
    #    if slices_same_compression:
    #        if slices_same_compression[0] < slices_same_compression[1]:
    #            slices_same_compression.append(slice)
    #            slices_combined.append(slices_same_compression)
    #    else:
    #        slices_combined.append(slice)
    for idx, _ in enumerate(slices):
        level_slice_dict[idx] = {}
    for idx, slice in enumerate(slices):
        level = df_level_slice_compression.loc[df_level_slice_compression['Slice (I->S)'] == slice, 'VertLevel'].to_list()[0]
        level_slice_dict[idx][level] = [slice]
    return level_slice_dict


# Functions for Step 2 (Get normalization metrics and slices)
# ==========================================================================================
def select_HC(fname_participants, sex=None, age=None):
    """
    Selects healthy controls to use for normalization based on sex and age range specified by the user.
    :param fname_participants: Filename of participants.tsv
    :param sex: Either F for female or M for male.
    :param age: list: List of age range to select subjects.
    :return list_to_include: list: List of participant_id
    """
    # Load participants information
    data = pd.read_csv(fname_participants, sep="\t")
    # Initialize lists
    list_sub_sex = []
    list_sub_age = []
    # select subject with same sex
    if sex:
        list_sub_sex = data.loc[data['sex'] == sex, 'participant_id'].to_list()
        if not age:
            list_to_include = list_sub_sex
    # select subjects within age range
    if age:
        list_sub_age = data.loc[data['age'].between(age[0], age[1]), 'participant_id'].to_list()
        if not sex:
            list_to_include = list_sub_age
    if age and sex:
        list_to_include = set(list_sub_age).intersection(list_sub_sex)
    printv(f'{len(list_to_include)} healthy controls are used for normalization')
    return list(list_to_include)


def average_hc(ref_folder, metric, list_HC):
    """
    Gets metrics of healthy controls in PAM50 anatomical dimensions and averages across subjects.
    :param ref_folder: path to folder where .csv fields of healthy controls are.
    :param metric: str: metric to perform normalization
    :param list_HC: list: List of healthy controls to include
    :return df:
    """
    # Create empty dict to put dataframe of each healthy control
    d = {}
    # Iterator to count number of healthy subjects
    i = 0
    # Loop through .csv files of healthy controls
    for file in os.listdir(ref_folder):
        if 'PAM50.csv' in file:
            subject = os.path.basename(file).split('_')[0]
            if list_HC:
                # Check if subject is in list to include
                if subject in list_HC:
                    d[file] = pd.read_csv(os.path.join(ref_folder, file)).astype({metric: float})
                    i = i+1
            else:
                d[file] = pd.read_csv(os.path.join(ref_folder, file)).astype({metric: float})
                i = i+1
    first_key = next(iter(d))
    # Create an empty dataframe with same columns
    df = pd.DataFrame(columns=d[first_key].columns)
    df['VertLevel'] = d[first_key]['VertLevel']
    df['Slice (I->S)'] = d[first_key]['Slice (I->S)']
    # Loop through all HC
    for key, values in d.items():
        for column in d[key].columns:
            if 'MEAN' in column:
                if df[column].isnull().values.all():
                    df[column] = d[key][column]
                else:
                    # Sum all columns that have MEAN key
                    df[column] = df[column] + d[key][column].tolist()
    # Divide by number of HC
    for column in df.columns:
        if 'MEAN' in column:
            df[column] = df[column]/i
    return df


def get_slices_in_PAM50(compressed_level_dict, df_metrics, df_metrics_PAM50):
    """
    Get corresponding slice of compression in PAM50 space.
    :param compressed_level_dict: dict: Dictionary of levels and corresponding slice(s).
    :param df_metrics: pandas.DataFrame: Metrics output of sct_process_segmentation.
    :param df_metrics_PAM50: pandas.DataFrame: Metrics output of sct_process_segmentation in PAM50 anatomical dimensions.
    :return compression_level_dict_PAM50:
    """
    compression_level_dict_PAM50 = {}
    # Drop empty columns
    df_metrics_PAM50 = df_metrics_PAM50.drop(columns=['SUM(length)', 'DistancePMJ'])
    # Drop empty rows so they are not included for interpolation
    df_metrics_PAM50 = df_metrics_PAM50.dropna(axis=0)
    # Loop across slices and levels with compression
    for i, info in compressed_level_dict.items():
        compression_level_dict_PAM50[i] = {}
        for level, slices in info.items():
            # Number of slices in native image
            nb_slices_level = len(df_metrics.loc[df_metrics['VertLevel'] == level, 'VertLevel'].to_list())
            # Number of slices in PAM50
            nb_slices_PAM50 = len(df_metrics_PAM50.loc[df_metrics_PAM50['VertLevel'] == level, 'VertLevel'].to_list())
            # Do interpolation from native space to PAM50
            x_PAM50 = np.arange(0, nb_slices_PAM50, 1)
            x = np.linspace(0, nb_slices_PAM50 - 1, nb_slices_level)
            new_slices_coord = np.interp(x_PAM50, x,
                                         df_metrics.loc[df_metrics['VertLevel'] == level, 'Slice (I->S)'].to_list())
            # find nearest index
            slices_PAM50 = np.array([])
            for slice in slices:
                # get index corresponding to the min value
                idx = np.argwhere((np.round(new_slices_coord) - slice) == 0).T[0]  # Round to get all slices within ±1 arround the slice
                new_slice = [df_metrics_PAM50.loc[df_metrics_PAM50['VertLevel'] == level, 'Slice (I->S)'].to_list()[id] for id in idx]
                slices_PAM50 = np.append(slices_PAM50, new_slice, axis=0)
            slices_PAM50 = slices_PAM50.tolist()
            compression_level_dict_PAM50[i][level] = slices_PAM50
    return compression_level_dict_PAM50


def get_centerline_object(img_seg, verbose):
    """
    Get centerline object in physical dimensions
    :param img_seg: Image(): spinal cord segmentation
    """
    # Compute spinalcordtoolbox.types.Centerline class from get_centerline wit physical coordinates
    param_centerline = ParamCenterline(
                       algo_fitting='bspline',  # TODO add as default arg
                       smooth=30,  # TODO add as default arg
                       minmax=True)  # Check if we want min max or not

    _, arr_ctl_phys, arr_ctl_der_phys, _ = get_centerline(img_seg, param_centerline,
                                                          verbose=verbose, space="phys")
    ctl_seg = Centerline(*arr_ctl_phys, *arr_ctl_der_phys)
    return ctl_seg


def get_slices_upper_lower_level_from_centerline(centerline, distance, extent, z_compressions, z_ref):
    """
    Get slices to average for the level above the highest compression and below the lowest compression from the centerline.
    (If arg -normalize-hc is not used; meaning no normalization)
    : param centerline: Centerline(): Spinal cord centerline object
    : param distance: float: distance (mm) from the compression from where to average healthy slices.
    : param extent: float: extent (mm) to average healthy slices.
    : param z_compressions: list: list of slices that have a compression.
    : param z_ref: list: z index corresponding to the segmentation since the centerline only includes slices of the segmentation.
    : return
    """
    length = centerline.incremental_length_inverse
    # Get z index of lowest (min) and highest (max) compression
    z_compression_below = min(z_compressions)
    z_compression_above = max(z_compressions)
    # Get slices range for level below the lowest compression
    idx = np.argwhere(z_ref == z_compression_below)[0][0]
    length_0 = length[idx]
    zmax_below = z_ref[np.argmin(np.array([np.abs(i - length_0 + distance) for i in length]))]
    zmin_below = z_ref[np.argmin(np.array([np.abs(i - length_0 + distance + extent) for i in length]))]

    # Get slices range for level above the highest compression
    idx = np.argwhere(z_ref == z_compression_above)[0][0]
    length_0 = length[idx]
    zmin_above = z_ref[np.argmin(np.array([np.abs(i - length_0 - distance) for i in length]))]
    zmax_above = z_ref[np.argmin(np.array([np.abs(i - length_0 - distance - extent) for i in length]))]

    # If zmin is equal to zmax, the range is not available, use the other level above/below
    if zmin_above == zmax_above and zmin_below == zmax_below:
        raise ValueError(f"No slices of level above and below with a distance of {distance} mm "
                         f"and extent of {extent} mm. Please provide another distance and extent.")
    if zmin_above == zmax_above:
        logger.warning("Level above all compressions is not available. Only level below will be used for normalization "
                       "instead. If you want to use the level above, please change distance and extent. ")
        zmin_above = zmin_below
        zmax_above = zmax_below
    if zmin_below == zmax_below:
        logger.warning("Level below all compressions is not available. Only level above will be used for normalization "
                       "instead. If you want to use the level below, please change distance and extent. ")
        zmin_below = zmin_above
        zmax_below = zmax_above
    slices_above = np.arange(zmin_above, zmax_above, 1)
    slices_below = np.arange(zmin_below, zmax_below, 1)
    return slices_below, slices_above


def get_slices_upper_lower_level_from_PAM50(compression_level_dict_PAM50, df_metrics_PAM50, distance, extent, slice_thickness_PAM50):
    """
    Get slices to average the level above the highest compression and below the lowest compression from the PAM50.
    : param compression_level_dict_PAM50: dict: Dictionary of levels and corresponding slice(s) in the PAM50 space.
    : param df_metrics_PAM50: pandas.DataFrame: Metrics output of sct_process_segmentation in PAM50 anatomical dimensions.
    : param distance: float: distance (mm) from the compression from where to average healthy slices.
    : param extent: float: extent (mm) to average healthy slices.
    : param slice_thickness_PAM50: float: Slice thickness of the PAM50.
    : return slices_below:
    : return slices_above:
    """
    min_slices = []
    max_slices = []
    # Get level above and below index
    for idx, info in compression_level_dict_PAM50.items():
        for level, slices in info.items():
            min_slices.append(min(slices))
            max_slices.append(max(slices))
    level_below = np.argmin(min_slices)
    level_above = np.argmax(max_slices)
    # Get slices to average at distance across the chosen extent for the level above all compressions
    zmin_above = int(max(list(compression_level_dict_PAM50[level_above].values())[0]) + distance/slice_thickness_PAM50)
    zmax_above = int(max(list(compression_level_dict_PAM50[level_above].values())[0]) + distance/slice_thickness_PAM50 + extent/slice_thickness_PAM50)
    # Get slices to average at distance across the chosen extent for level below all compressions
    zmin_below = int(min(list(compression_level_dict_PAM50[level_below].values())[0]) - distance/slice_thickness_PAM50 - extent/slice_thickness_PAM50)
    zmax_below = int(min(list(compression_level_dict_PAM50[level_below].values())[0]) - distance/slice_thickness_PAM50)
    # Check if slices have available metrics
    df_metrics_PAM50_short = (df_metrics_PAM50.dropna(how='all', axis=1)).dropna()
    not_above = False
    not_below = False
    if zmax_below not in df_metrics_PAM50_short['Slice (I->S)'].to_list():
        logger.warning("Level below all compressions is not available. Only the level above will be used for "
                       "normalization instead. If you want to use the level below, please change distance and extent. ")
        zmax_below = zmax_above
        zmin_below = zmin_above
        not_below = True
    if zmin_above not in df_metrics_PAM50_short['Slice (I->S)'].to_list():
        logger.warning("Level above all compressions is not available. Only the level below will be used for "
                       "normalization instead. If you want to use the level above, please change distance and extent. ")
        zmax_above = zmax_below
        zmin_above = zmin_below
        not_above = True
    if not_above and not_below:
        raise ValueError("No metrics of level above and below all compressions are available with a distance of "
                         f"{distance} mm and extent of {extent} mm. Please provide another distance and extent.")
    # Take last available slice if extent is out of range
    if zmin_below not in df_metrics_PAM50_short['Slice (I->S)'].to_list():
        zmin_below = min(df_metrics_PAM50_short['Slice (I->S)'].to_list())
    if zmax_above not in df_metrics_PAM50_short['Slice (I->S)'].to_list():
        zmin_below = min(df_metrics_PAM50_short['Slice (I->S)'].to_list())
    slices_above = np.arange(zmin_above, zmax_above, 1)
    slices_below = np.arange(zmin_below, zmax_below, 1)
    return slices_below, slices_above


# Functions for Step 3 (Computing MSCC using spinal cord morphometrics.)
# ==========================================================================================
def average_metric(df, metric, z_range_above, z_range_below, slices_avg):
    """
    Average metric at compression level, at level above and below.
    :param df: pandas.DataFrame: Metrics output of sct_process_segmentation.
    :param metric: str: metric to perform normalization
    :param z_range_above: list: list of slices of level above compression.
    :param z_range_below: list: list of slices of level below compression.
    :param slices_avg: list: list of slices at the level of compression.
    :return: ma: float64: Metric above the compression
    :return: mb: float64: Metric below the compression
    :return: mi: float64: Metric at the compression level
    """
    # find index of slices to average
    idx_compression = df['Slice (I->S)'].isin(slices_avg).tolist()
    idx_above = df['Slice (I->S)'].isin(z_range_above).tolist()
    idx_below = df['Slice (I->S)'].isin(z_range_below).tolist()
    ma = df.loc[idx_above, metric].mean()
    mb = df.loc[idx_below, metric].mean()
    mi = df.loc[idx_compression, metric].mean()
    return ma, mb, mi


def metric_ratio(ma, mb, mi):
    """
    Compute MSCC (Maximum Spinal Cord Compression) using the chosen metric of compression and of levels
    above and bellow.
    :param float: ma: metric of level above compression level.
    :param float: mb: metric of level above compression level.
    :param float: mi: metric at compression level.
    :return float: metric ratio in %
    """
    return (1 - float(mi) / ((ma + mb) / float(2))) * 100


def metric_ratio_norm(metrics_patients, metrics_HC):
    """
    Compute normalized MSCC (Maximum Spinal Cord Compression) using the chosen metric at the compression and
    levels above and bellow.
    Each metric is divided by the corresponding value in healthy controls.
    :param list: metrics_patients: list metric value of level above, below and at compression of patient.
    :param list: metrics_HC: list metric value of level above, below and at compression of healthy
    controls.
    :return float: MSCC normalized in %
    """
    ma = metrics_patients[0]/metrics_HC[0]
    mb = metrics_patients[1]/metrics_HC[1]
    mi = metrics_patients[2]/metrics_HC[2]
    return metric_ratio(ma, mb, mi)


def save_df_to_csv(dataframe, fname_out):
    """
    Save .csv file of MSCC results.
    :param fname_out:
    :return:
    """
    if os.path.isfile(fname_out):
        # Combine the data with the existing CSV file.
        # Rows with the same (filename, compression_level, slice) triple are merged together.
        # Metric values from the new dataframe take priority over old CSV file values.
        dataframe_old = pd.read_csv(fname_out, index_col=INDEX_COLUMNS)
        dataframe = dataframe.combine_first(dataframe_old)
    dataframe.to_csv(fname_out, na_rep='n/a')


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)    # values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG]

    # Step 0: Argument loading and validation
    # ---------------------------
    # Load input and output filenames
    fname_labels = arguments.l
    fname_segmentation = arguments.i
    fname_vertfile = arguments.vertfile
    distance = arguments.distance
    extent = arguments.extent
    sex = arguments.sex
    age = arguments.age
    mode = arguments.mode
    metric = f'MEAN({arguments.metric})'  # Adjust for csv file columns name
    if arguments.o is not None:
        fname_out = arguments.o
    else:
        path, file_name, ext = extract_fname(get_absolute_path(arguments.i))
        fname_out = os.path.join(path, f'{file_name}_compression_metrics.csv')
    # Check if segmentation, compression labels, and vertebral label files have same dimensions
    img_seg = Image(fname_segmentation).change_orientation('RPI')
    img_labels = Image(fname_labels).change_orientation('RPI')
    if arguments.vertfile:
        img_vertfile = Image(fname_vertfile).change_orientation('RPI')
        if not img_seg.data.shape == img_labels.data.shape == img_vertfile.data.shape:
            raise ValueError(f"Shape mismatch between compression labels [{img_labels.data.shape}], vertebral labels [{img_vertfile.data.shape}]"
                             f" and segmentation [{img_seg.data.shape}]). "
                             f"Please verify that your compression labels and vertebral labels were done in the same space as your input segmentation.")
    path_ref = os.path.join(__data_dir__, 'PAM50_normalized_metrics')
    # Fetch the subfolder that contains the "sub-{site}_{contrast}_PAM50.csv" files
    path_ref_hc = next((folder for (folder, _, filenames) in os.walk(path_ref)
                        if any((f.startswith('sub-') and f.endswith('.csv')) for f in filenames)), None)
    # Check if path_ref with normalized metrics exists
    if arguments.normalize_hc and not os.path.isdir(path_ref):
        raise FileNotFoundError(f"Directory with normalized PAM50 metrics {path_ref} does not exist.\n"
                                f"You can download it using 'sct_download_data -d PAM50_normalized_metrics'.")
    if arguments.normalize_hc and not path_ref_hc:
        raise FileNotFoundError(f"Directory with normalized PAM50 metrics {path_ref} does not contain any CSV files.\n"
                                f"You can try re-downloading it using 'sct_download_data -d PAM50_normalized_metrics'.")
    if arguments.normalize_hc and not arguments.vertfile:
        raise ValueError("Vertebral labeling file is required for the '-normalize-hc' flag. "
                         "Please provide it using the '-vertfile' argument.")

    # Print warning if sex or age are specified without normalized-hc
    if sex and not arguments.normalize_hc:
        parser.error("The 'sex' flag requires '-normalize-hc 1'.")
    if age and not arguments.normalize_hc:
        parser.error("The 'age' flag requires '-normalize-hc 1'.")

    if sex or age:
        fname_partcipants = get_absolute_path(os.path.join(path_ref, 'participants.tsv'))
    if age:
        age.sort()
        if any(n < 0 for n in age):
            parser.error(f'Age range needs to be positive, {age} was specified')

    # Step 1. Get subject metrics and compressed slices
    # -----------------------------------------------------------
    # Call sct_process_segmentation to get morphometrics perslice in native space
    path, file_name, ext = extract_fname(get_absolute_path(arguments.i))
    fname_metrics = os.path.join(path, f'{file_name}_metrics.csv')
    # If vertebral labeling file is provided, use it for the sct_process_segmentation call
    if arguments.vertfile:
        sct_process_segmentation.main(argv=['-i', fname_segmentation, '-vertfile', fname_vertfile, '-perslice', '1', '-o', fname_metrics])
    # But sometimes, the vertebral labeling file is not available (e.g., in severe injuries)
    else:
        sct_process_segmentation.main(argv=['-i', fname_segmentation, '-perslice', '1', '-o', fname_metrics])
    # Fetch metrics of subject
    df_metrics = pd.read_csv(fname_metrics).astype({metric: float})
    # Get compressed slices
    slices_compressed = get_compressed_slice(img_labels, df_metrics, mode)
    # Get vertebral level corresponding to the slice with the compression
    compressed_levels_dict = {}
    if arguments.vertfile:
        compressed_levels_dict = get_verterbral_level_from_slice(slices_compressed, df_metrics)

    # Step 2: Get normalization metrics and slices (using non-compressed subject slices)
    # -----------------------------------------------------------
    # Get spinal cord centerline object to compute the distance
    # Get max and min index of the segmentation with pmj
    _, _, Z = (img_seg.data > NEAR_ZERO_THRESHOLD).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)
    # Get the z index corresponding to the segmentation since the centerline only includes slices of the segmentation.
    z_ref = np.array(range(min_z_index.astype(int), max_z_index.max().astype(int) + 1))
    # Get centerline object
    centerline = get_centerline_object(img_seg, verbose=verbose)
    # Get healthy slices to average for level above and below
    z_range_centerline_above, z_range_centerline_below = get_slices_upper_lower_level_from_centerline(centerline, distance, extent, slices_compressed, z_ref)
    logger.debug(f'Slice range above: {z_range_centerline_above}')
    logger.debug(f'Slice range below: {z_range_centerline_below}')

    # Step 2: Get normalization metrics and slices (using PAM50 and reference dataset)
    # -----------------------------------------------------------
    if arguments.normalize_hc:
        # Select healthy controls based on sex and/or age range
        if sex or age:
            list_HC = select_HC(fname_partcipants, sex, age)
        else:
            list_HC = None
        # Call sct_process_segmentation to get morphometrics perslice in PAM50 space
        fname_metrics_PAM50 = os.path.join(path, f'{file_name}_metrics_PAM50.csv')
        sct_process_segmentation.main(argv=['-i', fname_segmentation, '-vertfile', fname_vertfile, '-normalize-PAM50', '1',
                                      '-perslice', '1', '-o', fname_metrics_PAM50])
        # Get PAM50 slice thickness
        fname_PAM50 = os.path.join(__data_dir__, 'PAM50', 'template', 'PAM50_t2.nii.gz')
        img_pam50 = Image(fname_PAM50).change_orientation('RPI')
        slice_thickness_PAM50 = get_slice_thickness(img_pam50)
        # Fetch metrics of PAM50 template
        df_metrics_PAM50 = pd.read_csv(fname_metrics_PAM50).astype({metric: float})
        # Average metrics of healthy controls
        df_avg_HC = average_hc(path_ref_hc, metric, list_HC)
        # Get slices correspondence in PAM50 space
        compressed_levels_dict_PAM50 = get_slices_in_PAM50(compressed_levels_dict, df_metrics, df_metrics_PAM50)
        z_range_PAM50_below, z_range_PAM50_above = get_slices_upper_lower_level_from_PAM50(compressed_levels_dict_PAM50, df_metrics_PAM50, distance, extent, slice_thickness_PAM50)

    # Step 3a. Compute MSCC metrics for each compressed level (vertebral labeling available)
    # ------------------------------------------------------
    if compressed_levels_dict:
        # Loop through all compressed levels (compute one MSCC per compressed level)
        rows = []
        for idx in compressed_levels_dict.keys():
            # Get compressed level and slice
            level = list(compressed_levels_dict[idx].keys())[0]  # TODO change if more than one level
            slice_list = compressed_levels_dict[idx][level]
            slice_num = slice_list[0]
            printv(f'\nCompression at level {int(level)} (slice {slice_num})', verbose=verbose, type='info')

            # Compute metric ratio (non-normalized)
            metrics_patient = average_metric(df_metrics, metric, z_range_centerline_above, z_range_centerline_below, slice_list)
            metric_ratio_result = metric_ratio(metrics_patient[0], metrics_patient[1], metrics_patient[2])

            if arguments.normalize_hc:
                # Compute metric ratio (normalized, PAM50)
                # NB: This PAM50 dict has the same structure as the regular dict, so it's safe to re-use `idx` here
                slice_avg_PAM50 = list(compressed_levels_dict_PAM50[idx].values())[0]
                metrics_patient_PAM50 = average_metric(df_metrics_PAM50, metric, z_range_PAM50_above, z_range_PAM50_below, slice_avg_PAM50)
                metric_ratio_PAM50_result = metric_ratio(metrics_patient_PAM50[0], metrics_patient_PAM50[1], metrics_patient_PAM50[2])
                # Get metrics of healthy controls
                metrics_HC = average_metric(df_avg_HC, metric, z_range_PAM50_above, z_range_PAM50_below, slice_avg_PAM50)
                logger.debug(f'\nmetric_a_HC = {metrics_HC[0]}, metric_b_HC = {metrics_HC[1]}, metric_i_HC = {metrics_HC[2]}')
                # Compute metric ratio (normalized, PAM50 + HC)
                metric_ratio_norm_result = metric_ratio_norm(metrics_patient_PAM50, metrics_HC)
            else:
                # If not `normalize_hc`, then skip computing normalized metrics
                metric_ratio_PAM50_result = None
                metric_ratio_norm_result = None

            rows.append([arguments.i, level, slice_num,
                         metric_ratio_result,
                         metric_ratio_PAM50_result,
                         metric_ratio_norm_result])

            # Display results
            logger.debug(f'\nmetric_a = {metrics_patient[0]}, metric_b = {metrics_patient[1]}, metric_i = {metrics_patient[2]}')
            printv(f'{arguments.metric}_ratio = {metric_ratio_result}', verbose=verbose, type='info')
            if arguments.normalize_hc:
                logger.debug(f'PAM50: metric_a = {metrics_patient_PAM50[0]}, metric_b = {metrics_patient_PAM50[1]}, metric_i = {metrics_patient_PAM50[2]}')
                printv(f'{arguments.metric}_ratio_PAM50 = {metric_ratio_PAM50_result}', verbose=verbose, type='info')
                printv(f'{arguments.metric}_ratio_PAM50_normalized = {metric_ratio_norm_result}', verbose=verbose, type='info')

        metric_columns = [
            f'{arguments.metric}_ratio',
            f'{arguments.metric}_ratio_PAM50',
            f'{arguments.metric}_ratio_PAM50_normalized',
        ]
        df_metric_ratios = pd.DataFrame.from_records(rows, index=INDEX_COLUMNS, columns=INDEX_COLUMNS + metric_columns)
        save_df_to_csv(df_metric_ratios, fname_out)
        printv(f'\nSaved: {os.path.abspath(fname_out)}')

    # Step 3b. Compute MSCC metrics for each compressed level (vertebral labeling not available)
    # ------------------------------------------------------
    if not compressed_levels_dict:
        # Loop through all compressed levels (compute one MSCC per compressed level)
        rows = []
        for slice_num in slices_compressed:
            slice_list = [slice_num]     # int --> list to make it compatible with the 'average_metric' function below

            # Compute metric ratio (non-normalized)
            ma, mb, mi = average_metric(df_metrics, metric, z_range_centerline_above, z_range_centerline_below, slice_list)
            metric_ratio_result = metric_ratio(ma, mb, mi)

            # The two last None values are placeholders for the PAM50 and normalized metrics (as we cannot compute them
            # without vertebral labeling)
            rows.append([arguments.i, None, slice_num, metric_ratio_result, None, None])
            metric_columns = [
                f'{arguments.metric}_ratio',
                f'{arguments.metric}_ratio_PAM50',
                f'{arguments.metric}_ratio_PAM50_normalized',
            ]
            df_metric_ratios = pd.DataFrame.from_records(rows, index=INDEX_COLUMNS, columns=INDEX_COLUMNS + metric_columns)
            save_df_to_csv(df_metric_ratios, fname_out)
            printv(f'\nSaved: {os.path.abspath(fname_out)}')


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
