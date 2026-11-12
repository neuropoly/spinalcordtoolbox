"""
Functions to interpolate metrics (from sct_process_segmentation) into the PAM50 anatomical dimensions

Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import numpy as np
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.aggregate_slicewise import Metric
from spinalcordtoolbox.template import get_slices_from_vertebral_levels, get_vertebral_level_from_slice


def interpolate_metrics(metrics, fname_vert_levels_PAM50, fname_vert_levels):
    """
    Interpolates metrics perlevel into the PAM50 anatomical dimensions.
    :param metrics: Dict of class Metric(). Output of spinalcordtoolbox.process_seg.compute_shape.
    :param fname_vert_levels_PAM50: Path to the PAM50_levels.nii.gz (PAM50 labeled segmentation).
    :param fname_vert_levels: Path to subject's vertebral labeling file.
    :return metrics_PAM50_space: Dict of class Metric() in PAM50 anatomical dimensions.
    """
    # Load PAM50 labeled segmentation
    im_seg_labeled_PAM50 = Image(fname_vert_levels_PAM50)
    im_seg_labeled_PAM50.change_orientation('RPI')
    # Load subject's labeled segmentation
    im_seg_labeled = Image(fname_vert_levels)
    im_seg_labeled.change_orientation('RPI')

    # Get unique integer vertebral levels (but exclude 0, 49, and 50, as these aren't vertebral levels)
    levels = sorted(int(level) for level in np.unique(im_seg_labeled.data) if 0 < int(level) < 49)

    # Get slices corresponding to each level
    level_slices_PAM50 = [get_slices_from_vertebral_levels(im_seg_labeled_PAM50, level) for level in levels]
    level_slices_im = [get_slices_from_vertebral_levels(im_seg_labeled, level) for level in levels]

    # Compute the mean scaling factor between PAM50 and native slices
    pairs = list(zip(level_slices_PAM50, level_slices_im))
    # Exclude the first/last levels to avoid edge effects (only if there are enough levels)
    trim = 1 if len(pairs) > 2 else 0
    scale_mean = np.mean([len(s_pam) / len(s_im) for s_pam, s_im in pairs[trim:len(pairs)-trim]])

    # Initialize a metrics dict filled by NaN with number of rows equal to number of slices in PAM50 template
    z = im_seg_labeled_PAM50.dim[2]  # z == number of slices
    metrics_PAM50_space_dict = {k: np.full([z], np.nan) for k in metrics.keys()}

    # Loop through slices per-level populating the metrics dict
    # NB: The iteration direction is a bit unintuitive here: we iterate from superior to inferior
    #                          I    <--         <--    S
    #                     "last level"           "first level"
    #  * Iteration:          i==2        i==1        i==0
    #  * Slice numbers:   00 01 02 03|04 05 06 07|08 09 10 11
    #  * Level slices:    C4 C4 C4 C4|C3 C3 C3 C3|C2 C2 C2 C2
    #                                |           |
    #  * Intervertebral discs:     c3-c4       c2-c3
    for i, (level, slices_PAM50, slices_im) in enumerate(zip(levels, level_slices_PAM50, level_slices_im)):
        is_first = (i == 0)
        is_last = (i == len(levels) - 1)

        # Set up the necessary input parameters for the interpolation function `np.interp`: x, xp, and fp.
        #    - (xp, fp) are a set of known input->output pairs (e.g. slice1->csa1, slice2->csa2, etc.)
        #    - (x) is a set of inputs we want (e.g. slices in the PAM50 space)
        #    - However, since our goal is just to go from one linearly-spaced grid to another, all we need to know is:
        #       * A. How many points are in the level in the subject space, and
        #       * B. How many points are in the level in the PAM50 space.
        #    - The plan would be to use these values to create the two linearly-spaced grids and interpolate between:
        #       * `x  = np.linspace(start=0, stop=1, num=n_pam50)`
        #       * `xp = np.linspace(start=0, stop=1, num=n_subj)`
        n_subj = len(slices_im)
        n_pam50 = len(slices_PAM50)
        # However, there is a caveat:
        #    - For the first/last levels in the subject space, the cord seg could be cut off (i.e. partial levels)
        #    - So, instead of using all the points from the corresponding PAM50 level, we estimate how many points the
        #      "partial" PAM50 level would have by using the mean ratio of subj:PAM50 points and multiplying.
        if is_first or is_last:
            n_pam50 = int(scale_mean * n_subj)

        # There is one other caveat here:
        #     - Right now, we are only interpolating within a vertebral level.
        #     - But, this neglects the space in *between* vertebral levels, e.g.:
        #         * Level slices:    C4 C4 C4 C4 C3 C3 C3 C3 C2 C2 C2 C2
        #                                       |           |
        #         * Intervertebral discs:     c3-c4       c2-c3
        #         * Interpolation range:       [0           1]
        #     - If we tried to interpolate the C3 level from the subj space to the PAM50 space, we need to include the
        #       information from the last C2 sample and the first C4 sample.
        #     - So, we inset the range by half of the spacing between points, such that the two discs fall on [0, 1]
        n_spaces_subj = (0 if is_last else 1/2) + (n_subj - 1) + (0 if is_first else 1/2)
        n_spaces_pam50 = (0 if is_last else 1/2) + (n_pam50 - 1) + (0 if is_first else 1/2)
        spacing_subj = (1 / n_spaces_subj) if n_spaces_subj > 0 else 1
        spacing_pam50 = (1 / n_spaces_pam50) if n_spaces_pam50 > 0 else 1
        inset_subj_l = 0 if is_last else (spacing_subj / 2)
        inset_subj_r = 0 if is_first else (spacing_subj / 2)
        inset_pam50_l = 0 if is_last else (spacing_pam50 / 2)
        inset_pam50_r = 0 if is_first else (spacing_pam50 / 2)
        x = np.linspace(start=0 + inset_pam50_l, stop=1 - inset_pam50_r, num=n_pam50)
        xp = np.linspace(start=0 + inset_subj_l,  stop=1 - inset_subj_r,  num=n_subj)

        # Loop through metrics
        for key, value in metrics.items():
            if key != 'length':
                # Get the metric values corresponding to the subject-space slices
                fp = value.data[slices_im]
                xp_full, fp_full = xp, fp

                # Fetch 1 metric from each of the subject's adjacent levels (if they exist) and concat to either side
                if not is_last:
                    xp_full = np.concatenate(([-inset_subj_l], xp_full))
                    fp_full = np.concatenate(([value.data[level_slices_im[i + 1][-1]]], fp_full))
                if not is_first:
                    xp_full = np.concatenate((xp_full, [1 + inset_subj_r]))
                    fp_full = np.concatenate((fp_full, [value.data[level_slices_im[i - 1][0]]]))

                # Interpolate from the full range (incl. adjacent levels) to the PAM50 range
                metrics_inter = np.interp(x=x, xp=xp_full, fp=fp_full)

                # Scale interpolation of first and last levels (to account for incomplete levels)
                diff = len(metrics_inter) - len(slices_PAM50)
                if is_first:
                    # If the first level, scale from level below
                    if diff > 0:
                        metrics_inter = metrics_inter[:-diff]
                    elif diff < 0:
                        slices_PAM50 = slices_PAM50[:-abs(diff)]
                elif is_last:
                    # If the last level, scale from level above
                    if diff > 0:
                        metrics_inter = metrics_inter[diff:]
                    elif diff < 0:
                        slices_PAM50 = slices_PAM50[abs(diff):]
                metrics_PAM50_space_dict[key][slices_PAM50] = metrics_inter

    # Convert dict of ndarrays to dict of Metric() objects
    return {k: Metric(data=np.array(v), label=k) for k, v in metrics_PAM50_space_dict.items()}


def build_pam50_agg_metric(agg_metric_native, nz_native, label_name, method,
                           fname_vert_level, fname_vert_level_PAM50):
    """
    Adapt `interpolate_metrics()` to the per-slice output of `extract_metric()`.
    :param agg_metric_native: per-slice native-space metrics (dict output of extract_metric() with perslice=True)
    :param nz_native: int: total z-slices in native image
    :param label_name: str: atlas label name (for 'Label' CSV column), e.g., 'white matter'
    :param method: str: extraction method ('wa', 'ml', 'map', 'bin', 'median', 'max')
    :param fname_vert_level: str: native vertebral levels file (centerline-masked)
    :param fname_vert_level_PAM50: str: PAM50 template PAM50_levels.nii.gz
    :return: dict keyed by (z,) PAM50 slice tuples, suitable for save_as_csv()
    """
    method_key_map = {
        'wa': 'WA()', 'ml': 'ML()', 'map': 'MAP()',
        'bin': 'BIN()', 'median': 'MEDIAN()', 'max': 'MAX()'
    }
    primary_key = method_key_map[method]

    # Convert metrics from extract_metric() form (one dict per slice, multiple metrics) to
    # compute_shape() form (one Metric object per metric, multiple slices), since that is the
    # form expected by interpolate_metrics()
    metric_1d = np.full(nz_native, np.nan)
    for (z,), entry in agg_metric_native.items():
        val = entry.get(primary_key)
        if val is not None:
            metric_1d[z] = val

    # Interpolate to PAM50 space; returns Dict[str, Metric] with 1D data of length z_PAM50
    metrics_pam50 = interpolate_metrics(
        {primary_key: Metric(data=metric_1d, label=primary_key)},
        fname_vert_level_PAM50,
        fname_vert_level
    )

    # Convert interpolated metrics back into the expected form, from one Metric object per metric
    # back into one dict per slice, since that is the form expected by save_as_csv()
    pam50_values = metrics_pam50[primary_key].data

    # Determine which vertebral levels are present in the native data to filter PAM50 output
    # (excluding 0 and values >=49, which are reserved/non-vertebral-level labels in the PAM50
    # convention: https://spinalcordtoolbox.com/stable/user_section/tutorials/vertebral-labeling/labeling-conventions.html)
    im_native_levels = Image(fname_vert_level).change_orientation('RPI')
    native_levels = set(
        int(v) for v in np.unique(im_native_levels.data) if 0 < int(v) < 49
    )

    # Map each PAM50 z-slice to a vertebral level and build the output agg_metric
    im_pam50_levels = Image(fname_vert_level_PAM50).change_orientation('RPI')

    agg_metric_pam50 = {}
    for z_pam50, val in enumerate(pam50_values):
        # nan means that there was no data in the native space for that slice, so we skip it in the PAM50 space as well
        if np.isnan(val):
            continue
        vert_level = get_vertebral_level_from_slice(im_pam50_levels, z_pam50)
        if vert_level is None or vert_level not in native_levels:
            continue
        entry = {
            'Label': label_name,
            'VertLevel': (vert_level,),
            'DistancePMJ': None,    # required by save_as_csv() but not relevant for PAM50 space
            primary_key: val,
        }
        agg_metric_pam50[(z_pam50,)] = entry

    return agg_metric_pam50
