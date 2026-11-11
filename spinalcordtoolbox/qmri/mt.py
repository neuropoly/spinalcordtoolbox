"""
This is the interface API to compute MT-related metrics

Code is based on QMRLab: https://github.com/neuropoly/qMRLab

Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def divide_after_removing_zero(dividend, divisor, threshold, replacement=np.nan):
    """
    Mask zero, divide, look for numbers larger than 'threshold', and replace masked elements.

    :param dividend: np.array
    :param divisor: np.array
    :param threshold: float
    :param replacement: float: value to replace masked value with.
    :return: result_full
    """
    ind_nonzero = np.where(divisor)
    n_zero = divisor.size - len(ind_nonzero[0])
    logger.info(f"Found {n_zero} voxels with value=0. These will be replaced by {replacement}.")
    # divide without zero element in divisor
    result = np.true_divide(dividend[ind_nonzero], divisor[ind_nonzero])
    # find aberrant values above threshold
    logger.info(f"Threshold to clip values: +/- {threshold}")
    np.clip(result, -threshold, threshold, out=result)
    # initiate resulting array with replacement values
    result_full = np.full_like(dividend, fill_value=replacement, dtype='float32')
    result_full[ind_nonzero] = result
    return result_full


def compute_mtr(nii_mt1, nii_mt0, threshold_mtr=100):
    """
    Compute Magnetization Transfer Ratio in percentage.

    :param nii_mt1: Image object
    :param nii_mt0: Image object
    :param threshold_mtr: float: value above which number will be clipped
    :return: nii_mtr
    """
    # Convert input to avoid numerical errors from int16 data
    # Related issue: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3636
    nii_mt1.change_type('float32')
    nii_mt0.change_type('float32')

    # Initialize Image object
    nii_mtr = nii_mt1.copy()

    # Compute MTR
    nii_mtr.data = divide_after_removing_zero(100 * (nii_mt0.data - nii_mt1.data), nii_mt0.data, threshold_mtr)
    return nii_mtr


def compute_mtsat(nii_mt, nii_pd, nii_t1,
                  tr_mt, tr_pd, tr_t1,
                  fa_mt, fa_pd, fa_t1,
                  nii_b1map=None):
    """
    Compute MTsat (in percent) and T1 map (in s) based on FLASH scans

    :param nii_mt: Image object for MTw
    :param nii_pd: Image object for PDw
    :param nii_t1: Image object for T1w
    :param tr_mt: Float: Repetition time (in s) for MTw image
    :param tr_pd: Float: Repetition time (in s) for PDw image
    :param tr_t1: Float: Repetition time (in s) for T1w image
    :param fa_mt: Float: Flip angle (in deg) for MTw image
    :param fa_pd: Float: Flip angle (in deg) for PDw image
    :param fa_t1: Float: Flip angle (in deg) for T1w image
    :param nii_b1map: Image object for B1-map (optional)
    :return: MTsat and T1map.
    """
    # params
    nii_t1map = \
        None  # it would be possible in the future to input T1 map from elsewhere (e.g. MP2RAGE). Note: this T1map
    # needs to be in s unit.
    b1correctionfactor = \
        0.4  # empirically defined in https://doi.org/10.3389/fnins.2013.00095
    # R1 threshold, below which values will be clipped.
    r1_threshold = 0.01  # R1=0.01 s^-1 corresponds to T1=100s which is a reasonable threshold
    # Similarly, we also set a threshold for MTsat values
    mtsat_threshold = 1  # we expect MTsat to be on the order of 0.01

    # Convert flip angles into radians
    fa_mt_rad = np.radians(fa_mt)
    fa_pd_rad = np.radians(fa_pd)
    fa_t1_rad = np.radians(fa_t1)

    # ignore warnings from division by zeros (will deal with that later)
    seterr_old = np.seterr(over='ignore', divide='ignore', invalid='ignore')

    # check if a T1 map was given in input; if not, compute R1
    if nii_t1map is None:
        # compute R1
        logger.info("Compute T1 map...")
        r1map = 0.5 * np.true_divide((fa_t1_rad / tr_t1) * nii_t1.data - (fa_pd_rad / tr_pd) * nii_pd.data,
                                     nii_pd.data / fa_pd_rad - nii_t1.data / fa_t1_rad)
        # apply B1 correction if available
        if nii_b1map is not None:
            b1_squared = np.multiply(nii_b1map.data, nii_b1map.data)
            r1map = np.multiply(r1map, b1_squared)
        # remove nans and clip unrelistic values
        r1map = np.nan_to_num(r1map)
        ind_unrealistic = np.where(r1map < r1_threshold)
        if ind_unrealistic[0].size:
            logger.warning(f"R1 values were found to be lower than {r1_threshold}. They will be set to inf, producing T1=0 for "
                           "these voxels.")
            r1map[ind_unrealistic] = np.inf  # set to infinity so that these values will be 0 on the T1map
        # compute T1
        nii_t1map = nii_mt.copy()
        nii_t1map.data = 1. / r1map
    else:
        logger.info("Use input T1 map.")
        r1map = 1. / nii_t1map.data

    # Compute A
    logger.info("Compute A...")
    a = (tr_pd * fa_t1_rad / fa_pd_rad - tr_t1 * fa_pd_rad / fa_t1_rad) * \
        np.true_divide(np.multiply(nii_pd.data, nii_t1.data, dtype=float),
                       tr_pd * fa_t1_rad * nii_t1.data - tr_t1 * fa_pd_rad * nii_pd.data)
    # apply B1 correction if available
    if nii_b1map is not None:
        a = np.divide(a, nii_b1map.data)

    # Compute MTsat
    logger.info("Compute MTsat...")
    nii_mtsat = nii_mt.copy()
    nii_mtsat.data = tr_mt * np.multiply((fa_mt_rad * np.true_divide(a, nii_mt.data) - 1),
                                         r1map, dtype=float) - (fa_mt_rad ** 2) / 2.
    # remove nans and clip unrelistic values
    nii_mtsat.data = np.nan_to_num(nii_mtsat.data)
    ind_unrealistic = np.where(np.abs(nii_mtsat.data) > mtsat_threshold)
    if ind_unrealistic[0].size:
        logger.warning(f"MTsat values were found to be larger than {mtsat_threshold}. They will be set to zero for these voxels.")
        nii_mtsat.data[ind_unrealistic] = 0
    # convert into percent unit (p.u.)
    nii_mtsat.data *= 100

    # Apply B1 correction to result
    # Weiskopf, N., Suckling, J., Williams, G., Correia, M.M., Inkster, B., Tait, R., Ooi, C., Bullmore, E.T., Lutti,
    # A., 2013. Quantitative multi-parameter mapping of R1, PD(*), MT, and R2(*) at 3T: a multi-center validation.
    # Front. Neurosci. 7, 95.
    if nii_b1map is not None:
        nii_mtsat.data = np.true_divide(nii_mtsat.data * (1 - b1correctionfactor),
                                        (1 - b1correctionfactor * nii_b1map.data))

    # set back old seterr settings
    np.seterr(**seterr_old)

    return nii_mtsat, nii_t1map
