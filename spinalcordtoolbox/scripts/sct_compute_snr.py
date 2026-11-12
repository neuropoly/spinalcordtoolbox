#!/usr/bin/env python
#
# Compute SNR in a given ROI according to different methods presented in Dietrich et al.,
# Measurement of signal-to-noise ratios in MR images: Influence of multichannel coils,
# parallel imaging, and reconstruction filters (2007).
#
# Also see this discussion: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3481
#
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
from typing import Sequence
import textwrap

import numpy as np

from spinalcordtoolbox.image import Image, empty_like, add_suffix
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser, parse_num_list


# PARAMETERS
class Param(object):
    # The constructor
    def __init__(self):
        self.almost_zero = np.finfo(float).eps


def get_parser():
    parser = SCTArgumentParser(
        description='Compute SNR using methods described in [Dietrich et al., Measurement of'
                    ' signal-to-noise ratios in MR images: Influence of multichannel coils, parallel '
                    'imaging, and reconstruction filters. J Magn Reson Imaging 2007; 26(2): 375-385].'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        help=textwrap.dedent("""
            Image to compute the SNR on. Example: `b0s.nii.gz`

              - For `-method diff` and `-method mult`, the image must be 4D, as SNR will be computed along the 4th dimension.
              - For `-method single`, the image can either be 3D or 4D. If a 4D image is passed, a specific 3D volume should be specified using the `-vol` argument.
        """),
        metavar=Metavar.file)

    optional = parser.optional_arggroup
    optional.add_argument(
        '-m',
        help='Binary (or weighted) mask within which SNR will be averaged. Example: `dwi_moco_mean_seg.nii.gz`',
        metavar=Metavar.file,
        default='')
    optional.add_argument(
        '-method',
        help="Method to use to compute the SNR:\n"
             "  - `diff`: Substract two volumes (defined by -vol) and estimate noise variance within the ROI "
             "(flag `-m` is required). Requires a 4D volume.\n"
             "  - `mult`: Estimate noise variance over time across volumes specified with `-vol`. Requires a 4D volume.\n"
             "  - `single`: Compute the mean signal in the mask specified by `-m` and estimate the noise variance in a "
             "mask specified by `-m-noise`. If the noise mask is in the background (air), the noise variance needs to "
             "be corrected for Rayleigh distribution (set `-rayleigh 1`). If the noise mask is located in a "
             "region with high signal (eg: tissue), noise distribution can be assumed Gaussian and there is no need to "
             "correct for Rayleigh distribution (use `-rayleigh 0`). This implementation corresponds to the SNRstdv "
             "in the Dietrich et al. article. Uses a 3D or a 4D volume. If a 4D volume is input, the volume to "
             "compute SNR on is specified by `-vol`.",
        choices=('diff', 'mult', 'single'),
        default='diff')
    optional.add_argument(
        '-m-noise',
        help="Binary (or weighted) mask within which noise will be calculated. Only valid for `-method single`.",
        metavar=Metavar.file,
        default='')
    optional.add_argument(
        '-vol',
        help=textwrap.dedent("""
            Volumes to compute SNR from. Separate with `,` (Example: `-vol 0,1`), or select range using `:` (Example: `-vol 2:50`).

            If this argument is not passed:

              - For `-method mult`, all volumes will be used.
              - For `-method diff`, the first two volumes will be used.
              - For `-method single`, the first volume will be used.
        """),
        metavar=Metavar.str,
        default='')
    optional.add_argument(
        '-rayleigh',
        type=int,
        help="Correct for Rayleigh distribution. It is recommended to always use this correction for the 'diff' method "
             "and to use it with the 'single' method in case the noise mask is taken in a region with low SNR (e.g., "
             "the air). ",
        default=1,
        choices=(0, 1))
    optional.add_argument(
        '-o',
        metavar=Metavar.str,
        type=str,
        help="File name to write the computed SNR to."
    )

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_tempfile_args()

    return parser


def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    Source: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return np.sqrt(variance)


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    # Default params
    param = Param()

    # Get parser info
    fname_data = arguments.i
    fname_mask = arguments.m
    fname_mask_noise = arguments.m_noise
    method = arguments.method
    file_name = arguments.o
    rayleigh_correction = arguments.rayleigh

    # Check parameters
    if method in ['diff', 'single']:
        if not fname_mask:
            parser.error(f"Argument '-m' must be specified when using '-method {method}'.")

    # Load data
    im_data = Image(fname_data)
    data = im_data.data
    dim = len(data.shape)
    nz = data.shape[2]
    if fname_mask:
        mask = Image(fname_mask).data

    # Check dimensionality
    if method in ['diff', 'mult']:
        if dim != 4:
            raise ValueError(f"Input data dimension: {dim}. Input dimension for this method should be 4.")
    if method in ['single']:
        if dim not in [3, 4]:
            raise ValueError(f"Input data dimension: {dim}. Input dimension for this method should be 3 or 4.")

    # Check dimensionality of mask
    if fname_mask:
        if len(mask.shape) != 3:
            raise ValueError(f"Mask should be a 3D image, but the input mask has shape '{mask.shape}'.")

    # Retrieve selected volumes
    index_vol = parse_num_list(arguments.vol)
    if not index_vol:
        if method == 'mult':
            index_vol = range(data.shape[3])
        elif method == 'diff':
            index_vol = [0, 1]
        elif method == 'single':
            index_vol = [0]

    # Compute SNR
    # NB: "time" is assumed to be the 4th dimension of the variable "data"
    if method == 'mult':
        # Compute mean and STD across time
        data_mean = np.mean(data[:, :, :, index_vol], axis=3)
        data_std = np.std(data[:, :, :, index_vol], axis=3, ddof=1)
        # Generate mask where std is different from 0
        mask_std_nonzero = np.where(data_std > param.almost_zero)
        snr_map = np.zeros_like(data_mean)
        snr_map[mask_std_nonzero] = data_mean[mask_std_nonzero] / data_std[mask_std_nonzero]
        # Output SNR map
        fname_snr = add_suffix(fname_data, f'_SNR-{method}')
        im_snr = empty_like(im_data)
        im_snr.data = snr_map
        im_snr.save(fname_snr, dtype=np.float32)
        # Output non-zero mask
        fname_stdnonzero = add_suffix(fname_data, f'_mask-STD-nonzero{method}')
        im_stdnonzero = empty_like(im_data)
        data_stdnonzero = np.zeros_like(data_mean)
        data_stdnonzero[mask_std_nonzero] = 1
        im_stdnonzero.data = data_stdnonzero
        im_stdnonzero.save(fname_stdnonzero, dtype=np.float32)
        # Compute SNR in ROI
        if fname_mask:
            snr_roi = np.average(snr_map[mask_std_nonzero], weights=mask[mask_std_nonzero])

    elif method == 'diff':
        # Check user selected exactly 2 volumes for this method.
        if not len(index_vol) == 2:
            raise ValueError(f"Number of selected volumes: {len(index_vol)}. The method 'diff' should be used with "
                             f"exactly 2 volumes. You can specify the number of volumes with the flag '-vol'.")
        data_2vol = np.take(data, index_vol, axis=3)
        # Compute mean across the two volumes
        data_mean = np.mean(data_2vol, axis=3)
        # Compute mean in ROI for each z-slice, if the slice in the mask is not null
        mean_in_roi = [np.average(data_mean[..., iz], weights=mask[..., iz])
                       for iz in range(nz) if np.any(mask[..., iz])]
        data_sub = np.subtract(data_2vol[:, :, :, 1], data_2vol[:, :, :, 0])
        # Compute STD in the ROI for each z-slice. The "np.sqrt(2)" results from the variance of the subtraction of two
        # distributions: var(A-B) = var(A) + var(B).
        # More context in: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3481
        std_in_roi = [weighted_std(data_sub[..., iz] / np.sqrt(2), weights=mask[..., iz])
                      for iz in range(nz) if np.any(mask[..., iz])]
        # Compute SNR
        snr_roi_slicewise = [m/s for m, s in zip(mean_in_roi, std_in_roi)]
        snr_roi = sum(snr_roi_slicewise) / len(snr_roi_slicewise)

    elif method == 'single':
        # Check that the input volume is 3D, or if it is 4D, that the user selected exactly 1 volume for this method.
        if dim == 3:
            data3d = data
        elif dim == 4:
            if not len(index_vol) == 1:
                raise ValueError(f"Selected volumes: {index_vol}. The method 'single' should be used with "
                                 f"exactly 1 volume. You can specify the index of the volume with the flag '-vol'.")
            data3d = np.squeeze(data[..., index_vol])
        # Check that input noise mask is provided
        if fname_mask_noise:
            mask_noise = Image(fname_mask_noise).data
        else:
            parser.error("A noise mask is mandatory with '-method single'.")
        # Check dimensionality of the noise mask
        if len(mask_noise.shape) != 3:
            raise ValueError(f"Input noise mask dimension: {dim}. Input dimension for the noise mask should be 3.")
        # Check that non-null slices are consistent between mask and mask_noise.
        for iz in range(nz):
            if not np.any(mask[..., iz]) == np.any(mask_noise[..., iz]):
                raise ValueError(f"Slice {iz} is empty in either mask or mask_noise. Non-null slices should be "
                                 f"consistent between mask and mask_noise.")
        # Compute mean in ROI for each z-slice, if the slice in the mask is not null
        mean_in_roi = [np.average(data3d[..., iz], weights=mask[..., iz])
                       for iz in range(nz) if np.any(mask[..., iz])]
        std_in_roi = [weighted_std(data3d[..., iz], weights=mask_noise[..., iz])
                      for iz in range(nz) if np.any(mask_noise[..., iz])]
        # Compute SNR
        snr_roi_slicewise = [m/s for m, s in zip(mean_in_roi, std_in_roi)]
        snr_roi = sum(snr_roi_slicewise) / len(snr_roi_slicewise)
        if rayleigh_correction:
            # Correcting for Rayleigh noise (see eq. A12 in Dietrich et al.)
            snr_roi *= np.sqrt((4 - np.pi) / 2)

    # Display result
    if fname_mask:
        printv(f'\nSNR_{method} = {snr_roi}\n', type='info')

    # Added function for text file
    if file_name is not None:
        with open(file_name, "w") as f:
            f.write(str(snr_roi))
            printv(f'\nFile saved to {file_name}')


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
