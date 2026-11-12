#!/usr/bin/env python
#
# Compute TSNR using inputed anat.nii.gz and fmri.nii.gz files.
#
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
import os
from typing import Sequence

import numpy as np

from spinalcordtoolbox.image import Image, add_suffix, empty_like, rpi_slice_to_orig_orientation
from spinalcordtoolbox.utils.sys import init_sct, set_loglevel, printv
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser, display_viewer_syntax
from spinalcordtoolbox.reports import qc2


class Param:
    def __init__(self):
        self.debug = 1
        self.verbose = 1


class Tsnr:
    def __init__(self, param=None, fmri=None, anat=None, mask=None, out=None, verbose=None):
        if param is not None:
            self.param = param
        else:
            self.param = Param()
        self.fmri = fmri
        self.anat = anat
        self.mask = mask
        self.out = out
        self.verbose = verbose

    def compute(self):

        fname_data = self.fmri
        # open data
        nii_data = Image(fname_data)
        orientation_fmri = nii_data.orientation
        if not orientation_fmri == 'RPI':
            nii_data.change_orientation('RPI')
        data = nii_data.data

        # compute mean
        data_mean = np.mean(data, 3)
        # compute STD
        data_std = np.std(data, 3, ddof=1)
        # compute TSNR
        data_tsnr = data_mean / data_std
        # Change nan to zeros
        data_tsnr[np.isnan(data_tsnr)] = 0
        # compute mean tSNR per slice if mask is there
        if self.mask is not None:
            mask = Image(self.mask)
            orientation_mask = mask.orientation
            if not orientation_mask == 'RPI':
                mask = mask.change_orientation('RPI')
            data_tsnr_masked = data_tsnr * mask.data
            for z in range(data_tsnr_masked.shape[-1]):
                # Display result
                tsnr_roi = (data_tsnr_masked[:, :, z])[data_tsnr_masked[:, :, z] != 0].mean()
                slice_orig = rpi_slice_to_orig_orientation(data.shape, orientation_fmri, z, 2)
                printv(f'\nSlice {slice_orig},  tSNR = {tsnr_roi:.2f}', type='info')
            tsnr_roi = (data_tsnr_masked)[data_tsnr_masked != 0].mean()
            printv(f'\ntSNR = {tsnr_roi:.2f}', type='info')
        # Roerient to original space

        # save TSNR
        fname_tsnr = self.out
        nii_tsnr = empty_like(nii_data)
        nii_tsnr.data = data_tsnr
        nii_tsnr.change_orientation(orientation_fmri)
        nii_tsnr.save(fname_tsnr, dtype=np.float32)


# PARSER
# ==========================================================================================
def get_parser():
    parser = SCTArgumentParser(
        description="Compute temporal SNR (tSNR) in fMRI time series."
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        help="Input fMRI data. Example: `fmri.nii.gz`"
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-m',
        help='Binary (or weighted) mask within which tSNR will be averaged. Example: `fmri_moco_mean_seg.nii.gz`',
        metavar=Metavar.file,
        )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help="tSNR data output file. Example: `fmri_tsnr.nii.gz`"
    )
    optional.add_argument(
        '-qc',
        metavar=Metavar.str,
        help='The path where the quality control generated content will be saved. Note: The `-m` parameter is '
             'required to generate the QC report, as it is necessary to center the QC on the region of interest.')
    optional.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the dataset the process was run on',)
    optional.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the subject the process was run on',)
    optional.add_argument(
        '-qc-contrast',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the contrast the process was run on.",)

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    param = Param()

    fname_src = arguments.i
    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject
    qc_contrast = arguments.qc_contrast

    # Check dimensionality of mask
    fname_mask = arguments.m
    if fname_mask is not None:
        mask = Image(fname_mask).data
        if len(mask.shape) != 3:
            raise ValueError(f"Mask should be a 3D image, but the input mask has shape '{mask.shape}'.")

    if path_qc and fname_mask is None:
        parser.error("The '-m' parameter is required to generate the QC report.")

    if arguments.o is not None:
        fname_dst = arguments.o
    else:
        fname_dst = add_suffix(fname_src, "_tsnr")

    # call main function
    tsnr = Tsnr(param=param, fmri=fname_src, mask=fname_mask, out=fname_dst, verbose=verbose)
    tsnr.compute()

    display_viewer_syntax([fname_dst], verbose=verbose)

    if path_qc is not None:
        qc2.sct_fmri_compute_tsnr(
            fname_input=fname_dst,
            fname_output=fname_dst,
            fname_seg=fname_mask,
            argv=argv,
            path_qc=os.path.abspath(path_qc),
            dataset=qc_dataset,
            subject=qc_subject,
            contrast=qc_contrast,
        )


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
