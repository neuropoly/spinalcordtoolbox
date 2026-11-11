#!/usr/bin/env python
#
# Utility function to denoise diffusion MRI images
#
# Copyright (c) 2022 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import os
import sys
from typing import Sequence
import textwrap

import numpy as np

from spinalcordtoolbox.image import add_suffix, Image
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, list_type, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, set_loglevel, printv, LazyLoader

patch2self = LazyLoader("patch2self", globals(), 'dipy.denoise.patch2self')
nib = LazyLoader("nib", globals(), "nibabel")


def get_parser():
    parser = SCTArgumentParser(
        description=textwrap.dedent("""
            Utility function to denoise diffusion MRI images. Returns the denoised image and also the difference between the input and the output. The Patch2Self denoising algorithm is based on self-supervised denoising via statistical independence of noise, as described in the following publications:

              - Fadnavis et al. Patch2Self: Denoising Diffusion MRI with Self-supervised Learning. NeurIPS, 2020, Vol. 33. (https://doi.org/10.48550/arXiv.2011.01355)
              - Schilling et al. Patch2Self denoising of diffusion MRI in the cervical spinal cord improves intra-cord contrast, signal modelling, repeatability, and feature conspicuity. medRxiv, 2021. (https://doi.org/10.1101/2021.10.04.21264389)

            The implementation is based on DIPY (https://docs.dipy.org/stable/examples_built/preprocessing/denoise_patch2self.html).
        """),  # noqa: E501 (line too long)
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        "-i",
        help="Input NIfTI image to be denoised. Example: `image_input.nii.gz`",
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        "-b",
        help="Input bvals file corresponding to the NIfTI file to be denoised. Example: `filename.bval`",
        metavar=Metavar.file,
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-model",
        help="Type of regression model used for self-supervised training within Patch2Self.",
        choices=('ols', 'ridge', 'lasso'),
        default='ols',
    )
    optional.add_argument(
        "-radius",
        help=textwrap.dedent("""
            Patch Radius used to generate p-neighbourhoods within Patch2Self. Notes:

              - A radius of `0` will use 1x1x1 p-neighbourhoods, a radius of `1` will use 3x3x3 p-neighbourhoods, and so on.
              - For anisotropic patch sizes, provide a comma-delimited list of 3 integers. (e.g. `-radius 0,1,0`). For isotropic patch sizes, provide a single int value (e.g. `-radius 0`).
        """),  # noqa: E501 (line too long)
        metavar=Metavar.int,
        default="0",
    )
    optional.add_argument(
        "-o",
        help="Name of the output NIFTI image.",
        metavar=Metavar.str,
    )

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    model = arguments.model
    if "," in arguments.radius:
        patch_radius = list_type(",", int)(arguments.radius)
    else:
        patch_radius = int(arguments.radius)

    file_to_denoise = arguments.i
    fname_bvals = arguments.b
    if arguments.o is not None:
        output_file_name_denoised = arguments.o
        output_file_name_diff = add_suffix(arguments.o, "_difference")
    else:
        _, filename = os.path.split(file_to_denoise)
        output_file_name_denoised = add_suffix(filename, "_patch2self_denoised")
        output_file_name_diff = add_suffix(filename, "_patch2self_difference")

    nii = nib.load(file_to_denoise)
    bvals = np.loadtxt(fname_bvals)
    hdr = nii.header
    data = nii.get_fdata()

    printv("Applying Patch2Self Denoising...")
    data_denoised = patch2self.patch2self(data, bvals, patch_radius, model, verbose=True)
    data_diff = np.absolute(data_denoised.astype('f8') - data.astype('f8'))

    if verbose == 2:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3)
        axial_middle = data.shape[2] // 2
        middle_vol = data.shape[3] // 2
        before = data[:, :, axial_middle, middle_vol].T
        ax[0].imshow(before, cmap='gray', origin='lower')
        ax[0].set_title("before")
        after = data_denoised[:, :, axial_middle, middle_vol].T
        ax[1].imshow(after, cmap='gray', origin='lower')
        ax[1].set_title("after")
        difference = data_diff[:, :, axial_middle, middle_vol].T
        ax[2].imshow(difference, cmap='gray', origin='lower')
        ax[2].set_title("difference")
        for i in range(3):
            ax[i].set_axis_off()
        plt.show()

    # Save files
    nii_denoised = Image(param=data_denoised, hdr=hdr)
    nii_diff = Image(param=data_diff, hdr=hdr)
    nii_denoised.save(output_file_name_denoised)
    nii_diff.save(output_file_name_diff)

    display_viewer_syntax([file_to_denoise, output_file_name_denoised, output_file_name_diff], verbose=verbose)


if __name__ == '__main__':
    init_sct()
    main(sys.argv[1:])
