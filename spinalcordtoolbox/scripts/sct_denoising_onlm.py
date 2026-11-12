#!/usr/bin/env python
#
# Utility function to denoise images
#
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
from time import time
from typing import Sequence

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.fs import extract_fname
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel, LazyLoader
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser, display_viewer_syntax

nlmeans = LazyLoader("nlmeans", globals(), "dipy.denoise.nlmeans")
nib = LazyLoader("nib", globals(), "nibabel")


# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1  # verbose
        self.remove_temp_files = 1
        self.parameter = "Rician"
        self.file_to_denoise = ''
        self.output_file_name = ''


def get_parser():
    parser = SCTArgumentParser(
        description='Utility function to denoise images. Return the denoised image and also the difference '
                    'between the input and the output. The denoising algorithm is based on the Non-local means '
                    'methods (Pierrick Coupe, Jose Manjon, Montserrat Robles, Louis Collins. “Adaptive Multiresolution '
                    'Non-Local Means Filter for 3D MR Image Denoising” IET Image Processing, Institution of '
                    'Engineering and Technology, 2011). The implementation is based on Dipy (https://dipy.org/).'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        "-i",
        help="Input NIFTI image to be denoised. Example: `image_input.nii.gz`",
        metavar=Metavar.file,
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-p",
        help='Type of assumed noise distribution.',
        choices=("Rician", "Gaussian"),
        default="Rician")
    optional.add_argument(
        "-d",
        type=int,
        help="Threshold value for what to be considered as noise. "
             "The standard deviation of the noise is calculated for values below this limit. "
             "Not relevant if `-std` value is precised.",
        metavar=Metavar.int,
        default="80")
    optional.add_argument(
        "-std",
        type=float,
        help="Standard deviation of the noise. "
             "If not specified, it is calculated using a background of point of values "
             "below the threshold value (parameter `-d`).",
        metavar=Metavar.float)
    optional.add_argument(
        "-o",
        help="Name of the output NIFTI image.",
        metavar=Metavar.str)

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_tempfile_args()

    return parser


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    parameter = arguments.p
    remove_temp_files = arguments.r
    noise_threshold = arguments.d

    file_to_denoise = arguments.i
    output_file_name = arguments.o
    std_noise = arguments.std

    param = Param()
    param.verbose = verbose
    param.remove_temp_files = remove_temp_files
    param.parameter = parameter

    path, file, ext = extract_fname(file_to_denoise)

    img = nib.load(file_to_denoise)
    hdr_0 = img.header

    data = np.asanyarray(img.dataobj)

    if min(data.shape) <= 5:
        printv('One of the image dimensions is <= 5 : reducing the size of the block radius.')
        block_radius = min(data.shape) - 1
    else:
        block_radius = 5  # default value

    # Process for manual detecting of background
    # mask = data[:, :, :] > noise_threshold
    # data = data[:, :, :]

    if arguments.std is not None:
        sigma = std_noise
        # Application of NLM filter to the image
        printv('Applying Non-local mean filter...')
        if param.parameter == 'Rician':
            den = nlmeans.nlmeans(data, sigma=sigma, mask=None, rician=True, block_radius=block_radius)
        else:
            den = nlmeans.nlmeans(data, sigma=sigma, mask=None, rician=False, block_radius=block_radius)
    else:
        # # Process for manual detecting of background
        mask = data > noise_threshold
        sigma = np.std(data[~mask])
        # Application of NLM filter to the image
        printv('Applying Non-local mean filter...')
        if param.parameter == 'Rician':
            den = nlmeans.nlmeans(data, sigma=sigma, mask=mask, rician=True, block_radius=block_radius)
        else:
            den = nlmeans.nlmeans(data, sigma=sigma, mask=mask, rician=False, block_radius=block_radius)

    t = time()
    printv(f"total time: {time() - t}")
    printv(f"vol size {den.shape}")

    axial_middle = int(data.shape[2] / 2)

    before = data[:, :, axial_middle].T
    after = den[:, :, axial_middle].T

    diff_3d = np.absolute(den.astype('f8') - data.astype('f8'))
    difference = np.absolute(after.astype('f8') - before.astype('f8'))
    if arguments.std is None:
        difference[~mask[:, :, axial_middle].T] = 0

    if param.verbose == 2:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(before, cmap='gray', origin='lower')
        ax[0].set_title('before')
        ax[1].imshow(after, cmap='gray', origin='lower')
        ax[1].set_title('after')
        ax[2].imshow(difference, cmap='gray', origin='lower')
        ax[2].set_title('difference')
        for i in range(3):
            ax[i].set_axis_off()

        plt.show()

    # Save files
    img_denoise = Image(param=den, hdr=hdr_0)
    img_diff = Image(param=diff_3d, hdr=hdr_0)
    if output_file_name is not None:
        output_file_name = output_file_name
    else:
        output_file_name = f'{file}_denoised{ext}'
    img_denoise.save(output_file_name)
    img_diff.save(f'{file}_difference{ext}')

    display_viewer_syntax(files=[file_to_denoise, output_file_name], verbose=verbose)


# =======================================================================================================================
# Start program
# =======================================================================================================================
if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
