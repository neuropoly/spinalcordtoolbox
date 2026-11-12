#!/usr/bin/env python
#
# Compute DTI.
#
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
from typing import Sequence
import textwrap

from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser


def get_parser():
    parser = SCTArgumentParser(
        description='Compute Diffusion Tensor Images (DTI) using dipy.'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        "-i",
        help='Input 4d file. Example: `dmri.nii.gz`',
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        "-bval",
        help='Bvals file. Example: `bvals.txt`',
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        "-bvec",
        help='Bvecs file. Example: `bvecs.txt`',
        metavar=Metavar.file,
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-method',
        help=textwrap.dedent("""
            Type of method to calculate the diffusion tensor:

              - `standard:` Standard equation [Basser, Biophys J 1994]
              - `restore`: Robust fitting with outlier detection [Chang, MRM 2005]
        """),
        default='standard',
        choices=('standard', 'restore'))
    optional.add_argument(
        "-evecs",
        type=int,
        help='Output tensor eigenvectors and eigenvalues.',
        default=0,
        choices=(0, 1))
    optional.add_argument(
        '-m',
        metavar=Metavar.file,
        help='Mask used to compute DTI in for faster processing. Example: `mask.nii.gz`')
    optional.add_argument(
        '-o',
        help='Output prefix.',
        metavar=Metavar.str,
        default='dti_')

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

    # initialization
    file_mask = ''
    fname_in = arguments.i
    fname_bvals = arguments.bval
    fname_bvecs = arguments.bvec
    prefix = arguments.o
    method = arguments.method
    evecs = arguments.evecs
    if arguments.m is not None:
        file_mask = arguments.m

    # compute DTI
    if not compute_dti(fname_in, fname_bvals, fname_bvecs, prefix, method, evecs, file_mask, verbose):
        printv('ERROR in compute_dti()', 1, 'error')


# compute_dti
# ==========================================================================================
def compute_dti(fname_in, fname_bvals, fname_bvecs, prefix, method, evecs, file_mask, verbose):
    """
    Compute DTI.
    :param fname_in: input 4d file.
    :param bvals: bvals txt file
    :param bvecs: bvecs txt file
    :param prefix: output prefix. Example: "dti_"
    :param method: algo for computing dti
    :param evecs: bool: output diffusion tensor eigenvectors and eigenvalues
    :return: True/False
    """
    # Open file.
    from spinalcordtoolbox.image import Image
    nii = Image(fname_in)
    data = nii.data
    printv(f'data.shape {data.shape}')

    # open bvecs/bvals
    from dipy.io import read_bvals_bvecs
    bvals, bvecs = read_bvals_bvecs(fname_bvals, fname_bvecs)
    from dipy.core.gradients import gradient_table
    gtab = gradient_table(bvals, bvecs)

    # mask and crop the data. This is a quick way to avoid calculating Tensors on the background of the image.
    if not file_mask == '':
        printv('Open mask file...', verbose)
        # open mask file
        nii_mask = Image(file_mask)
        mask = nii_mask.data

    # fit tensor model
    printv(f'Computing tensor using "{method}" method...', verbose)
    import dipy.reconst.dti as dti
    if method == 'standard':
        tenmodel = dti.TensorModel(gtab)
        if file_mask == '':
            tenfit = tenmodel.fit(data)
        else:
            tenfit = tenmodel.fit(data, mask)
    elif method == 'restore':
        import dipy.denoise.noise_estimate as ne
        sigma = ne.estimate_sigma(data)
        dti_restore = dti.TensorModel(gtab, fit_method='RESTORE', sigma=sigma)
        if file_mask == '':
            tenfit = dti_restore.fit(data)
        else:
            tenfit = dti_restore.fit(data, mask)

    # Compute metrics
    printv('Computing metrics...', verbose)
    # FA
    nii.data = tenfit.fa
    nii.save(f'{prefix}FA.nii.gz', dtype='float32')
    # MD
    nii.data = tenfit.md
    nii.save(f'{prefix}MD.nii.gz', dtype='float32')
    # RD
    nii.data = tenfit.rd
    nii.save(f'{prefix}RD.nii.gz', dtype='float32')
    # AD
    nii.data = tenfit.ad
    nii.save(f'{prefix}AD.nii.gz', dtype='float32')
    if evecs:
        data_evecs = tenfit.evecs
        data_evals = tenfit.evals
        # output 1st (V1), 2nd (V2) and 3rd (V3) eigenvectors as 4d data
        for idim in range(3):
            nii.data = data_evecs[:, :, :, :, idim]
            nii.save(f'{prefix}V{idim + 1}.nii.gz', dtype="float32")
            nii.data = data_evals[:, :, :, idim]
            nii.save(f'{prefix}E{idim + 1}.nii.gz', dtype="float32")

    return True


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
