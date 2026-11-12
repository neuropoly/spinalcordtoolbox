#!/usr/bin/env python
#
# Merge b=0 and dMRI data and output appropriate bvecs/bvals files
#
# Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE


import sys
from typing import Sequence
import textwrap

import numpy as np

from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel, LazyLoader
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser
from spinalcordtoolbox.image import Image, concat_data

fetcher = LazyLoader("fetcher", globals(), "dipy.data.fetcher")


def get_parser():
    parser = SCTArgumentParser(
        description=textwrap.dedent("""
            Concatenate b=0 scans with DWI time series and update the bvecs and bvals files.

            Example 1: Add two b=0 file at the beginning and one at the end of the DWI time series:

            ```
            sct_dmri_concat_b0_and_dwi -i b0-1.nii b0-2.nii dmri.nii b0-65.nii -bvec bvecs.txt -bval bvals.txt -order b0 b0 dwi b0 -o dmri_concat.nii -obval bvals_concat.txt -obvec bvecs_concat.txt
            ```

            Example 2: Concatenate two DWI series and add one b=0 file at the beginning:

            ```
            sct_dmri_concat_b0_and_dwi -i b0-1.nii dmri1.nii dmri2.nii -bvec bvecs1.txt bvecs2.txt -bval bvals1.txt bvals2.txt -order b0 dwi dwi -o dmri_concat.nii -obval bvals_concat.txt -obvec bvecs_concat.txt
            ```
        """),  # noqa: E501 (line too long)
    )
    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        nargs='+',
        help="Input 4d files, separated by space, listed in the right order of concatenation. Example: `b0.nii dmri1.nii dmri2.nii`",
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        '-bval',
        nargs='+',
        help="Bvals file(s). Example: `bvals.txt`",
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        '-bvec',
        nargs='+',
        help="Bvecs file(s). Example: `bvecs.txt`",
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        '-order',
        nargs='+',
        help="Order of b=0 and DWI files entered in flag `-i`, separated by space. Example: `b0 dwi dwi`",
        choices=['b0', 'dwi'],
    )
    mandatory.add_argument(
        '-o',
        help="Output 4d concatenated file. Example: `b0_dmri_concat.nii`",
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        '-obval',
        help="Output concatenated bval file. Example: `bval_concat.txt`",
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        '-obvec',
        help="Output concatenated bvec file. Example: `bvec_concat.txt`",
        metavar=Metavar.file,
    )

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


def main(argv: Sequence[str]):
    """
    Main function
    :param argv:
    :return:
    """
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    # check number of input args
    if not len(arguments.i) == len(arguments.order):
        raise Exception("Number of items between flags '-i' and '-order' should be the same.")
    if not len(arguments.bval) == len(arguments.bvec):
        raise Exception("Number of files for bval and bvec should be the same.")

    # Concatenate NIFTI files
    im_list = [Image(fname) for fname in arguments.i]
    im_concat = concat_data(im_list, dim=3, squeeze_data=False)
    im_concat.save(arguments.o)
    printv(f"Generated file: {arguments.o}")

    # Concatenate bvals and bvecs
    bvals_all = []
    bvecs_all = [[], [], []]
    i_dwi = 0  # counter for DWI files, to read in bvec/bval files
    for ftype, fname in zip(arguments.order, arguments.i, strict=True):
        if ftype == 'b0':
            # count number of b=0
            n_b0 = Image(fname).dim[3]
            bvals = np.zeros(n_b0)
            bvecs = np.zeros((n_b0, 3))
        else:
            assert ftype == 'dwi'  # enforced by `choices=['b0', 'dwi']` in `get_parser()`
            # read bval/bvec files
            bvals, bvecs = fetcher.read_bvals_bvecs(arguments.bval[i_dwi], arguments.bvec[i_dwi])
            i_dwi += 1
        # Concatenate bvals
        bvals_all.extend(str(n) for n in bvals)
        # Concatenate bvecs
        for i in range(3):
            bvecs_all[i].extend(f'{n:.16f}' for n in bvecs[:, i])
    bvals_concat = ' '.join(bvals_all)
    bvecs_concat = '\n'.join(' '.join(v) for v in bvecs_all)

    # Write files
    new_f = open(arguments.obval, 'w')
    new_f.write(bvals_concat)
    new_f.close()
    printv(f"Generated file: {arguments.obval}")
    new_f = open(arguments.obvec, 'w')
    new_f.write(bvecs_concat)
    new_f.close()
    printv(f"Generated file: {arguments.obvec}")


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
