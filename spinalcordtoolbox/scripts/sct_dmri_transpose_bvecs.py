#!/usr/bin/env python
#
# Transpose bvecs file (if necessary) to get nx3 structure
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
from typing import Sequence

from spinalcordtoolbox.utils.fs import extract_fname
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser


def get_parser():
    parser = SCTArgumentParser(
        description='Transpose bvecs file (if necessary) to get nx3 structure.'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-bvec',
        metavar=Metavar.file,
        help="Input bvecs file. Example: `bvecs.txt`"
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        default='',
        help="Output bvecs file. By default, input file is overwritten. Example: `bvecs_t.txt`"
    )

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

    fname_in = arguments.bvec
    fname_out = arguments.o

    # get bvecs in proper orientation
    from dipy.io import read_bvals_bvecs
    bvals, bvecs = read_bvals_bvecs(None, fname_in)

    # # Transpose bvecs
    # printv('Transpose bvecs...', verbose)
    # # from numpy import transpose
    # bvecs = bvecs.transpose()

    # Write new file
    if fname_out == '':
        path_in, file_in, ext_in = extract_fname(fname_in)
        fname_out = path_in + file_in + ext_in
    fid = open(fname_out, 'w')
    for iLine in range(bvecs.shape[0]):
        fid.write(f'{" ".join(str(i) for i in bvecs[iLine, :])}\n')
    fid.close()

    # display message
    printv(f'Created file:\n--> {fname_out}\n', verbose, 'info')


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
