#!/usr/bin/env python
#
# Concatenate bvecs files in time.
#
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
from typing import Sequence

from spinalcordtoolbox.utils.fs import extract_fname
from spinalcordtoolbox.utils.sys import init_sct, set_loglevel, LazyLoader
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser

fetcher = LazyLoader("fetcher", globals(), "dipy.data.fetcher")


def get_parser():
    parser = SCTArgumentParser(
        description='Concatenate bvec files in time. You can either use bvecs in lines or columns. '
                    'N.B.: Return bvecs in lines. If you need it in columns, please use '
                    'sct_dmri_transpose_bvecs afterwards.'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        "-i",
        nargs='+',
        help='List of the bvec files to concatenate. Example: `dmri_b700.bvec dmri_b2000.bvec`',
        metavar=Metavar.file,
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-o",
        metavar=Metavar.file,
        help='Output file with bvecs concatenated. Example: `dmri_b700_b2000_concat.bvec`')

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

    fname_bvecs_list = arguments.i
    # Build fname_out
    if arguments.o is not None:
        fname_out = arguments.o
    else:
        path_in, file_in, ext_in = extract_fname(fname_bvecs_list[0])
        fname_out = f'{path_in}bvecs_concat{ext_in}'

    # Open bvec files and collect values
    bvecs_all = [[], [], []]
    for i_fname in fname_bvecs_list:
        _, bvecs = fetcher.read_bvals_bvecs(None, i_fname)
        for i in range(3):
            bvecs_all[i].extend(f'{n:.16f}' for n in bvecs[:, i])

    # Concatenate
    bvecs_concat = '\n'.join(' '.join(v) for v in bvecs_all)

    # Write new bvec
    new_f = open(fname_out, 'w')
    new_f.write(bvecs_concat)
    new_f.close()


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
