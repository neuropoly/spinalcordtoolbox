#!/usr/bin/env python
#
# This program straightens the spinal cord of an anatomic image, apply a smoothing in the z dimension and apply
# the inverse warping field to get back the curved spinal cord but smoothed.
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

# TODO: maybe no need to convert RPI at the beginning because strainghten spinal cord already does it!

import sys
import os
import time
from typing import Sequence
import textwrap

import numpy as np

from spinalcordtoolbox.image import Image, generate_output_file, convert, add_suffix
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, list_type, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel, __version__
from spinalcordtoolbox.utils.fs import tmp_create, cache_save, cache_signature, cache_valid, copy, \
    extract_fname, rmtree
from spinalcordtoolbox.math import smooth

from spinalcordtoolbox.scripts import sct_apply_transfo, sct_straighten_spinalcord


class Param:
    # The constructor
    def __init__(self):
        self.algo_fitting = 'bspline'  # Fitting algorithm for centerline. See sct_straighten_spinalcord.

    # update constructor with user's parameters
    def update(self, param_user):
        # list_objects = param_user.split(',')
        for object in param_user:
            if len(object) < 2:
                printv('ERROR: Wrong usage.', 1, type='error')
            obj = object.split('=')
            setattr(self, obj[0], obj[1])


# PARSER
# ==========================================================================================
def get_parser():
    # initialize default parameters
    param_default = Param()

    # Initialize the parser
    parser = SCTArgumentParser(
        description=textwrap.dedent("""
            Smooth the spinal cord along its centerline. Steps are:

              1. Spinal cord is straightened (using centerline),
              2. a Gaussian kernel is applied in the superior-inferior direction,
              3. then cord is de-straightened as originally.
        """),
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        help="Image to smooth. Example: `data.nii.gz`"
    )
    mandatory.add_argument(
        '-s',
        metavar=Metavar.file,
        help="Spinal cord centerline or segmentation. Example: `data_centerline.nii.gz`"
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-smooth',
        metavar=Metavar.list,
        type=list_type(',', float),
        default=[0, 0, 3],
        help="Sigma (standard deviation) of the smoothing Gaussian kernel (in mm). For isotropic smoothing you only "
             "need to specify a value (e.g. `2`). For anisotropic smoothing specify a value for each axis, separated "
             "with a comma. The order should follow axes Right-Left, Antero-Posterior, Superior-Inferior "
             "(e.g.: `1,1,3`). For no smoothing, set value to `0`."
    )
    optional.add_argument(
        '-algo-fitting',
        metavar=Metavar.str,
        choices=['bspline', 'polyfit'],
        default=param_default.algo_fitting,
        help="Algorithm for curve fitting. For more information, see `sct_straighten_spinalcord`."
    )
    optional.add_argument(
        "-o",
        metavar=Metavar.file,
        help="Output filename. Example: `smooth_sc.nii.gz`. If not provided, the suffix `_smooth` will be added to the input file name."),

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_tempfile_args()

    return parser


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    # Initialization
    param = Param()
    start_time = time.time()

    fname_anat = arguments.i
    fname_centerline = arguments.s
    param.algo_fitting = arguments.algo_fitting

    if arguments.smooth is not None:
        sigmas = arguments.smooth
    remove_temp_files = arguments.r
    if arguments.o is not None:
        fname_out = arguments.o
    else:
        fname_out = f'{extract_fname(fname_anat)[1]}_smooth.nii.gz'

    # Display arguments
    printv('\nCheck input arguments...')
    printv(f'  Volume to smooth .................. {fname_anat}')
    printv(f'  Centerline ........................ {fname_centerline}')
    printv(f'  Sigma (mm) ........................ {sigmas}')
    printv(f'  Verbose ........................... {verbose}')

    # Check that input is 3D:
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_anat).dim
    dim = 4  # by default, will be adjusted later
    if nt == 1:
        dim = 3
    if nz == 1:
        dim = 2
    if dim == 4:
        printv('WARNING: the input image is 4D, please split your image to 3D before smoothing spinalcord using :\n'
               f'sct_image -i {fname_anat} -split t -o {fname_anat}', verbose, 'warning')
        printv('4D images not supported, aborting ...', verbose, 'error')

    # Extract path/file/extension
    path_anat, file_anat, ext_anat = extract_fname(fname_anat)
    path_centerline, file_centerline, ext_centerline = extract_fname(fname_centerline)

    path_tmp = tmp_create(basename="smooth-spinalcord")

    # Copying input data to tmp folder
    printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    copy(fname_anat, os.path.join(path_tmp, f'anat{ext_anat}'))
    copy(fname_centerline, os.path.join(path_tmp, f'centerline{ext_centerline}'))

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # convert to nii format
    im_anat = convert(Image(f'anat{ext_anat}'))
    im_anat.save('anat.nii', mutable=True, verbose=verbose)
    im_centerline = convert(Image(f'centerline{ext_centerline}'))
    im_centerline.save('centerline.nii', mutable=True, verbose=verbose)

    # Change orientation of the input image into RPI
    printv('\nOrient input volume to RPI orientation...')

    img_anat_rpi = Image("anat.nii").change_orientation("RPI")
    fname_anat_rpi = add_suffix(img_anat_rpi.absolutepath, "_rpi")
    img_anat_rpi.save(path=fname_anat_rpi, mutable=True)

    # Change orientation of the input image into RPI
    printv('\nOrient centerline to RPI orientation...')

    img_centerline_rpi = Image("centerline.nii").change_orientation("RPI")
    fname_centerline_rpi = add_suffix(img_centerline_rpi.absolutepath, "_rpi")
    img_centerline_rpi.save(path=fname_centerline_rpi, mutable=True)

    # Straighten the spinal cord
    # straighten segmentation
    printv('\nStraighten the spinal cord using centerline/segmentation...', verbose)
    cache_sig = cache_signature(
        input_files=[fname_anat_rpi, fname_centerline_rpi],
        input_params={"x": "spline", "version": __version__},
    )
    cachefile = os.path.join(curdir, "straightening.cache")
    if (
        cache_valid(cachefile, cache_sig)
        and os.path.isfile(os.path.join(curdir, 'warp_curve2straight.nii.gz'))
        and os.path.isfile(os.path.join(curdir, 'warp_straight2curve.nii.gz'))
        and os.path.isfile(os.path.join(curdir, 'straight_ref.nii.gz'))
    ):
        # if they exist, copy them into current folder
        printv('Reusing existing warping field which seems to be valid', verbose, 'warning')
        copy(os.path.join(curdir, 'warp_curve2straight.nii.gz'), 'warp_curve2straight.nii.gz')
        copy(os.path.join(curdir, 'warp_straight2curve.nii.gz'), 'warp_straight2curve.nii.gz')
        copy(os.path.join(curdir, 'straight_ref.nii.gz'), 'straight_ref.nii.gz')
        # apply straightening
        sct_apply_transfo.main(['-i', fname_anat_rpi, '-w', 'warp_curve2straight.nii.gz', '-d', 'straight_ref.nii.gz', '-o', 'anat_rpi_straight.nii', '-x', 'spline', '-v', '0'])
    else:
        sct_straighten_spinalcord.main([
            '-i', fname_anat_rpi,
            '-o', 'anat_rpi_straight.nii',
            '-s', fname_centerline_rpi,
            '-x', 'spline',
            '-param', f'algo_fitting={param.algo_fitting}',
            '-v', '0',
        ])
        cache_save(cachefile, cache_sig)
        # move warping fields and straight reference file from the tmpdir to the localdir (to use caching next time)
        copy('straight_ref.nii.gz', os.path.join(curdir, 'straight_ref.nii.gz'))
        copy('warp_curve2straight.nii.gz', os.path.join(curdir, 'warp_curve2straight.nii.gz'))
        copy('warp_straight2curve.nii.gz', os.path.join(curdir, 'warp_straight2curve.nii.gz'))

    # Smooth the straightened image along z
    printv('\nSmooth the straightened image...')

    img = Image("anat_rpi_straight.nii")
    out = img.copy()

    if len(sigmas) == 1:
        sigmas = [sigmas[0] for i in range(len(img.data.shape))]
    elif len(sigmas) != len(img.data.shape):
        raise ValueError("-smooth need the same number of inputs as the number of image dimension OR only one input")

    sigmas = [sigmas[i] / img.dim[i + 4] for i in range(3)]
    out.data = smooth(out.data, sigmas)
    out.save(path="anat_rpi_straight_smooth.nii")

    # Apply the reversed warping field to get back the curved spinal cord
    printv('\nApply the reversed warping field to get back the curved spinal cord...')
    sct_apply_transfo.main([
        '-i', 'anat_rpi_straight_smooth.nii',
        '-o', 'anat_rpi_straight_smooth_curved.nii',
        '-d', 'anat.nii',
        '-w', 'warp_straight2curve.nii.gz',
        '-x', 'spline',
        '-v', '0',
    ])

    # replace zeroed voxels by original image (issue #937)
    printv('\nReplace zeroed voxels by original image...', verbose)
    nii_smooth = Image('anat_rpi_straight_smooth_curved.nii')
    data_smooth = nii_smooth.data
    data_input = Image('anat.nii').data
    indzero = np.where(data_smooth == 0)
    data_smooth[indzero] = data_input[indzero]
    nii_smooth.data = data_smooth
    nii_smooth.save('anat_rpi_straight_smooth_curved_nonzero.nii')

    # come back
    os.chdir(curdir)

    # Generate output file
    printv('\nGenerate output file...')
    generate_output_file(os.path.join(path_tmp, "anat_rpi_straight_smooth_curved_nonzero.nii"), fname_out)

    # Remove temporary files
    if remove_temp_files == 1:
        printv('\nRemove temporary files...')
        rmtree(path_tmp)

    # Display elapsed time
    elapsed_time = time.time() - start_time
    printv(f'\nFinished! Elapsed time: {int(np.round(elapsed_time))}s\n')

    display_viewer_syntax([fname_anat, fname_out], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
