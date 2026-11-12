#!/usr/bin/env python
#
# Register a volume (e.g., EPI from fMRI or DTI scan) to an anatomical image.
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

# TODO: if user specified -param, then ignore the default paramreg
# TODO: check syn with shrink=4
# TODO: output name file for warp using "src" and "dest" file name, i.e. warp_filesrc2filedest.nii.gz
# TODO: testing script for all cases
# TODO: add following feature:
# -r of isct_antsRegistration at the initial step (step 0).
# -r [' dest ',' src ',0] --> align the geometric center of the two images
# -r [' dest ',' src ',1] --> align the maximum intensities of the two images I use that quite often...
# TODO: output reg for ants2d and centermass (2016-02-25)

# Note for the developer: DO NOT use --collapse-output-transforms 1, otherwise inverse warping field is not output

# TODO: make three possibilities:
# - one-step registration, using only image registration (by sliceReg or antsRegistration)
# - two-step registration, using first segmentation-based registration (based on sliceReg or antsRegistration) and
# second the image registration (and allow the choice of algo, metric, etc.)
# - two-step registration, using only segmentation-based registration

import sys
import os
import time
from copy import deepcopy
from typing import Sequence
import textwrap

import numpy as np

from spinalcordtoolbox.reports import qc2
from spinalcordtoolbox.registration.algorithms import Paramreg, ParamregMultiStep
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, list_type, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.image import check_dim, Image

from spinalcordtoolbox.registration.core import register_wrapper

# Default registration parameters
step0 = Paramreg(step='0', type='im', algo='syn', metric='MI', iter='0', shrink='1', smooth='0', gradStep='0.5',
                 slicewise='0', dof='Tx_Ty_Tz_Rx_Ry_Rz')  # only used to put src into dest space
step1 = Paramreg(step='1', type='im')
DEFAULT_PARAMREGMULTI = ParamregMultiStep([step0, step1])


def get_parser():
    # Initialize the parser
    parser = SCTArgumentParser(
        description=textwrap.dedent("""
                    This program co-registers two 3D volumes. The deformation is non-rigid and is constrained along Z direction (i.e., axial plane). Hence, this function assumes that orientation of the destination image is axial (RPI). If you need to register two volumes with large deformations and/or different contrasts, it is recommended to input spinal cord segmentations (binary mask) in order to achieve maximum robustness. The program outputs a warping field that can be used to register other images to the destination image. To apply the warping field to another image, use `sct_apply_transfo`

                    Tips:

                     - For a registration step using segmentations, use the MeanSquares metric. Also, simple algorithm will be very efficient, for example centermass as a 'preregistration'.
                     - For a registration step using images of different contrast, use the Mutual Information (MI) metric.
                     - Combine the steps by increasing the complexity of the transformation performed in each step, for example:

                       ```
                       -param step=1,type=seg,algo=slicereg,metric=MeanSquares:
                              step=2,type=seg,algo=affine,metric=MeanSquares,gradStep=0.2:
                              step=3,type=im,algo=syn,metric=MI,iter=5,shrink=2
                       ```
                     - When image contrast is low, a good option is to perform registration only based on the image segmentation, i.e. using type=seg
                     - Columnwise algorithm needs to be applied after a translation and rotation such as centermassrot algorithm. For example:

                      ```
                      -param step=1,type=seg,algo=centermassrot,metric=MeanSquares:
                             step=2,type=seg,algo=columnwise,metric=MeanSquares
                      ```
        """),  # noqa: E501 (line too long)
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        help="Image source. Example: `src.nii.gz`"
    )
    mandatory.add_argument(
        '-d',
        metavar=Metavar.file,
        help="Image destination. Example: `dest.nii.gz`"
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-iseg',
        metavar=Metavar.file,
        help="Segmentation source. Example: `src_seg.nii.gz`"
    )
    optional.add_argument(
        '-dseg',
        metavar=Metavar.file,
        help="Segmentation destination. Example: `dest_seg.nii.gz`"
    )
    optional.add_argument(
        '-ilabel',
        metavar=Metavar.file,
        help="Labels source."
    )
    optional.add_argument(
        '-dlabel',
        metavar=Metavar.file,
        help="Labels destination."
    )
    optional.add_argument(
        '-initwarp',
        metavar=Metavar.file,
        help="Initial warping field to apply to the source image."
    )
    optional.add_argument(
        '-initwarpinv',
        metavar=Metavar.file,
        help="Initial inverse warping field to apply to the destination image (only use if you wish to generate the "
             "dest->src warping field)"
    )
    optional.add_argument(
        '-m',
        metavar=Metavar.file,
        help="Mask that can be created with sct_create_mask to improve accuracy over region of interest. This mask "
             "will be used on the destination image. Masks will be binarized at 0.5. Example: `mask.nii.gz`"
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help="Name of output file. Example: `src_reg.nii.gz`"
    )
    optional.add_argument(
        '-owarp',
        metavar=Metavar.file,
        help="Name of output forward warping field."
    )
    optional.add_argument(
        '-owarpinv',
        metavar=Metavar.file,
        help="Name of output inverse warping field."
    )
    optional.add_argument(
        '-param',
        metavar=Metavar.list,
        type=list_type(':', str),
        help=(f"Parameters for registration. Separate arguments with `,`. Separate steps with `:`.\n"
              f"Example: step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=im,algo=syn,metric=MI,iter=5,"
              f"shrink=2\n"
              f"  - step: <int> Step number (starts at 1, except for type=label).\n"
              f"  - type: {{im, seg, imseg, label}} type of data used for registration. If you specify 'im', you must also "
              f"provide arguments -i and -d. If you specify 'seg', you must provide -iseg and -dseg. If you specify imseg, "
              f"you must provide all four arguments. If you specify -label, you must provide -ilabel and -dlabel. ((Note: "
              f"Use type=label only at step=0. Use type=imseg only for algo=centermassrot along with rot_method=hog or "
              f"rot_method=pca_hog.))\n"
              f"  - algo: The algorithm used to compute the transformation. Default={DEFAULT_PARAMREGMULTI.steps['1'].algo}\n"
              f"    * translation: translation in X-Y plane (2dof)\n"
              f"    * rigid: translation + rotation in X-Y plane (4dof)\n"
              f"    * affine: translation + rotation + scaling in X-Y plane (6dof)\n"
              f"    * syn: non-linear symmetric normalization\n"
              f"    * bsplinesyn: syn regularized with b-splines\n"
              f"    * slicereg: regularized translations (see: goo.gl/Sj3ZeU)\n"
              f"    * centermass: slicewise center of mass alignment (seg only).\n"
              f"    * centermassrot: slicewise center of mass and rotation alignment using method specified in "
              f"'rot_method'\n"
              f"    * columnwise: R-L scaling followed by A-P columnwise alignment (seg only).\n"
              f"    * dl: Contrast-agnostic, deep learning-based registration based on the SynthMorph architecture. "
              f"Can be run using: -param step=1,type=im,algo=dl\n"
              f"  - slicewise: <int> Slice-by-slice 2d transformation. "
              f"Default={DEFAULT_PARAMREGMULTI.steps['1'].slicewise}.\n"
              f"  - metric: {{CC, MI, MeanSquares}}. Default={DEFAULT_PARAMREGMULTI.steps['1'].metric}.\n"
              f"    * CC: The cross correlation metric compares the images based on their intensities but with a small "
              f"normalization. It can be used with images with the same contrast (for ex. T2-w with T2-w). In this "
              f"case it is very efficient but the computation time can be very long.\n"
              f"    * MI: the mutual information metric compares the images based on their entropy, therefore the "
              f"images need to be big enough to have enough information. It works well for images with different "
              f"contrasts (for example T2-w with T1-w) but not on segmentations.\n"
              f"    * MeanSquares: The mean squares metric compares the images based on their intensities. It can be "
              f"used only with images that have exactly the same contrast (with the same intensity range) or with "
              f"segmentations.\n"
              f"  - iter: <int> Number of iterations. Default={DEFAULT_PARAMREGMULTI.steps['1'].iter}.\n"
              f"  - shrink: <int> Shrink factor. A shrink factor of 2 will down sample the images by a factor of 2 to "
              f"do the registration, and thus allow bigger deformations (and be faster to compute). It is usually "
              f"combined with a smoothing. (only for syn/bsplinesyn). Default={DEFAULT_PARAMREGMULTI.steps['1'].shrink}.\n"
              f"  - smooth: <int> Smooth factor (in mm). Note: if algo={{centermassrot,columnwise}} the smoothing "
              f"kernel is: SxSx0. Otherwise it is SxSxS. Default={DEFAULT_PARAMREGMULTI.steps['1'].smooth}.\n"
              f"  - laplacian: <int> Laplace filter using Gaussian second derivatives, applied before registration. "
              f"The input number correspond to the standard deviation of the Gaussian filter. "
              f"Default={DEFAULT_PARAMREGMULTI.steps['1'].laplacian}.\n"
              f"  - gradStep: <float> The gradient step used by the function opitmizer. A small gradient step can lead "
              f"to a more accurate registration but will take longer to compute, with the risk to not reach "
              f"convergence. A bigger gradient step will make the registration faster but the result can be far from "
              f"an optimum. Default={DEFAULT_PARAMREGMULTI.steps['1'].gradStep}.\n"
              f"  - deformation: ?x?x?: Restrict deformation (for ANTs algo). Replace ? by 0 (no deformation) or 1 "
              f"(deformation). Default={DEFAULT_PARAMREGMULTI.steps['1'].deformation}.\n"
              f"  - init: Initial translation alignment based on:\n"
              f"    * geometric: Geometric center of images\n"
              f"    * centermass: Center of mass of images\n"
              f"    * origin: Physical origin of images\n"
              f"  - poly: <int> Polynomial degree of regularization (only for algo=slicereg). "
              f"Default={DEFAULT_PARAMREGMULTI.steps['1'].poly}.\n"
              f"  - filter_size: <float> Filter size for regularization (only for algo=centermassrot). "
              f"Default={DEFAULT_PARAMREGMULTI.steps['1'].filter_size}.\n"
              f"  - smoothWarpXY: <int> Smooth XY warping field (only for algo=columnwize). "
              f"Default={DEFAULT_PARAMREGMULTI.steps['1'].smoothWarpXY}.\n"
              f"  - pca_eigenratio_th: <int> Min ratio between the two eigenvalues for PCA-based angular adjustment "
              f"(only for algo=centermassrot and rot_method=pca). "
              f"Default={DEFAULT_PARAMREGMULTI.steps['1'].pca_eigenratio_th}.\n"
              f"  - dof: <str> Degree of freedom for type=label. Separate with `_`. "
              f"Default={DEFAULT_PARAMREGMULTI.steps['0'].dof}. T stands for translation, R stands for rotation, and S "
              f"stands for scaling. x, y, and z indicate the direction. Examples:\n"
              f"    * Tx_Ty_Tz_Rx_Ry_Rz would allow translation on x, y and z axes and rotation on x, y and z axes\n"
              f"    * Tx_Ty_Tz_Sz would allow translation on x, y and z axes and scaling only on z axis\n"
              f"  - rot_method {{pca, hog, pcahog}}: rotation method to be used with algo=centermassrot. If using hog "
              f"or pcahog, type should be set to imseg. Default={DEFAULT_PARAMREGMULTI.steps['1'].rot_method}\n"
              f"    * pca: approximate cord segmentation by an ellipse and finds it orientation using PCA's "
              f"eigenvectors (use with `type=seg`)\n"
              f"    * hog: finds the orientation using the symmetry of the image (use with `type=imseg`)\n"
              f"    * pcahog: tries method pca and if it fails, uses method hog (use with `type=imseg`).\n")
    )
    optional.add_argument(
        '-identity',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Supplying this option will skip registration optimization (e.g. translations, rotations, deformations) "
             "and will only rely on the qform (from the NIfTI header) of the source and destination images. Use this "
             "option if you wish to put the source image into the space of the destination image (i.e. match "
             "dimension, resolution and orientation)."
    )
    optional.add_argument(
        '-z',
        metavar=Metavar.int,
        type=int,
        default=Param().padding,
        help="Size of z-padding to enable deformation at edges when using SyN."
    )
    optional.add_argument(
        '-x',
        choices=['nn', 'linear', 'spline', 'label'],
        default='linear',
        help="Interpolation method to use when applying the final warping field to the source/dest images. "
             "(Refer to the help text for `sct_apply_transfo -x` for more details)."
    )
    optional.add_argument(
        '-ofolder',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="Output folder. Example: `reg_results`"
    )
    optional.add_argument(
        '-qc',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="The path where the quality control generated content will be saved. Note: This flag requires the `-dseg` "
             "flag."
    )
    optional.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the dataset the process was run on."
    )
    optional.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the subject the process was run on."
    )
    optional.add_argument(
        '-qc-contrast',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the contrast the process was run on."
    )

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_tempfile_args()

    return parser


# DEFAULT PARAMETERS

class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.outSuffix = "_reg"
        self.padding = 5
        self.remove_temp_files = 1


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    # initialize parameters
    param = Param()

    # Initialization
    fname_output = ''
    path_out = ''
    fname_src_seg = ''
    fname_dest_seg = ''
    fname_src_label = ''
    fname_dest_label = ''

    start_time = time.time()

    # get arguments
    fname_src = arguments.i
    fname_dest = arguments.d
    if arguments.iseg is not None:
        fname_src_seg = arguments.iseg
    if arguments.dseg is not None:
        fname_dest_seg = arguments.dseg
    if arguments.ilabel is not None:
        fname_src_label = arguments.ilabel
    if arguments.dlabel is not None:
        fname_dest_label = arguments.dlabel
    if arguments.o is not None:
        fname_output = arguments.o
    if arguments.ofolder is not None:
        path_out = arguments.ofolder
    if arguments.owarp is not None:
        fname_output_warp = arguments.owarp
    else:
        fname_output_warp = ''
    if arguments.owarpinv is not None:
        fname_output_warpinv = arguments.owarpinv
    else:
        fname_output_warpinv = ''
    if arguments.initwarp is not None:
        fname_initwarp = os.path.abspath(arguments.initwarp)
    else:
        fname_initwarp = ''
    if arguments.initwarpinv is not None:
        fname_initwarpinv = os.path.abspath(arguments.initwarpinv)
    else:
        fname_initwarpinv = ''
    if arguments.m is not None:
        fname_mask = arguments.m
    else:
        fname_mask = ''
    padding = arguments.z
    paramregmulti = deepcopy(DEFAULT_PARAMREGMULTI)
    if arguments.param is not None:
        paramregmulti_user = arguments.param
        # update registration parameters
        for paramStep in paramregmulti_user:
            paramregmulti.addStep(paramStep)
    # Raise error if arguments.qc is provided without arguments.dseg
    if arguments.qc is not None and fname_dest_seg == '':
        parser.error("The argument '-qc' requires the argument '-dseg'.")

    identity = arguments.identity
    interp = arguments.x
    remove_temp_files = arguments.r

    # printv(arguments)
    printv('\nInput parameters:')
    printv(f'  Source .............. {fname_src} {Image(fname_src).data.shape}')
    printv(f'  Destination ......... {fname_dest} {Image(fname_dest).data.shape}')
    printv(f'  Init transfo ........ {fname_initwarp}')
    printv(f'  Mask ................ {fname_mask}')
    printv(f'  Output name ......... {fname_output}')
    # printv(f'  Algorithm ........... {paramregmulti.algo}')
    # printv(f'  Number of iterations  {paramregmulti.iter}')
    # printv(f'  Metric .............. {paramregmulti.metric}')
    printv(f'  Remove temp files ... {remove_temp_files}')
    printv(f'  Verbose ............. {verbose}')

    # update param
    param.verbose = verbose
    param.padding = padding
    param.fname_mask = fname_mask
    param.remove_temp_files = remove_temp_files

    # Get if input is 3D
    printv('\nCheck if input data are 3D...', verbose)
    check_dim(fname_src, dim_lst=[3])
    check_dim(fname_dest, dim_lst=[3])

    # Check if user selected type=seg, but did not input segmentation data
    if 'paramregmulti_user' in locals():
        if True in ['type=seg' in paramregmulti_user[i] for i in range(len(paramregmulti_user))]:
            if fname_src_seg == '' or fname_dest_seg == '':
                parser.error("If you select 'type=seg' you must specify '-iseg' and '-dseg' arguments.")

    # Put source into destination space using header (no estimation -- purely based on header)
    # TODO: Check if necessary to do that
    # TODO: use that as step=0
    # printv('\nPut source into destination space using header...', verbose)
    # run_proc('isct_antsRegistration -d 3 -t Translation[0] -m MI[dest_pad.nii,src.nii,1,16] -c 0 -f 1 -s 0 -o
    # [regAffine,src_regAffine.nii] -n BSpline[3]', verbose)
    # if segmentation, also do it for seg

    fname_src2dest, fname_dest2src, _, _ = \
        register_wrapper(fname_src, fname_dest, param, paramregmulti, fname_src_seg=fname_src_seg,
                         fname_dest_seg=fname_dest_seg, fname_src_label=fname_src_label,
                         fname_dest_label=fname_dest_label, fname_mask=fname_mask, fname_initwarp=fname_initwarp,
                         fname_initwarpinv=fname_initwarpinv, identity=identity, interp=interp,
                         fname_output=fname_output,
                         fname_output_warp=fname_output_warp, fname_output_warpinv=fname_output_warpinv,
                         path_out=path_out)

    # display elapsed time
    elapsed_time = time.time() - start_time
    printv(f'\nFinished! Elapsed time: {int(np.round(elapsed_time))}s', verbose)

    if arguments.qc is not None:
        qc2.sct_register(
            fname_input=fname_src2dest,
            fname_output=fname_dest,
            fname_seg=fname_dest_seg,
            command='sct_register_multimodal',
            argv=argv,
            path_qc=os.path.abspath(arguments.qc),
            dataset=arguments.qc_dataset,
            subject=arguments.qc_subject,
            contrast=arguments.qc_contrast,
        )

    # If dest wasn't registered (e.g. unidirectional registration due to '-initwarp'), then don't output syntax
    if fname_dest2src:
        display_viewer_syntax([fname_src, fname_dest2src], verbose=verbose)
    display_viewer_syntax([fname_dest, fname_src2dest], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
