#!/usr/bin/env python
#
# Perform operations on images
#
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import os
import sys
from typing import Sequence
import textwrap

import numpy as np

from spinalcordtoolbox.scripts import sct_apply_transfo, sct_resample
from spinalcordtoolbox.image import (Image, concat_data, add_suffix, change_orientation, split_img_data, pad_image,
                                     create_formatted_header_string, HEADER_FORMATS,
                                     stitch_images, generate_stitched_qc_images)
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.utils.shell import (SCTArgumentParser, Metavar, display_viewer_syntax, ActionCreateFolder,
                                           list_type)
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel, LazyLoader
from spinalcordtoolbox.utils.fs import tmp_create, extract_fname, rmtree

nib = LazyLoader("nib", globals(), "nibabel")
nib_processing = LazyLoader("nib_processing", globals(), "nibabel.processing")

DIM_LIST = ['x', 'y', 'z', 't']


def get_parser():
    parser = SCTArgumentParser(
        description='Perform manipulations on images (e.g., pad, change space, split along dimension). '
                    'Inputs can be a number, a 4d image, or several 3d images separated with `,`'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        nargs='+',
        metavar=Metavar.file,
        help='Input file(s). Example: `data.nii.gz`\n'
             'Note: Only `-concat`, `-omc` or `-stitch` support multiple input files. In those cases, separate filenames using '
             'spaces. Example usage: `sct_image -i data1.nii.gz data2.nii.gz -concat`')

    optional = parser.optional_arggroup
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help='Output file. Example: `data_pad.nii.gz`')

    image = parser.add_argument_group('IMAGE OPERATIONS')
    image.add_argument(
        '-pad',
        metavar=Metavar.list,
        help='Pad 3D image. Specify padding as: `x,y,z` (in voxel). Example: `0,0,1`')
    image.add_argument(
        '-pad-asym',
        metavar=Metavar.list,
        help='Pad 3D image with asymmetric padding. Specify padding as: `x_i,x_f,y_i,y_f,z_i,z_f` (in voxel). '
             'Example: `0,0,5,10,1,1`')
    image.add_argument(
        '-split',
        help='Split data along the specified dimension. The suffix _DIM+NUMBER will be added to the intput file name.',
        choices=('x', 'y', 'z', 't'))
    image.add_argument(
        '-concat',
        help='Concatenate data along the specified dimension',
        choices=('x', 'y', 'z', 't'))
    image.add_argument(
        '-stitch',
        action='store_true',
        help='Stitch multiple images acquired in the same orientation utilizing '
             'the algorithm by Lavdas, Glocker et al. (https://doi.org/10.1016/j.crad.2019.01.012).'
             'The first image is used as the reference.'
             'Ensure to use the image with the largest dimensions first to avoid cropping of the images.'
             )
    image.add_argument(
        '-qc',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="The path where the quality control generated content will be saved. "
             "(Note: QC reporting is only available for `sct_image -stitch`)."
    )
    image.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the dataset the process was run on. "
             "(Note: QC reporting is only available for `sct_image -stitch`)."
    )
    image.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the subject the process was run on. '
             "(Note: QC reporting is only available for `sct_image -stitch`)."
    )
    image.add_argument(
        '-qc-contrast',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the contrast the process was run on. '
             "(Note: QC reporting is only available for `sct_image -stitch`)."
    )
    image.add_argument(
        '-remove-vol',
        metavar=Metavar.list,
        help='Remove specific volumes from a 4d volume. Separate with `,`. Example: `0,5,10`')
    image.add_argument(
        '-keep-vol',
        metavar=Metavar.list,
        help='Keep specific volumes from a 4d volume (remove others). Separate with `,`. Example: `1,2,3,11`')
    image.add_argument(
        '-type',
        help='Change file type',
        choices=('uint8', 'int16', 'int32', 'float32', 'complex64', 'float64', 'int8', 'uint16', 'uint32', 'int64', 'uint64'))

    header = parser.add_argument_group('HEADER OPERATIONS')
    header.add_argument(
        "-header",
        choices=HEADER_FORMATS,
        # 'const' and 'nargs' used because of https://stackoverflow.com/q/40324356
        const='sct',
        nargs='?',
        help="Print the header of a NIfTI file. You can select the output format of the header: 'sct' (default), 'nibabel' or 'fslhd'."
    )
    header.add_argument(
        '-copy-header',
        metavar=Metavar.file,
        help=textwrap.dedent("""
            Copy the NIfTI header of the source image (specified in `-i`) to the destination image (specified here) and save it into a new image (specified in `-o`).

            !! WARNING: This command should ONLY be run to fix a wrong header (e.g., where the qform and/or sform between an image and a mask of the image do not match). Also note that the image is NOT affected by this command, so if the dimensions of the source and destination images do not match, then you should probably NOT use this command.
        """),  # noqa: E501 (line too long)
        )
    affine_fixes = header.add_mutually_exclusive_group(required=False)
    affine_fixes.add_argument(
        '-set-sform-to-qform',
        help="Set the input image's sform matrix equal to its qform matrix. Use this option when you "
             "need to enforce matching sform and qform matrices. This option can be used by itself, or in combination "
             "with other functions.",
        action='store_true'
    )
    affine_fixes.add_argument(
        '-set-qform-to-sform',
        help="Set the input image's qform matrix equal to its sform matrix. Use this option when you "
             "need to enforce matching sform and qform matrices. This option can be used by itself, or in combination "
             "with other functions.",
        action='store_true'
    )

    orientation = parser.add_argument_group('ORIENTATION OPERATIONS')
    orientation.add_argument(
        '-getorient',
        help='Get orientation of the input image',
        action='store_true')
    orientation.add_argument(
        '-setorient',
        help='Set orientation of the input image (modifies BOTH the header and data array, similar to `fslswapdim`).',
        choices=[
            'LAS', 'LAI', 'LPS', 'LPI', 'LSA', 'LSP', 'LIA', 'LIP',
            'RAS', 'RAI', 'RPS', 'RPI', 'RSA', 'RSP', 'RIA', 'RIP',
            'ALS', 'ALI', 'ARS', 'ARI', 'ASL', 'ASR', 'AIL', 'AIR',
            'PLS', 'PLI', 'PRS', 'PRI', 'PSL', 'PSR', 'PIL', 'PIR',
            'SLA', 'SLP', 'SRA', 'SRP', 'SAL', 'SAR', 'SPL', 'SPR',
            'ILA', 'ILP', 'IRA', 'IRP', 'IAL', 'IAR', 'IPL', 'IPR',
        ])
    orientation.add_argument(
        '-flip',
        help=textwrap.dedent("""
            Flip an axis of the image's data array. (This will not change the header orientation string.)

              - WARNING: This option should only be used to fix the data array when it does not match the orientation string in the header. We recommend that you investigate and understand where the mismatch originated from in the first place before using this option.
              - Example: For an image with 'RPI' in its header, `-flip x` will flip the LR axis of the data array.
        """),  # noqa: E501 (line too long)
        choices=DIM_LIST)
    orientation.add_argument(
        '-transpose',
        metavar="ax1,ax2,ax3",
        help=textwrap.dedent("""
            Transpose the axes (x,y,z) of the image's data array. (This will not change the header orientation string.)

              - WARNING: This option should only be used to fix the data array when it does not match the orientation string in the header. We recommend that you investigate and understand where the mismatch originated from in the first place before using this option.
              - Example: For a 3D image with 'RPI' in its header, `-transpose z,y,x` will swap the LR and SI axes of the data array.
        """),  # noqa: E501 (line too long)
        type=list_type(',', str))

    multi = parser.add_argument_group('MULTI-COMPONENT OPERATIONS ON ITK COMPOSITE WARPING FIELDS')
    multi.add_argument(
        '-mcs',
        action='store_true',
        help='Multi-component split: Split ITK warping field into three separate displacement fields. '
             'The suffixes `_X`, `_Y` and `_Z` will be added to the input file name.')
    multi.add_argument(
        '-omc',
        action='store_true',
        help='Multi-component merge: Merge inputted images into one multi-component image. Requires several inputs.')

    warping = parser.add_argument_group('WARPING FIELD OPERATIONS')
    warping.add_argument(
        '-display-warp',
        action='store_true',
        help='Create a grid and deform it using provided warping field.')
    warping.add_argument(
        '-to-fsl',
        metavar=Metavar.file,
        help="Transform displacement field values to absolute FSL warps. To be used with FSL's applywarp function with the "
        "`--abs` flag. Input the file that will be used as the input (source) for applywarp and optionally the target "
        "(ref). The target file is necessary for the case where the warp is in a different space than the target. For "
        "example, the inverse warps generated by `sct_straighten_spinalcord`. This feature has not been extensively "
        "validated so consider checking the results of `applywarp` against `sct_apply_transfo` before using in FSL "
        "pipelines. Example syntax: `sct_image -i WARP_SRC2DEST -to-fsl IM_SRC (IM_DEST) -o WARP_FSL`, "
        "followed by FSL: `applywarp -i IM_SRC -r IM_DEST -w WARP_FSL --abs -o IM_SRC2DEST` ",
        nargs='*')

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

    # initializations
    output_type = None

    fname_in = arguments.i

    im_in_list = [Image(fname) for fname in fname_in]
    if len(im_in_list) > 1 and not arguments.concat and not arguments.omc and not arguments.stitch:
        parser.error("Multi-image input is only supported for the '-concat','-omc' and '-stitch' arguments.")

    # Apply initialization steps to all input images first
    if arguments.set_sform_to_qform:
        [im.set_sform_to_qform() for im in im_in_list]
    elif arguments.set_qform_to_sform:
        [im.set_qform_to_sform() for im in im_in_list]

    # Most sct_image options don't accept multi-image input, so here we simply separate out the first image
    # TODO: Extend the options so that they iterate through the list of images (to support multi-image input)
    im_in = im_in_list[0]

    if arguments.o is not None:
        fname_out = arguments.o
    else:
        # in case fname_out is not defined, use first element of input file name list
        fname_out = fname_in[0]

    # Run command
    # Arguments are sorted alphabetically (not according to the usage order)
    if arguments.concat is not None:
        dim = arguments.concat
        assert dim in DIM_LIST  # Enforced by add_argument(..., choices=...)
        dim = DIM_LIST.index(dim)
        im_out = [concat_data(im_in_list, dim)]

    elif arguments.copy_header is not None:
        if arguments.o is None:
            raise ValueError("Need to specify output image with -o!")
        im_dest = Image(arguments.copy_header)
        im_dest_new = im_in.copy()
        im_dest_new.data = im_dest.data.copy()
        # im_dest.header = im_in.header
        im_dest_new.absolutepath = im_dest.absolutepath
        im_out = [im_dest_new]

    elif arguments.display_warp:
        visualize_warp(im_warp=im_in, im_grid=None, step=3, rm_tmp=True)
        im_out = None

    elif arguments.getorient:
        orient = im_in.orientation
        im_out = None

    elif arguments.keep_vol is not None:
        index_vol = (arguments.keep_vol).split(',')
        for iindex_vol, vol in enumerate(index_vol):
            index_vol[iindex_vol] = int(vol)
        im_out = [remove_vol(im_in, index_vol, todo='keep')]

    elif arguments.mcs:
        if len(im_in.data.shape) != 5:
            printv(parser.error('ERROR: -mcs input need to be a multi-component image'))
        im_out = multicomponent_split(im_in)

    elif arguments.omc:
        im_ref = im_in_list[0]
        for im in im_in_list:
            if im.data.shape != im_ref.data.shape:
                printv(parser.error('ERROR: -omc inputs need to have all the same shapes'))
            del im
        im_out = [multicomponent_merge(im_in_list=im_in_list)]

    elif arguments.pad is not None:
        ndims = len(im_in.data.shape)
        if ndims != 3:
            printv('ERROR: you need to specify a 3D input file.', 1, 'error')
            return

        pad_arguments = arguments.pad.split(',')
        if len(pad_arguments) != 3:
            printv('ERROR: you need to specify 3 padding values.', 1, 'error')

        padx, pady, padz = pad_arguments
        padx, pady, padz = int(padx), int(pady), int(padz)
        im_out = [pad_image(im_in, pad_x_i=padx, pad_x_f=padx, pad_y_i=pady,
                            pad_y_f=pady, pad_z_i=padz, pad_z_f=padz)]

    elif arguments.pad_asym is not None:
        ndims = len(im_in.data.shape)
        if ndims != 3:
            printv('ERROR: you need to specify a 3D input file.', 1, 'error')
            return

        pad_arguments = arguments.pad_asym.split(',')
        if len(pad_arguments) != 6:
            printv('ERROR: you need to specify 6 padding values.', 1, 'error')

        padxi, padxf, padyi, padyf, padzi, padzf = pad_arguments
        padxi, padxf, padyi, padyf, padzi, padzf = int(padxi), int(padxf), int(padyi), int(padyf), int(padzi), int(padzf)
        im_out = [pad_image(im_in, pad_x_i=padxi, pad_x_f=padxf, pad_y_i=padyi, pad_y_f=padyf, pad_z_i=padzi, pad_z_f=padzf)]

    elif arguments.remove_vol is not None:
        index_vol = (arguments.remove_vol).split(',')
        for iindex_vol, vol in enumerate(index_vol):
            index_vol[iindex_vol] = int(vol)
        im_out = [remove_vol(im_in, index_vol, todo='remove')]

    elif arguments.setorient is not None:
        printv(im_in.absolutepath)
        im_out = [change_orientation(im_in, arguments.setorient)]

    elif arguments.flip is not None:
        if DIM_LIST.index(arguments.flip) >= len(im_in.data.shape):
            parser.error(f"Cannot flip axis '{arguments.flip}' on an image with n_dim = {len(im_in.data.shape)}.")
        im_out = im_in.copy()
        im_out.data = np.flip(im_out.data, axis=DIM_LIST.index(arguments.flip))
        im_out = [im_out]

    elif arguments.transpose is not None:
        if len(arguments.transpose) != len(im_in.data.shape):
            parser.error(f"Transpose argument '{','.join(arguments.transpose)}' must match number of image dimensions "
                         f"in the input image ({len(im_in.data.shape)}).")
        im_out = im_in.copy()
        transpose_numerical = [DIM_LIST.index(axis)  # Convert [x, y, z, t] to [0, 1, 2, 3]
                               for axis in arguments.transpose]
        im_out.data = np.transpose(im_out.data, axes=transpose_numerical)
        im_out = [im_out]

    elif arguments.header is not None:
        header = im_in.header
        # Necessary because of https://github.com/nipy/nibabel/issues/480#issuecomment-239227821
        im_file = nib.load(im_in.absolutepath)
        header.structarr['scl_slope'] = im_file.dataobj.slope
        header.structarr['scl_inter'] = im_file.dataobj.inter
        printv(create_formatted_header_string(header=header, output_format=arguments.header), verbose=verbose)
        im_out = None

    elif arguments.split is not None:
        dim = arguments.split
        assert dim in DIM_LIST  # Enforced by add_argument(..., choices=...)
        dim = DIM_LIST.index(dim)
        im_out = split_data(im_in, dim)

    elif arguments.stitch:
        im_out = [stitch_images(im_in_list)]

    elif arguments.type is not None:
        output_type = arguments.type
        im_out = [im_in]

    elif arguments.to_fsl is not None:
        space_files = arguments.to_fsl
        if len(space_files) > 2 or len(space_files) < 1:
            printv(parser.error('ERROR: -to-fsl expects 1 or 2 arguments'))
            return
        spaces = [Image(s) for s in space_files]
        if len(spaces) < 2:
            spaces.append(None)
        im_out = [displacement_to_abs_fsl(im_in, spaces[0], spaces[1])]

    # If these arguments are used standalone, simply pass the input image to the output (the affines were set earlier)
    elif arguments.set_sform_to_qform or arguments.set_qform_to_sform:
        im_out = [im_in]

    else:
        im_out = None
        printv(parser.error('ERROR: you need to specify an operation to do on the input image'))

    # Write output
    if im_out is not None:
        printv('Generate output files...', verbose)
        # if only one output
        if len(im_out) == 1 and arguments.split is None:
            im_out[0].save(fname_out, dtype=output_type, verbose=verbose)
            display_viewer_syntax([fname_out], verbose=verbose)
        if arguments.mcs:
            # use input file name and add _X, _Y _Z. Keep the same extension
            l_fname_out = []
            for i_dim in range(3):
                l_fname_out.append(add_suffix(fname_out or fname_in[0], f'_{DIM_LIST[i_dim].upper()}'))
                im_out[i_dim].save(l_fname_out[i_dim], verbose=verbose)
            display_viewer_syntax(l_fname_out, verbose=verbose)
        if arguments.split is not None:
            # use input file name and add _"DIM+NUMBER". Keep the same extension
            l_fname_out = []
            for i, im in enumerate(im_out):
                l_fname_out.append(add_suffix(fname_out or fname_in[0], f'_{DIM_LIST[dim].upper()}{i:04}'))
                im.save(l_fname_out[i])
            display_viewer_syntax(l_fname_out, verbose=verbose)

    # Generate QC report (for `sct_image -stitch` only)
    if arguments.qc is not None:
        if arguments.stitch is not None:
            printv("Generating QC Report...", verbose=verbose)
            # specify filenames to use in QC report
            path_tmp = tmp_create(basename="stitching-qc")
            fname_qc_concat = os.path.join(path_tmp, "concatenated_input_images.nii.gz")
            fname_qc_out = os.path.join(path_tmp, os.path.basename(fname_out))
            # generate 2 images to compare in QC report
            # (1. naively concatenated input images, and 2. stitched image) padded so both have same dimensions
            im_concat, im_out_padded = generate_stitched_qc_images(im_in_list, im_out[0])
            im_concat.save(fname_qc_concat)
            im_out_padded.save(fname_qc_out)
            # generate the QC report itself
            generate_qc(fname_in1=fname_qc_out, fname_in2=fname_qc_concat, args=sys.argv[1:],
                        path_qc=os.path.abspath(arguments.qc), dataset=arguments.qc_dataset,
                        subject=arguments.qc_subject, contrast=arguments.qc_contrast, process='sct_image_stitch')
        else:
            printv("WARNING: '-qc' is only supported for 'sct_image -stitch'. QC report will not be generated.",
                   type='warning')

    elif arguments.getorient:
        printv(orient)

    elif arguments.display_warp:
        printv('Warping grid generated.', verbose, 'info')


def displacement_to_abs_fsl(disp_im, src, tgt=None):
    """ Convert an ITK style displacement field to an FSL compatible absolute coordinate field.
        this can be applied using `applywarp` from FSL using the `--abs` flag. Or converted to a
        normal relative displacement field with `convertwarp --abs --relout`
        args:
          disp_im: An `Image` object representing an ITK displacement field
          src: An `Image` object in the same space as the images you'd like to transform. Usually the fixed or
               source image used in the registration generating the displacement field.
          tgt: An `Image` object, with the correct target space, for the unusual case when the deformation
               not in the target space.
    """

    def aff(x): return x.header.get_best_affine()

    def pad1_3vec(vec_arr):
        """ Pad a 3d array of 3 vectors by 1 to make a 3d array of 4 vectors for affine transformation """
        return np.pad(vec_arr, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1)

    def apply_affine(data, aff):
        """ Transform a 3d array of 3 vectors by a 4x4 affine matrix" s.t. y = A x for each x in the array """
        return np.dot(pad1_3vec(data), aff.T)[..., 0:3]

    # Drop 5d to 4d displacement
    disp_im.data = disp_im.data.squeeze(-2)

    # If the target and displacement are in different spaces, resample the displacement to
    # the target space
    if tgt is not None:
        hdr = disp_im.header.copy()
        shp = disp_im.data.shape
        hdr.set_data_shape(shp)
        disp_nib = nib.Nifti1Image(disp_im.data.copy(), aff(disp_im), hdr)
        disp_resampled = nib_processing.resample_from_to(disp_nib, (shp, aff(tgt)), order=1)
        disp_im.data = disp_resampled.dataobj.copy()
        disp_im.header = disp_resampled.header.copy()

    # Generate an array of voxel coordinates for the target (note disp_im is in the same space as the target)
    disp_dims = disp_im.data.shape[0:3]
    disp_coords = np.mgrid[0:disp_dims[0], 0:disp_dims[1], 0:disp_dims[2]].transpose((1, 2, 3, 0))

    # Convert those to world coordinates
    tgt_coords_world = apply_affine(disp_coords, aff(disp_im))

    # Apply the displacement. TODO check if this works for all tgt orientations [works for RPI]
    src_coords_world = tgt_coords_world + disp_im.data * [-1, -1, 1]

    # Convert these to voxel coordinates in the source image
    sw2v = np.linalg.inv(aff(src))  # world to voxel matrix for the source image
    src_coords = apply_affine(src_coords_world, sw2v)

    # Convert from voxel coordinates to FSL "mm" units (not really mm)
    # First we handle the case where the src or tgt image space has a positive
    # determinant by inverting the x coordinate
    if np.linalg.det(aff(src)) > 0:
        src_coords[..., 0] = (src.data.shape[0] - 1) - src_coords[..., 0]

    # Then we multiply by the voxel sizes and we're done
    src_coords_mm = src_coords * src.header.get_zooms()[0:3]

    out_im = disp_im.copy()
    out_im.data = src_coords_mm
    return out_im


def split_data(im_in, dim, squeeze_data=True):
    """
    """
    # backwards compat
    return split_img_data(src_img=im_in, dim=dim, squeeze_data=squeeze_data)


def remove_vol(im_in, index_vol_user, todo):
    """
    Remove specific volumes from 4D data.
    :param im_in: [str] input image.
    :param index_vol: [int] list of indices corresponding to volumes to remove
    :param todo: {keep, remove} what to do
    :return: 4d volume
    """
    # get data
    data = im_in.data
    nt = data.shape[3]
    # define index list of volumes to keep/remove
    if todo == 'remove':
        index_vol = [i for i in range(0, nt) if i not in index_vol_user]
    elif todo == 'keep':
        index_vol = index_vol_user
    else:
        printv('ERROR: wrong assignment of variable "todo"', 1, 'error')
    # define new 4d matrix with selected volumes
    data_out = data[:, :, :, index_vol]
    # save matrix inside new Image object
    im_out = im_in.copy()
    im_out.data = data_out
    return im_out


def multicomponent_split(im):
    """
    Convert composite image (e.g., ITK warping field, 5dim) into several 3d volumes.
    Replaces "c3d -mcs warp_comp.nii -oo warp_vecx.nii warp_vecy.nii warp_vecz.nii"
    :param im:
    :return:
    """
    data = im.data
    if len(data.shape) != 5:
        raise ValueError(f"Expected 5D image but got '{len(data.shape)}' dimensions instead.")
    data_out = []
    for i in range(data.shape[-1]):
        dat_out = data[:, :, :, :, i]
        '''
        while dat_out.shape[-1] == 1:
            dat_out = reshape(dat_out, dat_out.shape[:-1])
        '''
        data_out.append(dat_out)  # .astype('float32'))
    im_out = [im.copy() for j in range(len(data_out))]
    for i, im in enumerate(im_out):
        im.data = data_out[i]
        im.hdr.set_intent('vector', (), '')
        im.absolutepath = add_suffix(im.absolutepath, f'_{i}')
    return im_out


def multicomponent_merge(im_in_list: Sequence[Image]):
    # WARNING: output multicomponent is not optimal yet, some issues may be related to the use of this function

    im_0 = im_in_list[0]
    new_shape = list(im_0.data.shape)
    if len(new_shape) == 3:
        new_shape.append(1)
    new_shape.append(len(im_in_list))
    new_shape = tuple(new_shape)

    data_out = np.zeros(new_shape)
    for i, im in enumerate(im_in_list):
        dat = im.data
        if len(dat.shape) == 2:
            data_out[:, :, 0, 0, i] = dat.astype('float32')
        elif len(dat.shape) == 3:
            data_out[:, :, :, 0, i] = dat.astype('float32')
        elif len(dat.shape) == 4:
            data_out[:, :, :, :, i] = dat.astype('float32')
        del im
        del dat
    im_out = im_0.copy()
    im_out.data = data_out.astype('float32')
    im_out.hdr.set_intent('vector', (), '')
    im_out.absolutepath = add_suffix(im_out.absolutepath, '_multicomponent')
    return im_out


def visualize_warp(im_warp: Image, im_grid: Image = None, step=3, rm_tmp=True):
    fname_warp = im_warp.absolutepath
    if im_grid:
        fname_grid = im_grid.absolutepath
    else:
        tmp_dir = tmp_create(basename="visualize-warp")
        nx, ny, nz = im_warp.data.shape[0:3]
        curdir = os.getcwd()
        os.chdir(tmp_dir)
        sq = np.zeros((step, step))
        sq[step - 1] = 1
        sq[:, step - 1] = 1
        dat = np.zeros((nx, ny, nz))
        for i in range(0, dat.shape[0], step):
            for j in range(0, dat.shape[1], step):
                for k in range(dat.shape[2]):
                    if dat[i:i + step, j:j + step, k].shape == (step, step):
                        dat[i:i + step, j:j + step, k] = sq
        im_grid = Image(param=dat)
        grid_hdr = im_warp.hdr
        im_grid.hdr = grid_hdr
        fname_grid = f'grid_{step}.nii.gz'
        im_grid.save(fname_grid)
        fname_grid_resample = add_suffix(fname_grid, '_resample')
        sct_resample.main(argv=['-i', fname_grid, '-f', '3x3x1', '-x', 'nn', '-o', fname_grid_resample, '-v', '0'])
        fname_grid = os.path.join(tmp_dir, fname_grid_resample)
        os.chdir(curdir)
    path_warp, file_warp, ext_warp = extract_fname(fname_warp)
    grid_warped = os.path.join(path_warp, f'{extract_fname(fname_grid)[1]}_{file_warp}{ext_warp}')
    sct_apply_transfo.main(argv=['-i', fname_grid, '-d', fname_grid, '-w', fname_warp, '-o', grid_warped, '-v', '0'])
    if rm_tmp:
        rmtree(tmp_dir)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
