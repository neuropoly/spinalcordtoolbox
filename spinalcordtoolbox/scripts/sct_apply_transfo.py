#!/usr/bin/env python
#
# Apply transformations. This function is a wrapper for sct_WarpImageMultiTransform
#
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

# TODO: display message at the end
# TODO: interpolation methods

import sys
import os
import functools
from typing import Sequence
import textwrap

import numpy as np

from spinalcordtoolbox.image import Image, add_suffix
from spinalcordtoolbox.cropping import ImageCropper
from spinalcordtoolbox.math import dilate
from spinalcordtoolbox.labels import cubic_to_point
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, get_interpolation, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, run_proc, printv, set_loglevel
from spinalcordtoolbox.utils.fs import tmp_create, rmtree, extract_fname, copy

from spinalcordtoolbox.scripts import sct_image


# PARSER
# ==========================================================================================
def get_parser():
    parser = SCTArgumentParser(
        description='Apply transformations. This function is a wrapper for antsApplyTransforms (ANTs).'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        "-i",
        help='Input image. Example: `t2.nii.gz`',
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        "-d",
        help='Destination image. For warping input images, the destination image defines the spacing, origin, size, '
             'and direction of the output warped image. Example: `dest.nii.gz`',
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        "-w",
        nargs='+',
        help='Transformation(s), which can be warping fields (nifti image) or affine transformation matrix (text '
             'file). Separate with space. Example: `warp1.nii.gz warp2.nii.gz`',
        metavar=Metavar.file,
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-winv",
        help='Affine transformation(s) listed in flag -w which should be inverted before being used. Note that this '
             'only concerns affine transformation (not warping fields). If you would like to use an inverse warping '
             'field, then directly input the inverse warping field in flag -w.',
        nargs='+',
        metavar=Metavar.file,
        default=[])
    optional.add_argument(
        "-crop",
        help="Crop the output image using the extents of the warping field.\n"
             " - 0: no cropping (WARNING: may result in duplicated output if the destination image's FOV is "
             "      larger than the FOV of the warping field)\n"
             " - 1: crop using a rectangular bounding box around the warping field (setting outside voxels to 0)\n"
             " - 2: crop using a rectangular bounding box around the warping field (changing the size of the output "
             "      image)\n"
             " - 3: mask the output image (setting outside voxels to 0) using the warping field directly instead of "
             "      using a rectangular bounding box around the warping field. Useful if Option 1 does not zero out "
             "      enough voxels.",
        type=int,
        default=0,
        choices=(0, 1, 2, 3))
    optional.add_argument(
        "-o",
        help='Filename to use for the output image (i.e. the transformed image). Example: `out.nii.gz`',
        metavar=Metavar.file)
    optional.add_argument(
        "-x",
        help=textwrap.dedent("""
            Interpolation method.

            Note: The `label` method is a special interpolation method designed for single-voxel labels (e.g. disc labels used as registration landmarks, compression labels, etc.). This method is necessary because classical interpolation may corrupt the values of single-voxel labels, or cause them to disappear entirely. The function works by dilating each label, applying the transformation using nearest neighbour interpolation, then extracting the center-of-mass of each transformed 'blob' to get a single-voxel output label. Because the output is a single-voxel label, the `-x label` method is not appropriate for multi-voxel labeled segmentations (such as spinal cord or lesion masks).
        """),  # noqa: E501 (line too long)
        default='spline',
        choices=('nn', 'linear', 'spline', 'label'))

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_tempfile_args()

    return parser


def isct_antsApplyTransforms(dimensionality,
                             fname_input,
                             fname_output,
                             fname_reference,
                             transforms,
                             interpolation_args,
                             verbose):
    """
    Wrapper for CLI binary that allows for pre/postprocessing steps.

    Note: Most logic should still go in `sct_apply_transfo`, because our SCT
    script is essentially one big wrapper script for the ANTs call. Only use
    this wrapper to fix bugs in the binary itself, or to compensate for
    missing behavior/options in the ANTs source code. This wrapper should more
    or less function like the original binary (with only minor tweaks added).
    """
    run_proc(['isct_antsApplyTransforms',
              '-d', dimensionality,
              '-i', fname_input,
              '-o', fname_output,
              '-t'] + transforms +  # list of transforms
             ['-r', fname_reference] +
             interpolation_args,  # list of ['-n', interp_type]
             verbose=verbose,
             is_sct_binary=True)
    # Preserve integer datatype when interpolation is NearestNeighour to
    # counter ANTs behavior of always outputting float64 images
    im_out = Image(fname_output)
    dtype_in = Image(fname_input).data.dtype
    if 'NearestNeighbor' in interpolation_args:
        im_out.data = im_out.data.astype(dtype_in)
        im_out.hdr.set_data_dtype(dtype_in)
        im_out.save(mutable=True, verbose=0)
    # FIXME: Consider returning an `Image` type if the extra save/loads
    #        add significant overhead to `sct_apply_transfo`.


class Transform:
    def __init__(self, input_filename, fname_dest, output_filename, list_warp, list_warpinv=[], verbose=0, crop=0,
                 interp='spline', remove_temp_files=1, debug=0):
        self.input_filename = input_filename
        self.list_warp = list_warp
        self.list_warpinv = list_warpinv
        self.fname_dest = fname_dest
        self.output_filename = output_filename
        self.interp = interp
        self.crop = crop
        self.verbose = verbose
        self.remove_temp_files = remove_temp_files
        self.debug = debug

    def apply(self):
        # Initialization
        fname_src = self.input_filename  # source image (moving)
        list_warp = self.list_warp  # list of warping fields
        fname_out = self.output_filename  # output
        fname_dest = self.fname_dest  # destination image (fix)
        verbose = self.verbose
        remove_temp_files = self.remove_temp_files
        crop_reference = self.crop  # if = 1, put 0 everywhere around warping field, if = 2, real crop

        islabel = False
        if self.interp == 'label':
            islabel = True
            self.interp = 'nn'

        interp = get_interpolation('isct_antsApplyTransforms', self.interp)

        # Parse list of warping fields
        printv('\nParse list of warping fields...', verbose)
        use_inverse = []
        fname_warp_list_invert = []
        # list_warp = list_warp.replace(' ', '')  # remove spaces
        # list_warp = list_warp.split(",")  # parse with comma
        for idx_warp, path_warp in enumerate(self.list_warp):
            # Check if this transformation should be inverted
            if path_warp in self.list_warpinv:
                use_inverse.append('-i')
                # list_warp[idx_warp] = path_warp[1:]  # remove '-'
                fname_warp_list_invert.extend([[use_inverse[idx_warp], list_warp[idx_warp]]])
            else:
                use_inverse.append('')
                fname_warp_list_invert.append([path_warp])
            path_warp = list_warp[idx_warp]
            if path_warp.endswith((".nii", ".nii.gz")) \
                    and Image(list_warp[idx_warp]).header.get_intent()[0] != 'vector':
                raise ValueError(f"Displacement field in {path_warp} is invalid: should be encoded"
                                 " in a 5D file with vector intent code"
                                 " (see https://web.archive.org/web/20241009085040/https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h)")
        # need to check if last warping field is an affine transfo
        isLastAffine = False
        path_fname, file_fname, ext_fname = extract_fname(fname_warp_list_invert[-1][-1])
        if ext_fname in ['.txt', '.mat']:
            isLastAffine = True

        # check if destination file is 3d
        # check_dim(fname_dest, dim_lst=[3]) # PR 2598: we decided to skip this line.

        # N.B. Here we take the inverse of the warp list, because sct_WarpImageMultiTransform concatenates in the reverse order
        fname_warp_list_invert.reverse()
        fname_warp_list_invert = functools.reduce(lambda x, y: x + y, fname_warp_list_invert)

        # Extract path, file and extension
        path_src, file_src, ext_src = extract_fname(fname_src)
        path_dest, file_dest, ext_dest = extract_fname(fname_dest)

        # Get dimensions of data
        printv('\nGet dimensions of data...', verbose)
        img_src = Image(fname_src)
        nx, ny, nz, nt, px, py, pz, pt = img_src.dim
        # nx, ny, nz, nt, px, py, pz, pt = get_dimension(fname_src)
        printv(f'  {nx} x {ny} x {nz} x {nt}', verbose)

        # if 3d
        if nt == 1:
            # Apply transformation
            printv('\nApply transformation...', verbose)
            if nz in [0, 1]:
                dim = '2'
            else:
                dim = '3'
            # if labels, dilate before resampling
            if islabel:
                printv("\nDilate labels before warping...", verbose)
                path_tmp = tmp_create(basename="apply-transfo-3d-label")
                fname_dilated_labels = os.path.join(path_tmp, "dilated_data.nii")
                # dilate points (NB: 'ball' of size <5 can result in lost labels for single-voxel labels, see #4863)
                dilate(Image(fname_src), 5, 'ball', islabel=True).save(fname_dilated_labels)
                fname_src = fname_dilated_labels

            printv("\nApply transformation and resample to destination space...", verbose)
            isct_antsApplyTransforms(
                dimensionality=dim,
                fname_input=fname_src,
                fname_output=fname_out,
                transforms=fname_warp_list_invert,
                fname_reference=fname_dest,
                interpolation_args=interp,
                verbose=verbose,
            )

            img_out = Image(fname_out)

        # if 4d, loop across the T dimension
        else:
            if islabel:
                raise NotImplementedError

            dim = '4'
            path_tmp = tmp_create(basename="apply-transfo-4d")

            # convert to nifti into temp folder
            printv('\nCopying input data to tmp folder and convert to nii...', verbose)
            img_src.save(os.path.join(path_tmp, "data.nii"))
            copy(fname_dest, os.path.join(path_tmp, file_dest + ext_dest))
            fname_warp_list_tmp = []
            for fname_warp in list_warp:
                path_warp, file_warp, ext_warp = extract_fname(fname_warp)
                copy(fname_warp, os.path.join(path_tmp, file_warp + ext_warp))
                fname_warp_list_tmp.append(file_warp + ext_warp)
            fname_warp_list_invert_tmp = fname_warp_list_tmp[::-1]

            curdir = os.getcwd()
            os.chdir(path_tmp)

            # split along T dimension
            printv('\nSplit along T dimension...', verbose)

            im_dat = Image('data.nii')
            im_header = im_dat.hdr
            data_split_list = sct_image.split_data(im_dat, 3)
            for im in data_split_list:
                im.save()

            # apply transfo
            printv('\nApply transformation to each 3D volume...', verbose)
            for it in range(nt):
                file_data_split = f'data_T{it:04}.nii'
                file_data_split_reg = f'data_reg_T{it:04}.nii'

                isct_antsApplyTransforms(
                    dimensionality='3',
                    fname_input=file_data_split,
                    fname_output=file_data_split_reg,
                    transforms=fname_warp_list_invert_tmp,
                    fname_reference=(file_dest + ext_dest),
                    interpolation_args=interp,
                    verbose=verbose,
                )

            # Merge files back
            printv('\nMerge file back...', verbose)
            import glob
            # im_list = [Image(file_name) for file_name in glob.glob('data_reg_T*.nii')]
            # concat_data use to take a list of image in input, now takes a list of file names to open the files one by one (see issue #715)
            fname_list = glob.glob('data_reg_T*.nii')
            fname_list.sort()
            img_out = sct_image.concat_data(fname_list, 3, im_header['pixdim'])
            os.chdir(curdir)
            # Delete temporary folder if specified
            if remove_temp_files:
                printv('\nRemove temporary files...', verbose)
                rmtree(path_tmp, verbose=verbose)

        # Copy affine matrix from destination space to make sure qform/sform are the same
        printv("Copy affine matrix from destination space to make sure qform/sform are the same.", verbose)
        img_out.copy_affine_from_ref(Image(fname_dest))

        if islabel:
            printv("\nTake the center of mass of each registered dilated labels...", verbose)
            img_out = cubic_to_point(img_out)
            if remove_temp_files:
                printv('\nRemove temporary files...', verbose)
                rmtree(path_tmp, verbose=verbose)

        # Crop the resulting image using dimensions from the warping field
        warping_field = fname_warp_list_invert[-1]
        # If the last transformation is not an affine transfo, we need to compute the matrix space of the concatenated
        # warping field
        if not isLastAffine and crop_reference in [1, 2, 3]:
            printv('Last transformation is not affine.')
            # Extract only the first n dims of the warping field by creating a dummy image with the correct shape
            img_warp = Image(warping_field)
            warp_shape = img_warp.data.shape[:int(dim)]  # dim = {'2', '3', '4'}
            img_warp_ndim = Image(np.ones(warp_shape), hdr=img_warp.hdr)
            if crop_reference in [1, 2]:
                # Set zero to everything outside the warping field
                cropper = ImageCropper(img_out)
                cropper.get_bbox_from_ref(img_warp_ndim)
                if crop_reference == 1:
                    printv('Cropping strategy is: keep same matrix size, put 0 everywhere around warping field')
                    img_out = cropper.crop(background=0)
                elif crop_reference == 2:
                    printv('Cropping strategy is: crop around warping field (the size of warping field will '
                           'change)')
                    img_out = cropper.crop()
            elif crop_reference == 3:
                # Resample the warping field mask (in reference coordinates) into the space of the image to be cropped
                img_ref_r = resample_nib(img_warp_ndim, image_dest=img_out, interpolation='nn', mode='constant')
                # Simply mask the output image instead of doing a bounding-box-based crop
                img_out.data = img_out.data * img_ref_r.data

        # Only save at the end to avoid unnecessary save/load operations for large warped 4D images
        printv("Saving output image.", verbose)
        img_out.save(fname_out, mutable=True)


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    """
    Entry point for sct_apply_transfo
    :param argv: list of input arguments.
    :return:
    """
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    input_filename = arguments.i
    fname_out = arguments.o if arguments.o is not None else os.path.basename(add_suffix(input_filename, '_reg'))
    fname_dest = arguments.d
    warp_filename = arguments.w
    warpinv_filename = arguments.winv

    transform = Transform(input_filename=input_filename, fname_dest=fname_dest, list_warp=warp_filename,
                          list_warpinv=warpinv_filename, output_filename=fname_out)

    transform.crop = arguments.crop
    transform.interp = arguments.x
    transform.remove_temp_files = arguments.r
    transform.verbose = verbose

    transform.apply()

    display_viewer_syntax([fname_dest, fname_out], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
