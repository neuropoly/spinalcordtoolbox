#!/usr/bin/env python
#
# Perform mathematical operations on images
#
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
import pickle
import gzip
import argparse
from typing import Sequence, Union
import textwrap

import numpy as np

import spinalcordtoolbox.math as sct_math
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, list_type, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel, LazyLoader
from spinalcordtoolbox.utils.fs import extract_fname

plt = LazyLoader("plt", globals(), "matplotlib.pyplot")


def number_or_fname(arg: str) -> Union[float, str]:
    """
    Parse an argument as either a number or an image file name.

    Can be used as the `type` argument to `parser.add_argument`.

    Examples:
    - '-arg 5'             (int argument, converted to float)
    - '-arg 5.0'           (float argument)
    - '-arg image.nii.gz'  (image file name)
    """
    try:
        return float(arg)
    except ValueError:
        return arg


def denoise_params(arg: str) -> tuple[int, int]:
    """
    Parse the argument for `-denoise` into a pair (patch radius, block radius).

    Any sub-arguments that don't start with "p=" or "b=" are silently ignored.
    """
    p, b = 1, 5  # defaults
    for sub_arg in arg.split(","):
        if sub_arg.startswith("p="):
            p = int(sub_arg[2:])
        elif sub_arg.startswith("b="):
            b = int(sub_arg[2:])
    return (p, b)


def one_or_three_sigmas(arg: str) -> list[float]:
    """
    Parse the arguments for `-smooth` and `-laplacian`.

    Returns a list of exactly 3 floats, measured in millimeters.
    """
    values = [float(v) for v in arg.split(',')]
    if len(values) == 1:
        values *= 3
    elif len(values) != 3:
        raise ValueError(f"expected 1 or 3 values, got {len(values)}")
    return values


def one_two_three_ints(arg: str) -> list[int]:
    """
    Parse the arguments for `-dilate` and `-erode`.

    Returns a list of 1, 2, or 3 ints, measured in voxels.
    """
    values = [int(v) for v in arg.split('x')]
    if len(values) not in [1, 2, 3]:
        raise ValueError(f"expected 1, 2, or 3 values, got {len(values)}")
    return values


class AppendTodo(argparse.Action):
    """
    Store the arguments of an sct_maths operation in `arguments.todo`.

    The format is: (arg_name, arg_value).
    """
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.todo.append((self.dest, values))


def get_parser():
    parser = SCTArgumentParser(
        description='Perform mathematical operations on images.',
        argument_default=argparse.SUPPRESS,  # so that the operations to perform are only in arguments.todo
    )

    # Make sure the list of operations to perform gets initialized
    parser.set_defaults(todo=[])

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        "-i",
        metavar=Metavar.file,
        help="Input file. Example: `data.nii.gz`")
    mandatory.add_argument(
        "-o",
        metavar=Metavar.file,
        help='Output file. Example: `data_mean.nii.gz`')

    optional = parser.optional_arggroup
    optional.add_argument(
        '-volumewise',
        type=int,
        help=textwrap.dedent("""
            Specifying this option will process a 4D image in a "volumewise" manner:

              - Split the 4D input into individual 3D volumes
              - Apply the maths operations to each 3D volume
              - Merge the processed 3D volumes back into a single 4D output image
        """),
        choices=(0, 1),
        default=0
    )

    basic = parser.add_argument_group('BASIC OPERATIONS')
    basic.add_argument(
        "-add",
        metavar='',
        nargs="*",
        type=number_or_fname,
        action=AppendTodo,
        help=textwrap.dedent("""
            Add following input. Can be a number or one or more 3D/4D images (separated with space). Examples:

              - `sct_maths -i 3D.nii.gz -add 5`                       (Result: 3D image with `5` added to each voxel)
              - `sct_maths -i 3D.nii.gz -add 3D_2.nii.gz`             (Result: 3D image)
              - `sct_maths -i 4D.nii.gz -add 4D_2.nii.gz`             (Result: 4D image)
              - `sct_maths -i 4D_nii.gz -add 4D_2.nii.gz 4D_3.nii.gz` (Result: 4D image)

            Note: If your terminal supports it, you can also specify multiple images using a pattern:

              - `sct_maths -i 4D.nii.gz -add 4D_*.nii.gz` (Result: Adding `4D_2.nii.gz`, `4D_3.nii.gz`, etc.)

            Note: If the input image is 4D, you can also leave `-add` empty to sum the 3D volumes within the image:

              - `sct_maths -i 4D.nii.gz -add` (Result: 3D image, with 3D volumes summed within 4D image)
        """))
    basic.add_argument(
        "-sub",
        metavar='',
        nargs="+",
        type=number_or_fname,
        action=AppendTodo,
        help='Subtract following input. Can be a number, or one or more 3D/4D images (separated with space).')
    basic.add_argument(
        "-mul",
        metavar='',
        nargs="*",
        type=number_or_fname,
        action=AppendTodo,
        help='Multiply by following input. Can be a number, or one or more 3D/4D images (separated with space). '
             '(See `-add` for examples.)')
    basic.add_argument(
        "-div",
        metavar='',
        nargs="+",
        type=number_or_fname,
        action=AppendTodo,
        help='Divide by following input. Can be a number, or one or more 3D/4D images (separated with space).')
    basic.add_argument(
        '-mean',
        choices=('x', 'y', 'z', 't'),
        action=AppendTodo,
        help='Average data across dimension.')
    basic.add_argument(
        '-rms',
        choices=('x', 'y', 'z', 't'),
        action=AppendTodo,
        help='Compute root-mean-squared across dimension.')
    basic.add_argument(
        '-std',
        choices=('x', 'y', 'z', 't'),
        action=AppendTodo,
        help='Compute STD across dimension.')
    basic.add_argument(
        "-bin",
        metavar=Metavar.float,
        type=float,
        action=AppendTodo,
        help='Binarize image using specified threshold. Example: `0.5`')
    basic.add_argument(
        '-slicewise-mean',
        type=int,
        choices=(0, 1, 2),
        action=AppendTodo,
        help='Compute slicewise mean the specified dimension. Zeros are not inlcuded in the mean.')

    thresholding = parser.add_argument_group("THRESHOLDING METHODS")
    thresholding.add_argument(
        '-otsu',
        metavar=Metavar.int,
        type=int,
        action=AppendTodo,
        help='Threshold image using Otsu algorithm (from skimage). Specify the number of bins (e.g. 16, 64, 128)')
    thresholding.add_argument(
        "-adap",
        metavar=Metavar.list,
        type=list_type(',', int, 2),
        action=AppendTodo,
        help=textwrap.dedent("""
            Threshold image using Adaptive algorithm (from skimage). Provide 2 values separated by `,` that correspond to the parameters below. For example, `-adap 7,0` corresponds to a block size of 7 and an offset of 0.

              - Block size: Odd size of pixel neighborhood which is used to calculate the threshold value.
              - Offset: Constant subtracted from weighted mean of neighborhood to calculate the local threshold value. Suggested offset is 0.
        """),  # noqa: E501 (line too long)
        )
    thresholding.add_argument(
        "-otsu-median",
        metavar=Metavar.list,
        type=list_type(',', int, 2),
        action=AppendTodo,
        help=textwrap.dedent("""
            Threshold image using Median Otsu algorithm (from Dipy). Provide 2 values separated by `,` that correspond to the parameters below. For example, `-otsu-median 3,5` corresponds to a filter size of 3 repeated over 5 iterations.

              - Size: Radius (in voxels) of the applied median filter.
              - Iterations: Number of passes of the median filter.
        """),  # noqa: E501 (line too long)
        )
    thresholding.add_argument(
        '-percent',
        metavar=Metavar.int,
        type=int,
        action=AppendTodo,
        help="Threshold image using percentile of its histogram.",
        )
    thresholding.add_argument(
        "-thr",
        metavar=Metavar.float,
        type=float,
        action=AppendTodo,
        help='Lower threshold limit (zero below number).',
        )
    thresholding.add_argument(
        "-uthr",
        metavar=Metavar.float,
        type=float,
        action=AppendTodo,
        help='Upper threshold limit (zero above number).',
        )

    mathematical = parser.add_argument_group("MATHEMATICAL MORPHOLOGY")
    mathematical.add_argument(
        '-dilate',
        metavar=Metavar.int,
        type=one_two_three_ints,
        action=AppendTodo,
        help="Dilate binary or greyscale image.\n"
             "You can customize the structural element by combining the arguments `-dilate`, `-shape`, and `-dim`. "
             "(The values passed to `-dilate` will control the side length or radius of whatever shape is chosen.)\n"
             "You can provide either a single number, or 2/3 numbers separated by `x` (depending on the shape).\n"
             "\n"
             "Examples:\n"
             "  - `-shape cube -dilate 3`            -> Side length 3   -> A 3x3x3 cube.\n"
             "  - `-shape ball -dilate 2x2x5`        -> Radius 2x2x5    -> A 5x5x11 ball.\n"
             "  - `-shape disk -dilate 3 -dim 2`     -> Radius 3        -> An 7x7 disk in the X-Y plane, applied to each Z slice.\n"
             "  - `-shape square -dilate 1x4 -dim 0` -> Side length 1x4 -> A 1x4 rectangle in the Y-Z plane, applied to each X slice.",
        )
    mathematical.add_argument(
        '-erode',
        metavar=Metavar.int,
        type=one_two_three_ints,
        action=AppendTodo,
        help="Erode binary or greyscale image. The argument is interpreted the same way as for `-dilate`.",
        )
    mathematical.add_argument(
        '-shape',
        choices=('square', 'cube', 'disk', 'ball'),
        action='append',  # to output a warning if used more than once
        default=[],
        help=textwrap.dedent("""
            Shape of the structuring element for the mathematical morphology operation. Default: `ball`.

            If a 2D shape `{'disk', 'square'}` is selected, `-dim` must be specified.
        """),
        )
    mathematical.add_argument(
        '-dim',
        type=int,
        choices=(0, 1, 2),
        action='append',  # to output a warning if used more than once
        default=[],
        help="Dimension of the array which 2D structural element will be orthogonal to. For example, if you wish to "
             "apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be: shape=disk, dim=2.",
        )

    filtering = parser.add_argument_group("FILTERING METHODS")
    filtering.add_argument(
        "-smooth",
        metavar=Metavar.list,
        type=one_or_three_sigmas,
        action=AppendTodo,
        help='Gaussian smoothing filtering. Supply values for standard deviations in mm. If a single value is provided, '
             'it will be applied to each axis of the image. If multiple values are provided, there must be one value '
             'per image axis. (Examples: `-smooth 2.0,3.0,2.0` (3D image), `-smooth 2.0` (any-D image)).',
        )
    filtering.add_argument(
        '-laplacian',
        metavar=Metavar.list,
        type=one_or_three_sigmas,
        action=AppendTodo,
        help='Laplacian filtering. Supply values for standard deviations in mm. If a single value is provided, it will '
             'be applied to each axis of the image. If multiple values are provided, there must be one value per '
             'image axis. (Examples: `-laplacian 2.0,3.0,2.0` (3D image), `-laplacian 2.0` (any-D image)).',
        )
    filtering.add_argument(
        '-denoise',
        type=denoise_params,
        action=AppendTodo,
        help=textwrap.dedent("""
            Non-local means adaptative denoising from P. Coupe et al. as implemented in dipy. Separate with `,` Example: `p=1,b=3`

              - `p`: (patch radius) similar patches in the non-local means are searched for locally, inside a cube of side `2*p+1` centered at each voxel of interest. Default: `p=1`
              - `b`: (block radius) the size of the block to be used (2*b+1) in the blockwise non-local means implementation. Default: `b=5`.
                Note, block radius must be smaller than the smaller image dimension: default value is lowered for small images)

            To use default parameters, write `-denoise 1`
        """),  # noqa: E501 (line too long)
        )

    similarity = parser.add_argument_group("SIMILARITY METRIC")
    similarity.add_argument(
        '-mi',
        metavar=Metavar.file,
        action=AppendTodo,
        help='Compute the mutual information (MI) between both input files (`-i` and `-mi`) as in: '
             'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html',
        )
    similarity.add_argument(
        '-minorm',
        metavar=Metavar.file,
        action=AppendTodo,
        help='Compute the normalized mutual information (MI) between both input files (`-i` and `-mi`) as in: '
             'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html',
        )
    similarity.add_argument(
        '-corr',
        metavar=Metavar.file,
        action=AppendTodo,
        help='Compute the cross correlation (CC) between both input files (`-i` and `-corr`).',
        )

    misc = parser.misc_arggroup
    misc.add_argument(
        '-symmetrize',
        type=int,
        choices=(0, 1, 2),
        action=AppendTodo,
        help='Symmetrize data along the specified dimension.',
        )
    misc.add_argument(
        '-type',
        choices=('uint8', 'int16', 'int32', 'float32', 'complex64', 'float64',
                 'int8', 'uint16', 'uint32', 'int64', 'uint64'),
        default=None,  # needed because argument_default=argparse.SUPPRESS
        help='Output type.',
        )

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


def get_data_arrays(
    shape: tuple[int, ...],
    args: list[Union[float, str]],
) -> list[np.ndarray]:
    """
    Helper function to load the arguments of -add, -sub, -mul, -div.

    Returns a list of data arrays, where they all have the given shape.
    Raises a ValueError if there is a shape mismatch.
    """
    list_data = []
    for arg in args:
        if isinstance(arg, float):
            list_data.append(np.full(shape, arg))
        else:
            assert isinstance(arg, str)
            im = Image(arg)
            if im.data.shape != shape:
                raise ValueError(f"image {arg} has the wrong shape {im.data.shape} (expected: {shape})")
            list_data.append(im.data)
    return list_data


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    # Handle `-shape` and `-dim` for `-dilate` and `-erode`
    if len(arguments.shape) == 0:
        shape = 'ball'
    elif len(arguments.shape) == 1:
        shape = arguments.shape[0]
    else:
        parser.error("-shape cannot be specified more than once")

    if shape in ['disk', 'square']:
        # 2D kernels need the value of `-dim`
        if len(arguments.dim) == 0:
            parser.error(f"-dim is required for -shape {shape}")
        elif len(arguments.dim) == 1:
            dim = arguments.dim[0]
        else:
            parser.error("-dim cannot be specified more than once")
    else:
        # 3D kernels don't need `-dim`
        if len(arguments.dim) == 0:
            dim = None
        else:
            parser.error(f"-dim should not be specified for -shape {shape}")

    for arg_name, arg_value in arguments.todo:
        if arg_name in ['dilate', 'erode']:
            if shape in ['disk', 'square']:
                # 2D kernels need 1 or 2 values for the size argument
                # if 1 value is supplied, we repeat it to get 2 values
                if len(arg_value) == 1:
                    arg_value.extend(arg_value)
                elif len(arg_value) == 3:
                    parser.error(f"-{arg_name} needs 1 or 2 values for -shape {shape}, but got 3")
            else:
                assert shape in ['ball', 'cube']
                # 3D kernels need 1 or 3 values for the size argument
                # if 1 value is supplied, we repeat it to get 3 values
                if len(arg_value) == 1:
                    arg_value.extend(arg_value + arg_value)
                elif len(arg_value) == 2:
                    parser.error(f"-{arg_name} needs 1 or 3 values for -shape {shape}, but got 2")

    # Check that the list of operations makes sense
    if not arguments.todo:
        parser.error("there must be at least one operation to perform")
    for arg_name, _ in arguments.todo[:-1]:
        if arg_name in ['mi', 'minorm', 'corr']:
            parser.error(f"-{arg_name}: similarity metrics are only supported "
                         "as the last operation to perform")

    # Actually do the computations
    im = Image(arguments.i)

    # If volumewise is specified, treat 4D image as a set of 3D volumes
    if arguments.volumewise:
        data_in_list = [np.squeeze(arr, axis=3) for arr in np.array_split(im.data, im.data.shape[3], 3)]
    else:
        data_in_list = [im.data]

    data_out_list = []
    for data in data_in_list:
        for arg_name, arg_value in arguments.todo:
            assert data is not None

            if arg_name == "add":
                if data.ndim == 4 and not arg_value:
                    # special case to sum a 4D volume across the t axis
                    data = np.sum(data, axis=3)
                else:
                    try:
                        list_data = get_data_arrays(data.shape, arg_value)
                    except ValueError as e:
                        printv(f"ERROR: -{arg_name}: {e}", 1, 'error')
                    # for addition, this dtype is usually ok
                    safe_dtype = np.result_type(data, *list_data)
                    data = np.add(data, np.sum(list_data, axis=0, dtype=safe_dtype), dtype=safe_dtype)

            elif arg_name == "sub":
                try:
                    list_data = get_data_arrays(data.shape, arg_value)
                except ValueError as e:
                    printv(f"ERROR: -{arg_name}: {e}", 1, 'error')
                # for subtraction, make sure the dtype is at least signed by including int8
                safe_dtype = np.result_type(data, *list_data, np.int8)
                data = np.subtract(data, np.sum(list_data, axis=0, dtype=safe_dtype), dtype=safe_dtype)

            elif arg_name == "mul":
                if data.ndim == 4 and not arg_value:
                    # special case to multiply a 4D volume across the t axis
                    data = np.prod(data, axis=3)
                else:
                    try:
                        list_data = get_data_arrays(data.shape, arg_value)
                    except ValueError as e:
                        printv(f"ERROR: -{arg_name}: {e}", 1, 'error')
                    # for multiplication, this dtype is usually ok
                    safe_dtype = np.result_type(data, *list_data)
                    data = np.multiply(data, np.prod(list_data, axis=0, dtype=safe_dtype), dtype=safe_dtype)

            elif arg_name == "div":
                try:
                    list_data = get_data_arrays(data.shape, arg_value)
                except ValueError as e:
                    printv(f"ERROR: -{arg_name}: {e}", 1, 'error')
                # for division, make sure the dtype is at least floating point by including float32
                safe_dtype = np.result_type(data, *list_data, np.float32)
                data = np.divide(data, np.prod(list_data, axis=0, dtype=safe_dtype), dtype=safe_dtype)

            elif arg_name == "mean":
                axis = ('x', 'y', 'z', 't').index(arg_value)
                if axis >= data.ndim:
                    # Averaging a 3D image over time, nothing to do
                    pass
                else:
                    data = np.mean(data, axis)
                    if axis < 3:
                        # Averaging over a spatial axis, we should preserve it
                        data = np.expand_dims(data, axis)

            elif arg_name == "rms":
                data = data.astype(float)
                axis = ('x', 'y', 'z', 't').index(arg_value)
                if axis >= data.ndim:
                    # Taking the mean across time for a 3D image has no effect.
                    # Because of this, RMS is just squaring then sqrting (i.e. abs)
                    data = np.abs(data)
                else:
                    data = np.sqrt(np.mean(np.square(data), axis))
                    if axis < 3:
                        # Taking RMS over a spatial axis, we should preserve it
                        data = np.expand_dims(data, axis)

            elif arg_name == "std":
                axis = ('x', 'y', 'z', 't').index(arg_value)
                if axis >= data.ndim or data.shape[axis] == 1:
                    printv("ERROR: Zero division while taking -std along a singleton dimension", 1, 'error')
                else:
                    data = np.std(data, axis, ddof=1)
                    if axis < 3:
                        # Taking std over a spatial axis, we should preserve it
                        data = np.expand_dims(data, axis)

            elif arg_name == "bin":
                bin_thr = arg_value
                data = sct_math.binarize(data, bin_thr)

            elif arg_name == "slicewise_mean":
                axis = arg_value
                # TODO: add option to include zeros in mean.
                data = sct_math.slicewise_mean(data, axis)

            elif arg_name == "otsu":
                nbins = arg_value
                data = sct_math.otsu(data, nbins)

            elif arg_name == "adap":
                block_size, offset = arg_value
                data = sct_math.adap(data, block_size, offset)

            elif arg_name == "otsu_median":
                size, n_iter = arg_value
                data = sct_math.otsu_median(data, size, n_iter)

            elif arg_name == "percent":
                percentile = arg_value
                data = sct_math.perc(data, percentile)

            elif arg_name == "thr":
                threshold = arg_value
                data = sct_math.threshold(data, lthr=threshold)

            elif arg_name == "uthr":
                threshold = arg_value
                data = sct_math.threshold(data, uthr=threshold)

            elif arg_name == "dilate":
                # This uses the global `shape` and `dim` values
                size = arg_value
                data = sct_math.dilate(data, size, shape, dim)

            elif arg_name == "erode":
                # This uses the global `shape` and `dim` values
                size = arg_value
                data = sct_math.erode(data, size, shape, dim)

            elif arg_name in ["smooth", "laplacian"]:
                # Adjust sigmas from millimeters to voxels
                # This uses the resolution of the starting value `im`
                sigmas = [mm/pixdim for mm, pixdim in zip(arg_value, im.dim[4:7])]
                data = {
                    "smooth": sct_math.smooth,
                    "laplacian": sct_math.laplacian,
                }[arg_name](data, sigmas)

            elif arg_name == "denoise":
                patch_radius, block_radius = arg_value
                data = sct_math.denoise_nlmeans(data, patch_radius, block_radius)

            elif arg_name in ["mi", "minorm", "corr"]:
                # Reuses the header of the starting value `im`
                compute_similarity(
                    img1=Image(data, hdr=im.hdr),
                    img2=Image(arg_value),
                    fname_out=arguments.o,
                    metric=arg_name,
                    metric_full={
                        'mi': 'Mutual information',
                        'minorm': 'Normalized Mutual information',
                        'corr': 'Pearson correlation coefficient',
                    }[arg_name],
                    verbose=verbose,
                )
                printv(f"\nDone! File created: {arguments.o}", verbose, 'info')
                return

            else:
                assert arg_name == "symmetrize"
                axis = arg_value
                data = sct_math.symmetrize(data, axis)

        data_out_list.append(data)

    # Reconstruct output data array from processed volumes
    data = np.stack(data_out_list, axis=3) if len(data_out_list) > 1 else data_out_list[0]

    # Save the final image with the requested dtype
    Image(data, hdr=im.hdr).save(arguments.o, dtype=arguments.type)
    display_viewer_syntax([arguments.o], verbose=verbose)


def compute_similarity(img1: Image, img2: Image, fname_out: str, metric: str, metric_full: str, verbose):
    """
    Sanitize input and compute similarity metric between two images data.
    """
    if img1.data.size != img2.data.size:
        raise ValueError(
            "Input images don't have the same size!\n"
            "Please use  \"sct_register_multimodal -i im1.nii.gz -d im2.nii.gz -identity 1\"  "
            "to put the input images in the same space"
        )

    res, data1_1d, data2_1d = sct_math.compute_similarity(img1.data, img2.data, metric=metric)

    if verbose > 1:
        plt.plot(data1_1d, 'b')
        plt.plot(data2_1d, 'r')
        plt.title(f'Similarity: {metric_full} = {res}')
        plt.savefig('fig_similarity.png')

    path_out, filename_out, ext_out = extract_fname(fname_out)
    if ext_out not in ['.txt', '.pkl', '.pklz', '.pickle']:
        raise ValueError(f"The output file should be a text file or a pickle file. Received extension: {ext_out}")

    if ext_out == '.txt':
        with open(fname_out, 'w') as f:
            f.write(f'{metric_full}: \n{res}')
    elif ext_out == '.pklz':
        pickle.dump(res, gzip.open(fname_out, 'wb'), protocol=2)
    else:
        pickle.dump(res, open(fname_out, 'w'), protocol=2)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
