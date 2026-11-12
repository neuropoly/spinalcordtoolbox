#!/usr/bin/env python
#
# Create mask along z direction.
#
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

# TODO: scale size in mm.

import sys
import os
from typing import Sequence
import textwrap

import numpy as np

from scipy.ndimage import center_of_mass

from spinalcordtoolbox.image import Image, empty_like
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel, LazyLoader
from spinalcordtoolbox.utils.fs import tmp_create, check_file_exist, extract_fname, rmtree, copy
from spinalcordtoolbox.labels import create_labels
from spinalcordtoolbox.types import Coordinate

from spinalcordtoolbox.scripts.sct_image import concat_data

nib = LazyLoader("nib", globals(), "nibabel")


# DEFAULT PARAMETERS
class Param:
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.fname_out = ''
        self.process_list = ['coord', 'point', 'centerline', 'center']
        self.process = 'center'  # default method
        self.shape_list = ['cylinder', 'box', 'gaussian']
        self.shape = 'cylinder'  # default shape
        self.size = '41'  # in voxel. if gaussian, size corresponds to sigma.
        self.even = 0
        self.file_prefix = 'mask_'  # output prefix
        self.verbose = 1
        self.remove_temp_files = 1
        self.offset = '0,0'


def get_parser():
    # Initialize default parameters
    param_default = Param()

    parser = SCTArgumentParser(
        description='Create mask along z direction.'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        help='Image to create mask on. Only used to get header. Must be 3D. Example: `data.nii.gz`',
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        '-p',
        default=param_default.process,
        help=textwrap.dedent("""
            Process to generate mask.

              - `coord,XxY`: Center mask at the X,Y coordinates. Example: `coord,20x15`
              - `point,FILE`: Center mask at the X,Y coordinates of the label defined in input volume FILE. Example: `point,label.nii.gz`
              - `center`: Center mask in the middle of the FOV `[nx/2, ny/2]`.
              - `centerline,FILE`: At each slice, the mask is centered at the spinal cord centerline, defined by the input segmentation FILE. Example: `centerline,t2_seg.nii.gz`
        """),
        metavar=Metavar.str,
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-size',
        help='Diameter of the mask in the axial plane, given in pixel (Example: `35`) or in millimeter (Example: `35mm`). '
             'If shape=gaussian, size instead corresponds to "sigma" (Example: `45`).',
        metavar=Metavar.str,
        default=param_default.size)
    optional.add_argument(
        '-f',
        help='Shape of the mask',
        default=param_default.shape,
        choices=('cylinder', 'box', 'gaussian'))
    optional.add_argument(
        '-o',
        metavar=Metavar.str,
        help='Name of output mask, Example: `data.nii.gz`',)

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_tempfile_args()

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

    param = Param()
    param.fname_data = os.path.abspath(arguments.i)

    if arguments.p is not None:
        param.process = (arguments.p).split(',')
        if param.process[0] not in param.process_list:
            printv(parser.error(f'ERROR: Process {param.process[0]} is not recognized.'))
    if arguments.size is not None:
        param.size = arguments.size
    if arguments.f is not None:
        param.shape = arguments.f
    if arguments.o is not None:
        param.fname_out = os.path.abspath(arguments.o)
    if arguments.r is not None:
        param.remove_temp_files = arguments.r

    # run main program
    create_mask(param)

    display_viewer_syntax([param.fname_data, param.fname_out], im_types=['anat', 'seg'], opacities=['', '0.5'], verbose=verbose)


def create_mask(param):
    # parse argument for method
    method_type = param.process[0]
    # check method val
    if not method_type == 'center':
        method_val = param.process[1]

    # check existence of input files
    if method_type == 'centerline':
        check_file_exist(method_val, param.verbose)

    # Extract path/file/extension
    path_data, file_data, ext_data = extract_fname(param.fname_data)

    # Get output folder and file name
    if param.fname_out == '':
        param.fname_out = os.path.abspath(param.file_prefix + file_data + ext_data)

    path_tmp = tmp_create(basename="create-mask")

    printv('\nOrientation:', param.verbose)
    orientation_input = Image(param.fname_data).orientation
    printv(f'  {orientation_input}', param.verbose)

    # copy input data to tmp folder and re-orient to RPI
    Image(param.fname_data).change_orientation("RPI").save(os.path.join(path_tmp, "data_RPI.nii"))
    if method_type == 'centerline':
        Image(method_val).change_orientation("RPI").save(os.path.join(path_tmp, "centerline_RPI.nii"))
    if method_type == 'point':
        Image(method_val).change_orientation("RPI").save(os.path.join(path_tmp, "point_RPI.nii"))

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Get dimensions of data
    im_data = Image('data_RPI.nii')
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    printv('\nDimensions:', param.verbose)
    printv(im_data.dim, param.verbose)
    # in case user input 4d data
    if nt != 1:
        printv(f'WARNING in {os.path.basename(__file__)}: Input image is 4d but output mask will be 3D from first time slice.', param.verbose, 'warning')
        # extract first volume to have 3d reference
        nii = empty_like(Image('data_RPI.nii'))
        data3d = nii.data[:, :, :, 0]
        nii.data = data3d
        nii.save('data_RPI.nii')

    if method_type == 'coord':
        # parse to get coordinate
        coord = [x for x in map(int, method_val.split('x'))]

    if method_type == 'point':
        # extract coordinate of point
        printv('\nExtract coordinate of point...', param.verbose)
        coord = Image("point_RPI.nii").getNonZeroCoordinates()

    if method_type == 'center':
        # set coordinate at center of FOV
        coord = np.round(float(nx) / 2), np.round(float(ny) / 2)

    if method_type == 'centerline':
        # get name of centerline from user argument
        fname_centerline = 'centerline_RPI.nii'
    else:
        # generate volume with line along Z at coordinates 'coord'
        printv('\nCreate line...', param.verbose)
        fname_centerline = create_line(param, 'data_RPI.nii', coord, nz)

    # create mask
    printv('\nCreate mask...', param.verbose)
    centerline = nib.load(fname_centerline)  # open centerline
    hdr = centerline.header  # get header
    hdr.set_data_dtype('uint8')  # set imagetype to uint8
    # spacing = hdr.structarr['pixdim']
    data_centerline = np.asanyarray(centerline.dataobj)  # get centerline
    # if data is 2D, reshape with empty third dimension
    if len(data_centerline.shape) == 2:
        data_centerline_shape = list(data_centerline.shape)
        data_centerline_shape.append(1)
        data_centerline = data_centerline.reshape(data_centerline_shape)
    z_centerline_not_null = [iz for iz in range(0, nz, 1) if data_centerline[:, :, iz].any()]
    # get center of mass of the centerline
    cx = [0] * nz
    cy = [0] * nz
    for iz in range(0, nz, 1):
        if iz in z_centerline_not_null:
            cx[iz], cy[iz] = center_of_mass(np.array(data_centerline[:, :, iz]))
    # create 2d masks
    im_list = []
    for iz in range(nz):
        if iz not in z_centerline_not_null:
            im_list.append(Image(data_centerline[:, :, iz], hdr=hdr))
        else:
            center = np.array([cx[iz], cy[iz]])
            mask2d = create_mask2d(param, center, param.shape, param.size, im_data=im_data)
            im_list.append(Image(mask2d, hdr=hdr))
    im_out = concat_data(im_list, dim=2).save('mask_RPI.nii.gz')

    im_out.change_orientation(orientation_input)
    im_out.header = Image(param.fname_data).header
    im_out.save(param.fname_out)

    # come back
    os.chdir(curdir)

    # Remove temporary files
    if param.remove_temp_files == 1:
        printv('\nRemove temporary files...', param.verbose)
        rmtree(path_tmp)


def create_line(param, fname, coord, nz):
    """
    Create vertical line in 3D volume
    :param param:
    :param fname:
    :param coord:
    :param nz:
    :return:
    """

    # duplicate volume (assumes input file is nifti)
    copy(fname, 'line.nii', verbose=param.verbose)

    # set all voxels to zero
    img = Image('line.nii')
    img.data = np.zeros_like(img.data)
    img.save()

    labels = []

    if isinstance(coord[0], Coordinate):
        for x, y, _, _ in coord:
            labels.extend([Coordinate([x, y, iz, 1]) for iz in range(nz)])
    else:
        # backwards compat
        labels.extend([Coordinate([coord[0], coord[1], iz, 1]) for iz in range(nz)])

    create_labels(img, labels).save()

    return 'line.nii'


def create_mask2d(param, center, shape, size, im_data):
    """
    Create a 2D mask
    :param param:
    :param center:
    :param shape:
    :param size:
    :param im_data: Image object for input data.
    :return:
    """
    # get dim
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    # extract offset d = 2r+1 --> r=ceil((d-1)/2.0)
    offset = param.offset.split(',')
    offset[0] = int(offset[0])
    offset[1] = int(offset[1])
    # px, py = spacing[0], spacing[1]

    # initialize 2d grid
    xx, yy = np.mgrid[:nx, :ny]
    mask2d = np.zeros((nx, ny))
    xc = center[0]
    yc = center[1]
    if 'mm' in size:
        size = float(size[:-2])
        radius_x = np.ceil((int(np.round(size / px)) - 1) / 2.0)
        radius_y = np.ceil((int(np.round(size / py)) - 1) / 2.0)
    else:
        radius_x = np.ceil((int(size) - 1) / 2.0)
        radius_y = radius_x

    if shape == 'box':
        mask2d = ((abs(xx + offset[0] - xc) <= radius_x) & (abs(yy + offset[1] - yc) <= radius_y)) * 1

    elif shape == 'cylinder':
        mask2d = (((xx + offset[0] - xc) / radius_x) ** 2 + ((yy + offset[1] - yc) / radius_y) ** 2 <= 1) * 1

    elif shape == 'gaussian':
        sigma = float(radius_x)
        mask2d = np.exp(-(((xx + offset[0] - xc)**2) / (2 * (sigma**2)) + ((yy + offset[1] - yc)**2) / (2 * (sigma**2))))

    return mask2d


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
