# pytest unit tests for spinalcordtoolbox.centerline

import itertools
from datetime import datetime

import pytest
import numpy as np
import nibabel as nib

from spinalcordtoolbox.centerline.curve_fitting import bspline
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline, find_and_sort_coord
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import sct_test_path


# Enforce that all tests in this suite use verbose logging
pytestmark = pytest.mark.usefixtures("verbose_logging")


# Separate setting for get_centerline. Set to 2 to save images ("DEBUG"), 0 otherwise ("INFO")
VERBOSE = 0


def dummy_centerline(size_arr=(9, 9, 9), pixdim=(1, 1, 1), subsampling=1, dilate_ctl=0, hasnan=False, zeroslice=[],
                     outlier=[], orientation='RPI', debug=False):
    """
    Create a dummy Image centerline of small size. Return the full and sub-sampled version along z. Voxel resolution
    on fully-sampled data is 1x1x1 mm (so, 2x undersampled data along z would have resolution of 1x1x2 mm).
    :param size_arr: tuple: (nx, ny, nz)
    :param pixdim: tuple: (px, py, pz)
    :param subsampling: int >=1. Subsampling factor along z. 1: no subsampling. 2: centerline defined every other z.
    :param dilate_ctl: Dilation of centerline. E.g., if dilate_ctl=1, result will be a square of 3x3 per slice.
                         if dilate_ctl=0, result will be a single pixel per slice.
    :param hasnan: Bool: Image has non-numerical values: nan, inf. In this case, do not subsample.
    :param zeroslice: list int: zero all slices listed in this param
    :param outlier: list int: replace the current point with an outlier at the corner of the image for the slices listed
    :param orientation:
    :param debug: Bool: Write temp files
    :return:
    """
    nx, ny, nz = size_arr
    # create regularized curve, within X-Z plane, located at y=ny/4, passing through the following points:
    x = np.array([round(nx/4.), round(nx/2.), round(3*nx/4.)])
    z = np.array([0, round(nz/2.), nz-1])
    # we use bspline (instead of poly) in order to avoid bad extrapolation at edges
    # see: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2754
    xfit, _ = bspline(z, x, range(nz), 10)
    # p = P.fit(z, x, 3)
    # p = np.poly1d(np.polyfit(z, x, deg=3))
    data = np.zeros((nx, ny, nz))
    arr_ctl = np.array([xfit.astype(int),
                        [round(ny / 4.)] * len(range(nz)),
                        range(nz)], dtype=np.uint16)
    # Loop across dilation of centerline. E.g., if dilate_ctl=1, result will be a square of 3x3 per slice.
    for ixiy_ctl in itertools.product(range(-dilate_ctl, dilate_ctl+1, 1), range(-dilate_ctl, dilate_ctl+1, 1)):
        data[(arr_ctl[0] + ixiy_ctl[0]).tolist(),
             (arr_ctl[1] + ixiy_ctl[1]).tolist(),
             arr_ctl[2].tolist()] = 1
    # Zero specified slices
    if zeroslice is not []:
        data[:, :, zeroslice] = 0
    # Add outlier
    if outlier is not []:
        # First, zero all the slice
        data[:, :, outlier] = 0
        # Then, add point in the corner
        data[0, 0, outlier] = 1
    # Create image with default orientation LPI
    affine = np.eye(4)
    affine[0:3, 0:3] = affine[0:3, 0:3] * pixdim
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(data, hdr=nii.header, dim=nii.header.get_data_shape())
    # subsample data
    img_sub = img.copy()
    img_sub.data = np.zeros((nx, ny, nz))
    for iz in range(0, nz, subsampling):
        img_sub.data[..., iz] = data[..., iz]
    # Add non-numerical values at the top corner of the image
    if hasnan:
        img.data[0, 0, 0] = np.nan
        img.data[1, 0, 0] = np.inf
    # Update orientation
    img.change_orientation(orientation)
    img_sub.change_orientation(orientation)
    if debug:
        img_sub.save(f'tmp_dummy_seg_{datetime.now():%Y%m%d%H%M%S%f}.nii.gz')
    return img, img_sub, arr_ctl


# Generate a list of fake centerlines: (dummy_segmentation(params), dict of expected results)
im_ctl_find_and_sort_coord = [
    (dummy_centerline(size_arr=(41, 7, 9), subsampling=1, orientation='LPI'), None),
    ]

im_ctl_zeroslice = [
    (dummy_centerline(size_arr=(15, 7, 9), zeroslice=[0, 1], orientation='LPI'), (3, 7)),
    (dummy_centerline(size_arr=(15, 7, 9), zeroslice=[], orientation='LPI'), (3, 9)),
    ]

im_centerlines = [
    (dummy_centerline(size_arr=(41, 7, 9), subsampling=1, orientation='SAL'),
     {'median': 0, 'rmse': 0.4, 'laplacian': 2, 'norm': 3},
     {}),
    (dummy_centerline(size_arr=(41, 7, 9), pixdim=(0.5, 0.5, 10), subsampling=1, orientation='SAL'),
     {'median': 0, 'rmse': 0.3, 'laplacian': 2, 'norm': 3},
     {}),
    (dummy_centerline(size_arr=(9, 9, 9), subsampling=3),
     {'median': 0, 'rmse': 0.3, 'laplacian': 0.5, 'norm': 2},
     {'exclude_polyfit': True}),  # excluding polyfit because of poorly conditioned fitting
    (dummy_centerline(size_arr=(9, 9, 9), subsampling=1, hasnan=True),
     {'median': 0, 'rmse': 0.3, 'laplacian': 2, 'norm': 1.5},
     {}),
    # (dummy_centerline(size_arr=(30, 20, 9), subsampling=1, outlier=[5]),
    #  {'median': 0, 'rmse': 1, 'laplacian': 5, 'norm': 13.5},
    #  {}),
    (dummy_centerline(size_arr=(30, 20, 50), subsampling=1),
     {'median': 0, 'rmse': 0.3, 'laplacian': 0.5, 'norm': 2.1},
     {}),
    (dummy_centerline(size_arr=(30, 20, 50), subsampling=1, outlier=[20]),
     {'median': 0, 'rmse': 0.8, 'laplacian': 70, 'norm': 14},
     {'exclude_nurbs': True}),
    (dummy_centerline(size_arr=(30, 20, 50), subsampling=3, dilate_ctl=2, orientation='AIL'),
     {'median': 0, 'rmse': 0.25, 'laplacian': 0.2, 'norm': 3},
     {}),
    (dummy_centerline(size_arr=(30, 20, 50), subsampling=5),
     {'median': 0, 'rmse': 0.3, 'laplacian': 0.5, 'norm': 3.6},
     {}),
    (dummy_centerline(size_arr=(30, 20, 50), subsampling=10),
     {'median': 0, 'rmse': 0.1, 'laplacian': 0.5, 'norm': 3.8},
     {}),
]

param_optic = [
    ({'fname_image':
        sct_test_path('t2', 't2.nii.gz'),
      'contrast': 't2',
      'fname_centerline-optic':
        sct_test_path('t2', 't2_centerline-optic.nii.gz')}),
    ({'fname_image':
        sct_test_path('t2s', 't2s.nii.gz'),
      'contrast': 't2s',
      'fname_centerline-optic':
        sct_test_path('t2s/t2s_centerline-optic.nii.gz')}),
    ({'fname_image':
        sct_test_path('dmri', 'dwi_mean.nii.gz'),
      'contrast': 'dwi',
      'fname_centerline-optic':
        sct_test_path('dmri', 'dwi_mean_centerline-optic.nii.gz')}),
]


@pytest.mark.parametrize('img_ctl,expected', im_ctl_find_and_sort_coord)
def test_find_and_sort_coord(img_ctl, expected):
    img = img_ctl[0].copy()
    centermass = find_and_sort_coord(img)
    assert centermass.shape == (3, 9)
    assert np.linalg.norm(centermass - img_ctl[2]) == 0


@pytest.mark.parametrize('img_ctl,expected', im_ctl_zeroslice)
def test_get_centerline_polyfit_minmax(img_ctl, expected):
    """Test centerline fitting with minmax=True"""
    img_sub = img_ctl[1].copy()
    img_out, arr_out, _, _ = get_centerline(
        img_sub, ParamCenterline(algo_fitting='polyfit', degree=3, minmax=True), verbose=VERBOSE)
    # Assess output size
    assert arr_out.shape == expected


@pytest.mark.parametrize('img_ctl,expected,params', im_centerlines)
def test_get_centerline_polyfit(img_ctl, expected, params):
    """Test centerline fitting using polyfit"""
    if params.get('exclude_polyfit', False):
        return
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, arr_deriv_out, fit_results = get_centerline(
        img_sub, ParamCenterline(algo_fitting='polyfit', minmax=False), verbose=VERBOSE)
    assert np.median(find_and_sort_coord(img) - find_and_sort_coord(img_out)) == expected['median']
    assert np.max(np.absolute(np.diff(arr_deriv_out))) < expected['laplacian']
    # sort `arr_out` to match the sorting from `find_and_sort_coord`
    dim_si = [img.orientation.find(x) for x in ['I', 'S'] if img.orientation.find(x) != -1][0]
    sort_indices = np.argsort(arr_out[dim_si])
    arr_out_sorted = arr_out[:, sort_indices]
    assert np.linalg.norm(find_and_sort_coord(img) - arr_out_sorted) < expected['norm']


@pytest.mark.parametrize('img_ctl,expected,params', im_centerlines)
def test_get_centerline_bspline(img_ctl, expected, params):
    """Test centerline fitting using bspline"""
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, arr_deriv_out, fit_results = get_centerline(
        img_sub, ParamCenterline(algo_fitting='bspline', minmax=False), verbose=VERBOSE)
    assert np.median(find_and_sort_coord(img) - find_and_sort_coord(img_out)) == expected['median']
    assert fit_results.rmse < expected['rmse']
    assert fit_results.laplacian_max < expected['laplacian']


@pytest.mark.parametrize('img_ctl,expected,params', im_centerlines)
def test_get_centerline_linear(img_ctl, expected, params):
    """Test centerline fitting using linear interpolation"""
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, arr_deriv_out, fit_results = get_centerline(
        img_sub, ParamCenterline(algo_fitting='linear', minmax=False), verbose=VERBOSE)
    assert np.median(find_and_sort_coord(img) - find_and_sort_coord(img_out)) == expected['median']
    assert fit_results.laplacian_max < expected['laplacian']


@pytest.mark.parametrize('img_ctl,expected,params', im_centerlines)
def test_get_centerline_nurbs(img_ctl, expected, params):
    """Test centerline fitting using nurbs"""
    if params.get('exclude_nurbs', False):
        return
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, arr_deriv_out, fit_results = get_centerline(
        img_sub, ParamCenterline(algo_fitting='nurbs', minmax=False), verbose=VERBOSE)
    assert np.median(find_and_sort_coord(img) - find_and_sort_coord(img_out)) == expected['median']
    assert fit_results.laplacian_max < expected['laplacian']


@pytest.mark.parametrize('params', param_optic)
def test_get_centerline_optic(params):
    """Test centerline extraction with optic"""
    # TODO: add assert on the output .csv files for more precision
    im = Image(params['fname_image'])
    # Add non-numerical values at the top corner of the image for testing purpose
    im.change_type('float32')
    im.data[0, 0, 0] = np.nan
    im.data[1, 0, 0] = np.inf
    im_centerline, arr_out, _, _ = get_centerline(
        im, ParamCenterline(algo_fitting='optic', contrast=params['contrast'], minmax=False), verbose=VERBOSE)
    # Compare with ground truth centerline
    assert np.all(im_centerline.data == Image(params['fname_centerline-optic']).data)
