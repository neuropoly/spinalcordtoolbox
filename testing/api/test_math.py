# pytest unit tests for spinalcordtoolbox.math

import os
import datetime

import numpy as np
import nibabel as nib
import pytest

import spinalcordtoolbox.math as sct_math
from spinalcordtoolbox.image import Image

VERBOSE = int(os.getenv('SCT_VERBOSE', 0))
DUMP_IMAGES = bool(os.getenv('SCT_DEBUG_IMAGES', False))


def dummy_blob(size_arr=(9, 9, 9), pixdim=(1, 1, 1), coordvox=None):
    """
    Create an image with a non-null voxels at coordinates specified by coordvox.
    :param size_arr:
    :param pixdim:
    :param coordvox: If None: will create a single voxel in the middle of the FOV.
      If tuple: (x,y,z): Create single voxel at specified coordinate
      If list of tuples: [(x1,y1,z1), (x2,y2,z2)]: Create multiple voxels.
    :return: Image object
    """
    # nx, ny, nz = size_arr
    data = np.zeros(size_arr)
    # if not specified, voxel coordinate is set at the middle of the volume
    if coordvox is None:
        coordvox = tuple([round(i / 2) for i in size_arr])
    elif isinstance(coordvox, list):
        for icoord in coordvox:
            data[icoord] = 1
    elif isinstance(coordvox, tuple):
        data[coordvox] = 1
    else:
        ValueError("Wrong type for coordvox")
    # Create image with default orientation LPI
    affine = np.eye(4)
    affine[0:3, 0:3] = affine[0:3, 0:3] * pixdim
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(data, hdr=nii.header, dim=nii.header.get_data_shape())
    return img


@pytest.mark.parametrize('islabel', [False, True])
def test_dilate(islabel):

    # Create dummy image with single pixel in the middle
    im = dummy_blob(size_arr=(9, 9, 9), coordvox=(4, 4, 4))
    if DUMP_IMAGES:
        im.save(f'tmp_dummy_im_{datetime.now():%Y%m%d%H%M%S%f}.nii.gz')

    # cube (only asserting along one dimension for convenience)
    data_dil = sct_math.dilate(im.data, size=1, shape='cube', islabel=islabel)
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 0, 1, 0, 0]))
    data_dil = sct_math.dilate(im.data, size=2, shape='cube', islabel=islabel)
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 1, 1, 0, 0]))
    data_dil = sct_math.dilate(im.data, size=3, shape='cube', islabel=islabel)
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 1, 1, 1, 0]))

    # ball (only asserting along one dimension for convenience)
    data_dil = sct_math.dilate(im.data, size=0, shape='ball', islabel=islabel)
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 0, 1, 0, 0]))
    data_dil = sct_math.dilate(im.data, size=1, shape='ball', islabel=islabel)
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 1, 1, 1, 0]))
    data_dil = sct_math.dilate(im.data, size=2, shape='ball', islabel=islabel)
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([1, 1, 1, 1, 1]))

    # square in xy plane
    data_dil = sct_math.dilate(im.data, size=1, shape='disk', dim=1, islabel=islabel)
    assert np.array_equal(data_dil[2:7, 4, 4], np.array([0, 1, 1, 1, 0]))
    assert np.array_equal(data_dil[4, 4, 2:7], np.array([0, 1, 1, 1, 0]))
    data_dil = sct_math.dilate(im.data, size=1, shape='disk', dim=2, islabel=islabel)
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 1, 1, 1, 0]))
    assert np.array_equal(data_dil[2:7, 4, 4], np.array([0, 1, 1, 1, 0]))

    # test with Image as input
    im_dil = sct_math.dilate(im, size=1, shape='cube', islabel=islabel)
    assert np.array_equal(im_dil.data[4, 2:7, 4], np.array([0, 0, 1, 0, 0]))


def test_erode():
    # Create dummy image with single pixel
    im = dummy_blob(size_arr=(9, 9, 9), coordvox=(4, 4, 4))
    # Dilate it
    im_dil = sct_math.dilate(im, size=1, shape='ball')
    # Erode it
    im_dil_erode = sct_math.erode(im_dil, size=1, shape='ball')
    assert np.array_equal(np.where(im_dil_erode.data), (np.array([4]), np.array([4]), np.array([4])))
    # Test with data as input
    data_dil_erode = sct_math.erode(im_dil.data, size=1, shape='ball')
    assert np.array_equal(np.where(data_dil_erode), (np.array([4]), np.array([4]), np.array([4])))


def test_threshold():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    b = sct_math.threshold(a.copy(), 3)
    assert (b == np.array([0, 0, 3, 4, 5, 6, 7, 8, 9])).all()

    c = sct_math.threshold(a.copy(), uthr=5)
    assert (c == np.array([1, 2, 3, 4, 5, 0, 0, 0, 0])).all()

    d = sct_math.threshold(a.copy(), lthr=3, uthr=5)
    assert (d == np.array([0, 0, 3, 4, 5, 0, 0, 0, 0])).all()

    e = sct_math.threshold(a.copy())
    assert (e == a).all()
