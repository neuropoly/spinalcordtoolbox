# pytest unit tests for sct_get_centerline

import logging
import os

import numpy as np
import pytest

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_get_centerline
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.parametrize('space', ["pix", "phys"])
def test_sct_get_centerline_output_file_exists(tmp_path, tmp_path_qc, space):
    """This test checks the output file using default usage of the CLI script.

    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)
    Note: There actually used to be a check here, but it was replaced by 'test_get_centerline_optic' in
    'unit_testing/test_centerline.py'. For more details, see:
       * https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2774/commits/5e6bd57abf6bcf825cd110e0d74b8e465d298409
       * https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2774#discussion_r450546434"""
    sct_get_centerline.main(argv=['-i', sct_test_path('t2s', 't2s.nii.gz'), '-c', 't2s', '-space', space,
                                  '-qc', tmp_path_qc])
    for file in [sct_test_path('t2s', 't2s_centerline.nii.gz'), sct_test_path('t2s', 't2s_centerline.csv')]:
        assert os.path.exists(file)


@pytest.mark.sct_testing
@pytest.mark.parametrize('ext', ["", ".nii.gz"])
def test_sct_get_centerline_output_file_exists_with_o_arg(tmp_path, tmp_path_qc, ext):
    """This test checks the '-o' argument with and without an extension to
    ensure that the correct output file is created either way."""
    sct_get_centerline.main(argv=['-i', sct_test_path('t2s', 't2s.nii.gz'), '-c', 't2s', '-qc', tmp_path_qc,
                                  '-o', os.path.join(tmp_path, f't2s_centerline{ext}')])
    for file in [sct_test_path('t2s', 't2s_centerline.nii.gz'), sct_test_path('t2s', 't2s_centerline.csv')]:
        assert os.path.exists(file)


@pytest.mark.sct_testing
def test_sct_get_centerline_soft_sums_to_one_and_overlaps_with_bin(tmp_path, tmp_path_qc):
    """
    This test checks two necessary conditions of the soft centerline:

    1) The sum of the output intensities of the soft centerline is equal to 1 on all slices
    2) The output maximum of the soft centerline overlaps with the binary segmentation on all slices.
    """
    # Condition 1: All slices of the soft centerline sum to 1
    sct_get_centerline.main(argv=['-i', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                  '-method', 'fitseg', '-centerline-soft', '1', '-o',
                                  os.path.join(tmp_path, 't2_seg_centerline_soft.nii.gz'),
                                  '-qc', tmp_path_qc])
    im_soft = Image(os.path.join(tmp_path, 't2_seg_centerline_soft.nii.gz'))
    # Sum soft centerline intensities in the (x,y) plane, across all slices
    sum_over_slices = np.apply_over_axes(np.sum, im_soft.data, [0, 2]).flatten()
    # Test if the summation for each slice is equal to 1
    assert (sum_over_slices == 1).all()

    # Condition 2: The max voxels of the soft centerline overlap with the binary centerline
    sct_get_centerline.main(argv=['-i', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                  '-method', 'fitseg', '-centerline-soft', '0', '-o',
                                  os.path.join(tmp_path, 't2_seg_centerline_bin.nii.gz'),
                                  '-qc', tmp_path_qc])
    im_bin = Image(os.path.join(tmp_path, 't2_seg_centerline_bin.nii.gz'))
    # Find the maximum intensity voxel across all slices in soft centerline and binary centerline
    max_over_slices_soft = np.apply_over_axes(np.max, im_soft.data, [0, 2])
    max_over_slices_bin = np.apply_over_axes(np.max, im_bin.data, [0, 2])
    # Find the coordinates of the maximum in each slice in both soft and binary centerline
    max_coords_over_slices_soft = np.transpose(np.where(im_soft.data == max_over_slices_soft))
    max_coords_over_slices_bin = np.transpose(np.where(im_bin.data == max_over_slices_bin))
    # Test if the maximum are the same between soft and binary centerline for each slices
    assert (max_coords_over_slices_soft == max_coords_over_slices_bin).all()
