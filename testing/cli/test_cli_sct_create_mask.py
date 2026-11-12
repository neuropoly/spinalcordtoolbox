# pytest unit tests for sct_create_mask

import pytest
import logging

from spinalcordtoolbox.scripts import sct_create_mask
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.parametrize("path_input,process,size", [
    (sct_test_path('mt', 'mt1.nii.gz'), 'coord,15x17', '10'),
    (sct_test_path('mt', 'mt1.nii.gz'), f'point,{sct_test_path("mt", "mt1_point.nii.gz")}', '10'),
    (sct_test_path('mt', 'mt1.nii.gz'), 'center', '10'),
    (sct_test_path('mt', 'mt1.nii.gz'), f'centerline,{sct_test_path("mt", "mt1_seg.nii.gz")}', '5'),
    (sct_test_path('dmri', 'dmri.nii.gz'), 'center', '10')
])
def test_sct_create_mask_no_checks(path_input, process, size):
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_create_mask.main(argv=['-i', path_input, '-p', process, '-size', size, '-r', '0'])
