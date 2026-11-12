# pytest unit tests for sct_image

import logging
import os

import pytest
import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.fs import extract_fname
from spinalcordtoolbox.utils.sys import sct_test_path
from spinalcordtoolbox.scripts import sct_image, sct_crop_image

logger = logging.getLogger(__name__)


@pytest.fixture
def dmri_in():
    """Filepath for input 4D dMRI image."""
    return sct_test_path('dmri', 'dmri.nii.gz')


@pytest.fixture
def dmri_t_slices(tmp_path, dmri_in):
    """Filepaths for 4D dMRI image split across t axis."""
    fname_out = str(tmp_path / 'dmri.nii.gz')
    sct_image.main(argv=['-i', dmri_in, '-split', 't', '-o', fname_out])

    parent, filename, ext = extract_fname(fname_out)
    fname_slices = [os.path.join(parent, f'{filename}_T{i:04}{ext}') for i in range(7)]
    return fname_slices


@pytest.mark.sct_testing
def test_sct_image_split_dmri(dmri_t_slices):
    """Verify the output of '-split' matches reference image. Note: CLI script is run by the 'dmri_t_slices' fixture."""
    _, filename, ext = extract_fname(dmri_t_slices[0])
    ref = Image(sct_test_path('dmri', f'{filename}{ext}'))  # Reference image should exist inside working directory (sct_testing_data)
    new = Image(dmri_t_slices[0])         # New image should be generated inside tmp directory
    assert np.linalg.norm(ref.data - new.data) == 0


@pytest.mark.sct_testing
def test_sct_image_concat_dmri(tmp_path, dmri_t_slices, dmri_in):
    """Run the CLI script and verify concatenated image matches reference image."""
    path_out = str(tmp_path / 'dmri_concat.nii.gz')
    sct_image.main(argv=['-i'] + dmri_t_slices + ['-concat', 't', '-o', path_out])
    ref = Image(dmri_in)
    new = Image(path_out)
    assert np.linalg.norm(ref.data - new.data) == 0


@pytest.mark.sct_testing
@pytest.mark.parametrize("path_in", [sct_test_path('t2', 't2.nii.gz'),       # 3D
                                     sct_test_path('dmri', 'dmri.nii.gz')])  # 4D
def test_sct_image_getorient(path_in):
    """Run the CLI script and ."""
    sct_image.main(argv=['-i', path_in, '-getorient'])


@pytest.mark.sct_testing
def test_sct_image_pad():
    """Run the CLI script and test the '-pad' option."""
    pad = 2
    path_in = sct_test_path('mt', 'mtr.nii.gz')  # 3D
    path_out = 'sct_image_out.nii.gz'
    sct_image.main(argv=['-i', path_in, '-o', 'sct_image_out.nii.gz', '-pad', f'0,0,{pad}'])

    # z axis (dim[2]) should be padded, but all other values should be unchanged
    expected_dim = Image(path_in).dim[:2] + (Image(path_in).dim[2] + (2 * pad),) + Image(path_in).dim[3:]
    assert Image(path_out).dim == expected_dim


@pytest.mark.parametrize("output_format", ('sct', 'fslhd', 'nibabel'))
def test_sct_image_show_header_no_checks(output_format):
    """Run the CLI script without checking results. The rationale for not checking results is
    provided here: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3317#issuecomment-811429547"""
    sct_image.main(argv=['-i', sct_test_path('t2', 't2.nii.gz'), '-header', output_format])


def test_sct_image_display_warp_check_output_exists():
    """Run the CLI script and check that the warp image file was created."""
    fname_in = 'warp_template2anat.nii.gz'
    fname_out = f'grid_3_resample_{fname_in}'
    sct_image.main(argv=['-i', sct_test_path('t2', fname_in), '-display-warp'])
    assert os.path.exists(sct_test_path('t2', fname_out))


def test_sct_image_stitch(tmp_path, tmp_path_qc):
    """Run the CLI script and check that the stitched file was generated."""
    # crop images for testing stitching function
    path_in = sct_test_path('t2', 't2.nii.gz')
    fname_roi1 = os.path.join(tmp_path, 't2_roi1.nii.gz')
    fname_roi2 = os.path.join(tmp_path, 't2_roi2.nii.gz')
    sct_crop_image.main(argv=['-i', path_in, '-o', fname_roi1, '-xmin', '0', '-xmax', '59',
                              '-ymin', '20', '-ymax', '40', '-zmin', '0', '-zmax', '51'])
    sct_crop_image.main(argv=['-i', path_in, '-o', fname_roi2, '-xmin', '0', '-xmax', '59',
                              '-ymin', '0', '-ymax', '20', '-zmin', '0', '-zmax', '51'])

    fname_out = os.path.join(tmp_path, 'stitched.nii.gz')
    sct_image.main(argv=['-i', fname_roi1, fname_roi2, '-o', fname_out, '-stitch', '-qc', tmp_path_qc])
    assert os.path.exists(fname_out)
