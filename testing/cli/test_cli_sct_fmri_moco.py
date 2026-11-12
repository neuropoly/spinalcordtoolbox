# pytest unit tests for sct_fmri_moco

import numpy as np
import pytest
import logging

from spinalcordtoolbox.scripts import sct_fmri_moco, sct_deepseg, sct_maths, sct_create_mask, sct_image, sct_apply_transfo
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.fixture(scope='module')
def fmri_mean(tmp_path_factory, tmp_path_qc):
    """Mean segmented image for QC report generation."""
    tmp_path = tmp_path_factory.mktemp('fmri_mean')
    path_out = str(tmp_path / 'fmri_mean.nii.gz')

    sct_maths.main(argv=['-i', sct_test_path('fmri', 'fmri.nii.gz'), '-mean', 't', '-o', path_out])
    return path_out


@pytest.fixture(scope='module')
def fmri_mean_seg(tmp_path_factory, tmp_path_qc, fmri_mean):
    """Mean segmented image for QC report generation."""
    tmp_path = tmp_path_factory.mktemp('fmri_mean_seg')
    path_out = str(tmp_path / 'fmri_mean_seg.nii.gz')

    sct_deepseg.main(argv=['spinalcord', '-i', fmri_mean, '-o', path_out, '-qc', str(tmp_path_qc)])
    return path_out


@pytest.fixture(scope='module')
def fmri_mask(tmp_path_factory, fmri_mean, fmri_mean_seg):
    """Mask image for testing."""
    tmp_path = tmp_path_factory.mktemp('fmri_mask')
    path_out = str(tmp_path / 'fmri_mask.nii.gz')
    sct_create_mask.main(argv=['-i', fmri_mean, '-p', f'centerline,{fmri_mean_seg}', '-size', '35mm', '-o', path_out])
    return path_out


@pytest.mark.sct_testing
def test_sct_fmri_moco_no_checks(tmp_path_qc, fmri_mean_seg):
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_fmri_moco.main(argv=['-i', sct_test_path('fmri', 'fmri_r.nii.gz'), '-g', '4', '-x', 'nn', '-r', '0',
                             '-qc', tmp_path_qc, '-qc-seg', fmri_mean_seg])


@pytest.mark.parametrize('target_vol', ['ref', 0, 5])
def test_sct_fmri_moco_target_volume(capsys, tmp_path_qc, fmri_mean, fmri_mean_seg, target_vol):
    """Run the CLI script against a specified target volume (either `-ref` or `-param num_target`)."""
    # run `sct_fmri_moco` with a given target volume
    target_args = ['-ref', fmri_mean] if target_vol == 'ref' else ['-param', f'num_target={target_vol}']
    sct_fmri_moco.main(argv=['-i', sct_test_path('fmri', 'fmri_r.nii.gz'), '-g', '4', '-x', 'nn', '-r', '0',
                             '-qc', tmp_path_qc, '-qc-seg', fmri_mean_seg] + target_args)

    # make sure the target volume was mentioned in the logging output
    captured = capsys.readouterr()
    target_msgs = [line for line in captured.out.split("\n") if line.startswith("Target:")]
    assert len(target_msgs) > 0, "No target message found in output"
    assert str(target_vol) in target_msgs[0], f"Expected target volume {target_vol} not found in output"


@pytest.mark.parametrize("group_size", [-1, 0, 1.5, 'NaN'])
def test_sct_fmri_moco_invalid_group_values(tmp_path, tmp_path_qc, fmri_mean_seg, group_size):
    """Ensure that invalid group sizes return a parsing error."""
    with pytest.raises(SystemExit) as e:
        sct_fmri_moco.main(argv=['-i', sct_test_path('fmri', 'fmri.nii.gz'),
                                 '-g', str(group_size), '-x', 'nn', '-r', '0',
                                 '-ofolder', str(tmp_path),
                                 '-qc', tmp_path_qc, '-qc-seg', fmri_mean_seg])
    assert e.value.code == 2


@pytest.mark.sct_testing
@pytest.mark.parametrize("ref", [True, False])
def test_sct_fmri_moco_dl(tmp_path_qc, fmri_mean, fmri_mask, tmp_path, fmri_mean_seg, ref):
    """Run the CLI script with '-m' and optionally '-ref' using '-dl' algorithm."""
    fname_data = sct_test_path('fmri', 'fmri.nii.gz')
    argv = ['-i', fname_data, '-m', fmri_mask,
            '-ofolder', str(tmp_path), '-dl',
            '-qc', tmp_path_qc, '-qc-seg', fmri_mean_seg]
    if ref:
        argv.extend(['-ref', fmri_mean])
    sct_fmri_moco.main(argv=argv)

    def build_disp_from_params(fname_tx, fname_ty, fname_ref, fname_warp):
        """Re-build displacement field from translation parameters; Tx and Ty"""
        tx = Image(fname_tx).data[0, 0, :, :]  # (Z,T)
        ty = Image(fname_ty).data[0, 0, :, :]  # (Z,T)
        ref = Image(fname_ref)
        nx, ny, nz, nt = ref.data.shape
        disp = np.zeros((nx, ny, nz, nt, 3), dtype=np.float32)
        for t in range(nt):
            for z in range(nz):
                disp[:, :, z, t, 0] = tx[z, t]
                disp[:, :, z, t, 1] = ty[z, t]
                disp[:, :, z, t, 2] = 0.0
        im_disp = Image(disp, hdr=ref.hdr)
        im_disp.affine = ref.affine
        im_disp.hdr.set_data_shape(disp.shape)
        im_disp.hdr.set_intent('vector', (), '')
        im_disp.save(fname_warp)

    sct_image.main(argv=['-i', fname_data, '-split', 't', '-o', str(tmp_path / "fmri.nii.gz")])
    # test on the first volume of fmri
    t = 0
    fname_vol = str(tmp_path / f"fmri_T{t:04d}.nii.gz")
    fname_warp_4d = str(tmp_path / "warp_fmri_moco.nii.gz")
    # re-build displacement field
    build_disp_from_params(fname_tx=str(tmp_path / "moco_params_x.nii.gz"),
                           fname_ty=str(tmp_path / "moco_params_y.nii.gz"),
                           fname_ref=fname_data, fname_warp=fname_warp_4d)
    sct_image.main(argv=['-i', fname_warp_4d, '-split', 't'])
    fname_warp = str(tmp_path / f"warp_fmri_moco_T{t:04d}.nii.gz")
    fname_out = str(tmp_path / f"fmri_warped_T{t:04d}.nii.gz")

    sct_apply_transfo.main(argv=['-i', fname_vol, '-d', fmri_mean,
                                 '-w', fname_warp, '-o', fname_out, '-x', 'linear'])
    warped = Image(fname_out).data.astype(np.float32)
    moco_dl = Image(str(tmp_path / "fmri_mocoDL.nii.gz")).data.astype(np.float32)

    moco_vol0 = moco_dl[..., 0]
    # pick a representative slice: middle slice
    z = warped.shape[2] // 2
    # compare using phase cross-correlation
    from skimage.registration import phase_cross_correlation
    shift, _, _ = phase_cross_correlation(warped[:, :, z], moco_vol0[:, :, z], normalization=None, upsample_factor=10)
    print("shift:", shift)
    assert np.max(np.abs(shift)) < 1.0
