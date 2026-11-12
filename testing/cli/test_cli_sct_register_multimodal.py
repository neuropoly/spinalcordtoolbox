# pytest unit tests for sct_register_multimodal

import os

import pytest
import logging
import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import sct_dir_local_path, sct_test_path
from spinalcordtoolbox.scripts import sct_register_multimodal, sct_create_mask

logger = logging.getLogger(__name__)


def test_sct_register_multimodal_mask_files_exist(tmp_path, tmp_path_qc):
    """
    Run the script without validating results.

    - TODO: Write a check that verifies the registration results.
    - TODO: Parametrize this test to add '-initwarpinv warp_anat2template.nii.gz',
            after the file is added to sct_testing_data:
            https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3407#discussion_r646895013
    """
    fname_mask = str(tmp_path/'mask_mt1.nii.gz')
    sct_create_mask.main(['-i', sct_test_path('mt', 'mt1.nii.gz'),
                          '-p', f"centerline,{sct_test_path('mt', 'mt1_seg.nii.gz')}",
                          '-size', '35mm', '-f', 'cylinder', '-o', fname_mask])
    sct_register_multimodal.main([
        '-i', sct_dir_local_path('data/PAM50/template/', 'PAM50_t2.nii.gz'),
        '-iseg', sct_dir_local_path('data/PAM50/template/', 'PAM50_cord.nii.gz'),
        '-d', sct_test_path('mt', 'mt1.nii.gz'),
        '-dseg', sct_test_path('mt', 'mt1_seg.nii.gz'),
        '-param', 'step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3',
        '-m', fname_mask,
        '-initwarp', sct_test_path('t2', 'warp_template2anat.nii.gz'),
        '-ofolder', str(tmp_path), '-qc', tmp_path_qc
    ])

    for path in ["PAM50_t2_reg.nii.gz", "warp_PAM50_t22mt1.nii.gz"]:
        assert os.path.exists(tmp_path/path)

    # Because `-initwarp` was specified (but `-initwarpinv` wasn't) the dest->seg files should NOT exist
    for path in ["mt1_reg.nii.gz", "warp_mt12PAM50_t2.nii.gz"]:
        assert not os.path.exists(tmp_path/path)


@pytest.mark.sct_testing
@pytest.mark.parametrize("use_seg,param,fname_gt", [
    (False, 'step=1,algo=syn,type=im,iter=1,smooth=1,shrink=2,metric=MI',
     sct_test_path('mt', 'mt0_reg_syn_goldstandard.nii.gz')),
    (False, 'step=1,algo=slicereg,type=im,iter=5,smooth=0,metric=MeanSquares',
     sct_test_path('mt', 'mt0_reg_slicereg_goldstandard.nii.gz')),
    (False, 'step=1,algo=dl,type=im', None),
    (True, 'step=1,algo=centermassrot,type=seg,rot_method=pca', None),
    (True, 'step=1,algo=centermassrot,type=imseg,rot_method=hog', None),
    (True, 'step=1,algo=centermassrot,type=imseg,rot_method=pcahog', None),
    (True, 'step=1,algo=columnwise,type=seg,smooth=1', None),
])
def test_sct_register_multimodal_mt0_image_data_within_threshold(use_seg, param, fname_gt, tmp_path, tmp_path_qc):
    """Run the CLI script and verify that the output image data is close to a reference image (within threshold)."""
    fname_out_src = str(tmp_path/'mt0_reg.nii.gz')
    fname_out_dest = str(tmp_path/'mt0_reg_inv.nii.gz')
    fname_owarp = str(tmp_path/'warp_mt02mt1.nii.gz')
    fname_owarpinv = str(tmp_path/'warp_mt12mt0.nii.gz')

    argv = ['-i', sct_test_path('mt', 'mt0.nii.gz'),
            '-d', sct_test_path('mt', 'mt1.nii.gz'), '-x', 'linear', '-r', '0', '-param', param,
            '-o', fname_out_src, '-owarp', fname_owarp, '-owarpinv', fname_owarpinv]
    seg_argv = ['-iseg', sct_test_path('mt', 'mt0_seg.nii.gz'),
                '-dseg', sct_test_path('mt', 'mt1_seg.nii.gz'),
                '-qc', tmp_path_qc]  # qc requires -dseg
    sct_register_multimodal.main(argv=(argv + seg_argv) if use_seg else argv)

    for f in [fname_out_src, fname_out_dest, fname_owarp, fname_owarpinv]:
        assert os.path.isfile(f)

    # This check is skipped because of https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3372
    #############################################################################################
    # if fname_gt is not None:
    #     im_gt = Image(fname_gt)
    #     im_result = Image(fname_out)
    #     # get dimensions
    #     nx, ny, nz, nt, px, py, pz, pt = im_gt.dim
    #     # set the difference threshold to 1e-3 pe voxel
    #     threshold = 1e-3 * nx * ny * nz * nt
    #     # check if non-zero elements are present when computing the difference of the two images
    #     diff = im_gt.data - im_result.data
    #     # compare images
    #     assert abs(np.sum(diff)) < threshold  # FIXME: Use np.linalg.norm when this test is fixed


def test_sct_register_multimodal_with_softmask(tmp_path, tmp_path_qc):
    """
    Verify that softmask is actually applied during registration.

    NB: For 'gaussian', ANTs binaries can't handle softmasks natively, so SCT should be applying
        the mask directly to the image. Related links:
            * https://github.com/ivadomed/pipeline-hemis/issues/3.
            * https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3075
    """
    fname_mask = str(tmp_path/'mask_t2.nii.gz')
    fname_t2 = sct_test_path('t2', 't2.nii.gz')
    fname_t1 = sct_test_path('t1', 't1w.nii.gz')
    fname_warp = str(tmp_path/"warp_t1w2t2.nii.gz")

    sct_create_mask.main(['-i', fname_t2, '-p', f"centerline,{sct_test_path('t2', 't2_centerline-manual.nii.gz')}",
                          '-o', fname_mask, '-f', 'gaussian'])
    sct_register_multimodal.main(['-i', fname_t1, '-d', fname_t2, '-dseg', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                  '-param', "step=1,type=im,algo=slicereg,metric=CC", '-m', fname_mask,
                                  '-ofolder', str(tmp_path), '-r', '0', '-v', '2', '-qc', tmp_path_qc])

    # If registration was successful, the warping field should be non-empty
    assert np.any(Image(fname_warp).data)

    # TODO: Find a way to validate the quality of the registration to see if the mask actually has a benefit.
    #       The problem is, adding a mask seems to make the registration _worse_ for this specific data/params.
    #       (Dice score == 0.9 for no mask, and 0.85 for 'gaussian', and 0.7 for 'cylinder'.)
    #       Nonetheless, below is a rough sketch of what a test could look like:
    # from spinalcordtoolbox.math import dice
    # from spinalcordtoolbox.deepseg_sc.core import deep_segmentation_spinalcord
    # fname_t1_reg = str(tmp_path / "t1w_reg.nii.gz")
    # im_t1_reg_seg, _, _ = deep_segmentation_spinalcord(Image(fname_t1_reg), contrast_type='t1', ctr_algo='svm')
    # im_t2_seg, _, _ = deep_segmentation_spinalcord(Image(fname_t2), contrast_type='t2', ctr_algo='svm')
    # dice_score_t1_reg = dice(im_t2_seg.data, im_t1_reg_seg.data)
    # assert dice_score_t1_reg > 0.9


@pytest.mark.parametrize('algo', [',algo=rigid', ''])
def test_sct_register_multimodal_with_labels(capsys, tmp_path, tmp_path_qc, algo):
    """
    Test registration with '-param type=label' set.

    NB: Label-based registration is a little different from normal registration.
    The path of execution goes from 'register()' onto 'register_step_label()'
    and then 'register_landmarks()', which is its own ITK-based landmarks
    registration function separate from ANTs that entirely ignores the choice of 'algo'.

    Because of this, we run registration with and without 'algo' set, and ensure
    that 'algo' really is ignored, but that registration doesn't actually fail.

    See https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3893 for bug context.
    """
    # NB: Registering the t2 image with itself is non-representative, but it's the only
    #     sct_testing_data image we have that has an associated vertebral label file.
    sct_register_multimodal.main(['-i', sct_test_path('t2', 't2.nii.gz'),
                                  '-d', sct_test_path('t2', 't2.nii.gz'),
                                  '-ilabel', sct_test_path('t2', 'labels.nii.gz'),
                                  '-dlabel', sct_test_path('t2', 'labels.nii.gz'),
                                  '-param', f'step=0,type=label{algo}',
                                  '-ofolder', str(tmp_path)])
    for file in ['t2_dest_reg.nii.gz', 't2_src_reg.nii.gz', 'warp_t22t2.nii.gz']:
        assert os.path.isfile(tmp_path / file)
    # NB: Right now, a warning will be thrown regardless of whether `algo` is explicitly
    #     specified by the user, because `algo` has a default, non-empty setting.
    assert "has no effect for 'type=label' registration." in capsys.readouterr().out


def test_sct_register_multimodal_with_qc_without_dseg(capsys, tmp_path, tmp_path_qc):
    """
    Test if an error is raised when using QC ('-qc') without providing a destination segmentation ('-dseg').
    """
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        sct_register_multimodal.main(['-i', sct_test_path('t2', 't2.nii.gz'),
                                      '-d', sct_test_path('t2', 't2.nii.gz'),
                                      '-ofolder', str(tmp_path),
                                      '-qc', tmp_path_qc])
    assert pytest_wrapped_e.value.code == 2
