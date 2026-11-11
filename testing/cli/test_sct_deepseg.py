# pytest unit tests for spinalcordtoolbox.deepseg

import json
import os
import shlex
import shutil
from pathlib import Path
from unittest import mock

import pytest
import warnings
import numpy as np
from torch.serialization import SourceChangeWarning

import spinalcordtoolbox as sct
from spinalcordtoolbox.image import Image, compute_dice, add_suffix, check_image_kind
from spinalcordtoolbox.utils.sys import sct_test_path, __deepseg_dir__
import spinalcordtoolbox.deepseg.models

from spinalcordtoolbox.scripts import sct_deepseg, sct_resample, sct_qc


def test_model_dict():
    """
    Make sure all fields are present in each model.
    :return:
    """
    for key, value in sct.deepseg.models.MODELS.items():
        assert 'url' in value
        assert 'description' in value
        assert 'default' in value


@pytest.fixture()
def cleanup_model_dirs():
    """Fixture to clean up model directory after tests."""
    model_dirs = os.listdir(__deepseg_dir__)
    yield
    new_dirs = set(os.listdir(__deepseg_dir__)) - set(model_dirs)
    for dir_name in new_dirs:
        shutil.rmtree(os.path.join(__deepseg_dir__, dir_name))


@pytest.mark.parametrize('fname_image, fname_seg_manual, fname_out, task, thr, expected_dice', [
    (sct_test_path('t2s', 't2s_uncropped.nii.gz'),
     sct_test_path('t2s', 't2s_uncropped_gmseg_manual.nii.gz'),
     't2s_uncropped_seg_deepseg.nii.gz',
     'graymatter',
     None,
     0.91),  # Dice for GM is harder than SC seg due to complex GM shape
    (sct_test_path('t2', 't2.nii.gz'),
     sct_test_path('t2', 't2_seg-manual.nii.gz'),
     't2_seg_deepseg.nii.gz',
     'spinalcord',
     None,
     0.95),
    (sct_test_path('t2', 't2.nii.gz'),
     sct_test_path('t2', 't2_seg-deepseg_rootlets.nii.gz'),
     't2_seg_deepseg.nii.gz',
     'rootlets',
     None,
     None),  # no Dice score for rootlets model (we just make sure all the labels are present)
    (sct_test_path('t2', 't2.nii.gz'),  # dummy image since no EPI test data
     None,  # no ground truth, just test if it runs
     't2_seg_deepseg.nii.gz',
     'sc_epi',
     None,
     None),
    (sct_test_path('t2', 't2.nii.gz'),  # dummy image since no MP2RAGE test data
     None,  # no ground truth, just test if it runs
     't2_seg_deepseg.nii.gz',
     'lesion_ms_mp2rage',
     None,
     None),
    (sct_test_path('t2', 't2.nii.gz'),
     None,  # no ground truth, just test if it runs
     't2_seg_deepseg.nii.gz',
     'lesion_ms',
     None,
     None),
    (sct_test_path('t2', 't2.nii.gz'),
     None,  # no ground truth, just test if it runs
     't2_seg_deepseg.nii.gz',
     'canal',
     None,
     None),
])
@pytest.mark.usefixtures(cleanup_model_dirs.__name__)
def test_segment_nifti_binary_seg(fname_image, fname_seg_manual, fname_out, task, thr, expected_dice,
                                  tmp_path, tmp_path_qc):
    """
    Test binary output (produced using values other than `-thr 0`) with sct_deepseg postprocessing CLI arguments.
    """
    # Ignore warnings from ivadomed model source code changing
    warnings.filterwarnings("ignore", category=SourceChangeWarning)
    fname_out = str(tmp_path/fname_out)  # tmp_path for automatic cleanup
    args = [task, '-i', fname_image, '-o', fname_out, '-qc', tmp_path_qc]
    if thr is not None:
        args.extend(['-thr', str(thr)])
    if 'sc_' in task:
        # TODO: Replace the "general" testing of these arguments with specific tests with specific input data
        args.extend(['-largest', '1', '-fill-holes', '1', '-remove-small', '5mm3'])

    # try out `use_mirroring` for `lesion_ms` model only (due to longer inference time)
    # based on https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4995#issuecomment-3410672883
    # FIXME: This takes upwards of 1-2 hours to complete based on OS. Disable for now until we can speed up testing.
    # if task == 'lesion_ms':
    #     args.extend(['-test-time-aug'])

    sct_deepseg.main(argv=args)
    # Make sure output file exists
    assert os.path.isfile(fname_out)
    # Compare with ground-truth segmentation if provided
    if fname_seg_manual:
        im_seg = Image(fname_out)
        im_seg_manual = Image(fname_seg_manual)
        output_type = check_image_kind(im_seg_manual)
        if output_type in ['seg', 'softseg']:
            dice_segmentation = compute_dice(im_seg, im_seg_manual, mode='3d', zboundaries=False)
            assert dice_segmentation > expected_dice
        else:
            # Confirm the output type is a labelled segmentation instead
            assert output_type == 'seg-labeled', f"ground truth is unexpected type {output_type}"

            # Get all labels in the segmentation, and those that we expect to see
            expected_labels = {coord.value for coord in im_seg_manual.getCoordinatesAveragedByValue()}
            detected_labels = {coord.value for coord in im_seg.getCoordinatesAveragedByValue()}

            # See if any labels we expected to see are missing
            missing_labels = expected_labels - detected_labels
            if len(missing_labels) > 0:
                pytest.fail(f"Test expected label(s) '{missing_labels}' in segmentation which were not present.")

            # See if any labels we weren't expecting showed up.
            unexpected_labels = detected_labels - expected_labels
            if len(unexpected_labels) > 0:
                warnings.warn(
                    f"Test produced label(s) '{unexpected_labels}' in segmentation which were not expected to appear."
                )

    # Generate secondary (sct_qc) QC report for the segmentation, should be identical to normal `-qc`
    sct_qc.main(argv=['-i', fname_image, '-s', fname_out,
                      '-p', 'sct_deepseg', '-deepseg-task', task,
                      '-qc', tmp_path_qc])


@pytest.fixture(scope='session')
def t2_ax(tmp_path_factory):
    """Generate an approximation of an axially-acquired T2w anat image using resampling."""
    tmp_path = tmp_path_factory.mktemp('t2_ax')
    fname_out = str(tmp_path / 't2_ax.nii.gz')
    sct_resample.main(argv=["-i", sct_test_path('t2', 't2.nii.gz'), "-o", fname_out,
                            "-mm", "0.8x3x0.8", "-x", "spline"])
    return fname_out


@pytest.fixture(scope='session')
def t2_ax_sc_seg(tmp_path_factory):
    """Generate an approximation of an axially-acquired T2w segmentation using resampling."""
    tmp_path = tmp_path_factory.mktemp('t2_ax')
    fname_out = str(tmp_path / 't2_ax_sc_seg.nii.gz')
    sct_resample.main(argv=["-i", sct_test_path('t2', 't2_seg-manual.nii.gz'), "-o", fname_out,
                            "-mm", "0.8x3x0.8", "-x", "spline"])
    return fname_out


@pytest.mark.parametrize('fname_image, fnames_seg_manual, fname_out, suffixes, task, thr, expected_dice, extra_args', [
    (sct_test_path('t2', 't2_fake_lesion.nii.gz'),
     [sct_test_path('t2', 't2_fake_lesion_sc_seg.nii.gz'),
      sct_test_path('t2', 't2_fake_lesion_lesion_seg.nii.gz')],
     't2_deepseg.nii.gz',
     ["_sc_seg", "_lesion_seg"],
     'lesion_sci_t2',
     0.5,
     0.95,
     []),
    (t2_ax,          # Generate axial images on the fly
     [t2_ax_sc_seg,  # Just test against SC ground truth, because the model generates SC segs well
      None],         # The model performs poorly on our fake t2_ax() image, so skip evaluating on lesion seg
     't2_deepseg.nii.gz',
     ["_sc_seg", "_lesion_seg"],
     'lesion_ms_axial_t2',
     0.5,
     0.94,  # axial model is just barely under .95, so we'll accept .94
     []),
    (sct_test_path('t1', 't1_mouse.nii.gz'),
     [None, None],
     't1_deepseg.nii.gz',
     ["_GM_seg", "_WM_seg"],
     'gm_wm_mouse_t1',
     0.5,
     None,
     []),
    (sct_test_path('t2', 't2.nii.gz'),
     [None, None],
     't2_deepseg.nii.gz',
     ["_totalspineseg_discs", "_totalspineseg_all"],
     'spine',
     0,
     None,
     []),
    (sct_test_path('t2', 't2.nii.gz'),
     [None, None],
     't2_deepseg.nii.gz',
     ["_totalspineseg_discs", "_totalspineseg_all"],
     'spine',
     0,
     None,
     ["-label-vert", "1"]),
])
@pytest.mark.usefixtures(cleanup_model_dirs.__name__)
def test_segment_nifti_multiclass(fname_image, fnames_seg_manual, fname_out, suffixes, task, thr, expected_dice,
                                  extra_args, tmp_path, tmp_path_qc, request):
    """
    Uses the locally-installed sct_testing_data
    """
    # Skip mouse test if the file is not present locally
    # (We do not include the file in sct_testing_data as A. the mouse image is large and B. inference time is lengthy.)
    # If testing locally, you can get this file from our internal testing dataset -> copy to sct_testing_data/t1/
    # More info here: https://github.com/spinalcordtoolbox/spinalcordtoolbox/wiki/Testing%253A-Datasets
    if "mouse" in task and not os.path.exists(fname_image):
        pytest.skip("Mouse data must be manually downloaded to run this test.")
    # Fixtures can't be used in parametrization (https://stackoverflow.com/q/42014484)
    # So, we have to evaluate the fixture (i.e. generate the axial images) at test-time
    if "lesion_ms_axial_t2" in task:
        fname_image = request.getfixturevalue(fname_image.__name__)
        fnames_seg_manual[0] = request.getfixturevalue(fnames_seg_manual[0].__name__)

    fname_out = str(tmp_path / fname_out)
    sct_deepseg.main([task, '-i', fname_image, '-thr', str(thr), '-o', fname_out, '-qc', tmp_path_qc,
                      '-largest', '1'] + extra_args)
    # The `-o` argument takes a single filename, even though one (or more!) files might be output.
    # If multiple output files will be produced, `sct_deepseg` will take this singular `-o` and add suffixes to it.
    fnames_out = [add_suffix(fname_out, suffix) for suffix in suffixes]
    for fname_out, fname_seg_manual in zip(fnames_out, fnames_seg_manual):
        # Make sure output file exists
        assert os.path.isfile(fname_out)
        # Compare with ground-truth segmentation if provided
        if fname_seg_manual:
            im_seg = Image(fname_out)
            im_seg_manual = Image(fname_seg_manual)
            dice_segmentation = compute_dice(im_seg, im_seg_manual, mode='3d', zboundaries=False)
            assert dice_segmentation > expected_dice

    # Generate secondary (sct_qc) QC report for the segmentation, should be identical to normal `-qc`
    qc_args = ['-i', fname_image, '-s', fnames_out[0], '-s2', fnames_out[1],
               '-p', 'sct_deepseg', '-deepseg-task', task, '-qc', tmp_path_qc]
    sct_qc.main(argv=qc_args)


@pytest.mark.parametrize("qc_plane", ["Axial", "Sagittal"])
@pytest.mark.usefixtures(cleanup_model_dirs.__name__)
def test_deepseg_with_cropped_qc(qc_plane, tmp_path, tmp_path_qc):
    """
    Test that `-qc-seg` cropping works with both Axial and Sagittal QCs.
    """
    fname_out = str(tmp_path / "t2_deepseg.nii.gz")
    sct_deepseg.main(['lesion_sci_t2',
                      '-i', sct_test_path('t2', 't2_fake_lesion.nii.gz'),
                      '-o', fname_out,
                      '-qc', tmp_path_qc,
                      '-qc-plane', qc_plane,
                      '-qc-seg', sct_test_path('t2', 't2_fake_lesion_sc_seg.nii.gz')])


@pytest.fixture(scope='session')
def t2_zero(tmp_path_factory):
    """
    Create an empty-array version of `t2.nii.gz` to test failure case.
    """
    tmp_path = tmp_path_factory.mktemp('t2_zero')
    fname_out = str(tmp_path / 't2_zero.nii.gz')

    img = Image(sct_test_path('t2', 't2.nii.gz'))
    img.data = np.zeros_like(img.data)
    img.save(fname_out)

    return fname_out


def test_deepseg_totalspineseg_empty_output(t2_zero, tmp_path, tmp_path_qc):
    """
    Test that passing an empty input image will properly fail.
    """
    fname_out = str(tmp_path / "t2_deepseg.nii.gz")
    with pytest.raises(ValueError) as e:
        sct_deepseg.main(['spine',
                          '-i', t2_zero,
                          '-o', fname_out,
                          '-qc', tmp_path_qc])
    assert "step 1 failed to produce a valid segmentation" in str(e.value)


@pytest.mark.parametrize('box_overrides', [
    [],  # default: sc-crop pipeline via automatic detection only, no override
    ['-box-zmin', '0'],  # partial override: widen one face beyond the detected box
    ['-box-xmin', '10', '-box-xmax', '59', '-box-ymin', '0', '-box-ymax', '54',
     '-box-zmin', '3', '-box-zmax', '48'],  # all 6 faces given: skips detection entirely
])
@pytest.mark.usefixtures(cleanup_model_dirs.__name__)
def test_deepseg_crop_box_override(box_overrides, tmp_path, tmp_path_qc):
    """
    Test the sc-crop pipeline (spinalcord task) with no override, a partial `-box-*` override
    (patched onto the detected box), and a full `-box-*` override (skips detection entirely).
    """
    fname_out = str(tmp_path / "t2_seg_deepseg.nii.gz")
    sct_deepseg.main(['spinalcord', '-i', sct_test_path('t2', 't2.nii.gz'), '-o', fname_out,
                      '-qc', tmp_path_qc] + box_overrides)

    assert os.path.isfile(fname_out)
    assert os.path.isfile(add_suffix(fname_out, "_cropbox"))  # crop-active models always produce this

    im_seg = Image(fname_out)
    im_seg_manual = Image(sct_test_path('t2', 't2_seg-manual.nii.gz'))
    dice_segmentation = compute_dice(im_seg, im_seg_manual, mode='3d', zboundaries=False)
    assert dice_segmentation > 0.95


@pytest.mark.usefixtures(cleanup_model_dirs.__name__)
def test_deepseg_full_box_override_skips_detection(tmp_path, tmp_path_qc):
    """
    When all 6 `-box-*` faces are given, sc_crop.detect() should not be called at all --
    there's no point running detection just to overwrite every value it would have produced.
    """
    fname_out = str(tmp_path / "t2_seg_deepseg.nii.gz")
    with mock.patch("spinalcordtoolbox.deepseg.inference.sc_crop.detect") as mock_detect:
        sct_deepseg.main(['spinalcord', '-i', sct_test_path('t2', 't2.nii.gz'), '-o', fname_out,
                          '-qc', tmp_path_qc,
                          '-box-xmin', '10', '-box-xmax', '59', '-box-ymin', '0', '-box-ymax', '54',
                          '-box-zmin', '3', '-box-zmax', '48'])
    mock_detect.assert_not_called()


def test_deepseg_box_override_rejected_for_non_crop_model(tmp_path, tmp_path_qc, capsys):
    """
    `-box-*` isn't even exposed as an argument for models that don't use the sc-crop pipeline.
    """
    fname_out = str(tmp_path / "t2_seg_deepseg.nii.gz")
    with pytest.raises(SystemExit):
        sct_deepseg.main(['rootlets', '-i', sct_test_path('t2', 't2.nii.gz'), '-o', fname_out,
                          '-qc', tmp_path_qc, '-box-xmin', '0'])
    assert "unrecognized arguments: -box-xmin 0" in capsys.readouterr().err


def test_deepseg_crop_box_override_invalid_bbox_raises(tmp_path, tmp_path_qc):
    """
    Inverted `-box-*` coordinates (a face's min > its max) should raise a clear error rather
    than silently producing an empty or nonsensical crop.
    """
    fname_out = str(tmp_path / "t2_seg_deepseg.nii.gz")
    with pytest.raises(ValueError) as e:
        sct_deepseg.main(['spinalcord', '-i', sct_test_path('t2', 't2.nii.gz'), '-o', fname_out,
                          '-qc', tmp_path_qc, '-box-zmin', '40', '-box-zmax', '10'])
    assert "invalid bounding box" in str(e.value)


def test_deepseg_crop_detection_failure_raises_without_box_override(t2_zero, tmp_path, tmp_path_qc):
    """
    If sc-crop detection fails outright (no cord found) and no `-box-*` is given to fall back
    on, the error should propagate instead of being silently swallowed.
    """
    fname_out = str(tmp_path / "t2_seg_deepseg.nii.gz")
    with pytest.raises(RuntimeError) as e:
        sct_deepseg.main(['spinalcord', '-i', t2_zero, '-o', fname_out, '-qc', tmp_path_qc])
    assert "No spinal cord detected" in str(e.value)


def test_deepseg_crop_detection_failure_falls_back_to_full_image(t2_zero, tmp_path, tmp_path_qc):
    """
    If sc-crop detection fails outright but `-box-*` faces are given, fall back to the full
    image extent for the faces not overridden, instead of crashing.
    """
    fname_out = str(tmp_path / "t2_seg_deepseg.nii.gz")
    sct_deepseg.main(['spinalcord', '-i', t2_zero, '-o', fname_out, '-qc', tmp_path_qc,
                      '-box-zmin', '0'])
    assert os.path.isfile(fname_out)


@pytest.mark.usefixtures(cleanup_model_dirs.__name__)
def test_deepseg_crop_creates_output_subdirectory(tmp_path, tmp_path_qc):
    """
    `-o` pointing to a not-yet-existing subdirectory shouldn't crash when saving the cropbox
    file, which is written earlier in the pipeline than sct_deepseg.py's own output-directory
    creation.
    """
    fname_out = str(tmp_path / "subdir" / "t2_seg_deepseg.nii.gz")
    sct_deepseg.main(['spinalcord', '-i', sct_test_path('t2', 't2.nii.gz'), '-o', fname_out,
                      '-qc', tmp_path_qc])
    assert os.path.isfile(fname_out)
    assert os.path.isfile(add_suffix(fname_out, "_cropbox"))


@pytest.mark.usefixtures(cleanup_model_dirs.__name__)
def test_deepseg_crop_box_qc_report_entry(tmp_path, tmp_path_qc):
    """
    Crop-active models should get a separate QC report entry (plane='Ortho', i.e. an orthographic view) for the crop box
    itself, distinct from the segmentation entry.
    """
    # `tmp_path_qc` is session-scoped (shared across tests), so record which entries already
    # exist before this test runs its own segmentation.
    path_json = Path(tmp_path_qc) / '_json'
    existing = set(path_json.glob('qc_*.json')) if path_json.is_dir() else set()

    fname_out = str(tmp_path / "t2_seg_deepseg.nii.gz")
    sct_deepseg.main(['spinalcord', '-i', sct_test_path('t2', 't2.nii.gz'), '-o', fname_out,
                      '-qc', tmp_path_qc])
    fname_cropbox = add_suffix(fname_out, "_cropbox")
    assert os.path.isfile(fname_cropbox)

    new_entries = [json.loads(p.read_text()) for p in path_json.glob('qc_*.json') if p not in existing]
    cropbox_entries = [e for e in new_entries if e['plane'] == 'Ortho']  # Orthographic view
    assert len(cropbox_entries) == 1
    entry = cropbox_entries[0]
    assert entry['command'] == 'sc_crop'
    # Neither field should mention 'sct_deepseg', so searching for either tool's name in the QC
    # report reliably shows only that tool's rows -- not both, due to some shared substring.
    assert 'sct_deepseg' not in entry['command']
    assert 'sct_deepseg' not in entry['cmdline']
    cmdline = ["sc_crop", "-i", str(sct_test_path('t2', 't2.nii.gz')), "--bbox", str(fname_cropbox)]
    assert entry['cmdline'] == " ".join(shlex.quote(x) for x in cmdline)
    assert (Path(tmp_path_qc) / entry['backgroundImage']).is_file()
    assert (Path(tmp_path_qc) / entry['overlayImage']).is_file()


@pytest.mark.parametrize("task", ["spinalcord", "lesion_ms"])
def test_deepseg_soft_seg_produces_expected_output(tmp_path, tmp_path_qc, task):
    """
    Test that the softseg output is produced when requested.
    """
    fname_out = str(tmp_path / "t2_seg_deepseg_sosft_seg.nii.gz")
    sct_deepseg.main([task, '-i', sct_test_path('t2', 't2.nii.gz'), '-o', fname_out,
                      '-qc', tmp_path_qc, '-soft-seg'])
    im_out = Image(fname_out)
    # Check that the softseg output is a float image (not integer)
    assert np.issubdtype(im_out.data.dtype, np.floating)
