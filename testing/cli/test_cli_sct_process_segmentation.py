# pytest unit tests for sct_process_segmentation

import pytest
import logging
import numpy as np
import tempfile
import nibabel
import csv

from spinalcordtoolbox.scripts import sct_process_segmentation, sct_image
from spinalcordtoolbox.utils.sys import sct_test_path
from spinalcordtoolbox.image import add_suffix

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def dummy_3d_mask_nib():
    data = np.zeros([32, 32, 32], dtype=np.uint8)
    data[9:24, 9:24, 9:24] = 1
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.fixture(scope="session")
def dummy_3d_pmj_label():
    data = np.zeros([32, 32, 32], dtype=np.uint8)
    data[15, 15, 28] = 1
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


def test_sct_process_segmentation_check_values(tmp_path):
    """ Run sct_process_segmentation and check the results"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_process_segmentation.main(argv=['-i', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                        '-perslice', '1', '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        rows = list(reader)
        row = rows[10]
        assert row['Slice (I->S)'] == '10'
        assert float(row['MEAN(area)']) == pytest.approx(78.58643257836141)
        assert float(row['MEAN(diameter_AP)']) == pytest.approx(7.842507960401277)
        assert float(row['MEAN(diameter_RL)']) == pytest.approx(12.537767663676533)
        assert float(row['MEAN(orientation)']) == pytest.approx(-5.435920797812814)
        assert float(row['MEAN(eccentricity)']) == pytest.approx(0.7678714078545043)
        assert float(row['MEAN(solidity)']) == pytest.approx(0.9691052891744933)


def test_sct_process_segmentation_check_pmj(dummy_3d_mask_nib, dummy_3d_pmj_label, tmp_path, tmp_path_qc):
    """ Run sct_process_segmentation with -pmj, -pmj-distance and -pmj-extent and check the results"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_process_segmentation.main(argv=['-i', dummy_3d_mask_nib, '-pmj', dummy_3d_pmj_label,
                                        '-pmj-distance', '8', '-pmj-extent', '4', '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert row['Slice (I->S)'] == '18:21'
        assert row['DistancePMJ'] == '8.0'
        assert row['VertLevel'] == ''
        assert row['SUM(length)'] == '4.0'


def test_sct_process_segmentation_check_pmj_reoriented(dummy_3d_mask_nib, dummy_3d_pmj_label, tmp_path, tmp_path_qc):
    """
    Make sure that the results are the same regardless of the input orientation.

    This should never really fail, since `sct_process_segmentation` reorients to RPI prior to processing,
    but it will hopefully act as a canary for #4622, since PMJ CSA calls `get_centerline` in two different places.
    """
    # Note down filenames for easier iteration
    fname_dict = {
        "rpi": {
            "i": dummy_3d_mask_nib,
            "pmj": dummy_3d_pmj_label,
            "out": str(tmp_path / 'csa_rpi.csv'),
        },
        "sal": {
            "i": add_suffix(dummy_3d_mask_nib, suffix='_sal'),
            "pmj": add_suffix(dummy_3d_pmj_label, suffix='_sal'),
            "out": str(tmp_path / 'csa_sal.csv'),
        }
    }
    # Create duplicate images, reoriented to SAL
    sct_image.main(argv=["-setorient", "SAL", "-i", fname_dict['rpi']["i"], "-o", fname_dict['sal']["i"]])
    sct_image.main(argv=["-setorient", "SAL", "-i", fname_dict['rpi']["pmj"], "-o", fname_dict['sal']["pmj"]])
    # Run sct_process_segmentation on both sets of images, then compare results
    results = []
    for fnames in fname_dict.values():
        sct_process_segmentation.main(argv=['-i', fnames["i"], '-pmj', fnames["i"], '-o', fnames["out"],
                                            '-pmj-distance', '8', '-pmj-extent', '4'])
        with open(fnames["out"], "r") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            results.append(next(reader))
    # Check that the results are the same
    for key in results[0].keys():
        if key not in ['Timestamp', 'SCT Version', "Filename"]:
            assert results[0][key] == results[1][key]


def test_sct_process_segmentation_check_pmj_perslice(dummy_3d_mask_nib, dummy_3d_pmj_label, tmp_path, tmp_path_qc):
    """ Run sct_process_segmentation with -pmj, -perslice and check the results"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_process_segmentation.main(argv=['-i', dummy_3d_mask_nib, '-pmj', dummy_3d_pmj_label,
                                        '-perslice', '1', '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        rows = list(reader)
        row = rows[10]
        assert row['Slice (I->S)'] == '10'
        assert row['DistancePMJ'] == '18.11'
        assert row['VertLevel'] == ''
        assert row['SUM(length)'] == '1.0'


def test_sct_process_segmentation_missing_pmj_args(dummy_3d_mask_nib, dummy_3d_pmj_label):
    """ Run sct_process_segmentation with PMJ method when missing -pmj or -pmj-distance """
    for args in [['-i', dummy_3d_mask_nib, '-pmj', dummy_3d_pmj_label], ['-i', dummy_3d_mask_nib, '-pmj-distance', '4']]:
        with pytest.raises(SystemExit) as e:
            sct_process_segmentation.main(argv=args)
            assert e.value.code == 2


def test_sct_process_segmentation_check_normalize(dummy_3d_mask_nib, tmp_path):
    """ Run sct_process_segmentation with -normalize and check the results"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_process_segmentation.main(argv=['-i', dummy_3d_mask_nib, '-normalize', 'brain-volume',
                                        '960606.0', 'sex', '0', 'thalamus-volume', '13942.0', '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert float(row['MEAN(area)']) == pytest.approx(228.20973426502943)


def test_sct_process_segmentation_check_normalize_missing_value(dummy_3d_mask_nib, tmp_path):
    """ Run sct_process_segmentation with -normalize when missing a value"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_process_segmentation.main(argv=['-i', dummy_3d_mask_nib, '-normalize', 'brain-volume',
                                            '960606.0', 'sex', 'thalamus-volume', '13942.0', '-o', filename])
        assert e.value.code == 2


def test_sct_process_segmentation_check_normalize_missing_predictor(dummy_3d_mask_nib, tmp_path):
    """ Run sct_process_segmentation with -normalize when missing a predictor"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_process_segmentation.main(argv=['-i', dummy_3d_mask_nib, '-normalize',
                                            'sex', '0', 'thalamus-volume', '13942.0', '-o', filename])
        assert e.value.code == 2


def test_sct_process_segmentation_check_normalize_PAM50(tmp_path):
    """ Run sct_process_segmentation with -normalize PAM50"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_process_segmentation.main(argv=['-i', sct_test_path('t2', 't2_seg-manual.nii.gz'), '-normalize-PAM50', '1',
                                        '-perslice', '1', '-discfile', sct_test_path('t2', 'labels.nii.gz'), '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        rows = list(reader)
        row = rows[26]
        assert row['Slice (I->S)'] == '827'
        assert float(row['MEAN(area)']) == pytest.approx(71.16703270056989)
        assert row['VertLevel'] == '5'


def test_sct_process_segmentation_check_normalize_PAM50_missing_perslice(tmp_path):
    """ Run sct_process_segmentation with -normalize PAM50 when missing perslice argument"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_process_segmentation.main(argv=['-i', sct_test_path('t2', 't2_seg-manual.nii.gz'), '-normalize-PAM50', '1',
                                            '-discfile', sct_test_path('t2', 'labels.nii.gz'), '-o', filename])
        assert e.value.code == 2


def test_sct_process_segmentation_check_normalize_PAM50_missing_vertfile(tmp_path):
    """ Run sct_process_segmentation with -normalize PAM50 when missing -vertfile"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_process_segmentation.main(argv=['-i', sct_test_path('t2', 't2_seg-manual.nii.gz'), '-normalize-PAM50', '1',
                                            '-perslice', '1', '-o', filename])
        assert e.value.code == 2


def test_sct_process_segmentation_check_both_discfile_vertfile(tmp_path):
    """ Run sct_process_segmentation with -vertfile and -discfile"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_process_segmentation.main(argv=['-i', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                            '-vertfile', sct_test_path('t2', 't2_seg-manual_labeled.nii.gz'),
                                            '-discfile', sct_test_path('t2', 'labels.nii.gz'), '-o', filename])
        assert e.value.code == 2


def test_sct_process_segmentation_check_discfile(tmp_path):
    """ Run sct_process_segmentation with-discfile"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_process_segmentation.main(argv=['-i', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                        '-vert', '1:10', '-perslice', '1',
                                        '-discfile', sct_test_path('t2', 'labels.nii.gz'), '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        rows = list(reader)
        row = rows[10]
        assert row['Slice (I->S)'] == '10'
        assert row['DistancePMJ'] == ''
        assert row['VertLevel'] == '3'
        assert float(row['MEAN(area)']) == pytest.approx(78.58643257836141)


def test_sct_process_segmentation_anat_properties(tmp_path, tmp_path_qc):
    """ Run sct_process_segmentation with -anat and check the symmetry results"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_process_segmentation.main(argv=['-i', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                        '-vert', '1:10', '-perslice', '1',
                                        '-discfile', sct_test_path('t2', 'labels.nii.gz'), '-anat', sct_test_path('t2', 't2.nii.gz'), '-o', filename,
                                        '-qc', str(tmp_path_qc)])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        rows = list(reader)
        row = rows[10]
        assert row['Slice (I->S)'] == '10'
        assert float(row['MEAN(symmetry_dice_RL)']) == pytest.approx(0.8779774794566941)
        assert float(row['MEAN(symmetry_dice_AP)']) == pytest.approx(0.8778275640403707)
        assert float(row['MEAN(symmetry_hausdorff_RL)']) == pytest.approx(0.894427190999916)
        assert float(row['MEAN(symmetry_hausdorff_AP)']) == pytest.approx(0.894427190999916)
        assert float(row['MEAN(symmetry_difference_RL)']) == pytest.approx(9.686472446271557)
        assert float(row['MEAN(symmetry_difference_AP)']) == pytest.approx(9.756804249735488)


@pytest.mark.sct_testing
def test_sct_process_segmentation_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_process_segmentation.main(argv=['-i', sct_test_path('t2', 't2_seg-manual.nii.gz')])
