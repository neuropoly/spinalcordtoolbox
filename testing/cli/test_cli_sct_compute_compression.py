# pytest unit tests for sct_compute_compression

import pytest
import logging
import numpy as np
import tempfile
import nibabel
import csv

from spinalcordtoolbox.scripts import sct_compute_compression

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def dummy_3d_compression_label():
    data = np.zeros([32, 32, 81], dtype=np.uint8)
    data[15, 15, 48] = 1
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(prefix='compression', suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.fixture(scope="session")
def dummy_3d_multiple_compression_label():
    data = np.zeros([32, 32, 81], dtype=np.uint8)
    data[15, 15, 48] = 1    # Compression #1
    data[15, 15, 60] = 1    # Compression #2
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(prefix='multiple_compression', suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.fixture(scope="session")
def dummy_3d_lesion_label():
    data = np.zeros([32, 32, 81], dtype=np.uint8)
    # Create a binary mask with a 3D lesion
    data[13:17, 13:17, 43:46] = 1
    data[14:16, 14:16, 46:50] = 1
    data[13:17, 13:17, 50:55] = 1
    nii = nibabel.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(prefix='lesion', suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.fixture(scope="session")
def dummy_3d_vert_label():
    data = np.zeros([32, 32, 81], dtype=np.uint8)
    data[15, 15, 0:10] = 2.0
    data[15, 15, 10:25] = 3.0
    data[15, 15, 25:50] = 4.0
    data[15, 15, 25:50] = 5.0
    data[15, 15, 50:70] = 6.0
    data[15, 15, 70:81] = 7.0
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(prefix='vert', suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.fixture(scope="session")
def dummy_3d_mask_nib():
    data = np.zeros([32, 32, 81], dtype=np.uint8)
    data[9:24, 10:23, :] = 1
    data[9:24, 10:23, 48] = 0
    data[9:24, 11:21, 48] = 1
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(prefix='seg', suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.mark.sct_testing
def test_sct_compute_compression_value_against_groundtruth():
    """Run the CLI script and verify that computed mscc value is equivalent to known ground truth value."""
    di, da, db = 6.85, 7.65, 7.02
    mscc = sct_compute_compression.metric_ratio(mi=di, ma=da, mb=db)
    assert mscc == pytest.approx(6.612133606, abs=1e-4)


def test_sct_compute_compression_check_missing_input_segmentation(tmp_path, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression when missing -i"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-l', dummy_3d_compression_label, '-vertfile', dummy_3d_vert_label,
                                           '-o', filename, '-normalize-hc',  '1'])
        assert e.value.code == 2


def test_sct_compute_compression_check_missing_input_l(tmp_path, dummy_3d_mask_nib, dummy_3d_vert_label):
    """ Run sct_compute_mscc when missing -l"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-vertfile', dummy_3d_vert_label, '-o', filename,
                                           '-normalize-hc',  '1'])
        assert e.value.code == 2


def test_sct_compute_compression_check_missing_input_normalize_hc(tmp_path, dummy_3d_mask_nib, dummy_3d_vert_label):
    """ Run sct_compute_mscc when missing -normalize-hc"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-vertfile', dummy_3d_vert_label, '-o', filename])
        assert e.value.code == 2


def test_sct_compute_compression_check_wrong_sex(tmp_path, dummy_3d_mask_nib, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression with wrong value for sex"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_compression_label, '-vertfile', dummy_3d_vert_label,
                                           '-sex', 'J', '-o', filename, '-normalize-hc',  '1'])
        assert e.value.code == 2


def test_sct_compute_compression_check_wrong_age(tmp_path, dummy_3d_mask_nib, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression with wrong age"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_compression_label, '-vertfile', dummy_3d_vert_label,
                                           '-age', '20', '-o', filename, '-normalize-hc',  '1'])
        assert e.value.code == 2


def test_sct_compute_compression_check_wrong_metric(tmp_path, dummy_3d_mask_nib, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression when specifying a wrong metric"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_compression_label, '-vertfile', dummy_3d_vert_label,
                                           '-metric', 'MEAN', '20', '-o', filename, '-normalize-hc',  '0'])
        assert e.value.code == 2


def test_sct_compute_compression_no_normalization(tmp_path, dummy_3d_mask_nib, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression without normalization to a database of healthy controls"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_compression_label, '-vertfile', dummy_3d_vert_label,
                                       '-normalize-hc',  '0', '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert float(row['compression_level']) == 5.0
        # compressed diameter ~10mm
        # surrounding diameter ~13mm
        # expected ratio (1 - 10/13) ~= 23.07%
        assert float(row['diameter_AP_ratio']) == pytest.approx(23.07, abs=0.01)
        assert row['diameter_AP_ratio_PAM50'] == 'n/a'
        assert row['diameter_AP_ratio_PAM50_normalized'] == 'n/a'


def test_sct_compute_compression_no_normalization_multiple_compressions(tmp_path, dummy_3d_mask_nib, dummy_3d_multiple_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression on label file with multiple compression labels without normalization to a database of healthy controls"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_multiple_compression_label, '-vertfile', dummy_3d_vert_label,
                                       '-normalize-hc',  '0', '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        # First compression
        row = next(reader)
        assert float(row['Slice (I->S)']) == 48
        assert float(row['compression_level']) == 5.0
        # compressed diameter ~10mm
        # surrounding diameter ~13mm
        # expected ratio (1 - 10/13) ~= 23.07%
        assert float(row['diameter_AP_ratio']) == pytest.approx(23.07, abs=0.01)
        assert row['diameter_AP_ratio_PAM50'] == 'n/a'
        assert row['diameter_AP_ratio_PAM50_normalized'] == 'n/a'
        # Second compression
        row = next(reader)
        assert float(row['Slice (I->S)']) == 60
        assert float(row['compression_level']) == 6.0
        # this is a dummy compression label, with no actual compression in the mask
        # so expected ratio is (1 - 13/13) = 0%
        assert float(row['diameter_AP_ratio']) == pytest.approx(0, abs=0.01)
        assert row['diameter_AP_ratio_PAM50'] == 'n/a'
        assert row['diameter_AP_ratio_PAM50_normalized'] == 'n/a'


def test_sct_compute_compression(tmp_path, dummy_3d_mask_nib, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression and check mscc and normalized mscc"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_compression_label,
                                       '-vertfile', dummy_3d_vert_label, '-normalize-hc',  '1', '-o', filename])
    # Run sct_compute_compression a second time with a different metric than the default
    sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_compression_label,
                                       '-vertfile', dummy_3d_vert_label, '-normalize-hc',  '1', '-o', filename,
                                       '-metric', 'area'])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert float(row['compression_level']) == 5.0
        # compressed diameter ~10mm
        # surrounding diameter ~13mm
        # expected ratio (1 - 10/13) ~= 23.07%
        # PAM50 and PAM50_normalized are just taken from a test run
        assert float(row['diameter_AP_ratio']) == pytest.approx(23.07, abs=0.01)
        assert float(row['diameter_AP_ratio_PAM50']) == pytest.approx(14.33, abs=0.01)
        assert float(row['diameter_AP_ratio_PAM50_normalized']) == pytest.approx(18.43, abs=0.01)
        # compressed area 10*15 = 150
        # surrounding area 13*15 = 195
        # expected ratio (1 - 150/195) ~= 23.07%
        # PAM50 and PAM50_normalized are just taken from a test run
        assert float(row['area_ratio']) == pytest.approx(23.07, abs=0.01)
        assert float(row['area_ratio_PAM50']) == pytest.approx(14.33, abs=0.01)
        assert float(row['area_ratio_PAM50_normalized']) == pytest.approx(21.88, abs=0.01)
        # Ensure that there isn't a duplicate appended row from running sct_compute_compression twice
        with pytest.raises(StopIteration):
            next(reader)


def test_sct_compute_compression_sex_F(tmp_path, dummy_3d_mask_nib, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression and check mscc and normalized mscc"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_compression_label,
                                       '-vertfile', dummy_3d_vert_label, '-normalize-hc',  '1', '-sex', 'F',
                                       '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert float(row['compression_level']) == 5.0
        # compressed diameter ~10mm
        # surrounding diameter ~13mm
        # expected ratio (1 - 10/13) ~= 23.07%
        # PAM50 and PAM50_normalized are just taken from a test run
        assert float(row['diameter_AP_ratio']) == pytest.approx(23.07, abs=0.01)
        assert float(row['diameter_AP_ratio_PAM50']) == pytest.approx(14.33, abs=0.01)
        assert float(row['diameter_AP_ratio_PAM50_normalized']) == pytest.approx(17.99, abs=0.01)


def test_sct_compute_compression_lesion_no_vertfile(tmp_path, dummy_3d_mask_nib, dummy_3d_lesion_label):
    """ Run sct_compute_compression for a lesion with no vertfile provided"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_lesion_label, '-mode', 'lesion',
                                       '-metric', 'diameter_AP', '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert row['compression_level'] == 'n/a'     # 'n/a' because we do not provide a vertfile
        assert float(row['Slice (I->S)']) == 48
        # compressed diameter ~10mm
        # surrounding diameter ~13mm
        # expected ratio (1 - 10/13) ~= 23.07%
        assert float(row['diameter_AP_ratio']) == pytest.approx(23.07, abs=0.01)
        assert row['diameter_AP_ratio_PAM50'] == 'n/a'   # 'n/a' because we do not do normalization
        assert row['diameter_AP_ratio_PAM50_normalized'] == 'n/a'    # 'n/a' because we do not do normalization
        # Ensure that there isn't a duplicate appended row from running sct_compute_compression twice
        with pytest.raises(StopIteration):
            next(reader)


def test_sct_compute_compression_lesion_vertfile(tmp_path, dummy_3d_mask_nib, dummy_3d_lesion_label, dummy_3d_vert_label):
    """ Run sct_compute_compression for a lesion with provided vertfile"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_lesion_label, '-mode', 'lesion',
                                       '-vertfile', dummy_3d_vert_label, '-metric', 'diameter_AP', '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert float(row['compression_level']) == 5
        assert float(row['Slice (I->S)']) == 48
        # compressed diameter ~10mm
        # surrounding diameter ~13mm
        # expected ratio (1 - 10/13) ~= 23.07%
        assert row['diameter_AP_ratio_PAM50'] == 'n/a'   # 'n/a' because we do not do normalization
        assert row['diameter_AP_ratio_PAM50_normalized'] == 'n/a'    # 'n/a' because we do not do normalization
        # Ensure that there isn't a duplicate appended row from running sct_compute_compression twice
        with pytest.raises(StopIteration):
            next(reader)
