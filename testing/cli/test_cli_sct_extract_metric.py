# pytest unit tests for sct_extract_metric

import csv
import numpy as np

import pytest
import logging

from spinalcordtoolbox.scripts import sct_extract_metric
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_extract_metric_against_groundtruth():
    """Verify that computed values are equivalent to known ground truth values."""
    fname_out = 'quantif_mtr.csv'
    sct_extract_metric.main(argv=['-i', sct_test_path('mt', 'mtr.nii.gz'),
                                  '-f', sct_test_path('mt', 'label/atlas'),
                                  '-method', 'wa', '-l', '51', '-z', '1:2', '-o', fname_out])
    results = np.genfromtxt(fname_out, skip_header=1, delimiter=',')
    results = results[~np.isnan(results)]  # Remove non-numeric fields such as filename, SCT version, etc.
    assert results == pytest.approx([176.1532, 32.6404, 11.4573], abs=0.001)  # Size[vox], WA, and STD respectively


def test_sct_extract_metric_vertlevel_outside_bounds():
    """Make sure that specifying -vert values outside the bounds of the data does not produce an empty row."""
    fname_out = 'quantif_mtr.csv'
    sct_extract_metric.main(argv=['-i', sct_test_path('mt', 'mtr.nii.gz'),
                                  '-f', sct_test_path('mt', 'label/atlas'),
                                  '-vertfile', sct_test_path('mt', 'label', 'template', 'PAM50_levels.nii.gz'),
                                  '-vert', '1:12', '-perlevel', '1',
                                  '-method', 'wa', '-l', '51', '-o', fname_out])
    results = np.genfromtxt(fname_out, delimiter=',', dtype=None, encoding="utf-8")  # dtype=None preserves str data
    header_row, data_rows = results[0],  np.char.strip(results[1:], '"')
    idx_vertlevel = np.where(header_row == "VertLevel")[0][0]
    idx_size = np.where(header_row == "WA()")[0][0]
    assert list(data_rows[:, idx_vertlevel].astype(int)) == [5, 4]  # VertLevel column should only have 2 values
    assert list(data_rows[:, idx_size].astype(float)) == pytest.approx([35.268203117287605, 30.073466350619103], abs=0.001)


def test_sct_extract_metric_vertfile_doesnt_exists():
    fname_out = 'quantif_mtr.csv'
    with pytest.raises(SystemExit) as e:
        sct_extract_metric.main(argv=['-i', sct_test_path('mt', 'mtr.nii.gz'),
                                      '-f', sct_test_path('mt', 'label/atlas'),
                                      '-vertfile', sct_test_path('mt', 'label', 'template', 'levels.nii.gz'),
                                      '-vert', '1:12', '-perlevel', '1',
                                      '-method', 'wa', '-l', '51', '-o', fname_out])
        assert e.value.code == 2


def test_sct_extract_metric_normalize_pam50(tmp_path):
    """Verify that -normalize-PAM50 outputs per-slice metrics in PAM50 space with VertLevel populated."""
    fname_out = str(tmp_path / 'quantif_mtr_pam50.csv')
    sct_extract_metric.main(argv=[
        '-i', sct_test_path('mt', 'mtr.nii.gz'),
        '-f', sct_test_path('mt', 'label/atlas'),
        '-method', 'wa', '-l', '51',
        '-vertfile', sct_test_path('mt', 'label', 'template', 'PAM50_levels.nii.gz'),
        '-perslice', '1', '-normalize-PAM50', '1',
        '-o', fname_out
    ])
    with open(fname_out, 'r') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert all(r['Label'] == 'white matter' for r in rows)
    assert all(int(r['VertLevel']) in [4, 5] for r in rows)
    assert all((800 < int(r['Slice (I->S)']) < 870) for r in rows)
    assert all((30 < float(r['WA()']) < 36) for r in rows)


def test_sct_extract_metric_normalize_pam50_missing_perslice(tmp_path):
    """Verify that -normalize-PAM50 1 without -perslice 1 raises an error."""
    with pytest.raises(SystemExit) as e:
        sct_extract_metric.main(argv=[
            '-i', sct_test_path('mt', 'mtr.nii.gz'),
            '-f', sct_test_path('mt', 'label/atlas'),
            '-method', 'wa', '-l', '51',
            '-vertfile', sct_test_path('mt', 'label', 'template', 'PAM50_levels.nii.gz'),
            '-normalize-PAM50', '1',
            '-o', str(tmp_path / 'out.csv')
        ])
    assert e.value.code == 2


def test_sct_extract_metric_normalize_pam50_missing_vertfile(tmp_path):
    """Verify that -normalize-PAM50 1 with a missing vertfile raises an error."""
    with pytest.raises(SystemExit) as e:
        sct_extract_metric.main(argv=[
            '-i', sct_test_path('mt', 'mtr.nii.gz'),
            '-f', sct_test_path('mt', 'label/atlas'),
            '-method', 'wa', '-l', '51',
            '-vertfile', sct_test_path('mt', 'label', 'template', 'does_not_exist.nii.gz'),
            '-perslice', '1', '-normalize-PAM50', '1',
            '-o', str(tmp_path / 'out.csv')
        ])
    assert e.value.code == 2
