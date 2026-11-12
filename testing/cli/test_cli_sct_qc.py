# pytest unit tests for sct_qc

import pytest
import logging
import json
import os

from spinalcordtoolbox.scripts import sct_qc, sct_deepseg
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_qc_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_qc.main(argv=['-i', sct_test_path('t2', 't2.nii.gz'),
                      '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                      '-p', 'sct_deepseg_sc',
                      '-qc-dataset', 'sct_testing_data', '-qc-subject', 'dummy',
                      '-qc-contrast', 't2'])


# custom label to str mapping for totalspineseg
custom_label_json = {
    12: 'C2',
    13: 'C3',
    14: 'C4',
    15: 'C5',
    16: 'C6',
    17: 'C7',
    21: 'T1',
    22: 'T2',
    23: 'T3',
    24: 'T4',
    25: 'T5',
    26: 'T6',
    27: 'T7',
    28: 'T8',
    29: 'T9',
    30: 'T10',
    31: 'T11',
    32: 'T12',
    41: 'L1',
    42: 'L2',
    43: 'L3',
    44: 'L4',
    45: 'L5',
    50: 'sacrum',
    63: 'C2-C3',
    64: 'C3-C4',
    65: 'C4-C5',
    66: 'C5-C6',
    67: 'C6-C7',
    71: 'C7-T1',
    72: 'T1-T2',
    73: 'T2-T3',
    74: 'T3-T4',
    75: 'T4-T5',
    76: 'T5-T6',
    77: 'T6-T7',
    78: 'T7-T8',
    79: 'T8-T9',
    80: 'T9-T10',
    81: 'T10-T11',
    82: 'T11-T12',
    91: 'T12-L1',
    92: 'L1-L2',
    93: 'L2-L3',
    94: 'L3-L4',
    95: 'L4-L5',
    100: 'L5-S'
}


@pytest.fixture(scope="module")
def totalspineseg_labels(tmp_path_factory):
    # Generate the labeling using totalspineseg
    tmp_path = tmp_path_factory.mktemp("totalspineseg_labels")
    sct_deepseg.main(argv=["spine",
                           "-i", sct_test_path('t2', 't2.nii.gz'),
                           "-o", os.path.join(tmp_path, "spine.nii.gz")])
    return os.path.join(tmp_path, "spine_totalspineseg_all.nii.gz")


@pytest.mark.parametrize('custom_labels,err_msg', [
    (custom_label_json, ""),
    (None, ""),
    ([12, "C2"], "single dictionary"),          # JSON should be a dict
    ({"C2": "C2"}, "invalid literal for int"),  # Keys should be integers
    ({12: 12}, "Not a text label")              # Values should be strings
])
def test_sct_qc_totalspineseg_custom_labels(custom_labels, err_msg, totalspineseg_labels, tmp_path_qc):
    """Run the CLI script with custom mapping for total spine segmentation."""
    # construct the arguments for sct_qc
    args = ['-i', sct_test_path('t2', 't2.nii.gz'),
            '-s', totalspineseg_labels,
            '-p', 'sct_label_vertebrae',
            '-qc', tmp_path_qc,
            '-qc-dataset', 'sct_testing_data',
            '-qc-subject', 'dummy',
            '-qc-contrast', 't2']

    # add the custom labels as an argument (if provided)
    if custom_labels:
        custom_mapping_file = os.path.join(tmp_path_qc, 'custom_mapping.json')
        with open(custom_mapping_file, 'w') as f:
            json.dump(custom_labels, f)
        args.extend(['-custom-labels', custom_mapping_file])

    # If an err message was supplied, check for a raised error
    if err_msg:
        with pytest.raises(ValueError) as e:
            sct_qc.main(argv=args)
        assert err_msg in str(e.value)

    # Otherwise, run the script normally
    else:
        sct_qc.main(argv=args)
