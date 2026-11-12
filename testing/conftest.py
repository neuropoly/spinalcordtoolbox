"""
Configure tests so that pytest downloads testing data every time pytest is run
from the sct directory

Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import os
import logging
from typing import Mapping
from hashlib import md5
import tempfile
from glob import glob
import json
import csv

import pytest
from nibabel import Nifti1Header
from numpy import zeros

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import sct_test_path, __sct_dir__, __data_dir__, set_loglevel
from spinalcordtoolbox.download import install_named_dataset
from contrib.fslhd import generate_nifti_fields, generate_numpy_fields

logger = logging.getLogger(__name__)


def pytest_sessionstart():
    """Perform actions that must be done prior to test collection."""
    # Use a non-interactive backend so that no GUI plots will interrupt the test suite.
    # (NB: We do this here to ensure it is set before `matplotlib` is first imported.)
    if 'MPLBACKEND' not in os.environ:
        os.environ["MPLBACKEND"] = 'Agg'

    # Download sct_testing_data prior to test collection
    if not os.path.exists(sct_test_path()):
        logger.info("Downloading sct test data")
        install_named_dataset('sct_testing_data', dest_folder=sct_test_path())


def pytest_sessionfinish():
    """Perform actions that must be done after the test session."""
    # don't generate summary files locally, since they are time-consuming and delay local testing
    if "CI" not in os.environ:
        return

    # get the newest temporary path created by pytest
    pytest_tempdirs = glob(os.path.join(tempfile.gettempdir(), "pytest-of-*", "pytest-current"))
    tmp_path = max(pytest_tempdirs, key=lambda p: os.path.getctime(p), default="")

    # generate directory summaries for both sct_testing_data and the temporary directory
    for (folder, fname_out) in [(tmp_path, "pytest-tmp.json"),
                                (sct_test_path(), "sct_testing_data.json"),
                                (os.path.join(__data_dir__, "sct_course_data"), "sct_course_data.json")]:
        fname_out = os.path.join(__sct_dir__, "testing", fname_out)
        if os.path.isdir(folder):
            summary = summarize_files_in_folder(folder, exclude=['straightening.cache'])
            summary = sorted(summary, key=lambda d: d['path'])   # sort list-of-dicts by paths
            keys = [d.pop('path') for d in summary]              # remove paths from dicts
            summary = {key: d for key, d in zip(keys, summary)}  # convert to dict-of-dicts
            with open(fname_out, 'w', newline='\n') as jsonfile:
                json.dump(summary, jsonfile, indent=2)


def summarize_files_in_folder(folder, exclude=None):
    # Construct a list of dictionaries summarizing all the files in a folder
    summary = []
    for root, dirs, files in os.walk(folder, followlinks=True):
        for fname in files:
            if exclude and fname in exclude:
                continue
            fpath = os.path.join(root, fname)
            if fname.endswith(".csv"):
                fpath = filter_csv_columns(fpath, columns=["Timestamp", "SCT Version"])
            elif fname.endswith(".json"):
                fpath = filter_json_sidecars(fpath, fields=["Version", "CodeURL"])
            root_short = root.replace(folder, os.path.basename(folder))
            file_dict = {
                # Use consistent characters to make cross-platform diffing work
                "path": "/".join(os.path.split(os.path.join(root_short, fname))),
                "size": os.path.getsize(fpath),
                "md5": checksum(fpath),
            }
            if any(fname.endswith(ext) for ext in [".nii", ".nii.gz"]):
                img = Image(fpath)
                img_fields = generate_nifti_fields(img.header)
                arr_fields = generate_numpy_fields(img.data)
            else:
                img_fields = {k: '' for k in generate_nifti_fields(Nifti1Header()).keys()}
                arr_fields = {k: '' for k in generate_numpy_fields(zeros([1, 1, 1])).keys()}
            summary.append(file_dict | img_fields | arr_fields)
    return summary


def filter_csv_columns(fpath, columns):
    """
    Filter out columns from a CSV file that are not in the list `columns`.

    This helps when checking the filesize and md5 of CSV files, which can contain differing branches, timestamps, etc.
    """
    def read_csv(file_path):
        """Read CSV into a list of dictionaries."""
        # DictReader will automatically use the first row as a header. If the CSV doesn't have a header,
        # `csv` will interpret the first row of data as column names. Thus, if the first row has duplicate values,
        # the columns will be interpreted as duplicates, and one of the columns might get thrown away!
        # NOTE: The csv library has `csv.Sniffer.has_header()` to detect headers, but given that it's based
        #       on heuristics, it's probably safer to check the parsed headers after the fact.
        with open(file_path, mode='r', newline='', encoding='utf-8') as fp:
            reader = csv.DictReader(fp)
            return [row for row in reader]

    def write_csv(data, file_path):
        """Write a list of dictionaries to a CSV."""
        with open(file_path, mode='w', newline='', encoding='utf-8') as fp:
            writer = csv.DictWriter(fp, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    # return early if the parsed header doesn't contain any of the columns to filter (to avoid mangling no-header CSVs)
    csv_contents = read_csv(fpath)
    if not any(col in csv_contents[0].keys() for col in columns):
        return fpath

    # filter CSV contents
    csv_contents_filtered = [{k: v for k, v in row.items() if k not in columns}
                             for row in csv_contents]
    # write and return filtered CSV
    fpath_tmp = os.path.join(tempfile.mkdtemp(), os.path.basename(fpath))
    write_csv(csv_contents_filtered, fpath_tmp)

    return fpath_tmp


def filter_json_sidecars(fpath, fields):
    """
    Filter out fields from JSON sidecar files.

    This helps when checking the filesize and md5 of JSON files, which can contain differing branches, timestamps, etc.
    """
    def read_json(file_path):
        """Read JSON file into a dictionary."""
        with open(file_path, 'r', encoding='utf-8') as fp:
            return json.load(fp)

    def write_json(data, file_path):
        """Write a list of dictionary to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as fp:
            json.dump(data, fp, indent=4)

    # return early if json contents don't match the expected format of our JSON sidecar files
    # dict[list[dict]] (e.g. {'GeneratedBy': [{'Name': ..., 'Version': ... 'CodeUrl': ...}, ...]})
    json_contents = read_json(fpath)
    if not (isinstance(json_contents, dict) and
            all(isinstance(v, list) for v in json_contents.values()) and
            all(isinstance(item, dict) for v in json_contents.values() for item in v)):
        return fpath

    # filter JSON contents
    json_contents_filtered = {item_name: [{k: v for k, v in d.items() if k not in fields}
                                          for d in list_of_dicts]
                              for item_name, list_of_dicts in json_contents.items()}
    # write and return filtered JSON
    fpath_tmp = os.path.join(tempfile.mkdtemp(), os.path.basename(fpath))
    write_json(json_contents_filtered, fpath_tmp)

    return fpath_tmp


@pytest.fixture(autouse=True)
def run_in_tmp_path(tmp_path):
    """
    Temporarily change the working directory to a pytest temporary dir. This is needed to prevent tests from
    cluttering the working directory *or* the sct_testing_data directory with output files.

    Note: Because this uses the `tmp_path` fixture, which generates one temporary directory per test, this allows
          tests to _also_ call `tmp_path` and access the same directory (e.g. for checking outputs).
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(cwd)


@pytest.fixture(scope="session", autouse=True)
def test_data_integrity(request):
    files_checksums = dict()
    for root, _, files in os.walk(sct_test_path()):
        for f in files:
            fname = os.path.join(root, f)
            chksum = checksum(fname)
            files_checksums[fname] = chksum

    request.addfinalizer(lambda: check_testing_data_integrity(files_checksums))


@pytest.fixture(scope="module")
def verbose_logging():
    """
    Temporarily sets the logging state to be verbose.
    Module scoped to allow for tests to be run together w/o repeated calls to the logger
    """
    # Make logging verbose
    set_loglevel(verbose=True, caller_module_name=__name__)

    # Run our tests
    yield

    # Return the logger to be non-verbose
    set_loglevel(verbose=False, caller_module_name=__name__)


def checksum(fname: os.PathLike) -> str:
    with open(fname, 'rb') as f:
        data = f.read()
    return md5(data).hexdigest()


def check_testing_data_integrity(files_checksums: Mapping[os.PathLike, str]):
    changed = []
    new = []
    missing = []

    after = []

    for root, _, files in os.walk(sct_test_path()):
        for f in files:
            fname = os.path.join(root, f)
            chksum = checksum(fname)
            after.append(fname)

            if fname not in files_checksums:
                logger.warning(f"Discovered new file in sct_testing_data that didn't exist before: {(fname, chksum)}")
                new.append((fname, chksum))

            elif files_checksums[fname] != chksum:
                logger.error(f"Checksum mismatch for test data: {fname}. Got {chksum} instead of {files_checksums[fname]}")
                changed.append((fname, chksum))

    for fname, chksum in files_checksums.items():
        if fname not in after:
            logger.error(f"Test data missing after test:a: {fname}")
            missing.append((fname, chksum))

    assert not changed
    # assert not new
    assert not missing


@pytest.fixture(scope="session")
def tmp_path_qc(tmp_path_factory):
    """
    Create a session-scoped temporary directory for QC output.
    This is used to group generated QC reports into a single report, to allow for easier checking of outputs.
    """
    qc_path = tmp_path_factory.mktemp('qc-report-')
    return str(qc_path)  # most SCT scripts require string inputs for paths
