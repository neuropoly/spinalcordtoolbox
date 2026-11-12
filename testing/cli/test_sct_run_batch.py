# pytest unit tests for sct_run_batch

import glob
import os
import json
import yaml
import pathlib
import itertools

import pytest
from stat import S_IEXEC
from textwrap import dedent

from spinalcordtoolbox.scripts import sct_run_batch


@pytest.fixture
def dummy_script(tmp_path):
    """Dummy executable script that displays subject."""
    path_out = str(tmp_path / "dummy_script.sh")
    script_text = """
    #!/bin/bash
    SUBJECT=$1
    echo $SUBJECT
    """
    with open(path_out, 'w') as script:
        # indexing removes beginning newline
        script.write(dedent(script_text)[1:])
    # Set "Owner has execute permission" bit to 1 to ensure script is executable
    script_stat = os.stat(path_out)
    os.chmod(path_out, script_stat.st_mode | S_IEXEC)
    # NB: This does nothing on Windows. (https://docs.python.org/3/library/os.html#os.chmod)
    # > Although Windows supports chmod(), you can only set the file’s read-only flag with it
    # > (via the stat.S_IWRITE and stat.S_IREAD constants or a corresponding integer value).
    # > All other bits are ignored.
    return path_out


@pytest.fixture
def dummy_script_with_file_exclusion(tmp_path):
    """Dummy executable script that displays file only for non-excluded files."""
    path_out = str(tmp_path / "dummy_script_with_file_exclusion.sh")
    script_text = """
    #!/bin/bash
    SUBJECT=$1
    # check if FILE_T2 is in the list of excluded files
    FILE_T2="$SUBJECT"_T2w.nii.gz
    # process file if it is included
    if [[ -n "$INCLUDE_FILES" ]]; then
        if [[ " $INCLUDE_FILES " =~ " $FILE_T2 " ]]; then
            echo "$SUBJECT"
        fi
    fi
    # process file if it is not excluded
    if [[ -n "$EXCLUDE_FILES" ]]; then
        if [[ ! " $EXCLUDE_FILES " =~ " $FILE_T2 " ]]; then
            echo "$SUBJECT"
        fi
    fi
    """
    with open(path_out, 'w') as script:
        script.write(dedent(script_text)[1:])
    script_stat = os.stat(path_out)
    os.chmod(path_out, script_stat.st_mode | S_IEXEC)
    return path_out


@pytest.fixture(scope='module')
def subject_dirs(tmpdir_factory):
    """Generate dummy BIDS-like dataset with subject anat directories."""
    sub_names = ['sub-001', 'sub-002', 'sub-003', 'sub-010', 'sub-011', 'sub-012']
    sub_dirs = []
    data_dir = pathlib.Path(tmpdir_factory.mktemp('data'))
    for sub in sub_names:
        anat_dir = data_dir / sub / 'anat'
        anat_dir.mkdir(parents=True)
        sub_dirs.append(anat_dir)
    return data_dir, sub_names


def yml_file(tmp_path, subject_dirs, exclusion_type, yml_format, entry_type):
    """
    Function that generates an `include.yml` or `exclude.yml` file based on the given parameters.

    :param tmp_path: Pytest fixture providing a pathlib.Path temporary directory.
    :param subject_dirs: tuple containing the data directory path and list of subject names.
    :param exclusion_type: str, either 'include' or 'exclude', indicating the type of YML file to create.
    :param yml_format: str, either 'dict' or 'list', indicating the desired YML format.
    :param entry_type: str, either 'subject' or 'filename', indicating the type of entries in the YML file.
    """
    # construct the entries based on exclusion_type and entry_type
    _, sub_names = subject_dirs
    yml_to_write = (sub_names[::2] if exclusion_type == "exclude"  # every 2nd subject is excluded
                    else sub_names[1::2])                          # conversely, every other 2nd subject is included, too
    yml_to_write = (yml_to_write if entry_type == 'subject'
                    else [f"{sub}_T2w.nii.gz" for sub in yml_to_write])

    # if 'dict' is specified, then mimic the YML formatting of the QC report (dict of lists)
    if yml_format == 'dict':
        yml_to_write = {f"FILES_{value}": [value] for value in yml_to_write}

    # write the YML contents to a file
    fname_yml = tmp_path / f'{exclusion_type}.yml'
    with open(fname_yml, 'w') as fp:
        yaml.dump(yml_to_write, fp)

    return fname_yml


# Parametrize the YML generation fixtures by exclusion_type and yml_format
YML_PARAMETERS = list(itertools.product(
    ['include', 'exclude'],  # exclusion_type
    ['dict', 'list']         # yml_format
))


@pytest.fixture(params=YML_PARAMETERS, ids=["-".join(params) for params in YML_PARAMETERS])
def subject_yml(request, subject_dirs, tmp_path):
    """Generate a parametrized inclusion/exclusion YML file containing a list of subject names."""
    return yml_file(tmp_path, subject_dirs, *request.param, entry_type="subject")


# NB: We don't use @pytest.mark.parametrize() to set up the YML file because the YML fixture is parametrized instead
@pytest.mark.parametrize("use_config_file", [False, True], ids=["argv", "config"])
def test_yml_containing_subjects(tmp_path, subject_dirs, dummy_script, subject_yml, use_config_file):
    """Test that `-include-yml` and `-exclude-yml` properly filter subjects (e.g. sub-001, sub-002)."""
    # run script
    out = tmp_path / 'out'
    data_dir, sub_names = subject_dirs
    yml_type = 'include' if subject_yml.name == 'include.yml' else 'exclude'

    argv = ['-path-data', str(data_dir), '-path-output', str(out), '-script', dummy_script, f'-{yml_type}-yml', str(subject_yml)]
    if use_config_file:
        config_path = tmp_path / 'config.yml'
        config = {arg_name[1:].replace("-", "_"): val for arg_name, val in zip(argv[::2], argv[1::2])}
        with open(config_path, 'w') as fp:
            yaml.dump(config, fp)
        argv = ['-config', str(config_path)]
    sct_run_batch.main(argv=argv)

    # test log contents to see which subjects were processed
    # - excluded subjects: no log file (subject was not processed)
    # - included subjects: non-empty log file (`echo` should be run for that subject)
    script_name = os.path.splitext(os.path.basename(dummy_script))[0]
    for sub in sub_names:
        log_file = out / 'log' / f'{script_name}_{sub}.log'
        if sub in sub_names[::2]:  # every 2nd subject is excluded
            assert not log_file.exists()
        else:
            assert log_file.exists() and os.path.getsize(log_file) > 0


@pytest.fixture(params=YML_PARAMETERS, ids=["-".join(params) for params in YML_PARAMETERS])
def filename_yml(request, subject_dirs, tmp_path):
    """Generate a parametrized inclusion/exclusion YML file containing a list of filenames."""
    return yml_file(tmp_path, subject_dirs, *request.param,  entry_type="filename")


# NB: We don't use @pytest.mark.parametrize() to set up the YML file because the YML fixture is parametrized instead
def test_yml_containing_filenames(tmp_path, subject_dirs, dummy_script_with_file_exclusion, filename_yml):
    """Test that `-include-yml` and `-exclude-yml` properly filter subject filenames (e.g. sub-001_T2w.nii.gz)."""
    # run script
    out = tmp_path / 'out'
    data_dir, sub_names = subject_dirs
    yml_type = 'include' if filename_yml.name == 'include.yml' else 'exclude'
    sct_run_batch.main(argv=['-path-data', str(data_dir), '-path-out', str(out), '-script', dummy_script_with_file_exclusion,
                             f'-{yml_type}-yml', str(filename_yml)])

    # test log contents to see which subjects were processed
    # - excluded subjects: empty log file (subject was processed but file was skipped, so `echo` should not be run)
    # - included subjects: non-empty log file (`echo` should be run for that subject)
    script_name = os.path.splitext(os.path.basename(dummy_script_with_file_exclusion))[0]
    for sub in sub_names:
        log_file = out / 'log' / f'{script_name}_{sub}.log'
        if sub in sub_names[::2]:  # every 2nd subject is excluded
            assert log_file.exists() and os.path.getsize(log_file) == 0
        else:
            assert log_file.exists() and os.path.getsize(log_file) > 0


def test_config_with_args_warning(tmp_path, dummy_script):
    """
    Test that an error is thrown when trying to pass an argument ('-include') alongside a config file.
    """
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    config_path = tmp_path / 'config.json'

    with open(config_path, 'w') as f:
        config = {"jobs": 1, "path_data": str(data), "path_output": str(out)}
        json.dump(config, f)

    with pytest.warns(UserWarning, match=r'-config.*discouraged'):
        sct_run_batch.main(['-c', str(config_path), '-include', 'something', '-script', dummy_script])


def test_config_extra_value_warning(tmp_path, dummy_script):
    """
    Test that an error is thrown for passing a non-existent argument within a config file.
    """
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    config_path = str(tmp_path / 'config.yml')

    with open(config_path, 'w') as config:
        cnf_text = f"""
        jobs: 1
        path_data: {data}
        path_output: {out}
        unknowable: unknown
        """
        config.write(dedent(cnf_text))
        config.flush()

    with pytest.warns(UserWarning, match='unknowable'):
        sct_run_batch.main(['-c', config_path, '-script', dummy_script])


def test_only_one_include_exclude(tmp_path, dummy_script):
    """
    Test that an error is thrown for trying to pass both '-include'/'-include-list', or '-exclude'/'-exclude-list'.
    """
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    with pytest.raises(SystemExit) as e:
        sct_run_batch.main(['-include', 'arg', '-include-list', 'arg2',
                            '-path-data', str(data), '-path-out', str(out), '-script', dummy_script])
    assert e.value.code == 2  # Default error code given by `parser.error` within an ArgumentParser
    with pytest.raises(SystemExit) as e:
        sct_run_batch.main(['-exclude', 'arg', '-exclude-list', 'arg2',
                            '-path-data', str(data), '-path-out', str(out), '-script', dummy_script])
    assert e.value.code == 2  # Default error code given by `parser.error` within an ArgumentParser


def test_directory_inclusion_exclusion():
    """
    Test that -include, -include-list, -exclude, and -exclude-list properly filter directories.
    """
    filter_dirs = sct_run_batch._filter_directories  # for brevity

    # Test list of subjects by themselves
    sub_dir_list = ['sub001', 'sub002', 'sub003', 'sub010', 'sub011', 'sub012']
    assert filter_dirs(sub_dir_list, include="sub.*2") == ['sub002', 'sub012']
    assert filter_dirs(sub_dir_list, include_list=["sub001", "sub002"]) == ['sub001', 'sub002']
    assert filter_dirs(sub_dir_list, exclude="sub001") == ['sub002', 'sub003', 'sub010', 'sub011', 'sub012']
    assert filter_dirs(sub_dir_list, exclude_list=['sub010', 'sub011', 'sub012']) == ['sub001', 'sub002', 'sub003']

    # Test list of subjects with session subdirectories
    sess_dir_list = ['sub01/ses01', 'sub01/ses02', 'sub02/ses01', 'sub02/ses02', 'sub03/ses01', 'sub03/ses02']
    assert filter_dirs(sess_dir_list, include="sub01") == ['sub01/ses01', 'sub01/ses02']
    # NB: The `include_list` case below is tied to https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3598
    assert filter_dirs(sess_dir_list, include_list=["sub01", "sub02"]) == ['sub01/ses01', 'sub01/ses02',
                                                                           'sub02/ses01', 'sub02/ses02']
    assert filter_dirs(sess_dir_list, exclude="sub01") == ['sub02/ses01', 'sub02/ses02', 'sub03/ses01', 'sub03/ses02']
    assert filter_dirs(sess_dir_list, exclude_list=['sub01', 'sub02']) == ['sub03/ses01', 'sub03/ses02']


def test_directory_parsing(tmp_path):
    """
    Test that subject and session subdirectories are parsed correctly within a sample dataset structure.
    """

    parse_dir = sct_run_batch._parse_dataset_directory  # for brevity

    subject_names = ['sub-001', 'sub-002', 'sub-003']
    for sub in subject_names:
        (tmp_path / sub).mkdir()
        (tmp_path / f"{sub}.txt").touch()  # Add false positive files to ensure only directories are returned
    assert parse_dir(tmp_path) == subject_names
    assert parse_dir(tmp_path, subject_prefix="subject-") == []

    session_names = ['ses-001', 'ses-002']
    for sub in subject_names[:-1]:  # Skip creating 'ses' dirs in [-1] to test a mix of 'sub' and 'sub/ses' directories
        for ses in session_names:
            (tmp_path / sub / ses).mkdir()
            (tmp_path / sub / f"{ses}.txt").touch()  # Add false positive files to ensure only directories are returned
    assert parse_dir(tmp_path) == ([os.path.join(sub, ses) for sub in subject_names[:-1] for ses in session_names]
                                   + [subject_names[-1]])  # Last subject shouldn't have any session subdirectories
    assert parse_dir(tmp_path, ignore_ses=True) == subject_names


def test_non_executable_task(tmp_path, dummy_script):
    """
    Test that sct_run_batch can still process a non-executable script. (sct_run_batch will attempt
    to set the execute bit if passed a non-executable script.)
    """
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'

    # Set "Owner has execute permission" bit to 0 to ensure script is non-executable
    script_stat = os.stat(dummy_script)
    os.chmod(dummy_script, script_stat.st_mode & ~S_IEXEC)
    # NB: This does nothing on Windows. (https://docs.python.org/3/library/os.html#os.chmod)
    # > Although Windows supports chmod(), you can only set the file’s read-only flag with it
    # > (via the stat.S_IWRITE and stat.S_IREAD constants or a corresponding integer value).
    # > All other bits are ignored.

    sct_run_batch.main(['-include', '^t.*',
                        '-subject-prefix', '',
                        '-path-data', str(data), '-path-out', str(out),
                        '-script', dummy_script,
                        '-continue-on-error', 0])


def test_no_sessions(tmp_path, dummy_script):
    """
    Test that individual subjects (i.e. not in 'ses' subfolders) can be processed individually by sct_run_batch.
    """
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    # Create dummy BIDS directory without sessions
    for sub in ['01', '02']:
        (data / f'sub-{sub}' / 'anat').mkdir(parents=True)
    sct_run_batch.main(['-path-data', str(data), '-path-out', str(out), '-script', dummy_script])
    for sub in ['01', '02']:
        file_log = glob.glob(os.path.join(out, 'log', f'*sub-{sub}.log'))[0]
        assert f'sub-{sub}' in open(file_log, "r").read()


def test_separate_sessions(tmp_path, dummy_script):
    """
    Test that sessions ('ses') can be separated so that sct_run_batch can process each session folder separately.
    """
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    # Create dummy BIDS directory with sessions
    sub_ses_pairs = [('01', '01'), ('01', '02'), ('01', '03'), ('02', '01'), ('02', '02')]
    for sub, ses in sub_ses_pairs:
        (data / f'sub-{sub}' / f'ses-{ses}').mkdir(parents=True)
    sct_run_batch.main(['-path-data', str(data), '-path-out', str(out), '-script', dummy_script])
    for sub, ses in sub_ses_pairs:
        file_log = glob.glob(os.path.join(out, 'log', f'*sub-{sub}_ses-{ses}.log'))[0]
        assert os.path.join(f'sub-{sub}', f'ses-{ses}') in open(file_log, "r").read()
