"""
Filesystem related helpers and utilities

Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import sys
import io
import os
import shutil
import tempfile
import datetime
import logging
from pathlib import Path
from contextlib import contextmanager

import portalocker

from .sys import printv

logger = logging.getLogger(__name__)


def tmp_create(basename):
    """Create temporary folder and return its path
    """
    prefix = f"sct_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{basename}_"
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    logger.debug(f"Creating temporary folder ({tmpdir})")
    return tmpdir


# Modified from https://shallowsky.com/blog/programming/python-tee.html
class Tee:
    def __init__(self, _fd1, _fd2):
        self.fd1 = _fd1
        self.fd2 = _fd2

    def close(self):
        # We can't assume that we own the underlying files exclusively; maybe
        # another part of the code also has a reference to them, in which case
        # it would be rude to actually close them. But, we can give up our
        # references to them, and this should automatically close them if the
        # reference count goes to zero.
        del self.fd1, self.fd2

    def write(self, text):
        self.fd1.write(text)
        self.fd2.write(text)

    def flush(self):
        self.fd1.flush()
        self.fd2.flush()

    def isatty(self):
        # This method is needed to ensure that `printv` correctly applies color formatting when sys.stdout==Tee().
        # Use utils.csi_filter if you want to strip the color codes from only one half of the Tee
        return self.fd1.isatty() or self.fd2.isatty()


def copy_helper(src, dst, verbose=1):
    """Copy src to dst, almost like shutil.copy
    If src and dst are the same files, don't crash.
    """
    if not os.path.isfile(src):
        folder = os.path.dirname(src)
        contents = os.listdir(folder)
        raise ValueError(f"Couldn't find {os.path.basename(src)} in {folder} (contents: {contents})")

    try:
        logger.info(f"cp {src} {dst}")
        shutil.copy(src, dst)
    except Exception as e:
        if sys.hexversion < 0x03000000:
            if isinstance(e, shutil.Error) and "same file" in str(e):
                return
        else:
            if isinstance(e, shutil.SameFileError):
                return
        raise  # Must be another error


def rmtree(folder, verbose=1):
    """Recursively remove folder, almost like shutil.rmtree
    """
    printv(f"rm -rf {folder}", verbose=verbose, type="code")
    shutil.rmtree(folder, ignore_errors=True)


def extract_fname(fpath):
    """
    Split a full path into a parent folder component, filename stem and extension.

    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    """
    parent, filename = os.path.split(fpath)
    if filename.endswith(".nii.gz"):
        stem, ext = filename[:-7], ".nii.gz"
    else:
        stem, ext = os.path.splitext(filename)
    return parent, stem, ext


def get_absolute_path(fname):
    if os.path.isfile(fname) or os.path.isdir(fname):
        return os.path.realpath(fname)
    else:
        printv(f'\nERROR: {fname} does not exist. Exit program.\n', 1, 'error')


def check_file_exist(fname, verbose=1):
    if fname[0] == '-':
        # fname should be a warping field that will be inverted, ignore the "-"
        fname_to_test = fname[1:]
    else:
        fname_to_test = fname
    if os.path.isfile(fname_to_test):
        if verbose:
            printv(f'  OK: {fname}', verbose, 'normal')
        return True
    else:
        printv(f'\nERROR: The file {fname} does not exist. Exit program.\n', 1, 'error')
        return False


class TempFolder(object):
    """This class will create a temporary folder."""

    def __init__(self, basename, verbose=0):
        self.path_tmp = tmp_create(basename)
        self.previous_path = None
        self.verbose = verbose

    def chdir(self):
        """This method will change the working directory to the temporary folder."""
        self.previous_path = os.getcwd()
        os.chdir(self.path_tmp)

    def chdir_undo(self):
        """This method will return to the previous working directory, the directory
        that was the state before calling the chdir() method."""
        if self.previous_path is not None:
            os.chdir(self.previous_path)

    def get_path(self):
        """Return the temporary folder path."""
        return self.path_tmp

    def copy_from(self, filename):
        """This method will copy a specified file to the temporary folder.

        :param filename: The filename to copy into the folder.
        """
        file_fname = os.path.basename(filename)
        copy(filename, self.path_tmp, verbose=self.verbose)
        return os.path.join(self.path_tmp, file_fname)

    def cleanup(self):
        """Remove the created folder and its contents."""
        rmtree(self.path_tmp, verbose=self.verbose)


def cache_signature(input_files=[], input_params={}):
    """
    Create a signature to be used for caching purposes

    :param input_files: paths of input files (that can influence output)
    :param input_params: input parameters (that can influence output)

    Notes:

    - Use this with cache_valid (to validate caching assumptions)
      and cache_save (to store them)

    - Using this system, the outputs are not checked, only the
      inputs and parameters (to regenerate the outputs in case
      these change).
      If the outputs have been modified directly, then they may
      be reused.
      To prevent that, the next step would be to record the outputs
      signature in the cache file, so as to also verify them prior
      to taking a shortcut.

    """
    import hashlib
    h = hashlib.md5()
    for path in input_files:
        with io.open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
    for k, v in sorted(input_params.items()):
        h.update(str(type(k)).encode('utf-8'))
        h.update(str(k).encode('utf-8'))
        h.update(str(type(v)).encode('utf-8'))
        h.update(str(v).encode('utf-8'))

    return f"# Cache file generated by SCT\nDEPENDENCIES_SIG={h.hexdigest()}\n".encode()


def cache_valid(cachefile, sig_expected):
    """
    Verify that the cachefile contains the right signature
    """
    if not os.path.exists(cachefile):
        return False
    with io.open(cachefile, "rb") as f:
        sig_measured = f.read()
    return sig_measured == sig_expected


def cache_save(cachefile, sig):
    """
    Save signature to cachefile

    :param cachefile: path to cache file to be saved
    :param sig: cache signature created with cache_signature()
    """
    with io.open(cachefile, "wb") as f:
        f.write(sig)


def mv(src, dst, verbose=1):
    """Move a file from src to dst (adding a logging message)."""
    printv(f"mv {src} {dst}", verbose=verbose, type="code")
    # NB: We specify `shutil.copyfile` to override the default of `shutil.copy2`.
    #     (`copy2` copies file metadata, but doing so fails with a PermissionError on WSL installations where the
    #     src/dest are on different devices. So, we use `copyfile` instead, which doesn't preserve file metadata.)
    #     Fixes https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3832.
    shutil.move(src, dst, copy_function=shutil.copyfile)


def copy(src, dst, verbose=1):
    """Copy src to dst, almost like shutil.copy
    If src and dst are the same files, don't crash.
    """
    if not os.path.isfile(src):
        folder = os.path.dirname(src)
        contents = os.listdir(folder)
        raise Exception(f"Couldn't find {os.path.basename(src)} in {folder} (contents: {contents})")
    try:
        printv(f"cp {src} {dst}", verbose=verbose, type="code")
        shutil.copy(src, dst)
    except Exception as e:
        if sys.hexversion < 0x03000000:
            if isinstance(e, shutil.Error) and "same file" in str(e):
                return
        else:
            if isinstance(e, shutil.SameFileError):
                return
        raise  # Must be another error


def relpath_or_abspath(child_path, parent_path):
    """
    Try to find a relative path between a child path and its parent path. If it doesn't exist,
    then the child path is not within the parent path, so return its abspath instead.
    """
    abspath = Path(child_path).resolve()
    try:
        return abspath.relative_to(parent_path)
    except ValueError:
        return abspath


@contextmanager
def mutex(name):
    """
    Use a bounded semaphore as a mutex to prevent parallel processes from running.

    portalocker.BoundedSemaphore is very similar to threading.BoundedSemaphore,
    but works across multiple processes and across multiple operating systems. So,
    we can spawn multiple processes using `sct_run_batch`, while still ensuring
    that we create QC reports sequentially (without mangling the output files).

    We use a mutex over a lock because the mutex doesn't depend on the destination
    of the locked files, which allows us to avoid locking on e.g. NFS-mounted drives.
    """
    semaphore = portalocker.BoundedSemaphore(maximum=1, name=name, timeout=60, check_interval=0.1)
    semaphore.acquire()
    try:
        yield semaphore
    finally:
        semaphore.release()
