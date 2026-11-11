# This file is vendored from ivadomed 2.9.10 (https://github.com/ivadomed/ivadomed),
# originally `ivadomed/utils.py`, licensed under the MIT License.
# See the LICENSE.md file in this directory for the full license text.
#
# Modifications for SCT:
#   - Removed training and CLI helpers that SCT doesn't use.
#   - Fixed flake8 warnings, without changing behavior. This includes using a raw string for the `'[\s]+'` regex, since
#     invalid escape sequences raise a `SyntaxWarning` on Python 3.12+.

import os
import subprocess
import datetime
import platform

from loguru import logger
from pathlib import Path

AXIS_DCT = {'sagittal': 0, 'coronal': 1, 'axial': 2}


def cuda(input_var, cuda_available=True, non_blocking=False):
    """Passes input_var to GPU.

    Args:
        input_var (Tensor): either a tensor or a list of tensors.
        cuda_available (bool): If False, then return identity
        non_blocking (bool):

    Returns:
        Tensor
    """
    if cuda_available:
        if isinstance(input_var, list):
            return [t.cuda(non_blocking=non_blocking) for t in input_var]
        else:
            return input_var.cuda(non_blocking=non_blocking)
    else:
        return input_var


def define_device(gpu_id):
    """Define the device used for the process of interest.

    Args:
        gpu_id (int): GPU ID.

    Returns:
        Bool, device: True if cuda is available.
    """
    import torch
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        logger.info("Cuda is not available.")
        logger.info("Working on {}.".format(device))
    if cuda_available:
        # Set the GPU
        gpu_id = int(gpu_id)
        torch.cuda.set_device(gpu_id)
        logger.info(f"Using GPU ID {gpu_id}")
    return cuda_available, device


def _git_info(commit_env='IVADOMED_COMMIT', branch_env='IVADOMED_BRANCH'):
    """Get ivadomed version info from GIT.

    This functions retrieves the ivadomed version, commit, branch and installation type.

    Args:
        commit_env (str):
        branch_env (str):
    Returns:
        str, str, str, str: installation type, commit, branch, version.
    """
    ivadomed_commit = os.getenv(commit_env, "unknown")
    ivadomed_branch = os.getenv(branch_env, "unknown")
    if check_exe("git") and Path(__ivadomed_dir__, ".git").is_dir():
        ivadomed_commit = __get_commit() or ivadomed_commit
        ivadomed_branch = __get_branch() or ivadomed_branch

    if ivadomed_commit != 'unknown':
        install_type = 'git'
    else:
        install_type = 'package'

    path_version = Path(__ivadomed_dir__, 'ivadomed', 'version.txt')
    with path_version.open() as f:
        version_ivadomed = f.read().strip()

    return install_type, ivadomed_commit, ivadomed_branch, version_ivadomed


def check_exe(name):
    """Ensure that a program exists.

    Args:
        name (str): Name or path to program.
    Returns:
        str or None: path of the program or None
    """

    def is_exe(fpath):
        return Path(fpath).is_file() and os.access(fpath, os.X_OK)

    fpath = Path(name).parent
    if fpath and is_exe(name):
        return fpath
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = str(Path(path, name))
            if is_exe(exe_file):
                return exe_file

    return None


def __get_commit(path_to_git_folder=None):
    """Get GIT ivadomed commit.

    Args:
        path_to_git_folder (str): Path to GIT folder.
    Returns:
        str: git commit ID, with trailing '*' if modified.
    """
    if path_to_git_folder is None:
        path_to_git_folder = __ivadomed_dir__
    else:
        path_to_git_folder = Path(path_to_git_folder).expanduser().absolute()

    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        commit = output.decode().strip()
    else:
        commit = "?!?"

    p = subprocess.Popen(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        unclean = True
        for line in output.decode().strip().splitlines():
            line = line.rstrip()
            if line.startswith("??"):  # ignore ignored files, they can't hurt
                continue
            break
        else:
            unclean = False
        if unclean:
            commit += "*"

    return commit


def __get_branch():
    """Get ivadomed branch.

    Args:

    Returns:
        str: ivadomed branch.
    """
    p = subprocess.Popen(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, cwd=__ivadomed_dir__)
    output, _ = p.communicate()
    status = p.returncode

    if status == 0:
        return output.decode().strip()


def _version_string():
    install_type, ivadomed_commit, ivadomed_branch, version_ivadomed = _git_info()
    if install_type == "package":
        return version_ivadomed
    else:
        return "{install_type}-{ivadomed_branch}-{ivadomed_commit}".format(**locals())


__ivadomed_dir__ = Path(__file__).resolve().parent.parent
__version__ = _version_string()


def get_timestamp() -> str:
    """
    Return a datetime string in the format YYYY-MM-DDTHHMMSS.(sub-precision)
    Returns:
    """
    timestamp = datetime.datetime.now().isoformat().replace(":", "")
    return timestamp


def get_system_memory() -> float:
    """
    Return the system memory in GB.
    Returns:
    """
    current_platform = platform.system()
    if current_platform == "Linux":
        return get_linux_system_memory()
    elif current_platform == "Windows":
        return get_win_system_memory()
    elif current_platform == "Darwin":
        return get_mac_system_memory()


def get_win_system_memory() -> float:
    """
    Obtain the amount of memory available on Windows system.
    Returns: memory in GB
    Source: https://stackoverflow.com/a/21589439
    """
    process = os.popen('wmic memorychip get capacity')
    result = process.read()
    process.close()
    totalMem = 0
    for m in result.split("  \n\n")[1:-1]:
        totalMem += int(m)
    return totalMem / (1024 ** 3)


def get_linux_system_memory() -> float:
    """
    Obtain the amount of memory available on Linux system.
    Returns: memory in GB
    Source: https://stackoverflow.com/a/28161352
    """
    import os
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448
    mem_gib = mem_bytes / (1024. ** 3)  # e.g. 3.74
    return mem_gib


def get_mac_system_memory() -> float:
    """
    Obtain the amount of memory available on MacOS system.
    Returns: memory in GB
    Source: https://apple.stackexchange.com/a/4296
    """

    import subprocess
    import re

    # Get process info
    ps = subprocess.Popen(['ps', '-caxm', '-orss,comm'], stdout=subprocess.PIPE).communicate()[0].decode()

    # Iterate processes
    processLines = ps.split('\n')
    sep = re.compile(r'[\s]+')
    rssTotal = 0  # kB
    for row in range(1, len(processLines)):
        rowText = processLines[row].strip()
        rowElements = sep.split(rowText)
        try:
            rss = float(rowElements[0]) * 1024
        except Exception:
            rss = 0  # ignore...
        rssTotal += rss

    return rssTotal / 1024 ** 3
