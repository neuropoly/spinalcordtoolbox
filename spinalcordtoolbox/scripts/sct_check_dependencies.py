#!/usr/bin/env python
#
# Check the installation and environment variables of the toolbox and its dependencies.
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

# TODO: if fail, run with log and display message to send to sourceforge.
# TODO: check chmod of binaries
# TODO: find another way to create log file. E.g. print(). For color as well.
# TODO: manage .cshrc files

import sys
import io
import os
import platform
import importlib
import warnings
import psutil
import traceback
from typing import Sequence

import requirements
import torch.cuda
from torch import __version__ as __torch_version__

from spinalcordtoolbox.download import default_datasets, is_installed
from spinalcordtoolbox.utils.shell import SCTArgumentParser
from spinalcordtoolbox.utils.sys import (sct_dir_local_path, init_sct, run_proc, __version__, __sct_dir__,
                                         set_loglevel, ANSIColors16, _which_sct_binaries)


def _test_condition(condition):
    """Test condition formatted in requirements"""
    # Define Environment markers (https://www.python.org/dev/peps/pep-0508/#environment-markers)
    os_name = os.name  # noqa: F841
    platform_machine = platform.machine()  # noqa: F841
    platform_release = platform.release()  # noqa: F841
    platform_system = platform.system()  # noqa: F841
    platform_version = platform.version()  # noqa: F841
    python_full_version = platform.python_version()  # noqa: F841
    platform_python_implementation = platform.python_implementation()  # noqa: F841
    python_version = platform.python_version()[:3]  # noqa: F841
    sys_platform = sys.platform  # noqa: F841
    # Test condition
    return eval(condition)


def resolve_module(framework_name):
    """This function will resolve the framework name
    to the module name in cases where it is different.

    :param framework_name: the name of the framework.
    :return: the tuple (module name, supress stderr).
    """
    # Framework name : (module name, suppress stderr)
    modules_map = {
        'futures': ('concurrent.futures', False),
        'requirements-parser': ('requirements', False),
        'scikit-image': ('skimage', False),
        'scikit-learn': ('sklearn', False),
        'pyqt5': ('PyQt5', False),
        'pyqt5-qt5': ('PyQt5.QtCore', False),  # Importing Qt may catch issue #2523
        'pyqt5-sip': ('PyQt5.sip', False),
        'pyyaml': ('yaml', False),
        'futures': ("concurrent.futures", False),
        'opencv': ('cv2', False),
        'msvc-runtime': ('msvc_runtime', False),
        'mkl-service': (None, False),
        'pytest-cov': ('pytest_cov', False),
        'urllib3[secure]': ('urllib3', False),
        'pytest-xdist': ('xdist', False),
        'protobuf': ('google.protobuf', False),
        'pillow': ('PIL', False),
        # Importing `matplotlib_inline` requires IPython, but we don't install IPython (on purpose). This is because
        # `matplotlib_inline` is only needed to run SCT scripts in Jupyter notebooks, and IPython would already be
        # installed in the parent context. So, we map `matplotlib-inline` to None to skip import (which would fail).
        'matplotlib-inline': (None, False),
        # The `nnunetv2-neuropoly` fork maps to the `nnunetv2` name
        'nnunetv2-neuropoly': ('nnunetv2', False),
    }

    try:
        return modules_map[framework_name.lower()]
    except KeyError:
        return (framework_name, False)


def module_import(module_name, suppress_stderr=False):
    """Import a module using importlib.

    :param module_name: the name of the module.
    :param suppress_stderr: if the stderr should be suppressed.
    :return: the imported module.
    """
    if suppress_stderr:
        original_stderr = sys.stderr
        if sys.hexversion < 0x03000000:
            sys.stderr = io.BytesIO()
        else:
            sys.stderr = io.TextIOWrapper(io.BytesIO(), sys.stderr.encoding)
        try:
            module = importlib.import_module(module_name)
        except Exception:
            sys.stderr = original_stderr
            raise
        else:
            sys.stderr = original_stderr

    else:
        module = importlib.import_module(module_name)
    return module


def get_version(module):
    """
    Get module version. This function is required due to some exceptions in fetching module versions.
    :param module: the module to get version from
    :return: string: the version of the module
    """
    # Unfortunately importing PyQt5.Qt makes sklearn import crash on Ubuntu 14.04 (corresponding to Debian's jessie)
    # so we don't display the version for this distros.
    # See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2522#issuecomment-559310454
    if 'PyQt5' in module.__name__ and 'jessie' in platform.platform():
        version = None
    elif module.__name__ == 'PyQt5.QtCore':
        from PyQt5.Qt import QT_VERSION_STR
        version = QT_VERSION_STR
    elif module.__name__ == 'PyQt5':
        from PyQt5.Qt import PYQT_VERSION_STR
        version = PYQT_VERSION_STR
    else:
        version = getattr(module, "__version__", getattr(module, "__VERSION__", None))
    return version


def print_line(string):
    """print without carriage return"""
    sys.stdout.write(string.ljust(52, '.'))
    sys.stdout.flush()


def print_ok(more=None):
    print(f"[{ANSIColors16.LightGreen}OK{ANSIColors16.ResetAll}]{more if more is not None else ''}")


def print_warning(more=None):
    print(f"[{ANSIColors16.LightYellow}WARNING{ANSIColors16.ResetAll}]{more if more is not None else ''}")


def print_fail(more=None):
    print(f"[{ANSIColors16.LightRed}FAIL{ANSIColors16.ResetAll}]{more if more is not None else ''}")


def add_bash_profile(string):
    bash_profile = os.path.expanduser(os.path.join("~", ".bash_profile"))
    with io.open(bash_profile, "a") as file_bash:
        file_bash.write(f"\n{string}")


def get_dependencies(requirements_txt=None):
    if requirements_txt is None:
        requirements_txt = sct_dir_local_path("requirements.txt")

    requirements_txt = open(requirements_txt, "r", encoding="utf-8")

    # workaround for https://github.com/madpah/requirements-parser/issues/39
    warnings.filterwarnings(message='Private repos not supported', action='ignore', category=UserWarning)

    for req in requirements.parse(requirements_txt):
        if ';' in req.line:  # handle environment markers; TODO: move this upstream into requirements-parser
            condition = req.line.split(';', 1)[-1].strip()
            if not _test_condition(condition):
                continue
        pkg = req.name
        # TODO: just return req directly and make sure caller can deal with fancier specs
        ver = dict(req.specs).get("==", None)
        yield pkg, ver


def get_parser():
    parser = SCTArgumentParser(
        description='Check the installation and environment variables of the toolbox and its dependencies.'
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-complete',
        help="Complete test.",
        action="store_true")
    optional.add_argument(
        "-short",
        help="Short test. Only shows SCT version, CPU cores and RAM available.",
        action="store_true")

    # Add common arguments
    parser.add_common_args()

    return parser


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    complete_test = arguments.complete
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    print("\nSYSTEM INFORMATION"
          "\n------------------")

    print("SCT info:")
    print(f"- version: {__version__}")
    print(f"- path: {__sct_dir__}")

    # initialization
    install_software = 0
    e = 0

    # complete test
    if complete_test:
        print(run_proc('date', verbose))
        print(run_proc('whoami', verbose))
        print(run_proc('pwd', verbose))
        bash_profile = os.path.expanduser(os.path.join("~", ".bash_profile"))
        if os.path.isfile(bash_profile):
            with io.open(bash_profile, "r") as f:
                print(f.read())
        bashrc = os.path.expanduser(os.path.join("~", ".bashrc"))
        if os.path.isfile(bashrc):
            with io.open(bashrc, "r") as f:
                print(f.read())

    # check OS
    if sys.platform.startswith('darwin'):
        os_running = 'osx'
    elif sys.platform.startswith('linux'):
        os_running = 'linux'
    elif sys.platform.startswith('win32'):
        os_running = 'windows'
    else:
        os_running = 'unknown'

    print(f'OS: {os_running} ({platform.platform()})')
    print(f'CPU cores: Available: {psutil.cpu_count()}, Used by ITK functions: {int(os.getenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", 0))}')

    ram = psutil.virtual_memory()
    factor_MB = 1024 * 1024
    print(f'RAM: Total: {ram.total // factor_MB}MB, Used: {ram.used // factor_MB}MB, Available: {ram.available // factor_MB}MB')

    # Check if SCT was installed with GPU version of torch (NB: On macOS, `torch` will always be CPU-only)
    gpu_torch_installed = ("+cpu" not in __torch_version__) and not sys.platform.startswith('darwin')

    if gpu_torch_installed:
        try:
            status, n_gpus = run_proc([
                "nvidia-smi",
                "--query-gpu=count",
                "--format=csv,noheader",
                "-i", "0",
            ], verbose=0, raise_exception=False)
        except FileNotFoundError as e:
            e.strerror = "GPU version of torch is installed, but could not find NVIDIA's GPU software"
            raise e
        if status == 0:
            for n in range(int(n_gpus)):
                _, output = run_proc([
                    "nvidia-smi",
                    "-i", str(n),
                    "--query-gpu=gpu_name,driver_version,vbios_version,memory.total,memory.free",
                    "--format=csv,noheader",
                ], verbose=0, raise_exception=False)
                gpu_name, driver_version, vbios_version, mem_total, mem_free = [s.strip() for s in output.split(",")]
                print(f"GPU {n}: {gpu_name} "
                      f"(Driver: {driver_version}, vBIOS: {vbios_version}) "
                      f"[VRAM Free: {mem_free}/{mem_total}]")

    if arguments.short:
        sys.exit()

    # Print 'optional' header only if any of the 'optional' checks will be triggered
    if not sys.platform.startswith('win32') or gpu_torch_installed:
        print("\nOPTIONAL DEPENDENCIES"
              "\n---------------------")

    # Check version of FSLeyes
    # NB: We put this section first because typically, it will error out, since FSLeyes isn't installed by default.
    #     SCT devs want to have access to this information, but we don't want to scare our users into thinking that
    #     there's a critical error. So, we put it up top to allow the installation to end on a nice "OK" note.
    if not sys.platform.startswith('win32'):
        print_line('Check FSLeyes version')
        cmd = 'fsleyes --version'
        status, output = run_proc(cmd, verbose=0, raise_exception=False)
        # Exit code 0 - command has run successfully
        if status == 0:
            # Fetch only version number (full output of 'fsleyes --version' is 'fsleyes/FSLeyes version 0.34.2')
            fsleyes_version = output.split()[2]
            print_ok(more=f" ({fsleyes_version})")
        else:
            print('[  ]')
            print('  ', (status, output))

    # Check GPU dependencies (but only if the GPU version of torch was installed). We install CPU torch by default, so
    # if the GPU version is present, then the user must have gone out of their way to make GPU inference work.
    if gpu_torch_installed:
        print_line('Check if CUDA is available in PyTorch')
        if not torch.cuda.is_available():
            print_fail(" (torch.cuda.is_available() returned False)")
        else:
            # NB: If CUDA is available, we can perform further GPU tests
            print_ok()

            print_line('Check CUDA version used by PyTorch')
            for cuda_envvar in ['CUDA_HOME', 'CUDA_PATH']:
                if cuda_envvar in os.environ:
                    print_warning(f" ({torch.version.cuda} (NB: {cuda_envvar} is set to '{os.environ[cuda_envvar]}' "
                                  f"which may override torch's built-in CUDA))")
                    break
            else:
                print_ok(f" ({torch.version.cuda})")

            print_line('Check number of GPUs available to PyTorch')
            if torch.cuda.device_count():
                print_ok(f" (Device count: {torch.cuda.device_count()})")
            else:
                print_fail(f" (torch.version.count returned {torch.cuda.device_count()})")

            print_line('Testing PyTorch (torch.tensor multiplication)')
            try:
                torch.manual_seed(1337)
                A = torch.randn(30000, 10000, dtype=torch.float16, device=torch.device("cuda"))
                B = torch.randn(10000, 20000, dtype=torch.float16, device=torch.device("cuda"))
                A @ B
                print_ok()
            except Exception as err:
                print_fail()
                print(f"An error occurred -> {err}")
                print(f"Full traceback: {traceback.format_exc()}")

    print("\nMANDATORY DEPENDENCIES"
          "\n----------------------")

    # check if Python path is within SCT path
    print_line('Check Python executable')
    path_python = sys.executable
    if __sct_dir__ in path_python:
        print_ok()
        print(f'  Using bundled python {sys.version} at {path_python}')
    else:
        print_warning()
        print(f'  Using system python which is unsupported: {path_python}')

    # Import matplotlib.pyplot here (before PyQt can be imported) in order to mitigate a libgcc error
    # See also: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3511#issuecomment-912167649
    import matplotlib.pyplot as plt

    for dep_pkg, dep_ver_spec in get_dependencies():
        if dep_ver_spec is None:
            print_line(f'Check if {dep_pkg} is installed')
        else:
            print_line(f'Check if {dep_pkg} ({dep_ver_spec}) is installed')

        try:
            module_name, suppress_stderr = resolve_module(dep_pkg)
            # If a module cannot be imported, then its `dep_pkg` name should be resolved to `module_name=None`
            if module_name is None:
                print_ok()
                continue
            module = module_import(module_name, suppress_stderr)
            version = get_version(module)

            if dep_ver_spec is not None and version is not None and dep_ver_spec != version:
                print_warning(more=f" ({version} != {dep_ver_spec} mandated version))")
            elif version is not None:
                print_ok(more=f" ({version})")
            else:
                print_ok()

        except Exception as err:
            print_fail()
            print(f"An error occured while importing module {dep_pkg} -> {err}")
            print(f"Full traceback: {traceback.format_exc()}")
            install_software = 1

    print_line('Check if spinalcordtoolbox is installed')
    try:
        importlib.import_module('spinalcordtoolbox')
        print_ok()
    except ImportError:
        print_fail("Unable to import spinalcordtoolbox module.")
        install_software = 1

    # Check ANTs integrity
    print_line('Check ANTs compatibility with OS ')
    cmd = ["sct_testing", os.path.join(__sct_dir__, "testing", "dependencies", "test_ants.py")]
    status, output = run_proc(cmd, verbose=0, raise_exception=False, is_sct_binary=True)
    if status == 0:
        print_ok()
    else:
        print_fail()
        print(output)
        e = 1
    if complete_test:
        print(f'>> {cmd}')
        print((status, output), '\n')

    # check PropSeg compatibility with OS
    if sys.platform.startswith('win32'):
        print_line("Skipping PropSeg compatibility check ")
        print("[  ] (Not supported on 'native' Windows (without WSL))")
    else:
        print_line('Check PropSeg compatibility with OS ')
        status, output = run_proc('isct_propseg', verbose=0, raise_exception=False, is_sct_binary=True)
        if status in (0, 1):
            print_ok()
        else:
            print_fail()
            print(output)
            e = 1
        if complete_test:
            print((status, output), '\n')

    print_line('Check if figure can be opened with PyQt')
    if sys.platform.startswith("linux") and 'DISPLAY' not in os.environ:
        print_fail(" ($DISPLAY not set on X11-supporting system)")
    else:
        try:
            from PyQt5.QtWidgets import QApplication, QLabel
            _ = QApplication([])
            label = QLabel('Hello World!')
            label.show()
            label.close()
            print_ok()
        except Exception as err:
            print_fail()
            print(err)

    print_line('Check if figure can be opened with matplotlib')
    try:
        import matplotlib
        # If matplotlib is using a GUI backend, the default 'show()` function will be overridden
        # See: https://github.com/matplotlib/matplotlib/issues/20281#issuecomment-846467732
        fig = plt.figure()  # NB: `plt` was imported earlier in the script to avoid a libgcc error
        if getattr(fig.canvas.manager.show, "__func__", None) != matplotlib.backend_bases.FigureManagerBase.show:
            print_ok(f" (Using GUI backend: '{matplotlib.get_backend()}')")
        else:
            print_fail(f" (Using non-GUI backend '{matplotlib.get_backend()}')")
    except Exception as err:
        print_fail()
        print(err)

    # check if data folder contains the required default depedencies
    for dataset_name in default_datasets() + [_which_sct_binaries()]:
        print_line(f"Check data dependency '{dataset_name}'")
        if is_installed(dataset_name):
            print_ok()
        else:
            print_fail(f" Run 'sct_download_data -d {dataset_name}' to reinstall")

    print('')
    sys.exit(e + install_software)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
