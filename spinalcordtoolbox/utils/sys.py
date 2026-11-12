"""
System related utilities

Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import io
import sys
import os
import shutil
import logging
import subprocess
import shlex
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import tqdm

# Expose Tensorflow's LazyLoader class in `utils.sys` namespace
from contrib.tensorflow.lazy_loader import LazyLoader  # noqa
from spinalcordtoolbox.utils.profiling import begin_global_timer

logger = logging.getLogger(__name__)


def stylize(string, styles):
    """Helper function that mimics colored.stylize to reduce boilerplate when coloring text."""
    if not isinstance(styles, list):
        styles = [styles]
    # Convert style names into style codes
    style_codes = []
    for style in styles:
        # Already in code form, apply directly
        if style.startswith("\033"):
            style_codes.append(style)
        # Not in code form, fetch from class
        else:
            style_codes.append(getattr(ANSIColors16, style, ANSIColors16.ResetAll))
    return "".join(style_codes) + string + ANSIColors16.ResetAll


class ANSIColors16(object):
    """This class defines the ANSI color escape codes for terminals that support 16 colors.

    Notes:
        - Most terminals support 8 colors (base color set) or 16 colors (base colors + light colors).
        - We use these codes instead of dedicated color packages (colored, colorama) because those packages
          are meant for 256-color coloring, which only some terminals support. (Notably, the Windows Command Prompt
          does not support 256 colors.)
        - Further reading: https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html#rich-text
        - Source for codes: https://pkg.go.dev/github.com/whitedevops/colors
    """
    ResetAll = "\033[0m"

    Bold = "\033[1m"
    Dim = "\033[2m"
    Underlined = "\033[4m"
    Blink = "\033[5m"
    Reverse = "\033[7m"
    Hidden = "\033[8m"

    ResetBold = "\033[21m"
    ResetDim = "\033[22m"
    ResetUnderlined = "\033[24m"
    ResetBlink = "\033[25m"
    ResetReverse = "\033[27m"
    ResetHidden = "\033[28m"

    Default = "\033[39m"
    Black = "\033[30m"
    Red = "\033[31m"
    Green = "\033[32m"
    Yellow = "\033[33m"
    Blue = "\033[34m"
    Magenta = "\033[35m"
    Cyan = "\033[36m"
    LightGray = "\033[37m"
    DarkGray = "\033[90m"
    LightRed = "\033[91m"
    LightGreen = "\033[92m"
    LightYellow = "\033[93m"
    LightBlue = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan = "\033[96m"
    White = "\033[97m"

    BackgroundDefault = "\033[49m"
    BackgroundBlack = "\033[40m"
    BackgroundRed = "\033[41m"
    BackgroundGreen = "\033[42m"
    BackgroundYellow = "\033[43m"
    BackgroundBlue = "\033[44m"
    BackgroundMagenta = "\033[45m"
    BackgroundCyan = "\033[46m"
    BackgroundLightGray = "\033[47m"
    BackgroundDarkGray = "\033[100m"
    BackgroundLightRed = "\033[101m"
    BackgroundLightGreen = "\033[102m"
    BackgroundLightYellow = "\033[103m"
    BackgroundLightBlue = "\033[104m"
    BackgroundLightMagenta = "\033[105m"
    BackgroundLightCyan = "\033[106m"
    BackgroundWhite = "\033[107m"


# LOGGING GUIDELINES
# - Info: Messages that help the user understand what step in the processing
#         they are at (i.e. messages meant to avoid long periods of processing
#         without any feedback).
# - Debug: Info messages that do not meaningfully help the user understand what
#          processing step they are at (i.e. information that is primarily
#          useful in debugging)
# - Warning: Actionable steps the user can take (i.e. CLI options to consider,
#            potential data issues) but that SCT can automatically handle.
# - Error: Unhandled warnings (i.e. issues with CLI args or input data that
#          cannot be automatically handled)
LOGGING_COLOURS = {
    # SCT-specific `printv` logging colors
    'normal': ANSIColors16.ResetAll,
    'info': ANSIColors16.LightGreen,
    'code': ANSIColors16.LightBlue,
    'bold': ANSIColors16.Bold,
    'process': ANSIColors16.LightMagenta,

    # Overlap between SCT's `printv` log levels and `logging` library's log levels
    'debug': ANSIColors16.DarkGray,
    # 'info': ANSIColors16.Default,
    'warning': [ANSIColors16.LightYellow, ANSIColors16.Bold],
    'error': [ANSIColors16.LightRed, ANSIColors16.Bold],
    'critical': [ANSIColors16.Red, ANSIColors16.Bold],
}


# Allow numeric level values from `logging` (10, 20, 30...) to be used as keys as well
# (https://docs.python.org/3/library/logging.html#levels)
# NB: 'info' is already used by `printv` for LightGreen, so we can't map logging.INFO to 'info'.
#     Instead, we need to map logging.INFO to 'normal' to preserve the look of logging.info calls.
# FIXME: We could change the `printv` info to be 'notice' instead, and then unify 'info' to be the same color for
#        both `printv` and `logging`.
LOGGING_COLOURS[logging.NOTSET] = LOGGING_COLOURS['normal']      # 0
LOGGING_COLOURS[logging.DEBUG] = LOGGING_COLOURS['debug']        # 10
LOGGING_COLOURS[logging.INFO] = LOGGING_COLOURS['normal']        # 20
LOGGING_COLOURS[logging.WARNING] = LOGGING_COLOURS['warning']    # 30
LOGGING_COLOURS[logging.ERROR] = LOGGING_COLOURS['error']        # 40
LOGGING_COLOURS[logging.CRITICAL] = LOGGING_COLOURS['critical']  # 50


class CustomLoggingFormatter(logging.Formatter):
    """
    Static string manipulation class, applying SCT's predefined ANSI color
    codes from `printv` to the `logging` library's logs as well.
    """
    _formatters = {
        level: logging.Formatter(stylize("%(message)s", colour))
        for level, colour in LOGGING_COLOURS.items()
    }

    def format(self, record):
        formatter = self._formatters[record.levelno]
        return formatter.format(record)


def set_loglevel(verbose, caller_module_name):
    """
    Use SCT's verbosity values to set the logging level.

    The behavior of this function changes depending on how the caller module was invoked:
       1. If the module was invoked via the command line, the root (global) logger will be used.
       2. If the module was invoked via code (e.g. sct_script.main()), then only its local logger will be used.

    :verbosity: Verbosity value, typically from argparse (args.v). Values must adhere
    one of two schemes:
       - [0, 1, 2], which corresponds to [WARNING, INFO, DEBUG]. (Older scheme)
       - [False, True], which corresponds to [INFO, DEBUG].      (Newer scheme)
    :caller_module_name: The `__name__` of the caller module
    """
    dict_log_levels = {
        '0': 'WARNING', '1': 'INFO', '2': 'DEBUG',  # Older scheme
        'False': 'INFO', 'True': 'DEBUG',           # Newer scheme (See issue #2676)
    }

    if str(verbose) not in dict_log_levels.keys():
        raise ValueError(f"Invalid verbosity level '{verbose}' does not map to a log level, so cannot set.")

    log_level = dict_log_levels[str(verbose)]

    # Set logging level for this file
    logger.setLevel(getattr(logging, log_level))

    # Set logging level for the file that called this function
    caller_logger = logging.getLogger(caller_module_name)
    caller_logger.setLevel(getattr(logging, log_level))

    # Set CLI logging formatter globally
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(CustomLoggingFormatter())

    # Set the logging level globally, but only when scripts are directly invoked (e.g. from the command line)
    if caller_module_name == "__main__":
        root_logger.setLevel(getattr(logging, log_level))
        # matplotlib is particularly chatty, so keep it at the default level
        # see: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4086
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
    else:
        # NB: Nothing will be set if we're calling a CLI script in-code, i.e. <sct_cli_script>.main(). This keeps
        # the loglevel changes from leaking: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3341
        pass


# TODO: add test
def init_sct():
    """
    Initialize SCT for typical terminal usage, including logging initialization,
    as well as a status message with the SCT version and the command run.
    """

    def _format_wrap(old_format):
        def _format(record):
            res = old_format(record)
            if record.levelno >= logging.ERROR:
                res = f"\x1B[31;1m{res}\x1B[0m"
            elif record.levelno >= logging.WARNING:
                res = f"\x1B[33m{res}\x1B[0m"
            else:
                pass
            return res
        return _format

    # Initialize logging
    set_loglevel(verbose=False, caller_module_name=__name__)  # False => "INFO". For "DEBUG", must be called again with verbose=True.
    hdlr = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter()
    fmt.format = _format_wrap(fmt.format)
    hdlr.setFormatter(fmt)
    logging.root.addHandler(hdlr)

    # Display SCT version
    logger.info(f'\n--\nSpinal Cord Toolbox ({__version__})\n')

    # Display command (Only if called from CLI: check for .py in first arg)
    # Use next(iter()) to not fail on empty list (vs. sys.argv[0])
    if '.py' in next(iter(sys.argv), None):
        script = os.path.basename(sys.argv[0]).removesuffix(".py")
        arguments = ' '.join(sys.argv[1:])
        logger.info(f"{script} {arguments}\n"
                    f"--\n")

    # Enable timing
    begin_global_timer()


def send_email(addr_to, addr_from, subject, message='', passwd=None, filename=None, html=False, smtp_host=None, smtp_port=None, login=None):
    if smtp_host is None:
        smtp_host = os.environ.get("SCT_SMTP_SERVER", "smtp.gmail.com")
    if smtp_port is None:
        smtp_port = int(os.environ.get("SCT_SMTP_PORT", 587))
    if login is None:
        login = addr_from

    if html:
        msg = MIMEMultipart("alternative")
    else:
        msg = MIMEMultipart()

    msg['From'] = addr_from
    msg['To'] = addr_to
    msg['Subject'] = subject

    body = message
    if not isinstance(body, bytes):
        body = body.encode("utf-8")

    body_html = f"""
<html><pre style="font: monospace"><body>
{body}
</body></pre></html>""".encode()

    if html:
        msg.attach(MIMEText(body_html, 'html', "utf-8"))

    msg.attach(MIMEText(body, 'plain', "utf-8"))

    # filename = "NAME OF THE FILE WITH ITS EXTENSION"
    if filename:
        attachment = open(filename, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment', filename=filename)
        msg.attach(part)

    # send email
    server = smtplib.SMTP(smtp_host, smtp_port)
    server.starttls()
    if passwd is not None:
        server.login(login, passwd)
    text = msg.as_string()
    server.sendmail(addr_from, addr_to, text)
    server.quit()


def sct_progress_bar(*args, **kwargs):
    """Thin wrapper around `tqdm.tqdm` which checks `SCT_PROGRESS_BAR` muffling the progress
       bar if the user sets it to `no`, `off`, or `false` (case insensitive)."""
    do_pb = os.environ.get('SCT_PROGRESS_BAR', 'yes')
    if do_pb.lower() in ['off', 'no', 'false']:
        kwargs['disable'] = True
    # If not specified, then override the default 'ascii' value to account for Windows issues. For background, see:
    # https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2770#issuecomment-1399313735
    if 'ascii' not in kwargs:
        # On Windows, terminals often have limited unicode support, so it's recommended to set ascii=True
        if sys.platform.startswith('win32'):
            kwargs['ascii'] = True
        # On Linux/macOS, tqdm has built-in fallback login when ascii=None to account for various terminal capabilities
        else:
            kwargs['ascii'] = None

    return tqdm.tqdm(*args, **kwargs)


def _which_sct_binaries():
    """
    :return name of the sct binaries to use on this platform
    """

    if sys.platform.startswith("darwin"):
        return "binaries_osx"
    elif sys.platform.startswith("win32"):
        return "binaries_win"
    else:
        return "binaries_linux"


def list2cmdline(lst):
    return " ".join(shlex.quote(x) for x in lst)


def run_proc(cmd, verbose=1, raise_exception=True, cwd=None, env=None, is_sct_binary=False):
    if cwd is None:
        cwd = os.getcwd()

    if env is None:
        env = os.environ.copy()

    if is_sct_binary:
        # create an absolute path to the binary inside of SCT's bin directory
        name = cmd[0] if isinstance(cmd, list) else cmd.split(" ", 1)[0]
        path = os.path.join(__bin_dir__, name)
        if isinstance(cmd, list):
            cmd[0] = path
        elif isinstance(cmd, str):
            rem = cmd.split(" ", 1)[1:]
            cmd = path if len(rem) == 0 else f"{path} {rem[0]}"

        # also, for windows, add the bin directory to the path (to allow ANTs to access 'msvc-runtime' DLLs)
        # see also: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4655#issuecomment-2430178901
        if sys.platform.startswith("win32"):
            env['PATH'] = __bin_dir__ + os.pathsep + env['PATH']

    if isinstance(cmd, str):
        cmdline = cmd
    else:
        cmdline = list2cmdline(cmd)

    if verbose:
        printv(f"{cmdline} # in {cwd}", 1, 'code')

    shell = isinstance(cmd, str)

    process = subprocess.Popen(cmd, shell=shell, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    output_final = ''
    while True:
        # Watch out for deadlock!!!
        output = process.stdout.readline().decode("utf-8")
        if output == '' and process.poll() is not None:
            break
        if output:
            if verbose == 2:
                logger.debug(f"output => {output.strip()}")
            output_final = f'{output_final}{output.strip()}\n'

    status = process.returncode
    output = output_final.rstrip()

    if status != 0 and raise_exception:
        raise RuntimeError(output)

    return status, output


def printv(string, verbose=1, type='normal', file=None):
    """
    Print a message that is color-coded based on the message type, then exit if type='error'.

    Notes:
        - This function is intended for top-level, user-focused communication. So, it should be the sole
          form of message logging in CLI scripts (e.g. sct_propseg).
        - This function also acts as a way to throw an error without displaying a traceback. So, in CLI scripts,
           calling `printv(type='error')` should be prioritized over throwing Exceptions.
        - However, API modules *should not* use this function. They should instead use `logging` for
          info/warning messages, and Exceptions for throwing errors.
    """

    if file is None:
        # replicate the logic from print()
        # so that we can check file.isatty()
        file = sys.stdout

    if verbose:
        # The try/except is there in case file does not have isatty field (it did happen to me)
        try:
            # Print color only if the output is the terminal
            if file.isatty():
                color = LOGGING_COLOURS.get(type, ANSIColors16.ResetAll)
                print(stylize(string, color), file=file)
            else:
                print(string, file=file)
        except Exception:
            print(string)

    if type == 'error':
        sys.exit(1)


def sct_dir_local_path(*args):
    """Construct a directory path relative to __sct_dir__"""
    return os.path.join(__sct_dir__, *args)


def sct_test_path(*args):
    """Construct a directory path relative to the sct testing data. Consults
    the SCT_TESTING_DATA environment variable.
    If unset, use the default dataset location from sct_download_data."""

    test_path = os.environ.get('SCT_TESTING_DATA', '')
    if test_path:
        return os.path.join(test_path, 'sct_testing_data', *args)
    else:
        # NB: The default path written below is actually determined inside
        #     sct_download_data. But, trying to import the path from the script
        #     causes a circular dependency. So, we duplicate the path here.
        #     This could cause bugs if the default location ever changes.
        # TODO: Consider moving sct_test_path() inside testing/conftest.py,
        #       since it's a testing-specific function. This would mitigate the
        #       circular dependency, since we would no longer be importing
        #       from sct_download_data inside sys.py.
        return sct_dir_local_path('data', 'sct_testing_data', *args)


def check_exe(name):
    """
    Ensure that a program exists and can be executed

    :param name: str: name of program or path to program
    :return: boolean
    """
    _, filename = os.path.split(name)
    # Case 1: Check full filepath directly (which may point to a location not on the PATH)
    if os.path.isfile(name) and os.access(name, os.X_OK):
        return True
    # Case 2: Check filename only via the PATH
    elif shutil.which(filename) and os.access(shutil.which(filename), os.X_OK):
        return True
    else:
        return False


def _version_string():
    install_type, sct_commit, sct_branch, version_sct = _git_info()
    if install_type == "package":
        return version_sct
    return f"{install_type}-{sct_branch}-{sct_commit}"


def _git_info(commit_env='SCT_COMMIT', branch_env='SCT_BRANCH'):

    sct_commit = os.getenv(commit_env, "unknown")
    sct_branch = os.getenv(branch_env, "unknown")
    if check_exe("git") and os.path.isdir(os.path.join(__sct_dir__, ".git")):
        sct_commit = __get_commit() or sct_commit
        sct_branch = __get_branch() or sct_branch

    if sct_commit != 'unknown':
        install_type = 'git'
    else:
        install_type = 'package'

    with io.open(os.path.join(__sct_dir__, 'spinalcordtoolbox', 'version.txt'), 'r') as f:
        version_sct = f.read().rstrip()

    return install_type, sct_commit, sct_branch, version_sct


def __get_branch():
    """
    Fallback if for some reason the value vas no set by sct_launcher
    :return:
    """

    p = subprocess.Popen(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, cwd=__sct_dir__)
    output, _ = p.communicate()
    status = p.returncode

    if status == 0:
        return output.decode().strip()


def __get_commit(path_to_git_folder=None):
    """
    :return: git commit ID, with trailing '*' if modified
    """
    if path_to_git_folder is None:
        path_to_git_folder = __sct_dir__
    else:
        path_to_git_folder = os.path.abspath(os.path.expanduser(path_to_git_folder))

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


def __get_git_origin(path_to_git_folder=None):
    """
    :return: git origin url if available
    """
    if path_to_git_folder is None:
        path_to_git_folder = __sct_dir__
    else:
        path_to_git_folder = os.path.abspath(os.path.expanduser(path_to_git_folder))

    p = subprocess.Popen(["git", "remote", "get-url", "origin"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        origin = output.decode().strip()
    else:
        origin = "?!?"

    return origin


__sct_dir__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
__version__ = _version_string()
__data_dir__ = os.path.join(__sct_dir__, 'data')
__bin_dir__ = os.path.join(__sct_dir__, 'bin')
__deepseg_dir__ = os.path.join(__data_dir__, 'deepseg_models')
