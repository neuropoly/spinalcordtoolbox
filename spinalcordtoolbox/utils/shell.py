"""
Convenience/shell related utilities

Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import os
import sys
import re
import shutil
import logging
import argparse
import itertools

from enum import Enum
from functools import cached_property, partial

from .sys import check_exe, printv, ANSIColors16
from .fs import relpath_or_abspath
from .profiling import TimeProfilingAction, MemoryTracingAction

logger = logging.getLogger(__name__)


def display_open(file, message="Done! To view results"):
    """Print the syntax to open a file based on the platform."""
    cmd_open = None
    if sys.platform.startswith('linux'):
        # If user runs SCT within the official Docker distribution, or in WSL, then the command xdg-open will not be
        # working, therefore we prefer to instruct the user to manually open the file.
        # Source for WSL environment variables: https://stackoverflow.com/a/61036356
        if "DOCKER" not in os.environ and "IS_WSL" not in os.environ and "WSL_DISTRO_NAME" not in os.environ:
            cmd_open = 'xdg-open'
    elif sys.platform.startswith('darwin'):
        cmd_open = 'open'
    elif sys.platform.startswith('win32'):
        cmd_open = 'start'

    if cmd_open:
        printv(f'\n{message}, type:')
        printv(f"{cmd_open} {file}\n", type='info')
    else:
        printv(f'\n{message}, open the following file:')
        printv(f"{file}\n", type='info')


SUPPORTED_VIEWERS = ['fsleyes', 'fslview_deprecated', 'fslview', 'itk-snap', 'itksnap']
# - The 'fsleyes' colormaps are used for 'fsleyes'.
# - The 'fslview' colormaps are used for 'fslview' and 'fslview_deprecated'.
# - For 'itksnap', there are no colormaps. Instead, color behavior is dictated using CLI options '-g', '-o', and '-s'.
#   How imtypes are mapped to CLI options is a bit convoluted, but tl;dr: only 2 image types (gray, seg) are supported.
#   Surprisingly, softseg images can only be properly displayed as grayscale images, hence the use of 'gray' for them.
IMTYPES_COLORMAP = {
    'anat':        {'fsleyes': 'greyscale',             'fslview': 'Greyscale',             'itksnap': 'gray'},
    'seg-1':       {'fsleyes': 'red',                   'fslview': 'Red',                   'itksnap': 'seg'},
    'seg-2':       {'fsleyes': 'blue-lightblue',        'fslview': 'Blue-Lightblue',        'itksnap': 'seg'},
    'seg-3':       {'fsleyes': 'green',                 'fslview': 'Green',                 'itksnap': 'seg'},
    'seg-4':       {'fsleyes': 'yellow',                'fslview': 'Yellow',                'itksnap': 'seg'},
    'seg-labeled': {'fsleyes': 'subcortical',           'fslview': 'MGH-Subcortical',       'itksnap': 'seg'},
    'softseg-1':   {'fsleyes': 'YlOrRd',                'fslview': 'YlOrRd',                'itksnap': 'gray'},
    'softseg-2':   {'fsleyes': 'blue-lightblue',        'fslview': 'Blue-Lightblue',        'itksnap': 'gray'},
    'softseg-3':   {'fsleyes': 'brain_colours_2winter', 'fslview': 'Brain_Colours_2winter', 'itksnap': 'gray'},
    'softseg-4':   {'fsleyes': 'brain_colours_3warm',   'fslview': 'Brain_Colours_3warm',   'itksnap': 'gray'},
    'cropbox':     {'fsleyes': None,                    'fslview': 'Red',                   'itksnap': 'seg'},
}


def display_viewer_syntax(files, verbose, im_types=[], minmax=[], opacities=[], mode=''):
    """
    Print the syntax to open a viewer and display images for QC. To use default values, enter empty string: ''
    Parameters
    ----------
    files [list:string]: list of NIFTI file names
    im_types [list:string]: list of image type associated with each file. Available types: see IMTYPE_COLORMAPS
    minmax [list:string]: list of min,max brightness scale associated with each file. Separate with comma.
    opacities [list:string]: list of opacity associated with each file. Between 0 and 1.

    Returns
    -------
    cmd_strings [dict:string]: pairs of viewers and their corresponding syntax strings (that were printed).

    Example
    -------
    display_viewer_syntax([file1, file2, file3])
    display_viewer_syntax([file1, file2], im_types=['anat', 'softseg'], minmax=['', '0,1'], opacities=['', '0.7'])
    """
    # Try to convert the path to one that is relative to the CWD; if not possible, use the abspath instead.
    files = [str(relpath_or_abspath(filepath, parent_path=os.getcwd())) for filepath in files]

    available_viewers = [viewer for viewer in SUPPORTED_VIEWERS if check_exe(viewer)]

    if verbose:
        if len(available_viewers) == 0:
            return
        elif len(available_viewers) == 1:
            printv('\nDone! To view results, type:')
        elif len(available_viewers) >= 2:
            printv('\nDone! To view results, run one of the following commands (depending on your preferred viewer):')

    cmd_strings = {}
    for viewer in available_viewers:
        if viewer in ['fslview', 'fslview_deprecated']:
            cmd = _construct_fslview_syntax(viewer, files, im_types, minmax, opacities, mode)
        elif viewer in ['fsleyes']:
            cmd = _construct_fsleyes_syntax(viewer, files, im_types, minmax, opacities)
        elif viewer in ['itksnap', 'itk-snap']:
            cmd = _construct_itksnap_syntax(viewer, files, im_types)
        else:
            cmd = ""  # This should never be reached, because SUPPORTED_VIEWERS should match the 'if' cases exactly
        cmd_strings[viewer] = cmd

        if verbose:
            printv(f"{cmd}\n", verbose=1, type='info')

    return cmd_strings


def _construct_fslview_syntax(viewer, files, im_types, minmax, opacities, mode):
    cmd = viewer
    n = itertools.cycle([1, 2, 3, 4])  # There are 4 colormaps for segs
    # add mode (only supported by fslview for the moment)
    if mode:
        cmd = f'{cmd} -m {mode}'
    for i in range(len(files)):
        cmd = f'{cmd} {files[i]}'
        if im_types:
            if im_types[i]:
                key = im_types[i]
                # use different colormaps for each subsequent seg
                if key in ("seg", "softseg"):
                    key = f"{key}-{next(n)}"  # There are 4 colormaps for segs, so take modulo
                cmd = f'{cmd} -l {IMTYPES_COLORMAP[key]["fslview"]}'
        if minmax:
            if minmax[i]:
                cmd = f'{cmd} -b {minmax[i]}'  # a,b
        if opacities:
            if opacities[i]:
                cmd = f'{cmd} -t {opacities[i]}'
    cmd = f'{cmd} &'

    return cmd


def _construct_fsleyes_syntax(viewer, files, im_types, minmax, opacities):
    cmd = viewer
    n = itertools.cycle([1, 2, 3, 4])  # There are 4 colormaps for segs
    for i in range(len(files)):
        cmd = f'{cmd} {files[i]}'
        if im_types:
            if im_types[i]:
                key = im_types[i]
                # use different colormaps for each subsequent seg
                if key in ("seg", "softseg"):
                    key = f"{key}-{next(n)}"
                if key == 'cropbox':
                    cmd = f'{cmd} -ot mask -mc 1 1 0 -o -w 3 -d'
                else:
                    cmd = f'{cmd} -cm {IMTYPES_COLORMAP[key]["fsleyes"]}'
        if minmax:
            if minmax[i]:
                cmd = f'{cmd} -dr {" ".join(minmax[i].split(","))}'  # a b
        if opacities:
            if opacities[i]:
                cmd = f'{cmd} -a {float(opacities[i]) * 100}'  # in percentage
    cmd = f'{cmd} &'

    return cmd


def _construct_itksnap_syntax(viewer, files, im_types):
    # Split the image files into two categories: grayscale images (used for `-g`/`-o`) and seg images (used for `-s`)
    gray_images = []
    seg_images = []
    for i in range(len(files)):
        if not im_types:
            gray_images.append(files[i])
        else:
            # itksnap only has 1 colormap per seg, so always use cm "1" for segs
            key = im_types[i]
            if key in ("seg", "softseg"):
                key = f"{key}-1"
            colormap = IMTYPES_COLORMAP[key]['itksnap']
            if colormap == 'gray':
                gray_images.append(files[i])
            elif colormap == 'seg':
                seg_images.append(files[i])
            else:
                raise ValueError(f"ITKSnap does not support colormap '{colormap}'")

    # Construct ITKSnap command based on image types
    # 1. '-g' is the "main image" and it's mandatory: i.e. You can't just display a single seg image ('-s') by itself,
    #    or else itksnap will throw this error:
    #        "Error: Option -s must be used together with option -g"
    # NB: `-g` really should be a grayscale image, but if there are no gray images, we fall back to using a seg image.
    main_image = (gray_images or seg_images).pop(0)
    cmd = f"{viewer} -g {main_image}"
    # 2. '-o' is used for any remaining grayscale images not used as the main image (`-g`)
    if gray_images:
        cmd = f"{cmd} -o {' '.join(gray_images)}"
    # 3. '-s' is used for any images with 1 value (binary segmentations) or >1 values (labeled segmentations).
    #    NB: There can only be one segmentation per ITKSnap command. (ITKSnap can't toggle segmentations like FSLeyes.)
    #        To get around this, we duplicate the command so that there is one command per segmentation.
    if seg_images:
        cmd = "\n".join([f"{cmd} -s {seg_image}" for seg_image in seg_images])

    return cmd


class SCTArgumentParser(argparse.ArgumentParser):
    """
    Parser that centralizes initialization steps common across all SCT scripts.
    """
    def __init__(self, **kwargs):
        super(SCTArgumentParser, self).__init__(
            formatter_class=SmartFormatter,
            # Disable "add_help", because it won't properly add '-h' to our custom argument groups
            # (We use custom argument groups because of https://stackoverflow.com/a/24181138)
            add_help=False,
            # All arguments must be passed as keyword arguments (to encourage clarity at the caller level)
            **kwargs
        )

        # Update "usage:" message to match how SCT scripts are actually called (no '.py')
        self.prog = self.prog.removesuffix(".py")

    # == OVERRIDES == #
    def error(self, message):
        """
        Overridden parent method. Ensures that help is printed when called with invalid args.

        See https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3137.
        """
        # Source: https://stackoverflow.com/a/4042861
        self.print_help(sys.stderr)
        message_formatted = (f'{ANSIColors16.Bold}{ANSIColors16.LightRed}'
                             f'\n{self.prog}: error: {message}\n\n'
                             f'{ANSIColors16.ResetAll}')
        self.exit(2, message_formatted)

    # == STANDARD ARGUMENT GROUPS == #
    @cached_property
    def mandatory_arggroup(self):
        """
        The "MANDATORY" argument group.

        This should include arguments which are required for the CLI task to run.
        """
        mandatory = self.add_argument_group("MANDATORY ARGUMENTS")

        # Force arguments in this group to be required via in-place function override
        _f = partial(mandatory.add_argument, required=True)
        mandatory.add_argument = _f

        return mandatory

    @cached_property
    def optional_arggroup(self):
        """
        The "OPTIONAL" argument group.
        This should include arguments relevant to the task (such as slice in the MRI), but not strictly required
        by the user (i.e. they have a default value, or only enable optional functionality)
        """
        optional = self.add_argument_group("OPTIONAL ARGUMENTS")
        return optional

    @cached_property
    def misc_arggroup(self):
        """
        The "MISC" argument group; should contain arguments which do not directly affect the CLI task, but will change
        how the task is presented or logged to the user. The -h, -v, and -r arguments are placed here by default.
        """
        misc = self.add_argument_group("MISC ARGUMENTS")
        return misc

    # == COMMON ARGUMENT ADDITIONS == #
    def add_common_args(self, arg_group=None):
        """
        Adds two universally used arguments to the provided argument group:
            -h: The help flag. If present, displays help and terminates the program
            -v: The verbosity flag. If present, increases the verbosity of the program's output

        If no argument group is provided, the arguments are placed into the "MISC" argument group.
        """
        # If the user didn't specify an argument group, use the "MISC" group
        if not arg_group:
            arg_group = self.misc_arggroup

        # Add the help flag
        arg_group.add_argument(
            "-h", "--help",
            action="help",
            help="Show this help message and exit."
        )
        # Add the verbosity flag # TODO update/deprecate to address Issue #2676
        arg_group.add_argument(
            '-v',
            metavar=Metavar.int,
            type=int,
            choices=[0, 1, 2],
            default=1,
            # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
            help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode."
        )

        # Add a flag which allows time-based profiling the profiler
        arg_group.add_argument(
            '-profile-time',
            nargs='?',
            metavar=Metavar.file,
            action=TimeProfilingAction,
            help=argparse.SUPPRESS
            # TODO: Find a way to only show these "developer" util helpers in certain contexts
            # help="Enables time-based profiling of the program, dumping the results to the specified file.\n"
            #      "\n"
            #      "If no file is specified, human-readable results are placed into a 'time_profiling_results.txt' "
            #      f"document in the current directory ('{os.getcwd()}'). If the specified file is a `.prof` file, the "
            #      "file will instead be in binary format, ready for use with common post-profiler utilities (such as "
            #      "`snakeviz`)."
        )
        arg_group.add_argument(
            '-trace-memory',
            nargs='?',
            metavar=Metavar.folder,
            action=MemoryTracingAction,
            help=argparse.SUPPRESS
            # TODO: Find a way to only show these "developer" util helpers in certain contexts
            # help="Enables memory tracing of the program.\n"
            #      "\n"
            #      "When active, a measure of the peak memory (in KiB) will be output to the file `peak_memory.txt`. "
            #      "Optionally, developers can also modify the SCT code to add additional `snapshot_memory()` calls. "
            #      "These calls will 'snapshot' the memory usage at that moment, saving the memory trace at that point "
            #      "into a second file (`memory_snapshots.txt`).\n"
            #      "\n"
            #      f"By default, both outputs will be placed in the current directory ('{os.getcwd()}'). Optionally, you "
            #      "may provide an alternative directory (`-trace-memory <dir_name>`), in which case all files will "
            #      "be placed in that directory instead.\n"
            #      "Note that this WILL incur an overhead to runtime, so it is generally advised that you do not run "
            #      "this in conjunction with the time profiler or in time-sensitive contexts."
        )

        # Return the arg_group to allow for chained operations
        return arg_group

    def add_tempfile_args(self, arg_group=None):
        """
        Adds a single argument:
            -r: The remove_temp flag. If present, denotes that temporary files should be removed after the program ends.

        If no argument group is provided, the arguments are placed into the "MISC" argument group
        """
        # If the user didn't specify an argument group, use the "MISC" group
        if not arg_group:
            arg_group = self.misc_arggroup

        # TODO: Change this to a flag, rather than a number
        arg_group.add_argument(
            "-r",
            type=int,
            help="Remove temporary files.",
            default=1,
            choices=(0, 1)
        )


class ActionCreateFolder(argparse.Action):
    """
    Custom action: creates a new folder if it does not exist. If the folder
    already exists, do nothing.

    The action will strip off trailing slashes from the folder's name.
    Source: archive.is/pgUKF (`argparse_actions` -> project now 404s)
    """
    @staticmethod
    def create_folder(folder_name):
        """
        Create a new directory if not exist. The action might throw
        OSError, along with other kinds of exception
        """
        folder_name = os.path.normpath(folder_name)
        os.makedirs(folder_name, exist_ok=True)
        return folder_name

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            folders = list(map(self.create_folder, values))
        else:
            folders = self.create_folder(values)

        # Add the attribute
        setattr(namespace, self.dest, folders)


def list_type(delimiter, subtype, length=None):
    """
        Factory function that returns a list parsing function, which can be
        used with argparse's `type` option. This allows for more complex type
        parsing, and preserves the behavior of the old msct_parser.
    """
    def list_typecast_func(string):
        vals = [subtype(v) for v in string.split(delimiter)]
        if length is not None and len(vals) != length:
            raise ValueError(f"expected {length} values, got {len(vals)} values")
        return vals

    return list_typecast_func


def positive_int_type(input_value):
    """
        Type function which can be used with argparse's `type` option. This
        allows for more complex type parsing, enforcing the requirement of an
        input integer to be positive.
    """
    try:
        int_value = int(input_value)         # cast first to check for type errors
        if int_value != float(input_value):  # allow "1" and "1.0" but not "1.5"
            raise ValueError("non-integer number encountered")
        if int_value < 0:
            raise ValueError("negative integer encountered")
        if int_value == 0:
            raise ValueError("zero encountered")
    except ValueError as e:
        raise ValueError(f"expected positive integer, got {input_value}") from e
    return int_value


class Metavar(str, Enum):
    """
    This class is used to display intuitive input types via the metavar field of argparse
    """
    file = "<file>"
    str = "<str>"
    folder = "<folder>"
    int = "<int>"
    list = "<list>"
    float = "<float>"

    def __str__(self):
        return self.value


class SmartFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Custom formatter that inherits from HelpFormatter to apply the same
    tweaks across all of SCT's scripts."""

    # Pattern for an inline list of values, e.g. "(choices: item1, item2, item3) (default: item1)".
    # When such a line wraps, the wrapped lines are aligned with the first item, rather than with the label.
    INLINE_LIST_PATTERN = re.compile(
        r"""
        ^\s*(?P<label>\(?\w+:\s)  # "Word: ", with an optional opening bracket, e.g. "(choices: "
        [^\s,()]+                 # A single item: no spaces, commas, or brackets
        (?:,\s[^\s,()]+){2,}      # At least 2 more items, each preceded by ", "
        \)?\.?                    # An optional closing bracket, then an optional period
        (?:\s\(.*\))?$            # An optional trailing parenthetical, e.g. "(default: ...)"
        """,
        re.VERBOSE,
    )

    def __init__(self, *args, **kw):
        self._add_defaults = None
        super(SmartFormatter, self).__init__(*args, **kw)
        # Tweak: Update argparse's '_width' to match Terminal width
        try:
            self._width = shutil.get_terminal_size()[0]
        except (KeyError, ValueError):
            logger.warning('Not able to fetch Terminal width. Using default: %s', self._width)

    def _get_help_string(self, action):
        """
        Forced the "default" to not be printed if it's a meaningless default (i.e. an empty string)
        """
        # Return immediately if no default was specified, or if this is a suppressed "help" flag
        if action.default in [None, "", [], (), {}, '==SUPPRESS==']:
            return action.help

        # Otherwise, format the string as-usual
        return super()._get_help_string(action)

    def _fill_text(self, text, width, indent):
        """Overrides the default _fill_text method. It takes a single string
        (`text`) and rebuilds it so that each line wraps at the specified
        `width`, while also preserving newlines.

        This method is what gets called for the parser's `description` field.
        """
        # NB: We use our overridden split_lines method to apply indentation to the help description
        paragraphs = self._split_lines(text, width)
        return '\n'.join(paragraphs)

    def _split_lines(self, text, width):
        """Overrides the default _split_lines method. It takes a single string
        (`text`) and rebuilds it so that each line wraps at the specified
        `width`, while also preserving newlines, as well as any offsets within
        the text (e.g. indented lists).

        This method is what gets called for each argument's `help` field.
        """
        import textwrap
        # NB: text.splitlines() is what's used by argparse.RawTextHelpFormatter
        #     to preserve newline characters (`\n`) in text.
        lines = text.splitlines()
        # NB: The remaining code is fully custom
        while lines[0] == '':  # Discard empty start lines
            lines = lines[1:]
        offsets = [re.match("^[ \t]*", line).group(0) for line in lines]
        wrapped = []
        for i, li in enumerate(lines):
            li = li.rstrip()  # strip trailing whitespace
            if len(li) > 0:
                # Check for ANSI graphics control sequences, and increase width to compensate
                width_adjusted = width + len("".join(re.findall(r"\x1b\[[0-9;]+m", li)))
                # Split the line into two parts: the first line, and wrapped lines
                init_wrap = textwrap.fill(li, width_adjusted).splitlines()
                first = init_wrap[0]
                rest = "\n".join(init_wrap[1:])
                # Add an offset to the wrapped lines so that they're indented the same as the first line
                o = offsets[i]
                if re.match(r"^\s+[-*]\s\w.*$", li):  # List matching: " - Text" or " * Text"
                    o += "  "  # If the line is a list item, add extra indentation to the wrapped lines (#2889)
                # Inline list matching: "(choices: item1, item2, item3)" (see INLINE_LIST_PATTERN)
                elif inline_list := self.INLINE_LIST_PATTERN.match(li):
                    o += " " * len(inline_list.group("label"))  # Align wrapped lines with the first item
                ol = len(o)
                rest_wrap = textwrap.fill(rest, width_adjusted - ol).splitlines()
                offset_lines = [o + wl for wl in rest_wrap]
                # Merge the first line and the wrapped lines
                wrapped = wrapped + [first] + offset_lines
            else:
                wrapped = wrapped + [li]
        return wrapped


def parse_num_list(str_num):
    """
    Parse numbers in string based on delimiter ',' or ':'

    .. note::
        Examples:
        '' -> []
        '1,2,3' -> [1, 2, 3]
        '1:3,4' -> [1, 2, 3, 4]
        '1,1:4' -> [1, 2, 3, 4]

    :param str_num: str
    :return: list of ints
    """
    list_num = list()

    if not str_num:
        return list_num

    elements = str_num.split(",")
    for element in elements:
        m = re.match(r"^\d+$", element)
        if m is not None:
            val = int(element)
            if val not in list_num:
                list_num.append(val)
            continue
        m = re.match(r"^(?P<first>\d+):(?P<last>\d+)$", element)
        if m is not None:
            a = int(m.group("first"))
            b = int(m.group("last"))
            list_num.extend(x for x in range(a, b + 1) if x not in list_num)
            continue
        raise ValueError(f"unexpected group element {element} group spec {str_num}")

    return list_num


def get_interpolation(program, interp):
    """
    Get syntax on interpolation field depending on program. Supported programs: ants, flirt, WarpImageMultiTransform
    :param program:
    :param interp:
    :return:
    """
    # TODO: check if field and program exists
    interp_program = ''
    # FLIRT
    if program == 'flirt':
        if interp == 'nn':
            interp_program = ' -interp nearestneighbour'
        elif interp == 'linear':
            interp_program = ' -interp trilinear'
        elif interp == 'spline':
            interp_program = ' -interp spline'
    # ANTs
    elif program == 'ants' or program == 'ants_affine' or program == 'isct_antsApplyTransforms' \
            or program == 'isct_antsSliceRegularizedRegistration' or program == 'isct_antsRegistration':
        if interp == 'nn':
            interp_program = ' -n NearestNeighbor'
        elif interp == 'linear':
            interp_program = ' -n Linear'
        elif interp == 'spline':
            interp_program = ' -n BSpline[3]'
    # check if not assigned
    if interp_program == '':
        logger.warning('%s: interp_program not assigned. Using linear for ants_affine.', os.path.basename(__file__))
        interp_program = ' -n Linear'
    # return
    return interp_program.strip().split()


def parse_num_list_inv(list_int):
    """
    Take a list of numbers and output a string that reduce this list based on delimiter ';' or ':'

    .. note::
        Note: we use ; instead of , for compatibility with csv format.
        Examples:
        [] -> ''
        [1, 2, 3] --> '1:3'
        [1, 2, 3, 5] -> '1:3;5'

    :param list_int: list: list of ints
    :return: str_num: string
    """
    # deal with empty list
    if not list_int or list_int is None:
        return ''
    # Sort list in increasing number
    list_int = sorted(list_int)
    # initialize string
    str_num = str(list_int[0])
    colon_is_present = False
    # Loop across list elements and build string iteratively
    for i in range(1, len(list_int)):
        # if previous element is the previous integer: I(i-1) = I(i)-1
        if list_int[i] == list_int[i - 1] + 1:
            # if ":" already there, update the last chars (based on the number of digits)
            if colon_is_present:
                str_num = str_num[:-len(str(list_int[i - 1]))] + str(list_int[i])
            # if not, add it along with the new int value
            else:
                str_num = f'{str_num}:{list_int[i]}'
                colon_is_present = True
        # I(i-1) != I(i)-1
        else:
            str_num = f'{str_num};{list_int[i]}'
            colon_is_present = False

    return str_num
