#!/usr/bin/env python
#
# Function to compute the Ernst Angle
#
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
from typing import Sequence
import textwrap

from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel


class ErnstAngle:
    # The constructor
    def __init__(self, t1, tr=None, fname_output=None):
        self.t1 = t1
        self.tr = tr
        self.fname_output = fname_output

    # compute and return the Ernst Angle
    def getErnstAngle(self, tr):
        from numpy import arccos
        from numpy import exp
        from math import pi
        angle_rad = arccos(exp(-tr / self.t1))
        angle_deg = angle_rad * 180 / pi
        return angle_deg

    # draw the graph
    def draw(self, tr_min, tr_max):
        import matplotlib.pyplot as plt
        from numpy import arange
        step = (tr_max - tr_min) / 50
        tr_range = arange(tr_min, tr_max, step)
        theta_E = self.getErnstAngle(tr_range)

        printv("\nDrawing", type='info')
        plt.plot(tr_range, theta_E, linewidth=1.0)
        plt.xlabel(r"TR (in $ms$)")
        plt.ylabel(r"$\Theta_E$ (in degree)")
        plt.ylim(min(theta_E), max(theta_E) + 2)
        plt.title(f"Ernst Angle with T1={self.t1}ms")
        plt.grid(True)

        if self.tr is not None:
            plt.plot(self.tr, self.getErnstAngle(self.tr), 'ro')
        if self.fname_output is not None:
            printv("\nSaving figure", type='info')
            plt.savefig(self.fname_output, format='png')
        plt.show()


def get_parser():
    parser = SCTArgumentParser(
        description=textwrap.dedent("""
            Function to compute the Ernst Angle.

            For examples of T1 values in the brain, see Wansapura et al. NMR relaxation times in the human brain at 3.0 tesla. Journal of magnetic resonance imaging : JMRI (1999) vol. 9 (4) pp. 531-8. T1 in WM: 832msT1 in GM: 1331ms
        """),  # noqa: E501 (line too long)
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        "-tr",
        type=float,
        help='Value of TR (in ms) to get the Ernst Angle. Example: `2000`',
        metavar=Metavar.float,
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-t1",
        type=float,
        help='T1 value (in ms).',
        metavar=Metavar.float,
        default=832.0)
    optional.add_argument(
        "-b",
        type=float,
        nargs=2,
        metavar=Metavar.float,
        help='Min/Max range of TR (in ms) separated with space. Only use with `-v 2`.',
        default=[500, 3500])
    optional.add_argument(
        "-o",
        help="Name of the output file containing Ernst angle result.",
        metavar=Metavar.str,
        default="ernst_angle.txt")
    optional.add_argument(
        "-ofig",
        help="Name of the output graph. Only use with `-v 2`.",
        metavar=Metavar.str,
        default="ernst_angle.png")

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


# main
# =======================================================================================================================
def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    # Initialization
    input_t1 = arguments.t1
    input_fname_output = None
    input_tr = None
    fname_output_file = arguments.o
    if arguments.ofig is not None:
        input_fname_output = arguments.ofig
    if arguments.tr is not None:
        input_tr = arguments.tr

    graph = ErnstAngle(input_t1, tr=input_tr, fname_output=input_fname_output)
    if input_tr is not None:
        printv(f"\nValue of the Ernst Angle with T1={graph.t1}ms and TR={input_tr}ms :", verbose=verbose, type='info')
        printv(str(graph.getErnstAngle(input_tr)))
        # save text file
        f = open(fname_output_file, 'w')
        f.write(str(graph.getErnstAngle(input_tr)))
        f.close()

    if verbose == 2:
        # The upper and lower bounds of the TR values to use for plotting the Ernst angle curve
        input_tr_min = arguments.b[0]
        input_tr_max = arguments.b[1]
        # If the input TR value is outside the default plotting range, then widen the range
        if input_tr > input_tr_max:
            input_tr_max = input_tr + 500
        elif input_tr < input_tr_min:
            input_tr_min = 0
        graph.draw(input_tr_min, input_tr_max)


# =======================================================================================================================
# Start program
# =======================================================================================================================
if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
