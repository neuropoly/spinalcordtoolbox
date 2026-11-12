#!/usr/bin/env python
#
# Display bvecs
#
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
from typing import Sequence

import numpy as np

from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel, LazyLoader
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser

fetcher = LazyLoader("fetcher", globals(), 'dipy.data.fetcher')
pd = LazyLoader("pd", globals(), "pandas")
mpl_plt = LazyLoader("mpl_plt", globals(), "matplotlib.pyplot")
mpl_lines = LazyLoader("mpl_lines", globals(), "matplotlib.lines")
mpl_cm = LazyLoader("mpl_cm", globals(), "matplotlib.cm")

BZERO_THRESH = 0.0001  # b-zero threshold


def get_parser():
    parser = SCTArgumentParser(
        description='Display scatter plot of gradient directions from bvecs file. If you have multi-shell acquisition,'
                    'you can provide also bvals file to display individual shells in q-space.'
    )
    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-bvec',
        metavar=Metavar.file,
        help="Input bvecs file. Example: `sub-001_dwi.bvec`",
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-bval',
        metavar=Metavar.file,
        help="Input bval file (for multi-shell acquisition). Example: `sub-001_dwi.bval`",
    )

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


def plot_2dscatter(fig_handle=None, subplot=None, x=None, y=None, xlabel='X', ylabel='Y', bvals=None, colors=None):
    ax = fig_handle.add_subplot(subplot, aspect='equal')
    for i in range(0, len(x)):
        # if b=0, do not plot
        if not (abs(x[i]) < BZERO_THRESH and abs(x[i]) < BZERO_THRESH):
            ax.scatter(x[i], y[i], color=colors[bvals[i]], alpha=0.7)
    # mpl_plt.axis('equal')
    mpl_plt.xlabel(xlabel)
    mpl_plt.ylabel(ylabel)
    mpl_plt.xlim([-max(bvals), max(bvals)])
    mpl_plt.ylim([-max(bvals), max(bvals)])
    mpl_plt.grid()


def create_custom_legend(fig, shell_colors, bvals):
    """
    Create custom legend for whole figure containing number of directions (bvecs) for individual shells (bvals)
    TODO - if legend contains >4 rows (i.e., 12 unique bvals), it overlays charts - need to be fixed in the future
    :param fig: figure the legend will be created for
    :param shell_colors: dict with b-values and colors
    :param bvals: ndarray with all b-values
    :return:
    """

    # count number of bvals for individual shells
    bvals_per_shell = pd.Series(bvals[bvals > BZERO_THRESH]).value_counts()

    # Create single custom legend for whole figure with several subplots
    # Loop across legend elements
    lines = [mpl_lines.Line2D([0], [0], color=color, marker='o', markersize=8, alpha=0.7, linestyle='')
             for color in shell_colors.values()]
    labels = [f'b-values = {shell} (n = {bvals_per_shell[shell]})'
              for shell in shell_colors.keys()]

    # Insert legend below subplots, NB - this line has to be run after the mpl_plt.tight_layout() which is called before
    # function call
    legend = fig.legend(lines, labels, loc='lower left', bbox_to_anchor=(0.1, 0),
                        bbox_transform=mpl_plt.gcf().transFigure, ncol=3, fontsize=10)
    # Change box's frame color to black
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    # tight layout of whole figure and shift master title up
    fig.subplots_adjust(top=0.92, bottom=0.15)


# MAIN
# ==========================================================================================

def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    fname_bvecs = arguments.bvec
    fname_bvals = arguments.bval
    # Read bvals and bvecs files (if arguments.bval is not passed, bvals will be None)
    bvals, bvecs = fetcher.read_bvals_bvecs(fname_bvals, fname_bvecs)
    # if first dimension is not equal to 3 (x,y,z), transpose bvecs file
    if bvecs.shape[0] != 3:
        bvecs = bvecs.transpose()
    # if bvals file was not passed, create dummy unit bvals array (necessary fot scatter plots)
    if bvals is None:
        bvals = np.repeat(1, bvecs.shape[1])
    # multiply unit b-vectors by b-values
    x, y, z = bvecs[0] * bvals, bvecs[1] * bvals, bvecs[2] * bvals

    # Set different color for each shell (bval)
    shell_colors = {}
    # Create iterator with different colors from brg palette
    colors = iter(mpl_cm.nipy_spectral(np.linspace(0, 1, len(np.unique(bvals)))))
    for unique_bval in np.unique(bvals):
        # skip b=0
        if unique_bval < BZERO_THRESH:
            continue
        shell_colors[unique_bval] = tuple(next(colors))

    # Get total number of directions
    n_dir = len(x)

    # Get effective number of directions
    bvecs_eff = []
    n_b0 = 0
    for i in range(0, n_dir):
        add_direction = True
        # check if b=0
        if abs(x[i]) < BZERO_THRESH and abs(x[i]) < BZERO_THRESH and abs(x[i]) < BZERO_THRESH:
            n_b0 += 1
            add_direction = False
        else:
            # loop across bvecs_eff
            for j in range(0, len(bvecs_eff)):
                # if bvalue already present, then do not add to bvecs_eff
                if bvecs_eff[j] == [x[i], y[i], z[i]]:
                    add_direction = False
        if add_direction:
            bvecs_eff.append([x[i], y[i], z[i]])
    n_dir_eff = len(bvecs_eff)

    # Display scatter plot
    fig = mpl_plt.figure(facecolor='white', figsize=(9, 8))
    fig.suptitle(f'Number of b=0: {n_b0}, Number of b!=0: {n_dir - n_b0}, '
                 f'Number of effective directions (without duplicates): {n_dir_eff}')

    # Display three views
    plot_2dscatter(fig_handle=fig, subplot=221, x=x, y=y, xlabel='X', ylabel='Y', bvals=bvals, colors=shell_colors)
    plot_2dscatter(fig_handle=fig, subplot=222, x=x, y=z, xlabel='X', ylabel='Z', bvals=bvals, colors=shell_colors)
    plot_2dscatter(fig_handle=fig, subplot=223, x=y, y=z, xlabel='Y', ylabel='Z', bvals=bvals, colors=shell_colors)

    # 3D
    ax = fig.add_subplot(224, projection='3d')
    # ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
    for i in range(0, n_dir):
        # x, y, z = bvecs[0], bvecs[1], bvecs[2]
        # if b=0, do not plot
        if not (abs(x[i]) < BZERO_THRESH and abs(x[i]) < BZERO_THRESH and abs(x[i]) < BZERO_THRESH):
            ax.scatter(x[i], y[i], z[i], color=shell_colors[bvals[i]], alpha=0.7)
    ax.set_xlim3d(-max(bvals), max(bvals))
    ax.set_ylim3d(-max(bvals), max(bvals))
    ax.set_zlim3d(-max(bvals), max(bvals))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    mpl_plt.title('3D view (use mouse to rotate)')
    mpl_plt.axis('on')
    # mpl_plt.draw()

    mpl_plt.tight_layout()
    # add legend with b-values if bvals file was passed
    if arguments.bval is not None:
        create_custom_legend(fig, shell_colors, bvals)

    # Save image
    printv("Saving figure: bvecs.png\n")
    mpl_plt.savefig('bvecs.png')
    # mpl_plt.show()


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
