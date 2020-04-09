#!/usr/bin/env python
# -*- coding: utf-8
"""
This command-line tool is the interface for the deepseg API that performs segmentation using deep learning from the
ivadomed package.
"""

# TODO: implement feature with a new flag (e.g. path-to-model) that will give the possibility to point to a model
#  folder, in case a test model is not on OSF and is not listed in MODELS.

from __future__ import absolute_import

import sys
import os
import argparse

import spinalcordtoolbox as sct
import spinalcordtoolbox.deepseg.core
import spinalcordtoolbox.deepseg.models
from spinalcordtoolbox.utils import Metavar, SmartFormatter

from sct_utils import init_sct, printv


def get_parser():

    param_default = sct.deepseg.core.ParamDeepseg()

    parser = argparse.ArgumentParser(
        description="Segmentation using deep learning.",
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        required=True,
        help="Image to segment.",
        metavar=Metavar.file)

    seg = parser.add_argument_group('\nSEGMENTATION:')
    seg.add_argument(
        "-m",
        help="Model to use, from the list of official SCT models downloaded from the Internet.",
        choices=list(sct.deepseg.models.MODELS.keys()))
    seg.add_argument(
        "-mpath",
        help="Path to model, in case you would like to use a custom model. The model folder should follow the "
             "conventions listed in: URL.",
        metavar=Metavar.folder)

    misc = parser.add_argument_group('\nMISC')
    misc.add_argument(
        "-o",
        help="Output segmentation suffix. In case of multi-class segmentation, class-specific suffixes will be added.",
        metavar=str,
        default=param_default.output_suffix)
    misc.add_argument(
        "-v",
        type=int,
        help="Verbose: 0 = no verbosity, 1 = verbose.",
        choices=(0, 1),
        default=1)
    misc.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")

    return parser


def main():
    param = sct.deepseg.core.ParamDeepseg()

    parser = get_parser()
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    # TODO: instead of assigning each args param, we could pass args while instanciating ParamDeepseg(args), and the
    #  class would deal with assigning arguments to each field.
    if 'o' in args:
        param.output_suffix = args.o

    # Get model path
    if args.m:
        name_model = args.m
        sct.deepseg.models.is_model(name_model)
        if not spinalcordtoolbox.deepseg.models.is_installed(name_model):
            if not spinalcordtoolbox.deepseg.models.install(name_model):
                printv("Model needs to be installed.", 1, 'error')
                exit(RuntimeError)
    elif args.mpath:
        path_model = args.mpath

    sct.deepseg.core.segment_nifti(args.i, path_model)


if __name__ == '__main__':
    init_sct()
    main()
