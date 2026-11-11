"""
Load PyTorch models that were pickled by the `ivadomed` package.

Some ivadomed models (e.g. `mice_uqueensland_sc`) are distributed as `.pt` files containing whole pickled model
objects, which refer to classes by their original module path (e.g. `ivadomed.models.Unet`). This module provides
a `pickle_module` for `torch.load` that maps those paths onto SCT's vendored copy (`spinalcordtoolbox.ivadomed`),
so that the models can be loaded without the `ivadomed` package being installed.

Copyright (c) 2026 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import pickle
import sys
from pickle import *  # noqa: F401,F403  (`torch.load` expects the full `pickle` module interface)

import torch


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "ivadomed" or module.startswith("ivadomed."):
            module = f"spinalcordtoolbox.{module}"
        return super().find_class(module, name)


def load(file, **kwargs):
    return Unpickler(file, **kwargs).load()


def torch_load(f, map_location=None):
    """`torch.load` for ivadomed `.pt` files, which contain whole pickled models rather than just weights."""
    return torch.load(f, map_location=map_location, pickle_module=sys.modules[__name__], weights_only=False)
