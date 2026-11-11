# This file is vendored from ivadomed 2.9.10 (https://github.com/ivadomed/ivadomed),
# originally `ivadomed/loader/slice_filter.py`, licensed under the MIT License.
# See the LICENSE.md file in this directory for the full license text.
#
# Modifications for SCT:
#   - Imports point to `spinalcordtoolbox.ivadomed` instead of `ivadomed`.
#   - `torch.load` is replaced by `pickle_compat.torch_load`, so that ivadomed-pickled `.pt` models load with the vendored
#     classes.

import torch
import numpy as np
from spinalcordtoolbox.ivadomed import utils as imed_utils
from spinalcordtoolbox.ivadomed.pickle_compat import torch_load


class SliceFilter(object):
    """Filter 2D slices from dataset.

    If a slice does not meet certain conditions, it is discarded from the dataset.

    Args:
        filter_empty_mask (bool): If True, slices where all voxel labels are zeros are discarded.
        filter_absent_class (bool): If True, slices where all voxel labels are zero for one or more classes are
            discarded.
        filter_empty_input (bool): If True, slices where all voxel intensities are zeros are discarded.
        filter_classification (bool): If True, slices where all images fail a custom classifier filter are discarded.
        device (torch.device): Indicates the CPU or GPU ID.
        cuda_available (bool): If True, CUDA is available.

    Attributes:
        filter_empty_mask (bool): If True, slices where all voxel labels are zeros are discarded. Default: False.
        filter_absent_class (bool): If True, slices where all voxel labels are zero for one or more classes are
            discarded. Default: False.
        filter_empty_input (bool): If True, slices where all voxel intensities are zeros are discarded. Default: True.
        filter_classification (bool): If True, slices where all images fail a custom classifier filter are discarded.
            Default: False.
        device (torch.device): Indicates the CPU or GPU ID.
        cuda_available (bool): If True, CUDA is available.

    """

    def __init__(self, filter_empty_mask: bool = False,
                 filter_absent_class: bool = False,
                 filter_empty_input: bool = True,
                 filter_classification: bool = False,
                 classifier_path: any = None,
                 device: torch.device = None,
                 cuda_available: bool = None):
        self.filter_empty_mask = filter_empty_mask
        self.filter_absent_class = filter_absent_class
        self.filter_empty_input = filter_empty_input
        self.filter_classification = filter_classification
        self.device = device
        self.cuda_available = cuda_available

        if self.filter_classification:
            if cuda_available:
                self.classifier = torch_load(classifier_path, map_location=device)
            else:
                self.classifier = torch_load(classifier_path, map_location='cpu')

    def __call__(self, sample: dict) -> bool:
        """Extract input_data and gt_data lists from sample dict and discard them if they don't match certain
        conditions.

        """
        input_data, gt_data = sample['input'], sample['gt']

        if self.filter_empty_mask:
            # Discard slices that do not have ANY ground truth (i.e. all masks are empty)
            if not np.any(gt_data):
                return False
        if self.filter_absent_class:
            # Discard slices that have absent classes (i.e. one or more masks are empty)
            if not np.all([np.any(mask) for mask in gt_data]):
                return False
        if self.filter_empty_input:
            # Discard set of images if one of them is empty or filled with constant value (i.e. std == 0)
            if np.any([img.std() == 0 for img in input_data]):
                return False
        if self.filter_classification:
            if not np.all([int(
                    self.classifier(
                        imed_utils.cuda(torch.from_numpy(img.copy()).unsqueeze(0).unsqueeze(0),
                                        self.cuda_available))) for img in input_data]):
                return False

        return True
