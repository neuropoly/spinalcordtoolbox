# This file is vendored from ivadomed 2.9.10 (https://github.com/ivadomed/ivadomed),
# originally `ivadomed/loader/patch_filter.py`, licensed under the MIT License.
# See the LICENSE.md file in this directory for the full license text.
#
# Modifications for SCT: none.

import numpy as np


class PatchFilter(object):
    """Filter 2D or 3D patches from dataset.

    If a patch does not meet certain conditions, it is discarded from the dataset at training time.

    Args:
        filter_empty_mask (bool): If True, 2D or 3D patches where all voxel labels are zeros are discarded at training time.
        filter_absent_class (bool): If True, 2D or 3D patches where all voxel labels are zero for one or more classes are
            discarded at training time.
        filter_empty_input (bool): If True, 2D or 3D patches where all voxel intensities are zeros are discarded
            at training time.
        is_train (bool): Indicates if at training time.

    Attributes:
        filter_empty_mask (bool): If True, 2D or 3D patches where all voxel labels are zeros are discarded at training time.
            Default: False.
        filter_absent_class (bool): If True, 2D or 3D patches where all voxel labels are zero for one or more classes are
            discarded at training time. Default: False.
        filter_empty_input (bool): If True, 2D or 3D patches where all voxel intensities are zeros are discarded
            at training time. Default: False.
        is_train (bool): Indicates if at training time.

    """

    def __init__(self, filter_empty_mask: bool = False,
                 filter_absent_class: bool = False,
                 filter_empty_input: bool = False,
                 is_train: bool = False) -> None:
        self.filter_empty_mask = filter_empty_mask
        self.filter_absent_class = filter_absent_class
        self.filter_empty_input = filter_empty_input
        self.is_train = is_train

    def __call__(self, sample: dict) -> bool:
        """Extract input_data and gt_data lists from sample dict and discard them if they don't match certain
        conditions.

        """
        input_data, gt_data = sample['input'], sample['gt']

        if self.is_train:
            if self.filter_empty_mask:
                # Discard 2D or 3D patches that do not have ANY ground truth (i.e. all masks are empty) at training time
                if not np.any(gt_data):
                    return False
            if self.filter_absent_class:
                # Discard 2D or 3D patches that have absent classes (i.e. one or more masks are empty) at training time
                if not np.all([np.any(mask) for mask in gt_data]):
                    return False
            if self.filter_empty_input:
                # Discard set of 2D or 3D patches if one of them is empty or filled with constant value
                # (i.e. std == 0) at training time
                if np.any([img.std() == 0 for img in input_data]):
                    return False

        return True
