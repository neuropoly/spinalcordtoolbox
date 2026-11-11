# This file is vendored from VoxelMorph 0.2 (https://github.com/voxelmorph/voxelmorph),
# originally `voxelmorph/py/utils.py`, licensed under the Apache License, Version 2.0.
# See the LICENSE.md file in this directory for the full license text.
#
# Modifications for SCT:
#   - Removed everything except `default_unet_features`.


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features
