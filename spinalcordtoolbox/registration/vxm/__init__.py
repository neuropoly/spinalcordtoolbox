"""
Vendored subset of VoxelMorph 0.2 (PyTorch backend), used by `algo=dl` registration.

Only the code needed to build `VxmDense` is included, so that SCT doesn't depend on the
`voxelmorph`, `neurite`, and `pystrum` packages. See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4899

Source: https://github.com/voxelmorph/voxelmorph (v0.2), licensed under the Apache License, Version 2.0.
See the LICENSE.md file in this directory for the full license text.
"""

from . import layers
from . import networks

__all__ = ["layers", "networks"]
