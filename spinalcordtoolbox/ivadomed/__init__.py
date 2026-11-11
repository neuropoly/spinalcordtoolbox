# This file is vendored from ivadomed 2.9.10 (https://github.com/ivadomed/ivadomed),
# originally `ivadomed/__init__.py`, licensed under the MIT License.
# See the LICENSE.md file in this directory for the full license text.
#
# Modifications for SCT:
#   - Fixed flake8 warnings, without changing behavior.

"""
Vendored subset of ivadomed 2.9.10, used by `sct_deepseg` to run inference with ivadomed-format models.

Only the inference code path (`inference.segment_volume`) is included, so that SCT doesn't depend on the
`ivadomed` package and its training-only dependencies.
See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4899

Source: https://github.com/ivadomed/ivadomed (v2.9.10), licensed under the MIT License.
See the LICENSE.md file in this directory for the full license text.
"""

from .utils import __version__, __ivadomed_dir__

__all__ = ["__version__", "__ivadomed_dir__"]
