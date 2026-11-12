"""
SCT Image API

Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

# TODO: Sort out the use of Image.hdr and Image.header --> they seem to carry duplicated information.

import sys
import os
import errno
import itertools
import warnings
import logging
import math
from typing import Sequence, Tuple
from copy import deepcopy

import numpy as np
import pathlib
from contrib import fslhd
from scipy import ndimage

import transforms3d.affines as affines

from spinalcordtoolbox.types import Coordinate
from spinalcordtoolbox.utils.fs import extract_fname, mv, tmp_create
from spinalcordtoolbox.utils.sys import run_proc, LazyLoader

nib = LazyLoader("nib", globals(), "nibabel")

logger = logging.getLogger(__name__)


def _get_permutations(im_src_orientation, im_dst_orientation):
    """

    :param im_src_orientation str: Orientation of source image. Example: 'RPI'
    :param im_dest_orientation str: Orientation of destination image. Example: 'SAL'
    :return: list of axes permutations and list of inversions to achieve an orientation change
    """

    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}

    perm = [0, 1, 2]
    inversion = [1, 1, 1]
    for i, character in enumerate(im_src_orientation):
        try:
            perm[i] = im_dst_orientation.index(character)
        except ValueError:
            perm[i] = im_dst_orientation.index(opposite_character[character])
            inversion[i] = -1

    return perm, inversion


def rpi_slice_to_orig_orientation(dim, orig_orientation, slice_number, axis):
    """
    Convert slice number from RPI (right-posterior-inferior) to original orientation, e.g., AIL
    (anterior-inferior-left).
    :param dim: tuple, dimensions of the image in RPI orientation, e.g., (20, 640, 640).
    :param orig_orientation: str, original image orientation, e.g., AIL.
    :param slice_number: str, slice number in RPI orientation, e.g., 9.
    :param axis: int, axis of the slice in the RPI orientation. '0' corresponds to the x (R-L) axis,
    '1' to the y (A-P) axis, and '2' to the z (I-S) axis.
    :return: int, slice number in original orientation, e.g., 6.

    Example: considering an image with 20 sagittal slices (0-19) and sagittal slice number 13 in the AIL orientation,
    the corresponding slice number in the RPI orientation is 6:
            rpi_slice_to_orig_orientation((20, 640, 640), 'AIL', 13, 0) -> 6
    Note: we use 0 as the last arg in this example as it corresponds to the R-L direction (first axis in RPI)
    """
    # TODO: Consider rewriting this function to use `Coordinate.permute` in much the same way as `reorient_coordinates`.
    #       Since this function basically reimplements the same logic, it's probably overkill.
    # Get the inversions
    _, inversion = _get_permutations('RPI', orig_orientation)

    return (dim[axis] - 1 - slice_number) if inversion[axis] == -1 else slice_number


def orig_orientation_to_rpi_slice(slice_number, img_src):
    """Reorient an SI slice number to the corresponding SI slice number in RPI orientation."""
    # TODO: This is basically a special case of `reorient_coordinates`, with similar function signature. Like
    #       `rpi_slice_to_orig_orientation`, this function should be rewritten in a way that de-duplicates the API. The
    #       only reason this is kept separate is to highlight another niche case of slice-reorientation that should be
    #       considered when refactoring the API, and to consider various ways of doing the same thing.
    #       Of note about this implementation is that:
    #         1. We only need to convert a single value.
    #         2. We don't want to reorient a whole coordinate, just the SI slice.
    #         3. We already know the world axis (SI), and need to find the voxel axis (x/y/z) based on orig orientation.
    orig_orientation = img_src.orientation
    axis = next(orig_orientation.index(si) for si in ("I", "S") if si in orig_orientation)
    coord = [0, 0, 0]
    coord[axis] = slice_number
    coord_rpi = Coordinate(coord.copy()).permute(img_src, "RPI", mode='absolute')
    return coord_rpi[2]


def reorient_coordinates(coords, img_src, orient_dest, mode='absolute'):
    """
    Reorient coordinates from source image orientation to destination orientation.

    :param coords: iterable of [x, y, z] coordinates to be reoriented
    :param img_src: spinalcordtoolbox.Image() object. Must represent the space that the coordinate
                    is currently in. The source orientation and the dimensions are pulled from
                    this image, which are used to permute/invert the coordinate.
    :param orient_dest: The orientation to output the new coordinate in.
    :param mode: Determines how inversions are handled. If 'absolute', the coordinate is recomputed using
             a new origin based on the source image's maximum dimension for the inverted axes. If
             'relative', the coordinate is treated as vector and inverted by multiplying by -1.
    :return: numpy array with the new coordinates in the destination orientation.
    """

    return [Coordinate(list(coord)).permute(img_src, orient_dest, mode) for coord in coords]


class Slicer(object):
    """
    Provides a sliced view onto original image data.
    Can be used as a sequence.

    Notes:

    - The original image data is directly available without copy,
      which is a nice feature, not a bug! Use .copy() if you need copies...

    Example:

    .. code:: python

       for slice2d in msct_image.SlicerFancy(im3d, "RPI"):
           print(slice)

    """

    def __init__(self, im, orientation="LPI"):
        """

        :param im: image to iterate through
        :param orientation: "from" letters to indicate how to slice the image.
                     The slices are done on the last letter axis,
                     and they are defined as the first/second letter.
        """

        if not isinstance(im, Image):
            raise ValueError("Expecting an image")
        if orientation not in all_refspace_strings():
            raise ValueError("Invalid orientation spec")

        # Get a different view on data, as if we were doing a reorientation

        perm, inversion = _get_permutations(im.orientation, orientation)

        # axes inversion (flip)
        data = im.data[::inversion[0], ::inversion[1], ::inversion[2]]

        # axes manipulations (transpose)
        if perm == [1, 0, 2]:
            data = np.swapaxes(data, 0, 1)
        elif perm == [2, 1, 0]:
            data = np.swapaxes(data, 0, 2)
        elif perm == [0, 2, 1]:
            data = np.swapaxes(data, 1, 2)
        elif perm == [2, 0, 1]:
            data = np.swapaxes(data, 0, 2)  # transform [2, 0, 1] to [1, 0, 2]
            data = np.swapaxes(data, 0, 1)  # transform [1, 0, 2] to [0, 1, 2]
        elif perm == [1, 2, 0]:
            data = np.swapaxes(data, 0, 2)  # transform [1, 2, 0] to [0, 2, 1]
            data = np.swapaxes(data, 1, 2)  # transform [0, 2, 1] to [0, 1, 2]
        elif perm == [0, 1, 2]:
            # do nothing
            pass
        else:
            raise NotImplementedError()

        self._data = data
        self._orientation = orientation
        self._nb_slices = data.shape[2]

    def __len__(self):
        return self._nb_slices

    def __getitem__(self, idx):
        """

        :return: an image slice, at slicing index idx
        :param idx: slicing index (according to the slicing direction)
        """
        if not isinstance(idx, int):
            raise NotImplementedError()

        if idx >= self._nb_slices:
            raise IndexError(f"I just have {self._nb_slices} slices!")

        return self._data[:, :, idx]


class SlicerOneAxis(object):
    """
    Image slicer to use when you don't care about the 2D slice orientation,
    and don't want to specify them.
    The slicer will just iterate through the right axis that corresponds to
    its specification.

    Can help getting ranges and slice indices.
    """

    def __init__(self, im, axis="IS"):
        opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}
        axis_labels = "LRPAIS"
        if len(axis) != 2:
            raise ValueError()
        if axis[0] not in axis_labels:
            raise ValueError()
        if axis[1] not in axis_labels:
            raise ValueError()
        if axis[0] != opposite_character[axis[1]]:
            raise ValueError()

        for idx_axis in range(2):
            dim_nr = im.orientation.find(axis[idx_axis])
            if dim_nr != -1:
                break
        if dim_nr == -1:
            raise ValueError()

        # SCT convention
        from_dir = im.orientation[dim_nr]
        self.direction = +1 if axis[0] == from_dir else -1
        self.nb_slices = im.dim[dim_nr]
        self.im = im
        self.axis = axis
        self._slice = lambda idx: tuple([(idx if x in axis else slice(None)) for x in im.orientation])

    def __len__(self):
        return self.nb_slices

    def __getitem__(self, idx):
        """

        :return: an image slice, at slicing index idx
        :param idx: slicing index (according to the slicing direction)
        """
        if isinstance(idx, slice):
            raise NotImplementedError()

        if idx >= self.nb_slices:
            raise IndexError(f"I just have {self.nb_slices} slices!")

        if self.direction == -1:
            idx = self.nb_slices - 1 - idx

        return self.im.data[self._slice(idx)]


class SlicerMany(object):
    """
    Image*s* slicer utility class.

    Can help getting ranges and slice indices.
    Can provide slices (being an *iterator*).

    Use with great care for now, that it's not very documented.
    """

    def __init__(self, images, slicerclass, *args, **kw):
        if len(images) == 0:
            raise ValueError("Don't expect me to work on 0 images!")

        self.slicers = [slicerclass(im, *args, **kw) for im in images]

        nb_slices = [x._nb_slices for x in self.slicers]
        if len(set(nb_slices)) != 1:
            raise ValueError("All images must have the same number of slices along the slicing axis!")
        self._nb_slices = nb_slices[0]

    def __len__(self):
        return self._nb_slices

    def __getitem__(self, idx):
        return [x[idx] for x in self.slicers]


def check_affines_match(im):
    hdr = im.hdr
    hdr2 = hdr.copy()

    try:
        hdr2.set_qform(hdr.get_sform())
    except np.linalg.LinAlgError:
        # See https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3097
        logger.warning(f"The sform for {im.absolutepath} is uninitialized and may cause unexpected behaviour.")

        if im.absolutepath is None:
            logger.error("Internal code has produced an image with an uninitialized sform. "
                         "please report this on github at https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues "
                         "or on the SCT forums https://forum.spinalcordmri.org/.")

        return True

    return np.allclose(hdr.get_qform(), hdr2.get_qform(), atol=1e-3)


class Image(object):
    """
    Create an object that behaves similarly to nibabel's image object. Useful additions include: dim, check_sform and
    a few methods (load, save) that deal with image dtype.
    """

    def __init__(self, param=None, hdr=None, orientation=None, absolutepath=None, dim=None,
                 mmap=(not sys.platform.startswith('win32')), verbose=1, check_sform=False):
        """
        :param param: string indicating a path to a image file or an `Image` object.
        :param hdr: a nibabel header object to use as the header for the image (overwritten if `param` is provided)
        :param orientation: a three character orientation code (e.g. RPI).
        :param absolutepath: a relative path to associate with the image.
        :param dim: The dimensions of the image, defaults to automatically determined.
        :param mmap: Whether to load data arrays as np.memmaps. Defaults to False (i.e. np.array) on Windows, and True
          (i.e. np.memmap) on every other platform. (The context for defaulting to False on Windows is:
          https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3695)
        :param verbose: integer how verbose to be 0 is silent 1 is chatty.
        :param check_sform: whether or not to check whether the sform matches the qform. If this is set to `True`,
          `Image` will fail raise an error if they don't match.
        """

        # initialization of all parameters
        self.affine = None
        self.data = None
        self._path = None
        self.ext = ""

        if absolutepath is not None:
            self._path = os.path.abspath(absolutepath)

        # Case 1: load an image from file
        if isinstance(param, str):
            try:
                self.loadFromPath(param, mmap, verbose)
            except OSError as e:
                if e.errno == errno.EMFILE:
                    e.strerror += (". Please try increasing your system's file descriptor "
                                   "limit by using the command `ulimit -Sn`.")
                raise e
        # Case 2: create a copy of an existing `Image` object
        elif isinstance(param, type(self)):
            self.copy(param)
        # Case 3: create a blank image from a list of dimensions
        elif isinstance(param, list):
            self.data = np.zeros(param)
            self.hdr = hdr.copy() if hdr is not None else nib.Nifti1Header()
            self.hdr.set_data_shape(self.data.shape)
        # Case 4: create an image from an existing data array
        elif isinstance(param, (np.ndarray, np.generic)):
            self.data = param
            self.hdr = hdr.copy() if hdr is not None else nib.Nifti1Header()
            self.hdr.set_data_shape(self.data.shape)
        else:
            raise TypeError('Image constructor takes at least one argument.')

        # Fix any mismatch between the array's datatype and the header datatype
        self.fix_header_dtype()

        # set a more permissive threshold for reading the qform
        # (see https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3703 for details)
        self.hdr.quaternion_threshold = -1e-6

        # Make sure sform and qform are the same.
        # Context: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2429
        if check_sform and not check_affines_match(self):
            logger.error(f"Image {self._path} has different qform and sform matrices. This can produce incorrect "
                         f"results. Please use 'sct_image -i {self._path} -header' to check that both affine "
                         f"matrices are valid. Then, consider running either 'sct_image -set-sform-to-qform' or "
                         f"'sct_image -set-qform-to-sform' to fix any discrepancies you may find.")
            logger.error("If internal SCT code has produced an intermediate/temporary file with this issue, please report this on GitHub at "
                         "https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues or on the SCT forum https://forum.spinalcordmri.org/.")

            # Temporarily skip raising an error, because we now know that "orthogonal qform matrices" from reorientation can cause sform/qform
            # discrepancies that trigger this error.
            #
            # raise ValueError("Image sform does not match qform")
            #
            # Ideally, we would solve the sform/qform discrepancies at the source. But doing so produced an even greater breaking change.
            # So, the safest approach in the short term is to keep the existing results, but skip the above failure and just emit a message.
            # This way, if the above error would trigger, the user at least knows that there may be an issue.
            # Original issue: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4689
            # Secondary issue caused by the "fix" for the first issue: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4744

        self.affine = self.hdr.get_best_affine()

    @property
    def dim(self):
        return get_dimension(self)

    @property
    def orientation(self):
        return get_orientation(self)

    @property
    def absolutepath(self):
        """
        Storage path (either actual or potential)

        Notes:

        - As several tools perform chdir() it's very important to have absolute paths
        - When set, if relative:

          - If it already existed, it becomes a new basename in the old dirname
          - Else, it becomes absolute (shortcut)

        Usually not directly touched (use `Image.save`), but in some cases it's
        the best way to set it.
        """
        return self._path

    @absolutepath.setter
    def absolutepath(self, value):
        if value is None:
            self._path = None
            return
        elif not os.path.isabs(value) and self._path is not None:
            value = os.path.join(os.path.dirname(self._path), value)
        elif not os.path.isabs(value):
            value = os.path.abspath(value)
        self._path = value

    @property
    def header(self):
        return self.hdr

    @header.setter
    def header(self, value):
        self.hdr = value

    def __deepcopy__(self, memo):
        return type(self)(deepcopy(self.data, memo), deepcopy(self.hdr, memo), deepcopy(self.orientation, memo), deepcopy(self.absolutepath, memo), deepcopy(self.dim, memo))

    def copy(self, image=None):
        if image is not None:
            self.affine = deepcopy(image.affine)
            self.data = deepcopy(image.data)
            self.hdr = deepcopy(image.hdr)
            self._path = deepcopy(image._path)
        else:
            return deepcopy(self)

    def copy_affine_from_ref(self, im_ref):
        """
        Copy qform and sform and associated codes from a reference Image object

        :param im_ref:
        :return:
        """
        # Copy q/sform and code
        self.hdr.set_qform(im_ref.hdr.get_qform())
        self.hdr._structarr['qform_code'] = im_ref.hdr._structarr['qform_code']
        self.hdr.set_sform(im_ref.hdr.get_sform())
        self.hdr._structarr['sform_code'] = im_ref.hdr._structarr['sform_code']

    def set_sform_to_qform(self):
        """Use this (or set_qform_to_sform) when matching matrices are required."""
        self.hdr.set_sform(self.hdr.get_qform())
        self.hdr._structarr['sform_code'] = self.hdr._structarr['qform_code']

    def set_qform_to_sform(self):
        """Use this or (set_sform_to_qform) when matching matrices are required."""
        self.hdr.set_qform(self.hdr.get_sform())
        self.hdr._structarr['qform_code'] = self.hdr._structarr['sform_code']

    def fix_header_dtype(self):
        """
        Change the header dtype to the match the datatype of the array.
        """
        # Using bool for nibabel headers is unsupported, so use uint8 instead:
        # `nibabel.spatialimages.HeaderDataError: data dtype "bool" not supported`
        dtype_data = self.data.dtype
        if dtype_data == bool:
            dtype_data = np.uint8

        dtype_header = self.hdr.get_data_dtype()
        if dtype_header != dtype_data:
            logger.debug(f"Image header specifies datatype '{dtype_header}', but array is of type "
                         f"'{dtype_data}'. Header metadata will be overwritten to use '{dtype_data}'.")
            self.hdr.set_data_dtype(dtype_data)

    def loadFromPath(self, path, mmap, verbose):
        """
        This function load an image from an absolute path using nibabel library

        :param path: path of the file from which the image will be loaded
        :return:
        """

        self.absolutepath = os.path.abspath(path)
        im_file = nib.load(self.absolutepath, mmap=mmap)
        self.affine = im_file.affine.copy()
        self.data = np.asanyarray(im_file.dataobj)
        self.hdr = im_file.header.copy()
        if path != self.absolutepath:
            logger.debug("Loaded %s (%s) orientation %s shape %s", path, self.absolutepath, self.orientation, self.data.shape)
        else:
            logger.debug("Loaded %s orientation %s shape %s", path, self.orientation, self.data.shape)

    def change_shape(self, shape):
        """
        Change data shape (in-place)

        This is mostly useful for adding/removing a fourth dimension,
        you probably don't want to use this function.

        """
        change_shape(self, shape, self)
        return self

    def change_orientation(self, orientation, inverse=False):
        """
        Change orientation on image (in-place).

        :param orientation: orientation string (SCT "from" convention)

        :param inverse: if you think backwards, use this to specify that you actually\
                        want to transform *from* the specified orientation, not *to*\
                        it.

        """
        change_orientation(self, orientation, self, inverse=inverse)
        return self

    def change_type(self, dtype):
        """
        Change data type on image.

        Note: the image path is voided.
        """
        change_type(self, dtype, self)
        return self

    def save(self, path=None, dtype=None, verbose=1, mutable=False):
        """
        Write an image in a nifti file

        :param path: Where to save the data, if None it will be taken from the\
                     absolutepath member.\
                     If path is a directory, will save to a file under this directory\
                     with the basename from the absolutepath member.

        :param dtype: if not set, the image is saved in the same type as input data\
                      if 'minimize', image storage space is minimized\
                        (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),\
                        (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),\
                        (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),\
                        (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),\
                        (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),\
                        (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),\
                        (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),\
                        (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),\
                        (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),\
                        (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),\
                        (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),\
                        (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),\
                        (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),\
                        (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),

        :param mutable: whether to update members with newly created path or dtype
        """
        if mutable:  # do all modifications in-place
            # Case 1: `path` not specified
            if path is None:
                if self.absolutepath:  # Fallback to the original filepath
                    path = self.absolutepath
                else:
                    raise ValueError("Don't know where to save the image (no absolutepath or path parameter)")
            # Case 2: `path` points to an existing directory
            elif os.path.isdir(path):
                if self.absolutepath:  # Use the original filename, but save to the directory specified by `path`
                    path = os.path.join(os.path.abspath(path), os.path.basename(self.absolutepath))
                else:
                    raise ValueError("Don't know where to save the image (path parameter is dir, but absolutepath is "
                                     "missing)")
            # Case 3: `path` points to a file (or a *nonexistent* directory) so use its value as-is
            #    (We're okay with letting nonexistent directories slip through, because it's difficult to distinguish
            #     between nonexistent directories and nonexistent files. Plus, `nibabel` will catch any further errors.)
            else:
                pass

            if os.path.isfile(path) and verbose:
                logger.warning("File %s already exists. Will overwrite it.", path)
            if os.path.isabs(path):
                logger.debug("Saving image to %s orientation %s shape %s",
                             path, self.orientation, self.data.shape)
            else:
                logger.debug("Saving image to %s (%s) orientation %s shape %s",
                             path, os.path.abspath(path), self.orientation, self.data.shape)

            # Now that `path` has been set and log messages have been written, we can assign it to the image itself
            self.absolutepath = os.path.abspath(path)

            if dtype is not None:
                self.change_type(dtype)

            if self.hdr is not None:
                self.hdr.set_data_shape(self.data.shape)
                self.fix_header_dtype()

            # nb. that copy() is important because if it were a memory map, save() would corrupt it
            dataobj = self.data.copy() if isinstance(self.data, np.memmap) else self.data
            affine = None
            header = self.hdr.copy() if self.hdr is not None else None
            nib.save(nib.Nifti1Image(dataobj, affine, header), self.absolutepath)
            if not os.path.isfile(self.absolutepath):
                raise RuntimeError(f"Couldn't save image to {self.absolutepath}")
        else:
            # if we're not operating in-place, then make any required modifications on a throw-away copy
            self.copy().save(path, dtype, verbose, mutable=True)
        return self

    def getNonZeroCoordinates(self, sorting=None, reverse_coord=False):
        """
        This function return all the non-zero coordinates that the image contains.
        Coordinate list can also be sorted by x, y, z, or the value with the parameter sorting='x', sorting='y', sorting='z' or sorting='value'
        If reverse_coord is True, coordinate are sorted from larger to smaller.
        """
        n_dim = 1
        if self.dim[3] == 1:
            n_dim = 3
        else:
            n_dim = 4
        if self.dim[2] == 1:
            n_dim = 2

        if n_dim == 3:
            X, Y, Z = (self.data > 0).nonzero()
            list_coordinates = [Coordinate([X[i], Y[i], Z[i], self.data[X[i], Y[i], Z[i]]]) for i in range(0, len(X))]
        elif n_dim == 2:
            try:
                X, Y = (self.data > 0).nonzero()
                list_coordinates = [Coordinate([X[i], Y[i], 0, self.data[X[i], Y[i]]]) for i in range(0, len(X))]
            except ValueError:
                X, Y, Z = (self.data > 0).nonzero()
                list_coordinates = [Coordinate([X[i], Y[i], 0, self.data[X[i], Y[i], 0]]) for i in range(0, len(X))]

        if sorting is not None:
            if reverse_coord not in [True, False]:
                raise ValueError('reverse_coord parameter must be a boolean')

            if sorting == 'x':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.x, reverse=reverse_coord)
            elif sorting == 'y':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.y, reverse=reverse_coord)
            elif sorting == 'z':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.z, reverse=reverse_coord)
            elif sorting == 'value':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.value, reverse=reverse_coord)
            else:
                raise ValueError("sorting parameter must be either 'x', 'y', 'z' or 'value'")

        return list_coordinates

    def getNonZeroValues(self, sorting=True):
        """
        This function return all the non-zero unique values that the image contains.
        If sorting is set to True, the list will be sorted.
        """
        list_values = list(np.unique(self.data[self.data > 0]))
        if sorting:
            list_values.sort()
        return list_values

    def getCoordinatesAveragedByValue(self):
        """
        This function computes the mean coordinate of group of labels in the image. This is especially useful for label's images.

        :return: list of coordinates that represent the center of mass of each group of value.
        """
        # 1. Extraction of coordinates from all non-null voxels in the image. Coordinates are sorted by value.
        coordinates = self.getNonZeroCoordinates(sorting='value')

        # 2. Separate all coordinates into groups by value
        groups = dict()
        for coord in coordinates:
            if coord.value in groups:
                groups[coord.value].append(coord)
            else:
                groups[coord.value] = [coord]

        # 3. Compute the center of mass of each group of voxels and write them into the output image
        averaged_coordinates = []
        for value, list_coord in groups.items():
            averaged_coordinates.append(sum(list_coord) / float(len(list_coord)))

        averaged_coordinates = sorted(averaged_coordinates, key=lambda obj: obj.value, reverse=False)
        return averaged_coordinates

    def transfo_pix2phys(self, coordi, mode='absolute'):
        """
        This function returns the physical coordinates of all points of 'coordi'.

        :param coordi: sequence of (nb_points x 3) values containing the pixel coordinate of points.
        :param mode: either 'absolute' or 'relative'.
                     Use 'absolute' to transform absolute pixel coordinates, taking into account
                     the origin of the physical coordinate system. (For example, use 'absolute'
                     for individual voxels.)
                     Use 'relative' to transform relative pixel coordinates, ignoring the origin of
                     the physical coordinate system. (For example, use 'relative' for the
                     difference between two voxels, or for derivatives.)
        :return: sequence with the physical coordinates of the points in the space of the image.

        Example:

        .. code:: python

            img = Image('file.nii.gz')
            coordi_pix = [[1,1,1]]   # for points: (1,1,1). N.B. Important to write [[x,y,z]] instead of [x,y,z]
            coordi_pix = [[1,1,1],[2,2,2],[4,4,4]]   # for points: (1,1,1), (2,2,2) and (4,4,4)
            coordi_phys = img.transfo_pix2phys(coordi=coordi_pix)

        """
        coordi = np.asarray(coordi, dtype=np.float64)
        num_points, dimension = coordi.shape
        if dimension != 3:
            raise ValueError(f'wrong {dimension=}')
        if mode == 'absolute':
            affine_column = np.ones((num_points, 1), dtype=np.float64)
        elif mode == 'relative':
            affine_column = np.zeros((num_points, 1), dtype=np.float64)
        else:
            raise ValueError(f'invalid {mode=}')
        augmented_pix = np.hstack([coordi, affine_column])
        # The affine matrix usually transforms _column_ vectors of pix coordinates, but
        # `augmented_pix` takes the form of _row_ vectors. So, we transpose before and
        # after we do the matrix multiplication.
        affine_matrix = self.hdr.get_best_affine()
        augmented_phys = np.matmul(affine_matrix, augmented_pix.T).T
        return augmented_phys[:, :3]

    def transfo_phys2pix(self, coordi, real=True):
        """
        This function returns the pixels coordinates of all points of 'coordi'

        :param coordi: sequence of (nb_points x 3) values containing the pixel coordinate of points.
        :param real: whether to return real pixel coordinates
        :return: sequence with the physical coordinates of the points in the space of the image.
        """

        m_p2f = self.hdr.get_best_affine()
        m_f2p = np.linalg.inv(m_p2f)
        aug = np.hstack((np.asarray(coordi), np.ones((len(coordi), 1))))
        ret = np.empty_like(coordi, dtype=np.float64)
        for idx_coord, coord in enumerate(aug):
            phys = np.matmul(m_f2p, coord)
            ret[idx_coord] = phys[:3]
        if real:
            return np.int32(np.round(ret))
        else:
            return ret

    def get_values(self, coordi=None, interpolation_mode=0, border='constant', cval=0.0):
        """
        This function returns the intensity value of the image at the position coordi (can be a list of coordinates).

        :param coordi: continuouspix
        :param interpolation_mode: 0=nearest neighbor, 1= linear, 2= 2nd-order spline, 3= 2nd-order spline, 4= 2nd-order spline, 5= 5th-order spline
        :return: intensity values at continuouspix with interpolation_mode
        """
        return ndimage.map_coordinates(self.data, coordi, output=np.float32, order=interpolation_mode,
                                       mode=border, cval=cval)

    def get_transform(self, im_ref, mode='affine'):
        aff_im_self = self.affine
        aff_im_ref = im_ref.affine
        transform = np.matmul(np.linalg.inv(aff_im_self), aff_im_ref)
        if mode == 'affine':
            transform = np.matmul(np.linalg.inv(aff_im_self), aff_im_ref)
        else:
            T_self, R_self, Sc_self, Sh_self = affines.decompose44(aff_im_self)
            T_ref, R_ref, Sc_ref, Sh_ref = affines.decompose44(aff_im_ref)
            if mode == 'translation':
                T_transform = T_ref - T_self
                R_transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid':
                T_transform = T_ref - T_self
                R_transform = np.matmul(np.linalg.inv(R_self), R_ref)
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid_scaling':
                T_transform = T_ref - T_self
                R_transform = np.matmul(np.linalg.inv(R_self), R_ref)
                Sc_transform = Sc_ref / Sc_self
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            else:
                transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return transform

    def get_inverse_transform(self, im_ref, mode='affine'):
        aff_im_self = self.affine
        aff_im_ref = im_ref.affine
        if mode == 'affine':
            transform = np.matmul(np.linalg.inv(aff_im_ref), aff_im_self)
        else:
            T_self, R_self, Sc_self, Sh_self = affines.decompose44(aff_im_self)
            T_ref, R_ref, Sc_ref, Sh_ref = affines.decompose44(aff_im_ref)
            if mode == 'translation':
                T_transform = T_self - T_ref
                R_transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid':
                T_transform = T_self - T_ref
                R_transform = np.matmul(np.linalg.inv(R_ref), R_self)
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid_scaling':
                T_transform = T_self - T_ref
                R_transform = np.matmul(np.linalg.inv(R_ref), R_self)
                Sc_transform = Sc_self / Sc_ref
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            else:
                transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        return transform

    def get_directions(self):
        """
        This function return the X, Y, and Z axes of the image

        return: X, Y and Z axes of the image
        """
        direction_matrix = self.header.get_best_affine()
        T_self, R_self, Sc_self, Sh_self = affines.decompose44(direction_matrix)
        return R_self[0:3, 0], R_self[0:3, 1], R_self[0:3, 2]

    def interpolate_from_image(self, im_ref, fname_output=None, interpolation_mode=1, border='constant'):
        """
        This function interpolates an image by following the grid of a reference image.
        Example of use:

        .. code:: python

            from spinalcordtoolbox.image import Image
            im_input = Image(fname_input)
            im_ref = Image(fname_ref)
            im_input.interpolate_from_image(im_ref, fname_output, interpolation_mode=1)

        :param im_ref: reference Image that contains the grid on which interpolate.
        :param border: Points outside the boundaries of the input are filled according\
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap')
        :return: a new image that has the same dimensions/grid of the reference image but the data of self image.
        """
        nx, ny, nz, nt, px, py, pz, pt = im_ref.dim
        x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
        indexes_ref = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())))
        physical_coordinates_ref = im_ref.transfo_pix2phys(indexes_ref)

        # TODO: add optional transformation from reference space to image space to physical coordinates of ref grid.
        # TODO: add choice to do non-full transorm: translation, (rigid), affine
        # 1. get transformation
        # 2. apply transformation on coordinates

        coord_im = self.transfo_phys2pix(physical_coordinates_ref, real=False)
        interpolated_values = self.get_values(np.array([coord_im[:, 0], coord_im[:, 1], coord_im[:, 2]]), interpolation_mode=interpolation_mode, border=border)

        im_output = Image(im_ref)
        if interpolation_mode == 0:
            im_output.change_type('int32')
        else:
            im_output.change_type('float32')
        im_output.data = np.reshape(interpolated_values, (nx, ny, nz))
        if fname_output is not None:
            im_output.absolutepath = fname_output
            im_output.save()
        return im_output

    def mean(self, dim):
        """
        Average across specified dimension

        :param dim: int: axis used for averaging
        :return: Image object
        """
        im_out = empty_like(self)
        im_out.data = np.mean(self.data, dim)
        return im_out


def compute_dice(image1, image2, mode='3d', label=1, zboundaries=False):
    """
    This function computes the Dice coefficient between two binary images.
    :param image1: object Image
    :param image2: object Image
    :param mode: mode of computation of Dice.\
            3d: compute Dice coefficient over the full 3D volume\
            2d-slices: compute the 2D Dice coefficient for each slice of the volumes\
    :param: label: binary label for which Dice coefficient will be computed. Default=1
    :paaram: zboundaries: True/False. If True, the Dice coefficient is computed over a Z-ROI where both segmentations are\
                 present. Default=False.

    :return: Dice coefficient as a float between 0 and 1. Raises ValueError exception if an error occurred.
    """
    MODES = ['3d', '2d-slices']
    if mode not in MODES:
        raise ValueError(f'\n\nERROR: mode must be one of these values:{",  ".join(MODES)}')

    dice = 0.0  # default value of dice is 0

    # check if images are in the same coordinate system
    if image1.data.shape != image2.data.shape:
        raise ValueError(
            f"\n\nERROR: the data ({image1.absolutepath} and {image2.absolutepath}) don't have the same size."
            f"\nPlease use  \"sct_register_multimodal -i im1.nii.gz -d im2.nii.gz -identity 1\"  "
            f"to put the input images in the same space"
        )

    # if necessary, change orientation of images to RPI and compute segmentation boundaries
    if mode == '2d-slices' or (mode == '3d' and zboundaries):
        # changing orientation to RPI if necessary
        if image1.orientation != 'RPI':
            image1 = change_orientation(image1, "RPI")
        if image2.orientation != 'RPI':
            image2 = change_orientation(image2, "RPI")

        zmin, zmax = 0, image1.data.shape[2] - 1
        if zboundaries:
            # compute Z-ROI for which both segmentations are present.
            for z in range(zmin, zmax + 1):  # going from inferior to superior
                if np.any(image1.data[:, :, z]) and np.any(image2.data[:, :, z]):
                    zmin = z
                    break
            for z in range(zmax, zmin + 1, -1):  # going from superior to inferior
                if np.any(image1.data[:, :, z]) and np.any(image2.data[:, :, z]):
                    zmax = z
                    break

        if zmin > zmax:
            # segmentations do not overlap
            return 0.0

        if mode == '3d':
            # compute dice coefficient over Z-ROI
            data1 = image1.data[:, :, zmin:zmax]
            data2 = image2.data[:, :, zmin:zmax]

            dice = np.sum(data2[data1 == label]) * 2.0 / (np.sum(data1) + np.sum(data2))

        elif mode == '2d-slices':
            raise ValueError('2D slices Dice coefficient feature is not implemented yet')

    elif mode == '3d':
        # compute 3d dice coefficient
        dice = np.sum(image2.data[image1.data == label]) * 2.0 / (np.sum(image1.data) + np.sum(image2.data))

    return dice


def concat_data(im_in_list: Sequence[Image | str], dim, pixdim=None, squeeze_data=False):
    """
    Concatenate data

    :param im_in_list: list of Images
    :param dim: dimension: 0, 1, 2, 3.
    :param pixdim: pixel resolution to join to image header
    :param squeeze_data: bool: if True, remove the last dim if it is a singleton.
    :return im_out: concatenated image
    """
    # WARNING: calling concat_data in python instead of in command line causes a non-understood issue (results are
    # different with both options) from numpy import concatenate, expand_dims

    # determine the datatype of the final array
    arrs = [Image(im).data for im in im_in_list]
    dtypes = [arr.dtype for arr in arrs]
    dtypes_unique = list(set(dtypes))
    final_dtype = np.result_type(*dtypes_unique)
    if len(dtypes_unique) != 1:
        logger.warning(f"Concatenating images with different datatypes: {dtypes_unique}. Using dtype {final_dtype}.")

    # determine how many dimensions there should be in the final array
    shapes = [list(arr.shape) for arr in arrs]
    n_dims = max([dim+1] +                  # Option 1: dim+1 > len(shape), so a new axis will be added
                 [len(s) for s in shapes])  # Option 2: len(shape) >= dim+1, so we'll concatenate along an existing axis
    del arrs  # conserve memory

    # determine how many volumes there will be in the final array
    n_vols = 0
    volume_shapes = []
    for shape in shapes:
        # ensure we're not trying to, say, concatenate a 2D array with a 4D array
        if len(shape) not in [n_dims, n_dims - 1]:
            raise ValueError(f"Cannot concatenate shape of size {len(shape)} to expected size {n_dims}.")
        # we can pop the `dim` to simultaneously get the n_vols and also isolate the volume shape
        n_vols += shape.pop(dim) if len(shape) == n_dims else 1
        volume_shapes.append(tuple(shape))

    # make sure we're concatenating compatible shapes
    volume_shapes_unique = list(set(volume_shapes))
    if len(volume_shapes_unique) != 1:
        raise ValueError(f"Cannot concatenate incompatible shapes: {volume_shapes_unique}.")
    final_shape = list(volume_shapes_unique[0])
    final_shape.insert(dim, n_vols)

    # Preallocate the array in memory (to allow us to replace individual volumes one at a time)
    # NB: This can be very large for big images, such as `sct_apply_transfo` on a moco image warped to the PAM50
    # template. But, I don't think there's any way to avoid keeping 1 full copy in memory. I've tried using `np.memmap`
    # instead of `np.empty`, but calling it still allocates the full array in memory.
    data_concat = np.empty(final_shape, dtype=final_dtype)

    # Insert volumes one at a time to avoid doubling memory usage (as would occur with `np.concatenate(dat_list)`)
    i = 0
    for im in im_in_list:
        # make sure the source array has the same number of dimensions as the destination array
        vol = Image(im).data
        if len(vol.shape) < n_dims:
            vol = np.expand_dims(vol, axis=dim)
        # create slice object to index into the source and destination arrays
        j = i + vol.shape[dim]
        idx = [slice(None)] * n_dims
        idx[dim] = slice(i, j)
        data_concat[tuple(idx)] = vol
        i = j
        del vol  # conserve memory

    im_in_first = Image(im_in_list[0])
    im_out = empty_like(im_in_first)  # NB: empty_like reuses the header from the first input image for im_out
    if im_in_first.absolutepath is not None:
        im_out.absolutepath = add_suffix(im_in_first.absolutepath, '_concat')

    if pixdim is not None:
        im_out.hdr['pixdim'] = pixdim

    if squeeze_data and data_concat.shape[dim] == 1:
        # remove the last dim if it is a singleton.
        im_out.data = data_concat.reshape(
            tuple([x for (idx_shape, x) in enumerate(data_concat.shape) if idx_shape != dim]))
    else:
        im_out.data = data_concat

    # TODO: the line below fails because .dim is immutable. We should find a solution to update dim accordingly
    #  because as of now, this field contains wrong values (in this case, the dimension should be changed). Also
    #  see mean()
    # im_out.dim = im_out.data.shape[:dim] + (1,) + im_out.data.shape[dim:]

    return im_out


def find_zmin_zmax(im, threshold=0.1):
    """
    Find the min (and max) z-slice index below which (and above which) slices only have voxels below a given threshold.

    :param im: Image object
    :param threshold: threshold to apply before looking for zmin/zmax, typically corresponding to noise level.
    :return: [zmin, zmax]
    """
    slicer = SlicerOneAxis(im, axis="IS")

    # Make sure image is not empty
    if not np.any(slicer):
        logger.error('Input image is empty')

    # Iterate from bottom to top until we find data
    for zmin in range(0, len(slicer)):
        if np.any(slicer[zmin] > threshold):
            break

    # Conversely from top to bottom
    for zmax in range(len(slicer) - 1, zmin, -1):
        if np.any(slicer[zmax] > threshold):
            break

    return zmin, zmax


def get_dimension(im_file, verbose=1):
    """
    Get dimension from Image or nibabel object. Manages 2D, 3D or 4D images.

    :param: im_file: Image or nibabel object
    :return: nx, ny, nz, nt, px, py, pz, pt
    """
    if not isinstance(im_file, (nib.Nifti1Image, Image)):
        raise TypeError("The provided image file is neither a nibabel.nifti1.Nifti1Image instance nor an Image instance")
    # initializating ndims [nx, ny, nz, nt] and pdims [px, py, pz, pt]
    ndims = [1, 1, 1, 1]
    pdims = [1, 1, 1, 1]
    data_shape = im_file.header.get_data_shape()
    # NB: Nibabel stores dim info as float32. Arithmetic using float32 values can cause issues. So, store the values
    #     as rounded 64-bit float values, according to the spacing of the value.
    zooms = []
    for zoom_f32 in im_file.header.get_zooms():
        # extract the exponent from the spacing used by the float32 values (i.e. 1e-9 -> '-9')
        spacing = np.spacing(zoom_f32)
        spacing_exp = np.floor(np.log10(np.abs(spacing))).astype(int)
        # convert to python float, then round to remove any spurious digits
        zoom_f64 = float(zoom_f32)
        zoom_f64_rounded = round(zoom_f64, -spacing_exp - 1)
        zooms.append(zoom_f64_rounded)
    for i in range(min(len(data_shape), 4)):
        ndims[i] = data_shape[i]
        pdims[i] = zooms[i]
    return *ndims, *pdims


def all_refspace_strings():
    """

    :return: all possible orientation strings ['RAI', 'RAS', 'RPI', 'RPS', ...]
    """
    return [x for x in itertools.chain(*[["".join(x) for x in itertools.product(*seq)] for seq in itertools.permutations(("RL", "AP", "IS"), 3)])]


def get_orientation(im):
    """

    :param im: an Image
    :return: reference space string (ie. what's in Image.orientation)
    """
    res = "".join(nib.orientations.aff2axcodes(im.hdr.get_best_affine()))
    return orientation_string_nib2sct(res)
    return res  # for later ;)


def orientation_string_nib2sct(s):
    """

    :return: SCT reference space code from nibabel one
    """
    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}
    return "".join([opposite_character[x] for x in s])


orientation_string_sct2nib = orientation_string_nib2sct


def change_shape(im_src, shape, im_dst=None):
    """

    :param shape: shape to obtain (must be compatible with original one)
    :return: an image with changed shape

    .. note::
        The resulting image has no path
    """

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    if im_src.data.flags.f_contiguous:
        im_dst.data = im_src.data.reshape(shape, order="F")
    elif im_src.data.flags.c_contiguous:
        warnings.warn("Encountered an array with C order, strange!")
        im_dst.data = im_src.data.reshape(shape, order="C")
    else:
        # image data may be a view
        im_dst.data = im_src.data.copy().reshape(shape, order="F")

    pair = nib.nifti1.Nifti1Pair(im_dst.data, im_dst.hdr.get_best_affine(), im_dst.hdr)
    im_dst.hdr = pair.header
    return im_dst


def change_orientation(im_src, orientation, im_dst=None, inverse=False):
    """

    :param im_src: source image
    :param orientation: orientation string (SCT "from" convention)
    :param im_dst: destination image (can be the source image for in-place
                   operation, can be unset to generate one)
    :param inverse: if you think backwards, use this to specify that you actually
                    want to transform *from* the specified orientation, not *to* it.
    :return: an image with changed orientation

    .. note::
        - the resulting image has no path member set
        - if the source image is < 3D, it is reshaped to 3D and the destination is 3D
    """

    if len(im_src.data.shape) < 3:
        pass  # Will reshape to 3D
    elif len(im_src.data.shape) == 3:
        pass  # OK, standard 3D volume
    elif len(im_src.data.shape) == 4:
        pass  # OK, standard 4D volume
    elif len(im_src.data.shape) == 5 and im_src.header.get_intent()[0] == "vector":
        pass  # OK, physical displacement field
    else:
        raise NotImplementedError("Don't know how to change orientation for this image")

    im_src_orientation = im_src.orientation
    im_dst_orientation = orientation
    if inverse:
        im_src_orientation, im_dst_orientation = im_dst_orientation, im_src_orientation

    perm, inversion = _get_permutations(im_src_orientation, im_dst_orientation)

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    im_src_data = im_src.data
    if len(im_src_data.shape) < 3:
        im_src_data = im_src_data.reshape(tuple(list(im_src_data.shape) + ([1] * (3 - len(im_src_data.shape)))))

    # Update data by performing inversions and swaps

    # axes inversion (flip)
    data = im_src_data[::inversion[0], ::inversion[1], ::inversion[2]]

    # axes manipulations (transpose)
    if perm == [1, 0, 2]:
        data = np.swapaxes(data, 0, 1)
    elif perm == [2, 1, 0]:
        data = np.swapaxes(data, 0, 2)
    elif perm == [0, 2, 1]:
        data = np.swapaxes(data, 1, 2)
    elif perm == [2, 0, 1]:
        data = np.swapaxes(data, 0, 2)  # transform [2, 0, 1] to [1, 0, 2]
        data = np.swapaxes(data, 0, 1)  # transform [1, 0, 2] to [0, 1, 2]
    elif perm == [1, 2, 0]:
        data = np.swapaxes(data, 0, 2)  # transform [1, 2, 0] to [0, 2, 1]
        data = np.swapaxes(data, 1, 2)  # transform [0, 2, 1] to [0, 1, 2]
    elif perm == [0, 1, 2]:
        # do nothing
        pass
    else:
        raise NotImplementedError()

    # Update header

    im_src_aff = im_src.hdr.get_best_affine()
    aff = nib.orientations.inv_ornt_aff(
        np.array((perm, inversion)).T,
        im_src_data.shape)
    im_dst_aff = np.matmul(im_src_aff, aff)

    im_dst.header.set_qform(im_dst_aff)
    im_dst.header.set_sform(im_dst_aff)
    im_dst.header.set_data_shape(data.shape)
    im_dst.data = data

    return im_dst


def change_type(im_src, dtype, im_dst=None):
    """
    Change the voxel type of the image

    :param dtype:    if not set, the image is saved in standard type\
                    if 'minimize', image space is minimize\
                    if 'minimize_int', image space is minimize and values are approximated to integers\
                    (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),\
                    (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),\
                    (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),\
                    (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),\
                    (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),\
                    (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),\
                    (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),\
                    (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),\
                    (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),\
                    (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),\
                    (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),\
                    (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),\
                    (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),\
                    (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),
    :return:
    """

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    if dtype is None:
        return im_dst

    # get min/max from input image
    min_in = np.nanmin(im_src.data)
    max_in = np.nanmax(im_src.data)

    # find optimum type for the input image
    if dtype in ('minimize', 'minimize_int'):
        # warning: does not take intensity resolution into account, neither complex voxels

        # check if voxel values are real or integer
        isInteger = True
        if dtype == 'minimize':
            for vox in im_src.data.flatten():
                if int(vox) != vox:
                    isInteger = False
                    break

        if isInteger:
            if min_in >= 0:  # unsigned
                if max_in <= np.iinfo(np.uint8).max:
                    dtype = np.uint8
                elif max_in <= np.iinfo(np.uint16).max:
                    dtype = np.uint16
                elif max_in <= np.iinfo(np.uint32).max:
                    dtype = np.uint32
                elif max_in <= np.iinfo(np.uint64).max:
                    dtype = np.uint64
                else:
                    raise ValueError("Maximum value of the image is to big to be represented.")
            else:
                if max_in <= np.iinfo(np.int8).max and min_in >= np.iinfo(np.int8).min:
                    dtype = np.int8
                elif max_in <= np.iinfo(np.int16).max and min_in >= np.iinfo(np.int16).min:
                    dtype = np.int16
                elif max_in <= np.iinfo(np.int32).max and min_in >= np.iinfo(np.int32).min:
                    dtype = np.int32
                elif max_in <= np.iinfo(np.int64).max and min_in >= np.iinfo(np.int64).min:
                    dtype = np.int64
                else:
                    raise ValueError("Maximum value of the image is to big to be represented.")
        else:
            # if max_in <= np.finfo(np.float16).max and min_in >= np.finfo(np.float16).min:
            #    type = 'np.float16' # not supported by nibabel
            if max_in <= np.finfo(np.float32).max and min_in >= np.finfo(np.float32).min:
                dtype = np.float32
            elif max_in <= np.finfo(np.float64).max and min_in >= np.finfo(np.float64).min:
                dtype = np.float64

        dtype = to_dtype(dtype)
    else:
        dtype = to_dtype(dtype)

        # if output type is int, check if it needs intensity rescaling
        if "int" in dtype.name:
            # get min/max from output type
            min_out = np.iinfo(dtype).min
            max_out = np.iinfo(dtype).max
            # before rescaling, check if there would be an intensity overflow

            if (min_in < min_out) or (max_in > max_out):
                # This condition is important for binary images since we do not want to scale them
                logger.warning(f"To avoid intensity overflow due to convertion to +{dtype.name}+, intensity will be rescaled to the maximum quantization scale")
                # rescale intensity
                data_rescaled = im_src.data * (max_out - min_out) / (max_in - min_in)
                im_dst.data = data_rescaled - (data_rescaled.min() - min_out)

    # change type of data in both numpy array and nifti header
    im_dst.data = getattr(np, dtype.name)(im_dst.data)
    im_dst.hdr.set_data_dtype(dtype)
    return im_dst


def to_dtype(dtype):
    """
    Take a dtypeification and return an np.dtype

    :param dtype: dtypeification (string or np.dtype or None are supported for now)
    :return: dtype or None
    """
    # TODO add more or filter on things supported by nibabel

    if dtype is None:
        return None
    if isinstance(dtype, type):
        if isinstance(dtype(0).dtype, np.dtype):
            return dtype(0).dtype
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, str):
        return np.dtype(dtype)

    raise TypeError(f"data type {dtype.__class__}: {dtype} not understood")


def zeros_like(img, dtype=None):
    """

    :param img: reference image
    :param dtype: desired data type (optional)
    :return: an Image with the same shape and header, filled with zeros

    Similar to numpy.zeros_like(), the goal of the function is to show the developer's
    intent and avoid doing a copy, which is slower than initialization with a constant.

    """
    zimg = Image(np.zeros_like(img.data), hdr=img.hdr.copy())
    if dtype is not None:
        zimg.change_type(dtype)
    return zimg


def empty_like(img, dtype=None):
    """
    :param img: reference image
    :param dtype: desired data type (optional)
    :return: an Image with the same shape and header, whose data is uninitialized

    Similar to numpy.empty_like(), the goal of the function is to show the developer's
    intent and avoid touching the allocated memory, because it will be written to
    afterwards.

    """
    dst = change_type(img, dtype)
    return dst


def spatial_crop(im_src, spec, im_dst=None):
    """
    Crop an image in {0,1,2} dimension(s),
    properly altering the header to not change the physical-logical corresondance.

    :param spec: dict of dim -> [lo,hi] bounds (integer voxel coordinates)
    """

    # Compute bounds
    bounds = [(0, x - 1) for x in im_src.data.shape]
    for k, v in spec.items():
        bounds[k] = v

    bounds_ndslice = tuple([slice(a, b + 1) for (a, b) in bounds])

    bounds = np.array(bounds)

    # Crop data
    new_data = im_src.data[bounds_ndslice]

    # Update header
    #
    # Ref: https://mail.python.org/pipermail/neuroimaging/2017-August/001501.html
    # Given A, we want to find A' that is identical up to the intercept, such
    # that A * [x_0, y_0, z_0, 1]' == A' * [0, 0, 0, 1].
    # Conveniently, A' * [0, 0, 0, 1]' is the fourth row in the affine matrix, so
    # we're done as soon as we calculate the LHS:

    aff = im_src.header.get_best_affine()
    new_aff = aff.copy()
    new_aff[:, [3]] = aff.dot(np.vstack((bounds[:, [0]], [1])))

    new_img = nib.Nifti1Image(new_data, new_aff, im_src.header)

    if im_dst is None:
        im_dst = im_src.copy()

    im_dst.header = new_img.header
    im_dst.data = new_data

    return im_dst


def convert(img: Image, squeeze_data=True, dtype=None):
    """
    """
    if squeeze_data:
        img.data = np.squeeze(img.data)
    if dtype:
        img.change_type(dtype)
    return img


def split_img_data(src_img: Image, dim, squeeze_data=True):
    """
    Split data

    :param src_img: input image.
    :param dim: dimension: 0, 1, 2, 3.
    :return: list of split images
    """

    dim_list = ['x', 'y', 'z', 't']
    data = src_img.data

    # in case input volume is 3d and dim=t, create new axis
    if dim + 1 > len(np.shape(data)):
        data = data[..., np.newaxis]

    # in case splitting along the last dim, make sure to remove the last dim to avoid singleton
    if dim + 1 == len(np.shape(data)):
        if squeeze_data:
            do_reshape = True
        else:
            do_reshape = False
    else:
        do_reshape = False

    # Split data into list
    data_split = np.array_split(data, data.shape[dim], dim)

    # Write each file
    im_out_list = []
    for idx_img, dat in enumerate(data_split):
        im_out = empty_like(src_img)
        if do_reshape:
            im_out.data = dat.reshape(tuple([x for (idx_shape, x) in enumerate(data.shape) if idx_shape != dim]))
        else:
            im_out.data = dat
        im_out.absolutepath = add_suffix(src_img.absolutepath, f"_{dim_list[dim].upper()}{idx_img:04}")
        im_out_list.append(im_out)

    return im_out_list


def concat_warp2d(fname_list, fname_warp3d, fname_dest):
    """
    Concatenate 2d warping fields into a 3d warping field along z dimension. The 3rd dimension of the resulting warping
    field will be zeroed.

    :param fname_list: list of 2d warping fields (along X and Y).
    :param fname_warp3d: output name of 3d warping field
    :param fname_dest: 3d destination file (used to copy header information)
    :return: none
    """

    nx, ny = nib.load(fname_list[0]).shape[0:2]
    nz = len(fname_list)
    warp3d = np.zeros([nx, ny, nz, 1, 3])

    for iz, fname in enumerate(fname_list):
        img = Image(fname)
        warp2d = img.data
        warp3d[:, :, iz, 0, 0] = warp2d[:, :, 0, 0, 0]
        warp3d[:, :, iz, 0, 1] = warp2d[:, :, 0, 0, 1]
        del warp2d

    # save new image
    im_dest = Image(fname_dest)
    im_warp3d = Image(warp3d)
    im_warp3d.copy_affine_from_ref(im_dest)

    # set "intent" code to vector, to be interpreted as warping field
    im_warp3d.header.set_intent('vector', (), '')
    im_warp3d.save(fname_warp3d)


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension.

    :param fname: absolute or relative file name. Example: t2.nii
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii

    Examples:
    .. code:: python

        add_suffix(t2.nii, _mean) -> t2_mean.nii
        add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """
    stem, ext = splitext(fname)
    return stem + suffix + ext


def splitext(fname):
    """
    Split a fname (folder/file + ext) into a folder/file and extension.

    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    """
    dir_, filename = os.path.split(fname)
    for special_ext in ['.nii.gz', '.tar.gz']:
        if filename.endswith(special_ext):
            stem, ext = filename[:-len(special_ext)], special_ext
            break
    else:
        stem, ext = os.path.splitext(filename)

    return os.path.join(dir_, stem), ext


def check_dim(fname, dim_lst=[3]):
    """
    Check if input dimension matches the input dimension requirements specified in the dim list.
    Example: to check if an image is 2D or 3D: check_dim(my_file, dim_lst=[2, 3])
    :param fname:
    :return: True or False
    """
    dim = Image(fname).hdr['dim'][:4]

    if not dim[0] in dim_lst:
        raise ValueError(f"File {fname} has {dim[0]} dimensions! Accepted dimensions are: {dim_lst}.")


def generate_output_file(fname_in, fname_out, squeeze_data=True, verbose=1):
    """
    Copy fname_in to fname_out with a few convenient checks: make sure input file exists, if fname_out exists send a
    warning, if input and output NIFTI format are different (nii vs. nii.gz) convert by unzipping or zipping, and
    display nice message at the end.
    :param fname_in:
    :param fname_out:
    :param verbose:
    :return: fname_out
    """
    path_in, file_in, ext_in = extract_fname(fname_in)
    path_out, file_out, ext_out = extract_fname(fname_out)

    # create output path (ignore if it already exists)
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)

    # if input image does not exist, give error
    if not os.path.isfile(fname_in):
        raise IOError(f"File {fname_in} is not a regular file!")

    # if input and output fnames are the same, do nothing and exit function
    if fname_in == fname_out:
        logger.info("File created: %s", os.path.join(path_out, file_out + ext_out))
        return os.path.join(path_out, file_out + ext_out)

    # if fname_out already exists in nii or nii.gz format
    if os.path.isfile(os.path.join(path_out, file_out + ext_out)):
        logger.warning(f"File {os.path.join(path_out, file_out + ext_out)} already exists. Deleting it..")
        os.remove(os.path.join(path_out, file_out + ext_out))

    if ext_in != ext_out:
        img = Image(fname_in)
        img = convert(img, squeeze_data=squeeze_data)
        img.save(fname_out)
    else:
        # Generate output file without changing the extension
        mv(fname_in, fname_out, verbose=verbose)

    logger.info("File created: %s", os.path.join(path_out, file_out + ext_out))
    return os.path.join(path_out, file_out + ext_out)


def pad_image(im: Image, pad_x_i: int = 0, pad_x_f: int = 0, pad_y_i: int = 0, pad_y_f: int = 0, pad_z_i: int = 0, pad_z_f: int = 0):
    """
    Given an input image, create a copy with specified padding.
    """

    nx, ny, nz, nt, px, py, pz, pt = im.dim
    pad_x_i, pad_x_f, pad_y_i, pad_y_f, pad_z_i, pad_z_f = int(pad_x_i), int(pad_x_f), int(pad_y_i), int(pad_y_f), int(pad_z_i), int(pad_z_f)

    if len(im.data.shape) == 2:
        new_shape = list(im.data.shape)
        new_shape.append(1)
        im.data = im.data.reshape(new_shape)

    # initialize padded_data, with same type as im.data
    padded_data = np.zeros((nx + pad_x_i + pad_x_f, ny + pad_y_i + pad_y_f, nz + pad_z_i + pad_z_f), dtype=im.data.dtype)

    if pad_x_f == 0:
        pad_x_f = None
    elif pad_x_f > 0:
        pad_x_f *= -1
    if pad_y_f == 0:
        pad_y_f = None
    elif pad_y_f > 0:
        pad_y_f *= -1
    if pad_z_f == 0:
        pad_z_f = None
    elif pad_z_f > 0:
        pad_z_f *= -1

    padded_data[pad_x_i:pad_x_f, pad_y_i:pad_y_f, pad_z_i:pad_z_f] = im.data
    im_out = im.copy()
    # TODO: Do not copy the Image(), because the dim field and hdr.get_data_shape() will not be updated properly.
    #   better to just create a new Image() from scratch.
    im_out.data = padded_data  # done after the call of the function
    im_out.absolutepath = add_suffix(im_out.absolutepath, "_pad")

    # adapt the origin in the sform and qform matrix
    new_origin = np.dot(im_out.hdr.get_qform(), [-pad_x_i, -pad_y_i, -pad_z_i, 1])

    im_out.hdr.structarr['qoffset_x'] = new_origin[0]
    im_out.hdr.structarr['qoffset_y'] = new_origin[1]
    im_out.hdr.structarr['qoffset_z'] = new_origin[2]
    im_out.hdr.structarr['srow_x'][-1] = new_origin[0]
    im_out.hdr.structarr['srow_y'][-1] = new_origin[1]
    im_out.hdr.structarr['srow_z'][-1] = new_origin[2]

    return im_out


HEADER_FORMATS = ('sct', 'fslhd', 'nibabel')


def create_formatted_header_string(header, output_format='sct'):
    """
    Generate a string with formatted header fields for pretty-printing.

    :param header: Input header to apply formatting to.
    :param output_format: Specify how to format the output header.
    """
    if output_format == 'sct':
        formatted_fields = _apply_sct_header_formatting(fslhd.generate_nifti_fields(header))
        aligned_string = _align_dict(formatted_fields)
    elif output_format == 'fslhd':
        formatted_fields = fslhd.generate_nifti_fields(header)
        aligned_string = _align_dict(formatted_fields)
    elif output_format == 'nibabel':
        formatted_fields = {k: v[()] for k, v in dict(header).items()}
        aligned_string = _align_dict(formatted_fields, use_tabs=False, delimiter=": ")
    else:
        raise ValueError(f"Can't format header using '{output_format}' format. Available formats: {HEADER_FORMATS}")

    return aligned_string


def _apply_sct_header_formatting(fslhd_fields):
    """
    Tweak fslhd's header fields using SCT's visual preferences.

    :param fslhd_fields: Dict with fslhd's header fields.
    :return modified_fields: Dict with modified header fields.
    """
    modified_fields = {}
    dim, pixdim = [], []
    for key, value in fslhd_fields.items():
        # Replace split dim fields with one-line dim field
        if key.startswith('dim'):
            dim.append(value)
            if key == 'dim7':
                modified_fields['dim'] = dim
        # Replace split pixdim fields with one-line pixdim field
        elif key.startswith('pixdim'):
            pixdim.append(float(value))
            if key == 'pixdim7':
                modified_fields['pixdim'] = pixdim
        # Leave all other fields
        else:
            modified_fields[key] = value

    return modified_fields


def _align_dict(dictionary, use_tabs=True, delimiter=""):
    """
    Create a string with aligned padding from a dict's keys and values.

    :param dictionary: Variable of type dict.
    :param use_tabs: Whether to use tabs instead of spaces for padding.

    :return: String containing padded dict key/values.
    """
    len_max = max([len(str(name)) for name in dictionary.keys()]) + 2
    out = []
    for k, v in dictionary.items():
        if use_tabs:
            len_max = int(8 * round(float(len_max)/8))  # Round up to the nearest 8 to align with tab stops
            padding = "\t" * math.ceil((len_max - len(k))/8)
        else:
            padding = " " * (len_max - len(k))
        out.append(f"{k}{padding}{delimiter}{v}")
    return '\n'.join(out)


def compute_cross_corr_3d(image: Image, coord, xrange=list(range(-10, 10)), xshift=10, yshift=10, zshift=10):
    """
    Compute cross-correlation between image and its mirror using a sliding window in R-L direction to find the image symmetry and adjust R-L coordinate.
    Use a sliding window of 20x20x20 mm by default.
    :param image: image in RPI orientation
    :param coord: 3x1 array: coordinate where to start slidding window (in RPI)
    :param xrange:
    :param xshift:
    :param yshift:
    :param zshift:

    :return: R-L coordinate of the image symmetry
    """
    from spinalcordtoolbox.math import correlation

    nx, ny, nz, _, px, py, pz, _ = image.dim
    x, y, z = coord
    # initializations
    I_corr = np.zeros(len(xrange))
    allzeros = 0
    # current_z = 0
    ind_I = 0
    # Adjust parameters with physical dimensions
    xrange = [int(item//px) for item in xrange]
    xshift = int(xshift//px)
    yshift = int(yshift//py)
    zshift = int(zshift//pz)
    for ix in xrange:
        # if pattern extends towards left part of the image, then crop and pad with zeros
        if x + ix + 1 + xshift > nx:
            padding_size = x + ix + xshift + 1 - nx
            src = image.data[x + ix - xshift: x + ix + xshift + 1 - padding_size,
                             y - yshift:y + yshift + 1,
                             z - zshift: z + zshift + 1]
            src = np.pad(src, ((0, 0), (0, 0), (0, padding_size)), 'constant',
                         constant_values=0)
        # if pattern extends towards right part of the image, then crop and pad with zeros
        elif x + ix - xshift < 0:
            padding_size = abs(ix - xshift)
            src = image.data[x + ix - xshift + padding_size: x + ix + xshift + 1,
                             y - yshift:y + yshift + 1,
                             z - zshift: z + zshift + 1]

            src = np.pad(src, ((0, 0), (0, 0), (padding_size, 0)), 'constant',
                         constant_values=0)
        else:
            src = image.data[x + ix - xshift: x + xshift + ix + 1,
                             y - yshift:y + yshift + 1,
                             z - zshift: z + zshift + 1]
        target = src[::-1, :, :]  # Mirror of src (in R-L direction)
        # convert to 1d
        src_1d = src.ravel()
        target_1d = target.ravel()
        # check if src_1d contains at least one non-zero value
        if (src_1d.size == target_1d.size) and np.any(src_1d):
            I_corr[ind_I] = correlation(src_1d, target_1d)
        else:
            allzeros = 1
        ind_I = ind_I + 1
    if allzeros:
        logger.warning('Data contained zero. We probably hit the edge of the image.')
    if np.any(I_corr):
        # if I_corr contains at least a non-zero value
        ind_peak = [i for i in range(len(I_corr)) if I_corr[i] == max(I_corr)][0]  # index of max along x
        logger.info('.. Peak found: x=%s (correlation = %s)', xrange[ind_peak], I_corr[ind_peak])
        # TODO (maybe) check if correlation is high enough compared to previous R-L coord
    else:
        # if I_corr contains only zeros
        logger.warning('Correlation vector only contains zeros.')
    # Change adjust rl_coord
    logger.info('R-L coordinate adjusted from %s to  %s)', x, x + xrange[ind_peak])
    return x + xrange[ind_peak]


def stitch_images(im_list: Sequence[Image], fname_out: str = 'stitched.nii.gz', verbose: int = 1) -> Image:
    """
    Stitch two (or more) images utilizing the C++-precompiled binaries of Biomedia-MIRA's stitching toolkit
    (https://github.com/biomedia-mira/stitching) by placing a system call.

    :param im_list: list of Image objects to stitch
    :param fname_out: filename for stitched output image
    :param verbose: adjusts the verbosity of the logging
    :return: An Image object containing the stitched data array
    """
    # preserve original orientation (we assume it's consistent among all images)
    orig_ornt = im_list[0].orientation

    # reorient input files and save them to a temp directory
    path_tmp = tmp_create(basename="stitch-images")
    fnames_in = []
    for im_in in im_list:
        temp_file_path = os.path.join(path_tmp, os.path.basename(im_in.absolutepath))
        im_in_rpi = change_orientation(im_in, 'RPI')
        im_in_rpi.save(temp_file_path, verbose=verbose)
        fnames_in.append(temp_file_path)

    # C++ stitching module by Glocker et al. uses the first image as reference image
    # and allocates an array (to be filled by subsequent images along the z-axis)
    # based on the dimensions (x,y) of the reference image.
    # As subsequent images are padded to the x-/y- dimensions of the reference image,
    # it is important to use the image with the largest dimensions as the first
    # argument to the input of the C++ binary, to ensure the images are not cropped.

    # order fs_names in descending order based on dimensions (largest -> smallest)
    fnames_in_sorted = sorted(fnames_in, key=lambda fname: max(Image(fname).dim), reverse=True)

    # ensure that a tmp_path is used for the output of the stitching binary, since sct_image will re-save the image
    fname_out = os.path.join(path_tmp, os.path.basename(fname_out))

    cmd = ['isct_stitching', '-i'] + fnames_in_sorted + ['-o', fname_out, '-a']
    status, output = run_proc(cmd, verbose=verbose, is_sct_binary=True)
    if status != 0:
        raise RuntimeError(f"Subprocess call to `isct_stitching` returned exit code {status} along with the following "
                           f"output:\n{output}")

    # reorient the output image back to the original orientation of the input images
    im_out = change_orientation(Image(fname_out), orig_ornt)

    return im_out


def generate_stitched_qc_images(ims_in: Sequence[Image], im_out: Image) -> Tuple[Image, Image]:
    """
    Pad input and output images to the same dimensions, so that a QC report can compare between the output of
    'stitching', vs. the naive output of concatenating the input images together.

    :param ims_in: A sequence of Image objects (i.e. the input images that were stitched).
    :param im_out: An Image object (i.e. the output stitched image).
    :return: Two Image objects with the same dimensions, such that they can be toggled back and forth in a QC report:
               1. A naive concatenation of `ims_in` (so that the images can be displayed side by side)
               2. A padded version of `im_out` (so that it matches the dimensions of the naive concatenation)
    """
    # Work with copies of the images to avoid mutating the original output
    ims_in = deepcopy(ims_in)
    im_out = deepcopy(im_out)

    # Ensure all images are in RPI orientation (since we make the assumption that that (x,y,z) = (LR,AP,SI))
    for im in list(ims_in) + [im_out]:
        im.change_orientation("RPI")

    # find the max shape of all input images
    shape_max = [max(im.data.shape[0] for im in ims_in),
                 max(im.data.shape[1] for im in ims_in),
                 max(im.data.shape[2] for im in ims_in)]

    # pad any input images that are smaller than the max [x,y] shape
    # (the stitching tool can handle mismatched [x,y] image shapes natively, but we have to manage it ourselves)
    for im in ims_in:
        # the images get concatenated in the z direction,
        # so we only need to pad in the x, y directions
        x_diff = shape_max[0] - im.data.shape[0]
        y_diff = shape_max[1] - im.data.shape[1]
        if (x_diff, y_diff) != (0, 0):
            # note that (diff // 2) + ((diff + 1) // 2) == diff
            im.data = np.pad(im.data, [
                [x_diff // 2, (x_diff + 1) // 2],
                [y_diff // 2, (y_diff + 1) // 2],
                [0, 0],
            ])

    # create a 1-voxel blank image, to be used to create a gap between each input image to distinguish between them
    im_blank = Image([shape_max[0], shape_max[1], 1])

    # create a naively-stitched (RPI) image for comparison in QC report
    # NB: we reverse the list of images because numpy's origin location (bottom) is different than nibabel's (top)
    im_concat_list = [im_blank] * (2*len(ims_in) - 1)  # Preallocate a list of blank spacer images
    im_concat_list[::2] = reversed(ims_in)             # Assign so that: [im_in1, im_blank, im_in2, im_blank ...]
    im_concat = concat_data(im_concat_list, dim=2)     # Concatenate the input images and spacer images together

    # We assume that the [x,y] dimensions match for both of the two QC images
    if im_concat.data.shape[0:2] != im_out.data.shape[0:2]:
        raise ValueError(f"Mismatched image dimensions: {im_concat.data.shape[0:2]} != {im_out.data.shape[0:2]}")

    # However, we can't assume that the [z] dimensions match, because concatenating and stitching produce very
    # different results (lengthwise). So, we pad the smaller image to make the dimensions match.
    z_max = max(im_out.data.shape[2], im_concat.data.shape[2])
    for im in [im_out, im_concat]:
        z_diff = z_max - im.data.shape[2]
        if z_diff > 0:
            im.data = np.pad(im.data, [
                [0, 0],
                [0, 0],
                [z_diff // 2, (z_diff + 1) // 2],
            ])

    # Double-check that the shapes are identical (which is a necessary condition for toggling images in QC reports)
    assert im_concat.data.shape == im_out.data.shape

    return im_concat, im_out


def check_image_kind(img):
    """
    Identify the image as one of 4 image types (represented as one of 4 strings):

        - 'seg': Binary segmentation (0/1)
        - 'softseg': Nonbinary segmentation in the range [0, 1], where 0 and 1 are the majority of values
        - 'seg-labeled': Nonbinary, whole values (0, 1, 2...) where 0 is the majority of values
        - 'anat': Any other image

    Useful for determining A) which colormap should be applied to an image, or
                           B) which interpolation can be safely used on a given image

    Output strings should be compatible with display_viewer_syntax() function.
    """
    # Count the unique values in the image
    unique, counts = np.unique(np.round(img.data, decimals=1), return_counts=True)
    unique, counts = unique[np.argsort(counts)[::-1]], counts[np.argsort(counts)[::-1]]  # Sort by counts
    # This heuristic helps to detect binary and soft segmentations
    idx_zero = np.where(unique == 0.0)[0]
    idx_ones = np.where(unique == 1.0)[0]
    binary_percentage = ((counts[idx_zero[0]] if idx_zero.size > 0 else 0) +
                         (counts[idx_ones[0]] if idx_ones.size > 0 else 0)) / np.sum(counts)
    # This heuristic helps to distinguish between PSIR images and label images (2-10% zero vs. 99% zero)
    is_whole_only = np.equal(np.mod(unique, 1), 0).all()
    zero_most_common = float(unique[0]) == 0.0
    zero_percentage = np.sum(counts[0]) / np.sum(counts)
    if binary_percentage == 1.0:
        return 'seg'
    if 0.0 <= min(unique) <= max(unique) <= 1.0 and binary_percentage > 0.95:
        return 'softseg'
    if is_whole_only and zero_most_common and zero_percentage > 0.50:
        return 'seg-labeled'
    else:
        return 'anat'
