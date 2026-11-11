"""
Quality Control report generator

Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
import datetime
from hashlib import md5
import importlib.resources
import itertools as it
import json
import logging
import math
from pathlib import Path
import shutil
from typing import Optional, Sequence

import numpy as np
import skimage.exposure
from scipy import ndimage

from spinalcordtoolbox.centerline.core import get_centerline, ParamCenterline
from spinalcordtoolbox.cropping import ImageCropper, BoundingBox
from spinalcordtoolbox.image import Image, rpi_slice_to_orig_orientation
import spinalcordtoolbox.reports
from spinalcordtoolbox.reports.assets.py import refresh_qc_entries
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.utils.shell import display_open
from spinalcordtoolbox.utils.sys import __version__, list2cmdline, LazyLoader, __sct_dir__
from spinalcordtoolbox.utils.fs import mutex
from spinalcordtoolbox.aggregate_slicewise import Metric

pd = LazyLoader("pd", globals(), "pandas")
mpl_plt = LazyLoader("mpl_plt", globals(), "matplotlib.pyplot")
mpl_figure = LazyLoader("mpl_figure", globals(), "matplotlib.figure")
mpl_axes = LazyLoader("mpl_axes", globals(), "matplotlib.axes")
mpl_cm = LazyLoader("mpl_cm", globals(), "matplotlib.cm")
mpl_colors = LazyLoader("mpl_colors", globals(), "matplotlib.colors")
mpl_backend_agg = LazyLoader("mpl_backend_agg", globals(), "matplotlib.backends.backend_agg")
mpl_patheffects = LazyLoader("mpl_patheffects", globals(), "matplotlib.patheffects")
mpl_patches = LazyLoader("mpl_patches", globals(), "matplotlib.patches")
mpl_collections = LazyLoader("mpl_collections", globals(), "matplotlib.collections")
nib_orientations = LazyLoader("nib_orientations", globals(), "nibabel.orientations")

logger = logging.getLogger(__name__)

# Clarify some constants that will control the width of the output images
# (the height is automatically adjusted based on the aspect ratio of the image)
# Notes:
#   - This shouldn't ever need to be changed, since "target width" is related
#     to the design of the QC report interface. The images are currently rendered
#     at 1060px wide, so we should match that in the mosaic arrays we generate.
#     NOTE: This will result in arrays smaller than 1060px wide, since mosaic
#     grid wrapping won't exceed this value. But, a smaller visible mosaic is
#     worth the trade-off of pixel accuracy when rendering the final image.
#   - `matplotlib` uses inches/DPI to define the canvas. But, given we want
#     a fixed output size, we can choose some arbitrary values for both.
#     Presumably, choosing a different DPI value (and thus a different canvas
#     size in inches) wouldn't change the output image at all. (Is this true?)
TARGET_WIDTH_PIXL = 1060
DPI = 300
TARGET_WIDTH_INCH = TARGET_WIDTH_PIXL / DPI


@contextmanager
def create_qc_entry(
    path_input: Path,
    path_qc: Path,
    command: str,
    cmdline: str,
    plane: str,
    dataset: Optional[str],
    subject: Optional[str],
    contrast: Optional[str],
    image_extension: str = 'png',
) -> AbstractContextManager[dict[str, Path]]:
    """
    Generate a new QC report entry.

    This context manager yields a dict of two paths, to be used for the QC report images:
    'path_background_img': the path to `background_img.{image_extension}`, and
    'path_overlay_img': the path to `overlay_img.{image_extension}`.

    The body of the `with` block should create these two image files.
    When the `with` block exits, the QC report is updated, with proper file synchronization.
    """
    if plane not in ['Axial', 'Sagittal', 'Ortho']:
        raise ValueError(f'Invalid plane: {plane!r}')

    logger.info('\n*** Generating Quality Control (QC) html report ***')
    mod_date = datetime.datetime.now()

    # Not quite following BIDS convention, we derive the value of
    # dataset, subject, contrast from the input file by splitting it as
    # {dataset}/{subject}/{contrast}/filename
    if dataset is None:
        dataset = path_input.parent.parent.parent.name
    if subject is None:
        subject = path_input.parent.parent.name
    if contrast is None:
        contrast = path_input.parent.name
    timestamp = mod_date.strftime('%Y_%m_%d_%H%M%S.%f')

    # Make sure the image directory exists
    path_img = path_qc / dataset / subject / contrast / command / timestamp
    path_img.mkdir(parents=True, exist_ok=True)

    # Ask the caller to generate the image files for the entry
    imgs_to_generate = {
        'path_background_img': path_img / f'background_img.{image_extension}',
        'path_overlay_img':  path_img / f'overlay_img.{image_extension}',
    }
    yield imgs_to_generate
    # Double-check that the images were generated during the 'with:' block
    for img_type, path in imgs_to_generate.items():
        if not path.exists():
            raise FileNotFoundError(f"Required QC image '{img_type}' was not found at the expected path: '{path}'")

    # Use mutex to ensure that we're only generating shared QC assets using one process at a time
    realpath = path_qc.resolve()
    with mutex(f"sct_qc-{realpath.name}-{md5(str(realpath).encode('utf-8')).hexdigest()}"):
        # Create a json file for the new QC report entry
        path_json = path_qc / '_json'
        path_json.mkdir(parents=True, exist_ok=True)
        path_result = path_json / f'qc_{timestamp}.json'
        with path_result.open('w') as file_result:
            json.dump({
                'path': str(Path.cwd()),
                'cmdline': cmdline,
                'command': command,
                'sctVersion': __version__,
                'dataset': dataset,
                'subject': subject,
                'contrast': contrast,
                'inputFile': path_input.name,
                'plane': plane,
                'backgroundImage': str(imgs_to_generate['path_background_img'].relative_to(path_qc)),
                'overlayImage': str(imgs_to_generate['path_overlay_img'].relative_to(path_qc)),
                'date': mod_date.strftime("%Y-%m-%d %H:%M:%S.%f"),
                'rank': '',
                'qc': '',
            }, file_result, indent=1)

        # Copy any missing QC assets
        path_qc.mkdir(parents=True, exist_ok=True)
        path_assets = importlib.resources.files(spinalcordtoolbox.reports) / 'assets'
        shutil.copytree(path_assets, path_qc, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '__init__.py'))
        # Inject the JSON QC entries into the index.html file
        path_index_html = refresh_qc_entries.main(path_qc)

    logger.info('Successfully generated the QC results in %s', str(path_result))
    display_open(file=str(path_index_html), message="To see the results in a browser")


def reorient_grid(
    shape: tuple[int, int, int],  # image data array shape
    affine: np.ndarray,  # image affine, of shape (4, 4)
    orientation: str,  # for example, "SAL"
) -> tuple[
    tuple[int, int, int],  # reoriented shape
    np.ndarray,  # reoriented affine, of shape (4, 4)
]:
    """
    Reorient a nifti sampling grid to the given orientation.
    """
    opposite = dict(zip("RASLPI", "LPIRAS"))  # SCT and nibabel use opposite conventions
    ornt = nib_orientations.ornt_transform(
        nib_orientations.io_orientation(affine),
        nib_orientations.axcodes2ornt(tuple(opposite[char] for char in orientation))
    )

    new_shape = [0, 0, 0]
    for axis, new_axis in enumerate(ornt[:, 0]):
        new_shape[int(new_axis)] = shape[axis]

    new_affine = np.dot(affine, nib_orientations.inv_ornt_aff(ornt, shape))

    return tuple(new_shape), new_affine


def rescale_grid(
    shape: tuple[int, int, int],  # image data array shape
    affine: np.ndarray,  # image affine, of shape (4, 4)
    resolution: list[float | None],  # desired resolution for each axis
) -> tuple[
    tuple[int, int, int],  # rescaled shape
    np.ndarray,  # rescaled affine, of shape (4, 4)
]:
    """
    Rescale a nifti sampling grid to the given resolution for each axis.

    No rescaling is done for an axis if the requested resolution is `None`.
    """
    # Make a copy of the input before modifying it
    shape = list(shape)
    affine = np.array(affine)

    old_resolution = np.linalg.norm(affine[:, :3], axis=0)

    for axis, (old_res, new_res) in enumerate(zip(old_resolution, resolution)):
        if new_res is not None:
            affine[:, axis] *= new_res / old_res
            # Round up to make sure we cover the original FOV entirely
            shape[axis] = math.ceil(shape[axis] * old_res / new_res)

    return tuple(shape), affine


@dataclass(frozen=True)
class SlicingSpec:
    """
    Specs for extracting one or more 2D slices in the same way from multiple 3D images.

    affine:
        A numpy array of shape=(4, 4) and dtype=np.float64, as returned by
        `nib.Nifti1Header.get_best_affine()`. This matrix gives the conversion
        between voxel coordinates (i, j, k), which are non-negative integers,
        and physical coordinates (x, y, z), which are floating point numbers:

            [ x ]   [ * * * | * ]   [ i ]
            [ y ]   [ * * * | * ]   [ j ]
            [ z ] = [ * * * | * ] @ [ k ]
            [---]   [-------+---]   [---]
            [ 1 ]   [ 0 0 0 | 1 ]   [ 1 ]

        The first column gives the step size and direction (in physical space)
        for taking a 1-voxel step from i to i+1. The second and third columns
        are the same for j to j+1, or k to k+1. The last column gives the
        physical coordinates of the (0, 0, 0) voxel. The last row is fixed.

        The 2D slices to be extracted are all parallel to the (j, k) plane.

    offsets:
        An ordered dict of slice labels and slice offsets for the 2D slices
        to be extracted. Each slice offset is a numpy array of shape=(3,)
        and dtype=np.float64, and gives voxel coordinates (i, j, k) for a
        corner of a slice to be extracted. Even though they are in voxel space,
        these coordinates are allowed to be negative or fractional.

        The insertion order of items in the dict controls the position of each
        slice in the generated mosaic.

    shape:
        The (height, width) size of each 2D slice to be extracted. The height
        is measured in voxels in the j direction, and the width is measured
        in voxels in the k direction.

    axis_labels:
        Two pairs of strings used to display orientation labels:

            ((up == -j direction, down == +j direction),
             (left == -k direction, right == +k direction))

        Used for the top-left tile in a mosaic.
    """
    # ----------------------------------------------------------------
    # Dataclass fields
    # ----------------------------------------------------------------
    # The default @dataclass __init__ method, used internally by the
    # SlicingSpec factory functions, always assigns a value to these.

    affine: np.ndarray  # shape=(4, 4)
    offsets: dict[str, np.ndarray]  # each shape=(3,), insertion order is important
    shape: tuple[int, int]  # 2D shape of all output slices
    axis_labels: tuple[tuple[str, str], tuple[str, str]]  # ((top, bottom), (left, right))

    # ----------------------------------------------------------------
    # Factory functions
    # ----------------------------------------------------------------
    # This is the normal way for users of this class to create
    # instances. They are @staticmethod, so they are called as:
    # SlicingSpec.full_axial(...) etc.

    @staticmethod
    def full_axial(img: Image, p_resample: float) -> 'SlicingSpec':
        """
        A slicing spec to extract full axial slices.

        The field of view is given by `img`.
        The number of slices is the same as `img` in the S-I axis,
        but the other two axes are resampled to `p_resample` mm.
        The slice labels correspond to the original orientation of `img`,
        but the output slices are always ordered and oriented in "SAL".
        """
        return SlicingSpec.full_oriented(img, "SAL", p_resample)

    @staticmethod
    def full_sagittal(img: Image, p_resample: float) -> 'SlicingSpec':
        """
        A slicing spec to extract full sagittal slices.

        The field of view is given by `img`.
        The number of slices is the same as `img` in the R-L axis,
        but the other two axes are resampled to `p_resample` mm.
        The slice labels correspond to the original orientation of `img`,
        but the output slices are always ordered and oriented in "RSP".
        """
        return SlicingSpec.full_oriented(img, "RSP", p_resample)

    @staticmethod
    def full_oriented(img: Image, orientation: str, p_resample: float) -> 'SlicingSpec':
        """
        A slicing spec to extract full slices in a given orientation.

        The meaning of the `orientation` argument is that:
        - The first axis is perpendicular to the 2D slices produced.
        - The second axis is the "up" direction for the 2D slices produced.
        - The third axis is the "left" direction for the 2D slices produced.

        Each output slice is resampled to `p_resample` mm.

        The slice labels match the original orientation of `img`,
        but their ordering matches the requested orientation.
        """
        # For axis labels.
        opposite = dict(zip("RASLPI", "LPIRAS"))

        # Reorient and rescale.
        shape, affine = img.data.shape, img.affine
        shape, affine = reorient_grid(shape, affine, orientation)
        shape, affine = rescale_grid(shape, affine, [None, p_resample, p_resample])

        # Split up the 3D image shape into a number of slices and a 2D slice shape.
        num_slices, shape = shape[0], shape[1:3]

        # The slice labels in the original orientation of the input image.
        slice_labels = [str(i) for i in range(num_slices)]
        if opposite[orientation[0]] in img.orientation:
            slice_labels.reverse()

        return SlicingSpec(
            affine=affine,
            offsets={
                label: np.array([i, 0, 0], dtype=np.float64)
                for i, label in enumerate(slice_labels)
            },
            shape=shape,
            axis_labels=tuple(
                (letter, opposite[letter])
                for letter in orientation[1:3]
            ),
        )

    # ----------------------------------------------------------------
    # Transformer methods
    # ----------------------------------------------------------------
    # These methods can be called on an existing SlicingSpec, to
    # produce a new SlicingSpec. For example, they can be used to
    # center and crop slices around a segmentation, or to drop some
    # slices entirely.

    def center_patches(
        self: 'SlicingSpec', seg: Image, shape: tuple[int, int] = (30, 30)
    ) -> 'SlicingSpec':
        """
        A re-centered and cropped slicing spec, meant for axial views.

        Each slice is individually centered around the center of mass of the
        given segmentation before being cropped (in both axes).
        """
        # Recenter each slice around its center of mass, and interpolate missing
        # slices. We use a single (N, 3) array so that we can inf_nan_fill.
        offsets = np.array(list(self.offsets.values()))
        slices_seg = self.get_slices(seg, order=1)
        for offset, slice_seg in zip(offsets, slices_seg.values()):
            offset[1:3] += ndimage.center_of_mass(slice_seg)
        inf_nan_fill(offsets[:, 1])
        inf_nan_fill(offsets[:, 2])

        # Take into account the size of the cropping rectangle.
        offsets[:, 1:3] -= (np.array(shape) - 1) / 2

        return SlicingSpec(
            affine=self.affine,
            offsets=dict(zip(self.offsets.keys(), offsets)),
            shape=shape,  # new shape
            axis_labels=self.axis_labels,
        )

    def center_columns(self: 'SlicingSpec', seg: Image) -> 'SlicingSpec':
        """
        A re-centered and cropped slicing spec, meant for sagittal views.

        Each slice is individually centered in the last axis around the center
        of mass of the given segmentation. It is also cropped in the last axis
        to the larger of:
        - 110% of the maximum width of the segmentation, or
        - 50% of the total image.

        Any slices that are more than 2 slices away from the segmentation are
        also dropped.
        """
        # Check which slices contain the segmentation, and compute their width.
        slices_seg = self.get_slices(seg, order=1)
        width = {}
        for i, slice_seg in enumerate(slices_seg.values()):
            [columns] = slice_seg.any(axis=0).nonzero()
            if columns.size != 0:
                width[i] = int(columns.max()) - int(columns.min()) + 1
        if not width:
            raise ValueError("The mask image is empty. Cannot crop using an empty mask. Check the input (e.g. '-qc-seg').")

        # Compute the target slice width.
        shape = (
            self.shape[0],  # the full image height
            math.ceil(max(
                1.1 * max(width.values()),  # 110% of the segmentation width
                0.5 * self.shape[1],  # 50% of the full image width
            )),
        )

        # Drop slices that are far from the segmentation.
        slice_min = max(0, min(width.keys()) - 2)
        slice_max = min(len(slices_seg) - 1, max(width.keys()) + 2)
        offsets_dict = dict(list(self.offsets.items())[slice_min:slice_max+1])

        # Re-center each remaining slice.
        for slice_label, offset in offsets_dict.items():
            offset[2] += ndimage.center_of_mass(slices_seg[slice_label])[1]

        # Take into account the cropping width.
        offsets_array = np.array(list(offsets_dict.values()))
        inf_nan_fill(offsets_array[:, 2])
        offsets_array[:, 2] -= (shape[1] - 1) / 2
        offsets_dict = dict(zip(offsets_dict.keys(), offsets_array))

        return SlicingSpec(
            affine=self.affine,
            offsets=offsets_dict,  # new offsets
            shape=shape,  # new shape
            axis_labels=self.axis_labels,
        )

    def composite_sagittal(self: 'SlicingSpec', seg: Image) -> 'SlicingSpec':
        """
        Construct a composite 2D sagittal image out of axial slices (one
        row from each axial slice), similarly to `sct_flatten_sagittal`.
        (For each axial slice, the center of mass is used to identify the
        midsagittal row. The individual rows can then be stitched together
        to build a composite midsagittal 2D image.)
        """
        # Recenter each slice around its center of mass, and interpolate missing
        # slices. We use a single (N, 3) array so that we can inf_nan_fill.
        offsets = np.array(list(self.offsets.values()))
        slices_seg = self.get_slices(seg, order=1)
        for offset, slice_seg in zip(offsets, slices_seg.values()):
            offset[2] += ndimage.center_of_mass(slice_seg)[1]
        inf_nan_fill(offsets[:, 2])

        return SlicingSpec(
            affine=self.affine,
            offsets=dict(zip(self.offsets.keys(), offsets)),
            shape=(self.shape[0], 1),  # crop to a single line
            axis_labels=self.axis_labels,
        )

    # ----------------------------------------------------------------
    # Consumer methods
    # ----------------------------------------------------------------
    # These methods are called on an existing SlicingSpec to produce
    # some other result.

    def get_slices(
        self: 'SlicingSpec',
        img: Image,  # 3D image to sample from
        order: int,  # interpolation order, can be [0, 1, 2, 3, 4, 5]
    ) -> dict[str, np.ndarray]:  # shape=self.shape
        """
        Get several slices of the same shape from a single image by resampling.

        The offsets are measured in output voxels, and can be fractional.
        """
        # To fill regions outside of the image with zeros,
        # while avoiding sharp cutoffs around the edges.
        mode = "grid-constant"

        voxel_to_voxel = np.linalg.inv(img.affine).dot(self.affine)
        matrix = voxel_to_voxel[:3, :3]
        origin = voxel_to_voxel[:3, 3]

        if order > 1:
            # Doing this prefilter step outside of the loop is crucial for performance
            data_filtered = ndimage.spline_filter(img.data, order=order, mode=mode)
        else:
            data_filtered = img.data.astype(np.float64)
        return {
            slice_label: ndimage.affine_transform(
                data_filtered,
                matrix=matrix,
                offset=(origin + matrix.dot(offset)),
                output_shape=(1, *self.shape),
                order=order,
                mode=mode,
                prefilter=False,
            )[0]
            for slice_label, offset in self.offsets.items()
        }

    def get_mosaic(self: 'SlicingSpec', **kwargs) -> 'Mosaic':
        """
        Get a mosaic that can hold the result of `self.get_slices`.
        """
        return Mosaic(self.shape, list(self.offsets.keys()), self.axis_labels, **kwargs)


class Mosaic:
    """
    Convenience methods for rendering several 2D slices into a grid of images.

    The caller is expected to manipulate `self.fig` and/or `self.ax` directly
    before calling `self.save(path)`.
    """
    canvas: np.ndarray  # The data array for the entire mosaic.
    tiles: dict[str, tuple[slice, slice]]  # The canvas coordinates for each slice label.
    axis_labels: tuple[tuple[str, str], tuple[str, str]]  # ((top, bottom), (left, right))

    fig: 'mpl_figure.Figure'
    ax: 'mpl_axes.Axes'

    def __init__(
        self,
        shape: tuple[int, int],  # (height, width) of each tile
        slice_labels: list[str],
        axis_labels: tuple[tuple[str, str], tuple[str, str]],  # ((top, bottom), (left, right))
        *,  # keyword-only arguments after this
        rect: tuple[float, float, float, float] = (0, 0, 1, 1),  # passed to Figure.add_axes()
        scale: float = 2.5,
    ):
        # Make a canvas of the right size to hold all the tiles.
        num_col = max(math.floor(TARGET_WIDTH_PIXL / scale / shape[1]), 1)
        num_row = math.ceil(len(slice_labels) / num_col)
        self.canvas = np.zeros((num_row * shape[0], num_col * shape[1]))

        # Compute the canvas coordinates for each slice label.
        self.tiles = dict(zip(slice_labels, [
            (slice(x, x + shape[0]), slice(y, y + shape[1]))
            for x in range(0, self.canvas.shape[0], shape[0])
            for y in range(0, self.canvas.shape[1], shape[1])
        ], strict=False))

        # Save the axis labels for displaying later.
        self.axis_labels = axis_labels

        # Initialize the Figure and the Axes for rendering the mosaic later.
        self.fig = mpl_figure.Figure(
            # figsize is (width, height) but canvas shape is (height, width)
            figsize=(self.canvas.shape[1] * scale / DPI, self.canvas.shape[0] * scale / DPI),
            dpi=DPI,
        )
        mpl_backend_agg.FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes(rect)
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)

    def insert_slices(self, slices: dict[str, np.ndarray]):
        """
        Insert 2D slices into the canvas based on their slice labels.

        Note that this doesn't render them to `self.ax`, so that the caller
        can post-process `self.canvas` (for example, with `equalize_histogram`)
        and has full control of `self.ax.imshow(self.canvas, ...)`.
        """
        for slice_label, data in slices.items():
            self.canvas[self.tiles[slice_label]] = data

    def add_labels(self):
        """
        Add axis labels and slice labels to the mosaic figure.

        Axis labels are added to the first tile, and slice labels are added to
        the other tiles.
        """
        # Rendering options for all the labels.
        text_args = dict(
            color='yellow',
            fontsize=4,
            path_effects=[
                mpl_patheffects.Stroke(linewidth=1, foreground='black'),
                mpl_patheffects.Normal(),
            ],
        )
        for tile_num, (slice_label, coords) in enumerate(self.tiles.items()):
            # Get real coordinates from the `slice` objects.
            top, bottom, _ = coords[0].indices(self.canvas.shape[0])
            left, right, _ = coords[1].indices(self.canvas.shape[1])
            if tile_num == 0:
                # Axis labels for the first tile.
                ((top_label, bottom_label), (left_label, right_label)) = self.axis_labels
                self.ax.text((left + right) / 2, top, top_label, ha='center', va='top', **text_args)
                self.ax.text((left + right) / 2, bottom, bottom_label, ha='center', va='bottom', **text_args)
                self.ax.text(left, (top + bottom) / 2, left_label, ha='left', va='center', **text_args)
                self.ax.text(right, (top + bottom) / 2, right_label, ha='right', va='center', **text_args)
            else:
                # Slice labels for the other tiles.
                self.ax.text(left, top, slice_label, ha='left', va='top', **text_args)

    def save(self, path: Path):
        """Save the final figure."""
        self.fig.savefig(str(path), format='png', transparent=True)


def sct_register(
    fname_input: str,
    fname_output: str,
    fname_seg: str,
    command: str,
    argv: Sequence[str],
    path_qc: str,
    dataset: Optional[str],
    subject: Optional[str],
    contrast: Optional[str],
    p_resample: float | None = 0.6,
):
    """
    Generate a QC report for sct_register_multimodal or sct_register_to_template.

    Axial orientation, switch between input and output images.
    """
    if not command:
        raise ValueError("The 'command' argument must be provided for the sct_register QC report, as there are "
                         "multiple registration commands that can generate `sct_register` QC reports.")
    cmdline = [command]
    cmdline.extend(argv)

    with create_qc_entry(
        path_input=Path(fname_input).resolve(),
        path_qc=Path(path_qc),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane='Axial',
        dataset=dataset,
        subject=subject,
        contrast=contrast,
    ) as imgs_to_generate:

        img_input = Image(fname_input)
        img_output = Image(fname_output)
        img_seg = Image(fname_seg)

        slicing_spec = SlicingSpec.full_axial(img_input, p_resample).center_patches(img_seg)

        for img, path in [
            (img_input, imgs_to_generate['path_background_img']),
            (img_output, imgs_to_generate['path_overlay_img']),
        ]:
            mosaic = slicing_spec.get_mosaic()
            mosaic.insert_slices(slicing_spec.get_slices(img, order=2))
            mosaic.canvas = equalize_histogram(mosaic.canvas)
            mosaic.ax.imshow(mosaic.canvas, cmap='gray', interpolation='none')
            mosaic.add_labels()
            mosaic.save(path)


def sct_fmri_compute_tsnr(
    fname_input: str,
    fname_output: str,
    fname_seg: str,
    argv: Sequence[str],
    path_qc: str,
    dataset: Optional[str],
    subject: Optional[str],
    contrast: Optional[str],
    p_resample: float | None = 0.6,
    command: str = 'sct_fmri_compute_tsnr',
):
    """
    Generate a QC report for sct_fmri_compute_tsnr.

    Axial orientation, switch between two input images, with color bar and
    mean value in spinal cord.
    """
    cmdline = [command]
    cmdline.extend(argv)

    with create_qc_entry(
        path_input=Path(fname_input).resolve(),
        path_qc=Path(path_qc),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane='Axial',
        dataset=dataset,
        subject=subject,
        contrast=contrast,
    ) as imgs_to_generate:

        img_input = Image(fname_input)
        img_output = Image(fname_output)
        img_seg = Image(fname_seg)

        slicing_spec = SlicingSpec.full_axial(img_input, p_resample).center_patches(img_seg)

        slices_input = slicing_spec.get_slices(img_input, order=2)
        slices_output = slicing_spec.get_slices(img_output, order=2)

        # Quadratic resampling gives a sharper image, but it may extrapolate
        # beyond the original min/max values. We want the colorbar tick marks
        # to reflect the true range of the original images.
        linear_values = [
            *slicing_spec.get_slices(img_input, order=1).values(),
            *slicing_spec.get_slices(img_output, order=1).values(),
        ]
        vmin = int(np.nanmin(linear_values))
        vmax = int(np.nanmax(linear_values))

        for corner_label, slices, path in [
            ("1", slices_input, imgs_to_generate['path_background_img']),
            ("2", slices_output, imgs_to_generate['path_overlay_img']),
        ]:
            mosaic = slicing_spec.get_mosaic(rect=(0, 0, 0.93, 1))
            mosaic.insert_slices(slices)
            axes_image = mosaic.ax.imshow(
                mosaic.canvas,
                cmap='seismic',
                norm=mpl_colors.Normalize(vmin, vmax),
                interpolation='none',
            )
            colorbar = mosaic.fig.colorbar(
                axes_image,
                cax=mosaic.ax.inset_axes([1.01, 0.07, 0.011, 0.86]),
                orientation='vertical',
                pad=0.01,
                shrink=0.5,
                aspect=1,
                ticks=[vmin, vmax],
            )
            colorbar.ax.tick_params(labelsize=5, length=2, pad=1.7)
            mosaic.ax.text(1.5, 6, corner_label, color='white', size=3.25)
            mosaic.add_labels()
            mosaic.save(path)


def sct_label_vertebrae(
    fname_input: str,
    fname_seg: str,
    argv: Sequence[str],
    path_qc: str,
    dataset: Optional[str],
    subject: Optional[str],
    contrast: Optional[str],
    path_custom_labels: str,
    draw_text: bool = True,
    p_resample: Optional[float] = None,
    offset_text: bool = True,
    command: str = 'sct_label_vertebrae',
):
    """
    Generate a QC report for sct_label_vertebrae.

    Sagittal orientation, wavy single slice, display vertebral labels.
    """
    cmdline = [command]
    cmdline.extend(argv)

    with create_qc_entry(
        path_input=Path(fname_input).resolve(),
        path_qc=Path(path_qc),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane='Sagittal',
        dataset=dataset,
        subject=subject,
        contrast=contrast,
    ) as imgs_to_generate:

        img_input = Image(fname_input)
        img_labels = Image(fname_seg)

        # A version of img_labels with only 0-1 values, for center-of-mass computations.
        img_seg = img_labels.copy()
        img_seg.data = (img_seg.data != 0)

        # Take a single mid-sagittal line from each axial slice to compose a 2D image.
        slicing_spec = SlicingSpec.full_axial(img_input, p_resample).composite_sagittal(img_seg)

        # Quadratic resampling for the actual image.
        slices_input = slicing_spec.get_slices(img_input, order=2)
        data_input = equalize_histogram(np.array([s[:, 0] for s in slices_input.values()]))

        # Nearest-neighbour resampling for the segmentation labels.
        slices_labels = slicing_spec.get_slices(img_labels, order=0)
        data_labels = np.array([s[:, 0] for s in slices_labels.values()])

        # Aspect ratio, since the thickness of axial slices may not be == p_resample.
        p_height = next(
            p for p, letter in zip(
                img_input.dim[4:7],
                img_input.orientation,
            )
            if letter in 'SI'
        )
        p_width = p_resample if p_resample is not None else next(
            p for p, letter in zip(
                img_input.dim[4:7],
                img_input.orientation,
            )
            if letter in 'AP'
        )
        aspect = (data_input.shape[0] * p_height) / (data_input.shape[1] * p_width)

        # Draw the actual image on the background.
        # figsize is (width, height) in inches
        fig = mpl_figure.Figure(figsize=(5, 5*aspect), dpi=100)
        mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(data_input, cmap='gray', interpolation='none', aspect='auto')
        fig.savefig(str(imgs_to_generate['path_background_img']), format='png', transparent=True)

        # Draw the label regions and text in the overlay.
        # figsize is (width, height) in inches
        fig = mpl_figure.Figure(figsize=(5, 5*aspect), dpi=100)
        mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        img = np.rint(np.ma.masked_where(data_labels <= 0, data_labels))
        labels = np.unique(img[np.where(~img.mask)]).astype(int)  # get available labels
        color_list = assign_label_colors_by_groups(labels)
        ax.imshow(
            img,
            cmap=mpl_colors.ListedColormap(color_list),
            interpolation='none',
            alpha=1,
            aspect='auto',
        )

        if draw_text:
            # Get the mapping between voxel values and text labels
            try:
                dict_labels = json.loads(Path(path_custom_labels).read_text())
                if not isinstance(dict_labels, dict):
                    raise ValueError("The JSON file should contain a single dictionary")
                for label_text in dict_labels.values():
                    if not isinstance(label_text, str):
                        raise ValueError(f"Not a text label: {label_text!r}")
                dict_labels = {int(label_num): label_text for label_num, label_text in dict_labels.items()}
            except ValueError as e:
                example = Path(__sct_dir__) / 'spinalcordtoolbox' / 'reports' / 'sct_label_vertebrae_regions.json'
                raise ValueError(f"Invalid format for custom labels, see {example} for an example. ({e})")

            # Add the text labels
            for label_num in labels:
                if label_num in dict_labels:
                    # NB: We need to subtract `min` to convert the label value into an index for the color list
                    label_color = color_list[label_num - labels.min()]
                    # Position the label text
                    y, x = ndimage.center_of_mass(img == label_num)
                    if offset_text:
                        x += img.shape[1] / 25
                    # Draw text with a shadow
                    label_text = dict_labels[label_num]
                    ax.text(x, y, label_text, color=label_color, clip_on=True).set_path_effects(
                        [mpl_patheffects.Stroke(linewidth=2, foreground='black'), mpl_patheffects.Normal()]
                    )

        fig.savefig(str(imgs_to_generate['path_overlay_img']), format='png', transparent=True)


def sct_label_utils(
    fname_input: str,
    fname_seg: str,
    argv: Sequence[str],
    path_qc: str,
    dataset: Optional[str],
    subject: Optional[str],
    contrast: Optional[str],
    p_resample: Optional[float] = None,
    command: str = 'sct_label_utils',
):
    """
    Generate a QC report for sct_label_utils.

    Sagittal orientation, wavy single slice, with posterior labels.
    """
    cmdline = [command]
    cmdline.extend(argv)

    with create_qc_entry(
        path_input=Path(fname_input).resolve(),
        path_qc=Path(path_qc),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane='Sagittal',
        dataset=dataset,
        subject=subject,
        contrast=contrast,
    ) as imgs_to_generate:

        img_input = Image(fname_input)
        img_labels = Image(fname_seg)

        # A version of img_labels with only 0-1 values, for center-of-mass computations.
        img_seg = img_labels.copy()
        img_seg.data = (img_seg.data != 0)

        # Take a single mid-sagittal line from each axial slice to compose a 2D image.
        slicing_spec = SlicingSpec.full_axial(img_input, p_resample).composite_sagittal(img_seg)

        # Quadratic resampling for the actual image.
        slices_input = slicing_spec.get_slices(img_input, order=2)
        data_input = equalize_histogram(np.array([s[:, 0] for s in slices_input.values()]))

        # Nearest-neighbour resampling for the segmentation labels.
        slices_labels = slicing_spec.get_slices(img_labels, order=0)
        data_labels = np.array([s[:, 0] for s in slices_labels.values()])

        # Aspect ratio, since the thickness of axial slices may not be == p_resample.
        p_height = next(
            p for p, letter in zip(
                img_input.dim[4:7],
                img_input.orientation,
            )
            if letter in 'SI'
        )
        p_width = p_resample if p_resample is not None else next(
            p for p, letter in zip(
                img_input.dim[4:7],
                img_input.orientation,
            )
            if letter in 'AP'
        )
        aspect = (data_input.shape[0] * p_height) / (data_input.shape[1] * p_width)

        # Draw the actual image on the background.
        # figsize is (width, height) in inches
        fig = mpl_figure.Figure(figsize=(5, 5*aspect), dpi=100)
        mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(data_input, cmap='gray', interpolation='none', aspect='auto')
        fig.savefig(str(imgs_to_generate['path_background_img']), format='png', transparent=True)

        # Draw the label voxels and text in the overlay.
        # figsize is (width, height) in inches
        fig = mpl_figure.Figure(figsize=(5, 5*aspect), dpi=100)
        mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax.imshow(np.full_like(data_labels, np.nan), cmap='gray', alpha=0, aspect='auto')
        non_null_vox = np.nonzero(data_labels > 0)
        horiz_offset = data_labels.shape[1] / 50
        for y, x, val in zip(*non_null_vox, data_labels[non_null_vox]):
            ax.plot(x, y, 'o', color='lime', markersize=5)
            ax.text(
                x + horiz_offset, y, str(round(val)), color='lime',
                fontsize=15, verticalalignment='center', clip_on=True,
            ).set_path_effects([
                mpl_patheffects.Stroke(linewidth=2, foreground='black'),
                mpl_patheffects.Normal(),
            ])

        fig.savefig(str(imgs_to_generate['path_overlay_img']), format='png', transparent=True)


def add_slice_numbers(ax, num_slices, radius, margin: int = 2, reverse=False):
    """
    Overlay slice indices onto an Axial mosaic.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing a single imshow of the mosaic.
    num_slices : int
        Total number of axial slices (dim 0 of the original volume).
    radius : (int, int)
        The size (in pixels) of the x/y radii of each square cell in the mosaic.
    margin : int
        Margin (in pixels) from the top-left corner of each cell to place the text.
    reverse : bool
        Whether to reverse the slice numbering (for SI-flipped images). If false, slice 0 is
        at the top-left, otherwise slice 0 is at the bottom-right.
    """
    # Get the mosaic array we just plotted
    img_arr = ax.get_images()[0].get_array()
    patch_size = [2*r for r in radius]
    n_cols = int(img_arr.shape[1] // patch_size[1])
    slice_range = list(range(1, num_slices))  # skip 0
    for i in slice_range:
        row = i // n_cols
        col = i % n_cols
        # top-left inside each tile
        x = col * patch_size[1] + margin
        y = row * patch_size[0] + margin
        # if there was an SI flip, then the label should be reverse
        label = str(i) if not reverse else str((num_slices - 1) - i)
        txt = ax.text(x, y, label, ha='left', va='top', color='yellow', fontsize=4)
        # give it a thin black outline for readability
        txt.set_path_effects([
            mpl_patheffects.Stroke(linewidth=1, foreground='black'),
            mpl_patheffects.Normal()
        ])


def sct_register_multimodal(
    fname_input: str,
    fname_output: str,
    fname_seg: str,
    argv: Sequence[str],
    path_qc: str,
    dataset: Optional[str],
    subject: Optional[str],
    contrast: Optional[str],
    command: str = 'sct_register_multimodal',
):
    """
    Generate a QC report for sct_register_multimodal.
    """
    cmdline = [command]
    cmdline.extend(argv)

    # Axial orientation, switch between two input images
    with create_qc_entry(
        path_input=Path(fname_input).resolve(),
        path_qc=Path(path_qc),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane='Axial',
        dataset=dataset,
        subject=subject,
        contrast=contrast,
    ) as imgs_to_generate:

        # Resample images slice by slice
        p_resample = 0.6
        logger.info('Resample images to %fx%f mm', p_resample, p_resample)
        img_input = Image(fname_input).change_orientation('SAL')
        img_input = resample_nib(
            image=img_input,
            new_size=[img_input.dim[4], p_resample, p_resample],
            new_size_type='mm',
            interpolation='spline',
        )
        img_output = resample_nib(
            image=Image(fname_output).change_orientation('SAL'),
            image_dest=img_input,
            interpolation='spline',
        )
        img_seg = resample_nib(
            image=Image(fname_seg).change_orientation('SAL'),
            image_dest=img_input,
            interpolation='linear',
        )
        img_seg.data = (img_seg.data > 0.5) * 1

        # Each slice is centered on the segmentation
        logger.info('Find the center of each slice')
        centers = np.array([ndimage.center_of_mass(slice) for slice in img_seg.data])
        inf_nan_fill(centers[:, 0])
        inf_nan_fill(centers[:, 1])

        # Generate the first QC report image
        img = equalize_histogram(mosaic(img_input, centers))

        # Fix the width to a specific size, and vary the height based on how many rows there are.
        size_fig = [TARGET_WIDTH_INCH, TARGET_WIDTH_INCH * img.shape[0] / img.shape[1]]

        fig = mpl_figure.Figure()
        fig.set_size_inches(*size_fig, forward=True)
        mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(img, cmap='gray', interpolation='none', aspect=1.0)
        add_orientation_labels(ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_path = str(imgs_to_generate['path_background_img'])
        logger.debug('Save image %s', img_path)
        fig.savefig(img_path, format='png', transparent=True, dpi=DPI)

        # Generate the second QC report image
        img = equalize_histogram(mosaic(img_output, centers))
        fig = mpl_figure.Figure()
        fig.set_size_inches(*size_fig, forward=True)
        mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1), label='0')
        ax.imshow(img, cmap='gray', interpolation='none', aspect=1.0)
        add_orientation_labels(ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_path = str(imgs_to_generate['path_overlay_img'])
        logger.debug('Save image %s', img_path)
        fig.savefig(img_path, format='png', transparent=True, dpi=DPI)


def sct_process_segmentation(
        fname_input: str,
        fname_seg: str,
        metrics: dict[str, Metric],
        argv: Sequence[str],
        path_qc: str,
        dataset: Optional[str],
        subject: Optional[str],
        contrast: Optional[str],
        angle_qc: Optional[bool] = False,
        angle_type: Optional[str] = 'angle_hog',
        command: str = 'sct_process_segmentation'
):
    """
    Generate one (optionally two) QC report entries for sct_process_segmentation:
      1. AP split overlay: per-slice cyan/yellow lines showing anterior (length_anterior) and
         posterior (length_posterior) cord lengths from the cord center of mass.
      2. (Optional) Angle overlay: a line per slice showing the estimated cord orientation angle
         (obtained using spinalcordtoolbox.registration.algorithms.find_angle_hog when angle_type='angle_hog').
    """
    cmdline = [command]
    cmdline.extend(argv)

    # Load the input images
    img_input = Image(fname_input).change_orientation('SAL')
    img_seg = Image(fname_seg).change_orientation('SAL')
    # NOTE: We don't resample the image to 0.6mm iso as the angle_hog was computed on the original image
    # TODO: consider slice by slice resampling (used e.g. in sct_deepseg_sagittal)

    # Each slice is centered on the segmentation
    logger.info('Find the center of each slice')
    centers = np.array([ndimage.center_of_mass(slice) for slice in img_seg.data])
    inf_nan_fill(centers[:, 0])
    inf_nan_fill(centers[:, 1])

    # If seg is available, use it to generate the radius
    radius = get_max_axial_radius(img_seg) if fname_seg else (15, 15)
    scale = 3.2  # we can consider increasing the number, then the mosaic can be zoomed in/out using the "Full size" button in the QC report

    # Preprocess the input anat image to get the background image (used for both QC reports)
    img_bg = equalize_histogram(mosaic(img_input, centers, radius, scale=scale))
    # Fix the width to a specific size, and vary the height based on how many rows there are.
    size_fig = [TARGET_WIDTH_INCH, TARGET_WIDTH_INCH * img_bg.shape[0] / img_bg.shape[1]]

    # Preprocess the input seg image to get the base image used for the overlay
    img_temp = img_seg.copy()
    img_temp.data = np.full_like(img_seg.data, np.nan)
    # Create empty axes for the mosaic
    img = mosaic(img_temp, centers, radius=radius, scale=scale)
    img = np.ma.masked_less_equal(img, 0)
    img.set_fill_value(0)

    # QC entry 1: anterior (cyan) / posterior (yellow) cord lengths with RL axis (white).
    # Variables from the first with-block (img_input, img_temp, img_bg, centers, radius, scale,
    # size_fig) remain in scope in Python and are reused here to avoid redundant computation.
    # Reference: Kang et al. J Clin Med 2023. https://doi.org/10.3390/jcm12124111 (Fig. 1)
    with create_qc_entry(
        path_input=Path(fname_input).resolve(),
        path_qc=Path(path_qc),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane='Axial',
        dataset=dataset,
        subject=subject,
        contrast=contrast,
    ) as imgs_to_generate:
        # Generate the first QC report image - background image
        fig = mpl_figure.Figure()
        fig.set_size_inches(*size_fig, forward=True)
        mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(img_bg, cmap='gray', interpolation='none', aspect=1.0)
        add_orientation_labels(ax, radius=tuple(r for r in radius))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_path = str(imgs_to_generate['path_background_img'])
        logger.debug('Save image %s', img_path)
        fig.savefig(img_path, format='png', transparent=True, dpi=DPI)

        # Generate the second QC report image - overlay image with lines for anterior/posterior length
        fig = mpl_figure.Figure()
        fig.set_size_inches(*size_fig, forward=True)
        mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(img, aspect=1.0)
        # px_AP: pixel size in mm for the AP axis of the SAL-oriented image (dim[5])
        add_ap_split_lines(ax, img_temp.dim[0], metrics,
                           px_AP=img_input.dim[5],
                           centers=centers,
                           img_slice_shape=img_seg.data.shape[1:],
                           radius=radius, scale=scale)
        add_slice_numbers(ax, img_temp.dim[0], radius=radius, reverse=True)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_path = str(imgs_to_generate['path_overlay_img'])
        logger.debug('Save image %s', img_path)
        fig.savefig(img_path, format='png', transparent=True, dpi=DPI)

    # QC entry 2 (optional)
    if angle_qc:
        # Axial orientation, switch between one anat image and one seg image
        with create_qc_entry(
            path_input=Path(fname_input).resolve(),
            path_qc=Path(path_qc),
            command=command,
            cmdline=list2cmdline(cmdline),
            plane='Axial',
            dataset=dataset,
            subject=subject,
            contrast=contrast,
        ) as imgs_to_generate:
            # Generate the first QC report image - background image
            fig = mpl_figure.Figure()
            fig.set_size_inches(*size_fig, forward=True)
            mpl_backend_agg.FigureCanvasAgg(fig)
            ax = fig.add_axes((0, 0, 1, 1))
            ax.imshow(img_bg, cmap='gray', interpolation='none', aspect=1.0)
            add_orientation_labels(ax, radius=tuple(r for r in radius))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            img_path = str(imgs_to_generate['path_background_img'])
            logger.debug('Save image %s', img_path)
            fig.savefig(img_path, format='png', transparent=True, dpi=DPI)

            # Generate the second QC report image - overlay image with lines for HOG angles
            fig = mpl_figure.Figure()
            fig.set_size_inches(*size_fig, forward=True)
            mpl_backend_agg.FigureCanvasAgg(fig)
            ax = fig.add_axes((0, 0, 1, 1))
            ax.imshow(img, aspect=1.0)
            # Plot HOG angle lines directly on the empty axes
            add_angle_lines(ax, img_temp.dim[0], metrics,
                            angle_type=angle_type,
                            centers=centers,
                            img_slice_shape=img_seg.data.shape[1:],
                            radius=radius, scale=scale)
            add_slice_numbers(ax, img_temp.dim[0], radius=radius, reverse=True)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            img_path = str(imgs_to_generate['path_overlay_img'])
            logger.debug('Save image %s', img_path)
            fig.savefig(img_path, format='png', transparent=True, dpi=DPI)


def add_angle_lines(ax, num_slices, metrics, angle_type, centers, img_slice_shape,
                    radius: tuple[int, int] = (15, 15), scale: float = 2.5):
    """
    Overlay HOG angle lines and add angle text onto an Axial mosaic.
    """
    AP_dim, LR_dim = img_slice_shape
    num_col = math.floor(TARGET_WIDTH_PIXL / scale / (2 * radius[0]))
    # Loop across axial slices
    for i in range(num_slices):
        # Calculate mosaic position
        slice_index = num_slices - 1 - i  # I need to use this hacky workaround to go from RPI to SAL
        row = slice_index // num_col
        col = slice_index % num_col
        # Center of mass
        # Note 1: This is a bit hacky: I check metrics['centermass_x'] just to kwow if given slice contains the
        #  cord seg, if so, then I plot the center of mass point based on the mosaic properties.
        # Note 2: metrics['centermass_x'] and ['centermass_y'] seem to be the same as `centers`, but they are obtained differently:
        #   - metrics['centermass_x'] and ['centermass_y'] are obtained using PCA via spinalcordtoolbox.registration.algorithms.compute_pca
        #   - centers is obtained using scipy.ndimage.center_of_mass
        #   For simplicity, I'm using directly the mosaic coordinates based on `centers`
        x = metrics['centermass_x'].data[i] if 'centermass_x' in metrics else np.nan
        y = metrics['centermass_y'].data[i] if 'centermass_y' in metrics else np.nan
        if not np.isnan(x) and not np.isnan(y):
            # Reconstruct the actual cord centre within the mosaic, replicating the clamping done in mosaic().
            # When the cord centre is within radius pixels of the image edge, mosaic() clamps it, so the cord
            # is no longer at the tile's geometric centre.
            cy, cx = centers[slice_index]
            cy_c = max(radius[0], min(AP_dim - radius[0], int(cy)))
            cx_c = max(radius[1], min(LR_dim - radius[1], int(cx)))
            y_mosaic = row * (2 * radius[0]) + int(cy) + radius[0] - cy_c
            x_mosaic = col * (2 * radius[1]) + int(cx) + radius[1] - cx_c
            # add the decimal part of the center of mass back.
            # (The mosaic is based on the SC center of mass, truncated to get an integer voxel coord. Preserving
            #  the decimal part when plotting is possible in matplotlib, and leads to a more accurate plot.)
            cy, cx = centers[slice_index]
            y_mosaic += (cy - int(cy))
            x_mosaic += (cx - int(cx))
            # Uncomment the next line to plot the center of mass point
            # ax.plot(x_mosaic, y_mosaic, 'o', color='red', markersize=1.0)
        # HOG angle
        angle_deg = metrics[angle_type].data[i] if angle_type in metrics else np.nan
        if not np.isnan(angle_deg):
            # Compute the end points of the line
            x_start = x_mosaic - radius[0] / 2 * np.sin(-angle_deg * np.pi / 180)
            y_start = y_mosaic - radius[1] / 2 * np.cos(-angle_deg * np.pi / 180)
            x_end = x_mosaic + radius[0] / 2 * np.sin(-angle_deg * np.pi / 180)
            y_end = y_mosaic + radius[1] / 2 * np.cos(-angle_deg * np.pi / 180)
            # Plot the line
            ax.plot([x_start, x_end], [y_start, y_end], '-', color='red', linewidth=0.7)
            # Include the angle text in degrees
            # Flip sign to match PCA convention
            # See https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/ba30577e80a4e7387498820f0ff30b8965fbf2a4/spinalcordtoolbox/registration/algorithms.py#L834
            # TODO: figure out why the link below flip the angle sign only for src_hog but not for dest_hog
            ax.text(x_mosaic + radius[0] * 0.2, y_mosaic - radius[1] * 0.3,  # upper right corner
                    f'{angle_deg:.1f}°', color='red', fontsize=3.5,
                    path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')])


def add_ap_split_lines(ax, num_slices, metrics, px_AP, centers, img_slice_shape,
                       radius: tuple[int, int] = (15, 15), scale: float = 2.5):
    """
    Overlay anterior/posterior cord length lines on an axial mosaic.

    For each slice, two lines are drawn (inspiration: Fig. 1 of Kang et al. J Clin Med 2023):
      - Cyan line:   cord center of mass → anterior cord edge (length_anterior)
      - Yellow line: cord center of mass → posterior cord edge (length_posterior)

    Lines are oriented using the ellipse orientation when available (otherwise vertical).
    This matches the rotation applied in _measure_ap_diameter() (which uses -region.orientation,
    the ellipse principal axis), not the HOG angle.

    :param ax: matplotlib axes object of the mosaic figure.
    :param num_slices: number of axial slices (= first dim of the SAL image).
    :param metrics: dict of Metric objects from compute_shape(). Must contain 'length_anterior'
      and 'length_posterior'. Optionally 'orientation' (ellipse orientation in degrees) for line orientation.
    :param px_AP: pixel size in mm along the AP axis (= img.dim[5] for a SAL-oriented image).
    :param centers: (N, 2) array of cord center-of-mass coordinates in SAL image space, as used
      by mosaic(). Required to correctly compute the cord position within each tile when the center
      is clamped to the image boundary.
    :param img_slice_shape: (AP_dim, LR_dim) shape of each axial SAL slice, used for clamping.
    :param radius: (rows, cols) half-size in pixels of each mosaic tile.
    :param scale: display scale factor (same value used when building the mosaic).
    """
    if 'length_anterior' not in metrics or 'length_posterior' not in metrics:
        return

    AP_dim, LR_dim = img_slice_shape
    num_col = math.floor(TARGET_WIDTH_PIXL / scale / (2 * radius[0]))

    for i in range(num_slices):
        diam_ant = metrics['length_anterior'].data[i]
        diam_post = metrics['length_posterior'].data[i]
        if np.isnan(diam_ant) or np.isnan(diam_post):
            continue

        # Mosaic tile position (same index inversion as add_angle_lines: RPI → SAL)
        slice_index = num_slices - 1 - i
        row = slice_index // num_col
        col = slice_index % num_col

        # Reconstruct the actual cord centre within the mosaic, replicating the clamping done in mosaic().
        # When the cord centre is within radius pixels of the image edge, mosaic() clamps it, so the cord
        # is no longer at the tile's geometric centre.
        cy, cx = centers[slice_index]
        cy_c = max(radius[0], min(AP_dim - radius[0], int(cy)))
        cx_c = max(radius[1], min(LR_dim - radius[1], int(cx)))
        y_mosaic = row * (2 * radius[0]) + int(cy) + radius[0] - cy_c
        x_mosaic = col * (2 * radius[1]) + int(cx) + radius[1] - cx_c
        # add the decimal part back, since the lengths are based off of the float-valued center
        y_mosaic += (cy - int(cy))
        x_mosaic += (cx - int(cx))

        # Ellipse orientation determines the cord's AP axis orientation in the mosaic.
        # SAL orientation: in imshow coordinates, y increases downward.
        # Orientation labels confirm: A (Anterior) = top (small y), P (Posterior) = bottom (large y).
        # Unit vector toward Posterior at angle_deg=0: (sin(0), cos(0)) = (0, 1) = straight down ✓
        angle_deg = 0.0
        if 'orientation' in metrics:
            v = metrics['orientation'].data[i]
            if not np.isnan(v):
                angle_deg = v
        angle_rad = -angle_deg * np.pi / 180

        # AP unit vector (toward Posterior) and perpendicular RL unit vector (toward Left at 0°)
        ap_dx, ap_dy = np.sin(angle_rad), np.cos(angle_rad)

        # Convert mm lengths to mosaic pixel lengths
        ant_px = diam_ant / px_AP
        post_px = diam_post / px_AP

        stroke = [mpl_patheffects.withStroke(linewidth=0.6, foreground='black')]

        # Cyan: cord center of mass to anterior cord edge
        ax.plot([x_mosaic, x_mosaic - ant_px * ap_dx],
                [y_mosaic, y_mosaic - ant_px * ap_dy],
                '-', color='cyan', linewidth=0.75, solid_capstyle='butt')
        # Yellow: cord center of mass to posterior cord edge
        ax.plot([x_mosaic, x_mosaic + post_px * ap_dx],
                [y_mosaic, y_mosaic + post_px * ap_dy],
                '-', color='yellow', linewidth=0.75, solid_capstyle='butt')

        # # Center of mass
        # ax.plot(x_mosaic, y_mosaic, 'o', color='black', markersize=1, markeredgewidth=0.8)

        # Value labels (mm) at the line tips
        ax.text(x_mosaic - ant_px * ap_dx,
                y_mosaic - ant_px * ap_dy - 1.5,
                f'{diam_ant:.1f} mm', color='cyan', fontsize=3.5,
                ha='left', va='bottom', path_effects=stroke)
        ax.text(x_mosaic + post_px * ap_dx,
                y_mosaic + post_px * ap_dy + 1.5,
                f'{diam_post:.1f} mm', color='yellow', fontsize=3.5,
                ha='left', va='top', path_effects=stroke)


def sct_deepseg(
    fname_input: str,
    fname_seg: str,
    fname_seg2: Optional[str],
    species: str,
    argv: Sequence[str],
    path_qc: str,
    dataset: Optional[str],
    subject: Optional[str],
    contrast: Optional[str],
    plane: Optional[str],
    fname_qc_seg: Optional[str],
    command: str = 'sct_deepseg'
):
    """
    Generate a QC report for sct_deepseg, based on which task was used.
    """
    cmdline = [command]
    cmdline.extend(argv)

    with create_qc_entry(
        path_input=Path(fname_input).resolve(),
        path_qc=Path(path_qc),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane=plane,
        dataset=dataset,
        subject=subject,
        contrast=contrast,
    ) as imgs_to_generate:
        # Custom QC to handle multiclass segmentation outside the spinal cord
        if "rootlets" in argv:
            sct_deepseg_spinal_rootlets(
                imgs_to_generate, fname_input, fname_seg, fname_seg2, species,
                radius=(23, 23), base_scaling=2.5,  # standard upscale to see rootlets detail
                outline=True)  # add outlines (and labels) to highlight the difficult-to-see rootlets seg
        elif "spine" in argv:
            sct_deepseg_spinal_rootlets(
                imgs_to_generate, fname_input, fname_seg, fname_seg2, species,
                radius=(40, 40), base_scaling=1.0,  # skip upscaling to get "big picture" view of all slices
                outline=False)  # skip outlines (and labels) because the images will be too small to display them
        # Non-rootlets, axial/sagittal DeepSeg QC report
        elif plane == 'Axial':
            sct_deepseg_axial(
                imgs_to_generate, fname_input, fname_seg, fname_seg2, species, fname_qc_seg,
                radius=(27, 27) if "canal" in argv else (15, 15))
        else:
            assert plane == 'Sagittal', (f"`plane` must be either 'Axial' "
                                         f"or 'Sagittal', but got {plane}")
            sct_deepseg_sagittal(
                imgs_to_generate, fname_input, fname_seg, fname_seg2, species, fname_qc_seg)


def sct_deepseg_axial(
    imgs_to_generate: dict[str, Path],
    fname_input: str,
    fname_seg_sc: str,
    fname_seg_lesion: Optional[str],
    species: str,
    fname_qc_seg: Optional[str],
    radius: Sequence[int]
):
    """
    Generate a QC report for sct_deepseg, with varied colormaps depending on the type of segmentation.

    This refactor is based off of the `listed_seg` method in qc.py, adapted to support multiple images.
    """
    # Axial orientation, switch between one anat image and 1-2 seg images
    # FIXME: This code is more or less duplicated with the 'sct_register_multimodal' report, because both reports
    #        use the old qc.py method "_make_QC_image_for_3d_volumes" for generating the background img.
    # Resample images slice by slice
    p_resample = {'human': 0.6, 'mouse': 0.1}[species]
    img_input = Image(fname_input)
    orientation_orig = img_input.orientation
    img_input = img_input.change_orientation('SAL')
    img_seg_sc = Image(fname_seg_sc).change_orientation('SAL')
    img_seg_lesion = Image(fname_seg_lesion).change_orientation('SAL') if fname_seg_lesion else None
    img_qc_seg = Image(fname_qc_seg).change_orientation('SAL') if fname_qc_seg else None

    # Resample images slice by slice
    logger.info('Resample images to %fx%f vox', p_resample, p_resample)
    img_input = resample_nib(
        image=img_input,
        new_size=[img_input.dim[4], p_resample, p_resample],
        new_size_type='mm',
        interpolation='spline',
    )
    img_seg_sc = resample_nib(
        image=img_seg_sc,
        image_dest=img_input,
        interpolation='linear',
    )
    img_seg_sc.data = (img_seg_sc.data > 0.5) * 1
    img_seg_lesion = resample_nib(
        image=img_seg_lesion,
        image_dest=img_input,
        interpolation='linear',
    ) if fname_seg_lesion else None
    if fname_seg_lesion:
        img_seg_lesion.data = (img_seg_lesion.data > 0.5) * 1
    img_qc_seg = resample_nib(
        image=img_qc_seg,
        image_dest=img_input,
        interpolation='linear',
    ) if fname_qc_seg else None

    # Using -qc-seg mask if available, we remove slices which are empty (except for the first 3 and last 3 slices just around the segmented slices)
    for img_to_crop in [img_input, img_seg_sc, img_seg_lesion, img_qc_seg]:
        if fname_qc_seg and img_to_crop:
            crop_with_mask(img_to_crop, img_qc_seg, pad=3)

    # Each slice is centered on the segmentation
    logger.info('Find the center of each slice')
    # Use the -qc-seg mask if available, otherwise use the spinal cord mask
    img_centers = img_qc_seg if fname_qc_seg else img_seg_sc
    centers = np.array([ndimage.center_of_mass(slice) for slice in img_centers.data])
    inf_nan_fill(centers[:, 0])
    inf_nan_fill(centers[:, 1])

    # If -qc-seg is available, use it to generate the radius
    radius = get_max_axial_radius(img_qc_seg) if fname_qc_seg else radius

    # Generate the first QC report image
    img = equalize_histogram(mosaic(img_input, centers, radius))

    # Fix the width to a specific size, and vary the height based on how many rows there are.
    size_fig = [TARGET_WIDTH_INCH, TARGET_WIDTH_INCH * img.shape[0] / img.shape[1]]

    fig = mpl_figure.Figure()
    fig.set_size_inches(*size_fig, forward=True)
    mpl_backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, cmap='gray', interpolation='none', aspect=1.0)
    add_orientation_labels(ax, radius=radius)
    add_slice_numbers(ax, img_input.dim[0], radius=radius, reverse=("I" in orientation_orig))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_background_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=DPI)

    # Generate the second QC report image
    fig = mpl_figure.Figure()
    fig.set_size_inches(*size_fig, forward=True)
    mpl_backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    colormaps = [mpl_colors.ListedColormap(["#ff0000"]),  # Red for first image
                 mpl_colors.ListedColormap(["#00ffff"])]  # Cyan for second
    for i, image in enumerate([img_seg_sc, img_seg_lesion]):
        if not image:
            continue
        img = mosaic(image, centers, radius)
        img = np.ma.masked_less_equal(img, 0)
        img.set_fill_value(0)
        ax.imshow(img,
                  cmap=colormaps[i],
                  norm=mpl_colors.Normalize(vmin=0.5, vmax=1),
                  # img==1 -> opaque, but soft regions -> more transparent as value decreases
                  alpha=1.0,
                  interpolation='none',
                  aspect=1.0)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_overlay_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=DPI)


def sct_deepseg_spinal_rootlets(
    imgs_to_generate: dict[str, Path],
    fname_input: str,
    fname_seg_sc: str,
    fname_seg_lesion: Optional[str],
    species: str,
    radius: Sequence[int],
    base_scaling: float = 2.5,  # scale up mosaic slices to make them more readable by default
    outline: bool = True
):
    """
    Generate a QC report for `sct_deepseg rootlets`.

    This refactor is based off of the `listed_seg` method in qc.py, adapted to support multiple images.
    """
    # Axial orientation, switch between one anat image and 1-2 seg images
    # FIXME: This code is more or less duplicated with the 'sct_register_multimodal' report, because both reports
    #        use the old qc.py method "_make_QC_image_for_3d_volumes" for generating the background img.

    # Load the input images
    img_input = Image(fname_input).change_orientation('SAL')
    img_seg_sc = Image(fname_seg_sc).change_orientation('SAL')
    img_seg_lesion = Image(fname_seg_lesion).change_orientation('SAL') if fname_seg_lesion else None

    # - Normally, we would apply isotropic resampling to the image to a specific mm resolution (based on the species).
    p_resample = {'human': 0.6, 'mouse': 0.1}[species]
    #   Choosing a fixed resolution allows us to crop the image around the spinal cord at a fixed radius that matches the chosen resolution,
    #   while also handling anisotropic images (so that they display correctly on an isotropic grid).
    # - However, we cannot apply resampling here because rootlets labels are often small (~1vox wide), and so resampling might
    #   corrupt the labels and cause them to be displayed unfaithfully.
    # - So, instead of resampling the image to fit the default crop radius, we scale the crop radius to suit the original resolution.
    p_original = (img_seg_sc.dim[5], img_seg_sc.dim[6])  # Image may be anisotropic, so use both resolutions (H,W)
    p_ratio = tuple(p_resample / p for p in p_original)
    radius = tuple(int(r * p) for r, p in zip(radius, p_ratio))
    # - One problem with this, however, is that if the crop radius ends up being smaller than the default, the QC will in turn be smaller as well.
    #   So, to ensure that the QC is still readable, we scale up whenever the p_ratio is < 1
    scale = max((1 / ratio) for ratio in p_ratio)  # e.g. 0.8mm human => p_ratio == 0.6/0.8 == 0.75; scale == 1/p_ratio == 1/0.75 == 1.33
    # - Note: `mosaic()` already has a base scaling factor of 2.5 (to help make the QC readable).
    #          Since resolution-based scaling would overwrite this, we need to preserve the base scaling factor.
    # - Note2: For `spine`, we actually _don't_ want to upscale by 2.5, so that's why `base_scaling` has
    #          been parametrized, allowing per-QC customization.
    scale *= base_scaling
    # - One other problem is that for anisotropic images, the aspect ratio won't be 1:1 between width/height.
    #   So, we use `aspect` to adjust the image via imshow, and `radius` to know where to place the text in x/y coords
    aspect = p_ratio[1] / p_ratio[0]

    # Each slice is centered on the segmentation
    logger.info('Find the center of each slice')
    centerline_param = ParamCenterline(algo_fitting="optic", contrast="t2")
    img_centerline, _, _, _ = get_centerline(img_input, param=centerline_param)
    centers = np.array([ndimage.center_of_mass(slice) for slice in img_centerline.data])
    inf_nan_fill(centers[:, 0])
    inf_nan_fill(centers[:, 1])

    # Generate the first QC report image
    img = equalize_histogram(mosaic(img_input, centers, radius, scale))

    # Fix the width to a specific size, and vary the height based on how many rows there are.
    size_fig = [TARGET_WIDTH_INCH, TARGET_WIDTH_INCH * (img.shape[0] / img.shape[1]) * aspect]

    fig = mpl_figure.Figure()
    fig.set_size_inches(*size_fig, forward=True)
    mpl_backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, cmap='gray', interpolation='none', aspect=aspect)
    add_orientation_labels(ax, radius=tuple(r for r in radius))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_background_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=DPI)

    # Generate the second QC report image
    fig = mpl_figure.Figure()
    fig.set_size_inches(*size_fig, forward=True)
    mpl_backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    # get available labels
    img = np.rint(np.ma.masked_where(img_seg_sc.data < 1, img_seg_sc.data))
    labels = np.unique(img[np.where(~img.mask)]).astype(int)
    colormaps = [mpl_colors.ListedColormap(assign_label_colors_by_groups(labels))]
    for i, image in enumerate([img_seg_sc, img_seg_lesion]):
        if not image:
            continue
        img = mosaic(image, centers, radius, scale)
        img = np.ma.masked_less_equal(img, 0)
        img.set_fill_value(0)
        ax.imshow(img,
                  cmap=colormaps[i],
                  norm=None,
                  alpha=1.0,
                  interpolation='none',
                  aspect=aspect)

        # only display outlines and segmentation labels if opted into
        # (in practice, this saves them from being added to the tiny totalspineseg QC)
        if outline:
            add_segmentation_labels(ax, img, colors=colormaps[i].colors, radius=tuple(r for r in radius))
            # linewidth 0.5 is too thick, 0.25 is too thin
            plot_outlines(img, ax=ax, facecolor='none', edgecolor='black', linewidth=0.3)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_overlay_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=DPI)


def sct_deepseg_sagittal(
    imgs_to_generate: dict[str, Path],
    fname_input: str,
    fname_seg_sc: str,
    fname_seg_lesion: Optional[str],
    species: str,
    fname_qc_seg: Optional[str],
):
    """
    Generate a QC report for sct_deepseg, with varied colormaps depending on the type of segmentation.

    This refactor is based off of the `listed_seg` method in qc.py, adapted to support multiple images.
    """
    # Axial orientation, switch between one anat image and 1-2 seg images
    # FIXME: This code is more or less duplicated with the 'sct_register_multimodal' report, because both reports
    #        use the old qc.py method "_make_QC_image_for_3d_volumes" for generating the background img.
    # Resample images slice by slice
    p_resample = {'human': 0.6, 'mouse': 0.1}[species]
    img_input = Image(fname_input).change_orientation('RSP')
    img_seg_sc = Image(fname_seg_sc).change_orientation('RSP')
    img_seg_lesion = Image(fname_seg_lesion).change_orientation('RSP') if fname_seg_lesion else None
    img_qc_seg = Image(fname_qc_seg).change_orientation('RSP') if fname_qc_seg else None

    # Resample images slice by slice
    R_L_resolution = img_input.dim[4]
    logger.info('Resample images to %fx%fx%f vox', R_L_resolution, p_resample, p_resample)
    img_input = resample_nib(
        image=img_input,
        new_size=[R_L_resolution, p_resample, p_resample],
        new_size_type='mm',
        interpolation='spline',
    )
    img_seg_sc = resample_nib(
        image=img_seg_sc,
        image_dest=img_input,
        interpolation='linear',
    )
    img_seg_sc.data = (img_seg_sc.data > 0.5) * 1
    img_seg_lesion = resample_nib(
        image=img_seg_lesion,
        image_dest=img_input,
        interpolation='linear',
    ) if fname_seg_lesion else None
    if fname_seg_lesion:
        img_seg_lesion.data = (img_seg_lesion.data > 0.5) * 1
    img_qc_seg = resample_nib(
        image=img_qc_seg,
        image_dest=img_input,
        interpolation='linear',
    ) if fname_qc_seg else None

    # Using -qc-seg mask if available, we remove slices which are empty (except for the first 2 and last 2 slices just around the segmented slices)
    # If -qc-seg mask isn't available, but image would create a mosaic that's too large, keep only the center 30 sagittal slices (using segmnetation as a reference)
    for img_to_crop in [img_input, img_seg_sc, img_seg_lesion, img_qc_seg]:
        if img_to_crop is None:
            continue  # don't crop missing images
        if fname_qc_seg:
            crop_with_mask(img_to_crop, img_qc_seg, pad=2)
        elif img_input.dim[0] > 30:
            crop_with_mask(img_to_crop, img_seg_sc, max_slices=30)
            if img_to_crop == img_input:  # display a message only once
                logger.warning("Source image is too large to display in a sagittal mosaic. Applying automatic cropping around segmentation.\n"
                               "Please consider using `sct_deepseg -qc-seg` option to customize the crop. You can use `sct_create_mask` to create a suitable mask to pass to "
                               "`-qc-seg`. If this message still occurs after cropping, please consider resampling your image to a lower resolution using `sct_resample`.")

    logger.info('Find the center of each slice')
    # Use the -qc-seg mask if available to get crop radius (as well as the center of mass) for each slice
    if fname_qc_seg:
        radius = get_max_sagittal_radius(img_qc_seg)
        centers = np.array([ndimage.center_of_mass(slice) for slice in img_qc_seg.data])
        inf_nan_fill(centers[:, 0])
        inf_nan_fill(centers[:, 1])
    # otherwise, if -qc-seg isn't provided, display the full sagittal slice and use the center of the uncropped image
    else:
        radius = (img_input.dim[1] // 2, img_input.dim[2] // 2)
        centers = np.array([radius] * img_input.data.shape[0])

    # Generate the first QC report image
    img = equalize_histogram(mosaic(img_input, centers, radius=radius))

    # Fix the width to a specific size, and vary the height based on how many rows there are.
    size_fig = [TARGET_WIDTH_INCH, TARGET_WIDTH_INCH * img.shape[0] / img.shape[1]]

    fig = mpl_figure.Figure()
    fig.set_size_inches(*size_fig, forward=True)
    mpl_backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, cmap='gray', interpolation='none', aspect=1.0)
    add_orientation_labels(ax, radius=radius, letters=['S', 'I', 'P', 'A'])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_background_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=DPI)

    # Generate the second QC report image
    fig = mpl_figure.Figure()
    fig.set_size_inches(*size_fig, forward=True)
    mpl_backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    colormaps = [mpl_colors.ListedColormap(["#ff0000"]),  # Red for first image
                 mpl_colors.ListedColormap(["#00ffff"])]  # Cyan for second
    for i, image in enumerate([img_seg_sc, img_seg_lesion]):
        if not image:
            continue
        img = mosaic(image, centers, radius=radius)
        img = np.ma.masked_less_equal(img, 0)
        img.set_fill_value(0)
        ax.imshow(img,
                  cmap=colormaps[i],
                  norm=mpl_colors.Normalize(vmin=0.5, vmax=1),
                  # img==1 -> opaque, but soft regions -> more transparent as value decreases
                  alpha=1.0,
                  interpolation='none',
                  aspect=1.0)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_overlay_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=DPI)


# Mimic the 'Ortho view' from FSLeyes using a single slice from each of the 3 orthogonal planes.
# - The plane order (sagittal -> coronal -> axial) follows the default FSLeyes order
# - The orientations below are chosen simply so that, after extracting a panel,
#   the x/y axes of the 2D array correspond to the labels shown below.
# - NB: FSLeyes uses radiological orientation (R -> L), but historically SCT's QC reports have used
#   neurological orientation (L -> R), so we follow that precedent here.
_ORTHO_QC_PLANES = {
    'Sagittal': ('RSP', ('S', 'I'), ('P', 'A')),
    'Coronal':  ('ASL', ('S', 'I'), ('L', 'R')),
    'Axial':    ('SAL', ('A', 'P'), ('L', 'R')),
}


def _extract_ortho_panel(fname: str, plane: str, bbox_min: np.ndarray, bbox_max: np.ndarray,
                         margin_frac: float = 0.25) -> np.ndarray:
    """
    Extract a single 2D panel from `fname` for use in constructing an orthographic view of a 3D image.

    Use the slice through the box's center, reoriented for `plane` (see `_ORTHO_QC_PLANES`) and cropped to
    the box's own extent (`bbox_min`/`bbox_max`, already in that same orientation) plus a margin, for
    anatomical context, instead of showing the full image.
    """
    orientation, _, _ = _ORTHO_QC_PLANES[plane]
    data = Image(fname).change_orientation(orientation).data

    def bounds(axis):
        lo, hi = int(bbox_min[axis]), int(bbox_max[axis])
        margin = max(5, round((hi - lo) * margin_frac))
        return max(0, lo - margin), min(data.shape[axis], hi + margin + 1)

    # Axis 0 is always the axis fixed at the box center, axis 1 always vertical, axis 2 always
    # horizontal -- by construction of the per-plane orientation codes above, so no further axis
    # bookkeeping is needed here (unlike a single-shared-orientation approach would require).
    panel = data[int((bbox_min[0] + bbox_max[0]) // 2)]
    y0, y1 = bounds(1)
    x0, x1 = bounds(2)
    return panel[y0:y1, x0:x1]


def sct_deepseg_cropbox(
    fname_input: str,
    fname_cropbox: str,
    path_qc: str,
    dataset: Optional[str],
    subject: Optional[str],
    contrast: Optional[str],
):
    """
    Generate a QC report entry for the sc-crop bounding box: one slice per orthogonal plane
    (axial, coronal, sagittal), each centered on the box, with the box's extent outlined. This is
    a separate report entry from the segmentation QC report, so users can tell a bad crop (box
    missed the cord) apart from a bad segmentation (box is fine, but the prediction inside it is
    wrong) — the segmentation QC report is always cropped around the segmentation and can't show
    whether the box itself was placed correctly.
    """
    # NB: labeled 'sc_crop' (rather than 'sct_deepseg', the tool that actually produced this
    # entry) with a standalone `sc_crop --bbox` command that reproduces the same crop, instead of
    # the `sct_deepseg` command that was really run. This is still a real, runnable command (not
    # a fabricated one) -- and it means neither "command" nor "cmdline" contains "sct_deepseg"
    # for this entry, so searching "sct_deepseg" in the QC report reliably shows *only*
    # segmentation rows, and searching "sc_crop" shows only crop-box rows, in either direction.
    command = 'sc_crop'
    cmdline = ['sc_crop', '-i', fname_input, '--bbox', fname_cropbox]

    with create_qc_entry(
        # NB: identify this entry by the cropbox file rather than the anatomical input, so it's
        # distinguishable from the segmentation QC entry in the report's "File" column (the two
        # would otherwise be identical -- same input file, same subject/contrast, same timestamp
        # down to the second).
        path_input=Path(fname_cropbox).resolve(),
        path_qc=Path(path_qc),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane='Ortho',
        dataset=dataset,
        subject=subject,
        contrast=contrast,
    ) as imgs_to_generate:
        # Extract every panel up front so widths can be derived from their actual aspect ratio
        # (width / height) instead of splitting the figure into equal-width columns: with a fixed
        # pixel aspect (imshow's aspect=1.0, needed for anatomically-correct proportions), an
        # equal-width column whose shape doesn't match its panel's aspect ratio gets letterboxed
        # by matplotlib -- visible as the empty margins around each slice.
        # Each panel is windowed to the box's own extent (+25% margin for anatomical context)
        # instead of the full image, so the box fills most of the available space -- consistent
        # with every other report in this file (e.g. sct_deepseg_axial's radius-based mosaic),
        # none of which display the full, un-zoomed image either.
        images = {'input': fname_input, 'box': fname_cropbox}
        panels = {name: {} for name in images}
        for plane, (orientation, _, _) in _ORTHO_QC_PLANES.items():
            # The box's own extent (in this plane's orientation) determines both the center slice
            # and the crop window, so it's recomputed per plane rather than shared across planes.
            box_data = Image(fname_cropbox).change_orientation(orientation).data
            coords = np.argwhere(box_data > 0)
            if coords.size == 0:
                raise ValueError(f"Crop box mask '{fname_cropbox}' is empty — nothing to display.")
            bbox_min, bbox_max = coords.min(axis=0), coords.max(axis=0)
            for image_name, fname in images.items():
                panels[image_name][plane] = _extract_ortho_panel(fname, plane, bbox_min, bbox_max)
        panels_input, panels_box = panels['input'], panels['box']

        # Fix the total width to TARGET_WIDTH_INCH (as every other report does) and derive a shared
        # row height so that each panel's own width (at that height) sums up to the total width.
        aspect_ratios = {label: p.shape[1] / p.shape[0] for label, p in panels_input.items()}
        height_fig = TARGET_WIDTH_INCH / sum(aspect_ratios.values())
        panel_widths = {label: height_fig * ratio for label, ratio in aspect_ratios.items()}

        # Background image: the anatomical slices, one per plane
        text_args = dict(
            color='yellow',
            fontsize=4,
            path_effects=[
                mpl_patheffects.Stroke(linewidth=1, foreground='black'),
                mpl_patheffects.Normal(),
            ],
        )
        fig = mpl_figure.Figure()
        fig.set_size_inches(TARGET_WIDTH_INCH, height_fig, forward=True)
        mpl_backend_agg.FigureCanvasAgg(fig)
        x_offset = 0.0
        for label, (_, (top, bottom), (left, right)) in _ORTHO_QC_PLANES.items():
            w_frac = panel_widths[label] / TARGET_WIDTH_INCH
            ax = fig.add_axes((x_offset, 0, w_frac, 1))
            panel = panels_input[label]
            # origin='upper' (row 0 at the top), matching every other report in this file --
            # correct here since row 0 of `panel` is always the panel's own "top" letter, by
            # construction of the per-plane orientation codes in `_ORTHO_QC_PLANES`.
            ax.imshow(equalize_histogram(panel), cmap='gray', origin='upper',
                      interpolation='none', aspect=1.0)
            ax.set_title(label, fontsize=6)
            h, w = panel.shape
            # Inset the anchor point a few pixels in from the edge, so the label sits fully
            # inside the panel instead of being centered on the border (where it would be half
            # cut off) -- no need for xlim/ylim padding (blank margin) or clip_on=False either.
            inset = max(2, round(0.03 * min(w, h)))
            ax.text(w / 2, inset, top, ha='center', va='top', **text_args)
            ax.text(w / 2, h - inset, bottom, ha='center', va='bottom', **text_args)
            ax.text(inset, h / 2, left, ha='left', va='center', **text_args)
            ax.text(w - inset, h / 2, right, ha='right', va='center', **text_args)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            x_offset += w_frac
        img_path = str(imgs_to_generate['path_background_img'])
        logger.debug('Save image %s', img_path)
        fig.savefig(img_path, format='png', transparent=True, dpi=DPI)

        # Overlay image: the box outline, one per plane, same layout as the background. Yellow to
        # match the crop box's appearance when viewed as an overlay in FSLeyes (see sct_deepseg.py).
        fig = mpl_figure.Figure()
        fig.set_size_inches(TARGET_WIDTH_INCH, height_fig, forward=True)
        mpl_backend_agg.FigureCanvasAgg(fig)
        x_offset = 0.0
        for label in _ORTHO_QC_PLANES:
            w_frac = panel_widths[label] / TARGET_WIDTH_INCH
            ax = fig.add_axes((x_offset, 0, w_frac, 1))
            panel = panels_box[label]
            ys, xs = np.where(panel > 0)
            if len(xs):
                rect = mpl_patches.Rectangle(
                    (xs.min(), ys.min()), xs.max() - xs.min(), ys.max() - ys.min(),
                    fill=False, edgecolor='yellow', linewidth=1.5)
                ax.add_patch(rect)
            ax.set_xlim(0, panel.shape[1])
            # Inverted (vs. the background's origin='upper'): row 0 (small y) must map to the top
            # of the panel here too, so the box outline stays aligned with the anatomical image.
            ax.set_ylim(panel.shape[0], 0)
            ax.set_aspect(1.0)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            x_offset += w_frac
        img_path = str(imgs_to_generate['path_overlay_img'])
        logger.debug('Save image %s', img_path)
        fig.savefig(img_path, format='png', transparent=True, dpi=DPI)


def sct_analyze_lesion(
    fname_input: str,
    fname_label: str,
    fname_sc: str,
    measure_pd: 'pd.DataFrame',
    argv: Sequence[str],
    path_qc: str,
    dataset: Optional[str],
    subject: Optional[str],
    contrast: Optional[str],
    command: str = 'sct_analyze_lesion'
):
    """
    Generate a QC report for sct_analyze_lesion, specifically highlighting the tissue bridge widths.
    """
    cmdline = [command]
    cmdline.extend(argv)

    # Axial orientation, switch between one anat image and 1-2 seg images
    with create_qc_entry(
        path_input=Path(fname_input).resolve(),
        path_qc=Path(path_qc),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane='Sagittal',
        dataset=dataset,
        subject=subject,
        contrast=contrast,
    ) as imgs_to_generate:
        # Load the spinal cord segmentation mask
        im_sc = Image(fname_sc)
        im_sc.change_orientation("RPI")
        im_sc_data = im_sc.data

        # Load the labeled lesion mask
        im_lesion = Image(fname_label)
        # Store the original orientation of the lesion mask before reorienting it to RPI
        orig_orientation = im_lesion.orientation
        im_lesion.change_orientation("RPI")
        im_lesion_data = im_lesion.data
        # Restrict the lesion mask to the spinal cord mask, as lesions should not occur outside the cord
        im_lesion_data = im_lesion_data * im_sc_data
        label_lst = [label for label in np.unique(im_lesion_data) if label]

        # Get the total number of lesions; this will represent the number of rows in the figure. For example, if we have
        # 2 lesions, we will have two rows. One row per lesion.
        num_of_lesions = len(label_lst)
        # Get the sagittal lesion slices
        sagittal_lesion_slices = np.unique(np.where(im_lesion_data)[0])
        # Get the minimum sagittal slice with lesion. For example, if a lesion cover slices 7,8,9, get 7
        min_sag_slice = min(sagittal_lesion_slices)
        # Get the maximum sagittal slice with lesion. For example, if a lesion cover slices 7,8,9, get 9
        max_sag_slice = max(sagittal_lesion_slices)
        # Get the number of slices containing the lesion. For example, if a lesion cover slices 7,8,9, get 3
        num_of_sag_slices = max_sag_slice - min_sag_slice + 1

        #  Create a figure
        #  The figure has one row per lesion and one column per sagittal slice containing the lesion
        # TODO: This report breaks the assumption that the width is fixed in pixel size
        #       The figure could be very wide horizontally, which wouldn't work with the "new QC"
        #       See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4887
        fig, axes = mpl_plt.subplots(num_of_lesions,
                                     num_of_sag_slices,
                                     figsize=(num_of_sag_slices * 5, num_of_lesions * 5))

        # Force axes to be a 2-dimensional array (to avoid indexing issues if we have only a single lesion or a single
        # sagittal slice)
        axes = np.asanyarray(axes).reshape((num_of_lesions, num_of_sag_slices))

        # Loop across lesions
        for idx_row, lesion_label in enumerate(label_lst):
            # NOTE: As the 'label_lesion()' function has been called at the beginning of the script, im_lesion_data is
            # now "labeled" meaning that different lesions have different values, e.g., 1, 2, 3
            # As we are looping across lesions, we get the lesion mask for the current lesion label
            im_label_data_cur = im_lesion_data == lesion_label

            # Loop across sagittal slices
            for idx_col, sagittal_slice in enumerate(range(min_sag_slice, max_sag_slice + 1)):
                # Get spinal cord and lesion masks data for the selected sagittal slice
                slice_sc = im_sc_data[sagittal_slice]
                slice_lesion = im_label_data_cur[sagittal_slice]

                # Convert the sagittal slice to the original orientation
                # '0' because of the R-L direction (first in RPI)
                sagittal_slice = rpi_slice_to_orig_orientation(im_lesion.dim, orig_orientation,
                                                               sagittal_slice, 0)

                # Plot spinal cord and lesion masks
                axes[idx_row, idx_col].imshow(np.swapaxes(slice_sc, 1, 0),
                                              cmap='gray', origin="lower")
                axes[idx_row, idx_col].imshow(np.swapaxes(slice_lesion, 1, 0),
                                              cmap='jet', alpha=0.8, interpolation='nearest', origin="lower")

                # Add title for each column
                if idx_row == 0:
                    axes[idx_row, idx_col].set_title(f'Sagittal slice #{sagittal_slice}')

                # Add title to each row, (i.e., y-axis)
                if idx_col == 0:
                    axes[idx_row, idx_col].set_ylabel(f'Lesion #{lesion_label}\n'
                                                      f'Inferior-Superior')
                else:
                    axes[idx_row, idx_col].set_ylabel('Inferior-Superior')

                # Add x-axis label
                axes[idx_row, idx_col].set_xlabel('Anterior-Posterior')

                # Crop the slice around the lesion (to zoom in)
                axes[idx_row, idx_col].set_xlim(np.min(np.where(im_label_data_cur)[1]) - 20,
                                                np.max(np.where(im_label_data_cur)[1]) + 20)
                axes[idx_row, idx_col].set_ylim(np.min(np.where(im_label_data_cur)[2]) - 20,
                                                np.max(np.where(im_label_data_cur)[2]) + 20)

                # --------------------------------------
                # Add text for tissue bridges
                # --------------------------------------
                # Check if the [idx_row][sagittal_slice, 'dorsal'] key exists in the tissue_bridges_plotting_data
                # If the key exists, it means that we have tissue bridges for the current lesion and sagittal slice
                if idx_row < len(measure_pd):
                    col_name_dorsal = f"slice_{int(sagittal_slice)}_dorsal_bridge_width [mm]"
                    if col_name_dorsal in measure_pd.columns:
                        dorsal_bridge_width_mm = measure_pd[col_name_dorsal][idx_row]
                        if not pd.isna(dorsal_bridge_width_mm):
                            axes[idx_row, idx_col].text(min(np.where(slice_lesion)[0]) - 3,
                                                        min(np.where(slice_lesion)[1]),
                                                        f'Dorsal bridge\n{np.round(dorsal_bridge_width_mm, 2)} mm',
                                                        color='red', fontsize=12, ha='left', va='bottom')

                    col_name_ventral = f"slice_{int(sagittal_slice)}_ventral_bridge_width [mm]"
                    if col_name_ventral in measure_pd.columns:
                        ventral_bridge_width_mm = measure_pd[col_name_ventral][idx_row]
                        if not pd.isna(ventral_bridge_width_mm):
                            axes[idx_row, idx_col].text(max(np.where(slice_lesion)[0]) + 3,
                                                        min(np.where(slice_lesion)[1]),
                                                        f'Ventral bridge\n{np.round(ventral_bridge_width_mm, 2)} mm',
                                                        color='red', fontsize=12, ha='right', va='bottom')

                # Swap x-axis to anterior-posterior (from the current posterior-anterior), so that ventral tissue
                # bridges are on the left and dorsal tissue bridges on the right
                # Context: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4608#issue-2482923134
                axes[idx_row, idx_col].invert_xaxis()

        # tight layout
        mpl_plt.tight_layout()
        for fname in imgs_to_generate.values():
            mpl_plt.savefig(fname)


def inf_nan_fill(A: np.ndarray):
    """
    Interpolate inf and NaN values with neighboring values in a 1D array, in-place.
    If only inf and NaNs, fills the array with zeros.
    """
    valid = np.isfinite(A)
    invalid = ~valid
    if np.all(invalid):
        A.fill(0)
    elif np.any(invalid):
        A[invalid] = np.interp(
            np.nonzero(invalid)[0],
            np.nonzero(valid)[0],
            A[valid])


def mosaic(img: Image, centers: np.ndarray, radius: tuple[int, int] = (15, 15), scale: float = 2.5):
    """
    Arrange the slices of `img` into a grid of images.

    Each slice is centered at the approximate coordinates given in `centers`,
    and cropped to `radius` pixels in each direction (horizontal, vertical).

    If `img` has N slices, then `centers` should have shape (N, 2).
    """
    # Note: This function used to hardcode a max row width of 600 pixels
    #       In practice, because the canvas size is fixed to 1500 pixels, this
    #       resulted in a permanent upscaling by 2.5x when saving the image.
    #       To make things clearer, we now use a variable.
    max_row_width = TARGET_WIDTH_PIXL / scale

    # Fit as many slices as possible in each row
    num_col = math.floor(max_row_width / (2*radius[1]))

    # Center and crop each axial slice
    cropped = []
    for center, slice in zip(centers.astype(int), img.data):
        # If the `center` coordinate is close to the edge, then move it away from the edge to capture more of the image
        # In other words, make sure the `center` coordinate is at least `radius` pixels away from the edge
        for i in [0, 1]:
            center[i] = min(slice.shape[i] - radius[i], center[i])  # Check far edge first
            center[i] = max(radius[i],                  center[i])  # Then check 0 edge last
        # Add a margin before cropping, in case the center is still too close to one of the edges
        cropped.append(np.pad(slice, [[r] for r in radius])[
            center[0]:center[0] + 2*radius[0],
            center[1]:center[1] + 2*radius[1],
        ])

    # Pad the list with empty arrays, to get complete rows of num_col
    empty = np.zeros((2*radius[0], 2*radius[1]))
    cropped.extend([empty] * (-len(cropped) % num_col))

    # Arrange the images into a grid
    return np.block([cropped[i:i+num_col] for i in range(0, len(cropped), num_col)])


def add_orientation_labels(ax: 'mpl_axes.Axes', radius: tuple[int, int] = (15, 15),
                           letters: tuple[str, str, str, str] = ('A', 'P', 'L', 'R')):
    """
    Add orientation labels (A, P, L, R) to a figure, yellow with a black outline.
    """
    # Ensure that letter locations are determined as a function of the bounding box. For a 15,15 radius (30x30):
    #    A                    [12,  6]
    # L     R   -->  [0, 17]            [24, 17]
    #    P                    [12, 28]
    for letter, x, y, in [
        (letters[0], radius[1] - 3,   6),
        (letters[1], radius[1] - 3,   radius[0]*2 - 2),
        (letters[2], 0,               radius[0] + 2),
        (letters[3], radius[1]*2 - 6, radius[0] + 2)
    ]:
        ax.text(x, y, letter, color='yellow', size=4).set_path_effects([
            mpl_patheffects.Stroke(linewidth=1, foreground='black'),
            mpl_patheffects.Normal(),
        ])


def add_segmentation_labels(ax: 'mpl_axes.Axes', seg_mosaic: np.ndarray, colors: list[str],
                            radius: tuple[int, int] = (15, 15)):
    """
    Add labels corresponding to the value of the segmentation for each slice in the mosaic.
    """
    # Fetch mosaic shape properties
    bbox = [2*radius[0], 2*radius[1]]
    grid_shape = [s // bb for s, bb in zip(seg_mosaic.shape, bbox)]
    # Fetch set of labels in the mosaic (including labels in between min/max)
    labels = [float(val) for val in range(int(np.unique(seg_mosaic).min()),
                                          int(np.unique(seg_mosaic).max())+1)]
    # Iterate over each sub-array in the mosaic
    for row in range(grid_shape[0]):
        for col in range(grid_shape[1]):
            # Fetch sub-array from mosaic
            extents = (slice(row*bbox[0], (row+1)*bbox[0]),
                       slice(col*bbox[1], (col+1)*bbox[1]))
            arr = seg_mosaic[extents]
            # Check for nonzero labels, then draw text for each label found
            labels_in_arr = [v for v in np.unique(arr) if v]
            for idx_pos, l_arr in enumerate(labels_in_arr, start=1):
                lr_shift = -4 * (len(str(int(l_arr))) - 1)
                y, x = (extents[0].stop - 6*idx_pos + 3,  # Shift each subsequent label up in case there are >1
                        extents[1].stop - 6 + lr_shift)   # Shift labels left if double/triple digit
                color = colors[0] if len(colors) == 1 else colors[labels.index(l_arr)]
                ax.text(x, y, str(int(l_arr)), color=color, size=4).set_path_effects([
                    mpl_patheffects.Stroke(linewidth=1, foreground='black'),
                    mpl_patheffects.Normal(),
                ])


def equalize_histogram(img: np.ndarray):
    """
    Perform histogram equalization using CLAHE.

    Notes:
    - Image value range is preserved
    - Workaround for adapthist artifact by padding (#1664)
    """
    winsize = 16
    min_, max_ = img.min(), img.max()
    b = (np.float32(img) - min_) / (max_ - min_)
    b[b >= 1] = 1  # 1+eps numerical error may happen (#1691)

    h, w = b.shape
    h1 = (h + (winsize - 1)) // winsize * winsize
    w1 = (w + (winsize - 1)) // winsize * winsize
    if h != h1 or w != w1:
        b1 = np.zeros((h1, w1), dtype=b.dtype)
        b1[:h, :w] = b
        b = b1
    c = skimage.exposure.equalize_adapthist(b, kernel_size=(winsize, winsize))
    if h != h1 or w != w1:
        c = c[:h, :w]

    return np.array(c * (max_ - min_) + min_, dtype=img.dtype)


def plot_outlines(img: np.ndarray, ax: 'mpl_axes.Axes', **kwargs):
    """
    Draw the outlines of every equal-value area of a 2D Numpy array with Matplotlib.

    kwargs are forwarded to `matplotlib.collections.PolyCollection` for styling.
    """
    # To see where each (row, column) index shows up in the graphic
    # (left, right, top, bottom), see:
    # https://matplotlib.org/stable/users/explain/artists/imshow_extent.html#default-extent

    # Add a frame of zeros around the image to account for cells on the border
    padded = np.pad(img, 1)

    # First we compute the endpoints of every horizontal segment of the outline

    # edge[r, c] is True when there should be a visible horizontal outline
    # between img[r, c+1] and img[r+1, c+1]
    edge = (padded[:-1, :] != padded[1:, :])
    # run[r, c] is True when a run of horizontal edges starts or stops at the
    # upper left corner of img[r, c]
    run = (edge[:, :-1] != edge[:, 1:])
    # Get the coordinates from the big boolean array. They are sorted so that
    # the horizontal segments start at even vertices and stop at the next (odd)
    # vertex: v0--v1, v2--v3, v4--v5, ...
    vertices = sorted(map(tuple, np.argwhere(run)))
    # Given a vertex, we want to quickly travel to its horizontal neighbour
    # i^1 == i+1 when i is even (v0->v1, v2->v3, ...)
    # i^1 == i-1 when i is odd (v1->v0, v3->v2, ...)
    horizontal_segment = {v: vertices[i ^ 1] for i, v in enumerate(vertices)}

    # Second, we compute endpoints for the vertical segments

    # edge[r, c] is True when there should be a visible vertical outline
    # between img[r+1, c] and img[r+1, c+1]
    edge = (padded[:, :-1] != padded[:, 1:])
    # run[r, c] is True when a run of vertical edges starts or stops at the
    # upper left corner of img[r, c]
    run = (edge[:-1, :] != edge[1:, :])
    # Get the coordinates from the big boolean array. They are sorted so that
    # the vertical segments start at even vertices and stop at the next (odd)
    # vertex: v0--v1, v2--v3, v4--v5, ...
    vertices = sorted(map(tuple, np.argwhere(run)), key=lambda v: v[::-1])
    # Given a vertex, we want to quickly travel to its vertical neighbour
    # i^1 == i+1 when i is even (v0->v1, v2->v3, ...)
    # i^1 == i-1 when i is odd (v1->v0, v3->v2, ...)
    vertical_segment = {v: vertices[i ^ 1] for i, v in enumerate(vertices)}

    # Now we need to collect the horizontal and vertical segments into a list
    # of polygons. We may need some open-ended polygons, which start and end
    # at vertices which are in the middle of a perpendicular segment. And we
    # may need some closed polygons, which loop back to their starting vertex.

    # The open-ended polygons:
    open_polygons = []
    open_vertices = set(horizontal_segment).symmetric_difference(vertical_segment)
    while open_vertices:
        polygon = []
        vertex = open_vertices.pop()
        # Build the polygon by listing its vertices in order. We remove
        # vertices from horizontal_segment and vertical_segment as we trace
        # over them with the polygon.
        while True:
            polygon.append(vertex)
            if vertex in horizontal_segment:
                # Move to the other endpoint of the segment, and make sure to
                # remove both endpoints from the dictionary.
                vertex = horizontal_segment.pop(vertex)
                del horizontal_segment[vertex]
            elif vertex in vertical_segment:
                # Move to the other endpoint of the segment, and make sure to
                # remove both endpoints from the dictionary.
                vertex = vertical_segment.pop(vertex)
                del vertical_segment[vertex]
            else:
                # We have reached the end of the current open-ended polygon.
                open_vertices.remove(vertex)
                # Convert (row, column) coordinates to (x, y)
                open_polygons.append(np.array(polygon)[:, ::-1] - 0.5)
                break

    # The closed polygons:
    closed_polygons = []
    while horizontal_segment:
        polygon = []
        # Remove both endpoints of an arbitrary segment.
        vertex, other = horizontal_segment.popitem()
        del horizontal_segment[other]
        while True:
            polygon.append(vertex)
            if vertex in horizontal_segment:
                # Move to the other endpoint of the segment, and make sure to
                # remove both endpoints from the dictionary.
                vertex = horizontal_segment.pop(vertex)
                del horizontal_segment[vertex]
            elif vertex in vertical_segment:
                # Move to the other endpoint of the segment, and make sure to
                # remove both endpoints from the dictionary.
                vertex = vertical_segment.pop(vertex)
                del vertical_segment[vertex]
            else:
                # We have reached the end of the current closed polygon.
                # Convert (row, column) coordinates to (x, y)
                closed_polygons.append(np.array(polygon)[:, ::-1] - 0.5)
                break
    assert not vertical_segment

    # Draw the outline
    ax.add_collection(mpl_collections.PolyCollection(
        open_polygons, closed=False, **kwargs))
    ax.add_collection(mpl_collections.PolyCollection(
        closed_polygons, closed=True, **kwargs))


def assign_label_colors_by_groups(labels):
    """
    This function handles label colors when labels may not be in a single continuous group:

        - Normal vertebral labels:  [2, 3, 4, 5, 6, ...]
        - Edge case, TotalSegmentator labels: [31, 32, 33, 200, 201, 217, 218, 219]

    We assume that the subgroups of labels (1: [31, 32, 33, ...], 2: [200, 201], 3: [217, 218, 219, ...])
    should each be assigned their own distinct colormap, as to group them semantically.
    """
    # Arrange colormaps for max contrast between colormaps, and max contrast between colors in colormaps
    # Put reds first because SC labels tend to be ~1, so they will usually be first in a list of labels
    distinct_colormaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']
    colormap_sampling = [0.25, 0.5, 0.75, 0.5]  # light -> medium -> dark -> medium -> (repeat)

    # Split labels into subgroups --> we split the groups wherever the difference between labels is > 1
    start_end = [0, len(labels)]
    for idx, (prev, curr) in enumerate(zip(labels, labels[1:]), start=1):
        if curr - prev > 1:
            start_end.insert(len(start_end) - 1, idx)
    label_groups = [labels[start:end] for start, end in zip(start_end, start_end[1:])]

    # Handle the usual case: A single continuous group (likely vertebral labels)
    # Contrast ratios against #000000 taken from https://webaim.org/resources/contrastchecker/
    labels_color = [
        "#ffffff",  # White       (21.00:1)
        "#F28C28",  # Orange      ( 8.55:1)
        "#0096FF",  # Blue        ( 6.80:1)
        "#ffee00",  # Yellow      (17.48:1)
        "#ff0000",  # Red         ( 5.25:1)
        "#50ff30",  # Green       (15.68:1)
        "#F749FD",  # Magenta     ( 7.32:1)
    ]
    if len(label_groups) == 1:
        # repeat high-contrast colors until we have enough to cover the range of labels
        n_colors = labels.max() - labels.min() + 1
        color_list = list(it.islice(it.cycle(labels_color), n_colors))

    # Handle the edge case: Multiple continuous groups
    else:
        # Initialize a list by repeating the color black (#000000) to fill in the gaps between colors.
        # We do this because matplotlib applies colormaps by scaling both the data and the colormap to [0, 1].
        # Without filling in the gaps between groups, the colormap would be scaled incorrectly relative to the data.
        # ((Note that, if done right, the #000000 color should never be assigned to our label values.))
        color_list = ['#000000'] * (labels.max() - labels.min() + 1)
        # Assign a colormap to each group of labels (while sampling the colormap at different points)
        for i, label_group in enumerate(label_groups):
            colormap = mpl_plt.get_cmap(distinct_colormaps[i % len(distinct_colormaps)])
            sampled_colors = [mpl_colors.to_hex(c) for c in [colormap(n) for n in colormap_sampling]]
            # Then, assign a color to each label within the group
            for j, label in enumerate(label_group):
                color_list[label - labels.min()] = sampled_colors[j % len(sampled_colors)]

    return color_list


def crop_with_mask(img_to_crop, img_ref, pad=3, max_slices=None):
    """
    Crop array along a specific axis based on nonzero slices in the reference image.

    Use `pad` if you want to pad around the mask (no matter how big the mask is).

    Use `max_slices` to pad only until that amount of slices is reached (overrides `pad`). If
    the segmentation spans more slices than `max_slices`, then no padding will occur. Instead,
    all slices containing the segmentation will be used (to preserve the segmentation).
    """
    if not np.count_nonzero(img_ref.data):
        raise ValueError("The mask image is empty. Cannot crop using an empty mask. Check the input (e.g. '-qc-seg').")
    # QC images are reoriented to SAL (axial) or RSP (sagittal) such that axis=0 is always the slice index
    axis = 0
    # get extents of segmentation used for cropping
    first_slice = min(np.where(img_ref.data)[axis])
    last_slice = max(np.where(img_ref.data)[axis])
    # if `max_slices` is specified, then override `pad`
    if max_slices is not None:
        # use `max(0, ...)` to avoid cropping the segmentation if it would exceed `max_slices`
        pad_total = max(0, max_slices - (last_slice - first_slice + 1))
        l_pad = math.floor(pad_total / 2)
        r_pad = math.ceil(pad_total / 2)
    # otherwise, use the provided value as-is
    else:
        l_pad = r_pad = pad
    # pad (but make sure the index slices are within the image bounds)
    start_slice = max(first_slice - l_pad, 0)
    stop_slice = min(last_slice + r_pad, img_ref.data.shape[axis] - 1)
    # crop the image at the specified axis
    cropper = ImageCropper(img_in=img_to_crop)
    cropper.bbox = BoundingBox(xmin=start_slice, xmax=stop_slice,
                               ymin=0, ymax=img_to_crop.data.shape[1]-1,
                               zmin=0, zmax=img_to_crop.data.shape[2]-1)
    img_cropped = cropper.crop()
    # since `ImageCropper` returns a copy of the image, we need to update the original image
    # we could instead just return the new copy, but that would require refactoring the qc
    # function to no longer iterate over the images in-place, which would be a larger change
    img_to_crop.data = img_cropped.data
    img_to_crop.hdr = img_cropped.hdr


def get_max_axial_radius(img):
    """
    Determine the maximum slicewise width/height of the nonzero voxels in img.

    Input images should be SAL, such that each SI axial slice is composed of [1] -> AP and [2] -> LR.
    """
    # In Axial plane, the radius is the maximum width/height of the spinal cord mask dilated by 20% or 15, whichever is larger.
    dilation = 1.2
    radius_default = 15
    heights = [np.max(np.where(slc)[0]) - np.min(np.where(slc)[0]) if np.sum(slc) > 0 else 0 for slc in img.data]
    widths = [np.max(np.where(slc)[1]) - np.min(np.where(slc)[1]) if np.sum(slc) > 0 else 0 for slc in img.data]
    radii_h = [int((h * dilation)//2) for h in heights]
    radii_w = [int((w * dilation)//2) for w in widths]
    max_radius = max([radius_default] + radii_h + radii_w)
    return max_radius, max_radius


def get_max_sagittal_radius(img):
    """
    Determine the maximum slicewise width/height of the nonzero voxels for sagittal images.

    Input images should be RSP, such that each LR sagittal slice is composed of [1] -> SI and [2] -> PA.
    """
    # In Sagittal plane, the radius is the maximum width of the spinal cord mask dilated by 20% or 1/2 of the image width, whichever is larger.
    # The height is always the entirety of the image height (for example, to view possible lesions in the brain stem)
    widths = [np.max(np.where(slice)[1]) - np.min(np.where(slice)[1]) if np.sum(slice) > 0 else 0 for slice in img.data]
    widths = [w//2 + 0.1*w//2 for w in widths]
    height = np.floor(img.data.shape[2]/2).astype(int)
    return height, max(np.floor(img.data.shape[1]/4).astype(int), int(max(widths)))
