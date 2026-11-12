"""
Quality Control report generator

Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import logging
from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np
import skimage
import skimage.io
import skimage.exposure

from spinalcordtoolbox.image import Image, check_image_kind
from spinalcordtoolbox.reports.qc2 import create_qc_entry, add_slice_numbers
from spinalcordtoolbox.reports.slice import Slice, Axial, Sagittal
from spinalcordtoolbox.utils.sys import list2cmdline, LazyLoader

mpl_figure = LazyLoader("mpl_figure", globals(), "matplotlib.figure")
mpl_axes = LazyLoader("mpl_axes", globals(), "matplotlib.axes")
mpl_colors = LazyLoader("mpl_colors", globals(), "matplotlib.colors")
mpl_backend_agg = LazyLoader("mpl_backend_agg", globals(), "matplotlib.backends.backend_agg")
mpl_animation = LazyLoader("mpl_animation", globals(), "matplotlib.animation")
mpl_patheffects = LazyLoader("mpl_patheffects", globals(), "matplotlib.patheffects")


logger = logging.getLogger(__name__)


class QcImage:
    """
    Class used to create a .png file from a 2d image produced by the class "Slice"
    """
    _color_bin_green = ["#ffffff", "#00ff00"]
    _color_bin_red = ["#ffffff", "#ff0000"]
    _seg_colormap = ["#4d0000", "#ff0000"]
    _ctl_colormap = ["#ff000099", '#ffff00']

    def listed_seg(self, mask, ax):
        """Create figure with red segmentation. Common scenario."""
        img = np.ma.masked_equal(mask, 0)
        ax.imshow(img,
                  cmap=mpl_colors.LinearSegmentedColormap.from_list("", self._seg_colormap),
                  norm=mpl_colors.Normalize(vmin=0.5, vmax=1),
                  interpolation=self.interpolation,
                  aspect=float(self.aspect_mask))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def template(self, mask, ax):
        """Show template statistical atlas"""
        values = mask
        values[values < 0.5] = 0
        color_white = mpl_colors.colorConverter.to_rgba('white', alpha=0.0)
        color_blue = mpl_colors.colorConverter.to_rgba('blue', alpha=0.7)
        color_cyan = mpl_colors.colorConverter.to_rgba('cyan', alpha=0.8)
        cmap = mpl_colors.LinearSegmentedColormap.from_list('cmap_atlas',
                                                            [color_white, color_blue, color_cyan], N=256)
        ax.imshow(values,
                  cmap=cmap,
                  interpolation=self.interpolation,
                  aspect=self.aspect_mask)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def no_seg_seg(self, mask, ax):
        """Create figure with image overlay. Notably used by sct_registration_to_template"""
        ax.imshow(mask, cmap='gray', interpolation=self.interpolation, aspect=self.aspect_mask)
        self._add_orientation_label(ax)

        # If Axial, overlay slice numbers in the top‐left of each tile
        if self.plane == 'Axial':
            add_slice_numbers(
                ax,
                num_slices=self.num_slices,
                radius=(15, 15),  # same “radius” as in slice.py
                margin=2,         # pixels inset from top‐left corner
                reverse=("I" in self.orig_orientation)
            )

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def highlight_pmj(self, mask, ax):
        """Hook to show a rectangle where PMJ is on the slice"""
        y, x = np.where(mask == 50)
        img = np.full_like(mask, np.nan)
        ax.imshow(img, cmap='gray', alpha=0, aspect=float(self.aspect_mask))
        ax.plot(x, y, 'x', color='lime', markersize=6)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def label_centerline(self, mask, ax):
        """Create figure with red label. Common scenario."""
        results_mask_pixels = np.where(mask > 0)
        # TODO: maybe we only need one pixel per centerline (currently, it's a 1x2 matrix of pixels)
        listOfCoordinates = list(zip(results_mask_pixels[0], results_mask_pixels[1]))
        for cord in listOfCoordinates:
            ax.plot(cord[1], cord[0], 'ro', markersize=1)
            # ax.text(cord[1]+5,cord[0]+5, str(mask[cord]), color='lime', clip_on=True)
        img = np.rint(np.ma.masked_where(mask < 1, mask))
        ax.imshow(img,
                  cmap=mpl_colors.ListedColormap(self._color_bin_red),
                  norm=mpl_colors.Normalize(vmin=0, vmax=1),
                  interpolation=self.interpolation,
                  alpha=1,
                  aspect=float(self.aspect_mask))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def vertical_line(self, mask, ax):
        """Centered vertical line to assess quality of straightening"""
        img = np.full_like(mask, np.nan)
        ax.imshow(img, cmap='gray', alpha=0, aspect=float(self.aspect_mask))
        ax.axvline(x=img.shape[1] / 2.0, color='r', linewidth=2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def grid(self, mask, ax):
        """Centered grid to assess quality of motion correction"""
        grid = np.full_like(mask, 0)
        ax.imshow(grid, cmap='gray', alpha=0, aspect=float(self.aspect_mask))
        for center_mosaic in self._centermass:
            x0, y0 = center_mosaic[0], center_mosaic[1]
            ax.axvline(x=x0, color='w', linestyle='-', linewidth=0.5)
            ax.axhline(y=y0, color='w', linestyle='-', linewidth=0.5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def smooth_centerline(self, mask, ax):
        """Display smoothed centerline"""
        mask = mask/mask.max()
        mask[mask < 0.05] = 0  # Apply 0.5 threshold
        img = np.ma.masked_equal(mask, 0)
        ax.imshow(img,
                  cmap=mpl_colors.LinearSegmentedColormap.from_list("", self._ctl_colormap),
                  norm=mpl_colors.Normalize(vmin=0, vmax=1),
                  interpolation=self.interpolation,
                  aspect=float(self.aspect_mask))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def _make_QC_image_for_3d_volumes(self, img, mask, plane, imgs_to_generate):
        """
        Create overlay and background images for all processes that deal with 3d volumes
        (all except sct_fmri_moco and sct_dmri_moco)

        :param img: The base image to display underneath the overlays (typically anatomical)
        :param mask: A list of images to be processed and overlaid on top of `img`
        :return:
        """

        img = self._equalize_histogram(img)

        # NB: `size_fig` is in inches. So, when size_fig == 5", then: dpi=300 --> 1500px, dpi=100 --> 500px, etc.
        size_fig = [
            5, (5
                # Sagittal slice QCs for isotropic images will often have similar heights and widths. However,
                # anisotropic images (and slice mosaics) will often have smaller or larger height than width:
                #  - Axial mosaic: e.g. WxH = 20x3 slice images.
                #  - Sagittal mosaic: e.g. WxH = 3x20 slice images
                # So, we want to scale the fig height to match the height of the image:
                * (img.shape[0] / img.shape[1])

                # However, we also need to account for the resolutions of each axis when scaling the height. Otherwise,
                # highly anisotropic images (e.g. T2*) will be shrunken down to a tiny height. (See #4563).
                # NB: self.aspect_img == (pix[0] / pix[1]), so combined, this is: (shape[0]*pix[0] / shape[1]*pix[1])
                * float(self.aspect_img))
        ]

        fig = mpl_figure.Figure()
        fig.set_size_inches(size_fig[0], size_fig[1], forward=True)
        mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        QcImage.no_seg_seg(self, img, ax)

        logger.info(str(imgs_to_generate['path_background_img']))
        self._save(fig, str(imgs_to_generate['path_background_img']), dpi=self.dpi)

        fig = mpl_figure.Figure()
        fig.set_size_inches(size_fig[0], size_fig[1], forward=True)
        mpl_backend_agg.FigureCanvasAgg(fig)
        for i, action in enumerate(self.action_list):
            logger.debug('Action List %s', action.__name__)
            if action.__name__ in ("no_seg_seg",):
                logger.debug("Mask type %s", mask[i].dtype)
                mask[i] = self._equalize_histogram(mask[i])
            ax = fig.add_axes((0, 0, 1, 1), label=str(i))
            action(self, mask[i], ax)
        self._save(fig, str(imgs_to_generate['path_overlay_img']), dpi=self.dpi)

    def _make_QC_image_for_4d_volumes(self, images_after_moco, images_before_moco, imgs_to_generate):
        """
        Generate background and overlay gifs for sct_fmri_moco and sct_dmri_moco

        :param images_after_moco: list of mosaic images after motion correction
        :param images_before_moco: list of mosaic images before motion correction
        :return:
        """

        size_fig = [5, 10 * images_after_moco[0].shape[0] / images_after_moco[0].shape[1] + 0.5]
        for i in range(len(images_after_moco)):
            images_after_moco[i] = self._equalize_histogram(images_after_moco[i])
            images_before_moco[i] = self._equalize_histogram(images_before_moco[i])

        self._generate_and_save_gif(images_before_moco, images_after_moco, imgs_to_generate, size_fig)
        self._generate_and_save_gif(images_before_moco, images_after_moco, imgs_to_generate, size_fig, is_mask=True)

    def _equalize_histogram(self, img):
        """
        Perform histogram equalization using CLAHE

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

    def _add_orientation_label(self, ax):
        """
        Add orientation labels on the figure

        :param fig: MPL figure handler
        :return:
        """
        if self.plane == 'Axial':
            # If mosaic of axial slices, display orientation labels
            text_a = ax.text(12, 6, 'A', color='yellow', size=4)
            text_p = ax.text(12, 28, 'P', color='yellow', size=4)
            text_l = ax.text(0, 18, 'L', color='yellow', size=4)
            text_r = ax.text(24, 18, 'R', color='yellow', size=4)
            # Add a black outline surrounding the text
            text_a.set_path_effects([mpl_patheffects.Stroke(linewidth=1, foreground='black'), mpl_patheffects.Normal()])
            text_p.set_path_effects([mpl_patheffects.Stroke(linewidth=1, foreground='black'), mpl_patheffects.Normal()])
            text_l.set_path_effects([mpl_patheffects.Stroke(linewidth=1, foreground='black'), mpl_patheffects.Normal()])
            text_r.set_path_effects([mpl_patheffects.Stroke(linewidth=1, foreground='black'), mpl_patheffects.Normal()])

    def _generate_and_save_gif(self, top_images, bottom_images, imgs_to_generate, size_fig, is_mask=False):
        """
        Create figure with two images for sct_fmri_moco and sct_dmri_moco and save gif

        :param top_images: list of images of mosaic before motion correction
        :param bottom_images: list of images of mosaic after motion correction
        :param size_fig: size of figure in inches
        :param is_mask: display grid on top of mosaic
        :return:
        """

        if is_mask:
            aspect = self.aspect_mask
        else:
            aspect = self.aspect_img

        fig = mpl_figure.Figure()
        mpl_backend_agg.FigureCanvasAgg(fig)
        fig.set_size_inches(size_fig[0], size_fig[1], forward=True)
        fig.subplots_adjust(left=0, top=0.9, bottom=0.1)

        ax1 = fig.add_subplot(211)
        null_image = np.zeros(np.shape(top_images[0]))
        img1 = ax1.imshow(null_image, cmap='gray', aspect=float(aspect))
        ax1.set_title('Before motion correction', fontsize=8, loc='left', pad=2)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        self._add_orientation_label(ax1)
        if is_mask:
            QcImage.grid(self, top_images[0], ax1)

        ax2 = fig.add_subplot(212)
        img2 = ax2.imshow(null_image, cmap='gray', aspect=float(aspect))
        ax2.set_title('After motion correction', fontsize=8, loc='left', pad=2)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        self._add_orientation_label(ax2)
        if is_mask:
            QcImage.grid(self, bottom_images[0], ax2)

        ann = ax2.annotate('', xy=(0, .025), xycoords='figure fraction', horizontalalignment='left',
                           verticalalignment='bottom', fontsize=6)

        def update_figure(i):
            img1.set_data(top_images[i])
            img1.set_clim(vmin=np.amin(top_images[i]), vmax=np.amax(top_images[i]))
            img2.set_data(bottom_images[i])
            img2.set_clim(vmin=np.amin(bottom_images[i]), vmax=np.amax(bottom_images[i]))
            ann.set_text(f'Volume: {i + 1}/{len(top_images)}')

        # FuncAnimation creates an animation by repeatedly calling the function update_figure for each frame
        ani = mpl_animation.FuncAnimation(fig, update_figure, frames=len(top_images))

        if is_mask:
            gif_out_path = str(imgs_to_generate['path_overlay_img'])
        else:
            gif_out_path = str(imgs_to_generate['path_background_img'])

        if self._fps is None:
            self._fps = 5
        writer = mpl_animation.PillowWriter(self._fps)
        logger.info('Saving gif %s', gif_out_path)
        ani.save(gif_out_path, writer=writer, dpi=self.dpi)

    def _save(self, fig, img_path, format='png', bbox_inches='tight', pad_inches=0.00, dpi=300):
        """
        Save the current figure into an image.

        :param fig: Figure handler
        :param img_path: str: path of the folder where the image is saved
        :param format: str: image format
        :param bbox_inches: str
        :param pad_inches: float
        :param dpi: int: Output resolution of the image
        :return:
        """
        logger.debug('Save image %s', img_path)
        fig.savefig(img_path,
                    format=format,
                    bbox_inches=None,
                    transparent=True,
                    dpi=dpi)


def generate_qc(fname_in1, fname_in2=None, fname_seg=None, plane=None, args=None, path_qc=None, dataset=None,
                subject=None, contrast=None, process=None, fps=None, p_resample=None):
    """
    Generate a QC entry allowing to quickly review results. This function is the entry point and is called by SCT
    scripts (e.g. sct_propseg).

    :param fname_in1: str: File name of input image #1 (mandatory)
    :param fname_in2: str: File name of input image #2
    :param fname_seg: str: File name of input segmentation
    :param plane: str: Orientation of the QC. Can be: Axial, Sagittal.
    :param args: args from parent function
    :param path_qc: str: Path to save QC report
    :param dataset: str: Dataset name
    :param subject: str: Subject name
    :param contrast: str: Contrast/acquisition name. If not provided, this will fall back to the name of the
                          parent directory of `fname_in1` (which may not be meaningful for all directory layouts).
    :param process: str: Name of SCT function. e.g., sct_propseg
    :param fps: float: Number of frames per second for output gif images. Used only for sct_frmi_moco and sct_dmri_moco.
    :param p_resample: float: Resolution (in mm) to resample the image to. If not provided, resampling will fall back
                              to the default value of the specific QC report layout (typically no resampling, or 0.6mm).
                              To turn off resampling, pass `p_resample==0`.
    :return: None
    """
    dpi = 300  # Output resolution of the image
    p_resample_default = 0.6  # Resolution in mm to resample the image to

    # The following are the expected types for some variables that get values
    # assigned in all branches of the big `if...elif...elif...` construct below
    qcslice: Slice
    action_list: List[Callable[[QcImage, np.ndarray, mpl_axes.Axes], None]]
    qcslice_layout: Callable[[Slice],
                             Union[List[np.ndarray],
                                   Tuple[List[List[np.ndarray]], List[Tuple[int, int]]]]]

    # Get QC specifics based on SCT process
    # Axial orientation, switch between the image and the segmentation
    if process in ['sct_propseg', 'sct_deepseg_sc', 'sct_deepseg_gm']:
        plane = 'Axial'
        im_list = [Image(fname_in1), Image(fname_seg)]
        action_list = [QcImage.listed_seg]
        def qcslice_layout(x): return x.mosaic()
    # Axial orientation, switch between the image and the centerline
    elif process in ['sct_get_centerline']:
        plane = 'Axial'
        p_resample_default = None
        im_list = [Image(fname_in1), Image(fname_seg)]
        action_list = [QcImage.label_centerline]
        def qcslice_layout(x): return x.mosaic()
    # Axial orientation, switch between the image and the white matter segmentation (linear interp, in blue)
    elif process in ['sct_warp_template']:
        plane = 'Axial'
        im_list = [Image(fname_in1), Image(fname_seg)]
        action_list = [QcImage.template]
        def qcslice_layout(x): return x.mosaic()
    # Axial orientation, switch between gif image (before and after motion correction) and grid overlay
    elif process in ['sct_dmri_moco', 'sct_fmri_moco']:
        plane = 'Axial'
        if fname_seg is None:
            raise ValueError("Segmentation is needed to ensure proper cropping around spinal cord.")
        im_list = [Image(fname_in1), Image(fname_in2), Image(fname_seg)]
        action_list = [QcImage.grid]
        def qcslice_layout(x): return x.mosaics_through_time()
    # Sagittal orientation, display PMJ box
    elif process in ['sct_detect_pmj']:
        plane = 'Sagittal'
        p_resample_default = None
        im_list = [Image(fname_in1), Image(fname_seg)]
        action_list = [QcImage.highlight_pmj]
        def qcslice_layout(x): return x.single()
    # Sagittal orientation, static image
    elif process in ['sct_straighten_spinalcord']:
        plane = 'Sagittal'
        p_resample_default = None
        dpi = 100
        im_list = [Image(fname_in1), Image(fname_in1)]
        action_list = [QcImage.vertical_line]
        def qcslice_layout(x): return x.single()
    # Metric outputs (only graphs)
    elif process in ['sct_process_segmentation']:
        plane = 'Sagittal'
        p_resample_default = None
        dpi = 100  # bigger picture is needed for this special case, hence reduce dpi
        fname_list = [fname_in1]
        # fname_seg should be a list of 4 images: 3 for each operation in `action_list`, plus an extra
        # centerline image, which is needed to make `Sagittal.get_center_spit` work correctly
        fname_list.extend(fname_seg)
        im_list = [Image(fname) for fname in fname_list]
        action_list = [QcImage.smooth_centerline, QcImage.highlight_pmj, QcImage.listed_seg]
        def qcslice_layout(x): return x.single()
    elif process in ['sct_image_stitch']:
        plane = 'Sagittal'
        p_resample_default = None
        dpi = 150
        im_list = [Image(fname_in1), Image(fname_in2)]
        action_list = [QcImage.no_seg_seg]
        def qcslice_layout(x): return x.single()
    elif process in ['sct_deepseg_lesion']:
        # Note, spinal cord segmentation (fname_seg) is used to crop the input image.
        # Then, the input image (fname_in1) is overlaid by the lesion (fname_in2).
        im_list = [Image(fname_in1), Image(fname_in2), Image(fname_seg)]
        action_list = [QcImage.listed_seg]
        def qcslice_layout(x): return x.mosaic()[:2]
    else:
        raise ValueError(f"Unrecognized process: {process}")

    slice_subtypes = {'Axial': Axial, 'Sagittal': Sagittal}
    if plane not in slice_subtypes.keys():
        raise ValueError(f"Invalid plane '{plane}'. Valid choices are {slice_subtypes.keys()}.")
    SliceSubtype = slice_subtypes[plane]

    if p_resample is None:  # If no resample value is specified, fall back to default
        p_resample = p_resample_default
    elif p_resample == 0:   # If user specified `-resample 0`, turn off resampling
        p_resample = None

    qcslice = object.__new__(SliceSubtype)
    qcslice._images = list()  # 3d volumes
    qcslice._4d_images = list()  # 4d volumes
    qcslice._image_seg = None  # for cropping
    qcslice._absolute_paths = list()  # Used because change_orientation removes the field absolute_path
    image_ref = None  # first pass: we don't have a reference image to resample to
    orig_orientations = list()
    num_slices = list()
    for i, image in enumerate(im_list):
        img = image.copy()
        qcslice._absolute_paths.append(img.absolutepath)  # change_orientation removes the field absolute_path
        orig_orientations.append(img.orientation)
        img.change_orientation('SAL')
        num_slices.append(img.dim[0])  # number of z-slices in the image
        if p_resample:
            type_img = 'seg' if ('seg' in check_image_kind(img)) else 'im'  # condense seg/softseg into just 'seg'
            img_r = qcslice._resample_slicewise(img, p_resample, type_img=type_img, image_ref=image_ref)
        else:
            img_r = img.copy()
        if img_r.dim[3] == 1:   # If image is 3D, nt = 1
            qcslice._images.append(img_r)
            image_ref = qcslice._images[0]  # 2nd and next passes: we resample any image to the space of the first one
        else:
            qcslice._4d_images.append(img_r)
            # image_ref = qcslice._4d_images[0]  # img_dest is not covered for 4D volumes in resample_nib()

    qc_image = object.__new__(QcImage)
    qc_image.plane = plane
    qc_image.dpi = dpi
    qc_image.interpolation = 'none'
    qc_image.action_list = action_list
    qc_image.process = process
    qc_image._fps = fps
    qc_image._centermass = None  # center of mass returned by slice.Axial.get_center()
    qc_image.orig_orientation = orig_orientations[0]  # just take the first orientation
    qc_image.num_slices = num_slices[0]  # pass the number of z-slices in the images (for mosaic labeling)

    # Get the aspect ratio (height/width) based on pixel size. Consider only the first 2 slices.
    qc_image.aspect_img, qc_image.aspect_mask = qcslice.aspect()[:2]

    path_input = Path(fname_in1).resolve()
    path_qc = Path(path_qc)
    command = process
    cmdline = [command]
    if isinstance(args, str):
        cmdline = f"{command} {args}"
    elif isinstance(args, list):
        cmdline = list2cmdline([command] + args)
    else:
        cmdline = command

    if qc_image.process in ['sct_fmri_moco', 'sct_dmri_moco']:
        [images_after_moco, images_before_moco], centermass = qcslice_layout(qcslice)
        qc_image._centermass = centermass
        with create_qc_entry(
            path_input, path_qc, command, cmdline, plane, dataset, subject, contrast,
            image_extension='gif',
        ) as imgs_to_generate:
            qc_image._make_QC_image_for_4d_volumes(
                images_after_moco,
                images_before_moco,
                imgs_to_generate,
            )
    else:
        img, *mask = qcslice_layout(qcslice)
        with create_qc_entry(
            path_input, path_qc, command, cmdline, plane, dataset, subject, contrast,
        ) as imgs_to_generate:
            qc_image._make_QC_image_for_3d_volumes(img, mask, plane, imgs_to_generate)
