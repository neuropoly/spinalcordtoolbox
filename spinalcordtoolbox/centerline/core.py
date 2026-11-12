"""
Core functions dealing with centerline extraction from 3D data

Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import os
import logging
import numpy as np

from spinalcordtoolbox.image import Image, zeros_like, add_suffix, reorient_coordinates
from spinalcordtoolbox.centerline import curve_fitting
from spinalcordtoolbox.utils.fs import tmp_create

logger = logging.getLogger(__name__)


class ParamCenterline:
    """Default parameters for centerline fitting"""
    def __init__(self, algo_fitting='bspline', degree=5, smooth=20, contrast=None, minmax=True, soft=0):
        """

        :param algo_fitting: str:
          polyfit: Polynomial fitting\
          bspline: b-spline fitting\
          linear: linear interpolation followed by smoothing with Hanning window\
          nurbs: Non-Uniform Rational Basis Splines. See [De Leener et al., MIA, 2016]\
          optic: Automatic segmentation using SVM and HOG. See [Gros et al., MIA 2018].\
        :param degree: Degree of polynomial function. Only for polyfit.
        :param smooth: Degree of smoothness. Only for bspline and linear. When using algo_fitting=linear, it corresponds
        :param contrast: Contrast type for algo_fitting=optic.
        :param minmax: Crop output centerline where the segmentation starts/end. If False, centerline will span all\
          slices to the size of a Hanning window (in mm).
        :param soft: Binary or soft centerline. Only relevant with algo_fitting polyfit, bspline, linear, nurbs.
        """
        self.algo_fitting = algo_fitting
        self.contrast = contrast
        self.degree = degree
        self.smooth = smooth
        self.soft = soft
        self.minmax = minmax


class FitResults:
    """
    Collection of metrics to assess fitting performance
    """
    def __init__(self):

        class Data:
            """Raw and fitted data"""
            def __init__(self):
                self.xmean = None
                self.xfit = None
                self.ymean = None
                self.yfit = None
                self.zmean = None
                self.zref = None

        self.rmse = None  # RMSE
        self.laplacian_max = None  # Maximum of 2nd derivatives
        self.data = Data()  # Raw and fitted data (for plotting in QC report)
        self.param = None  # ParamCenterline()


def find_and_sort_coord(img):
    """
    Find x,y,z coordinate of centerline and output an array which is sorted along SI direction. Removes any duplicate
    along the SI direction by averaging across the same ind_SI.

    :param img: Image(): Input image. Could be any orientation.
    :return: nx3 numpy array with X, Y, Z coordinates of center of mass
    """
    # TODO: deal with nan, etc.
    # Get indices of non-zero values
    arr = np.array(np.where(img.data))
    # Sort indices according to SI axis
    dim_si = [img.orientation.find(x) for x in ['I', 'S'] if img.orientation.find(x) != -1][0]
    # Loop across SI axis and average coordinates within duplicate SI values
    sorted_avg = []
    for i_si in np.unique(arr[dim_si]):
        sorted_avg.append(arr[:, arr[dim_si] == i_si].mean(axis=1))
    return np.array(sorted_avg).T


def get_centerline(im_seg, param=ParamCenterline(), verbose=1, remove_temp_files=1, space='pix'):
    """
    Extract centerline from an image (using optic) or from a binary or weighted segmentation (using the center of mass).

    :param im_seg: Image(): Input segmentation or series of points along the centerline.
    :param param: ParamCenterline() class:
    :param verbose: int: verbose level
    :param remove_temp_files: int: Whether to remove temporary files. 0 = no, 1 = yes.
    :param space: string: Defines the coordinate space in which to output Centerline arrays (`arr_*`).
                          (NOTE: This parameter has no bearing on the orientation of the output coordinates.)
                             - `space='pix'`: The coordinates are in pixel space, and the orientation is the same as the input image.
                             - `space='phys'`: The coordinates are in physical (scanner) space, and the orientation will be RAS+, as
                                               defined by the affine matrix transformation according to the NIFTI spec.
    :return: im_centerline: Image: Centerline in discrete coordinate (int)
    :return: arr_centerline: 3x1 array: Centerline in continuous coordinate (float) for each slice.
    :return: arr_centerline_deriv: 3x1 array: Derivatives of x and y centerline wrt. z for each slice.
    :return: fit_results: FitResults class
    """
    if not isinstance(im_seg, Image):
        raise ValueError("Expecting an image")
    if space not in ['pix', 'phys']:
        raise ValueError(f"'space' parameter must be either 'pix' or 'phys', but '{space}' was passed instead.")

    # Handle whether we should keep the centerline as a temporary output
    if remove_temp_files:
        fname_ctr_temp = None  # Rather than creating then deleting the temp output, we can just skip creating it at all
    else:
        temp_path = tmp_create(basename="get-centerline")
        fname_ctr = (add_suffix(os.path.basename(im_seg.absolutepath), "_ctr") if im_seg.absolutepath
                     else "centerline.nii.gz")
        fname_ctr_temp = os.path.join(temp_path, fname_ctr)

    # Open image and change to RPI orientation
    native_orientation = im_seg.orientation
    im_seg = im_seg.copy().change_orientation('RPI')  # Copy to avoid mutating input image

    # The 'optic' method is particular compared to the other methods, as here we estimate the centerline based on the
    # image itself (not the segmentation). This means we do not apply the same fitting procedure and centerline
    # creation that is used for the other methods.
    if param.algo_fitting == 'optic':
        from spinalcordtoolbox.centerline import optic
        if param.contrast is None:
            raise ValueError("Missing param.contrast")
        im_centerline = optic.detect_centerline(im_seg, param.contrast, remove_temp_files, verbose)
        x_centerline_fit, y_centerline_fit, z_centerline = find_and_sort_coord(im_centerline)
        # Compute derivatives using polynomial fit
        _, x_centerline_deriv = curve_fitting.polyfit_1d(z_centerline, x_centerline_fit, z_centerline, deg=param.degree)
        _, y_centerline_deriv = curve_fitting.polyfit_1d(z_centerline, y_centerline_fit, z_centerline, deg=param.degree)
        # rename 'z' variable to match the name used the 'non-optic' methods
        z_ref = z_centerline
        # Fit metrics aren't computed for 'optic`, so return 'None'
        fit_results = None

    # All other 'non-optic' methods involve segmentation-based curve fitting, which involves a number of pre- and
    # post-processing steps that are separate from the 'optic' method.
    else:
        # get pixdim (i.e. voxel sizes) along x/y/z axes (used in some centerline methods, and to estimate fit metrics)
        px, py, pz = im_seg.dim[4:7]
        # Take the center of mass at each slice to avoid: https://stackoverflow.com/q/2009379
        x_mean, y_mean, z_mean = find_and_sort_coord(im_seg)
        # Crop output centerline to where the segmentation starts/end
        if param.minmax:
            z_ref = np.array(range(z_mean.min().astype(int), z_mean.max().astype(int) + 1))
        else:
            z_ref = np.array(range(im_seg.dim[2]))
        index_mean = np.array([list(z_ref).index(i) for i in z_mean])

        # Choose a non-optic method
        if param.algo_fitting == 'polyfit':
            x_centerline_fit, x_centerline_deriv = curve_fitting.polyfit_1d(z_mean, x_mean, z_ref, deg=param.degree)
            y_centerline_fit, y_centerline_deriv = curve_fitting.polyfit_1d(z_mean, y_mean, z_ref, deg=param.degree)
            fig_title = f'Algo={param.algo_fitting}, Deg={param.degree}'

        elif param.algo_fitting == 'bspline':
            x_centerline_fit, x_centerline_deriv = curve_fitting.bspline(z_mean, x_mean, z_ref, param.smooth, pz=pz)
            y_centerline_fit, y_centerline_deriv = curve_fitting.bspline(z_mean, y_mean, z_ref, param.smooth, pz=pz)
            fig_title = f'Algo={param.algo_fitting}, Smooth={param.smooth}'

        elif param.algo_fitting == 'linear':
            # Simple linear interpolation
            x_centerline_fit, x_centerline_deriv = curve_fitting.linear(z_mean, x_mean, z_ref, param.smooth, pz=pz)
            y_centerline_fit, y_centerline_deriv = curve_fitting.linear(z_mean, y_mean, z_ref, param.smooth, pz=pz)
            fig_title = f'Algo={param.algo_fitting}, Smooth={param.smooth}'

        elif param.algo_fitting == 'nurbs':
            from spinalcordtoolbox.centerline.nurbs import b_spline_nurbs
            point_number = 3000
            # Interpolate such that the output centerline has the same length as z_ref
            x_mean_interp, _ = curve_fitting.linear(z_mean, x_mean, z_ref, 0)
            y_mean_interp, _ = curve_fitting.linear(z_mean, y_mean, z_ref, 0)
            x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv, \
                z_centerline_deriv, error = b_spline_nurbs(x_mean_interp, y_mean_interp, z_ref, nbControl=None,
                                                           point_number=point_number, all_slices=True)
            # Normalize derivatives to z_deriv
            x_centerline_deriv = x_centerline_deriv / z_centerline_deriv
            y_centerline_deriv = y_centerline_deriv / z_centerline_deriv
            fig_title = f'Algo={param.algo_fitting}, NumberPoints={point_number}'

        else:
            logger.error(f'algo_fitting "{param.algo_fitting}" does not exist.')
            raise ValueError

        # Create an image with the centerline
        im_centerline = im_seg.copy()
        im_centerline.data = np.zeros(im_centerline.data.shape)
        # Binarized
        if param.soft == 0:
            # Assign value=1 to centerline. Make sure to clip to avoid array overflow.
            im_centerline.data[np.clip(x_centerline_fit.round().astype(int), 0, im_centerline.data.shape[0] - 1),
                               np.clip(y_centerline_fit.round().astype(int), 0, im_centerline.data.shape[1] - 1),
                               z_ref] = 1
        # Soft (accounting for partial volume effect)
        else:
            # Clip to avoid array overflow
            # Note: '-1' here is necessary because python does matrix indexing from 0
            x_max, y_max = im_centerline.data.shape[:2]
            x_centerline_fit = x_centerline_fit.clip(0, x_max - 1)
            y_centerline_fit = y_centerline_fit.clip(0, y_max - 1)

            x_floor = np.floor(x_centerline_fit).astype(int)
            y_floor = np.floor(y_centerline_fit).astype(int)

            x_ceil = np.ceil(x_centerline_fit).astype(int)
            y_ceil = np.ceil(y_centerline_fit).astype(int)

            x_frac = x_centerline_fit - x_floor
            y_frac = y_centerline_fit - y_floor
            # Note: we add ('+='), rather than assign ('=')
            # (see https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4026#discussion_r1096093021)
            im_centerline.data[x_floor, y_floor, z_ref] += (1 - x_frac) * (1 - y_frac)
            im_centerline.data[x_floor, y_ceil, z_ref] += (1 - x_frac) * y_frac
            im_centerline.data[x_ceil, y_floor, z_ref] += x_frac * (1 - y_frac)
            im_centerline.data[x_ceil, y_ceil, z_ref] += x_frac * y_frac
            # You can check if the sum of the voxels is equal to 1 with:
            # np.apply_over_axes(np.sum, im_centerline.data, [0, 1]).flatten()

        # Compute fitting metrics
        fit_results = FitResults()
        fit_results.rmse = np.sqrt(np.mean((x_mean - x_centerline_fit[index_mean]) ** 2) * px +
                                   np.mean((y_mean - y_centerline_fit[index_mean]) ** 2) * py)
        fit_results.laplacian_max = np.max([
            np.absolute(np.gradient(np.array(x_centerline_deriv * px))).max(),
            np.absolute(np.gradient(np.array(y_centerline_deriv * py))).max()])
        fit_results.data.zmean = z_mean
        fit_results.data.zref = z_ref
        fit_results.data.xmean = x_mean
        fit_results.data.xfit = x_centerline_fit
        fit_results.data.ymean = y_mean
        fit_results.data.yfit = y_centerline_fit
        fit_results.param = param

        # Display fig of fitted curves
        if verbose == 2:
            from datetime import datetime
            import matplotlib.pyplot as plt
            plt.figure(figsize=(16, 10))
            plt.subplot(3, 1, 1)
            plt.title(f"{fig_title}\nRMSE[mm]={fit_results.rmse:0.2f}, LaplacianMax={fit_results.laplacian_max:0.2f}")
            plt.plot(z_mean * pz, x_mean * px, 'ro')
            plt.plot(z_ref * pz, x_centerline_fit * px, 'k')
            plt.plot(z_ref * pz, x_centerline_fit * px, 'k.')
            plt.ylabel("X [mm]")
            plt.legend(['Reference', 'Fitting', 'Fitting points'])

            plt.subplot(3, 1, 2)
            plt.plot(z_mean * pz, y_mean * py, 'ro')
            plt.plot(z_ref * pz, y_centerline_fit * py, 'b')
            plt.plot(z_ref * pz, y_centerline_fit * py, 'b.')
            plt.xlabel("Z [mm]")
            plt.ylabel("Y [mm]")
            plt.legend(['Reference', 'Fitting', 'Fitting points'])

            plt.subplot(3, 1, 3)
            plt.plot(z_ref * pz, x_centerline_deriv * px, 'k.')
            plt.plot(z_ref * pz, y_centerline_deriv * py, 'b.')
            plt.grid(axis='y', color='grey', linestyle=':', linewidth=1)
            plt.axhline(color='grey', linestyle='-', linewidth=1)
            # plt.plot(z_ref * pz, z_centerline_deriv * pz, 'r.')
            plt.ylabel("dX/dZ, dY/dZ")
            plt.xlabel("Z [mm]")
            plt.legend(['X-deriv', 'Y-deriv'])

            plt.savefig(f'fig_centerline_{datetime.now():%y%m%d-%H%M%S%f}_{param.algo_fitting}.png')
            plt.close()

    # Construct the outputs (still in RPI- orientation)
    im_centerline_rpi = im_centerline
    arr_ctl_rpi = np.array([x_centerline_fit, y_centerline_fit, z_ref])
    arr_ctl_der_rpi = np.array([x_centerline_deriv, y_centerline_deriv, np.ones_like(z_ref)])

    # Reorient the outputs back to the original orientation (copy to avoid mutating `im_centerline_rpi`)
    im_centerline = im_centerline_rpi.copy().change_orientation(native_orientation)
    # Note: We transpose here because we need to go from [x_list, y_list, z_list] to list[[x, y, z], ...], then back
    arr_ctl = np.array(reorient_coordinates(arr_ctl_rpi.T, im_centerline_rpi, native_orientation, mode='absolute')).T
    arr_ctl_der = np.array(reorient_coordinates(arr_ctl_der_rpi.T, im_centerline_rpi, native_orientation, mode='relative')).T

    # If 'phys' is specified, adjust centerline coordinates (`Centerline.points`) and
    # derivatives (`Centerline.derivatives`) to physical ("phys") space and native (`im_seg`) orientation.
    if space == 'phys':
        # Transform centerline to physical coordinate system
        # Note: We **must** use the reoriented image (`im_centerline`), because the orientation of the affine must
        #       match the orientation of the coordinates. See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4622
        arr_ctl = im_centerline.transfo_pix2phys(arr_ctl.T, mode='absolute').T
        arr_ctl_der = im_centerline.transfo_pix2phys(arr_ctl_der.T, mode='relative').T

    # Save outputs
    if fname_ctr_temp:  # Preserve centerline image in tempdir for debugging purposes
        im_centerline.save(fname_ctr_temp, mutable=True)
    return (im_centerline,
            arr_ctl,
            arr_ctl_der,
            fit_results)


def _call_viewer_centerline(im_data, interslice_gap=20.0):
    # TODO: _call_viewer_centerline should not be "internal" anymore, i.e., remove the "_"
    """
    FIXME doc
    Call Qt viewer for manually selecting labels.

    :param im_data:
    :param interslice_gap:
    :return: Image() of labels.
    """
    from spinalcordtoolbox.gui.base import AnatomicalParams
    from spinalcordtoolbox.gui.centerline import launch_centerline_dialog

    if not isinstance(im_data, Image):
        raise ValueError("Expecting an image")

    # Get the number of slice along the (IS) axis
    im_tmp = im_data.copy().change_orientation('RPI')
    _, _, nz, _, _, _, pz, _ = im_tmp.dim
    del im_tmp

    params = AnatomicalParams()
    # setting maximum number of points to a reasonable value
    params.num_points = np.ceil(nz * pz / interslice_gap) + 2
    params.interval_in_mm = interslice_gap
    # Starting at the top slice minus 1 in cases where the first slice is almost zero, due to gradient
    # non-linearity correction:
    # https://forum.spinalcordmri.org/t/centerline-viewer-enhancements-sct-v5-0-1/605/4?u=jcohenadad
    params.starting_slice = 'top_minus_one'

    im_mask_viewer = zeros_like(im_data)
    launch_centerline_dialog(im_data, im_mask_viewer, params)

    return im_mask_viewer
