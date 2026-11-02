"""
Functions processing segmentation data

Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import math
import platform
import numpy as np
from skimage import measure, transform
import skimage
from scipy.ndimage import map_coordinates, gaussian_filter1d, zoom
import logging

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.aggregate_slicewise import Metric
from spinalcordtoolbox.centerline.core import get_centerline
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.registration.algorithms import compute_pca, find_angle_hog
from spinalcordtoolbox.utils.shell import parse_num_list_inv
from spinalcordtoolbox.utils.sys import sct_progress_bar

from spinalcordtoolbox.process_seg_debug import (
    create_regularized_hog_angle_plot, create_ap_diameter_plots,
    create_quadrant_area_plots, create_symmetry_plots
)

# NB: We use a threshold to check if an array is empty, instead of checking if it's exactly 0. This is because
# resampling can change 0 -> ~0 (e.g. 1e-16). See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3402
NEAR_ZERO_THRESHOLD = 1e-6
KEYS_DEFAULT = ['area',
                'angle_AP',
                'angle_RL',
                'diameter_AP',
                'diameter_AP_ellipse',
                'diameter_RL',
                'length_anterior',
                'length_posterior',
                'eccentricity',
                'orientation',
                'solidity',
                'length'
                ]
KEYS_HOG = ['centermass_x', 'centermass_y', 'angle_hog']
KEYS_QUADRANT = ['area_quadrant_anterior_left', 'area_quadrant_anterior_right',
                 'area_quadrant_posterior_left', 'area_quadrant_posterior_right']
KEYS_SYMMETRY = ['symmetry_dice_RL', 'symmetry_hausdorff_RL', 'symmetry_difference_RL',
                 'symmetry_dice_AP', 'symmetry_hausdorff_AP', 'symmetry_difference_AP']


def get_missing_slices(im, im_mask):
    """Check for slices in `im` that are not covered by `im_mask`."""
    def nonzero_axial_slices(arr):
        return set(np.where(np.any(arr != 0, axis=(0, 1)))[0])
    im, im_mask = im.copy().change_orientation('RPI'), im_mask.copy().change_orientation('RPI')
    return list(nonzero_axial_slices(im.data) - nonzero_axial_slices(im_mask.data))


def compute_shape(segmentation, image=None, angle_correction=True, centerline_path=None, param_centerline=None,
                  verbose=1, remove_temp_files=1, filter_size=5):
    """
    Compute morphometric measures of the spinal cord in the transverse (axial) plane from the segmentation.
    The segmentation could be binary or weighted for partial volume [0,1].

    :param segmentation: input segmentation. Could be either an Image or a file name.
    :param image: input image. Could be either an Image or a file name. Note that the image is necessary to turn on HOG/symmetry-based metrics
    :param angle_correction:
    :param centerline_path: path to image file to be used as a centerline for computing angle correction.
    :param param_centerline: see centerline.core.ParamCenterline()
    :param verbose:
    :param remove_temp_files: int: Whether to remove temporary files. 0 = no, 1 = yes.
    :param filter_size: int: size of the gaussian filter for regularization along z for rotation angle. 0: no regularization
    :return metrics: Dict of class Metric(). If a metric cannot be calculated, its value will be nan.
    :return fit_results: class centerline.core.FitResults()
    """
    property_list = KEYS_DEFAULT
    # HOG-related properties that are only available when image (`sct_process_segmentation -i`) is provided
    property_list_image = KEYS_HOG + KEYS_QUADRANT + KEYS_SYMMETRY
    if image is not None:
        property_list = property_list[:1] + property_list_image + property_list[1:]

    im_seg = Image(segmentation).change_orientation('RPI')
    if image is not None:
        im = Image(image).change_orientation('RPI')
        # Make sure the input image and segmentation have the same dimensions
        if im_seg.dim[:3] != im.dim[:3]:
            raise ValueError(
                f"The input segmentation image ({im_seg.path}) and the input image ({im.path}) do not have the same "
                f"dimensions. Please provide images with the same dimensions."
            )
    if centerline_path is not None:
        # check for slices in the input mask not covered by the centerline
        missing_slices = get_missing_slices(im=im_seg, im_mask=Image(centerline_path))
        if missing_slices:
            raise ValueError(
                f"The provided centerline does not cover slice(s) {parse_num_list_inv(missing_slices)} "
                "of the input mask. Cannot compute angle correction. Please fix the input images or disable angle "
                "correction."
            )

    # Getting image dimensions. x, y and z respectively correspond to RL, PA and IS.
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    pr = 0.1   # we use a fixed value to be independent from the input image resolution
    data_seg = im_seg.data
    X, Y, Z = (data_seg > NEAR_ZERO_THRESHOLD).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # Initialize dictionary of property_list, with 1d array of nan (default value if no property for a given slice).
    shape_properties = {key: np.full(nz, np.nan, dtype=np.double) for key in property_list}

    fit_results = None

    if angle_correction:
        # allow the centerline image to be bypassed (in case `im_seg` is irregularly shaped, e.g. GM/WM)
        if centerline_path:
            im_centerline = Image(centerline_path).change_orientation('RPI')
            im_centerline_r = resample_nib(im_centerline, new_size=[px, py, pz], new_size_type='mm',
                                           interpolation='linear')
        else:
            im_centerline_r = im_seg
        # compute the spinal cord centerline based on the spinal cord segmentation
        _, arr_ctl, arr_ctl_der, fit_results = get_centerline(im_centerline_r, param=param_centerline, verbose=verbose,
                                                              remove_temp_files=remove_temp_files)
        # the third column of `arr_ctl` contains the integer slice numbers, and the first two
        # columns of `arr_ctl_der` contain the x and y components of the centerline derivative
        deriv = {int(z_ref): arr_ctl_der[:2, index] for index, z_ref in enumerate(arr_ctl[2])}

    # Loop across z and compute shape analysis
    current_tforms = {}
    current_patches = {}
    for iz in sct_progress_bar(range(min_z_index, max_z_index + 1), unit='iter', unit_scale=False, desc="Compute shape analysis",
                               ncols=80):
        # Extract 2D patch
        current_patch = im_seg.data[:, :, iz]
        if angle_correction:
            # Extract tangent vector to the centerline (i.e. its derivative)
            tangent_vect = np.array([deriv[iz][0] * px, deriv[iz][1] * py, pz])
            # Compute the angle about AP axis between the centerline and the normal vector to the slice
            angle_AP_rad = math.atan2(tangent_vect[0], tangent_vect[2])
            # Compute the angle about RL axis between the centerline and the normal vector to the slice
            angle_RL_rad = math.atan2(tangent_vect[1], tangent_vect[2])
            # Apply affine transformation to account for the angle between the centerline and the normal to the patch
            tform = transform.AffineTransform(scale=(np.cos(angle_RL_rad), np.cos(angle_AP_rad)))
            # Convert to float64, to avoid problems in image indexation causing issues when applying transform.warp
            current_patch = current_patch.astype(np.float64)
            # TODO: make sure pattern does not go extend outside of image border
            current_patch_scaled = transform.warp(current_patch,
                                                  tform.inverse,
                                                  output_shape=current_patch.shape,
                                                  order=1,
                                                  )
        else:
            current_patch_scaled = current_patch
            angle_AP_rad, angle_RL_rad = 0.0, 0.0

        # Compute shape properties for this slice
        shape_property = _properties2d(current_patch_scaled, [px, py], pr, iz, verbose=verbose)
        if shape_property is not None:
            # Add custom fields
            shape_property['angle_AP'] = angle_AP_rad * 180.0 / math.pi     # convert to degrees
            shape_property['angle_RL'] = angle_RL_rad * 180.0 / math.pi     # convert to degrees
            shape_property['length'] = pz / (np.cos(angle_AP_rad) * np.cos(angle_RL_rad))
            # Loop across properties and assign values for function output
            for property_name in property_list:
                shape_properties[property_name][iz] = shape_property.get(property_name, np.nan)

        # Store the patches and transforms to use later after regularization
        current_tforms[iz] = tform if angle_correction else None
        current_patches[iz] = {
            'patch': current_patch_scaled,
            'angle_AP_rad': angle_AP_rad,
            'angle_RL_rad': angle_RL_rad
        }

    # Compute image-based shape properties
    if image is not None:
        shape_properties_image = _properties_image(
            im, nz, px, py, pz, pr, min_z_index, max_z_index, property_list_image,
            current_patches, current_tforms, angle_correction, filter_size, verbose
        )
        shape_properties.update(shape_properties_image)

    metrics = {}
    for key, value in shape_properties.items():
        # Making sure all entries added to metrics have results
        value = np.array(value)
        if value.size > 0:
            metrics[key] = Metric(data=value, label=key)

    return metrics, fit_results


def _properties_image(im, nz, px, py, pz, pr, min_z_index, max_z_index, property_list,
                      current_patches, current_tforms, angle_correction, filter_size, verbose):
    """
    Compute morphometric measures of the spinal cord in the transverse (axial) plane that specifically
    require an anatomical image (namely symmetry, quadrant areas, and HOG-based angle).
    """
    # Initialize empty dictionary to store shape properties
    shape_properties = {key: np.full(nz, np.nan, dtype=np.double) for key in property_list}

    # Initialize lists to store slice indices and angles
    z_indices = []
    angle_hog_values = []
    centermass_values = []
    for iz in sct_progress_bar(range(min_z_index, max_z_index + 1), unit='iter', unit_scale=False, desc="Compute shape analysis",
                               ncols=80):
        # Skip any slices with empty segmentations
        current_patch_scaled = current_patches[iz]['patch']
        if np.count_nonzero(current_patch_scaled) * pr <= 1:
            logging.warning(f'Skipping slice {iz} as the segmentation contains only a single pixel.')
            continue

        # Apply angle correction transformation to the anatomical image slice
        current_patch_im = im.data[:, :, iz]
        if angle_correction:
            tform = current_tforms[iz]
            current_patch_im_scaled = transform.warp(current_patch_im.astype(np.float64),
                                                     tform.inverse,
                                                     output_shape=current_patch_im.shape,
                                                     order=1,
                                                     )
        else:
            current_patch_im_scaled = current_patch_im

        # compute PCA and get center or mass based on segmentation; centermass_src: [RL, AP] (assuming RPI orientation)
        coord_src, pca_src, centermass_src = compute_pca(current_patch_scaled)
        # Finds the angle of the image
        # TODO: explore different sigma values for the HOG method, i.e., the influence how far away pixels will vote for the orientation.
        # TODO: double-check if sigma is in voxel or mm units.
        # TODO: do we want to use the same sigma for all slices? As the spinal cord sizes vary across the z-axis.
        angle_hog = find_angle_hog(
            current_patch_im_scaled,
            centermass_src,
            px, py,
        )  # taken from registration.algorithms.register2d_centermassrot

        z_indices.append(iz)
        angle_hog_values.append(angle_hog)
        centermass_values.append(centermass_src)

    # Apply regularization to HOG angles along the z-axis if filter_size > 0
    if filter_size > 0 and len(z_indices) > 0:
        angle_hog_regularized = gaussian_filter1d(np.array(angle_hog_values), filter_size)
        if verbose == 2:
            create_regularized_hog_angle_plot(
                np.array(z_indices), filter_size,
                np.array(angle_hog_values), angle_hog_regularized
            )
    # If filter_size <= 0, then just use the original angles
    else:
        angle_hog_regularized = angle_hog_values

    # Now compute shape properties using the new angles
    for iz, angle_hog, centermass_src in zip(z_indices, angle_hog_regularized, centermass_values):
        current_patch_scaled = current_patches[iz]['patch']

        # Compute shape properties with regularized angle_hog
        shape_property = _properties2d(current_patch_scaled, [px, py], pr, iz, angle_hog=angle_hog, verbose=verbose)

        if shape_property is not None:
            # Add custom fields
            shape_property['centermass_x'] = centermass_src[0]
            shape_property['centermass_y'] = centermass_src[1]
            shape_property['angle_hog'] = -angle_hog * 180.0 / math.pi     # degrees, and change sign to match negative if left rotation
            # Loop across properties and assign values for function output
            for property_name in property_list:
                shape_properties[property_name][iz] = shape_property[property_name]
        else:
            logging.warning(f'\nNo properties for slice: {iz}')

    return shape_properties


def _properties2d(seg, dim, pr, iz, angle_hog=None, verbose=1):
    """
    Compute shape property of the input 2D segmentation. Accounts for partial volume information.
    :param seg: 2D input segmentation in uint8 or float (weighted for partial volume) that has a single object. seg.shape[0] --> RL; seg.shape[1] --> PA
    :param dim: [px, py]: Physical dimension of the segmentation (in mm). X,Y respectively correspond to RL,PA.
    :param iz: Integer slice number (z index) of the segmentation. Used for plotting purposes.
    :param angle_hog: Optional angle in radians to rotate the segmentation to align with AP/RL axes.
    :return:
    """
    pad = 3  # padding used for cropping
    # Check if slice is empty
    if np.all(seg < NEAR_ZERO_THRESHOLD):
        logging.debug('The slice is empty.')
        return None
    # Normalize between 0 and 1 (also check if slice is empty)
    seg_norm = (seg - seg.min()) / (seg.max() - seg.min())
    # Convert to float64
    seg_norm = seg_norm.astype(np.float64)
    # Binarize segmentation using threshold at NEAR_ZERO_THRESHOLD Necessary input for measure.regionprops
    # Note: even when the input segmentation is binary, it might be soft now due to the angle correction
    seg_bin = np.array(seg_norm > NEAR_ZERO_THRESHOLD, dtype='uint8')
    # Get all closed binary regions from the segmentation (normally there is only one)
    regions = measure.regionprops(seg_bin, intensity_image=seg_norm)
    # Check number of regions
    if len(regions) > 1:
        logging.debug('There is more than one object on this slice.')
        return None
    region = regions[0]
    # Get bounding box of the object
    minx, miny, maxx, maxy = region.bbox
    # Use those bounding box coordinates to crop the segmentation (for faster processing)
    seg_crop = seg_norm[np.clip(minx-pad, 0, seg_bin.shape[0]): np.clip(maxx+pad, 0, seg_bin.shape[0]),
                        np.clip(miny-pad, 0, seg_bin.shape[1]): np.clip(maxy+pad, 0, seg_bin.shape[1])]
    # Apply resampling to the cropped segmentation:
    zoom_factors = (dim[0]/pr, dim[1]/pr)
    seg_crop_r = zoom(seg_crop, zoom=zoom_factors, order=1, mode='grid-constant', grid_mode=True)  # make pixel size isotropic
    # Update dim to isotropic pixel size
    dim = [pr, pr]
    # Binarize segmentation using threshold at 0.5 Necessary input for measure.regionprops
    seg_crop_r_bin = np.array(seg_crop_r > 0.5, dtype='uint8')
    # Get all closed binary regions from the segmentation (normally there is only one)
    regions = measure.regionprops(seg_crop_r_bin, intensity_image=seg_crop_r)
    region = regions[0]
    # Compute area with weighted segmentation and adjust area with physical pixel size
    area = np.sum(seg_crop_r) * dim[0] * dim[1]
    # Compute ellipse orientation, modulo pi, in deg, and between [0, 90]
    orientation = fix_orientation(region.orientation)
    # Find RL and AP diameter based on major/minor axes and cord orientation # TODO: remove
    [diameter_AP, diameter_RL] = \
        _find_AP_and_RL_diameter(region.major_axis_length, region.minor_axis_length, orientation,
                                 dim)
    # Deal with https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2307
    if any(x in platform.platform() for x in ['Darwin-15', 'Darwin-16']):
        solidity = np.nan
    else:
        solidity = region.solidity

    # Fill up dictionary
    properties = {
        'area': area,
        'diameter_AP_ellipse': diameter_AP,
        'diameter_RL': diameter_RL,
        'centroid': region.centroid,        # Why do we store this? It is not used in the code.
        'eccentricity': region.eccentricity,
        'orientation': -region.orientation,  # in radians
        'solidity': solidity,  # convexity measure
    }
    # Rotate the segmentation by the orientation to align with AP/RL axes
    seg_crop_r_rotated = _rotate_segmentation_by_angle(seg_crop_r, -region.orientation)

    # Measure diameters along AP in the rotated segmentation
    rotated_properties = _measure_ap_diameter(seg_crop_r, seg_crop_r_rotated, dim, region.orientation,
                                              iz, properties, verbose)
    # Update the properties dictionary with the rotated properties
    properties.update(rotated_properties)
    properties['orientation'] = -properties['orientation'] * 180.0 / math.pi  # convert to degrees

    if angle_hog is not None:
        # If angle_hog is provided, rotate the segmentation by this angle to align with AP/RL axes, and use for symmetry assessment
        seg_crop_r_rotated_hog = _rotate_segmentation_by_angle(seg_crop_r, angle_hog)
        # Compute quadrant areas and RL and AP symmetry
        quadrant_areas = compute_quadrant_areas(seg_crop_r_rotated_hog, region.centroid, orientation_deg=0,
                                                area=area, diameter_AP=diameter_AP, diameter_RL=diameter_RL,
                                                dim=dim, iz=iz, verbose=verbose)
        # Update the properties dictionary with the rotated properties
        properties.update(quadrant_areas)
        symmetry_metrics = _calculate_symmetry(seg_crop_r_rotated_hog, region.centroid, iz=iz, dim=dim, verbose=verbose)
        properties.update(symmetry_metrics)
    return properties


def fix_orientation(orientation):
    """Re-map orientation from skimage.regionprops from [-pi/2,pi/2] to [0,90] and rotate by 90deg because image axis
    are inverted"""
    orientation_new = orientation * 180.0 / math.pi
    if 360 <= abs(orientation_new) <= 540:
        orientation_new = 540 - abs(orientation_new)
    if 180 <= abs(orientation_new) <= 360:
        orientation_new = 360 - abs(orientation_new)
    if 90 <= abs(orientation_new) <= 180:
        orientation_new = 180 - abs(orientation_new)
    return abs(orientation_new)


def _find_AP_and_RL_diameter(major_axis, minor_axis, orientation, dim):
    """
    This script checks the orientation of the and assigns the major/minor axis to the appropriate dimension, right-
    left (RL) or antero-posterior (AP). It also multiplies by the pixel size in mm.
    :param major_axis: major ellipse axis length calculated by regionprops
    :param minor_axis: minor ellipse axis length calculated by regionprops
    :param orientation: orientation in degree. Ranges between [0, 90]
    :param dim: pixel size in mm.
    :return: diameter_AP, diameter_RL
    """
    if 0 <= orientation < 45.0:
        diameter_AP = minor_axis
        diameter_RL = major_axis
    else:
        diameter_AP = major_axis
        diameter_RL = minor_axis
    # Adjust with pixel size
    diameter_AP *= dim[0]
    diameter_RL *= dim[1]
    return diameter_AP, diameter_RL


def _rotate_segmentation_by_angle(seg_crop_r, angle):
    """
    Rotate the segmentation by the angle (HOG angle found from the image) to align with AP/RL axes.

    :param seg_crop_r: 2D input segmentation
    :param angle: Rotation angle in radians (positive values correspond to counter-clockwise rotation)
    :return seg_crop_r_rotated: Rotated segmentation
    """
    # get center of mass (which is computed in the PCA function)
    _, _, [y0, x0] = compute_pca(seg_crop_r)        # same as `y0, x0 = region.centroid`
    # Create coordinate grid
    rows, cols = seg_crop_r.shape
    Y, X = np.mgrid[0:rows, 0:cols]
    # Center the coordinates
    Xc = X - x0
    Yc = Y - y0
    # Rotate coordinates
    # If the angle is negative, the rotation is clockwise (from Left to Right).
    # If it is positive, the rotation is counter-clockwise (from Right to Left).
    Xr = Xc * np.cos(angle) - Yc * np.sin(angle)
    Yr = Xc * np.sin(angle) + Yc * np.cos(angle)
    # Shift the rotated coordinates back to their original position in the image. This ensures that the rotated
    # segmentation is positioned correctly in the output image, with the rotation happening around the center of the
    # object rather than around the origin of the coordinate system.
    Xr = Xr + x0
    Yr = Yr + y0

    # Create coordinate mapping for interpolation
    coords = np.column_stack([np.ravel(Yr), np.ravel(Xr)])

    # Apply transformation
    seg_crop_r_rotated = map_coordinates(seg_crop_r,
                                         [coords[:, 0], coords[:, 1]],
                                         order=1).reshape(seg_crop_r.shape)      # order of the spline interpolation --> order=1: linear interpolation

    return seg_crop_r_rotated


def _measure_ap_diameter(seg_crop_r, seg_crop_r_rotated, dim, angle, iz, properties, verbose):
    """
    Measure the AP diameter in the rotated segmentation.
    This function counts the number of pixels along the AP axis in the rotated segmentation and converts them
    to physical dimensions using the provided pixel size.

    :param seg_crop_r: Original cropped segmentation (used only for plotting).
    :param seg_crop_r_rotated: Rotated segmentation (after applying angle) used to measure diameters. seg.shape[0] --> RL; seg.shape[1] --> PA
    :param dim: Physical dimensions of the segmentation (in mm). X,Y respectively correspond to RL,PA.
    :param angle: Rotation angle in radians (HOG angle found from the image)
    :param iz: Integer slice number (z index) of the segmentation. Used for plotting purposes.
    :param properties: Dictionary containing the properties of the original (not-rotated) segmentation. Used for plotting purposes.
    :param verbose: Verbosity level. If 2, debug figures are created.
    :return result: Dictionary containing the measured diameters and pixel counts along AP and RL axes.
    """
    # Get center of mass (which is computed in the PCA function); centermass_src: [RL, AP] (assuming RPI orientation)
    # Note: I'm using [rl0, ap0] instead of [y0, x0] to make it easier to track the axes as numpy handle them in a bit unintuitive way :-D
    rotated_bin = np.array(seg_crop_r_rotated > 0.5, dtype='uint8')  # binarize the rotated segmentation for PCA
    # Check if segmentation is empty after rotation
    if np.all(rotated_bin < NEAR_ZERO_THRESHOLD):
        logging.debug('The rotated segmentation is empty.')
        return {
            'ap_pixel_count': np.nan,
            'diameter_AP': np.nan,
            'length_anterior': np.nan,
            'length_posterior': np.nan,
        }
    else:
        _, _, [rl0, ap0] = compute_pca(rotated_bin)    # same as `y0, x0 = region.centroid`
        rl0_r, ap0_r = round(rl0), round(ap0)

        # Note: seg_crop_r_rotated is soft (due to the rotation) so we sum its values to account for softness
        # Sum non-zero pixels along AP axis, i.e., the number of pixels in the row corresponding to the center of mass along the RL axis
        # Compute AP diameter average across 3 mm extent centered at rl0_r

        # Use centroid for AP diameter
        extent_avg = 30  # extent of 3 mm (because seg was resampled to 0.1mm) used for averaging the minimum (to account for noise)
        indices = np.array([i for i in range(rl0_r - extent_avg//2, rl0_r + extent_avg//2)])
        # Ensure indices are within the segmentation bounds
        indices = indices[(indices >= 0) & (indices < seg_crop_r_rotated.shape[0])]
        # Average the voxels across the 3 mm RL extent to get the mean AP array
        ap_arr_mean = seg_crop_r_rotated[indices, :].mean(axis=0)

        # Split the AP diameter at the cord CoM and sum to get anterior/posterior lengths (vox -> mm).
        # The split column (ap0_r) is the CoM of the actual segmentation, computed by compute_pca() on the binary
        # rotated segmentation (equivalent to region.centroid from measure.regionprops; see above).
        # measure.regionprops fits the equivalent ellipse centered at region.centroid (the segmentation CoM),
        # so the RL major axis (diameter_RL) also passes through this column.
        # Inspiration: Kang et al. J Clin Med 2023, https://doi.org/10.3390/jcm12124111 (Fig. 1).
        # In RPI orientation, axis 1 (columns) is the PA direction: low index = posterior, high index = anterior.
        length_posterior = ap_arr_mean[:ap0_r].sum() * dim[1]
        length_anterior = ap_arr_mean[ap0_r:].sum() * dim[1]
        ap_diameter = length_posterior + length_anterior

        # Store all the rotated properties
        result = {
            'ap_pixel_count': ap_arr_mean.sum(),
            'diameter_AP': ap_diameter,
            'length_anterior': length_anterior,
            'length_posterior': length_posterior,
        }

        # Debug plotting
        if verbose == 2:
            create_ap_diameter_plots(angle, ap0_r, ap_diameter, dim, iz, properties, rl0_r, properties["diameter_RL"],
                                     seg_crop_r_rotated, seg_crop_r)

    return result


def _calculate_symmetry(seg_crop_r_rotated, centroid, dim, iz=None, verbose=1):
    """
    Compute symmetry metrics by flipping the segmentation along the RL and AP axes and comparing with the original.
    Calculates Dice coefficient, symmetric difference, and Hausdorff distance for both axes.
    See figure in https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4958#issue-3212203682 for details.
    :param seg_crop_r_rotated: Rotated segmentation (after applying angle) used to measure diameters. seg.shape[0] --> RL; seg.shape[1] --> PA
    :param centroid: (y, x) coordinates of the centroid in the upsampled image space.
    :param dim: [px, py] pixel dimensions in mm.
    :param iz: Optional slice index for debug plotting.
    :param verbose: Verbosity level for debug plotting.
    :return: symmetry_metrics: Dictionary with symmetry metrics for RL and AP axes.
    """
    y0, x0 = centroid
    y0 = int(round(y0))
    x0 = int(round(x0))
    # Create an empty array for flipped version of AP axis
    seg_crop_r_flipped_AP = np.zeros_like(seg_crop_r_rotated)

    # Get bounding box of the segmentation (on the cropped segmentation, not the full one)
    coords = np.argwhere(seg_crop_r_rotated > 0)
    # Center of segmentation along RL axis (x)

    # Flip around segmentation center
    for y, x in coords[:, :2]:
        x_mirror = 2 * x0 - x
        if 0 <= x_mirror < seg_crop_r_rotated.shape[1]:
            seg_crop_r_flipped_AP[y, x_mirror] = seg_crop_r_rotated[y, x]
    # Erase half of the segmentation to avoid bias
    seg_crop_r_flipped_AP[:, :x0] = 0
    seg_crop_r_rotated_cut = np.copy(seg_crop_r_rotated)
    seg_crop_r_rotated_cut[:, :x0] = 0
    # Compute the intersection and union for the Dice coefficient
    intersection_AP = np.sum(seg_crop_r_rotated_cut * seg_crop_r_flipped_AP)
    union_AP = np.sum(seg_crop_r_rotated_cut) + np.sum(seg_crop_r_flipped_AP)

    # Compute the Dice coefficient
    symmetry_dice_AP = 2 * intersection_AP / union_AP if union_AP > 0 else 0
    symmetric_difference_AP = (union_AP - (2*intersection_AP)) * dim[0] * dim[1]

    # Compute Hausdorff distance as additional metric
    hausdorff_distance_AP = skimage.metrics.hausdorff_distance(seg_crop_r_rotated_cut > 0.5, seg_crop_r_flipped_AP > 0.5) * dim[0]

    # Create an empty array for flipped version of RL axis
    seg_crop_r_flipped_RL = np.zeros_like(seg_crop_r_rotated)

    # Flip around segmentation center (RL axis)
    for y, x in coords[:, :2]:
        y_mirror = 2 * y0 - y
        if 0 <= y_mirror < seg_crop_r_rotated.shape[0]:
            seg_crop_r_flipped_RL[y_mirror, x] = seg_crop_r_rotated[y, x]
    seg_crop_r_flipped_RL[:y0, :] = 0
    seg_crop_r_rotated_cut_RL = np.copy(seg_crop_r_rotated)
    seg_crop_r_rotated_cut_RL[:y0, :] = 0
    # Compute the intersection and union for the Dice coefficient
    intersection_RL = np.sum(seg_crop_r_rotated_cut_RL * seg_crop_r_flipped_RL)
    union_RL = np.sum(seg_crop_r_rotated_cut_RL) + np.sum(seg_crop_r_flipped_RL)

    # Compute the Dice coefficient
    symmetry_dice_RL = 2 * intersection_RL / union_RL if union_RL > 0 else 0
    symmetric_difference_RL = (union_RL - (2*intersection_RL)) * dim[0] * dim[1]

    # Compute Hausdorff distance as additional metric
    hausdorff_distance_RL = skimage.metrics.hausdorff_distance(seg_crop_r_rotated_cut_RL > 0.5, seg_crop_r_flipped_RL > 0.5) * dim[0]

    symmetry_metrics = {
        'symmetry_dice_RL': symmetry_dice_RL,
        'symmetry_dice_AP': symmetry_dice_AP,
        'symmetry_hausdorff_RL': hausdorff_distance_RL,
        'symmetry_hausdorff_AP': hausdorff_distance_AP,
        'symmetry_difference_RL': symmetric_difference_RL,
        'symmetry_difference_AP': symmetric_difference_AP,
    }

    # Create a debug plot
    if verbose == 2:
        create_symmetry_plots(
            seg_crop_r_rotated, centroid,
            seg_crop_r_rotated_cut, seg_crop_r_rotated_cut_RL,
            seg_crop_r_flipped_RL, seg_crop_r_flipped_AP,
            symmetry_metrics, iz
        )

    return symmetry_metrics


def compute_quadrant_areas(image_crop_r: np.ndarray, centroid: tuple[float, float], orientation_deg: float,
                           area: float, diameter_AP: float, diameter_RL: float,
                           dim: list[float], iz: int, verbose=1) -> tuple[dict, dict]:
    """
    Compute the cross-sectional area of the four spinal cord quadrants in the axial plane.
    Also calculates the symmetry of the spinal cord in the right-left (RL) and anterior-posterior (AP) directions.
    The function rotates the coordinate system based on the spinal cord orientation and
    partitions the segmentation into four quadrants: posterior right, anterior right,
    posterior left, and anterior left. It then calculates the area of each quadrant in mm².
    :param image_crop_r: 2D upsampled non-binary (due to the angle correction) segmentation mask of the spinal cord.
    :param centroid: (y, x) coordinates of the centroid in the upsampled image space.
    :param orientation_deg: Orientation angle of the spinal cord in degrees (from regionprops).
    :param area: Total area of the spinal cord in mm² (used for symmetry calculations).
    :param diameter_AP: AP diameter of the spinal cord in mm (used for debug plots).
    :param diameter_RL: RL diameter of the spinal cord in mm (used for debug plots).
    :param dim: [px, py] pixel dimensions in mm. X,Y respectively correspond to AP, RL.
    :param iz: Slice index used for filename in debug plot.
    :return: quadrant_areas (dict)
        quadrant_areas is a dictionary with the area in mm² for each quadrant:
                 {
                    'area_quadrant_posterior_right': float,
                    'area_quadrant_anterior_right': float,
                    'area_quadrant_posterior_left': float,
                    'area_quadrant_anterior_left': float
                 }
    """
    y0, x0 = centroid
    orientation_rad = np.radians(orientation_deg)
    rows, cols = image_crop_r.shape
    Y, X = np.mgrid[0:rows, 0:cols]

    # Translate coordinate grid to centroid
    Xc = X - x0
    Yc = Y - y0

    # Rotate coordinates to align with AP/RL axes
    # This rotation is needed to accurately define anatomical quadrants.
    # Without the rotation, quadrants would be based on image axes (top/bottom/left/right) rather than true anatomical
    # orientation (posterior/anterior/left/right) resulting in unprecise area calculations whenever the spinal cord is
    # not perfectly aligned with the image axes.
    # TODO remove this rotation as the segmentation is already rotated
    Xr = Xc * np.cos(-orientation_rad) - Yc * np.sin(-orientation_rad)
    Yr = Xc * np.sin(-orientation_rad) + Yc * np.cos(-orientation_rad)

    # Apply quadrant masks - use the intensity values directly to account for the mask softness
    post_r_mask = (Yr < 0) & (Xr < 0)    # Posterior Right
    ant_r_mask = (Yr < 0) & (Xr >= 0)    # Anterior Right
    post_l_mask = (Yr >= 0) & (Xr < 0)   # Posterior Left
    ant_l_mask = (Yr >= 0) & (Xr >= 0)   # Anterior Left

    # Calculate physical area in mm²
    pixel_area_mm2 = (dim[0] * dim[1])

    # Sum areas for each quadrant - multiply by intensity values to account for the mask softness
    quadrant_areas = {
        'area_quadrant_posterior_right': np.sum(image_crop_r[post_r_mask]) * pixel_area_mm2,
        'area_quadrant_anterior_right': np.sum(image_crop_r[ant_r_mask]) * pixel_area_mm2,
        'area_quadrant_posterior_left': np.sum(image_crop_r[post_l_mask]) * pixel_area_mm2,
        'area_quadrant_anterior_left': np.sum(image_crop_r[ant_l_mask]) * pixel_area_mm2
    }

    if verbose == 2:
        create_quadrant_area_plots(
            image_crop_r, centroid, orientation_deg, dim,
            ant_r_mask, ant_l_mask, post_r_mask, post_l_mask, quadrant_areas,
            diameter_AP, diameter_RL, iz,
        )

    return quadrant_areas
