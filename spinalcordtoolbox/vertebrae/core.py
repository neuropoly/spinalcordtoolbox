"""
Core functions dealing with vertebral labeling

Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

# TODO: remove i/o as much as possible

import os
import logging

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, center_of_mass as ndimage_center_of_mass

from spinalcordtoolbox.image import Image, add_suffix
from spinalcordtoolbox.metadata import get_file_label
from spinalcordtoolbox.math import dilate, mutual_information
from spinalcordtoolbox.centerline.core import get_centerline
from spinalcordtoolbox.types import MissingDiscsError, EmptyArrayError

logger = logging.getLogger(__name__)


def label_vert(fname_seg, fname_label):
    """
    Label segmentation using vertebral labeling information. No orientation expected.

    :param fname_seg: file name of segmentation.
    :param fname_label: file name for a labelled segmentation that will be used to label the input segmentation
    :param fname_out: file name of the output labeled segmentation. If empty, will add suffix "_labeled" to fname_seg
    """
    # retrieve all labels
    coord_labels = Image(fname_label).change_orientation("RPI").getNonZeroCoordinates()
    if not coord_labels:
        raise MissingDiscsError(f'No disc labels present in file {fname_label}')
    # '-1' to use the convention "disc labelvalue=3 ==> disc C2/C3"
    discs = [(cl.z, cl.value - 1) for cl in reversed(coord_labels)]
    discs.sort(reverse=True)
    # label segmentation
    label_segmentation(fname_seg, discs)
    label_discs(fname_seg, discs)


def vertebral_detection(fname, fname_seg, contrast, param, init_disc, verbose=1, path_template='', path_output='..',
                        scale_dist=1.):
    """
    Find intervertebral discs in straightened image using template matching

    :param fname: file name of straigthened spinal cord
    :param fname_seg: file name of straigthened spinal cord segmentation
    :param contrast: t1 or t2
    :param param:  advanced parameters
    :param init_disc:
    :param verbose:
    :param path_template:
    :param path_output: output path for verbose=2 pictures
    :param scale_dist: float: Scaling factor to adjust average distance between two adjacent intervertebral discs
    :return:
    """
    logger.info('Look for template...')
    logger.info('Path template: %s', path_template)

    # adjust file names if MNI-Poly-AMU template is used (by default: PAM50)
    fname_level = get_file_label(os.path.join(path_template, 'template'), id_label=7, output='filewithpath')  # label = spinal cord mask with discrete vertebral levels
    id_label_dct = {'T1': 0, 'T2': 1, 'T2S': 2}
    fname_template = get_file_label(os.path.join(path_template, 'template'), id_label=id_label_dct[contrast.upper()], output='filewithpath')  # label = *-weighted template

    # Open template and vertebral levels
    logger.info('Open template and vertebral levels...')
    data_template = Image(fname_template).data
    data_disc_template = Image(fname_level).data

    # open anatomical volume
    im_input = Image(fname)
    data = im_input.data

    # smooth data
    smooth_factor = [3, 1, 1]
    data = gaussian_filter(data, smooth_factor, output=None, mode="reflect")

    # get dimension of src
    nx, ny, nz = data.shape
    # define xc and yc (centered in the field of view)
    xc = int(np.round(nx / 2))  # direction RL
    yc = int(np.round(ny / 2))  # direction AP
    # get dimension of template
    nxt, nyt, nzt = data_template.shape
    # define xc and yc (centered in the field of view)
    xct = int(np.round(nxt / 2))  # direction RL
    yct = int(np.round(nyt / 2))  # direction AP

    # define mean distance (in voxel) between adjacent discs: [C1/C2 -> C2/C3], [C2/C3 -> C3/C4], ..., [L1/L2 -> L2/L3]
    centerline_level = data_disc_template[xct, yct, :]
    # attribute value to each disc. Starts from max level, then decrease.
    min_level = centerline_level[centerline_level.nonzero()].min()
    max_level = centerline_level[centerline_level.nonzero()].max()
    list_disc_value_template = list(range(min_level, max_level))
    # add disc above top one
    list_disc_value_template.insert(int(0), min_level - 1)
    logger.info('Disc values from template: %s', list_disc_value_template)
    # get diff to find transitions (i.e., discs)
    diff_centerline_level = np.diff(centerline_level)
    # get disc z-values
    list_disc_z_template = diff_centerline_level.nonzero()[0].tolist()
    list_disc_z_template.reverse()
    logger.info('Z-values for each disc: %s', list_disc_z_template)
    list_distance_template = (
        np.diff(list_disc_z_template) * (-1)).tolist()  # multiplies by -1 to get positive distances
    # Update distance with scaling factor
    list_distance_template = [i * scale_dist for i in list_distance_template]
    logger.info('Distances between discs (in voxel): %s', list_distance_template)

    # display init disc
    if verbose == 2:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        fig_disc = Figure()
        FigureCanvas(fig_disc)
        ax_disc = fig_disc.add_subplot(111)
        # ax_disc = fig_disc.add_axes((0, 0, 1, 1))
        # get percentile for automatic contrast adjustment
        data_display = np.mean(data[xc - param['size_RL']:xc + param['size_RL'], :, :], axis=0).transpose()
        percmin = np.percentile(data_display, 10)
        percmax = np.percentile(data_display, 90)
        # display image
        ax_disc.matshow(data_display, cmap='gray', clim=[percmin, percmax], origin='lower')
        ax_disc.set_title('Anatomical image')
        # ax.autoscale(enable=False)  # to prevent autoscale of axis when displaying plot
        ax_disc.scatter(yc + param['shift_AP_visu'], init_disc[0], c='yellow', s=10)
        ax_disc.text(yc + param['shift_AP_visu'] + 4, init_disc[0], f'{init_disc[1]}/{init_disc[1] + 1}',
                     verticalalignment='center', horizontalalignment='left', color='pink', fontsize=7)

    # FIND DISCS
    # ===========================================================================
    logger.info('Detect intervertebral discs...')
    current_z, current_value = init_disc
    if (current_value not in list_disc_value_template) or (current_value-1 not in list_disc_value_template):
        logger.error('Initial disc (%s) is not in template. Cannot detect intervertebral discs.', current_value)
        raise ValueError('Initial disc is not in template.')
    # create list for z and disc
    list_disc_z = []
    list_disc_value = []
    zrange = list(range(-10, 10))
    direction = 'superior'
    search_next_disc = True
    while search_next_disc:
        logger.info('Current disc: %s (z=%s). Direction: %s', current_value, current_z, direction)
        try:
            # get z corresponding to current disc on template
            current_z_template = list_disc_z_template[current_value]
        except IndexError:
            # in case reached the bottom (see issue #849)
            logger.warning('Reached the bottom of the template. Stop searching.')
            break
        # find next disc
        # N.B. Do not search for C1/C2 disc (because poorly visible), use template distance instead
        if current_value != 1:
            current_z = compute_corr_3d(data, data_template, x=xc, xshift=0, xsize=param['size_RL'],
                                        y=yc, yshift=param['shift_AP'], ysize=param['size_AP'],
                                        z=current_z, zshift=0, zsize=param['size_IS'],
                                        xtarget=xct, ytarget=yct, ztarget=current_z_template,
                                        zrange=zrange, verbose=verbose, save_suffix=f'_disc{current_value}',
                                        path_output=path_output)

        # display new disc
        if verbose == 2:
            ax_disc.scatter(yc + param['shift_AP_visu'], current_z, c='yellow', s=10)
            ax_disc.text(yc + param['shift_AP_visu'] + 4, current_z, f'{current_value}/{current_value + 1}',
                         verticalalignment='center', horizontalalignment='left', color='yellow', fontsize=7)

        # append to main list
        if direction == 'superior':
            # append at the beginning
            list_disc_z.insert(0, current_z)
            list_disc_value.insert(0, current_value)
        elif direction == 'inferior':
            # append at the end
            list_disc_z.append(current_z)
            list_disc_value.append(current_value)

        # adjust correcting factor based on already-identified discs
        if len(list_disc_z) > 1:
            # compute distance between already-identified discs
            list_distance_current = (np.diff(list_disc_z) * (-1)).tolist()
            # retrieve the template distance corresponding to the already-identified discs
            index_disc_identified = [i for i, j in enumerate(list_disc_value_template) if j in list_disc_value[:-1]]
            list_distance_template_identified = [list_distance_template[i] for i in index_disc_identified]
            # divide subject and template distances for the identified discs
            list_subject_to_template_distance = [float(list_distance_current[i]) / list_distance_template_identified[i] for i in range(len(list_distance_current))]
            # average across identified discs to obtain an average correcting factor
            correcting_factor = np.mean(list_subject_to_template_distance)
            logger.info('.. correcting factor: %s', correcting_factor)
        else:
            correcting_factor = 1
        # update list_distance specific for the subject
        list_distance = [int(np.round(list_distance_template[i] * correcting_factor)) for i in range(len(list_distance_template))]

        # assign new current_z and disc value
        if direction == 'superior':
            try:
                approx_distance_to_next_disc = list_distance[list_disc_value_template.index(current_value - 1)]
            except ValueError:
                logger.warning('Disc value not included in template. Using previously-calculated distance: %s', approx_distance_to_next_disc)
            # assign new current_z and disc value
            current_z = current_z + approx_distance_to_next_disc
            current_value = current_value - 1
        elif direction == 'inferior':
            try:
                approx_distance_to_next_disc = list_distance[list_disc_value_template.index(current_value)]
            except ValueError:
                logger.warning('Disc value not included in template. Using previously-calculated distance: %s', approx_distance_to_next_disc)
            # assign new current_z and disc value
            current_z = current_z - approx_distance_to_next_disc
            current_value = current_value + 1

        # if current_z is larger than searching zone, switch direction (and start from initial z minus approximate
        # distance from updated template distance)
        if current_z >= nz or current_value == 0:
            logger.info('.. Switching to inferior direction.')
            direction = 'inferior'
            current_value = init_disc[1] + 1
            try:
                current_z = init_disc[0] - list_distance[list_disc_value_template.index(current_value)]
            except ValueError:
                logger.info('No disc is inferior to the initial disc.')
                search_next_disc = False
        # if current_z is lower than searching zone, stop searching
        if current_z <= 0:
            search_next_disc = False

    if verbose == 2:
        fig_disc.savefig('fig_label_discs.png')

    # if upper disc is not 1, add disc above top disc based on mean_distance_adjusted
    upper_disc = min(list_disc_value)
    # if not upper_disc == 1:
    logger.info('Adding top disc based on adjusted template distance: #%s', upper_disc - 1)
    approx_distance_to_next_disc = list_distance[list_disc_value_template.index(upper_disc - 1)]
    next_z = max(list_disc_z) + approx_distance_to_next_disc
    logger.info('.. approximate distance: %s', approx_distance_to_next_disc)
    # make sure next disc does not go beyond FOV in superior direction
    if next_z > nz:
        list_disc_z.insert(0, nz)
    else:
        list_disc_z.insert(0, next_z)
    # assign disc value
    list_disc_value.insert(0, upper_disc - 1)

    # Label segmentation
    discs = list(zip(list_disc_z, list_disc_value))
    label_segmentation(fname_seg, discs)
    label_discs(fname_seg, discs)


def center_of_mass(x):
    """
    :return: array center of mass
    """
    if (x == 0).all():
        raise EmptyArrayError("Center of mass can't be calculated on empty arrays.")
    return ndimage_center_of_mass(x)


def create_label_z(fname_seg, z, value, fname_labelz='labelz.nii.gz'):
    """
    Create a label at coordinates x_center, y_center, z

    :param fname_seg: segmentation
    :param z: int
    :param fname_labelz: string file name of output label
    :return: fname_labelz
    """
    nii = Image(fname_seg)
    orientation_origin = nii.orientation
    nii = nii.change_orientation("RPI")
    nx, ny, nz, nt, px, py, pz, pt = nii.dim  # Get dimensions
    # find x and y coordinates of the centerline at z using center of mass
    x, y = center_of_mass(np.array(nii.data[:, :, z]))
    x, y = int(np.round(x)), int(np.round(y))
    nii.data[:, :, :] = 0
    nii.data[x, y, z] = value
    # dilate label to prevent it from disappearing due to nearestneighbor interpolation
    nii.data = dilate(nii.data, 3, 'ball', islabel=True)
    nii.change_orientation(orientation_origin)  # put back in original orientation
    nii.save(fname_labelz)
    return fname_labelz


def get_z_and_disc_values_from_label(fname_label):
    """
    Find z-value and label-value based on labeled image in RPI orientation

    :param fname_label: image in RPI orientation that contains label
    :return: [z_label, value_label] int list
    """
    nii = Image(fname_label)
    # get center of mass of label
    x_label, y_label, z_label = center_of_mass(nii.data)
    x_label, y_label, z_label = int(np.round(x_label)), int(np.round(y_label)), int(np.round(z_label))
    # get label value
    value_label = int(nii.data[x_label, y_label, z_label])
    return [z_label, value_label]


def crop_labels(im_labeled_seg, im_seg):
    """
    Remove voxels in labeled segmentation that are not in segmentation,
    modifying on the labeled segmentation in-place.

    :param Image im_labeled_seg: labeled segmentation
    :param Image im_seg: segmentation (can be softseg or binary)
    :return: None
    """
    boolean_mask = (im_seg.data != 0)    # Nonzero -> True | Zero -> False
    im_labeled_seg.data *= boolean_mask  # Keep only voxels that were nonzero in the segmentation image
    # Note: NumPy will cast the bool type to the integer type, with False and True converted to 0 and 1 respectively.


def expand_labels(im_labeled_seg):
    """
    Fill in labels for all the voxels in the labeled segmentation,
    using the nearest label.
    Modifies the labeled segmentation in-place.

    :param Image im_labeled_seg: labeled segmentation
    :return: None
    """
    # for each voxel, find the coordinates of the nearest nonzero label
    indices = distance_transform_edt(
        (im_labeled_seg.data == 0),
        return_distances=False,
        return_indices=True)
    # label all voxels
    im_labeled_seg.data = im_labeled_seg.data[tuple(indices)]


def compute_corr_3d(src, target, x, xshift, xsize, y, yshift, ysize, z, zshift, zsize, xtarget, ytarget, ztarget, zrange, verbose, save_suffix, path_output):
    """
    FIXME doc
    Find z that maximizes correlation between src and target 3d data.

    :param src: 3d source data
    :param target: 3d target data
    :param x:
    :param xshift:
    :param xsize:
    :param y:
    :param yshift:
    :param ysize:
    :param z:
    :param zshift:
    :param zsize:
    :param xtarget:
    :param ytarget:
    :param ztarget:
    :param zrange:
    :param verbose:
    :param save_suffix:
    :return:
    """
    # parameters
    thr_corr = 0.2  # disc correlation threshold. Below this value, use template distance.
    # get dimensions from src
    nx, ny, nz = src.shape
    # Get pattern from template
    pattern = target[xtarget - xsize: xtarget + xsize + 1,
                     ytarget + yshift - ysize: ytarget + yshift + ysize + 1,
                     ztarget + zshift - zsize: ztarget + zshift + zsize + 1]
    pattern1d = pattern.ravel()
    # initializations
    I_corr = np.zeros(len(zrange))
    allzeros = 0
    # current_z = 0
    ind_I = 0
    # loop across range of z defined by src
    for iz in zrange:
        # if pattern extends towards the top part of the image, then crop and pad with zeros
        if z + iz + zsize + 1 > nz:
            padding_size = z + iz + zsize + 1 - nz
            data_chunk3d = src[x - xsize: x + xsize + 1,
                               y + yshift - ysize: y + yshift + ysize + 1,
                               z + iz - zsize: z + iz + zsize + 1 - padding_size]
            data_chunk3d = np.pad(data_chunk3d, ((0, 0), (0, 0), (0, padding_size)), 'constant',
                                  constant_values=0)
        # if pattern extends towards bottom part of the image, then crop and pad with zeros
        elif z + iz - zsize < 0:
            padding_size = abs(iz - zsize)
            data_chunk3d = src[x - xsize: x + xsize + 1,
                               y + yshift - ysize: y + yshift + ysize + 1,
                               z + iz - zsize + padding_size: z + iz + zsize + 1]
            data_chunk3d = np.pad(data_chunk3d, ((0, 0), (0, 0), (padding_size, 0)), 'constant',
                                  constant_values=0)
        else:
            data_chunk3d = src[x - xsize: x + xsize + 1,
                               y + yshift - ysize: y + yshift + ysize + 1,
                               z + iz - zsize: z + iz + zsize + 1]

        # convert subject pattern to 1d
        data_chunk1d = data_chunk3d.ravel()
        # check if data_chunk1d contains at least one non-zero value
        if (data_chunk1d.size == pattern1d.size) and np.any(data_chunk1d):
            I_corr[ind_I] = mutual_information(data_chunk1d, pattern1d, nbins=16, normalized=False)
        else:
            allzeros = 1
        ind_I = ind_I + 1
    # ind_y = ind_y + 1
    if allzeros:
        logger.warning('Data contained zero. We probably hit the edge of the image.')

    I_corr = np.array(I_corr, dtype=float)

    # Find global maximum
    if np.any(I_corr):
        # if I_corr contains at least a non-zero value
        ind_peak = [i for i in range(len(I_corr)) if I_corr[i] == max(I_corr)][0]  # index of max along z
        logger.info('.. Peak found: z=%s (correlation = %s)', zrange[ind_peak], I_corr[ind_peak])
        # check if correlation is high enough
        if I_corr[ind_peak] < thr_corr:
            logger.warning('Correlation is too low. Using adjusted template distance.')
            ind_peak = zrange.index(0)  # approx_distance_to_next_disc
    else:
        # if I_corr contains only zeros
        logger.warning('Correlation vector only contains zeros. Using adjusted template distance.')
        ind_peak = zrange.index(0)  # approx_distance_to_next_disc

    # display patterns and correlation
    if verbose == 2:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        fig = Figure(figsize=(15, 7))
        FigureCanvas(fig)
        # display template pattern
        ax = fig.add_subplot(131)
        ax.imshow(np.flipud(np.mean(pattern[:, :, :], axis=0).transpose()), origin='upper', cmap='gray',
                  interpolation='none')
        ax.set_title('Template pattern')
        # display subject pattern at best z
        ax = fig.add_subplot(132)
        iz = zrange[ind_peak]
        data_chunk3d = src[x - xsize: x + xsize + 1,
                           y + yshift - ysize: y + yshift + ysize + 1,
                           z + iz - zsize: z + iz + zsize + 1]
        ax.imshow(np.flipud(np.mean(data_chunk3d[:, :, :], axis=0).transpose()), origin='upper', cmap='gray',
                  clim=[0, 800], interpolation='none')
        ax.set_title(f'Subject at iz={iz}')
        # display correlation curve
        ax = fig.add_subplot(133)
        ax.plot(zrange, I_corr)
        ax.set_title('Mutual Info')
        ax.plot(zrange[ind_peak], I_corr[ind_peak], 'ro')
        ax.axvline(x=zrange.index(0), linewidth=1, color='black', linestyle='dashed')
        ax.axhline(y=thr_corr, linewidth=1, color='r', linestyle='dashed')
        ax.grid()
        # save figure
        fig.savefig(f'fig_pattern{save_suffix}.png')

    # return z-origin (z) + z-displacement minus zshift (to account for non-centered disc)
    return z + zrange[ind_peak] - zshift


def label_segmentation(fname_seg, discs):
    """
    Label segmentation image

    :param fname_seg: fname of the segmentation, no orientation expected
    :param discs: list of (z, value) pairs, one for each disc
    """

    # open segmentation
    seg = Image(fname_seg)
    init_orientation = seg.orientation
    seg.change_orientation("RPI")

    nz = seg.dim[2]
    for iz in range(nz):
        # get value of the disc right above iz
        try:
            # +1 because iz is BELOW the disc
            _, vertebral_level = min((z, value+1) for (z, value) in discs if z > iz)
        except ValueError:
            # if no discs are above iz, default to 0
            vertebral_level = 0
        # get voxels in mask
        ind_nonzero = np.nonzero(seg.data[:, :, iz])
        seg.data[ind_nonzero[0], ind_nonzero[1], iz] = vertebral_level

    # write file
    seg.change_orientation(init_orientation).save(add_suffix(fname_seg, '_labeled'))


def label_discs(fname_seg, discs):
    """
    Create file with single voxel label in the middle of the spinal cord for each disc.

    :param fname_seg: fname of the segmentation, no orientation expected
    :param discs: list of (z, value) pairs, one for each disc
    """
    seg = Image(fname_seg)
    init_orientation = seg.orientation
    seg.change_orientation("RPI")
    disc_data = np.zeros_like(seg.data)
    nx, ny, nz = seg.data.shape

    for disc_z, disc_value in discs:
        if disc_z < nz:
            slices = seg.data[:, :, disc_z]
            if (slices == 0).all():
                logger.warning("During disc labeling, center of mass calculation failed due to discontinuities in "
                               "segmented spinal cord; please check the quality of your segmentation. Using "
                               "interpolated centerline as a fallback.")
                interpolated_centerline, _, _, _ = get_centerline(seg)
                slices = interpolated_centerline.data[:, :, disc_z]
            cx, cy = [int(x) for x in np.round(center_of_mass(slices)).tolist()]

            # Disc value are offset by one due to legacy code
            disc_data[cx, cy, disc_z] = disc_value + 1

    seg.data = disc_data
    seg.change_orientation(init_orientation).save(add_suffix(fname_seg, '_labeled_disc'))
