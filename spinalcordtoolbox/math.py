"""
Mathematical operations on an image

Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import logging

import numpy as np
from skimage.morphology import erosion, dilation
from skimage.filters import threshold_local, threshold_otsu, rank
from scipy.ndimage import gaussian_filter, gaussian_laplace, label, generate_binary_structure

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import LazyLoader

dipy_patch2self = LazyLoader("dipy_patch2self", globals(), "dipy.denoise.patch2self")
dipy_mask = LazyLoader("dipy_mask", globals(), "dipy.segment.mask")
dipy_nlmeans = LazyLoader("dipy_nlmeans", globals(), "dipy.denoise.nlmeans")
dipy_noise = LazyLoader("dipy_noise", globals(), "dipy.denoise.noise_estimate")
skl_metrics = LazyLoader("skl_metrics", globals(), "sklearn.metrics")
scipy_stats = LazyLoader("scipy_stats", globals(), "scipy.stats")

logger = logging.getLogger(__name__)

ALMOST_ZERO = 0.000000001


def _get_footprint(shape, size, dim):
    """
    Create footprint (prev. terminology: structuring element) of desired shape and radius

    :param shape: str: Shape of the footprint. See available options below in the code
    :param size: int or list[int]: size of the element.
            Can be specified as 2 or 3 values (depending on the 2D/3D shape) for anisotropic images.
    :param dim: {0, 1, 2}: Dimension of the array which 2D structural element will be orthogonal to. For example, if
    you wish to apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be: shape=disk, dim=2.
    :return: numpy array: footprint
    """
    dimensionality = {
        'square': 2,
        'disk': 2,
        'cube': 3,
        'ball': 3,
    }
    if shape not in dimensionality:
        raise ValueError(f"This shape is not a valid entry: {shape}")

    # normalize `size` to be a list or 2 or 3 ints, depending on the 2D/3D kernel
    if isinstance(size, int):
        size = [size]
    if len(size) == 1:
        size = list(size) * dimensionality[shape]
    if len(size) != dimensionality[shape]:
        raise ValueError(f"Invalid size for shape {shape}: {size}")

    if shape == 'disk':
        # ellipse equation is: ((x - center_x)/radius_x)**2 + ((y - center_y)/radius_y)**2 <= 1
        # clearing denominators: ((x - center_x)*radius_y)**2 + ((y - center_y)*radius_x)**2 <= (radius_x*radius_y)**2
        r_x, r_y = size
        # the thin arrays fp_x and fp_y will be efficiently broadcast to a full rectangle by numpy
        fp_x = ((r_y * np.arange(-r_x, r_x+1))**2).reshape((2*r_x+1, 1))
        fp_y = ((r_x * np.arange(-r_y, r_y+1))**2).reshape((1, 2*r_y+1))
        footprint = ((fp_x + fp_y) <= (r_x * r_y)**2).astype(np.uint8)
    elif shape == 'ball':
        # ellipsoid equation, after clearing denominators:
        # ((x - center_x)*radius_y*radius_z)**2
        # + ((y - center_y)*radius_x*radius_z)**2
        # + ((z - center_z)*radius_x*radius_y)**2
        # <= (radius_x*radius_y*radius_z)**2
        r_x, r_y, r_z = size
        fp_x = ((r_y * r_z * np.arange(-r_x, r_x+1))**2).reshape((2*r_x+1, 1, 1))
        fp_y = ((r_x * r_z * np.arange(-r_y, r_y+1))**2).reshape((1, 2*r_y+1, 1))
        fp_z = ((r_x * r_y * np.arange(-r_z, r_z+1))**2).reshape((1, 1, 2*r_z+1))
        footprint = ((fp_x + fp_y + fp_z) <= (r_x * r_y * r_z)**2).astype(np.uint8)
    else:
        assert shape in ['square', 'cube']
        footprint = np.ones(size, dtype=np.uint8)

    # If 2d kernel, replicate it along the specified dimension
    if len(footprint.shape) == 2:
        if dim not in [0, 1, 2]:
            raise ValueError("dim can only take values: {0, 1, 2}")
        footprint = np.expand_dims(footprint, dim)

    return footprint


def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.

    :param im1 : array-like, bool\
        Any array of arbitrary size. If not boolean, will be converted.
    :param im2 : array-like, bool\
        Any other array of identical size. If not boolean, will be converted.
    :return dice : float\
        Dice coefficient as a float on range [0,1].\
        Maximum similarity = 1\
        No similarity = 0

    .. note::
        The order of inputs for `dice` is irrelevant. The result will be
        identical if `im1` and `im2` are switched.

    Source: https://gist.github.com/JDWarner/6730747
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def dilate(data, size, shape, dim=None, islabel=False):
    """
    Dilate data using ball structuring element

    :param data: Image or numpy array: 2d or 3d array
    :param size: int or list[int]: If shape={'square', 'cube'}: Corresponds to the length of an edge (size=1 has no effect).\
        If shape={'disk', 'ball'}: Corresponds to the radius, not including the center element (size=0 has no effect).
        Can be specified as 2 or 3 values (depending on the 2D/3D shape) for anisotropic images.
    :param shape: {'square', 'cube', 'disk', 'ball'}
    :param dim: {0, 1, 2}: Dimension of the array which 2D structural element will be orthogonal to. For example, if\
    you wish to apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be: shape=disk, dim=2.
    :return: numpy array: data dilated
    """
    if isinstance(data, Image):
        im_out = data.copy()
        im_out.data = dilate(data.data, size, shape, dim, islabel)
        return im_out
    else:
        footprint = _get_footprint(shape, size, dim)
        if islabel:
            return _dilate_point_labels(data, footprint=footprint)
        else:
            if data.dtype in ['uint8', 'uint16']:
                return rank.maximum(data, footprint=footprint)
            else:
                return dilation(data, footprint=footprint)


def _dilate_point_labels(data, footprint):
    """
    A more efficient dilation algorithm when we know the image is mostly zero (i.e. point label image)
    """
    dim = len(data.shape)  # 2D or 3D
    if dim != len(footprint.shape):
        raise ValueError(f"incompatible shapes: {data.shape=} {footprint.shape=}")
    if footprint.size == 0:
        raise ValueError(f"cannot handle empty footprint: {footprint.shape=}")

    # To dilate each voxel, we multiply the footprint by the voxel's value.
    #
    # But, to preserve overlapping dilated voxels, we'd like to use `np.maximum`
    # to take the maximum (so that the '0' voxels in the footprint don't overwrite
    # any existing nonzero voxels in the image).
    #
    # To perform `np.maximum` in place, we use the `np.ufunc.at` syntax, which
    # requires an `indices` argument to specify *where* to perform the maximum at.
    # So, we just need to compute the bounding box surrounding the voxel,
    # specified as a tuple of `slice` objects, which acts like
    # data_out[x1:x2, y1:y2, z1:z2] (in 3D) or data_out[x1:x2, y1:y2] (in 2D).
    #
    # As an added complication, we may also need to crop the footprint, if the
    # pixel to be dilated is close to an edge of the array. So, we also need to
    # compute a bounding box surrounding the center of the footprint, also as
    # a tuple of `slice` objects.

    data_corners = np.array([[0]*dim, data.shape], dtype=int)
    fp_corners = np.array([[0]*dim, footprint.shape], dtype=int)

    # For odd-length dimensions, the center of the footprint is unique.
    # For even-length dimensions, we round up.
    fp_center = (fp_corners[0] + (fp_corners[1] - 1) + 1) // 2

    data_out = np.zeros_like(data)
    for data_coords in np.argwhere(data):
        # data_coords is a numpy array, which behaves differently from tuples
        # when indexing another numpy array. We want the tuple behaviour.
        data_value = data[tuple(data_coords)]

        # Make sure that the footprint doesn't exceed the image boundaries
        # If the footprint fits inside the image, then:
        #    - (width of footprint) <= (distance from voxel to image boundary)
        # However, if the footprint would go outside the boundaries, then:
        #    - (distance from voxel to image boundary) < (width of footprint)
        # So we take the minimum to always use the shorter of the two distances.
        distances = [
            # Distance from 0 --> voxel
            np.minimum(data_coords - data_corners[0],
                       fp_center - fp_corners[0]),
            # Distance from voxel --> data boundaries
            np.minimum(data_corners[1] - data_coords,
                       fp_corners[1] - fp_center),
        ]
        data_start = data_coords - distances[0]
        data_stop = data_coords + distances[1]
        fp_start = fp_center - distances[0]
        fp_stop = fp_center + distances[1]

        data_indices = tuple(slice(start, stop) for start, stop in zip(data_start, data_stop))
        fp_indices = tuple(slice(start, stop) for start, stop in zip(fp_start, fp_stop))

        # Now we can apply the (possibly truncated) footprint to the right part
        # of the data array, blending in the data value with the "max" function
        np.maximum.at(
            data_out, data_indices,
            data_value*footprint[fp_indices],
        )

    return data_out


def erode(data, size, shape, dim=None):
    """
    Dilate data using ball structuring element

    :param data: Image or numpy array: 2d or 3d array
    :param size: int or list[int]: If shape={'square', 'cube'}: Corresponds to the length of an edge (size=1 has no effect).\
        If shape={'disk', 'ball'}: Corresponds to the radius, not including the center element (size=0 has no effect).
        Can be specified as 2 or 3 values (depending on the 2D/3D shape) for anisotropic images.
    :param shape: {'square', 'cube', 'disk', 'ball'}
    :param dim: {0, 1, 2}: Dimension of the array which 2D structural element will be orthogonal to. For example, if\
    you wish to apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be: shape=disk, dim=2.
    :return: numpy array: data dilated
    """
    if isinstance(data, Image):
        im_out = data.copy()
        im_out.data = erode(data.data, size, shape, dim)
        return im_out
    else:
        footprint = _get_footprint(shape, size, dim)
        if data.dtype in ['uint8', 'uint16']:
            return rank.minimum(data, footprint=footprint)
        else:
            return erosion(data, footprint=footprint, out=None)


def mutual_information(x, y, nbins=32, normalized=False):
    """
    Compute mutual information

    :param x: 1D numpy.array : flatten data from an image
    :param y: 1D numpy.array : flatten data from an image
    :param nbins: number of bins to compute the contingency matrix (only used if normalized=False)
    :return: float non negative value : mutual information
    """
    if normalized:
        mi = skl_metrics.normalized_mutual_info_score(x, y)
    else:
        c_xy = np.histogram2d(x, y, nbins)[0]
        mi = skl_metrics.mutual_info_score(None, None, contingency=c_xy)
    return mi


def correlation(x, y, type='pearson'):
    """
    Compute pearson or spearman correlation coeff
    Pearson's R is parametric whereas Spearman's R is non parametric (less sensitive)

    :param x: 1D numpy.array : flatten data from an image
    :param y: 1D numpy.array : flatten data from an image
    :param type: str:  'pearson' or 'spearman': type of R correlation coeff to compute
    :return: float value : correlation coefficient (between -1 and 1)
    """

    if type == 'pearson':
        corr = scipy_stats.pearsonr(x, y)[0]
    if type == 'spearman':
        corr = scipy_stats.spearmanr(x, y)[0]

    return corr


def smooth(data, sigmas):
    """
    Smooth data by convolving Gaussian kernel
    :param data: input 3D numpy array
    :param sigmas: Kernel SD in voxel
    :return:
    """
    if len(data.shape) != len(sigmas):
        raise ValueError(f"Expected {len(data.shape)} sigmas, but got {len(sigmas)}")
    return gaussian_filter(data.astype(float), sigmas, order=0, truncate=4.0)


def laplacian(data, sigmas):
    """
    Apply Laplacian filter
    """
    if len(data.shape) != len(sigmas):
        raise ValueError(f"Expected {len(data.shape)} sigmas, but got {len(sigmas)}")
    return gaussian_laplace(data.astype(float), sigmas)


def compute_similarity(data1, data2, metric):
    '''
    Compute a similarity metric between two images data

    :param data1: numpy.array 3D data
    :param data2: numpy.array 3D data
    :param fname_out: file name of the output file. Output file should be either a text file ('.txt') or a pickle file ('.pkl', '.pklz' or '.pickle')
    :param metric: 'mi' for mutual information or 'corr' for pearson correlation coefficient
    :return: tuple with computetd results of similarity, data1 flattened array, data2 flattened array
    '''
    data1_1d = data1.ravel()
    data2_1d = data2.ravel()

    # get indices of non-null voxels from the intersection of both data
    data_mult = data1_1d * data2_1d
    ind_nonnull = np.where(data_mult > ALMOST_ZERO)[0]

    # set new variables with non-null voxels
    data1_1d = data1_1d[ind_nonnull]
    data2_1d = data2_1d[ind_nonnull]

    # compute similarity metric
    if metric == 'mi':
        res = mutual_information(data1_1d, data2_1d, normalized=False)
    elif metric == 'minorm':
        res = mutual_information(data1_1d, data2_1d, normalized=True)
    elif metric == 'corr':
        res = correlation(data1_1d, data2_1d)
    else:
        raise ValueError(f"Don't know what metric to use! Got unsupported: {metric}")

    return res, data1_1d, data2_1d


def otsu(data, nbins):
    thresh = threshold_otsu(data, nbins)
    return data > thresh


def adap(data, block_size, offset):
    mask = data
    for iz in range(data.shape[2]):
        adaptive_thresh = threshold_local(data[:, :, iz], block_size,
                                          method='gaussian', offset=offset)
        mask[:, :, iz] = mask[:, :, iz] > adaptive_thresh
    return mask


def otsu_median(data, size, n_iter):
    data, mask = dipy_mask.median_otsu(data, size, n_iter)
    return mask


def threshold(data, lthr=None, uthr=None):
    if lthr is not None:
        data[data < lthr] = 0
    if uthr is not None:
        data[data > uthr] = 0
    return data


def perc(data, perc_value):
    perc = np.percentile(data, perc_value)
    return data > perc


def binarize(data, bin_thr=0):
    return data > bin_thr


def concatenate_along_last_dimension(data):
    """
    Concatenate multiple data arrays, while ensuring that the last axis of the
    array ("N") is safe to use for operations involving "axis=-1" (e.g. `np.sum(axis=-1)`).

      * 3D (X,Y,Z)   -> 4D (X,Y,Z,N)
      * 4D (X,Y,Z,T) -> 5D (X,Y,Z,T,N)
      * 3D + 4D      -> 4D (X,Y,Z,N)

    :param data: List of ndarrays.
    :return data_concat: concatenate([data])
    """
    ndims = set([arr.ndim for arr in data])

    # Case 1: All images have the same ndim, so add a new axis to every image
    if ndims == {3} or ndims == {4}:
        data = [arr[..., np.newaxis] for arr in data]

    # Case 2: Mix of 3D and 4D images --> No longer supported
    elif ndims == {3, 4}:
        raise ValueError(f"Can only process images with the same number of dimensions, but got mix: {ndims}")

    # Case 3: 2D/5D/etc. images --> Not supported
    else:
        raise ValueError(f"Can only process 3D/4D images, but received images with ndim = {ndims - {3, 4}}")

    return np.concatenate(data, axis=-1)


def denoise_nlmeans(data_in, patch_radius=1, block_radius=5):
    """
    :param data_in: nd_array to denoise

    .. note::
        for more info about patch_radius and block radius, please refer to the dipy website: https://docs.dipy.org/stable/reference/dipy.denoise.html#dipy.denoise.nlmeans.nlmeans
    """

    data_in = np.asarray(data_in)

    block_radius_max = min(data_in.shape) - 1
    block_radius = block_radius_max if block_radius > block_radius_max else block_radius

    sigma = dipy_noise.estimate_sigma(data_in)
    denoised = dipy_nlmeans.nlmeans(data_in, sigma, patch_radius=patch_radius, block_radius=block_radius)

    return denoised


def symmetrize(data, dim):
    """
    Symmetrize data along specified dimension.
    :param data: numpy.array 3D data.
    :param dim: dimension of array to symmetrize along.

    :return data_out: symmetrized data
    """
    data_out = (data + np.flip(data, axis=dim)) / 2.0
    return data_out


def slicewise_mean(data, dim, exclude_zeros=False):
    """
    Compute slicewise mean the specified dimension. Zeros are not inlcuded in the mean.
    :param data: numpy.array 3D data.
    :param dim: dimension of array to symmetrize along.

    :return data_out: slicewise averaged data
    """
    if exclude_zeros:
        data[data == 0] = np.nan
    data_out = np.zeros_like(data)
    for slices in range(data.shape[dim]):
        idx_to_slice = [slice(None)] * data.ndim
        idx_to_slice[dim] = slices
        idx_to_slice = tuple(idx_to_slice)  # requirement of numpy indexing
        if np.isnan(data[idx_to_slice]).all():
            mean_data = [[0]]
        else:
            mean_data = np.nanmean(data[idx_to_slice], keepdims=True)
        data_out[idx_to_slice] = np.broadcast_to(mean_data, data[idx_to_slice].shape)

    return data_out


def denoise_patch2self(data_in, bvals_in, patch_radius=0, model='ols'):
    """
    :param data_in: 4d array to denoise
    :param bvals_in: b-values associated with the 4D DWI data
    :param patch_radius: radius of the p-neighbourhoods defined in the Patch2Self algorithm
    :param model: regression model required to learn the mapping within Patch2Self

    .. note::
        for more info about patch_radius and model, please refer to the dipy website:
        https://docs.dipy.org/stable/examples_built/preprocessing/denoise_patch2self.html
    """
    denoised = dipy_patch2self.patch2self(data_in, bvals_in, patch_radius=patch_radius, model=model)

    return denoised


def remove_small_objects(data, dim_lst, unit, thr):
    """Removes all unconnected objects smaller than the minimum specified size.

    Adapted from:
    https://github.com/ivadomed/ivadomed/blob/master/ivadomed/postprocessing.py#L327
    and
    https://github.com/ivadomed/ivadomed/blob/master/ivadomed/postprocessing.py#L224

    Args:
        data (ndarray): Input data.
        dim_lst (list): Dimensions of a voxel in mm.
        unit (str): Indicates the units of the objects: "mm3" or "vox"
        thr (int or list): Minimal object size to keep in input data.

    Returns:
        ndarray: Array with small objects.
    """
    px, py, pz = dim_lst
    # if there are more than 1 classes, `data` is a 4D array with the 1st
    # dimension representing number of classes. For e.g.
    # for spinal cord (SC) segmentation, num_classes=1,
    # for region-based models with both SC and lesion segmentations, num_classes=2
    num_classes = data.shape[0] if len(data.shape) == 4 else 1

    # structuring element that defines feature connections
    bin_structure = generate_binary_structure(3, 2)

    data_label, n = label(data, structure=bin_structure)

    if isinstance(thr, list) and (num_classes != len(thr)):
        raise ValueError(
            f"Length mismatch for remove small object postprocessing step: threshold length of {len(thr)} "
            f"while the number of predicted class is {num_classes}."
        )
    thr = thr[0] if num_classes == 1 else thr

    if unit == 'vox':
        size_min = thr
    elif unit == 'mm3':
        size_min = np.round(thr / (px * py * pz))
    else:
        logger.error('Please choose a different unit for removeSmall. Choices: vox or mm3')
        exit()

    for idx in range(1, n + 1):
        data_idx = (data_label == idx).astype(int)
        n_nonzero = np.count_nonzero(data_idx)

        if n_nonzero < size_min:
            data[data_label == idx] = 0

    return data
