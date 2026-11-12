"""
Resample data using nibabel

Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

# TODO: remove resample_file (not needed)

import logging

import numpy as np

from spinalcordtoolbox.image import Image, add_suffix
from spinalcordtoolbox.utils.shell import display_viewer_syntax

from spinalcordtoolbox.utils.sys import LazyLoader

nib = LazyLoader("nib", globals(), "nibabel")
nib_processing = LazyLoader("nib_processing", globals(), "nibabel.processing")

logger = logging.getLogger(__name__)


def resample_nib(image, new_size=None, new_size_type=None, image_dest=None, interpolation='linear', mode='nearest',
                 preserve_codes=False):
    """
    Resample a nibabel or Image object based on a specified resampling factor.
    Can deal with 2d, 3d or 4d image objects.

    :param image: nibabel or Image image.
    :param new_size: list of float: Resampling factor, final dimension or resolution, depending on new_size_type.
    :param new_size_type: {'vox', 'factor', 'mm'}: Feature used for resampling. Examples:
        new_size=[128, 128, 90], new_size_type='vox' --> Resampling to a dimension of 128x128x90 voxels
        new_size=[2, 2, 2], new_size_type='factor' --> 2x isotropic upsampling
        new_size=[1, 1, 5], new_size_type='mm' --> Resampling to a resolution of 1x1x5 mm
    :param image_dest: Destination image to resample the input image to. In this case, new_size and new_size_type
        are ignored
    :param interpolation: {'nn', 'linear', 'spline'}. The interpolation type
    :param mode: Outside values are filled with 0 ('constant') or nearest value ('nearest').
    :param preserve_codes: bool: Whether to preserve the qform/sform codes from the original image.
        - If set to False, nibabel will overwrite the existing qform/sform codes with `0` and `2` respectively.
        - This option is set to False by default, as resampling typically implies that the new image is in a different
          space. Therefore, since the image is no longer aligned to any scan, the previous codes are now invalid.
        - However, if you plan to eventually resample back to the native space later on, you may wish to set this
          option to True to preserve the codes. (For more information, see:
          https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3005)
    :return: The resampled nibabel or Image image (depending on the input object type).
    """

    # set interpolation method
    dict_interp = {'nn': 0, 'linear': 1, 'spline': 2}

    # If input is an Image object, create nibabel object from it
    if isinstance(image, nib.Nifti1Image):
        img = image
    elif isinstance(image, Image):
        img = nib.Nifti1Image(image.data, image.hdr.get_best_affine(), image.hdr)
    else:
        raise TypeError(f'Invalid image type: {type(image)}')

    # convert to floating point if we're doing arithmetic interpolation
    if interpolation != 'nn' and img.get_data_dtype().kind in 'biu':
        original_dtype = img.get_data_dtype()
        img = nib.Nifti1Image(img.get_fdata(), img.header.get_best_affine(), img.header)
        img.set_data_dtype(img.dataobj.dtype)
        logger.debug("Converting image from type '%s' to type '%s' for %s interpolation",
                     original_dtype, img.get_data_dtype(), interpolation)

    if image_dest is None:
        # Get dimensions of data
        p = img.header.get_zooms()
        shape = img.header.get_data_shape()

        # determine the number of dimensions that should be resampled
        ndim_r = img.ndim
        if img.ndim == 4:
            # For 4D images, only resample (x, y, z) dim, and preserve 't' dim
            ndim_r = 3

        # compute new shape based on specific resampling method
        if new_size_type == 'vox':
            shape_r = tuple([int(new_size[i]) for i in range(ndim_r)])
        elif new_size_type == 'factor':
            if len(new_size) == 1:
                # isotropic resampling
                new_size = tuple([new_size[0] for i in range(ndim_r)])
            # compute new shape as: shape_r = shape * f
            shape_r = tuple([int(np.round(shape[i] * float(new_size[i]))) for i in range(ndim_r)])
        elif new_size_type == 'mm':
            if len(new_size) == 1:
                # isotropic resampling
                new_size = tuple([new_size[0] for i in range(ndim_r)])
            # compute new shape as: shape_r = shape * (p_r / p)
            shape_r = tuple([int(np.round(shape[i] * float(p[i]) / float(new_size[i]))) for i in range(ndim_r)])
        else:
            raise ValueError("'new_size_type' is not recognized.")

        if img.ndim == 4:
            # Copy over 't' dim (i.e. number of volumes should be unaffected)
            shape_r = shape_r + (shape[3],)

        # Generate 3d affine transformation: R
        affine = img.affine[:4, :4]
        affine[3, :] = np.array([0, 0, 0, 1])  # satisfy to nifti convention. Otherwise it grabs the temporal
        logger.debug(f'Affine matrix: \n{affine}')
        R = np.eye(4)
        for i in range(3):
            try:
                R[i, i] = img.shape[i] / float(shape_r[i])
            except ZeroDivisionError as e:
                raise ValueError(
                    f"Requested resampling (`-{new_size_type} {'x'.join(new_size)}`) would resample the input image from {shape} to {shape_r}. "
                    f"Please double-check the requested resampling parameters. (Note that for the 'factor' and 'mm' resampling types, "
                    f"voxels will be rounded to the nearest whole number, which may result in a shape of [0].)"
                ) from e

        affine_r = np.dot(affine, R)
        reference = (shape_r, affine_r)

    # If reference is provided
    else:
        if isinstance(image_dest, nib.Nifti1Image):
            reference = image_dest
        elif isinstance(image_dest, Image):
            reference = nib.Nifti1Image(image_dest.data, affine=image_dest.hdr.get_best_affine(), header=image_dest.hdr)
        else:
            raise TypeError(f'Invalid image type: {type(image_dest)}')

    if img.ndim == 3:
        # we use mode 'nearest' to overcome issue #2453
        img_r = nib_processing.resample_from_to(
            img, to_vox_map=reference, order=dict_interp[interpolation], mode=mode, cval=0.0, out_class=None)

    elif img.ndim == 4:
        # TODO: Cover img_dest with 4D volumes
        # Import here instead of top of the file because this is an isolated case and nibabel takes time to import
        data4d = np.zeros(shape_r)
        # Loop across 4th dimension and resample each 3d volume
        for it in range(img.shape[3]):
            # Create dummy 3d nibabel image
            data3d = np.asanyarray(img.dataobj)[..., it]
            nii_tmp = nib.Nifti1Image(data3d, affine, dtype=data3d.dtype)
            img3d_r = nib_processing.resample_from_to(
                nii_tmp, to_vox_map=(shape_r[:-1], affine_r), order=dict_interp[interpolation], mode=mode,
                cval=0.0, out_class=None)
            data4d[..., it] = np.asanyarray(img3d_r.dataobj)
        # Create 4d nibabel Image
        img_r = nib.Nifti1Image(data4d, affine_r)  # Can't be int64 (#4408)
        # Copy over the TR parameter from original 4D image (otherwise it will be incorrectly set to 1)
        img_r.header.set_zooms(list(img_r.header.get_zooms()[0:3]) + [img.header.get_zooms()[3]])

    # preserve the codes from the original image, which will otherwise get overwritten with 0/2
    if preserve_codes:
        img_r.header['qform_code'] = img.header['qform_code']
        img_r.header['sform_code'] = img.header['sform_code']

    # Convert back to proper type
    if isinstance(image, nib.Nifti1Image):
        return img_r
    else:
        assert isinstance(image, Image)  # already checked at the start of the function
        return Image(np.asanyarray(img_r.dataobj), hdr=img_r.header, orientation=image.orientation, dim=img_r.header.get_data_shape())


def resample_file(fname_data, fname_out, new_size, new_size_type, interpolation, verbose, fname_ref=None):
    """This function will resample the specified input
    image file to the target size.
    Can deal with 2d, 3d or 4d image objects.

    :param fname_data: The input image filename.
    :param fname_out: The output image filename.
    :param new_size: The target size, i.e. 0.25x0.25
    :param new_size_type: Unit of resample (mm, vox, factor)
    :param interpolation: The interpolation type
    :param verbose: verbosity level
    :param fname_ref: Reference image to resample input image to
    """
    # Load data
    logger.info('load data...')
    img = Image(fname_data)
    if fname_ref is not None:
        img_ref = Image(fname_ref)
    else:
        img_ref = None

    img_r = resample_nib(img, new_size.split('x'), new_size_type, image_dest=img_ref, interpolation=interpolation)

    # build output file name
    if fname_out == '':
        fname_out = add_suffix(fname_data, '_r')
    else:
        fname_out = fname_out

    # save data
    img_r.save(fname_out)

    # to view results
    display_viewer_syntax([fname_out], verbose=verbose)

    return img_r
