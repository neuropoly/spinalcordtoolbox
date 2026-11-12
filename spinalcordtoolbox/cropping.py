"""
Functions dealing with image cropping

Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import logging
import numpy as np

from .image import Image, zeros_like


logger = logging.getLogger(__name__)


class BoundingBox(object):
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax


class ImageCropper(object):
    def __init__(self, img_in=None):
        """
        :param img_in:
        """
        self.img_in = img_in
        # Call one of the `get_bbox_from_*` methods to set the bounding box.
        self.bbox = None

    def crop(self, img_in=None, background=None, dilate=None):
        """
        Crop image (change dimension)

        :param img_in: Image: Image to be cropped. If not provided, `self.img_in` will be cropped instead.
        :param background: int: If set, the output image will not be cropped. Instead, voxels outside the bounding
        box will be set to the value specified by this parameter.
        :param dilate: If set, a list of 3 integers specifying an extra margin to keep around the bounding box,
        in each of the x-, y-, and z-directions.
        :return Image: img_out
        """
        if img_in is None:
            if self.img_in:
                img_in = self.img_in
            else:
                raise ValueError('No image to crop. Please provide `img_in` when initializing the cropper, or when '
                                 'calling the `crop` method.')

        bbox = self.bbox
        if bbox is None:
            raise ValueError(
                'Use one of the `get_bbox_from_*` methods to set the bounding '
                'box before calling `ImageCropper.crop()`.')

        if dilate is not None:
            bbox = BoundingBox(
                xmin=max(bbox.xmin-dilate[0], 0), xmax=min(bbox.xmax+dilate[0], img_in.dim[0]-1),
                ymin=max(bbox.ymin-dilate[1], 0), ymax=min(bbox.ymax+dilate[1], img_in.dim[1]-1),
                zmin=max(bbox.zmin-dilate[2], 0), zmax=min(bbox.zmax+dilate[2], img_in.dim[2]-1),
            )

        logger.info(f"Bounding box: x=[{bbox.xmin}, {bbox.xmax+1}], y=[{bbox.ymin}, {bbox.ymax+1}], z=[{bbox.zmin}, {bbox.zmax+1}]")

        # Crop the image
        if background is None:
            logger.info("Cropping the image...")
            data_crop = img_in.data[bbox.xmin:bbox.xmax+1, bbox.ymin:bbox.ymax+1, bbox.zmin:bbox.zmax+1]
            img_out = Image(param=data_crop, hdr=img_in.hdr)
            # adapt the origin in the qform matrix
            new_origin = np.dot(img_out.hdr.get_qform(), [bbox.xmin, bbox.ymin, bbox.zmin, 1])
            img_out.hdr.structarr['qoffset_x'] = new_origin[0]
            img_out.hdr.structarr['qoffset_y'] = new_origin[1]
            img_out.hdr.structarr['qoffset_z'] = new_origin[2]
            # adapt the origin in the sform matrix
            new_origin = np.dot(img_out.hdr.get_sform(), [bbox.xmin, bbox.ymin, bbox.zmin, 1])
            img_out.hdr.structarr['srow_x'][-1] = new_origin[0]
            img_out.hdr.structarr['srow_y'][-1] = new_origin[1]
            img_out.hdr.structarr['srow_z'][-1] = new_origin[2]

        # Set voxels outside the bbox to the value 'background'
        else:
            logger.info(f"Setting voxels outside the bounding box to: {background}")
            img_out = img_in.copy()
            img_out.data[:] = background
            img_out.data[bbox.xmin:bbox.xmax+1, bbox.ymin:bbox.ymax+1, bbox.zmin:bbox.zmax+1] = \
                img_in.data[bbox.xmin:bbox.xmax+1, bbox.ymin:bbox.ymax+1, bbox.zmin:bbox.zmax+1]

        return img_out

    def get_bbox_from_minmax(self, xmin, xmax, ymin, ymax, zmin, zmax):
        """
        Get voxel bounding box from xmin, xmax, ymin, ymax, zmin, zmax user input.
        Replaces '-1' with max dim along each axis, '-2' with max dim minus 1, etc.
        """
        if xmax < 0:
            xmax += self.img_in.dim[0]
        if ymax < 0:
            ymax += self.img_in.dim[1]
        if zmax < 0:
            zmax += self.img_in.dim[2]
        self.bbox = BoundingBox(xmin, xmax, ymin, ymax, zmin, zmax)

    def get_bbox_from_mask(self, img_mask):
        """
        Get bounding box from input binary mask, by looking at min/max values of the binary object in each dimension.
        """
        data_nonzero = np.nonzero(img_mask.data)
        # find min and max boundaries of the mask
        self.bbox = BoundingBox(
            min(data_nonzero[0]), max(data_nonzero[0]),
            min(data_nonzero[1]), max(data_nonzero[1]),
            min(data_nonzero[2]), max(data_nonzero[2]))

    def get_bbox_from_ref(self, img_ref):
        """
        Get bounding box from input reference image, by looking at min/max indices in each dimension.
        img_ref and self.img_in should have the same dimensions.
        """
        from spinalcordtoolbox.resampling import resample_nib
        #  Check that img_ref has the same length as img_in
        if not len(img_ref.data.shape) == len(self.img_in.data.shape):
            logger.error(f"Inconsistent dimensions: n_dim(img_ref)={len(img_ref.data.shape)}; n_dim(img_in)={len(self.img_in.data.shape)}")
            raise ValueError('Inconsistent dimensions')
        # Fill reference data with ones
        img_ref.data[:] = 1
        # Resample new image (in reference coordinates) into input image
        img_ref_r = resample_nib(img_ref, image_dest=self.img_in, interpolation='nn', mode='constant')
        # img_ref_r.save('test.nii')  # for debug
        # Get bbox from this resampled mask
        self.get_bbox_from_mask(img_ref_r)

    def get_bbox_from_gui(self):
        """
        Launch a GUI. The medial sagittal plane of the image is shown. User selects two points: top-left and bottom-
        right of the cropping window.
        Note: There is no cropping along the right-left direction.
        :return:
        """
        from spinalcordtoolbox.gui import base
        from spinalcordtoolbox.gui.sagittal import launch_sagittal_dialog

        # Change orientation to SAL (for displaying sagittal view in the GUI)
        native_orientation = self.img_in.orientation
        self.img_in.change_orientation('SAL')

        # Launch GUI
        params = base.AnatomicalParams()
        params.vertebraes = [1, 2]  # TODO: Have user draw a sliding rectangle instead (more intuitive)
        params.subtitle = "Click on the top-left (Label 1) and bottom-right (Label 2) of the image to select your " \
                          "cropping window."
        img_labels = zeros_like(self.img_in)
        launch_sagittal_dialog(self.img_in, img_labels, params)

        # Extract coordinates
        img_labels.change_orientation(native_orientation)
        cropping_coord = img_labels.getNonZeroCoordinates(sorting='value')
        # Since there is no cropping along the R-L direction, xmin/xmax are based on image dimension
        self.bbox = BoundingBox(
            0, img_labels.dim[0],
            min(cropping_coord[0].y, cropping_coord[1].y),
            max(cropping_coord[0].y, cropping_coord[1].y),
            min(cropping_coord[0].z, cropping_coord[1].z),
            max(cropping_coord[0].z, cropping_coord[1].z))
        # Put back input image in native orientation
        self.img_in.change_orientation(native_orientation)
