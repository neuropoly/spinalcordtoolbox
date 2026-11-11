"""
Functions dealing with spinal cord straightening

Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

# TODO: only input Image instead of file names

import os
import time
import logging
import bisect

import numpy as np

from spinalcordtoolbox.types import Centerline
from spinalcordtoolbox.image import Image, spatial_crop, generate_output_file, pad_image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.utils.sys import sct_progress_bar
from spinalcordtoolbox.utils.fs import tmp_create, rmtree, copy, mv, extract_fname

from spinalcordtoolbox.scripts import sct_apply_transfo, sct_resample, sct_image

logger = logging.getLogger(__name__)


class SpinalCordStraightener(object):
    def __init__(self, input_filename, centerline_filename, debug=0, param_centerline=ParamCenterline(),
                 interpolation_warp='spline', rm_tmp_files=1, verbose=1, precision=2.0, threshold_distance=10,
                 safe_zone=0, output_filename='', path_output=''):
        self.input_filename = input_filename
        self.centerline_filename = centerline_filename
        self.output_filename = output_filename
        self.debug = debug
        self.interpolation_warp = interpolation_warp
        self.remove_temp_files = rm_tmp_files  # remove temporary files
        self.verbose = verbose
        self.precision = precision
        self.threshold_distance = threshold_distance
        self.safe_zone = safe_zone
        self.path_output = path_output
        self.use_straight_reference = False
        self.centerline_reference_filename = ""
        self.discs_input_filename = ""
        self.discs_ref_filename = ""
        self.speed_factor = 1.0  # Speed parameter
        self.xy_size = 70  # in mm
        self.param_centerline = param_centerline

        # QC metrics
        self.accuracy_results = 0
        self.mse_straightening = 0.0
        self.max_distance_straightening = 0.0
        self.elapsed_time = 0.0
        self.elapsed_time_accuracy = 0.0

        # Outputs
        self.curved2straight = True
        self.straight2curved = True
        self.path_qc = None

        self.template_orientation = 0

    def straighten(self):
        """
        Straighten spinal cord. Steps: (everything is done in physical space)
        1. open input image and centreline image
        2. extract bspline fitting of the centreline, and its derivatives
        3. compute length of centerline
        4. compute and generate straight space
        5. compute transformations
            for each voxel of one space: (done using matrices --> improves speed by a factor x300)
                a. determine which plane of spinal cord centreline it is included
                b. compute the position of the voxel in the plane (X and Y distance from centreline, along the plane)
                c. find the correspondant centreline point in the other space
                d. find the correspondance of the voxel in the corresponding plane
        6. generate warping fields for each transformations
        7. write warping fields and apply them

        step 5.b: how to find the corresponding plane?
            The centerline plane corresponding to a voxel correspond to the nearest point of the centerline.
            However, we need to compute the distance between the voxel position and the plane to be sure it is part of the plane and not too distant.
            If it is more far than a threshold, warping value should be 0.

        step 5.d: how to make the correspondance between centerline point in both images?
            Both centerline have the same lenght. Therefore, we can map centerline point via their position along the curve.
            If we use the same number of points uniformely along the spinal cord (1000 for example), the correspondance is straight-forward.

        :return:
        """
        # Initialization
        fname_anat = self.input_filename
        fname_centerline = self.centerline_filename
        fname_output = self.output_filename
        remove_temp_files = self.remove_temp_files
        verbose = self.verbose

        # start timer
        start_time = time.time()

        # Extract path/file/extension
        path_anat, file_anat, ext_anat = extract_fname(fname_anat)

        path_tmp = tmp_create(basename="straighten-spinalcord")

        # Copying input data to tmp folder
        logger.info('Copy files to tmp folder...')
        Image(fname_anat, check_sform=True).save(os.path.join(path_tmp, "data.nii"))
        Image(fname_centerline, check_sform=True).save(os.path.join(path_tmp, "centerline.nii.gz"))

        if self.use_straight_reference:
            Image(self.centerline_reference_filename, check_sform=True).save(os.path.join(path_tmp, "centerline_ref.nii.gz"))
        if self.discs_input_filename != '':
            Image(self.discs_input_filename, check_sform=True).save(os.path.join(path_tmp, "labels_input.nii.gz"))
        if self.discs_ref_filename != '':
            Image(self.discs_ref_filename, check_sform=True).save(os.path.join(path_tmp, "labels_ref.nii.gz"))

        # go to tmp folder
        curdir = os.getcwd()
        os.chdir(path_tmp)

        # Change orientation of the input centerline into RPI
        image_centerline = Image("centerline.nii.gz").change_orientation("RPI").save("centerline_rpi.nii.gz",
                                                                                     mutable=True)

        # Get dimension
        nx, ny, nz, nt, px, py, pz, pt = image_centerline.dim
        if self.speed_factor != 1.0:
            intermediate_resampling = True
            px_r, py_r, pz_r = px * self.speed_factor, py * self.speed_factor, pz * self.speed_factor
        else:
            intermediate_resampling = False

        if intermediate_resampling:
            mv('centerline_rpi.nii.gz', 'centerline_rpi_native.nii.gz')
            pz_native = pz
            # TODO: remove system call
            sct_resample.main([
                '-i', 'centerline_rpi_native.nii.gz',
                '-mm', f'{px_r}x{py_r}x{pz_r}',
                '-o', 'centerline_rpi.nii.gz',
                '-v', '0',
            ])
            image_centerline = Image('centerline_rpi.nii.gz')
            nx, ny, nz, nt, px, py, pz, pt = image_centerline.dim

        if np.min(image_centerline.data) < 0 or np.max(image_centerline.data) > 1:
            image_centerline.data[image_centerline.data < 0] = 0
            image_centerline.data[image_centerline.data > 1] = 1
            image_centerline.save()

        # 2. extract bspline fitting of the centerline, and its derivatives
        img_ctl = Image('centerline_rpi.nii.gz')
        _, arr_ctl_phys, arr_ctl_der_phys, _ = get_centerline(img_ctl, self.param_centerline,
                                                              verbose=verbose, space="phys")
        centerline = Centerline(*arr_ctl_phys, *arr_ctl_der_phys)
        number_of_points = centerline.number_of_points

        # ==========================================================================================
        logger.info('Create the straight space and the safe zone')
        # 3. compute length of centerline
        # compute the length of the spinal cord based on fitted centerline and size of centerline in z direction

        # Computation of the safe zone.
        # The safe zone is defined as the length of the spinal cord for which an axial segmentation will be complete
        # The safe length (to remove) is computed using the safe radius and the angle of the
        # last centerline point with the inferior-superior direction. Formula: Ls = Rs * sin(angle)
        # Calculate Ls for both edges and remove appropriate number of centerline points
        radius_safe = 0.0  # mm

        # inferior edge
        u = centerline.derivatives[0]
        v = np.array([0, 0, -1])

        angle_inferior = np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
        length_safe_inferior = radius_safe * np.sin(angle_inferior)

        # superior edge
        u = centerline.derivatives[-1]
        v = np.array([0, 0, 1])
        angle_superior = np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
        length_safe_superior = radius_safe * np.sin(angle_superior)

        # remove points
        inferior_bound = bisect.bisect(centerline.progressive_length, length_safe_inferior) - 1
        superior_bound = centerline.number_of_points - bisect.bisect(centerline.progressive_length_inverse,
                                                                     length_safe_superior)

        z_centerline = centerline.points[:, 2]
        length_centerline = centerline.length
        size_z_centerline = z_centerline[-1] - z_centerline[0]

        # compute the size factor between initial centerline and straight bended centerline
        factor_curved_straight = length_centerline / size_z_centerline
        middle_slice = (z_centerline[0] + z_centerline[-1]) / 2.0

        bound_curved = [z_centerline[inferior_bound], z_centerline[superior_bound]]
        start_point = (z_centerline[inferior_bound] - middle_slice) * factor_curved_straight + middle_slice
        end_point = (z_centerline[superior_bound] - middle_slice) * factor_curved_straight + middle_slice

        logger.info(f'Length of spinal cord: {length_centerline}')
        logger.info(f'Size of spinal cord in z direction: {size_z_centerline}')
        logger.info(f'Ratio length/size: {factor_curved_straight}')
        logger.info(f'Safe zone boundaries (curved space): {bound_curved}')

        # 4. compute and generate straight space
        # points along curved centerline are already regularly spaced.
        # calculate position of points along straight centerline

        # Create straight NIFTI volumes.
        # ==========================================================================================
        # TODO: maybe this if case is not needed?
        if self.use_straight_reference:
            image_centerline_pad = Image('centerline_rpi.nii.gz')
            nx, ny, nz, nt, px, py, pz, pt = image_centerline_pad.dim

            fname_ref = 'centerline_ref_rpi.nii.gz'
            image_centerline_straight = Image('centerline_ref.nii.gz') \
                .change_orientation("RPI") \
                .save(fname_ref, mutable=True)
            _, arr_ctl_phys, arr_ctl_der_phys, _ = get_centerline(image_centerline_straight, self.param_centerline,
                                                                  verbose=verbose, space="phys")
            centerline_straight = Centerline(*arr_ctl_phys, *arr_ctl_der_phys)
            nx_s, ny_s, nz_s, nt_s, px_s, py_s, pz_s, pt_s = image_centerline_straight.dim

            # Prepare warping fields headers
            hdr_warp = image_centerline_pad.hdr.copy()
            hdr_warp.set_data_dtype('float32')
            hdr_warp_s = image_centerline_straight.hdr.copy()
            hdr_warp_s.set_data_dtype('float32')

            if self.discs_input_filename != "" and self.discs_ref_filename != "":
                discs_input_image = Image('labels_input.nii.gz')
                coord = discs_input_image.getNonZeroCoordinates(sorting='z', reverse_coord=True)
                coord_physical = []
                for c in coord:
                    c_p = discs_input_image.transfo_pix2phys([[c.x, c.y, c.z]]).tolist()[0]
                    c_p.append(c.value)
                    coord_physical.append(c_p)
                centerline.compute_vertebral_distribution(coord_physical)
                centerline.save_centerline(image=discs_input_image, fname_output='discs_input_image.nii.gz')

                discs_ref_image = Image('labels_ref.nii.gz')
                coord = discs_ref_image.getNonZeroCoordinates(sorting='z', reverse_coord=True)
                coord_physical = []
                for c in coord:
                    c_p = discs_ref_image.transfo_pix2phys([[c.x, c.y, c.z]]).tolist()[0]
                    c_p.append(c.value)
                    coord_physical.append(c_p)
                centerline_straight.compute_vertebral_distribution(coord_physical)
                centerline_straight.save_centerline(image=discs_ref_image, fname_output='discs_ref_image.nii.gz')

        else:
            logger.info(f'Start/end points (straight space): {[start_point, end_point]}')
            logger.info('Pad input volume to account for spinal cord length...')
            offset_z = 0

            # if the destination image is resampled, we still create the straight reference space with the native
            # resolution.
            # TODO: Maybe this if case is not needed?
            if intermediate_resampling:
                padding_z = int(np.ceil(1.5 * ((length_centerline - size_z_centerline) / 2.0) / pz_native))
                sct_image.main([
                    '-i', 'centerline_rpi_native.nii.gz',
                    '-o', 'tmp.centerline_pad_native.nii.gz',
                    '-pad', f'0,0,{padding_z}',
                    '-v', '0',
                ])
                image_centerline_pad = Image('centerline_rpi_native.nii.gz')
                nx, ny, nz, nt, px, py, pz, pt = image_centerline_pad.dim
                start_point_coord_native = image_centerline_pad.transfo_phys2pix([[0, 0, start_point]])[0]
                end_point_coord_native = image_centerline_pad.transfo_phys2pix([[0, 0, end_point]])[0]
                straight_size_x = int(self.xy_size / px)
                straight_size_y = int(self.xy_size / py)
                warp_space_x = [int(np.round(nx / 2)) - straight_size_x, int(np.round(nx / 2)) + straight_size_x]
                warp_space_y = [int(np.round(ny / 2)) - straight_size_y, int(np.round(ny / 2)) + straight_size_y]
                if warp_space_x[0] < 0:
                    warp_space_x[1] += warp_space_x[0] - 2
                    warp_space_x[0] = 0
                if warp_space_y[0] < 0:
                    warp_space_y[1] += warp_space_y[0] - 2
                    warp_space_y[0] = 0

                spec = dict((
                    (0, warp_space_x),
                    (1, warp_space_y),
                    (2, (0, end_point_coord_native[2] - start_point_coord_native[2])),
                ))
                spatial_crop(Image("tmp.centerline_pad_native.nii.gz"), spec).save(
                    "tmp.centerline_pad_crop_native.nii.gz")

                fname_ref = 'tmp.centerline_pad_crop_native.nii.gz'
                offset_z = 4
            else:
                fname_ref = 'tmp.centerline_pad_crop.nii.gz'

            nx, ny, nz, nt, px, py, pz, pt = image_centerline.dim
            padding_z = int(np.ceil(1.5 * ((length_centerline - size_z_centerline) / 2.0) / pz)) + offset_z
            image_centerline_pad = pad_image(image_centerline, pad_z_i=padding_z, pad_z_f=padding_z)
            nx, ny, nz = image_centerline_pad.data.shape
            hdr_warp = image_centerline_pad.hdr.copy()
            hdr_warp.set_data_dtype('float32')
            start_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, start_point]])[0]
            end_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, end_point]])[0]

            straight_size_x = int(self.xy_size / px)
            straight_size_y = int(self.xy_size / py)
            warp_space_x = [int(np.round(nx / 2)) - straight_size_x, int(np.round(nx / 2)) + straight_size_x]
            warp_space_y = [int(np.round(ny / 2)) - straight_size_y, int(np.round(ny / 2)) + straight_size_y]

            if warp_space_x[0] < 0:
                warp_space_x[1] += warp_space_x[0] - 2
                warp_space_x[0] = 0
            if warp_space_x[1] >= nx:
                warp_space_x[1] = nx - 1
            if warp_space_y[0] < 0:
                warp_space_y[1] += warp_space_y[0] - 2
                warp_space_y[0] = 0
            if warp_space_y[1] >= ny:
                warp_space_y[1] = ny - 1

            spec = dict((
                (0, warp_space_x),
                (1, warp_space_y),
                (2, (0, end_point_coord[2] - start_point_coord[2] + offset_z)),
            ))
            image_centerline_straight = spatial_crop(image_centerline_pad, spec)

            nx_s, ny_s, nz_s, nt_s, px_s, py_s, pz_s, pt_s = image_centerline_straight.dim
            hdr_warp_s = image_centerline_straight.hdr.copy()
            hdr_warp_s.set_data_dtype('float32')

            if self.template_orientation == 1:
                raise NotImplementedError()

            start_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, start_point]])[0]
            end_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, end_point]])[0]

            number_of_voxel = nx * ny * nz
            logger.debug(f'Number of voxels: {number_of_voxel}')

            time_centerlines = time.time()

            coord_straight = np.empty((number_of_points, 3))
            coord_straight[:, 0] = int(np.round(nx_s / 2))
            coord_straight[:, 1] = int(np.round(ny_s / 2))
            coord_straight[:, 2] = np.linspace(0, end_point_coord[2] - start_point_coord[2], number_of_points)
            coord_phys_straight = image_centerline_straight.transfo_pix2phys(coord_straight, mode='absolute')
            derivs_straight = np.empty((number_of_points, 3))
            derivs_straight[:] = image_centerline_straight.transfo_pix2phys([[0, 0, 1]], mode='relative')
            centerline_straight = Centerline(
                coord_phys_straight[:, 0], coord_phys_straight[:, 1], coord_phys_straight[:, 2],
                derivs_straight[:, 0], derivs_straight[:, 1], derivs_straight[:, 2],
            )

            time_centerlines = time.time() - time_centerlines
            logger.info(f'Time to generate centerline: {np.round(time_centerlines * 1000.0)} ms')

        if verbose == 2:
            # TODO: use OO
            import matplotlib.pyplot as plt
            from datetime import datetime
            curved_points = centerline.progressive_length
            straight_points = centerline_straight.progressive_length
            range_points = np.linspace(0, 1, number_of_points)
            dist_curved = np.zeros(number_of_points)
            dist_straight = np.zeros(number_of_points)
            for i in range(1, number_of_points):
                dist_curved[i] = dist_curved[i - 1] + curved_points[i - 1] / centerline.length
                dist_straight[i] = dist_straight[i - 1] + straight_points[i - 1] / centerline_straight.length
            plt.plot(range_points, dist_curved)
            plt.plot(range_points, dist_straight)
            plt.grid(True)
            plt.savefig(f'fig_straighten_{datetime.now():%y%m%d%H%M%S%f}.png')
            plt.close()

        lookup_curved2straight = list(range(centerline.number_of_points))
        if self.discs_input_filename != "":
            # create look-up table curved to straight
            for index in range(centerline.number_of_points):
                disc_label = centerline.l_points[index]
                relative_position = centerline.dist_points_rel[index]
                idx_closest = centerline_straight.get_closest_index(disc_label, relative_position,
                                                                    backup_index=index,
                                                                    backup_centerline=centerline)
                if idx_closest is not None:
                    lookup_curved2straight[index] = idx_closest
                else:
                    lookup_curved2straight[index] = 0
        # If we're generating a curved2straight warping field, we can update the boundaries of the straight-space safe
        # zone by finding out the straight-space indices that the "safe zone" curved-space indices get mapped to.
        coord_bound_straight_inferior = lookup_curved2straight[inferior_bound]
        coord_bound_straight_superior = lookup_curved2straight[superior_bound]
        z_centerline_straight = centerline_straight.points[:, 2]
        bound_straight = [z_centerline_straight[coord_bound_straight_inferior],
                          z_centerline_straight[coord_bound_straight_superior]]
        logger.info(f'Safe zone boundaries (straight space): {bound_straight}')
        # Remove duplicates from the start and end of the lookup table
        # This is necessary because `get_closest_index` will repeat itself once the first/last indexes are reached
        # NB: Any slice index set to `0` will have its warping field value set to `-100000` (i.e. no warping)
        for p in range(0, len(lookup_curved2straight) - 1):
            if lookup_curved2straight[p] == lookup_curved2straight[p + 1]:
                lookup_curved2straight[p] = 0
            else:
                break
        for p in range(len(lookup_curved2straight) - 1, 0, -1):
            if lookup_curved2straight[p] == lookup_curved2straight[p - 1]:
                lookup_curved2straight[p] = 0
            else:
                break
        lookup_curved2straight = np.array(lookup_curved2straight)

        lookup_straight2curved = list(range(centerline_straight.number_of_points))
        if self.discs_input_filename != "":
            for index in range(centerline_straight.number_of_points):
                disc_label = centerline_straight.l_points[index]
                relative_position = centerline_straight.dist_points_rel[index]
                idx_closest = centerline.get_closest_index(disc_label, relative_position,
                                                           backup_index=index,
                                                           backup_centerline=centerline_straight)
                if idx_closest is not None:
                    lookup_straight2curved[index] = idx_closest
        # Remove duplicates from the start and end of the lookup table
        # This is necessary because `get_closest_index` will repeat itself once the first/last indexes are reached
        # NB: Any slice index set to `0` will have its warping field value set to `-100000` (i.e. no warping)
        for p in range(0, len(lookup_straight2curved) - 1):
            if lookup_straight2curved[p] == lookup_straight2curved[p + 1]:
                lookup_straight2curved[p] = 0
            else:
                break
        for p in range(len(lookup_straight2curved) - 1, 0, -1):
            if lookup_straight2curved[p] == lookup_straight2curved[p - 1]:
                lookup_straight2curved[p] = 0
            else:
                break
        lookup_straight2curved = np.array(lookup_straight2curved)

        # Create volumes containing curved and straight warping fields
        data_warp_curved2straight = np.zeros((nx_s, ny_s, nz_s, 1, 3))
        data_warp_straight2curved = np.zeros((nx, ny, nz, 1, 3))

        # 5. compute transformations
        # Curved and straight images and the same dimensions, so we compute both warping fields at the same time.
        # b. determine which plane of spinal cord centreline it is included

        if self.curved2straight:
            for u in sct_progress_bar(range(nz_s)):
                x_s, y_s, z_s = np.mgrid[0:nx_s, 0:ny_s, u:u + 1]
                indexes_straight = np.array(list(zip(x_s.ravel(), y_s.ravel(), z_s.ravel())))
                physical_coordinates_straight = image_centerline_straight.transfo_pix2phys(indexes_straight)
                nearest_indexes_straight = centerline_straight.find_nearest_indexes(physical_coordinates_straight)
                distances_straight = centerline_straight.get_distances_from_planes(physical_coordinates_straight,
                                                                                   nearest_indexes_straight)
                lookup = lookup_straight2curved[nearest_indexes_straight]
                # NB: Any indexes that were mapped to `0` will have their warping field value set to `-100000` (i.e. no warping),
                #     regardless of the value of `self.threshold_distance`.
                indexes_out_distance_straight = np.logical_or(
                    np.logical_or(distances_straight > self.threshold_distance,
                                  distances_straight < -self.threshold_distance), lookup == 0)
                projected_points_straight = centerline_straight.get_projected_coordinates_on_planes(
                    physical_coordinates_straight, nearest_indexes_straight)
                coord_in_planes_straight = centerline_straight.get_in_plans_coordinates(projected_points_straight,
                                                                                        nearest_indexes_straight)

                coord_straight2curved = centerline.get_inverse_plans_coordinates(coord_in_planes_straight, lookup)
                displacements_straight = coord_straight2curved - physical_coordinates_straight
                # Invert Z coordinate as ITK & ANTs physical coordinate system is LPS- (RAI+)
                # while ours is LPI-
                # Refs: https://sourceforge.net/p/advants/discussion/840261/thread/2a1e9307/#fb5a
                #       https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html
                displacements_straight[:, 2] = -displacements_straight[:, 2]
                displacements_straight[indexes_out_distance_straight] = [100000.0, 100000.0, 100000.0]

                data_warp_curved2straight[indexes_straight[:, 0], indexes_straight[:, 1], indexes_straight[:, 2], 0, :]\
                    = -displacements_straight

        if self.straight2curved:
            for u in sct_progress_bar(range(nz)):
                x, y, z = np.mgrid[0:nx, 0:ny, u:u + 1]
                indexes = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())))
                physical_coordinates = image_centerline_pad.transfo_pix2phys(indexes)
                nearest_indexes_curved = centerline.find_nearest_indexes(physical_coordinates)
                distances_curved = centerline.get_distances_from_planes(physical_coordinates,
                                                                        nearest_indexes_curved)
                lookup = lookup_curved2straight[nearest_indexes_curved]
                # NB: Any indexes that were mapped to `0` will have their warping field value set to `-100000` (i.e. no warping),
                #     regardless of the value of `self.threshold_distance`.
                indexes_out_distance_curved = np.logical_or(
                    np.logical_or(distances_curved > self.threshold_distance,
                                  distances_curved < -self.threshold_distance), lookup == 0)
                projected_points_curved = centerline.get_projected_coordinates_on_planes(physical_coordinates,
                                                                                         nearest_indexes_curved)
                coord_in_planes_curved = centerline.get_in_plans_coordinates(projected_points_curved,
                                                                             nearest_indexes_curved)

                coord_curved2straight = centerline_straight.get_inverse_plans_coordinates(coord_in_planes_curved, lookup)
                displacements_curved = coord_curved2straight - physical_coordinates

                displacements_curved[:, 2] = -displacements_curved[:, 2]
                displacements_curved[indexes_out_distance_curved] = [100000.0, 100000.0, 100000.0]

                data_warp_straight2curved[indexes[:, 0], indexes[:, 1], indexes[:, 2], 0, :] = -displacements_curved

        # Creation of the safe zone based on pre-calculated safe boundaries
        coord_bound_curved_inf, coord_bound_curved_sup = image_centerline_pad.transfo_phys2pix(
            [[0, 0, bound_curved[0]]]), image_centerline_pad.transfo_phys2pix([[0, 0, bound_curved[1]]])
        coord_bound_straight_inf, coord_bound_straight_sup = image_centerline_straight.transfo_phys2pix(
            [[0, 0, bound_straight[0]]]), image_centerline_straight.transfo_phys2pix([[0, 0, bound_straight[1]]])

        # note: we need to ensure that straight bounds don't go outside the warping field's z-dimension, since
        #       `transfo_phys2pix` has no safeguards, and could return negative or too large indices.
        coord_bound_straight_inf[0][2] = max(0, coord_bound_straight_inf[0][2])
        coord_bound_straight_sup[0][2] = min(nz_s - 1, coord_bound_straight_sup[0][2])

        if self.safe_zone:
            logger.info('Applying safe zone to warping fields')
            # use -100000 to match the value used by `self.threshold_distance` earlier (due to `-displacements_curved`)
            # NB: make sure to _not_ include the safe zone slices themselves when applying the safe zone
            #     e.g. if the image has z dimension [160] and the safe zone is [0, 159] (i.e. the full image is safe),
            #     then nothing should be overwritten here (and it won't, since `:0` and `160:` both return empty arrays)
            data_warp_curved2straight[:, :, :coord_bound_straight_inf[0][2], 0, :] = -100000.0
            data_warp_curved2straight[:, :, (coord_bound_straight_sup[0][2]+1):, 0, :] = -100000.0
            data_warp_straight2curved[:, :, :coord_bound_curved_inf[0][2], 0, :] = -100000.0
            data_warp_straight2curved[:, :, (coord_bound_curved_sup[0][2]+1):, 0, :] = -100000.0

        # Generate warp files as a warping fields
        hdr_warp_s.set_intent('vector', (), '')
        hdr_warp_s.set_data_dtype('float32')
        hdr_warp.set_intent('vector', (), '')
        hdr_warp.set_data_dtype('float32')
        if self.curved2straight:
            img = Image(param=data_warp_curved2straight, hdr=hdr_warp_s)
            img.save('tmp.curve2straight.nii.gz')
            logger.info('Warping field generated: tmp.curve2straight.nii.gz')

        if self.straight2curved:
            img = Image(param=data_warp_straight2curved, hdr=hdr_warp)
            img.save('tmp.straight2curve.nii.gz')
            logger.info('Warping field generated: tmp.straight2curve.nii.gz')

        image_centerline_straight.save(fname_ref)
        if self.curved2straight:
            logger.info('Apply transformation to input image...')
            sct_apply_transfo.main(['-i', 'data.nii',
                                    '-d', fname_ref,
                                    '-w', 'tmp.curve2straight.nii.gz',
                                    '-o', 'tmp.anat_rigid_warp.nii.gz',
                                    '-x', 'spline',
                                    '-v', '0'])

        if self.accuracy_results:
            time_accuracy_results = time.time()
            # compute the error between the straightened centerline/segmentation and the central vertical line.
            # Ideally, the error should be zero.
            # Apply deformation to input image
            logger.info('Apply transformation to centerline image...')
            sct_apply_transfo.main(['-i', 'centerline.nii.gz',
                                    '-d', fname_ref,
                                    '-w', 'tmp.curve2straight.nii.gz',
                                    '-o', 'tmp.centerline_straight.nii.gz',
                                    '-x', 'nn',
                                    '-v', '0'])
            file_centerline_straight = Image('tmp.centerline_straight.nii.gz', verbose=verbose)
            nx, ny, nz, nt, px, py, pz, pt = file_centerline_straight.dim
            coordinates_centerline = file_centerline_straight.getNonZeroCoordinates(sorting='z')
            mean_coord = []
            for z in range(coordinates_centerline[0].z, coordinates_centerline[-1].z):
                temp_mean = [coord.value for coord in coordinates_centerline if coord.z == z]
                if temp_mean:
                    mean_value = np.mean(temp_mean)
                    mean_coord.append(
                        np.mean([[coord.x * coord.value / mean_value, coord.y * coord.value / mean_value]
                                 for coord in coordinates_centerline if coord.z == z], axis=0))

            # compute error between the straightened centerline and the straight line.
            x0 = file_centerline_straight.data.shape[0] / 2.0
            y0 = file_centerline_straight.data.shape[1] / 2.0
            count_mean = 0
            if number_of_points >= 10:
                mean_c = mean_coord[2:-2]  # we don't include the four extrema because there are usually messy.
            else:
                mean_c = mean_coord
            for coord_z in mean_c:
                if not np.isnan(np.sum(coord_z)):
                    dist = ((x0 - coord_z[0]) * px) ** 2 + ((y0 - coord_z[1]) * py) ** 2
                    self.mse_straightening += dist
                    dist = np.sqrt(dist)
                    if dist > self.max_distance_straightening:
                        self.max_distance_straightening = dist
                    count_mean += 1
            self.mse_straightening = np.sqrt(self.mse_straightening / float(count_mean))

            self.elapsed_time_accuracy = time.time() - time_accuracy_results

        os.chdir(curdir)

        # Generate output file (in current folder)
        # TODO: do not uncompress the warping field, it is too time consuming!
        logger.info('Generate output files...')
        if self.curved2straight:
            generate_output_file(os.path.join(path_tmp, "tmp.curve2straight.nii.gz"),
                                 os.path.join(self.path_output, "warp_curve2straight.nii.gz"), verbose)
        if self.straight2curved:
            generate_output_file(os.path.join(path_tmp, "tmp.straight2curve.nii.gz"),
                                 os.path.join(self.path_output, "warp_straight2curve.nii.gz"), verbose)

        # create ref_straight.nii.gz file that can be used by other SCT functions that need a straight reference space
        if self.curved2straight:
            copy(os.path.join(path_tmp, "tmp.anat_rigid_warp.nii.gz"),
                 os.path.join(self.path_output, "straight_ref.nii.gz"))
            # move straightened input file
            if fname_output == '':
                fname_straight = generate_output_file(os.path.join(path_tmp, "tmp.anat_rigid_warp.nii.gz"),
                                                      os.path.join(self.path_output,
                                                                   f"{file_anat}_straight{ext_anat}"), verbose)
            else:
                fname_straight = generate_output_file(os.path.join(path_tmp, "tmp.anat_rigid_warp.nii.gz"),
                                                      os.path.join(self.path_output, fname_output),
                                                      verbose)  # straightened anatomic

        # Remove temporary files
        if remove_temp_files:
            logger.info('Remove temporary files...')
            rmtree(path_tmp)

        if self.accuracy_results:
            logger.info(f'Maximum x-y error: {self.max_distance_straightening} mm')
            logger.info(f'Accuracy of straightening (MSE): {self.mse_straightening} mm')

        # display elapsed time
        self.elapsed_time = int(np.round(time.time() - start_time))

        return fname_straight
