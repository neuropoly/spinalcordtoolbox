"""
Custom object and error types.

Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

from operator import itemgetter

from numpy import dot, cross, array, einsum, stack, zeros
from numpy.linalg import norm, inv
import numpy as np
import scipy


class Coordinate:
    """
    Class to represent 3D coordinates.

    :param coord: a list with coordinates [x, y, z] or [x, y, z, value] or a string with coordinates delimited by commas.

    Example:
    .. code:: python

        coord = Coordinate([x, y, z])
        coord = Coordinate([x, y, z, value])
    """
    def __init__(self, coord=None):
        if coord is None:
            coord = [0, 0, 0, 0]
        elif isinstance(coord, str):
            coord = [float(c) for c in coord.split(',')]

        if not isinstance(coord, list):
            raise ValueError("Parameter must be a list with coordinates [x, y, z] or [x, y, z, value] or a "
                             "string with coordinates delimited by commas.")

        if len(coord) == 3:
            coord.append(0)
        elif len(coord) != 4:
            raise ValueError("Parameter must be a list with coordinates [x, y, z] or [x, y, z, value].")

        try:
            [float(c) for c in coord]
        except ValueError:
            raise TypeError(f"All coordinates and the value must be float or int: {coord}") from None

        self.x, self.y, self.z, self.value = coord

    def __iter__(self):
        # Allows for this usage: "for x, y, z, v in [list of Coordinate]"
        return iter((self.x, self.y, self.z, self.value))

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z}, {self.value})"

    def __str__(self):
        return f"{self.x},{self.y},{self.z},{self.value}"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def permute(self, img_src, orient_dest, mode='absolute'):
        """
        Permute coordinate based on source and destination orientation.

        :param img_src : spinalcordtoolbox.Image() object. Must represent the space that the coordinate
                         is currently in. The source orientation and the dimensions are pulled from
                         this image, which are used to permute/invert the coordinate.
        :param orient_dest: The orientation to output the new coordinate in.
        :param mode: Determines how inversions are handled. If 'absolute', the coordinate is recomputed using
                     a new origin based on the source image's maximum dimension for the inverted axes. If
                     'relative', the coordinate is treated as vector and inverted by multiplying by -1.
        :return: numpy array with the new coordinates in the destination orientation.

        Example:
        .. code:: python

            coord.permute(Image('data.nii.gz'), 'RPI')
        """
        # convert coordinates to array
        coord_arr = np.array([self.x, self.y, self.z])
        dim_arr = np.array(img_src.dim[0:3])
        # permutes
        from spinalcordtoolbox.image import _get_permutations
        perm, inversion = _get_permutations(orient_dest, img_src.orientation)  # we need to invert src and dest for this to work
        coord_permute = np.array([coord_arr[perm[0]], coord_arr[perm[1]], coord_arr[perm[2]]])
        dim_permute = np.array([dim_arr[perm[0]], dim_arr[perm[1]], dim_arr[perm[2]]])
        # invert indices based on maximum dimension for each axis
        for i in range(3):
            if inversion[i] == -1:
                if mode == 'absolute':
                    coord_permute[i] = (dim_permute[i] - 1) - coord_permute[i]
                elif mode == 'relative':
                    coord_permute[i] = -1 * coord_permute[i]
                else:
                    raise ValueError(f"Permutations can be 'absolute' or 'relative', but a '{mode}' permutation was "
                                     f"requested instead.")
        return coord_permute

    def __add__(self, other):
        if other == 0:  # this check is necessary for using the function sum() of list
            other = Coordinate()
        return Coordinate([self.x + other.x, self.y + other.y, self.z + other.z, self.value])

    def __radd__(self, other):
        return self + other

    def __truediv__(self, scalar):
        return Coordinate([self.x / float(scalar), self.y / float(scalar), self.z / float(scalar), self.value])


class Centerline:
    """
    This class represents a centerline in an image. Its coordinates can be in voxel space as well as in physical space.
    A centerline is defined by its points and the derivatives of each point.
    When initialized, the lenght of the centerline is computed as well as the coordinate reference system of each plane.
    # TODO: Check if the description above is correct. I've tried to input voxel space coordinates, and it broke the
    #  code. For example, the (removed) method extract_perpendicular_square() is (i think) expecting physical coordinates.
    """
    labels_regions = {'PMJ': 50, 'PMG': 49,
                      'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
                      'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13, 'T7': 14, 'T8': 15, 'T9': 16,
                      'T10': 17, 'T11': 18, 'T12': 19,
                      'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24,
                      'S1': 25, 'S2': 26, 'S3': 27, 'S4': 28, 'S5': 29,
                      'Co': 30}

    convert_vertlabel2disclabel = {'PMJ': 'PMJ', 'PMG': 'PMG',
                                   'C1': 'PMG-C1', 'C2': 'C1-C2', 'C3': 'C2-C3', 'C4': 'C3-C4', 'C5': 'C4-C5',
                                   'C6': 'C5-C6', 'C7': 'C6-C7',
                                   'T1': 'C7-T1', 'T2': 'T1-T2', 'T3': 'T2-T3', 'T4': 'T3-T4', 'T5': 'T4-T5',
                                   'T6': 'T5-T6', 'T7': 'T6-T7', 'T8': 'T7-T8', 'T9': 'T8-T9',
                                   'T10': 'T9-T10', 'T11': 'T10-T11', 'T12': 'T11-T12',
                                   'L1': 'T12-L1', 'L2': 'L1-L2', 'L3': 'L2-L3', 'L4': 'L3-L4', 'L5': 'L4-L5',
                                   'S1': 'L5-S1', 'S2': 'S1-S2', 'S3': 'S2-S3', 'S4': 'S3-S4', 'S5': 'S4-S5',
                                   'Co': 'S5-Co'}

    regions_labels = {50: 'PMJ', 49: 'PMG',
                      1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
                      8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
                      15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12',
                      20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5',
                      25: 'S1', 26: 'S2', 27: 'S3', 28: 'S4', 29: 'S5',
                      30: 'Co'}

    average_vert_length = {'PMJ': 30.0, 'PMG': 15.0, 'C1': 0.0,
                           'C2': 20.176514191661337, 'C3': 17.022090519403065, 'C4': 17.842111671016056,
                           'C5': 16.800356992319429, 'C6': 16.019212889311383, 'C7': 15.715854192723905,
                           'T1': 16.84466163681078, 'T2': 19.865049296865475, 'T3': 21.550165130933905,
                           'T4': 21.761237991438083, 'T5': 22.633281372803687, 'T6': 23.801974227738132,
                           'T7': 24.358357813758332, 'T8': 25.200266294477885, 'T9': 25.315272064638506,
                           'T10': 25.501856729317133, 'T11': 27.619238824308123, 'T12': 29.465119270009946,
                           'L1': 31.89272719870084, 'L2': 33.511890474486449, 'L3': 35.721413718617441}

    # {
    #     "T10": ["T10", 25.543101799896391, 2.0015883550878457],
    #     "T11": ["T11", 27.192970855618441, 1.9996136135271434],
    #     "T12": ["T12", 29.559890137292335, 2.0204112073304121],
    #     "PMG": ["PMG", 12.429867526011929, 2.9899172582983007],
    #     "C3": ["C3", 18.229087873095388, 1.3299710200291315],
    #     "C2": ["C2", 18.859365127066937, 1.5764843286826156],
    #     "C1": ["C1", 0.0, 0.0],
    #     "C7": ["C7", 15.543004729447034, 1.5597730786882851],
    #     "C6": ["C6", 15.967482996580138, 1.4698898678270345],
    #     "PMJ": ["PMJ", 11.38265467206886, 1.5641456310519117],
    #     "C4": ["C4", 17.486130819790912, 1.5888243108648978],
    #     "T8": ["T8", 25.649136105105754, 4.6835454011234718],
    #     "T9": ["T9", 25.581999112288241, 1.9565018840832449],
    #     "T6": ["T6", 23.539740893750668, 1.9073272889977211],
    #     "T7": ["T7", 24.388589291326571, 1.828160893366733],
    #     "T4": ["T4", 22.076131620822075, 1.726133989579701],
    #     "T5": ["T5", 22.402770293433733, 2.0157113843189087],
    #     "T2": ["T2", 19.800131846755267, 1.7600195442391204],
    #     "T3": ["T3", 21.287064228802027, 1.8123109081532691],
    #     "T1": ["T1", 16.525065003339993, 1.6130238001641826],
    #     "L2": ["L2", 34.382300279977912, 2.378543023223767],
    #     "L3": ["L3", 34.804841812064133, 2.7878124182683481],
    #     "L1": ["L1", 32.02934490161379, 2.7697447948338381],
    #     "C5": ["C5", 16.878263370935201, 1.6952832966782569],
    # }

    list_labels = [50, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    potential_list_labels = [50, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                             23, 24, 25, 26, 27, 28, 29, 30]

    def __init__(self, points_x=None, points_y=None, points_z=None, deriv_x=None, deriv_y=None, deriv_z=None,
                 fname=None):
        # initialization of variables
        self.length = 0.0
        self.progressive_length = [0.0]
        self.progressive_length_inverse = [0.0]
        self.incremental_length = [0.0]
        self.incremental_length_inverse = [0.0]

        # variables used for vertebral distribution
        self.first_label, self.last_label = None, None
        self.discs_levels = None
        self.label_reference = None

        self.compute_init_distribution = False

        if fname is not None:
            # Load centerline data from file
            centerline_file = np.load(fname)

            self.points = centerline_file['points']
            self.derivatives = centerline_file['derivatives']

            # This uses the old spelling 'disks_levels' to maintain backwards compatibility
            if 'disks_levels' in centerline_file:
                self.discs_levels = centerline_file['disks_levels'].tolist()
                # convertion of levels to int for future use
                for i in range(len(self.discs_levels)):
                    self.discs_levels[i][3] = int(self.discs_levels[i][3])
                self.label_reference = str(centerline_file['label_reference'])
                self.compute_init_distribution = True
        else:
            # Load centerline data from points and derivatives in parameters
            if points_x is None or points_y is None or points_z is None or deriv_x is None or deriv_y is None or deriv_z is None:
                raise ValueError('Data must be provided to centerline to be initialized')
            self.points = np.array(list(zip(points_x, points_y, points_z)))
            self.derivatives = array(list(zip(deriv_x, deriv_y, deriv_z)))

        self.number_of_points = len(self.points)

        # computation of centerline features, based on points and derivatives
        self.compute_length()
        self.coordinate_system = [self.compute_coordinate_system(index) for index in range(0, self.number_of_points)]
        self.plans_parameters = [self.get_plan_parameters(index) for index in range(0, self.number_of_points)]

        self.matrices = stack([item[4] for item in self.coordinate_system])
        self.inverse_matrices = stack([item[5] for item in self.coordinate_system])
        self.offset_plans = array([item[3] for item in self.plans_parameters])

        # initialization of KDTree for enabling computation of nearest points in centerline
        self.tree_points = scipy.spatial.cKDTree(self.points)

        if self.compute_init_distribution:
            self.compute_vertebral_distribution(discs_levels=self.discs_levels, label_reference=self.label_reference)

    def compute_length(self):
        for i in range(0, self.number_of_points - 1):
            distance = np.sqrt((self.points[i][0] - self.points[i + 1][0]) ** 2 +
                               (self.points[i][1] - self.points[i + 1][1]) ** 2 +
                               (self.points[i][2] - self.points[i + 1][2]) ** 2)
            self.length += distance
            self.progressive_length.append(distance)
            self.incremental_length.append(self.incremental_length[-1] + distance)
        for i in range(self.number_of_points - 1, 0, -1):
            distance = np.sqrt((self.points[i][0] - self.points[i - 1][0]) ** 2 +
                               (self.points[i][1] - self.points[i - 1][1]) ** 2 +
                               (self.points[i][2] - self.points[i - 1][2]) ** 2)
            self.progressive_length_inverse.append(distance)
            self.incremental_length_inverse.append(self.incremental_length_inverse[-1] + distance)

    def find_nearest_index(self, coord):
        """
        This function returns the index of the nearest point from centerline.
        Returns None if list of centerline points is empty.
        Raise an exception is the input coordinate has wrong format.

        :param coord: must be a numpy array [x, y, z]
        :return: index
        """
        if len(self.points) == 0:
            return None

        dist, result_index = self.tree_points.query(coord)

        return result_index

    def find_nearest_indexes(self, array_coordinates):
        dist, result_indexes = self.tree_points.query(array_coordinates)
        return result_indexes

    def get_plan_parameters(self, index):
        """
        This function returns the parameters of the parametric equation of the plane at index.

        :param index: int
        :return: List of parameters [a, b, c, d], corresponding to plane parametric equation a*x + b*y + c*z + d = 0
        """
        if 0 <= index < self.number_of_points:
            a = self.derivatives[index][0]
            b = self.derivatives[index][1]
            c = self.derivatives[index][2]
            d = - (a * self.points[index][0] + b * self.points[index][1] + c * self.points[index][2])
        else:
            raise IndexError(f'ERROR in types.Centerline.get_plan_parameters: index ({index}) should be '
                             f'within [0, {self.number_of_points}[.')

        return [a, b, c, d]

    def get_distances_from_planes(self, coordinates, indexes):
        return (einsum('ij,ij->i', self.derivatives[indexes], coordinates) + self.offset_plans[indexes]) / norm(self.derivatives[indexes], axis=1)

    def compute_coordinate_system(self, index):
        """
        This function computes the cordinate reference system (X, Y, and Z axes) for a given index of centerline.

        :param index: int
        :return:
        """
        if 0 <= index < self.number_of_points:
            origin = self.points[index].copy()
            z_prime_axis = self.derivatives[index].copy()
            z_prime_axis /= norm(z_prime_axis)
            y_axis = array([0, 1, 0])
            y_prime_axis = (y_axis - dot(y_axis, z_prime_axis) * z_prime_axis)
            y_prime_axis /= norm(y_prime_axis)
            x_prime_axis = cross(y_prime_axis, z_prime_axis)
            x_prime_axis /= norm(x_prime_axis)

            matrix_base = array([[x_prime_axis[0], y_prime_axis[0], z_prime_axis[0]],
                                 [x_prime_axis[1], y_prime_axis[1], z_prime_axis[1]],
                                 [x_prime_axis[2], y_prime_axis[2], z_prime_axis[2]]])

            inverse_matrix = inv(matrix_base)
        else:
            raise IndexError(f'ERROR in types.Centerline.compute_coordinate_system: index ({index}) '
                             f'should be within [0, {self.number_of_points}[.')

        return origin, x_prime_axis, y_prime_axis, z_prime_axis, matrix_base, inverse_matrix

    def get_projected_coordinates_on_planes(self, coordinates, indexes):
        unit_derivatives = self.derivatives[indexes]
        unit_derivatives /= np.expand_dims(norm(unit_derivatives, axis=1), axis=1)
        dot_products = np.expand_dims(np.sum((coordinates - self.points[indexes]) * unit_derivatives, axis=1), axis=1)
        return coordinates - unit_derivatives * dot_products

    def get_in_plans_coordinates(self, coordinates, indexes):
        return einsum('rmn,rn->rm',  # matmul Rx2D and Rx1D to get Rx1D
                      self.inverse_matrices[indexes],      # [r, m, n]
                      coordinates - self.points[indexes])  # [r, n]

    def get_inverse_plans_coordinates(self, coordinates, indexes):
        return einsum('rmn,rn->rm',  # matmul Rx2D and Rx1D to get Rx1D
                      self.matrices[indexes],              # [r, m, n]
                      coordinates) + self.points[indexes]  # [r, n]

    def compute_vertebral_distribution(self, discs_levels, label_reference='C1'):
        """
        This function computes the vertebral distribution along the centerline, based on the position of
        intervertebral discs in space. A reference label can be provided (default is top of C1) so that relative
        distances are computed from this reference.

        :param discs_levels: list: list of coordinates with value [[x, y, z, value], [x, y, z, value], ...]
                        the value corresponds to the intervertebral disc label. These coordinates should
                        be defined in the same space as the Centerline (`self`) object. This means that if
                        your Centerline object is defined in physical (`phys`) space, the [x,y,z] coordinates
                        should be defined in physical space. If your Centerline object is defined in pixel
                        (`pix`) space, the [x,y,z] coordinates should be defined in pixel space.
        :param label_reference: reference label from which relative position will be computed.
                        Must be on of self.labels_regions
        """
        self.discs_levels = discs_levels

        # special case for C2, which might not be present because it is difficult to identify
        is_C2_here = False
        C1, C3 = None, None
        for level in discs_levels:
            if level[3] == 2:
                is_C2_here = True
            elif level[3] == 1:
                C1 = level
            elif level[3] == 3:
                C3 = level
        if not is_C2_here and C1 is not None and C3 is not None:
            discs_levels.append([(C1[0] + C3[0]) / 2.0, (C1[1] + C3[1]) / 2.0, (C1[2] + C3[2]) / 2.0, 2])

        self.l_points = [0] * self.number_of_points
        self.dist_points = [0] * self.number_of_points
        self.dist_points_rel = [0] * self.number_of_points
        self.index_disc, index_disc_inv = {}, []

        # extracting each level based on position and computing its nearest point along the centerline
        index_first_label, index_last_label = None, None
        for level in discs_levels:
            if level[3] in self.list_labels:
                coord_level = [level[0], level[1], level[2]]
                disc = self.regions_labels[int(level[3])]
                nearest_index = self.find_nearest_index(coord_level)
                self.index_disc[disc] = nearest_index
                index_disc_inv.append([nearest_index, disc])

                # Finding minimum and maximum label, based on list_labels, which is ordered from top to bottom.
                index_label = self.list_labels.index(int(level[3]))
                if index_first_label is None or index_label < index_first_label:
                    index_first_label = index_label
                if index_last_label is None or index_label > index_last_label:
                    index_last_label = index_label

        # Assign first/last labels once we guarantee that there even *are* valid labels
        if index_last_label is None or index_last_label is None:
            raise ValueError(f"None of the provided disc levels {[lev[3] for lev in discs_levels]} are present in the"
                             f"list of expected disc levels: {list(self.list_labels)}.")
        self.first_label = self.list_labels[index_first_label]
        self.last_label = self.list_labels[index_last_label]

        index_disc_inv.append([0, 'bottom'])
        index_disc_inv.sort(key=itemgetter(0))

        progress_length = zeros(self.number_of_points)
        for i in range(self.number_of_points - 1):
            progress_length[i + 1] = progress_length[i] + self.progressive_length[i]

        # If the reference label (default 'C1' - 1) is not present in the disc labels, then use the uppermost label
        if label_reference not in self.index_disc.keys():
            label_reference = self.regions_labels[self.first_label]  # int label -> str label
        self.label_reference = label_reference

        # Add a special label to handle points above the uppermost label
        # (NB: We only do this if the reference label is not 'C1'. If the reference label *is* 'C1', then we
        #  want to leave the points above C1 as their default value ('0'), since there is further logic that
        #  relies on the assumption that 0 == "above C1".)
        if self.label_reference != 'C1':
            # Get the label that theoretically would be above the reference label
            label_reference_int = self.labels_regions[self.label_reference]       # str label -> int label
            above_label_reference = self.regions_labels[label_reference_int - 1]  # int label -> str label
            # Then, assign it to the last point in the cord
            index_disc_inv.append([self.number_of_points, above_label_reference])

        self.distance_from_C1label = {}
        progress_length_reference = progress_length[self.index_disc[self.label_reference]]
        for disc, i in self.index_disc.items():
            self.distance_from_C1label[disc] = progress_length_reference - progress_length[i]

        # Use the disc indexes to label the points according to the vertebral level they belong to
        for i in range(1, len(index_disc_inv)):
            disc = index_disc_inv[i][1]
            for j in range(index_disc_inv[i - 1][0], index_disc_inv[i][0]):
                self.l_points[j] = disc

        for i in range(self.number_of_points):
            self.dist_points[i] = progress_length_reference - progress_length[i]

        for i in range(self.number_of_points):
            current_label = self.l_points[i]

            if current_label == 0:
                if 'PMG' in self.index_disc:
                    self.dist_points_rel[i] = self.dist_points[i] - self.dist_points[self.index_disc['PMG']]
                    continue
                else:
                    current_label = 'PMG'

            index_current_label = self.list_labels.index(self.labels_regions[current_label])
            if index_current_label < index_first_label:
                reference_level_position = self.dist_points[self.index_disc[self.regions_labels[self.first_label]]]
                self.dist_points_rel[i] = self.dist_points[i] - reference_level_position

            elif index_current_label >= index_last_label:
                reference_level_position = self.dist_points[self.index_disc[self.regions_labels[self.last_label]]]
                self.dist_points_rel[i] = self.dist_points[i] - reference_level_position

            else:  # index_first_label <= index_current_label < index_last_label
                numer = self.dist_points[i] - self.dist_points[self.index_disc[current_label]]

                next_label = self.regions_labels[self.list_labels[index_current_label + 1]]
                if next_label in self.index_disc:
                    denom = abs(self.dist_points[self.index_disc[next_label]] -
                                self.dist_points[self.index_disc[current_label]])
                else:
                    denom = self.average_vert_length[current_label]

                if current_label in ['PMJ', 'PMG']:
                    self.dist_points_rel[i] = 1 - numer/denom
                else:
                    self.dist_points_rel[i] = numer/denom

    def get_closest_index(self, vertebral_level, relative_position, backup_index, backup_centerline):
        # Parse 'list_labels' ([50, 49, 1, 2, 3, ...]) to find the index that corresponds to the vertebral level
        if vertebral_level in self.labels_regions.keys():
            vertebral_number = self.labels_regions[vertebral_level]
            vertebral_index = self.potential_list_labels.index(vertebral_number)
        elif vertebral_level == 0:  # level == 0 --> above the C1 vertebral level
            vertebral_index = -1    # so, ensure index is always outside C1 index
        else:
            raise ValueError(f"vertebral_level must be a level string (C1, C2, C3...) or 0, but got {vertebral_level}.")

        # If the vertebral level is within the centerline, compute the closest index using relative position
        if self.list_labels.index(self.first_label) <= vertebral_index < self.list_labels.index(self.last_label):
            closest_index = self.get_closest_to_relative_position(vertebral_level, relative_position)

        # Otherwise, the vertebral level is outside the centerline. So, we first replace the level with the closest
        # level that IS in the centerline, then we compute the closest index using absolute position instead.
        else:
            if vertebral_index < self.list_labels.index(self.first_label):
                label = self.first_label
            else:
                assert vertebral_index >= self.list_labels.index(self.last_label)
                label = self.last_label
            closest_centerline_level = self.regions_labels[label]
            closest_index = self.get_closest_to_absolute_position(closest_centerline_level,
                                                                  backup_index, backup_centerline)

        return closest_index

    def get_closest_to_relative_position(self, vertebral_level, relative_position):
        """
        :param vertebral_level: the name of a vertebral level, as a string
        :param relative_position: the relative position [0, 1] from upper disc
        """
        dist_indices = [
            (abs(self.dist_points_rel[i] - relative_position), i)
            for i in range(self.number_of_points)
            if self.l_points[i] == vertebral_level
        ]
        if dist_indices:
            return min(dist_indices)[1]
        else:
            return None

    def get_closest_to_absolute_position(self, reference_level, backup_index, backup_centerline):
        """
        :param reference_level: the name of a vertebral level, as a string. this should be the closest vertebral level
                                from the centerline, since 'backup_index' is outside the centerline's top/bottom levels.
        :backup_index: the index of the desired slice, in the 'backup_centerline' space.
        :backup_centerline: an alternate centerline that contains both 'reference_level' and the 'backup_index'
        """
        # Get the slice index of the reference vertebral level within the backup centerline
        backup_index_reference = backup_centerline.index_disc[reference_level]

        # Get the 'mm' positions of both the desired slice index and the reference vertebral level
        backup_position = backup_centerline.dist_points[backup_index]
        backup_position_reference = backup_centerline.dist_points[backup_index_reference]

        # Compute the relative distance between the reference level and the desired slice
        relative_distance_backup = backup_position_reference - backup_position

        # The steps below are a mathematical equivalent to:
        #   1. Find the 'mm' position of the reference level in the 'self' centerline
        #   2. Add the distance that was computed for the 'backup' centerline, which results in new 'mm'
        #      position corresponding to the position of the 'backup_index' slice, but in the 'self' centerline space.
        #   3. For this new position, find the corresponding index in the 'self' centerline.
        # -------------------------------------------------------------------------------------------------------
        # Normalize the positions of the current centerline's points using the position of the reference level.
        # This makes it so that the reference level is '0.0', and all other points are +/- from it.
        dist_points = np.array(self.dist_points)
        dist_points_norm = dist_points - self.dist_points[self.index_disc[reference_level]]
        # Add the computed distance so that there is a new "center point" between the + and - values.
        dist_points_shifted = dist_points_norm + relative_distance_backup
        # Then, take the absolute value of all the points, causing the center point to become a minimum.
        # The index of this minimum is our desired index.
        closest_index = np.argmin(np.abs(dist_points_shifted))

        return closest_index

    def save_centerline(self, image=None, fname_output='centerline.sct'):
        if image is not None:
            image_output = image.copy()
            image_output.data = np.zeros_like(image_output.data, dtype=np.float32)

            for i in range(self.number_of_points):
                current_label = self.l_points[i]
                current_coord = self.points[i]
                current_dist_rel = self.dist_points_rel[i]
                if current_label in self.labels_regions:
                    coord_pix = image.transfo_phys2pix([current_coord])[0]
                    image_output.data[int(coord_pix[0]), int(coord_pix[1]), int(coord_pix[2])] = float(self.labels_regions[current_label]) + current_dist_rel

            image_output.save(fname_output, dtype='float32')
        else:
            # save a .centerline file containing the centerline
            if self.discs_levels is None:
                np.savez(fname_output, points=self.points, derivatives=self.derivatives)
            else:
                # This uses the old spelling 'disks_levels' to maintain backwards compatibility
                np.savez(fname_output, points=self.points, derivatives=self.derivatives,
                         disks_levels=self.discs_levels, label_reference=self.label_reference)

    def display(self, mode='absolute'):
        """
        This function display the centerline using matplotlib. Two modes are available: absolute and relative.
        The absolute mode display the absolute position of centerline points.
        The relative mode display the centerline position relative to the reference label (default is C1). This mode
        requires the computation of vertebral distribution beforehand.

        :param mode: {absolute, relative} see description of function for details
        """

        import matplotlib.pyplot as plt

        plt.figure(1)
        ax = plt.subplot(211)  # noqa: F841

        if mode == 'absolute':
            plt.plot([coord[2] for coord in self.points], [coord[0] for coord in self.points])
        else:
            position_reference = self.points[self.index_disc[self.label_reference]]
            plt.plot([coord[2] - position_reference[2] for coord in self.points],
                     [coord[0] - position_reference[0] for coord in self.points])
            for label_disc in self.labels_regions:
                if label_disc in self.index_disc:
                    point = self.points[self.index_disc[label_disc]]
                    plt.scatter(point[2] - position_reference[2], point[0] - position_reference[0], s=5)

        plt.grid()
        plt.title("X")
        # ax.set_aspect('equal')
        plt.xlabel('z')
        plt.ylabel('x')
        ax = plt.subplot(212)  # noqa: F841

        if mode == 'absolute':
            plt.plot([coord[2] for coord in self.points], [coord[1] for coord in self.points])
        else:
            position_reference = self.points[self.index_disc[self.label_reference]]
            plt.plot([coord[2] - position_reference[2] for coord in self.points],
                     [coord[1] - position_reference[1] for coord in self.points])
            for label_disc in self.labels_regions:
                if label_disc in self.index_disc:
                    point = self.points[self.index_disc[label_disc]]
                    plt.scatter(point[2] - position_reference[2], point[1] - position_reference[1], s=5)

        plt.grid()
        plt.title("Y")
        # ax.set_aspect('equal')
        plt.xlabel('z')
        plt.ylabel('y')
        plt.show()


class EmptyArrayError(ValueError):
    """Custom exception to distinguish between general SciPy ValueErrors."""


class MissingDiscsError(ValueError):
    """Custom exception to indicate that no disc labels were found."""
