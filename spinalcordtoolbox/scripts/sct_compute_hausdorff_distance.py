#!/usr/bin/env python
#
# Thinning with the Zhang-Suen algorithm (1984) --> code taken from  https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm
# Computation of the distances between two skeleton
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Sara Dupont
# CREATED: 2015-07-15
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import os

import numpy as np

from spinalcordtoolbox.image import Image, add_suffix, empty_like, change_orientation
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar
from spinalcordtoolbox.utils.sys import init_sct, run_proc, printv, set_global_loglevel
from spinalcordtoolbox.utils.fs import tmp_create, copy, extract_fname
from spinalcordtoolbox.math import binarize

# TODO: display results ==> not only max : with a violin plot of h1 and h2 distribution ? see dev/straightening --> seaborn.violinplot
# TODO: add the option Hyberbolic Hausdorff's distance : see  choi and seidel paper

# ----------------------------------------------------------------------------------------------------------------------
# PARAM ----------------------------------------------------------------------------------------------------------------


class Param:
    def __init__(self):
        self.debug = 0
        self.thinning = True
        self.verbose = 1


# ----------------------------------------------------------------------------------------------------------------------
# THINNING -------------------------------------------------------------------------------------------------------------
class Thinning:
    def __init__(self, im, v=1):
        printv('Thinning ... ', v, 'normal')
        self.image = im
        self.image.data = bin_data(self.image.data)
        self.dim_im = len(self.image.data.shape)

        if self.dim_im == 2:
            self.thinned_image = empty_like(self.image)
            self.thinned_image.data = self.zhang_suen(self.image.data)
            self.thinned_image.absolutepath = add_suffix(self.image.absolutepath, "_thinned")

        elif self.dim_im == 3:
            if not self.image.orientation == 'IRP':
                printv('-- changing orientation ...')
                self.image.change_orientation('IRP')

            thinned_data = np.asarray([self.zhang_suen(im_slice) for im_slice in self.image.data])

            self.thinned_image = empty_like(self.image)
            self.thinned_image.data = thinned_data
            self.thinned_image.absolutepath = add_suffix(self.image.absolutepath, "_thinned")

    # ------------------------------------------------------------------------------------------------------------------
    def get_neighbours(self, x, y, image):
        """
        Return 8-neighbours of image point P1(x,y), in a clockwise order
        code from https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm
        :param x:
        :param y:
        :param image:
        :return:
        """
        # now = time.time()
        x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
        neighbours = [image[x_1][y], image[x_1][y1], image[x][y1], image[x1][y1],     # P2,P3,P4,P5
                      image[x1][y], image[x1][y_1], image[x][y_1], image[x_1][y_1]]    # P6,P7,P8,P9
        # t = time.time() - now
        # printv('t neighbours: ', t)
        return neighbours

    # ------------------------------------------------------------------------------------------------------------------
    def transitions(self, neighbours):
        """
        No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence
        code from https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm
        :param neighbours:
        :return:
        """
        # now = time.time()
        n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
        s = np.sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)
        # t = time.time() - now
        # printv('t transitions sum: ', t)
        return s

    # ------------------------------------------------------------------------------------------------------------------
    def zhang_suen(self, image):
        """
        the Zhang-Suen Thinning Algorithm
        code from https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm
        :param image:
        :return:
        """
        # now = time.time()
        image_thinned = image.copy()  # deepcopy to protect the original image
        changing1 = changing2 = 1  # the points to be removed (set as 0)
        while changing1 or changing2:  # iterates until no further changes occur in the image
            # Step 1
            changing1 = []
            max = len(image_thinned) - 1
            pass_list = [1, max]
            # rows, columns = image_thinned.shape  # x for rows, y for columns

            # for x in range(1, rows - 1):         # No. of  rows
            # for y in range(1, columns - 1):  # No. of columns
            for x, y in non_zero_coord(image_thinned):
                if x not in pass_list and y not in pass_list:
                    # if image_thinned[x][y] == 1:  # Condition 0: Point P1 in the object regions
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = self.get_neighbours(x, y, image_thinned)
                    if (2 <= sum(n) <= 6 and    # Condition 1: 2<= N(P1) <= 6
                        P2 * P4 * P6 == 0 and    # Condition 3
                        P4 * P6 * P8 == 0 and   # Condition 4
                            self.transitions(n) == 1):    # Condition 2: S(P1)=1
                        changing1.append((x, y))
            for x, y in changing1:
                image_thinned[x][y] = 0
            # Step 2
            changing2 = []

            # for x in range(1, rows - 1):
            # for y in range(1, columns - 1):
            for x, y in non_zero_coord(image_thinned):
                if x not in pass_list and y not in pass_list:
                    # if image_thinned[x][y] == 1:  # Condition 0
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = self.get_neighbours(x, y, image_thinned)
                    if (2 <= sum(n) <= 6 and       # Condition 1
                        P2 * P4 * P8 == 0 and       # Condition 3
                        P2 * P6 * P8 == 0 and  # Condition 4
                            self.transitions(n) == 1):    # Condition 2
                        changing2.append((x, y))
            for x, y in changing2:
                image_thinned[x][y] = 0
        # t = time.time() - now
        # printv('t thinning: ', t)
        return image_thinned


# ----------------------------------------------------------------------------------------------------------------------
# HAUSDORFF'S DISTANCE -------------------------------------------------------------------------------------------------
class HausdorffDistance:
    def __init__(self, data1, data2, v=1):
        """
        the hausdorff distance between two sets is the maximum of the distances from a point in any of the sets to the nearest point in the other set
        :return:
        """
        # now = time.time()
        printv('Computing 2D Hausdorff\'s distance ... ', v, 'normal')
        self.data1 = bin_data(data1)
        self.data2 = bin_data(data2)

        self.min_distances_1 = self.relative_hausdorff_dist(self.data1, self.data2, v)
        self.min_distances_2 = self.relative_hausdorff_dist(self.data2, self.data1, v)

        # relatives hausdorff's distances in pixel
        self.h1 = np.max(self.min_distances_1)
        self.h2 = np.max(self.min_distances_2)

        # Hausdorff's distance in pixel
        self.H = max(self.h1, self.h2)
        # t = time.time() - now
        # printv('Hausdorff dist time :', t)

    # ------------------------------------------------------------------------------------------------------------------
    def relative_hausdorff_dist(self, dat1, dat2, v=1):
        h = np.zeros(dat1.shape)
        nz_coord_1 = non_zero_coord(dat1)
        nz_coord_2 = non_zero_coord(dat2)
        if len(nz_coord_1) != 0 and len(nz_coord_2) != 0:
            for x1, y1 in nz_coord_1:
                # for x1 in range(dat1.shape[0]):
                # for y1 in range(dat1.shape[1]):
                # if dat1[x1, y1] == 1:
                d_p1_dat2 = []
                p1 = np.asarray([x1, y1])
                for x2, y2 in nz_coord_2:
                    # for x2 in range(dat2.shape[0]):
                    # for y2 in range(dat2.shape[1]):
                    # if dat2[x2, y2] == 1:
                    p2 = np.asarray([x2, y2])
                    d_p1_dat2.append(np.linalg.norm(p1 - p2))  # Euclidean distance between p1 and p2
                h[x1, y1] = min(d_p1_dat2)
        else:
            printv('Warning: an image is empty', v, 'warning')
        return h


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTE DISTANCES ----------------------------------------------------------------------------------------------------
class ComputeDistances:
    def __init__(self, im1, im2=None, param=None):
        self.im1 = im1
        self.im2 = im2
        self.dim_im = len(self.im1.data.shape)
        self.dim_pix = 0
        self.distances = None
        self.res = ''
        self.param = param
        self.dist1_distribution = None
        self.dist2_distribution = None

        if self.dim_im == 3:
            self.orientation1 = self.im1.orientation
            if self.orientation1 != 'IRP':
                self.im1.change_orientation('IRP', generate_path=True)

            if self.im2 is not None:
                self.orientation2 = self.im2.orientation
                if self.orientation2 != 'IRP':
                    self.im2.change_orientation('IRP', generate_path=True)

        if self.param.thinning:
            self.thinning1 = Thinning(self.im1, self.param.verbose)
            self.thinning1.thinned_image.save()

            if self.im2 is not None:
                self.thinning2 = Thinning(self.im2, self.param.verbose)
                self.thinning2.thinned_image.save()

        if self.dim_im == 2 and self.im2 is not None:
            self.compute_dist_2im_2d()

        if self.dim_im == 3:
            if self.im2 is None:
                self.compute_dist_1im_3d()
            else:
                self.compute_dist_2im_3d()

        if self.dim_im == 2 and self.distances is not None:
            self.dist1_distribution = self.distances.min_distances_1[np.nonzero(self.distances.min_distances_1)]
            self.dist2_distribution = self.distances.min_distances_2[np.nonzero(self.distances.min_distances_2)]
        if self.dim_im == 3:
            self.dist1_distribution = []
            self.dist2_distribution = []

            for d in self.distances:
                if np.nonzero(d.min_distances_1)[0].size:  # Exist non zero values
                    self.dist1_distribution.append(d.min_distances_1[np.nonzero(d.min_distances_1)])
                else:  # all values are zero
                    self.dist1_distribution.append(0)
                if np.nonzero(d.min_distances_2)[0].size:  # Exist non zero values
                    self.dist2_distribution.append(d.min_distances_2[np.nonzero(d.min_distances_2)])
                else:  # all values are zero
                    self.dist2_distribution.append(0)

            self.res = 'Hausdorff\'s distance  -  First relative Hausdorff\'s distance median - Second relative Hausdorff\'s distance median(all in mm)\n'
            for i, d in enumerate(self.distances):
                med1 = np.median(self.dist1_distribution[i])
                med2 = np.median(self.dist2_distribution[i])
                if self.im2 is None:
                    self.res += 'Slice ' + str(i) + ' - slice ' + str(i + 1) + ': ' + str(d.H * self.dim_pix) + '  -  ' + str(med1 * self.dim_pix) + '  -  ' + str(med2 * self.dim_pix) + ' \n'
                else:
                    self.res += 'Slice ' + str(i) + ': ' + str(d.H * self.dim_pix) + '  -  ' + str(med1 * self.dim_pix) + '  -  ' + str(med2 * self.dim_pix) + ' \n'

        printv('-----------------------------------------------------------------------------\n' +
                   self.res, self.param.verbose, 'normal')

        if self.param.verbose == 2:
            self.show_results()

    # ------------------------------------------------------------------------------------------------------------------
    def compute_dist_2im_2d(self):
        nx1, ny1, nz1, nt1, px1, py1, pz1, pt1 = self.im1.dim
        nx2, ny2, nz2, nt2, px2, py2, pz2, pt2 = self.im2.dim

        assert np.isclose(px1, px2) and np.isclose(py1, py2) and np.isclose(px1, py1)
        self.dim_pix = py1

        if self.param.thinning:
            dat1 = self.thinning1.thinned_image.data
            dat2 = self.thinning2.thinned_image.data
        else:
            dat1 = bin_data(self.im1.data)
            dat2 = bin_data(self.im2.data)

        self.distances = HausdorffDistance(dat1, dat2, self.param.verbose)
        self.res = 'Hausdorff\'s distance : ' + str(self.distances.H * self.dim_pix) + ' mm\n\n' \
                   'First relative Hausdorff\'s distance : ' + str(self.distances.h1 * self.dim_pix) + ' mm\n' \
                   'Second relative Hausdorff\'s distance : ' + str(self.distances.h2 * self.dim_pix) + ' mm'

    # ------------------------------------------------------------------------------------------------------------------
    def compute_dist_1im_3d(self):
        nx1, ny1, nz1, nt1, px1, py1, pz1, pt1 = self.im1.dim
        self.dim_pix = py1

        if self.param.thinning:
            dat1 = self.thinning1.thinned_image.data
        else:
            dat1 = bin_data(self.im1.data)

        self.distances = []
        for i, dat_slice in enumerate(dat1[:-1]):
            self.distances.append(HausdorffDistance(bin_data(dat_slice), bin_data(dat1[i + 1]), self.param.verbose))

    # ------------------------------------------------------------------------------------------------------------------
    def compute_dist_2im_3d(self):
        nx1, ny1, nz1, nt1, px1, py1, pz1, pt1 = self.im1.dim
        nx2, ny2, nz2, nt2, px2, py2, pz2, pt2 = self.im2.dim
        # assert np.round(pz1, 5) == np.round(pz2, 5) and np.round(py1, 5) == np.round(py2, 5)
        assert nx1 == nx2
        self.dim_pix = py1

        if self.param.thinning:
            dat1 = self.thinning1.thinned_image.data
            dat2 = self.thinning2.thinned_image.data
        else:
            dat1 = bin_data(self.im1.data)
            dat2 = bin_data(self.im2.data)

        self.distances = []
        for slice1, slice2 in zip(dat1, dat2):
            self.distances.append(HausdorffDistance(slice1, slice2, self.param.verbose))

    # ------------------------------------------------------------------------------------------------------------------
    def show_results(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        plt.hold(True)
        sns.set(style="whitegrid", palette="pastel", color_codes=True)
        plt.figure(figsize=(35, 20))

        data_dist = {"distances": [], "image": [], "slice": []}

        if self.dim_im == 2:
            data_dist["distances"].append([dist * self.dim_pix for dist in self.dist1_distribution])
            data_dist["image"].append(len(self.dist1_distribution) * [1])
            data_dist["slice"].append(len(self.dist1_distribution) * [0])

            data_dist["distances"].append([dist * self.dim_pix for dist in self.dist2_distribution])
            data_dist["image"].append(len(self.dist2_distribution) * [2])
            data_dist["slice"].append(len(self.dist2_distribution) * [0])

        if self.dim_im == 3:
            for i in range(len(self.distances)):
                data_dist["distances"].append([dist * self.dim_pix for dist in self.dist1_distribution[i]])
                data_dist["image"].append(len(self.dist1_distribution[i]) * [1])
                data_dist["slice"].append(len(self.dist1_distribution[i]) * [i])
                data_dist["distances"].append([dist * self.dim_pix for dist in self.dist2_distribution[i]])
                data_dist["image"].append(len(self.dist2_distribution[i]) * [2])
                data_dist["slice"].append(len(self.dist2_distribution[i]) * [i])

        for k in data_dist.keys():  # flatten the lists in data_dist
            data_dist[k] = [item for sublist in data_dist[k] for item in sublist]

        data_dist = pd.DataFrame(data_dist)
        sns.violinplot(x="slice", y="distances", hue="image", data=data_dist, split=True, inner="point", cut=0)
        plt.savefig('violin_plot.png')
        # plt.show()


# ----------------------------------------------------------------------------------------------------------------------
def bin_data(data):
    return np.asarray((data > 0).astype(int))


# ----------------------------------------------------------------------------------------------------------------------
def resample_image(fname, suffix='_resampled.nii.gz', binary=False, npx=0.3, npy=0.3, thr=0.0, interpolation='spline'):
    """
    Resampling function: add a padding, resample, crop the padding
    :param fname: name of the image file to be resampled
    :param suffix: suffix added to the original fname after resampling
    :param binary: boolean, image is binary or not
    :param npx: new pixel size in the x direction
    :param npy: new pixel size in the y direction
    :param thr: if the image is binary, it will be thresholded at thr (default=0) after the resampling
    :param interpolation: type of interpolation used for the resampling
    :return: file name after resampling (or original fname if it was already in the correct resolution)
    """
    im_in = Image(fname)
    orientation = im_in.orientation
    if orientation != 'RPI':
        fname = im_in.change_orientation(im_in, 'RPI', generate_path=True).save().absolutepath

    nx, ny, nz, nt, px, py, pz, pt = im_in.dim

    if np.round(px, 2) != np.round(npx, 2) or np.round(py, 2) != np.round(npy, 2):
        name_resample = extract_fname(fname)[1] + suffix
        if binary:
            interpolation = 'nn'

        if nz == 1:
            # when data is 2d: we convert it to a 3d image in order to avoid conversion problem with 2d data
            # TODO: check if this above problem is still present (now that we are using nibabel instead of nipy)
            run_proc(['sct_image', '-i', ','.join([fname, fname]), '-concat', 'z', '-o', fname])

        run_proc(['sct_resample', '-i', fname, '-mm', str(npx) + 'x' + str(npy) + 'x' + str(pz), '-o', name_resample, '-x', interpolation])

        if nz == 1:  # when input data was 2d: re-convert data 3d-->2d
            run_proc(['sct_image', '-i', name_resample, '-split', 'z'])
            im_split = Image(name_resample.split('.nii.gz')[0] + '_Z0000.nii.gz')
            im_split.save(name_resample)

        if binary:
            img = Image(name_resample)
            img.data = binarize(img.data, thr)
            img.save()

        if orientation != 'RPI':
            name_resample = Image(name_resample) \
                .change_orientation(orientation, generate_path=True) \
                .save() \
                .absolutepath

        return name_resample
    else:
        if orientation != 'RPI':
            fname = add_suffix(fname, "_RPI")
            im_in = change_orientation(im_in, orientation).save(fname)

        printv('Image resolution already ' + str(npx) + 'x' + str(npy) + 'xpz')
        return fname


# ----------------------------------------------------------------------------------------------------------------------
def non_zero_coord(data):
    dim = len(data.shape)
    if dim == 3:
        X, Y, Z = (data > 0).nonzero()
        list_coordinates = [(X[i], Y[i], Z[i]) for i in range(0, len(X))]
    elif dim == 2:
        X, Y = (data > 0).nonzero()
        list_coordinates = [(X[i], Y[i]) for i in range(0, len(X))]
    return list_coordinates


def get_parser():
    parser = SCTArgumentParser(
        description='Compute the Hausdorff\'s distance between two binary images which can be thinned (ie skeletonized).'
                    ' If only one image is inputted, it will be only thinned'
    )

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        "-i",
        required=True,
        help='First Image on which you want to find the skeleton Example: t2star_manual_gmseg.nii.gz',
        metavar=Metavar.file,
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit")
    optional.add_argument(
        "-d",
        help='Second Image on which you want to find the skeleton Example: t2star_manual_gmseg.nii.gz',
        metavar=Metavar.file,
        required=False,
        default=None)
    optional.add_argument(
        "-thinning",
        type=int,
        help="Thinning : find the skeleton of the binary images using the Zhang-Suen algorithm (1984) and use it to "
             "compute the hausdorff's distance",
        required=False,
        default=1,
        choices=(0, 1))
    optional.add_argument(
        "-resampling",
        type=float,
        help="pixel size in mm to resample to Example: 0.5",
        metavar=Metavar.float,
        required=False,
        default=0.1)
    optional.add_argument(
        "-o",
        help='Name of the output file Example: my_hausdorff_dist.txt',
        metavar=Metavar.str,
        required=False,
        default='hausdorff_distance.txt')
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)

    param = Param()
    if param.debug:
        printv('\n*** WARNING: DEBUG MODE ON ***\n')
    else:
        input_fname = arguments.i
        input_second_fname = ''
        output_fname = 'hausdorff_distance.txt'
        resample_to = 0.1

        if arguments.d is not None:
            input_second_fname = arguments.d
        if arguments.thinning is not None:
            param.thinning = bool(arguments.thinning)
        if arguments.resampling is not None:
            resample_to = arguments.resampling
        if arguments.o is not None:
            output_fname = arguments.o
        param.verbose = verbose

        tmp_dir = tmp_create()
        im1_name = "im1.nii.gz"
        copy(input_fname, os.path.join(tmp_dir, im1_name))
        if input_second_fname != '':
            im2_name = 'im2.nii.gz'
            copy(input_second_fname, os.path.join(tmp_dir, im2_name))
        else:
            im2_name = None

        curdir = os.getcwd()
        os.chdir(tmp_dir)

        # now = time.time()
        input_im1 = Image(resample_image(im1_name, binary=True, thr=0.5, npx=resample_to, npy=resample_to))
        input_im1.absolutepath = os.path.basename(input_fname)
        if im2_name is not None:
            input_im2 = Image(resample_image(im2_name, binary=True, thr=0.5, npx=resample_to, npy=resample_to))
            input_im2.absolutepath = os.path.basename(input_second_fname)
        else:
            input_im2 = None

        computation = ComputeDistances(input_im1, im2=input_im2, param=param)

        # TODO change back the orientatin of the thinned image
        if param.thinning:
            computation.thinning1.thinned_image.save(
                os.path.join(curdir, add_suffix(os.path.basename(input_fname), '_thinned')))
            if im2_name is not None:
                computation.thinning2.thinned_image.save(
                    os.path.join(curdir, add_suffix(os.path.basename(input_second_fname), '_thinned')))

        os.chdir(curdir)

        res_fic = open(output_fname, 'w')
        res_fic.write(computation.res)
        res_fic.write('\n\nInput 1: ' + input_fname)
        res_fic.write('\nInput 2: ' + input_second_fname)
        res_fic.close()

        # printv('Total time: ', time.time() - now)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
