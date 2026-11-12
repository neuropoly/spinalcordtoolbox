"""
Iterative closest point algorithm

This file contains an implementation of the iterative closest point algorithm.
This algo registers two sets of points (3D coordinates) together.

Adapted from:
https://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python

NOTES ON ITK Transform Files:
https://intranet.neuro.polymtl.ca/geek-tips/image-processing-software/advanced-normalization-tools-ants.html#itk-transform-file

Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

# TODO: homogeneize input parameters: (src=src, dest=dest), instead of (dest, src).
# TODO: add full affine transfo
# TODO: normalize SSE: currently, it depends on the number of landmarks

import os
import logging

import numpy as np
from scipy.optimize import minimize

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import LazyLoader

nib = LazyLoader("nib", globals(), "nibabel")

logger = logging.getLogger(__name__)

sse_results = []
ini_param_rotation = 0.5  # rad
ini_param_trans_x = 270.0  # pix
ini_param_trans_y = -150.0  # pix
initial_step = 2


def register_landmarks(fname_src, fname_dest, dof, fname_affine='affine.txt', verbose=1):
    """
    Register two NIFTI volumes containing landmarks
    :param fname_src: fname of source landmarks
    :param fname_dest: fname of destination landmarks
    :param dof: degree of freedom. Separate with "_". Example: Tx_Ty_Tz_Rx_Ry_Sz
    :param fname_affine: output affine transformation
    :param verbose: 0, 1, 2
    :return:
    """
    # open src label
    im_src = Image(fname_src)
    # coord_src = im_src.getNonZeroCoordinates(sorting='value')  # landmarks are sorted by value
    coord_src = im_src.getCoordinatesAveragedByValue()  # landmarks are sorted by value
    # open dest labels
    im_dest = Image(fname_dest)
    # coord_dest = im_dest.getNonZeroCoordinates(sorting='value')
    coord_dest = im_dest.getCoordinatesAveragedByValue()
    # Reorganize landmarks

    points_src, points_dest = [], []
    for coord in coord_src:
        point_src = im_src.transfo_pix2phys([[coord.x, coord.y, coord.z]])
        # convert NIFTI to ITK world coordinate
        # points_src.append([point_src[0][0], point_src[0][1], point_src[0][2]])
        points_src.append([-point_src[0][0], -point_src[0][1], point_src[0][2]])
    for coord in coord_dest:
        point_dest = im_dest.transfo_pix2phys([[coord.x, coord.y, coord.z]])
        # convert NIFTI to ITK world coordinate
        # points_dest.append([point_dest[0][0], point_dest[0][1], point_dest[0][2]])
        points_dest.append([-point_dest[0][0], -point_dest[0][1], point_dest[0][2]])

    # display
    logger.info(f"Labels src: {str(points_src)}")
    logger.info(f"Labels dest: {str(points_dest)}")
    logger.info(f"Degrees of freedom (dof): {dof}")

    if len(coord_src) != len(coord_dest):
        raise Exception('Error: number of source and destination landmarks are not the same, so landmarks cannot be paired.')

    # estimate transformation
    # N.B. points_src and points_dest are inverted below, because ITK uses inverted transformation matrices, i.e., src->dest is defined in dest instead of src.
    # (rotation_matrix,
    #  translation_array,
    #  points_moving_reg,
    #  points_moving_barycenter) = getRigidTransformFromLandmarks(points_dest,
    #                                                             points_src,
    #                                                             constraints=dof,
    #                                                             verbose=verbose,
    #                                                             path_qc=path_qc)
    (rotation_matrix,
     translation_array,
     points_moving_reg,
     points_moving_barycenter) = getRigidTransformFromLandmarks(points_src,
                                                                points_dest,
                                                                constraints=dof,
                                                                verbose=verbose)
    # writing rigid transformation file
    # N.B. x and y dimensions have a negative sign to ensure compatibility between Python and ITK transfo
    open(fname_affine, 'w').write(
        "#Insight Transform File V1.0\n"
        "#Transform 0\n"
        "Transform: AffineTransform_double_3_3\n"
        "Parameters: "
        f"{rotation_matrix[0, 0]:.9f} {rotation_matrix[0, 1]:.9f} {rotation_matrix[0, 2]:.9f} "
        f"{rotation_matrix[1, 0]:.9f} {rotation_matrix[1, 1]:.9f} {rotation_matrix[1, 2]:.9f} "
        f"{rotation_matrix[2, 0]:.9f} {rotation_matrix[2, 1]:.9f} {rotation_matrix[2, 2]:.9f} "
        f"{translation_array[0, 0]:.9f} {translation_array[0, 1]:.9f} {translation_array[0, 2]:.9f}\n"
        "FixedParameters: "
        f"{points_moving_barycenter[0]:.9f} "
        f"{points_moving_barycenter[1]:.9f} "
        f"{points_moving_barycenter[2]:.9f}\n"
    )


def SSE(pointsA, pointsB):
    """
    Compute sum of squared error between pair-wise landmarks
    :param pointsA:
    :param pointsB:
    :return:
    """
    return np.sum(np.array(pointsA[:, 0:3] - pointsB[:, 0:3])**2.0)


def real_optimization_parameters(param_from_optimizer, initial_param=0, initial_step=10):
    # The initial step for the Nelder-Mead algorithm is of (initial_param * 5e-2) which is too small when we want initial_param = 30 pix and step = 5 or 10.
    # This function allows to choose the scale of the steps after the first movement
    step_factor = float(initial_step) / float(initial_param * 5e-2)
    real_param = initial_param + (param_from_optimizer - initial_param) * step_factor
    return real_param


def Metric_Images(imageA, imageB, type=''):

    data_A_list = np.asanyarray(nib.load(imageA).dataobj).tolist()
    data_B_list = np.asanyarray(nib.load(imageA).dataobj).tolist()

    # Define both list of intensity
    list_A = []
    list_B = []
    for i in range(len(data_A_list)):
        list_A = list_A + data_A_list[i]
        list_B = list_B + data_B_list[i]
    # Calculate metric depending on the type
    if type == 'MeanSquares':
        result_metric = 1.0 / (len(list_A)) * np.sum(np.array([list_A[i][0] - list_B[i][0] for i in range(len(list_A))])**2)
        # result_metric = 1/(len(list_A)) * np.sum(np.array(list_A - list_B)**2)

    if type == 'Correlation':
        result_metric = 1.0 / (len(list_A)) * np.sum(np.absolute(np.array([list_A[i][0] - list_B[i][0] for i in range(len(list_A))])))

    if type == 'MI':
        logger.info("\nto do: MI")

    # Return results
    logger.info(f"\nResult of metric is: {str(result_metric)}")
    return result_metric


def minimize_transform(params, points_dest, points_src, constraints):
    """
    Cost function to minimize
    :param params:
    :param points_dest:
    :param points_src:
    :param constraints:
    :return: sum of squared error between pair-wise landmarks
    """
    # initialize dof
    dof = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    # initialize dictionary to relate constraints index to dof
    dict_dof = {'Tx': 0, 'Ty': 1, 'Tz': 2, 'Rx': 3, 'Ry': 4, 'Rz': 5, 'Sx': 6, 'Sy': 7, 'Sz': 8}
    # extract constraints
    list_constraints = constraints.split('_')
    # loop across constraints and update dof
    for i in range(len(list_constraints)):
        dof[dict_dof[list_constraints[i]]] = params[i]
    # convert dof to more intuitive variables
    tx, ty, tz, alpha, beta, gamma, scx, scy, scz = dof[0], dof[1], dof[2], dof[3], dof[4], dof[5], dof[6], dof[7], dof[8]
    # build rotation matrix
    rotation_matrix = np.array([
        [
            np.cos(alpha)*np.cos(beta),
            np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma),
            np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma),
        ], [
            np.sin(alpha)*np.cos(beta),
            np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma),
            np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma),
        ], [
            np.sin(beta)*-1,
            np.cos(beta)*np.sin(gamma),
            np.cos(beta)*np.cos(gamma),
        ]
    ])
    # build scaling matrix
    scaling_matrix = np.array([[scx, 0.0, 0.0], [0.0, scy, 0.0], [0.0, 0.0, scz]])
    # compute rotation+scaling matrix
    rotsc_matrix = scaling_matrix @ rotation_matrix
    # compute center of mass from moving points (src)
    points_src_barycenter = np.mean(points_src, axis=0)
    # apply transformation to moving points (src)
    points_src_reg = ((rotsc_matrix @ (np.array(points_src) - points_src_barycenter).T).T + points_src_barycenter) + np.array([tx, ty, tz])
    # record SSE for later display
    sse_results.append(SSE(np.array(points_dest), points_src_reg))
    # return SSE
    return SSE(np.array(points_dest), points_src_reg)


def getRigidTransformFromLandmarks(points_dest, points_src, constraints='Tx_Ty_Tz_Rx_Ry_Rz', verbose=0):
    """
    Compute affine transformation to register landmarks
    :param points_src:
    :param points_dest:
    :param constraints:
    :param verbose: 0, 1, 2
    :return: rotsc_matrix, translation_array, points_src_reg, points_src_barycenter
    """
    # TODO: check input constraints

    # initialize default parameters
    init_param = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    # initialize parameters for optimizer
    init_param_optimizer = []
    # initialize dictionary to relate constraints index to dof
    dict_dof = {'Tx': 0, 'Ty': 1, 'Tz': 2, 'Rx': 3, 'Ry': 4, 'Rz': 5, 'Sx': 6, 'Sy': 7, 'Sz': 8}
    # extract constraints
    list_constraints = constraints.split('_')
    # loop across constraints and build initial_parameters
    for i in range(len(list_constraints)):
        init_param_optimizer.append(init_param[dict_dof[list_constraints[i]]])

    # launch optimizer
    # res = minimize(
    #     minimize_transform, x0=init_param_optimizer,
    #     args=(points_src, points_dest, constraints),
    #     method='Nelder-Mead', tol=1e-8,
    #     options={'xtol': 1e-8, 'ftol': 1e-8, 'maxiter': 10000, 'maxfev': 10000, 'disp': show},
    # )
    res = minimize(
        minimize_transform, x0=init_param_optimizer,
        args=(points_dest, points_src, constraints),
        method='Powell', tol=1e-8,
        options={'xtol': 1e-8, 'ftol': 1e-8, 'maxiter': 100000, 'maxfev': 100000, 'disp': verbose},
    )
    # res = minimize(
    #     minAffineTransform, x0=initial_parameters, args=points, method='COBYLA', tol=1e-8,
    #     options={'tol': 1e-8, 'rhobeg': 0.1, 'maxiter': 100000, 'catol': 0, 'disp': show},
    # )
    # loop across constraints and update dof
    dof = init_param
    for i in range(len(list_constraints)):
        dof[dict_dof[list_constraints[i]]] = res.x[i]
    # convert dof to more intuitive variables
    tx, ty, tz, alpha, beta, gamma, scx, scy, scz = dof[0], dof[1], dof[2], dof[3], dof[4], dof[5], dof[6], dof[7], dof[8]
    # convert results to intuitive variables
    # tx, ty, tz, alpha, beta, gamma, scx, scy, scz = res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], res.x[6], res.x[7], res.x[8]
    # build translation matrix
    translation_array = np.expand_dims(np.array([tx, ty, tz]), 0)
    # build rotation matrix
    rotation_matrix = np.array([
        [
            np.cos(alpha)*np.cos(beta),
            np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma),
            np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma),
        ], [
            np.sin(alpha)*np.cos(beta),
            np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma),
            np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma),
        ], [
            np.sin(beta)*-1,
            np.cos(beta)*np.sin(gamma),
            np.cos(beta)*np.cos(gamma),
        ]
    ])
    # build scaling matrix
    scaling_matrix = np.array([[scx, 0.0, 0.0], [0.0, scy, 0.0], [0.0, 0.0, scz]])
    # compute rotation+scaling matrix
    rotsc_matrix = scaling_matrix @ rotation_matrix
    # compute center of mass from moving points (src)
    points_src_barycenter = np.mean(points_src, axis=0)
    # apply transformation to moving points (src)
    points_src_reg = ((rotsc_matrix @ (np.array(points_src) - points_src_barycenter).T).T + points_src_barycenter) + translation_array

    logger.info(f"Matrix:\n {rotation_matrix}")
    logger.info(f"Center:\n {points_src_barycenter}")
    logger.info(f"Translation:\n {translation_array}")

    if verbose == 2:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        points_src_matrix = np.array(points_src)
        points_dest_matrix = np.array(points_dest)

        number_points = len(points_dest)

        ax.scatter([points_dest_matrix[i, 0] for i in range(0, number_points)],
                   [points_dest_matrix[i, 1] for i in range(0, number_points)],
                   [points_dest_matrix[i, 2] for i in range(0, number_points)], c='g', marker='+', s=500, label='dest')
        ax.scatter([points_src_matrix[i, 0] for i in range(0, number_points)],
                   [points_src_matrix[i, 1] for i in range(0, number_points)],
                   [points_src_matrix[i, 2] for i in range(0, number_points)], c='r', label='src')
        ax.scatter([points_src_reg[i, 0] for i in range(0, number_points)],
                   [points_src_reg[i, 1] for i in range(0, number_points)],
                   [points_src_reg[i, 2] for i in range(0, number_points)], c='b', label='src_reg')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect('auto')
        plt.legend()
        # plt.show()
        plt.savefig('getRigidTransformFromLandmarks_plot.png')

        plt.figure()  # fig2
        plt.plot(sse_results)
        plt.grid()
        plt.title(f'#Iterations: {res.nit}, #FuncEval: {res.nfev}, Error: {res.fun}')
        plt.show()
        plt.savefig(os.path.join('getRigidTransformFromLandmarks_iterations.png'))

    # transform numpy matrix to list structure because it is easier to handle
    points_src_reg = points_src_reg.tolist()

    return rotsc_matrix, translation_array, points_src_reg, points_src_barycenter
