# pytest unit tests for isct_antsRegistration

import os

import numpy as np
import nibabel as nib

from spinalcordtoolbox.utils.sys import run_proc, printv
from spinalcordtoolbox.utils.fs import tmp_create, rmtree

from spinalcordtoolbox.scripts import sct_dice_coefficient


def test_ants_registration():
    """
    This function test the integrity of ANTs output, given that some versions of ANTs give a wrong BSpline transform,
    notably when using sct_ANTSUseLandmarkImagesToGetBSplineDisplacementField.
    """
    # Initialization
    size_data = 61
    size_label = 1  # put zero for labels that are single points.
    dice_acceptable = 0.39  # computed DICE should be 0.931034
    verbose = 1

    path_tmp = tmp_create(basename="test-ants")

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Initialise numpy volumes
    data_src = np.zeros((size_data, size_data, size_data), dtype=np.int16)
    data_dest = np.zeros((size_data, size_data, size_data), dtype=np.int16)

    # add labels for src image (curved).
    # Labels can be big (more than single point), because when applying NN interpolation, single points might disappear
    data_src[20 - size_label:20 + size_label + 1, 20 - size_label:20 + size_label + 1, 10 - size_label:10 + size_label + 1] = 1
    data_src[30 - size_label:30 + size_label + 1, 30 - size_label:30 + size_label + 1, 30 - size_label:30 + size_label + 1] = 2
    data_src[20 - size_label:20 + size_label + 1, 20 - size_label:20 + size_label + 1, 50 - size_label:50 + size_label + 1] = 3

    # add labels for dest image (straight).
    # Here, no need for big labels (bigger than single point) because these labels will not be re-interpolated.
    data_dest[30 - size_label:30 + size_label + 1, 30 - size_label:30 + size_label + 1, 10 - size_label:10 + size_label + 1] = 1
    data_dest[30 - size_label:30 + size_label + 1, 30 - size_label:30 + size_label + 1, 30 - size_label:30 + size_label + 1] = 2
    data_dest[30 - size_label:30 + size_label + 1, 30 - size_label:30 + size_label + 1, 50 - size_label:50 + size_label + 1] = 3

    # save as nifti
    img_src = nib.Nifti1Image(data_src, np.eye(4))
    nib.save(img_src, 'data_src.nii.gz')
    img_dest = nib.Nifti1Image(data_dest, np.eye(4))
    nib.save(img_dest, 'data_dest.nii.gz')

    # Estimate rigid transformation
    printv('\nEstimate rigid transformation between paired landmarks...', verbose)
    # TODO fixup isct_ants* parsers
    run_proc(['isct_antsRegistration',
              '-d', '3',
              '-t', 'syn[1,3,1]',
              '-m', 'MeanSquares[data_dest.nii.gz,data_src.nii.gz,1,3]',
              '-f', '2',
              '-s', '0',
              '-o', '[src2reg,data_src_reg.nii.gz]',
              '-c', '5',
              '-v', '1',
              '-n', 'NearestNeighbor'], verbose, is_sct_binary=True)

    # # Apply rigid transformation
    # printv('\nApply rigid transformation to curved landmarks...', verbose)
    # sct_apply_transfo.main(["-i", "data_src.nii.gz", "-o", "data_src_rigid.nii.gz", "-d", "data_dest.nii.gz", "-w", "curve2straight_rigid.txt", "-p", "nn", "-v", "0"])
    #
    # # Estimate b-spline transformation curve --> straight
    # printv('\nEstimate b-spline transformation: curve --> straight...', verbose)
    # run_proc('isct_ANTSLandmarksBSplineTransform data_dest.nii.gz data_src_rigid.nii.gz warp_curve2straight_intermediate.nii.gz 5x5x5 3 2 0', verbose)
    #
    # # Concatenate rigid and non-linear transformations...
    # printv('\nConcatenate rigid and non-linear transformations...', verbose)
    # cmd = 'isct_ComposeMultiTransform 3 warp_curve2straight.nii.gz -R data_dest.nii.gz warp_curve2straight_intermediate.nii.gz curve2straight_rigid.txt'
    # printv(f'>> {cmd}', verbose)
    # run_proc(cmd)
    #
    # # Apply deformation to input image
    # printv('\nApply transformation to input image...', verbose)
    # sct_apply_transfo.main(["-i", "data_src.nii.gz", "-o", "data_src_warp.nii.gz", "-d", "data_dest.nii.gz", "-w", "warp_curve2straight.nii.gz", "-p", "nn", "-v", "0"])
    #
    # Compute DICE coefficient between src and dest
    printv('\nCompute DICE coefficient...', verbose)
    sct_dice_coefficient.main([
        "-i", "data_dest.nii.gz",
        "-d", "data_src_reg.nii.gz",
        "-o", "dice.txt",
        "-v", "0",
    ])
    with open("dice.txt", "r") as file_dice:
        dice = float(file_dice.read().replace('3D Dice coefficient = ', ''))
    printv(f'Dice coeff = {dice} (should be above {dice_acceptable})', verbose)

    # come back
    os.chdir(curdir)

    printv('\nDelete temporary files...', verbose)
    rmtree(path_tmp)

    assert dice > dice_acceptable
