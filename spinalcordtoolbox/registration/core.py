"""
Core entry points for registration

Copyright (c) 2022 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

# TODO: for -ref subject, crop data, otherwise registration is too long
# TODO: testing script for all cases
# TODO: enable vertebral alignment with -ref subject

import os

import logging

from spinalcordtoolbox.registration import algorithms

from spinalcordtoolbox.image import Image, add_suffix, generate_output_file
from spinalcordtoolbox.utils.fs import extract_fname, rmtree, tmp_create
from spinalcordtoolbox.utils.shell import printv
from spinalcordtoolbox.utils.sys import run_proc
from spinalcordtoolbox.scripts import sct_apply_transfo

logger = logging.getLogger(__name__)


def register_wrapper(fname_src, fname_dest, param, paramregmulti, fname_src_seg='', fname_dest_seg='', fname_src_label='',
                     fname_dest_label='', fname_mask='', fname_initwarp='', fname_initwarpinv='', identity=False,
                     interp='linear', fname_output='', fname_output_warp='', fname_output_warpinv='',
                     path_out='', same_space=False):
    """
    Wrapper for image registration.

    :param fname_src:
    :param fname_dest:
    :param param: Class Param(): See definition in sct_register_multimodal
    :param paramregmulti: Class ParamregMultiStep(): See definition in algorithms.py
    :param fname_src_seg:
    :param fname_dest_seg:
    :param fname_src_label:
    :param fname_dest_label:
    :param fname_mask:
    :param fname_initwarp: str: File name of initial transformation
    :param fname_initwarpinv: str: File name of initial inverse transformation
    :param identity:
    :param interp:
    :param fname_output:
    :param fname_output_warp:
    :param fname_output_warpinv:
    :param path_out:
    :param same_space: Bool: Source and destination images are in the same physical space (i.e. same coordinates).
    :return: fname_src2dest, fname_dest2src, fname_output_warp, fname_output_warpinv
    """
    # TODO: move interp inside param.
    # TODO: merge param inside paramregmulti by having a "global" sets of parameters that apply to all steps

    # Extract path, file and extension
    path_src, file_src, ext_src = extract_fname(fname_src)
    path_dest, file_dest, ext_dest = extract_fname(fname_dest)

    # check if source and destination images have the same name (related to issue #373)
    # If so, change names to avoid conflict of result files and warns the user
    suffix_src, suffix_dest = '_reg', '_reg'
    if file_src == file_dest:
        suffix_src, suffix_dest = '_src_reg', '_dest_reg'

    # define output folder and file name
    if fname_output == '':
        path_out = '' if not path_out else path_out  # output in user's current directory
        file_out = file_src + suffix_src
        file_out_inv = file_dest + suffix_dest
        ext_out = ext_src
    else:
        path, file_out, ext_out = extract_fname(fname_output)
        path_out = path if not path_out else path_out
        file_out_inv = f'{file_out}_inv'

    # create temporary folder
    path_tmp = tmp_create(basename="register-wrapper")

    printv("\nCopying input data to tmp folder and convert to nii...", param.verbose)
    Image(fname_src).save(os.path.join(path_tmp, "src.nii"))
    Image(fname_dest).save(os.path.join(path_tmp, "dest.nii"))

    if fname_src_seg:
        Image(fname_src_seg).save(os.path.join(path_tmp, "src_seg.nii"))

    if fname_dest_seg:
        Image(fname_dest_seg).save(os.path.join(path_tmp, "dest_seg.nii"))

    if fname_src_label:
        Image(fname_src_label).save(os.path.join(path_tmp, "src_label.nii"))
        Image(fname_dest_label).save(os.path.join(path_tmp, "dest_label.nii"))

    if fname_mask != '':
        Image(fname_mask).save(os.path.join(path_tmp, "mask.nii.gz"))

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # reorient destination to RPI
    Image('dest.nii').change_orientation("RPI").save('dest_RPI.nii')
    if fname_dest_seg:
        Image('dest_seg.nii').change_orientation("RPI").save('dest_seg_RPI.nii')
    if fname_dest_label:
        Image('dest_label.nii').change_orientation("RPI").save('dest_label_RPI.nii')
    if fname_mask:
        # TODO: change output name
        Image('mask.nii.gz').change_orientation("RPI").save('mask.nii.gz')

    if identity:
        # overwrite paramregmulti and only do one identity transformation
        step0 = algorithms.Paramreg(step='0', type='im', algo='syn', metric='MI', iter='0', shrink='1', smooth='0', gradStep='0.5')
        paramregmulti = algorithms.ParamregMultiStep([step0])

    # initialize list of warping fields
    warp_forward = []
    warp_forward_winv = []
    warp_inverse = []
    warp_inverse_winv = []
    generate_warpinv = 1

    # initial warping is specified, update list of warping fields and skip step=0
    if fname_initwarp:
        printv("\nSkip step=0 and replace with initial transformations:", param.verbose)
        printv(f"  {fname_initwarp}", param.verbose)
        # copy(fname_initwarp, 'warp_forward_0.nii.gz')
        warp_forward.append(fname_initwarp)
        start_step = 1
        if fname_initwarpinv:
            warp_inverse.append(fname_initwarpinv)
        else:
            printv("\nWARNING: No initial inverse warping field was specified, therefore the registration will be "
                   "src->dest only, and the inverse warping field will NOT be generated.", param.verbose, 'warning')
            generate_warpinv = 0
    else:
        if same_space:
            start_step = 1
        else:
            start_step = 0

    # loop across registration steps
    for i_step in range(start_step, len(paramregmulti.steps)):
        step = paramregmulti.steps[str(i_step)]
        printv(f"\n--\nESTIMATE TRANSFORMATION FOR STEP #{i_step}", param.verbose)
        # identify which is the src and dest
        if step.type == 'im':
            src = ['src.nii']
            dest = ['dest_RPI.nii']
            interp_step = ['spline']
        elif step.type == 'seg':
            src = ['src_seg.nii']
            dest = ['dest_seg_RPI.nii']
            interp_step = ['nn']
        elif step.type == 'imseg':
            src = ['src.nii', 'src_seg.nii']
            dest = ['dest_RPI.nii', 'dest_seg_RPI.nii']
            interp_step = ['spline', 'nn']
        elif step.type == 'label':
            src = ['src_label.nii']
            dest = ['dest_label_RPI.nii']
            interp_step = ['nn']
        elif step.type == 'rootlet':
            src = ['src.nii', 'src_label.nii']
            dest = ['dest_RPI.nii', 'dest_label_RPI.nii']
            interp_step = ['spline']  # Maybe do or linear?
        else:
            printv("ERROR: Wrong image type: {step.type}", 1, 'error')

        # if step>0, apply warp_forward_concat to the src image to be used
        if (not same_space and i_step > 0) or (same_space and i_step > 1):
            printv("\nApply transformation from previous step", param.verbose)
            for ifile in range(len(src)):
                sct_apply_transfo.main(argv=[
                    '-i', src[ifile],
                    '-d', dest[ifile],
                    '-o', add_suffix(src[ifile], '_reg'),
                    '-x', interp_step[ifile],
                    '-crop', '3' if (i_step == 1 and fname_initwarp) else '0',
                    '-v', '0',
                    '-w'] + warp_forward
                )
                src[ifile] = add_suffix(src[ifile], '_reg')

        # register src --> dest
        warp_forward_out, warp_inverse_out = register(src=src, dest=dest, step=step, param=param)

        # deal with transformations with "-" as prefix. They should be inverted with calling isct_ComposeMultiTransform.
        if warp_forward_out[0] == "-":
            warp_forward_out = warp_forward_out[1:]
            warp_forward_winv.append(warp_forward_out)
        if warp_inverse_out[0] == "-":
            warp_inverse_out = warp_inverse_out[1:]
            warp_inverse_winv.append(warp_inverse_out)

        # update list of forward/inverse transformations
        warp_forward.append(warp_forward_out)
        warp_inverse.insert(0, warp_inverse_out)

    # Concatenate transformations
    printv("\nConcatenate transformations...", param.verbose)

    # if a warping field needs to be inverted, remove it from warp_forward
    warp_forward = [f for f in warp_forward if f not in warp_forward_winv]
    dimensionality = len(Image("dest.nii").hdr.get_data_shape())
    cmd = ['isct_ComposeMultiTransform', f"{dimensionality}", 'warp_src2dest.nii.gz', '-R', 'dest.nii']

    if warp_forward_winv:
        cmd.append('-i')
        cmd.extend(reversed(warp_forward_winv))
    if warp_forward:
        cmd.extend(reversed(warp_forward))

    status, output = run_proc(cmd, is_sct_binary=True)
    if status != 0:
        raise RuntimeError(f"Subprocess call {cmd} returned non-zero: {output}")

    # if an inverse warping field needs to be inverted, remove it from warp_inverse_winv
    warp_inverse = [f for f in warp_inverse if f not in warp_inverse_winv]
    cmd = ['isct_ComposeMultiTransform', f"{dimensionality}", 'warp_dest2src.nii.gz', '-R', 'src.nii']
    dimensionality = len(Image("dest.nii").hdr.get_data_shape())

    if warp_inverse_winv:
        cmd.append('-i')
        cmd.extend(reversed(warp_inverse_winv))
    if warp_inverse:
        cmd.extend(reversed(warp_inverse))

    status, output = run_proc(cmd, is_sct_binary=True)
    if status != 0:
        raise RuntimeError(f"Subprocess call {cmd} returned non-zero: {output}")

    # TODO: make the following code optional (or move it to sct_register_multimodal)
    # Apply warping field to src data
    printv("\nApply transfo source --> dest...", param.verbose)
    sct_apply_transfo.main(argv=[
        '-i', 'src.nii',
        '-d', 'dest.nii',
        '-w', 'warp_src2dest.nii.gz',
        '-o', 'src_reg.nii',
        '-x', interp,
        '-v', '0',
    ])

    if generate_warpinv:
        printv("\nApply transfo dest --> source...", param.verbose)
        sct_apply_transfo.main(argv=[
            '-i', 'dest.nii',
            '-d', 'src.nii',
            '-w', 'warp_dest2src.nii.gz',
            '-o', 'dest_reg.nii',
            '-x', interp,
            '-v', '0',
        ])

    # come back
    os.chdir(curdir)

    # Generate output files
    # ------------------------------------------------------------------------------------------------------------------

    printv("\nGenerate output files...", param.verbose)

    # generate src -> dest output files
    fname_src2dest = os.path.join(path_out, file_out + ext_out)
    generate_output_file(os.path.join(path_tmp, "src_reg.nii"), fname_src2dest, param.verbose)
    if fname_output_warp == '':
        fname_output_warp = os.path.join(path_out, f'warp_{file_src}2{file_dest}.nii.gz')
    generate_output_file(os.path.join(path_tmp, "warp_src2dest.nii.gz"), fname_output_warp, param.verbose)

    # generate dest -> src output files
    if generate_warpinv:
        fname_dest2src = os.path.join(path_out, file_out_inv + ext_dest)
        generate_output_file(os.path.join(path_tmp, "dest_reg.nii"), fname_dest2src, param.verbose)
        if fname_output_warpinv == '':
            fname_output_warpinv = os.path.join(path_out, f'warp_{file_dest}2{file_src}.nii.gz')
        generate_output_file(os.path.join(path_tmp, "warp_dest2src.nii.gz"), fname_output_warpinv, param.verbose)
    else:
        # we skip generating files if there is no inverse warping field (i.e. we're doing a one-way registration)
        fname_dest2src = None
        fname_output_warpinv = None

    # Delete temporary files
    if param.remove_temp_files:
        printv("\nRemove temporary files...", param.verbose)
        rmtree(path_tmp, verbose=param.verbose)

    return fname_src2dest, fname_dest2src, fname_output_warp, fname_output_warpinv


# register images
# ==========================================================================================
def register(src, dest, step, param):
    """
    Register src onto dest image. Output affine transformations that need to be inverted will have the prefix "-".
    """
    # initiate default parameters of antsRegistration transformation
    ants_registration_params = {'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '',
                                'bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}

    output = ''  # default output if problem

    # If the input type is either im or seg, we can convert the input list into a string for improved code clarity
    if not step.type == 'imseg' and not step.type == 'rootlet':
        src = src[0]
        dest = dest[0]

    # display arguments
    printv("Registration parameters:", param.verbose)
    printv(f"  type ........... {step.type}", param.verbose)
    printv(f"  algo ........... {step.algo}", param.verbose)
    printv(f"  slicewise ...... {step.slicewise}", param.verbose)
    printv(f"  metric ......... {step.metric}", param.verbose)
    printv(f"  samplStrategy .. {step.samplingStrategy}", param.verbose)
    printv(f"  samplPercent ... {step.samplingPercentage}", param.verbose)
    printv(f"  iter ........... {step.iter}", param.verbose)
    printv(f"  smooth ......... {step.smooth}", param.verbose)
    printv(f"  laplacian ...... {step.laplacian}", param.verbose)
    printv(f"  shrink ......... {step.shrink}", param.verbose)
    printv(f"  gradStep ....... {step.gradStep}", param.verbose)
    printv(f"  deformation .... {step.deformation}", param.verbose)
    printv(f"  init ........... {step.init}", param.verbose)
    printv(f"  poly ........... {step.poly}", param.verbose)
    printv(f"  filter_size .... {step.filter_size}", param.verbose)
    printv(f"  dof ............ {step.dof}", param.verbose)
    printv(f"  smoothWarpXY ... {step.smoothWarpXY}", param.verbose)
    printv(f"  rot_method ..... {step.rot_method}", param.verbose)

    # validate `-param` here to fail early if invalid (but after displaying them to the user)
    if step.algo == 'centermassrot':
        valid_types = {'pca': 'seg', 'hog': 'imseg', 'pcahog': 'imseg'}
        if step.type != valid_types[step.rot_method]:
            raise ValueError(f"Parameter 'rot_method={step.rot_method}' is only valid for type={valid_types[step.rot_method]}', "
                             f"not '{step.type}'. Please update your `-param` settings to use a valid combination.")

    # set metricSize
    if step.metric == 'MI':
        metricSize = '32'  # corresponds to number of bins
    else:
        metricSize = '4'  # corresponds to radius (for CC, MeanSquares...)

    # set masking
    if param.fname_mask:
        fname_mask = 'mask.nii.gz'
        masking = ['-x', 'mask.nii.gz']
    else:
        fname_mask = ''
        masking = []

    # # landmark-based registration
    if step.type in ['label']:
        if step.algo:
            printv(f"Parameter 'algo={step.algo}' has no effect for 'type=label' registration.", type='warning')
        warp_forward_out, warp_inverse_out = algorithms.register_step_label(
            src=src,
            dest=dest,
            step=step,
            verbose=param.verbose,
        )

    elif step.algo == 'slicereg':
        warp_forward_out, warp_inverse_out, _ = algorithms.register_step_ants_slice_regularized_registration(
            src=src,
            dest=dest,
            step=step,
            metricSize=metricSize,
            fname_mask=fname_mask,
            verbose=param.verbose,
        )

    elif step.type in ['rootlet']:
        warp_forward_out, warp_inverse_out = algorithms.register_rootlet(
            src=src,
            dest=dest,
            step=step,
            ants_registration_params=ants_registration_params,
            metricSize=metricSize,
            padding=param.padding,
            verbose=param.verbose,
        )

    # ANTS 3d
    elif step.algo.lower() in ants_registration_params and step.slicewise == '0':  # FIXME [AJ]
        warp_forward_out, warp_inverse_out = algorithms.register_step_ants_registration(
            src=src,
            dest=dest,
            step=step,
            masking=masking,
            ants_registration_params=ants_registration_params,
            padding=param.padding,
            metricSize=metricSize,
            verbose=param.verbose,
        )

    # ANTS 2d
    elif step.algo.lower() in ants_registration_params and step.slicewise == '1':  # FIXME [AJ]
        warp_forward_out, warp_inverse_out = algorithms.register_step_slicewise_ants(
            src=src,
            dest=dest,
            step=step,
            ants_registration_params=ants_registration_params,
            fname_mask=fname_mask,
            remove_temp_files=param.remove_temp_files,
            verbose=param.verbose,
        )

    # slice-wise transfo
    elif step.algo in ['centermass', 'centermassrot', 'columnwise']:
        # check if user provided a mask-- if so, inform it will be ignored
        if fname_mask:
            printv(f"\nWARNING: algo {step.algo} will ignore the provided mask.\n", 1, 'warning')

        warp_forward_out, warp_inverse_out = algorithms.register_step_slicewise(
            src=src,
            dest=dest,
            step=step,
            ants_registration_params=ants_registration_params,
            remove_temp_files=param.remove_temp_files,
            verbose=param.verbose,
        )

    # deep learning registration using cascaded models
    elif step.algo == 'dl':
        # check if user provided a mask-- if so, inform it will be ignored
        if fname_mask:
            printv(f"\nWARNING: algo {step.algo} will ignore the provided mask.\n", 1, 'warning')

        warp_forward_out, warp_inverse_out = algorithms.register_step_dl_multimodal_cascaded_reg(
            src=src,
            dest=dest,
            step=step,
            verbose=param.verbose,
        )

    else:
        printv(f"\nERROR: algo {step.algo} does not exist. Exit program\n", 1, 'error')

    if not os.path.isfile(warp_forward_out):
        # no forward warping field for rigid and affine
        printv(f"\nERROR: file {warp_forward_out} doesn't exist (or is not a file).\n{output}\n"
               "ERROR: ANTs failed. Exit program.\n", 1, 'error')
    elif not os.path.isfile(warp_inverse_out) and \
            step.algo not in ['rigid', 'affine', 'translation'] and \
            step.type not in ['label']:
        # no inverse warping field for rigid and affine
        printv(f"\nERROR: file {warp_inverse_out} doesn't exist (or is not a file).\n{output}\n"
               "ERROR: ANTs failed. Exit program.\n", 1, 'error')
    else:
        # rename warping fields
        _, _, output_ext = extract_fname(warp_forward_out)
        if output_ext in ['.txt', '.mat']:
            warp_forward = f'warp_forward_{step.step}{output_ext}'
            os.rename(warp_forward_out, warp_forward)
            warp_inverse = f'-warp_forward_{step.step}{output_ext}'
        else:
            warp_forward = f'warp_forward_{step.step}.nii.gz'
            warp_inverse = f'warp_inverse_{step.step}.nii.gz'
            os.rename(warp_forward_out, warp_forward)
            os.rename(warp_inverse_out, warp_inverse)

    return warp_forward, warp_inverse
