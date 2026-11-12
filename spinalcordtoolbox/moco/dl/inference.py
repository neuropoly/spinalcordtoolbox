#!/usr/bin/env python
#
# Inference of DL-based motion correction for dMRI/fMRI
#
# Copyright (c) 2025 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import os
import time
import textwrap
import numpy as np
from shutil import copyfile

from spinalcordtoolbox.image import add_suffix, generate_output_file, Image, reorient_coordinates
from spinalcordtoolbox.scripts import sct_dmri_separate_b0_and_dwi
from spinalcordtoolbox.utils.sys import sct_dir_local_path, LazyLoader, printv
from spinalcordtoolbox.utils.fs import tmp_create, extract_fname, rmtree

torch = LazyLoader("torch", globals(), "torch")
moco_dl_model = LazyLoader("moco_dl_model", globals(), "spinalcordtoolbox.moco.dl.model")


def check_dl_args(argv):
    # mask (-m) is mandatory for DL module
    if "-m" not in argv:
        raise ValueError("\n[moco-dl] Missing required argument: -m <mask>. A spinal cord mask is required for DL-based motion correction.")

    # check if the raw user arguments contain any forbidden args
    forbidden = []
    for forbidden_arg in ["-g", "-x", "-param", "-bvalmin"]:
        if any(arg == forbidden_arg for arg in argv):
            forbidden.append(forbidden_arg)

    if forbidden:
        raise ValueError(
            f"The following options cannot be used together with -dl (DL-based motion correction): {', '.join(forbidden)}\n"
            "DL module does not support b-value threshold (-bvalmin), grouping (-g), final interpolation (-x), "
            "or advanced ANTs parameters (-param)."
        )


def moco_dl(fname_data, fname_mask='', fname_ref='', path_out='', mode="fmri", fname_bvals='', fname_bvecs=''):
    """
        Deep-learning motion correction (DenseRigidNet) for dMRI/fMRI.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(path_out, exist_ok=True)

    ckpt_path = sct_dir_local_path("data", "moco-dl_models", f"{mode}.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("\n[moco-dl] MocoDL models not found. Please install them using: sct_download_data -d moco-dl_models.")

    printv("\n[INFO] Running DL-based motion correction (DenseRigidNet)...\n")
    # Display parameter summary
    printv(textwrap.dedent(f"""
        Input parameters (DL-based motion correction):
        ----------------------------------------------------
        Input file:            {fname_data}
        Output folder:         {os.path.abspath(path_out)}
        Mode:                  {mode}
        Mask:                  {fname_mask if fname_mask != '' else 'None'}
        Reference image:       {fname_ref if fname_ref != '' else 'first volume of input (t=0)'}
        bvecs (dmri only):     {fname_bvecs}
        bvals (dmri only):     {fname_bvals}
    """))

    # Create tmp folder
    path_tmp = tmp_create(basename="moco-dl")

    # Load native inputs
    im_data_native = Image(fname_data)
    im_mask_native = Image(fname_mask) if fname_mask != '' else None
    if fname_ref:
        im_ref_native = Image(fname_ref)
    else:
        ref_data = im_data_native.data[..., 0]
        im_ref_native = Image(param=ref_data, hdr=im_data_native.hdr)
    orig_orient = im_data_native.orientation
    printv(f"\n[moco-dl] Native input orientation: {orig_orient}")

    # Validate orientation before reorienting to RPI
    if orig_orient[2] in 'IS':
        printv("[moco-dl] Input is treated as axial (last axis = S/I).")
    else:
        raise ValueError(
            f"Unsupported orientation: {orig_orient}. "
            f"DL-based moco currently supports only inputs where the last axis is S/I."
        )

    # Reorient inputs to RPI for inference
    printv("\n[moco-dl] Reorienting inputs to RPI for inference...")
    im_data_rpi = im_data_native.copy()
    if im_data_rpi.orientation != "RPI":
        im_data_rpi.change_orientation("RPI")
    im_mask_rpi = im_mask_native.copy()
    if im_mask_rpi.orientation != "RPI":
        im_mask_rpi.change_orientation("RPI")
    im_ref_rpi = im_ref_native.copy()
    if im_ref_rpi.orientation != "RPI":
        im_ref_rpi.change_orientation("RPI")

    # Copying input data to tmp folder
    printv("\nCopying input data to tmp folder...")
    fname_data_rpi = os.path.join(path_tmp, "data_rpi.nii.gz")
    im_data_rpi.save(fname_data_rpi)
    if im_mask_rpi is not None:
        fname_mask_rpi = os.path.join(path_tmp, "mask_rpi.nii.gz")
        im_mask_rpi.save(fname_mask_rpi)
    if im_ref_rpi is not None:
        fname_ref_rpi = os.path.join(path_tmp, "ref_rpi.nii.gz")
        im_ref_rpi.save(fname_ref_rpi)
    if fname_bvals != '':
        _, _, ext_bvals = extract_fname(fname_bvals)
        file_bvals = f"bvals.{ext_bvals}"
        copyfile(fname_bvals, os.path.join(path_tmp, file_bvals))
        fname_bvals = file_bvals
    if fname_bvecs != '':
        _, _, ext_bvecs = extract_fname(fname_bvecs)
        file_bvecs = f"bvecs.{ext_bvecs}"
        copyfile(fname_bvecs, os.path.join(path_tmp, file_bvecs))
        fname_bvecs = file_bvecs

    mov_img = im_data_rpi.data.astype(np.float32)
    mask_img = im_mask_rpi.data.astype(np.float32)
    ref_img = im_ref_rpi.data.astype(np.float32)
    affine_rpi = im_data_rpi.affine
    header_rpi = im_data_rpi.header

    # Build absolute output path and go to tmp folder
    curdir = os.getcwd()
    path_out_abs = os.path.abspath(path_out)
    os.chdir(path_tmp)

    # Load DL model
    printv(f"\n[moco-dl] Loading checkpoint: {ckpt_path}")
    model = moco_dl_model.DenseRigidReg()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model = model.to(device)
    model.eval()
    model.warp = moco_dl_model.RigidWarp(mode="bilinear")

    moving = torch.from_numpy(mov_img).unsqueeze(0).unsqueeze(0).to(device)
    fixed = torch.from_numpy(ref_img).unsqueeze(0).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).to(device)

    # Run inference
    start_time = time.time()
    printv("\n[moco-dl] Starting inference...")
    with torch.no_grad():
        warped, flow, Tx, Ty = model(moving, fixed, mask)

    inference_time = time.time() - start_time
    printv(f"\nInference time: {inference_time:.2f} sec")

    warped = warped.squeeze().cpu().numpy()  # (H,W,D,T)
    flow = flow.squeeze().cpu().numpy()  # (3,H,W,D,T) -- not exported
    Tx = Tx.squeeze().cpu().numpy()  # (D,T)
    Ty = Ty.squeeze().cpu().numpy()  # (D,T)

    # Convert voxel shifts predicted by the model to physical displacement (mm)
    # using voxel spacing derived from the input image affine.
    A = affine_rpi[:3, :3].astype(np.float32)
    # voxel spacing in mm
    sx = float(np.linalg.norm(A[:, 0]))
    sy = float(np.linalg.norm(A[:, 1]))
    printv(f"\n[moco-dl] Scaling by voxel size from affine: sx={sx:.3f}mm, sy={sy:.3f}mm")
    # convert voxel shifts -> mm
    Tx_mm_rpi = Tx.astype(np.float32) * sx
    Ty_mm_rpi = Ty.astype(np.float32) * sy * -1.0
    # Ty is flipped when converting from array coordinates to physical coordinates.

    # Save Moco output
    im_moco_rpi = Image(warped, hdr=header_rpi)
    im_moco_rpi.affine = affine_rpi
    im_moco_rpi.hdr.set_data_shape(im_moco_rpi.data.shape)
    fname_moco_rpi_tmp = os.path.join(path_tmp, "mocoDL_rpi.nii.gz")
    im_moco_rpi.save(fname_moco_rpi_tmp)

    # Reorient corrected image back to native orientation
    if orig_orient != "RPI":
        printv(f"\n[moco-dl] Reorienting corrected image from RPI back to native orientation: {orig_orient}")
        im_moco_native = im_moco_rpi.change_orientation(orig_orient)
    else:
        im_moco_native = im_moco_rpi.copy()
    fname_moco_tmp = os.path.join(path_tmp, "mocoDL.nii.gz")
    im_moco_native.save(fname_moco_tmp)

    # Save mean output
    printv("\n[moco-dl] Computing time-averaged output volume...")
    if mode == "dmri":
        args = ['-i', fname_moco_tmp, '-bvec', fname_bvecs, '-a', '1', '-v', '0']
        if fname_bvals:
            args.extend(['-bval', fname_bvals])
        _, fname_b0_mean, _, fname_dwi_mean = sct_dmri_separate_b0_and_dwi.main(argv=args)
    else:
        fname_moco_mean = add_suffix(fname_moco_tmp, '_mean')
        im_moco_native.mean(dim=3).save(fname_moco_mean)

    # Reorient translation params back to native orientation
    printv(f"\n[moco-dl] Reorienting translation params back to native orientation: {orig_orient}")
    if orig_orient != "RPI":
        D, T = Tx_mm_rpi.shape
        Tx_mm_native = np.zeros((D, T), dtype=np.float32)
        Ty_mm_native = np.zeros((D, T), dtype=np.float32)
        for d in range(D):
            for t in range(T):
                # Treat slice-wise motion as a displacement vector in RPI space
                vec_rpi = [[float(Tx_mm_rpi[d, t]), float(Ty_mm_rpi[d, t]), 0.0]]
                # Reorient vector from RPI -> native using relative mode
                vec_native = reorient_coordinates(vec_rpi, im_data_rpi, orig_orient, mode="relative")[0]
                Tx_mm_native[d, t] = vec_native[0]
                Ty_mm_native[d, t] = vec_native[1]
    else:
        Tx_mm_native = Tx_mm_rpi.astype(np.float32)
        Ty_mm_native = Ty_mm_rpi.astype(np.float32)

    # Save Tx and Ty in world-coordinates (mm unit)
    printv("\n[moco-dl] Saving translation params...")
    im_Tx = Image(Tx_mm_native[np.newaxis, np.newaxis, ...], hdr=im_data_native.header)
    im_Tx.affine = im_data_native.affine
    im_Tx.hdr.set_data_shape(im_Tx.data.shape)  # change header with new shape (1, 1, D, T)
    tx_tmp = os.path.join(path_tmp, "Tx_mm.nii.gz")
    im_Tx.save(tx_tmp)
    im_Ty = Image(Ty_mm_native[np.newaxis, np.newaxis, ...], hdr=im_data_native.header)
    im_Ty.affine = im_data_native.affine
    im_Ty.hdr.set_data_shape(im_Ty.data.shape)  # change header with new shape (1, 1, D, T)
    ty_tmp = os.path.join(path_tmp, "Ty_mm.nii.gz")
    im_Ty.save(ty_tmp)

    # Generate output files
    printv('\nGenerate output files...')
    # motion corrected data
    fname_moco = os.path.join(path_out_abs, add_suffix(os.path.basename(fname_data), "_mocoDL"))
    generate_output_file(fname_moco_tmp, fname_moco)
    # mean volume
    if mode == "dmri":
        generate_output_file(fname_b0_mean, add_suffix(fname_moco, "_b0_mean"))
        generate_output_file(fname_dwi_mean, add_suffix(fname_moco, "_dwi_mean"))
    else:
        generate_output_file(fname_moco_mean, add_suffix(fname_moco, "_mean"))
    # rigid translation parameter (Tx, Ty)
    generate_output_file(tx_tmp, os.path.join(path_out_abs, "moco_params_x.nii.gz"))
    generate_output_file(ty_tmp, os.path.join(path_out_abs, "moco_params_y.nii.gz"))

    # Delete temporary files
    printv('\nDelete temporary files...')
    rmtree(path_tmp)

    # come back to working directory
    os.chdir(curdir)

    printv(f"\n[moco-dl] Outputs saved in: {os.path.abspath(path_out)}")
    return fname_moco
