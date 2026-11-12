"""
Command-line interface for sc_crop.

Three modes, selected automatically from the arguments:

  Detect (default — no coordinates):
    sc_crop -i t2.nii.gz [-o t2_cropbox.nii.gz]
    sc_crop -i t2.nii.gz --detect [-o t2_cropbox.nii.gz]

  Crop (coordinates provided, or --crop flag):
    sc_crop -i t2.nii.gz -xmin 12 -xmax 54 -ymin 72 -ymax 164 -zmin 0 -zmax 311
    sc_crop -i t2.nii.gz --crop -xmin 12 -xmax 54 -ymin 72 -ymax 164 -zmin 0 -zmax 311 [-o out.nii.gz]

  Detect + crop (default — no flag needed):
    sc_crop -i t2.nii.gz [-o out.nii.gz]

  Download models:
    sc_crop download

  Write crop_metadata.json for a model release (e.g. for SCT's sct_deepseg):
    sc_crop write-metadata --pad-superior 40 --pad-inferior 100 --pad-left 15 --pad-right 15 --pad-anterior 15 --pad-posterior 22
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from .crop import detect, crop, _read_bbox_txt, _write_bbox_txt, _stem, _warn_overwrite, BBox3D
from .download import download
from .metadata import _write_metadata_cli
from .qc import save_bbox_nifti

GREEN, RESET = "\033[32m", "\033[0m"


def _print_fsleyes(input_path, crop_path):
    print("\nDone! To view results, type:")
    print(f"  {GREEN}fsleyes {input_path} {crop_path} &{RESET}")


def _crop_from_coords(input_path, xmin, xmax, ymin, ymax, zmin, zmax,
                      output, translate):
    """Crop a NIfTI using explicit voxel coordinates."""
    parent, stem = _stem(input_path)
    bbox = {
        "xmin": xmin, "xmax": xmax,
        "ymin": ymin, "ymax": ymax,
        "zmin": zmin, "zmax": zmax,
    }
    cropped = crop(nib.load(input_path), bbox, translate=translate)
    crop_path = Path(output) if output else parent / f"{stem}_crop.nii.gz"
    _warn_overwrite(crop_path)
    nib.save(cropped, crop_path)
    print(f"Crop    : {crop_path}  shape={cropped.shape}")
    _print_fsleyes(input_path, crop_path)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "write-metadata":
        _write_metadata_cli(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        description="Spinal cord detection and cropping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes:
  detect + crop (default)  sc_crop -i t2.nii.gz
  detect only              sc_crop -i t2.nii.gz --detect
  crop from bbox mask      sc_crop -i t2.nii.gz --bbox t2_cropbox.nii.gz
  crop from coords         sc_crop -i t2.nii.gz -xmin 12 -xmax 54 -ymin 72 -ymax 164 -zmin 0 -zmax 311

examples:
  sc_crop -i t2.nii.gz                                            # detect + crop → t2_crop.nii.gz + t2_cropbox.nii.gz
  sc_crop -i t2.nii.gz --detect                                  # detect only → t2_cropbox.nii.gz + t2_bbox.txt
  sc_crop -i t2.nii.gz --bbox t2_cropbox.nii.gz                  # crop from bbox NIfTI mask
  sc_crop -i label.nii.gz --bbox t2_cropbox.nii.gz               # crop label with same bbox
  sc_crop -i t2.nii.gz -xmin 12 -xmax 54 -ymin 72 -ymax 164 -zmin 0 -zmax 311
  sc_crop -i t2.nii.gz --pad-sup 50 --pad-inf 80
""",
    )

    # ── Input / output ────────────────────────────────────────────────────────
    parser.add_argument("input", nargs="?",
                        help="Input NIfTI volume (.nii or .nii.gz)")
    parser.add_argument("-i", dest="input_flag", default=None,
                        help="Input NIfTI volume — SCT-style alias for positional input")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path for the cropped volume (default: input with '_crop' suffix)")

    # ── Mode flags ────────────────────────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--detect",      action="store_true",
                      help="Detect only — write cropbox NIfTI mask and bbox.txt, do not crop")
    mode.add_argument("--crop",        action="store_true",
                      help="Crop using --bbox file or explicit -xmin/-xmax/… coordinates (no detection)")

    # ── Crop coordinates ──────────────────────────────────────────────────────
    coords = parser.add_argument_group("crop coordinates (voxel indices, inclusive)")
    coords.add_argument("--bbox", default=None, metavar="FILE",
                        help="Cropbox NIfTI mask (.nii/.nii.gz) or legacy bbox .txt file — alternative to passing coordinates")
    coords.add_argument("-xmin", type=int, default=None, metavar="N", help="x min (inclusive)")
    coords.add_argument("-xmax", type=int, default=None, metavar="N", help="x max (inclusive)")
    coords.add_argument("-ymin", type=int, default=None, metavar="N", help="y min (inclusive)")
    coords.add_argument("-ymax", type=int, default=None, metavar="N", help="y max (inclusive)")
    coords.add_argument("-zmin", type=int, default=None, metavar="N", help="z min (inclusive)")
    coords.add_argument("-zmax", type=int, default=None, metavar="N", help="z max (inclusive)")

    # ── Crop options ──────────────────────────────────────────────────────────
    parser.add_argument("--las", action="store_true",
                        help="Output cropped volume in LAS orientation (detect-crop only)")
    parser.add_argument("--no-translate", dest="translate", action="store_false",
                        help="Do not update affine origin after cropping")
    parser.set_defaults(translate=True)

    # ── Detection options ─────────────────────────────────────────────────────
    parser.add_argument("--model", default=None,
                        help="Path to a local detector ONNX file (override auto-downloaded model)")
    pad = parser.add_argument_group("padding (mm, detect / detect-crop) — individual > symmetric > default")
    pad.add_argument("--pad-sup",  type=float, default=None, dest="pad_superior",  metavar="MM")
    pad.add_argument("--pad-inf",  type=float, default=None, dest="pad_inferior",  metavar="MM")
    pad.add_argument("--pad-si",   type=float, default=None, dest="pad_si",        metavar="MM")
    pad.add_argument("--pad-left", type=float, default=None, dest="pad_left",      metavar="MM")
    pad.add_argument("--pad-right", type=float, default=None, dest="pad_right",     metavar="MM")
    pad.add_argument("--pad-rl",   type=float, default=None, dest="pad_rl",        metavar="MM")
    pad.add_argument("--pad-ant",  type=float, default=None, dest="pad_anterior",  metavar="MM")
    pad.add_argument("--pad-post", type=float, default=None, dest="pad_posterior", metavar="MM")
    pad.add_argument("--pad-ap",   type=float, default=None, dest="pad_ap",        metavar="MM")
    parser.add_argument("--conf", type=float, default=None,
                        help="Detection confidence threshold (default: from model config)")
    parser.add_argument("--regularization", default=None, choices=["cls", "graphtrim", "none"],
                        help="Post-processing regularization strategy (default: from model config)")
    parser.add_argument("--cls-conf", type=float, default=None, dest="cls_conf",
                        help="Confidence threshold for the classification head (cls regularization)")
    parser.add_argument("--device", default=None,
                        help="Inference device: 'cpu', 'cuda', 'cuda:0', … (default: auto)")
    # default=None (NOT "volume") is required: this lets detect() fall back to
    # config.yaml's norm_scope, i.e. whatever normalisation the shipped model was
    # actually trained with. Hardcoding a default here would silently override the
    # model's training-time normalisation for anyone who omits --norm-scope.
    parser.add_argument("--norm-scope", dest="norm_scope", default=None,
                        choices=["volume", "slice_all", "slice"],
                        help="Intensity normalisation scope (default: from model config). "
                             "volume: percentile 0.5/99.5 over the whole volume, computed once. "
                             "slice_all: percentile 0.5/99.5 per slice, all voxels including background. "
                             "slice: percentile 0.5/99.5 per slice, foreground voxels only.")
    args = parser.parse_args()
    input_path = args.input_flag or args.input
    if not input_path:
        parser.error("an input file is required (positional or -i)")

    coords_provided = any(v is not None for v in
                          [args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax])

    # ── Mode: crop from bbox file (NIfTI mask or legacy txt) ─────────────────
    if args.bbox:
        bbox_path = str(args.bbox)
        if bbox_path.endswith(".nii") or bbox_path.endswith(".nii.gz"):
            mask_nii = nib.load(bbox_path)
            nz = np.nonzero(np.asanyarray(mask_nii.dataobj))
            if nz[0].size == 0:
                raise ValueError(f"--bbox: mask {bbox_path} is empty (no non-zero voxels).")
            xmin, xmax = int(nz[0].min()), int(nz[0].max())
            ymin, ymax = int(nz[1].min()), int(nz[1].max())
            zmin, zmax = int(nz[2].min()), int(nz[2].max())
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = _read_bbox_txt(args.bbox)
        _crop_from_coords(input_path, xmin, xmax, ymin, ymax, zmin, zmax,
                          args.output, args.translate)
        return

    # ── Mode: crop from coordinates ───────────────────────────────────────────
    if args.crop or coords_provided:
        missing = [name for name, val in [
            ("-xmin", args.xmin), ("-xmax", args.xmax),
            ("-ymin", args.ymin), ("-ymax", args.ymax),
            ("-zmin", args.zmin), ("-zmax", args.zmax),
        ] if val is None]
        if missing:
            parser.error(f"crop mode requires all 6 coordinates or --bbox file — missing: {', '.join(missing)}")
        _crop_from_coords(input_path,
                          args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax,
                          args.output, args.translate)
        return

    # ── Detect (shared by detect and detect-crop modes) ───────────────────────
    bbox = detect(
        input_path,
        model_path=args.model,
        pad_superior=args.pad_superior,
        pad_inferior=args.pad_inferior,
        pad_si=args.pad_si,
        pad_left=args.pad_left,
        pad_right=args.pad_right,
        pad_rl=args.pad_rl,
        pad_anterior=args.pad_anterior,
        pad_posterior=args.pad_posterior,
        pad_ap=args.pad_ap,
        conf=args.conf,
        regularization=args.regularization,
        cls_conf=args.cls_conf,
        device=args.device,
        norm_scope=args.norm_scope,
    )

    parent, stem = _stem(input_path)
    xmin, xmax = bbox["xmin"], bbox["xmax"]
    ymin, ymax = bbox["ymin"], bbox["ymax"]
    zmin, zmax = bbox["zmin"], bbox["zmax"]

    # ── Mode: detect only ─────────────────────────────────────────────────────
    if args.detect:
        cropbox_path = parent / f"{stem}_cropbox.nii.gz"
        bbox_txt_path = parent / f"{stem}_bbox.txt"
        _warn_overwrite(cropbox_path)
        save_bbox_nifti(bbox, nib.load(input_path), cropbox_path)
        print(f"Cropbox : {cropbox_path}")
        _write_bbox_txt(bbox_txt_path, BBox3D(xmin, xmax + 1, ymin, ymax + 1, zmin, zmax + 1))
        print(f"BBox    : {bbox_txt_path}")
        crop_path = parent / f"{stem}_crop.nii.gz"
        print("\nTo crop and view in FSLeyes:")
        print(f"  {GREEN}sc_crop -i {input_path} --bbox {cropbox_path}{RESET}")
        print(f"  {GREEN}fsleyes {input_path} {crop_path} {cropbox_path} -ot mask -mc 1 0 0 --outline -w 3 &{RESET}")
        return

    # ── Mode: detect + crop (default) ────────────────────────────────────────
    if args.las:
        cropped = bbox["_bbox_pad_las"].crop(bbox["_img_las"], translate=args.translate)
        crop_path = Path(args.output) if args.output else parent / f"{stem}_crop_las.nii.gz"
    else:
        cropped = crop(nib.load(input_path), bbox, translate=args.translate)
        crop_path = Path(args.output) if args.output else parent / f"{stem}_crop.nii.gz"
    cropbox_path = parent / f"{stem}_cropbox.nii.gz"
    bbox_txt_path = parent / f"{stem}_bbox.txt"
    _warn_overwrite(crop_path)
    _warn_overwrite(cropbox_path)
    nib.save(cropped, crop_path)
    save_bbox_nifti(bbox, nib.load(input_path), cropbox_path)
    _write_bbox_txt(bbox_txt_path, BBox3D(xmin, xmax + 1, ymin, ymax + 1, zmin, zmax + 1))
    print(f"Crop    : {crop_path}  shape={cropped.shape}")
    print(f"Cropbox : {cropbox_path}")
    print(f"BBox    : {bbox_txt_path}")
    print("\nTo view in FSLeyes:")
    print(f"  {GREEN}fsleyes {input_path} {crop_path} {cropbox_path} -ot mask -mc 1 0 0 --outline -w 3 &{RESET}")


if __name__ == "__main__":
    main()
