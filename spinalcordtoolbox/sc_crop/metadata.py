"""
`crop_metadata.json` — the padding contract a model release hands to a downstream consumer
(e.g. SpinalCordToolbox's `sct_deepseg`) so it can reproduce, at inference time, the exact
cropping used to build the training data.

`write_crop_metadata()` is meant to be called right where `detect()` is called during
training-crop generation, with the *same* padding kwargs passed to both -- so the recorded
values (and the recorded `sc_crop` version) can't drift from what was actually used:

    pad_kwargs = dict(pad_superior=40, pad_inferior=100, pad_left=15,
                      pad_right=15, pad_anterior=15, pad_posterior=22)
    bbox = sc_crop.detect(img, **pad_kwargs)
    sc_crop.write_crop_metadata("crop_metadata.json", **pad_kwargs)
"""
import argparse
import json

PAD_FACES = ("superior", "inferior", "left", "right", "anterior", "posterior")
CROP_PAD_KEYS = tuple(f"pad_{face}" for face in PAD_FACES)


def write_crop_metadata(path, *, pad_superior, pad_inferior, pad_left, pad_right, pad_anterior, pad_posterior):
    """
    Write a `crop_metadata.json` recording the padding (mm) used to build training crops, plus
    the `sc_crop` version active at the time of writing.

    :param path: str: Output path (conventionally `crop_metadata.json`, at the root of the
        model's release .zip -- include the same file, unmodified, in every mirror/fold zip
        that makes up the release).
    :param pad_superior: float: Padding (mm) on the superior face.
    :param pad_inferior: float: Padding (mm) on the inferior face.
    :param pad_left: float: Padding (mm) on the left face.
    :param pad_right: float: Padding (mm) on the right face.
    :param pad_anterior: float: Padding (mm) on the anterior face.
    :param pad_posterior: float: Padding (mm) on the posterior face.
    """
    import sc_crop  # local import: avoids a circular import with sc_crop/__init__.py

    metadata = {
        "sc_crop_version": sc_crop.__version__,
        "pad_superior": pad_superior,
        "pad_inferior": pad_inferior,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_anterior": pad_anterior,
        "pad_posterior": pad_posterior,
    }
    with open(path, "w") as fp:
        json.dump(metadata, fp, indent=2)


def _write_metadata_cli(argv):
    """CLI entry point for the `sc_crop write-metadata` subcommand."""
    parser = argparse.ArgumentParser(
        prog="sc_crop write-metadata",
        description="Write a crop_metadata.json for a model release consumed by SpinalCordToolbox (SCT).",
    )
    parser.add_argument("-o", "--output", default="crop_metadata.json")
    for face in PAD_FACES:
        parser.add_argument(f"--pad-{face}", type=float, required=True, dest=f"pad_{face}",
                            help=f"Padding (mm) used at training time for the {face} face.")
    args = parser.parse_args(argv)

    write_crop_metadata(args.output, **{key: getattr(args, key) for key in CROP_PAD_KEYS})
    print(f"Wrote {args.output} -- include it at the root of your release .zip.")
