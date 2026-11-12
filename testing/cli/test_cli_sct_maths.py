# pytest unit tests for sct_maths

import os
import pytest
import traceback as tb
import logging

import numpy as np

from spinalcordtoolbox.image import Image, compute_dice
from spinalcordtoolbox.utils.sys import sct_test_path
from spinalcordtoolbox.scripts import sct_maths

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("ofolder", ["tmp_path", None])
@pytest.mark.parametrize("args_to_test, fname_in, fname_gt", [
    (["-denoise", "1"],       sct_test_path("t2", "t2.nii.gz"), None),  # TODO: Add gt
    (["-otsu-median", "5,5"], sct_test_path("t2", "t2.nii.gz"), None),  # TODO: Add gt
], ids=["-denoise", "-otsu-median"])
def test_sct_maths_basic(args_to_test, fname_in, fname_gt, ofolder, tmp_path):
    """Basic test to ensure core functionality runs without error."""
    # base args to test
    args = ["-i", fname_in]
    args.extend(args_to_test)

    # test all combinations of both (user-supplied and default) output filenames/folders
    fname_out = "img_out.nii.gz"
    path_out = (str(tmp_path) if ofolder == "tmp_path" else os.getcwd())
    fpath_out = os.path.join(path_out, fname_out)
    # sct_maths doesn't have an -ofolder argument, so we simulate it by passing the absolute path (if ofolder)
    args.extend(["-o", fpath_out if ofolder else fname_out])

    # run script
    sct_maths.main(argv=args)

    # test all output files exist
    assert os.path.exists(fpath_out)

    # test against ground truth (if specified)
    if fname_gt:
        im_out = Image(fpath_out)
        im_gt = Image(fname_gt)
        # replace with allclose, etc. depending on ground truth
        dice_segmentation = compute_dice(im_out, im_gt, mode='3d', zboundaries=False)
        assert dice_segmentation > 0.95

    # remove output files if tmp_path wasn't used
    if ofolder != "tmp_path":
        os.unlink(fpath_out)


@pytest.mark.sct_testing
def test_sct_maths_percent_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_maths.main(argv=['-i', sct_test_path('mt', 'mtr.nii.gz'), '-percent', '95', '-o', 'test.nii.gz'])


@pytest.mark.sct_testing
def test_sct_maths_add_integer_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_maths.main(argv=['-i', sct_test_path('mt', 'mtr.nii.gz'), '-add', '1', '-o', 'test.nii.gz'])


@pytest.mark.parametrize('axis_name, axis', [('x', 0), ('y', 1), ('z', 2), ('t', 3)])
def test_sct_maths_std_4d_output(axis_name, axis, tmp_path):
    """Run -std on 4D data and verify that the output remains an array with the expected shape."""
    data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    path_in = str(tmp_path / "im_4d.nii.gz")
    path_out = str(tmp_path / f"im_std_{axis_name}.nii.gz")
    Image(data).save(path_in)

    sct_maths.main(['-i', path_in, '-std', axis_name, '-o', path_out])

    expected = np.std(data, axis=axis, ddof=1)
    if axis < 3:
        expected = np.expand_dims(expected, axis)
    actual = Image(path_out).data
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', ['0', '1', '2'])
def test_sct_maths_symmetrize(dim, tmp_path):
    """Run the CLI script, then verify that symmetrize properly flips and
    averages the image data."""
    path_in = sct_test_path('t2', 't2.nii.gz')
    path_out = str(tmp_path/f't2_sym_{dim}.nii.gz')
    sct_maths.main(argv=['-i', path_in, '-symmetrize', str(dim),
                         '-o', path_out])
    im_in = Image(path_out)
    im_out = Image(path_out)
    assert np.array_equal(im_out.data,
                          (im_in.data + np.flip(im_in.data, axis=int(dim))) / 2.0)


def test_arithmetic_mixed_dtype(tmp_path):
    """Test whether arithmetic operations can be performed on images with different dtypes without crashing."""
    # make a dummy integer image
    path_im_int = str(tmp_path / "im_int.nii.gz")
    Image(np.ones((2, 3, 4), dtype=int)).save(path_im_int)

    # make a dummy floating point image
    path_im_float = str(tmp_path / "im_float.nii.gz")
    Image(np.ones((2, 3, 4), dtype=float)).save(path_im_float)

    # output filename, will be overwritten several times
    path_im_out = str(tmp_path / "im_out.nii.gz")

    # for each operation, test various combinations
    for op in ["-add", "-sub", "-mul", "-div"]:
        sct_maths.main([
            "-i", path_im_int,
            op, path_im_float, path_im_int,
            "-o", path_im_out,
        ])
        sct_maths.main([
            "-i", path_im_int,
            op, path_im_int, path_im_float,
            "-o", path_im_out,
        ])
        sct_maths.main([
            "-i", path_im_float,
            op, path_im_int,
            "-o", path_im_out,
        ])


def test_sub_between_uints(tmp_path):
    """Make sure subtracting uints results in a signed integer, rather than a wrapped-around uint."""
    # make two dummy unsigned integer images, one zeros and one ones
    path_im_uint_zeros = str(tmp_path / "im_uint_zeros.nii.gz")
    Image(np.zeros((2, 3, 4), dtype=np.uint8)).save(path_im_uint_zeros)
    path_im_uint_ones = str(tmp_path / "im_uint_ones.nii.gz")
    Image(np.ones((2, 3, 4), dtype=np.uint8)).save(path_im_uint_ones)

    # test subtraction
    path_im_out = str(tmp_path / "im_out.nii.gz")
    sct_maths.main([
        "-i", path_im_uint_zeros,
        "-sub", path_im_uint_ones,
        "-o", path_im_out,
    ])

    # ensure that output matches expected value
    expected = np.ones((2, 3, 4), dtype=np.int16) * -1
    actual = Image(path_im_out).data
    assert np.array_equal(expected, actual)
    assert expected.dtype == actual.dtype


def test_div_between_uints(tmp_path):
    """Make sure dividing uints results in a float, rather than a truncated uint."""
    # make two dummy unsigned integer images, one ones and one twos
    path_im_uint_twos = str(tmp_path / "im_uint_zeros.nii.gz")
    Image(np.ones((2, 3, 4), dtype=np.uint8) * 2).save(path_im_uint_twos)
    path_im_uint_ones = str(tmp_path / "im_uint_ones.nii.gz")
    Image(np.ones((2, 3, 4), dtype=np.uint8)).save(path_im_uint_ones)

    # test division
    path_im_out = str(tmp_path / "im_out.nii.gz")
    sct_maths.main([
        "-i", path_im_uint_ones,
        "-div", path_im_uint_twos,
        "-o", path_im_out,
    ])

    # ensure that output matches expected value
    expected = np.ones((2, 3, 4), dtype=np.float32) * 0.5
    actual = Image(path_im_out).data
    assert np.array_equal(expected, actual)
    assert expected.dtype == actual.dtype


def run_arithmetic_operation(tmp_path, dims, ops):
    """Generate some dummy data, then run -add/-sub/-mul/-div on the data."""
    assert len(dims) > 1  # dim[0] -> -i, dim[1:] -> ops
    args = []
    im_list = []
    # Generate input images
    for i, dim in enumerate(dims):
        path_im = str(tmp_path / f"im_{i}.nii.gz")
        Image(np.ones(dim)).save(path_im)
        if i == 0:
            args.extend(["-i", path_im])
        else:
            im_list.append(path_im)
    # Generate arg string
    if not isinstance(ops, list):
        ops = [ops]
    for op in ops:
        args.append(op)
        args.extend(im_list)
    # Add output image to argument list
    path_out = str(tmp_path / f"im_out{''.join(ops)}.nii.gz")
    args.extend(["-o", path_out])
    # Call sct_maths and return output data
    sct_maths.main(args)
    return Image(path_out).data


@pytest.mark.parametrize('ndims', [(3, 3), (4, 4), (3, 3, 3), (4, 4, 4)])
@pytest.mark.parametrize('op', ['-add', '-mul', '-sub', '-div'])
def test_arithmetic_operation_output_dimensions(tmp_path, ndims, op):
    """Test that arithmetic operations return the correct dimensions across various combinations
       of 3D and 4D images."""
    possible_dims = {3: [10, 15, 20], 4: [10, 15, 20, 5]}
    dims = [possible_dims[n] for n in ndims]
    data_out = run_arithmetic_operation(tmp_path, dims, op)
    dim_expected = dims[0]  # Expected dimension should match the input image
    dim_out = list(data_out.shape)
    assert dim_out == dim_expected, f"Calling {op} on ndims {ndims} resulted in mismatch."


@pytest.mark.parametrize('ndims', [(3, 3), (4, 4), (3, 3, 3), (4, 4, 4)])
def test_chained_arithmetic_operations_output_dimensions(tmp_path, ndims):
    """Test that arithmetic operations return the correct dimensions across various combinations
       of 3D and 4D images."""
    ops = ['-add', '-mul', '-sub', '-div']
    possible_dims = {3: [10, 15, 20], 4: [10, 15, 20, 5]}
    dims = [possible_dims[n] for n in ndims]
    data_out = run_arithmetic_operation(tmp_path, dims, ops)
    dim_expected = dims[0]  # Expected dimension should match the input image
    dim_out = list(data_out.shape)
    assert dim_out == dim_expected, f"Calling {ops} on ndims {ndims} resulted in mismatch."


@pytest.mark.parametrize('ndims', [(3, 4), (3, 4, 5)])
@pytest.mark.parametrize('op', ['-add', '-mul', '-sub', '-div'])
def test_mismatched_dimensions_error(tmp_path, ndims, op):
    """Test that passing images of mismatched dimensions returns a user-friendly error."""
    possible_dims = {3: [10, 15, 20], 4: [10, 15, 20, 5], 5: [10, 15, 20, 5, 1]}
    dims = [possible_dims[n] for n in ndims]
    with pytest.raises(SystemExit) as e:
        run_arithmetic_operation(tmp_path, dims, op)
    assert 'printv(f"ERROR: -{arg_name}: {e}"' in str(tb.format_list(tb.extract_tb(e.tb)))


@pytest.mark.parametrize('op', ['-add', '-mul'])
def test_add_mul_4d_image_with_no_argument(op, tmp_path):
    """Test that passing a 4D image with bare '-add' or '-mul' operates on the 3D volumes."""
    # Generate input image
    base_dim = [20, 20, 20]
    n_vol = 5
    dim = base_dim + [n_vol]
    path_im = str(tmp_path / "im.nii.gz")
    val = 2
    Image(np.ones(dim) * val).save(path_im)
    # Generate output image
    path_out = str(tmp_path / "im_out.nii.gz")
    sct_maths.main(["-i", path_im, op, "-o", path_out])
    # Validate output data
    data_out = Image(path_out).data
    dim_out = list(data_out.shape)
    assert dim_out == base_dim
    if op == '-add':
        assert np.all(data_out == val * n_vol)
    elif op == '-mul':
        assert np.all(data_out == val ** n_vol)


@pytest.mark.parametrize('op', ['-add', '-mul'])
def test_add_mul_4d_image_with_volumewise(op, tmp_path):
    """Test that passing a 4D image with '-volumewise' operates on the 3D volumes."""
    # Generate input image
    base_dim = [20, 20, 20]
    n_vol = 5
    dim = base_dim + [n_vol]
    path_im = str(tmp_path / "im.nii.gz")
    val = 2
    val_op = 5
    Image(np.ones(dim) * val).save(path_im)
    # Generate output image
    path_out = str(tmp_path / "im_out.nii.gz")
    sct_maths.main(["-i", path_im, op, str(val_op), "-volumewise", "1", "-o", path_out])
    # Validate output data
    data_out = Image(path_out).data
    dim_out = list(data_out.shape)
    assert dim_out == dim
    if op == '-add':
        assert np.all(data_out == val + val_op)
    elif op == '-mul':
        assert np.all(data_out == val * val_op)


@pytest.mark.parametrize('similarity_arg', ['-mi', '-minorm', '-corr'])
def test_similarity_metrics(tmp_path, similarity_arg):
    # Compute similarity between identical images
    path_in = sct_test_path('t2', 't2.nii.gz')
    path_out = str(tmp_path / f"output_metric{similarity_arg}.txt")
    sct_maths.main(['-i', path_in, similarity_arg, path_in, '-o', path_out])
    # Validate output values
    with open(path_out, 'r') as f:
        output = f.read().split("\n")
    assert len(output) == 2  # Text file takes the form "<metric name>: \n<value>"
    similarity_metric = float(output[1])
    if similarity_arg == "-mi":
        # NB: This number is image-dependent, since MI between identical signals is just the entropy of the signal.
        #     (https://stats.stackexchange.com/a/372685). So, we just hardcode the value for t2.nii.gz here.
        assert np.isclose(similarity_metric, 1.80845908)
    else:
        # Otherwise, adjusted similarity metrics should be ~1 for identical signals
        assert np.isclose(similarity_metric, 1)
