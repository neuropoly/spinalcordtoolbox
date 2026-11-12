# pytest unit tests for spinalcordtoolbox.image

import sys
import os

import pytest
import numpy as np
import nibabel
import nibabel.orientations
import logging

import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.utils.fs import tmp_create


@pytest.fixture(scope="session")
def image_paths():
    ret = []
    sct_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(sct_dir, "data")
    for cwd, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".nii.gz", ".nii")):
                path = os.path.join(cwd, file)
                ret.append(path)
    return ret


def fake_3dimage_custom(data):
    """
    :return: a Nifti1Image (3D) in RAS+ space
    """
    affine = np.eye(4)
    return nibabel.nifti1.Nifti1Image(data, affine)


def fake_3dimage_sct_custom(data):
    """
    :return: an Image (3D) in RAS+ (aka SCT LPI) space
    """
    i = fake_3dimage_custom(data)
    img = msct_image.Image(np.asanyarray(i.dataobj), hdr=i.header,
                           orientation="LPI",
                           dim=i.header.get_data_shape(),
                           )
    return img


def fake_3dimage_vis():
    """
    :return: a Nifti1Image (3D) in RAS+ space

    Following characteristics:

    - shape[LR] = 7
    - shape[PA] = 8
    - shape[IS] = 9


    Visual thing using voxel art...

    """
    shape = (7, 8, 9)
    data = np.zeros(shape, dtype=np.float32, order="F")

    # "L"
    indices = np.array([
        (0, 1, 6),
        (0, 1, 5),
        (0, 1, 4),
        (0, 1, 3),
        (0, 1, 2),
        (0, 1, 1),
        (0, 2, 1),
        (0, 3, 1),
        (0, 4, 1),
        (0, 5, 1),
    ]).T
    data[indices[0], indices[1], indices[2]] = 7

    # "P"
    indices = np.array([
        (1, 0, 6),
        (1, 0, 5),
        (1, 0, 4),
        (1, 0, 3),
        (1, 0, 2),
        (1, 0, 1),
        (2, 0, 6),
        (3, 0, 5),
        (3, 0, 4),
        (2, 0, 3),
        (1, 0, 2),
    ]).T
    data[indices[0], indices[1], indices[2]] = 9

    # "I"
    indices = np.array([
        (3, 1, 0),
        (2, 1, 0),
        (1, 1, 0),
        (4, 1, 0),
        (5, 1, 0),
        (3, 1, 0),
        (3, 2, 0),
        (3, 3, 0),
        (3, 4, 0),
        (3, 5, 0),
        (3, 6, 0),
        (3, 7, 0),
        (2, 7, 0),
        (1, 7, 0),
        (4, 7, 0),
        (5, 7, 0),
    ]).T
    data[indices[0], indices[1], indices[2]] = 9

    affine = np.eye(4)
    return nibabel.nifti1.Nifti1Image(data, affine)


@pytest.fixture(scope="session")
def fake_3dimage_sct_vis():
    """
    :return: an Image (3D) in RAS+ (aka SCT LPI) space
    """
    i = fake_3dimage_vis()
    img = msct_image.Image(np.asanyarray(i.dataobj), hdr=i.header,
                           orientation="LPI",
                           dim=i.header.get_data_shape(),
                           )
    return img


def fake_3dimage():
    """
    :return: a Nifti1Image (3D) in RAS+ space

    Following characteristics:

    - shape[LR] = 7
    - shape[PA] = 8
    - shape[IS] = 9
    """
    shape = (7, 8, 9)
    data = np.zeros(shape, dtype=np.float32, order="F")

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                data[x, y, z] = (1 + x) * 1 + (1 + y) * 10 + (1 + z) * 100

    if 0:
        for z in range(shape[2]):
            for y in range(shape[1]):
                for x in range(shape[0]):
                    sys.stdout.write(f" {data[x, y, z]: 3d}")
                sys.stdout.write("\n")
            sys.stdout.write("\n")

    affine = np.eye(4)
    return nibabel.nifti1.Nifti1Image(data, affine)


@pytest.fixture(name="fake_3dimage", scope="session")
def fake_3dimage_fixture():
    return fake_3dimage()


def fake_3dimage2():
    """
    :return: a Nifti1Image (3D) in RAS+ space

    Following characteristics:

    - shape[LR] = 1
    - shape[PA] = 2
    - shape[IS] = 3
    """
    shape = (1, 2, 3)
    data = np.zeros(shape, dtype=np.float32, order="F")

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                data[x, y, z] = (1 + x) * 1 + (1 + y) * 10 + (1 + z) * 100

    affine = np.eye(4)
    return nibabel.nifti1.Nifti1Image(data, affine)


def fake_4dimage():
    """
    :return: a Nifti1Image (3D) in RAS+ space

    Following characteristics:

    - shape[LR] = 2
    - shape[PA] = 3
    - shape[IS] = 4
    - shape[t] = 5
    """
    shape = (2, 3, 4, 5)
    data = np.zeros(shape, dtype=np.float32, order="F")

    for t in range(shape[3]):
        for z in range(shape[2]):
            for y in range(shape[1]):
                for x in range(shape[0]):
                    data[x, y, z, t] = (1 + x) * 1 + (1 + y) * 10 + (1 + z) * 100 + (1 + t) * 1000

    affine = np.eye(4)
    return nibabel.nifti1.Nifti1Image(data, affine)


@pytest.fixture(scope="session")
def fake_4dimage_sct():
    """
    :return: an Image (4D) in RAS+ (aka SCT LPI) space
    """
    i = fake_4dimage()
    img = msct_image.Image(np.asanyarray(i.dataobj), hdr=i.header,
                           orientation="LPI",
                           dim=i.header.get_data_shape(),
                           )
    return img


@pytest.fixture(scope="session")
def fake_3dimage_sct():
    """
    :return: an Image (3D) in RAS+ (aka SCT LPI) space
    """
    i = fake_3dimage()
    img = msct_image.Image(np.asanyarray(i.dataobj), hdr=i.header,
                           orientation="LPI",
                           dim=i.header.get_data_shape(),
                           )
    return img


@pytest.fixture(scope="session")
def fake_3dimage_sct2():
    """
    :return: an Image (3D) in RAS+ (aka SCT LPI) space
    """
    i = fake_3dimage2()
    img = msct_image.Image(np.asanyarray(i.dataobj), hdr=i.header,
                           orientation="LPI",
                           dim=i.header.get_data_shape(),
                           )
    return img


def test_slicer(fake_3dimage_sct, fake_3dimage_sct2):
    im3d = fake_3dimage_sct.copy()
    slicer = msct_image.Slicer(im3d, "LPI")
    if 0:
        for im2d in slicer:
            print(im2d)
    assert slicer._nb_slices == 9
    assert 100 < np.mean(slicer[0]) < 200

    slicer = msct_image.Slicer(im3d, "LPS")
    if 0:
        for im2d in slicer:
            print(im2d)
    assert slicer._nb_slices == 9
    assert 900 < np.mean(slicer[0]) < 1000

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(im3d, "LI")

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(im3d, "L")

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(im3d, "Lx")

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(im3d, "LA")

    im3d2 = fake_3dimage_sct.copy()
    im3d2.data += 1000
    slicer = msct_image.SlicerMany((im3d, im3d2), msct_image.Slicer, orientation="LPI")
    for idx_slice, (im2d_a, im2d_b) in enumerate(slicer):
        assert np.all(im2d_b == im2d_a + 1000)

    with pytest.raises(ValueError):
        slicer = msct_image.SlicerMany((im3d, im3d2), msct_image.Slicer, orientation="IS")

    im3d2 = fake_3dimage_sct2.copy()
    with pytest.raises(ValueError):
        slicer = msct_image.SlicerMany((im3d, im3d2), msct_image.Slicer, orientation="LPI")

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(1)

    im3d = fake_3dimage_sct.copy()

    slicer = msct_image.Slicer(im3d, "RPI")
    if 0:
        for im2d in slicer:
            print(im2d)
    slicer = msct_image.Slicer(im3d, "ASR")
    if 0:
        for im2d in slicer:
            print(im2d)

    msct_image.Slicer(im3d, "ASR")[0][0, 0] = -1

    for idx_slice, (slice2d_new, slice2d_old) in enumerate(msct_image.SlicerMany((im3d, fake_3dimage_sct.copy()), msct_image.Slicer, orientation="ASR")):
        if idx_slice == 0:
            assert slice2d_new[0, 0] == -1
            slice2d_new[0, 0] = slice2d_old[0, 0]
        assert (slice2d_new == slice2d_old).all()


def test_nibabel(fake_3dimage):
    img = fake_3dimage
    print(img.header)


def test_nibabel_reorient(fake_3dimage):
    """
    nibabel can reorient since recent versions,
    it's not the best idea to use that feature yet.
    """
    return
    src = fake_3dimage
    ornt = (
        [1, 1],
        [2, 1],
        [0, -1],
    )
    dst = src.as_reoriented(ornt)
    print(dst.header)


def test_change_orientation(tmp_path, fake_3dimage_sct, fake_3dimage_sct_vis):
    print("Spot-checking that physical coordinates don't change")
    for shape_is in (1, 2, 3):
        shape = (1, 1, shape_is)
        print(f"Simple image with shape {shape}")

        data = np.ones(shape, order="F")
        data[:, :, shape_is - 1] += 1
        im_src = fake_3dimage_sct_custom(data)
        im_dst = msct_image.change_orientation(im_src, "ASR")
        # Basic check
        assert im_dst.orientation == "ASR"
        # Basic data check
        assert im_dst.data.mean() == im_src.data.mean()
        # Basic header check: check that the same voxel
        # remains at the same physical position
        aff_src = im_src.header.get_best_affine()
        aff_dst = im_dst.header.get_best_affine()

        # Take the extremities "a" & "z"...
        # consider the original LPI position
        pta_src = np.array([[0, 0, 0, 1]]).T
        ptz_src = np.array([[0, 0, shape_is - 1, 1]]).T
        # and the position in ASR
        pta_dst = np.array([[0, shape_is - 1, 0, 1]]).T
        ptz_dst = np.array([[0, 0, 0, 1]]).T
        # The physical positions should be:
        posa_src = np.matmul(aff_src, pta_src)
        posa_dst = np.matmul(aff_dst, pta_dst)
        print(f"A at src {posa_src.T}")
        print(f"A at dst {posa_dst.T}")
        posz_src = np.matmul(aff_src, ptz_src)
        posz_dst = np.matmul(aff_dst, ptz_dst)
        # and they should be equal
        assert (posa_src == posa_dst).all()
        assert (posz_src == posz_dst).all()
        fn = "".join(str(x) for x in im_src.data.shape)
        im_src.save(str(tmp_path / f"{fn}-src.nii"))
        im_dst.save(str(tmp_path / f"{fn}-dst.nii"))

    np.random.seed(0)

    print("More checking that physical coordinates don't change")
    if 1:
        shape = (7, 8, 9)
        print(f"Simple image with shape {shape}")

        data = np.ones(shape, order="F") * 10
        data[4, 4, 4] = 4
        data[3, 3, 3] = 3
        data[0, 0, 0] = 0
        values = (0, 3, 4)

        im_ref = fake_3dimage_sct_custom(data)
        im_ref.header.set_xyzt_units("mm", "msec")

        import scipy.linalg

        def rand_rot():
            q, _ = scipy.linalg.qr(np.random.randn(3, 3))
            if scipy.linalg.det(q) < 0:
                q[:, 0] = -q[:, 0]
            return q

        affine = im_ref.header.get_best_affine()
        affine[:3, :3] = rand_rot()
        affine[3, :3] = 0.0
        affine[:3, 3] = np.random.random((3))
        affine[3, 3] = 1.0

        affine[0, 0] *= 2
        im_ref.header.set_sform(affine, code='scanner')
        im_ref.header.set_qform(affine, code='scanner')

        orientations = msct_image.all_refspace_strings()
        for ori_src in orientations:
            for ori_dst in orientations:
                print(f"{ori_src} -> {ori_dst}")
                im_src = msct_image.change_orientation(im_ref, ori_src)
                im_dst = msct_image.change_orientation(im_src, ori_dst)

                assert im_src.orientation == ori_src
                assert im_dst.orientation == ori_dst
                assert im_dst.data.mean() == im_src.data.mean()

                # Basic header check: check that the same voxel
                # remains at the same physical position
                aff_src = im_src.header.get_best_affine()
                aff_dst = im_dst.header.get_best_affine()

                data_src = np.array(im_src.data)
                data_dst = np.array(im_dst.data)

                for value in values:
                    pt_src = np.argwhere(data_src == value)[0]
                    pt_dst = np.argwhere(data_dst == value)[0]

                    pos_src = np.matmul(aff_src, np.hstack((pt_src, [1])).reshape((4, 1)))
                    pos_dst = np.matmul(aff_dst, np.hstack((pt_dst, [1])).reshape((4, 1)))
                    if 0:
                        print(f"P at src {pos_src.T}")
                        print(f"P at dst {pos_dst.T}")
                    assert np.allclose(pos_src, pos_dst, atol=1e-3)


def test_more_change_orientation(tmp_path, fake_3dimage_sct, fake_3dimage_sct_vis):

    im_src = fake_3dimage_sct.copy()
    im_src.save(os.path.join(tmp_path, "src.nii"), mutable=True)

    print(im_src.orientation, im_src.data.shape)

    def orient2shape(orient):
        # test-data-specific thing
        letter2dim = dict(
            L=7,
            R=7,
            A=8,
            P=8,
            I=9,
            S=9,
        )
        return tuple([letter2dim[x] for x in orient])

    orientation = im_src.orientation  # LPI
    assert im_src.header.get_best_affine()[:3, 3].tolist() == [0, 0, 0]
    im_dst = msct_image.change_orientation(im_src, "RPI")
    print(im_dst.orientation, im_dst.data.shape)
    assert im_dst.data.shape == orient2shape("RPI")
    assert im_dst.header.get_best_affine()[:3, 3].tolist() == [7 - 1, 0, 0]

    # spot check
    orientation = im_src.orientation  # LPI
    im_dst = msct_image.change_orientation(im_src, "IRP")
    print(im_dst.orientation, im_dst.data.shape)
    assert im_dst.data.shape == orient2shape("IRP")

    # to & fro
    im_dst2 = msct_image.change_orientation(im_dst, orientation)
    print(im_dst2.orientation, im_dst2.data.shape)
    assert im_dst2.orientation == im_src.orientation
    assert im_dst2.data.shape == orient2shape(orientation)
    assert (im_dst2.data == im_src.data).all()
    assert np.allclose(im_src.header.get_best_affine(), im_dst2.header.get_best_affine())

    # fn = os.path.join(path_tmp, "pouet.nii")
    im_ref = fake_3dimage_sct.copy()
    im_src = fake_3dimage_sct.copy()
    orientation = im_src.orientation
    im_src.change_orientation("ASR").change_orientation(orientation)
    assert im_src.orientation == im_ref.orientation
    assert (im_dst2.data == im_src.data).all()
    assert np.allclose(im_src.header.get_best_affine(), im_ref.header.get_best_affine())

    im_dst2 = msct_image.change_orientation(im_dst, orientation)
    print(im_dst2.orientation, im_dst2.data.shape)
    assert im_dst2.orientation == im_src.orientation
    assert im_dst2.data.shape == orient2shape(orientation)
    assert (im_dst2.data == im_src.data).all()
    assert np.allclose(im_src.header.get_best_affine(), im_dst2.header.get_best_affine())

    # copy
    im_dst = im_src.copy().change_orientation("IRP")
    assert im_dst.data.shape == orient2shape("IRP")
    print(im_dst.orientation, im_dst.data.shape)

    print("Testing orientation persistence")
    img = im_src.copy()
    orientation = img.orientation
    fn = os.path.join(tmp_path, "pouet.nii")
    img.change_orientation("PIR").save(fn)
    assert img.data.shape == orient2shape("PIR")
    img = msct_image.Image(fn)
    assert img.orientation == "PIR"
    assert img.data.shape == orient2shape("PIR")
    print(img.orientation, img.data.shape)

    # typical pattern
    img = fake_3dimage_sct_vis.copy()
    print(img.header.get_best_affine())
    orientation = img.orientation
    fn = os.path.join(tmp_path, "vis.nii")
    img.save(fn, mutable=True).change_orientation("ALS")
    fn2 = msct_image.add_suffix(img.absolutepath, "_als")
    img.save(path=fn2, mutable=True)
    img = msct_image.Image(fn2)
    assert img.orientation == "ALS"
    assert img.data.shape == orient2shape("ALS")
    print(img.header.get_best_affine())

    img.save(fn, mutable=True).change_orientation("RAS")
    fn2 = msct_image.add_suffix(img.absolutepath, "_ras")
    img.save(path=fn2, mutable=True)
    img = msct_image.Image(fn2)
    assert img.orientation == "RAS"
    assert img.data.shape == orient2shape("RAS")
    print(img.header.get_best_affine())

    img.save(fn, mutable=True).change_orientation("RPI")
    fn2 = msct_image.add_suffix(img.absolutepath, "_rpi")
    img.save(path=fn2, mutable=True)
    img = msct_image.Image(fn2)
    assert img.orientation == "RPI"
    assert img.data.shape == orient2shape("RPI")
    print(img.header.get_best_affine())

    img.save(fn, mutable=True).change_orientation("PLI")
    fn2 = msct_image.add_suffix(img.absolutepath, "_pli")
    img.save(path=fn2, mutable=True)
    img = msct_image.Image(fn2)
    assert img.orientation == "PLI"
    assert img.data.shape == orient2shape("PLI")
    print(img.header.get_best_affine())

    # print(src.header)
    possibilities = ["ASR", "SRA", "RAS", ]
    possibilities = msct_image.all_refspace_strings()
    for orientation in possibilities:
        dst = msct_image.change_orientation(im_src, orientation)
        # dst.save(f"pouet-{dst.orientation}.nii")
        print(orientation, dst.orientation, dst.data.shape, dst.dim)
        assert orientation == dst.orientation
        # assert dst.data.shape[:] == np.array(dst.dim)[:3]
        # print(dst.header)


def test_change_nd_orientation(fake_4dimage_sct):
    im_src = fake_4dimage_sct.copy()
    path_tmp = tmp_create(basename="test-reorient")
    im_src.save(os.path.join(path_tmp, "src.nii"), mutable=True)

    print(im_src.orientation, im_src.data.shape)

    def orient2shape(orient):
        # test-data-specific thing
        letter2dim = dict(
            L=2,
            R=2,
            A=3,
            P=3,
            I=4,
            S=4,
        )
        return tuple([letter2dim[x] for x in orient] + [5])

    orientation = im_src.orientation
    assert orientation == "LPI"
    assert im_src.header.get_best_affine()[:3, 3].tolist() == [0, 0, 0]

    im_dst = msct_image.change_orientation(im_src, "RPI")
    assert im_dst.orientation == "RPI"
    assert im_dst.data.shape == orient2shape("RPI")
    assert im_dst.header.get_best_affine()[:3, 3].tolist() == [2 - 1, 0, 0]


def test_crop(fake_3dimage_sct):

    im_src = fake_3dimage_sct.copy()

    crop_spec = dict(((0, (1, 3)), (1, (2, 4)), (2, (3, 5))))
    print(crop_spec)

    im_dst = msct_image.spatial_crop(im_src, crop_spec)

    print("Check shape")
    assert im_dst.data.shape == (3, 3, 3)
    print("Check world pos")
    aff_src = im_src.header.get_best_affine()
    aff_dst = im_dst.header.get_best_affine()
    pos_src = np.matmul(aff_src, np.array([[1, 2, 3, 1]]).T)
    pos_dst = np.matmul(aff_dst, np.array([[0, 0, 0, 1]]).T)
    assert (pos_src == pos_dst).all()


def test_change_shape(fake_3dimage_sct):

    # Add dimension
    im_src = fake_3dimage_sct
    shape = tuple(list(im_src.data.shape) + [1])
    im_dst = msct_image.change_shape(im_src, shape)
    path_tmp = tmp_create(basename="test-reshape")
    src_path = os.path.join(path_tmp, "src.nii")
    dst_path = os.path.join(path_tmp, "dst.nii")
    im_src.save(src_path)
    im_dst.save(dst_path)
    im_src = msct_image.Image(src_path)
    im_dst = msct_image.Image(dst_path)
    assert im_dst.data.shape == shape

    data_src = im_src.data
    data_dst = im_dst.data

    assert (data_dst.reshape(data_src.shape) == data_src).all()

    # Remove dimension
    im_dst = im_dst.change_shape(im_src.data.shape)
    assert im_dst.data.shape == im_src.data.shape


def test_sequences(fake_3dimage_sct):
    """
    Test correct behaviour in some Image manipulation sequences
    """

    img = fake_3dimage_sct.copy()

    path_tmp = tmp_create(basename="test-sequences")

    path_a = os.path.join(path_tmp, 'a.nii')
    path_b = os.path.join(path_tmp, 'b.nii')

    img.save(path_a)
    assert img.absolutepath is None

    img.save(path_b, mutable=True)
    assert img._path is not None
    assert img.absolutepath is not None
    assert img.absolutepath == os.path.abspath(path_b)

    img.save(path_a) \
        .change_orientation("RPI") \
        .save(path_b, mutable=True)
    assert img.absolutepath is not None
    assert img.absolutepath == os.path.abspath(path_b)


def test_add_suffix():
    assert msct_image.add_suffix('t2.nii', '_mean') == 't2_mean.nii'
    assert msct_image.add_suffix('t2.nii.gz', 'a') == 't2a.nii.gz'
    assert msct_image.add_suffix(os.path.join('var', 'lib', 'usr', 't2.nii.gz'), 'sfx') \
           == os.path.join('var', 'lib', 'usr', 't2sfx.nii.gz')
    assert msct_image.add_suffix(os.path.join('var', 'lib.version.3', 'usr', 't2.nii.gz'), 'sfx') \
           == os.path.join('var', 'lib.version.3', 'usr', 't2sfx.nii.gz')


def test_splitext():
    assert msct_image.splitext('image.nii') == ('image', '.nii')
    assert msct_image.splitext('image.nii.gz') == ('image', '.nii.gz')
    assert msct_image.splitext(os.path.join('folder', 'image.nii.gz')) == (os.path.join('folder', 'image'), '.nii.gz')
    assert msct_image.splitext('nice.image.nii.gz') == ('nice.image', '.nii.gz')
    assert msct_image.splitext(os.path.join('nice.folder', 'image.nii.gz')) == (os.path.join('nice.folder', 'image'), '.nii.gz')
    assert msct_image.splitext('image.tar.gz') == ('image', '.tar.gz')


def test_tolerance_of_affine_mismatch_check(caplog):
    """Verify that affine mismatch error is thrown only for mismatches above a certain tolerance."""
    # ERROR NOT EXPECTED (Affine matrices have slight differences, but are close enough to be equivalent)
    # NB: Specific values taken from anonymized data from https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3251
    qform_affine = np.array([[-0.0000000613307, -0.0032542832702, -0.8999945527288, 36.8009071350098],
                             [-0.9322916865349, -0.0000000613307, 0.0000000594134, 214.2190246582031],
                             [0.0000000615451, -0.9322860067718, 0.0031415651366, 122.8873901367188],
                             [0.0000000000000, 0.0000000000000, 0.0000000000000, 1.0000000000000]])
    sform_affine = np.array([[-0.0000000000007, -0.0032542543486, -0.8999945521355, 36.8009071350098],
                             [-0.9322916865349, 0.0000000001912, -0.0000000000000, 214.2190246582031],
                             [-0.0000000001912, -0.9322860240936, 0.0031415862031, 122.8873901367188],
                             [0.0000000000000, 0.0000000000000, 0.0000000000000, 1.0000000000000]])
    header_e7 = nibabel.Nifti1Header()
    header_e7.set_sform(affine=sform_affine)
    header_e7.set_qform(affine=qform_affine)
    msct_image.Image(param=[1, 1, 1], hdr=header_e7, check_sform=True)

    # ERROR NOT EXPECTED (A bigger discrepancy is introduced, but it doesn't exceed the tolerance of the check)
    qform_affine_e3 = qform_affine.copy()
    qform_affine_e3[0, 0] += 1e-3
    header_e3 = nibabel.Nifti1Header()
    header_e3.set_sform(affine=sform_affine)
    header_e3.set_qform(affine=qform_affine_e3)
    msct_image.Image(param=[1, 1, 1], hdr=header_e3, check_sform=True)

    # ERROR EXPECTED (A bigger discrepancy is introduced, and it does exceed the tolerance of the check)
    qform_affine_e2 = qform_affine.copy()
    qform_affine_e2[0, 0] += 1e-2
    header_e2 = nibabel.Nifti1Header()
    header_e2.set_sform(affine=sform_affine)
    header_e2.set_qform(affine=qform_affine_e2)

    # Temporarily avoid the ValueError, but still check for the message in log output
    # with pytest.raises(ValueError) as e:
    # assert "Image sform does not match qform" in str(e.value)
    # See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4745
    with caplog.at_level(logging.ERROR):
        msct_image.Image(param=[1, 1, 1], hdr=header_e2, check_sform=True)
    assert "has different qform and sform matrices" in caplog.text
