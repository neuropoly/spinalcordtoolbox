# Author: charley
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file LICENSE.TXT
import matplotlib.pyplot as plt
import numpy as np
import scripts.sct_utils as sct
import skimage
import torch
import torchvision
import yaml
from skimage.feature import peak_local_max
from spinalcordtoolbox.vertebrae.models import *
from torchvision import transforms


def normalize(arr):
    ma = arr.max()
    mi = arr.min()
    return (arr - mi) / (ma - mi)


def prediction_coordinates(Image, model, aim='full', threshold=0.3, heatmap=0, cud_test=False):
    """ take an Image as input and output the predicted coordinates.
     Post processing to remove obvious false positive
     Compute metrics as well and add it to previously existing table
     param: aim 'full' or 'c2'. Changes the threshold fro point retrieval
     param: threshold  when heatmap=0 (label fil is returned)
     changes the threshold of the relative max to retrieve data points (sklear_peak_local_max function)
     param: heatmap: if 1 return heatmap image output by the network else, retrieve point and apply post processing
     param: cud_test use to specify if you don't want to use cuda (used for performance testing)
     """
    global cuda_available
    cuda_available = cud_test
    shape_im = Image.shape
    shape_im = sorted(shape_im)
    if aim == 'c2':
        final, coordinates = infer_image(Image, model, thr=0.99)
        return (coordinates)
    else:
        final, coordinates = infer_image(Image, model, thr=threshold)
    if heatmap == 1:
        print(np.max(final))
        return final
    sct.printv('post_processing')
    print(coordinates)
    if len(coordinates) > 1:
        coord_out = post_processing(coordinates)

        if len(coord_out) < 2:
            coord_out = coordinates
            return coord_out
        else:
            return (coord_out)
    else:
        return (100)


def post_processing(coordinates):
    coordinates_tmp = []
    coordinates = sorted(coordinates, key=lambda x: x[0])
    width_pos = [x[1] for x in coordinates]
    height_pos = [x[0] for x in coordinates]

    # Remove points that are misaligned with the other.
    mean = np.median(width_pos)
    to_remove = []
    for i in range(len(width_pos)):
        if abs(width_pos[i] - mean) > 15:
            to_remove.append(i)
    for i in range(len(coordinates)):
        if i in to_remove:
            pass
        else:
            coordinates_tmp.append(coordinates[i])

    width_pos_c = [x[1] for x in coordinates_tmp]
    height_pos_c = [x[0] for x in coordinates_tmp]
    new_height = []
    new_width = []
    i = 0
    while i < len(height_pos_c):
        j = i
        tmp = []

        while j < len(height_pos_c) - 2 and abs(height_pos_c[j] - height_pos_c[j + 1]) < 10:
            if len(tmp) > 0:
                if abs(height_pos_c[tmp[0]] - height_pos_c[j + 1]) > 20:
                    break
            tmp.append(j)
            j += 1

        if len(tmp) > 0:

            h = np.round(np.mean(height_pos_c[tmp[0]:tmp[len(tmp) - 1] + 1]))
            w = np.round(np.mean(width_pos_c[tmp[0]:tmp[len(tmp) - 1] + 1]))

            new_height.append(h)
            new_width.append(w)
            i += len(tmp) + 1

        else:
            h = height_pos_c[i]
            w = width_pos_c[i]

            new_height.append(h)
            new_width.append(w)
            i += 1
    coordinates_tmp = []
    for a in range(len(new_height)):
        coordinates_tmp.append([new_height[a], new_width[a]])
    coordinates = sorted(coordinates_tmp, key=lambda x: x[0])
    width_pos_c = [x[1] for x in coordinates_tmp]
    height_pos_c = [x[0] for x in coordinates_tmp]
    distance = []
    coord_out = []

    to_remove = []
    for i in range(len(height_pos_c) - 1):
        distance.append(height_pos_c[i + 1] - height_pos_c[i])
    dis_mean_g = np.mean(distance)
    for i in range(1, len(height_pos_c) - 1):
        dis_mean = dis_mean_g
        if i < len(height_pos_c) - 4:
            distance_2 = []
            for j in range(4):
                distance_2.append(height_pos_c[i + j] - height_pos_c[i + j - 1])
                dis_mean = np.median(distance_2)
            # Debugging print: print(dis_mean)
        if abs(height_pos_c[i] - height_pos_c[i - 1]) < dis_mean - 0.1 * dis_mean:
            if abs(height_pos_c[i + 1] - height_pos_c[i]) < dis_mean - 0.7 * dis_mean:
                to_remove.append(i)
    for i in range(len(coordinates_tmp)):
        if i in to_remove:
            pass
        else:
            coord_out.append(coordinates_tmp[i])

    return coord_out


def infer_image(image, model, c=0.02, thr=0.3):
    """Inference function apply transformation to image to feed it to the network
       param: image input image
       param model: model object
       param c: CLAHE clip limit value
       param thr : relative threshold used for local max retrieval. """
    coord_out = []
    shape_im = image.shape
    final = np.zeros((shape_im[0], shape_im[1]))
    # retrieve 2-D for transformation (CLAHE & Normalization )
    patch = image[:, :, 0]
    patch = normalize(patch)
    # patch = skimage.exposure.equalize_adapthist(patch, kernel_size=10, clip_limit=0.02)
    patch = np.expand_dims(patch, axis=-1)
    patch = transforms.ToTensor()(patch).unsqueeze(0)
    if cuda_available:
        patch = patch.cuda()
    patch = patch.float()
    patch_out = model(patch)
    patch_out = patch_out.data.cpu().numpy()
    plt.imshow(patch_out[0, 0, :, :])
    plt.show()
    plt.savefig('heat_test.png')

    # retrieveal of coordinates by looking at local max which value are > th determined previously
    coordinates_tmp = peak_local_max(patch_out[0, 0, :, :], min_distance=5, threshold_rel=thr)
    final = patch_out[0, 0, :, :]
    for x in coordinates_tmp:
        coord_out.append([x[1], x[0]])
    if coord_out == []:
        coord_out = [0, 0]
    return final, coord_out