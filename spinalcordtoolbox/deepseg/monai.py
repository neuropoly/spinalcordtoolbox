import os
import glob
import math
import multiprocessing as mp

import torch
import torch.nn.functional as F
import torch.nn as nn

from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0

from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (Compose, EnsureTyped, Invertd, Spacingd,
                              LoadImaged, NormalizeIntensityd, EnsureChannelFirstd,
                              DivisiblePadd, Orientationd, ResizeWithPadOrCropd, ThresholdIntensityd)

# NNUNET global params
INIT_FILTERS = 32
ENABLE_DS = True

nnunet_plans = {
    "UNet_class_name": "PlainConvUNet",
    "UNet_base_num_features": INIT_FILTERS,
    "n_conv_per_stage_encoder": [2, 2, 2, 2, 2, 2],
    "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
    "pool_op_kernel_sizes": [
        [1, 1, 1],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [1, 2, 2]
    ],
    "conv_kernel_sizes": [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3]
    ],
    # NOTE: starting from contrast-agnostic v2.5, the monai-based nnunet model has more features at
    # the deeper layers of the network, hence update the max features in the `plans` dict
    "unet_max_num_features": 384,
}


def create_nnunet_from_plans(path_model, device: torch.device):
    """
    Adapted from nnUNet's source code:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/get_network_from_plans.py#L9

    """
    plans = nnunet_plans
    num_input_channels = 1
    num_classes = 1
    deep_supervision = ENABLE_DS

    num_stages = len(plans["conv_kernel_sizes"])

    dim = len(plans["conv_kernel_sizes"][0])
    conv_op = convert_dim_to_conv_op(dim)

    segmentation_network_class_name = plans["UNet_class_name"]
    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    assert segmentation_network_class_name in mapping.keys(), (
        "The network architecture specified by the plans file is non-standard "
        "(maybe your own?). You'll have to dive into either this function "
        "(get_network_from_plans) or the init of your nnUNetModule to "
        "accomodate that."
    )
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        'n_conv_per_stage'
        if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': plans["n_conv_per_stage_encoder"],
        'n_conv_per_stage_decoder': plans["n_conv_per_stage_decoder"]
    }

    # network class name!!
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(plans["UNet_base_num_features"] * 2 ** i,
                                plans["unet_max_num_features"]) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=plans["conv_kernel_sizes"],
        strides=plans["pool_op_kernel_sizes"],
        num_classes=num_classes,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)

    # this loop only takes about 0.2s on average on a CPU
    chkp_paths = glob.glob(os.path.join(path_model, '**', '*.ckpt'), recursive=True)
    if not chkp_paths:
        raise FileNotFoundError(f"Could not find .ckpt (i.e. model checkpoint) file in {path_model}")
    chkp_path = chkp_paths[0]
    checkpoint = torch.load(chkp_path, map_location=torch.device(device))["state_dict"]
    # NOTE: remove the 'net.' prefix from the keys because of how the model was initialized in lightning
    # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
    for key in list(checkpoint.keys()):
        if 'net.' in key:
            checkpoint[key.replace('net.', '')] = checkpoint[key]
            del checkpoint[key]

    # load the trained model weights
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def prepare_data(path_image, crop_size=(64, 160, 320), padding='edge'):
    # define test transforms
    transforms_test = inference_transforms_single_image(crop_size=crop_size, padding=padding)

    # define post-processing transforms for testing; taken (with explanations) from
    # https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py#L66
    test_post_pred = Compose([
        EnsureTyped(keys=["pred"]),
        Invertd(keys=["pred"], transform=transforms_test,
                orig_keys=["image"],
                meta_keys=["pred_meta_dict"],
                nearest_interp=False, to_tensor=True),
        # The output softseg includes garbage low-intensity values around ~0.0, so we filter them with thresholding
        ThresholdIntensityd(keys=["pred"], threshold=0.1),
        # With spline interpolation, resampled softsegs can end up with values >1.0, so clip to 1.0.
        # TODO: Replace with bilinear once https://github.com/Project-MONAI/MONAI/issues/7836 is solved.
        ThresholdIntensityd(keys=["pred"], threshold=1.0, above=False, cval=1.0)
    ])
    test_ds = Dataset(data=[{"image": path_image}], transform=transforms_test)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=math.ceil(mp.cpu_count() / 2), pin_memory=True)

    return test_loader, test_post_pred


def inference_transforms_single_image(crop_size, padding='edge'):
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RPI"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=2),   # "2" refers to spline interpolation
        ResizeWithPadOrCropd(keys=["image"], spatial_size=crop_size, mode=padding),
        DivisiblePadd(keys=["image"], k=2 ** 5, mode=padding),
        # pad inputs to ensure divisibility by no. of layers nnUNet has (5)
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
    ])


def postprocessing(batch, test_post_pred):
    # take only the highest resolution prediction
    batch["pred"] = batch["pred"][0]

    # NOTE: monai's models do not normalize the output, so we need to do it manually
    if bool(F.relu(batch["pred"]).max()):
        batch["pred"] = F.relu(batch["pred"]) / F.relu(batch["pred"]).max()
    else:
        batch["pred"] = F.relu(batch["pred"])

    post_test_out = [test_post_pred(i) for i in decollate_batch(batch)]

    pred = post_test_out[0]['pred'].cpu()

    return pred
