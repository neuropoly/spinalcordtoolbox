# This file is vendored from ivadomed 2.9.10 (https://github.com/ivadomed/ivadomed),
# originally `ivadomed/models.py`, licensed under the MIT License.
# See the LICENSE.md file in this directory for the full license text.
#
# Modifications for SCT:
#   - Fixed flake8 warnings, without changing behavior.
#   - `torch.load` is replaced by `pickle_compat.torch_load`, so that ivadomed-pickled `.pt` models load with the vendored
#     classes.

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.nn import init
from pathlib import Path
import torchvision.models

from spinalcordtoolbox.ivadomed.pickle_compat import torch_load


# Modified from torchvision.models.resnet.Resnet
class ResNet(nn.Module):
    """ResNet model based on
     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385.pdf>`_

    Args:
        block (nn.Module): Basic block of the network (such as conv + bn + non-nonlinearity)
        layers (int list): Number of blocks to stack (network depth) after each downsampling step.
        num_classes (int): Number of GT classes.
        zero_init_residual (bool): if True, zero-initialize the last BN in each residual branch,
        so that the residual branch starts with zeros, and each residual block behaves like an identity.
        groups (int): Number of parallel branches in the network, see `"paper" <https://arxiv.org/pdf/1611.05431.pdf>`_
        width_per_group (int): base width of the blocks
        replace_stride_with_dilation (tuple): each element in the tuple indicates if we replace the 2x2 stride with a dilated convolution
        norm_layer (layer): Custom normalization layer, if not provided BatchNorm2d is used

    """

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, torchvision.models.resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, torchvision.models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                torchvision.models.resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        preds = F.softmax(x, dim=1)
        # Remove background class
        preds = preds[:, 1:]
        return preds

    def forward(self, x):
        return self._forward_impl(x)


class resnet18(ResNet):
    def __init__(self, **kwargs):
        super().__init__(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        dropout_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"article" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, dropout_rate=0.3, num_classes=2, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = torchvision.models.densenet._DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=dropout_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = torchvision.models.densenet._Transition(num_input_features=num_features,
                                                                num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        preds = F.softmax(out, dim=1)
        # Remove background class
        preds = preds[:, 1:]
        return preds


class densenet121(DenseNet):
    def __init__(self, **kwargs):
        super().__init__(32, (6, 12, 24, 16), 64)


class DownConv(Module):
    """Two successive series of down convolution, batch normalization and dropout in 2D.
    Used in U-Net's encoder.

    Args:
        in_feat (int): Number of channels in the input image.
        out_feat (int): Number of channels in the output image.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.

    Attributes:
        conv1 (Conv2d): First 2D down convolution with kernel size 3 and padding of 1.
        conv1_bn (BatchNorm2d): First 2D batch normalization.
        conv1_drop (Dropout2d): First 2D dropout.
        conv2 (Conv2d): Second 2D down convolution with kernel size 3 and padding of 1.
        conv2_bn (BatchNorm2d): Second 2D batch normalization.
        conv2_drop (Dropout2d): Second 2D dropout.
    """

    def __init__(self, in_feat, out_feat, dropout_rate=0.3, bn_momentum=0.1, is_2d=True):
        super(DownConv, self).__init__()
        if is_2d:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
            dropout = nn.Dropout2d
        else:
            conv = nn.Conv3d
            bn = nn.InstanceNorm3d
            dropout = nn.Dropout3d

        self.conv1 = conv(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = bn(out_feat, momentum=bn_momentum)
        self.conv1_drop = dropout(dropout_rate)

        self.conv2 = conv(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = bn(out_feat, momentum=bn_momentum)
        self.conv2_drop = dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x


class UpConv(Module):
    """2D down convolution.
    Used in U-Net's decoder.

    Args:
        in_feat (int): Number of channels in the input image.
        out_feat (int): Number of channels in the output image.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.

    Attributes:
        downconv (DownConv): Down convolution.
    """

    def __init__(self, in_feat, out_feat, dropout_rate=0.3, bn_momentum=0.1, is_2d=True):
        super(UpConv, self).__init__()
        self.is_2d = is_2d
        self.downconv = DownConv(in_feat, out_feat, dropout_rate, bn_momentum, is_2d)

    def forward(self, x, y):
        # For retrocompatibility purposes
        if not hasattr(self, "is_2d"):
            self.is_2d = True
        mode = 'bilinear' if self.is_2d else 'trilinear'
        dims = -2 if self.is_2d else -3
        x = F.interpolate(x, size=y.size()[dims:], mode=mode, align_corners=True)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x


class Encoder(Module):
    """Encoding part of the U-Net model.
    It returns the features map for the skip connections

    Args:
        in_channel (int): Number of channels in the input image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.
        n_filters (int):  Number of base filters in the U-Net.

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        down_path (ModuleList): List of module operations done during encoding.
        conv_bottom (DownConv): Bottom down convolution.
        film_bottom (FiLMlayer): FiLM layer applied to bottom convolution.
    """

    def __init__(self, in_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1, n_metadata=None, film_layers=None,
                 is_2d=True, n_filters=64):
        super(Encoder, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        # first block
        self.down_path.append(DownConv(in_channel, n_filters, dropout_rate, bn_momentum, is_2d))
        self.down_path.append(FiLMlayer(n_metadata, n_filters) if film_layers and film_layers[0] else None)
        max_pool = nn.MaxPool2d if is_2d else nn.MaxPool3d
        self.down_path.append(max_pool(2))

        for i in range(depth - 1):
            self.down_path.append(DownConv(n_filters, n_filters * 2, dropout_rate, bn_momentum, is_2d))
            self.down_path.append(FiLMlayer(n_metadata, n_filters * 2) if film_layers and film_layers[i + 1] else None)
            self.down_path.append(max_pool(2))
            n_filters = n_filters * 2

        # Bottom
        self.conv_bottom = DownConv(n_filters, n_filters, dropout_rate, bn_momentum, is_2d)
        self.film_bottom = FiLMlayer(n_metadata, n_filters) if film_layers and film_layers[self.depth] else None

    def forward(self, x, context=None):
        features = []

        # First block
        x = self.down_path[0](x)
        if self.down_path[1]:
            x, w_film = self.down_path[1](x, context, None)
        features.append(x)
        x = self.down_path[2](x)

        # Down-sampling path (other blocks)
        for i in range(1, self.depth):
            x = self.down_path[i * 3](x)
            if self.down_path[i * 3 + 1]:
                x, w_film = self.down_path[i * 3 + 1](x, context, None if 'w_film' not in locals() else w_film)
            features.append(x)
            x = self.down_path[i * 3 + 2](x)

        # Bottom level
        x = self.conv_bottom(x)
        if self.film_bottom:
            x, w_film = self.film_bottom(x, context, None if 'w_film' not in locals() else w_film)
        features.append(x)

        return features, None if 'w_film' not in locals() else w_film


class Decoder(Module):
    """Decoding part of the U-Net model.

    Args:
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        hemis (bool): Boolean indicating if HeMIS is on or not.
        final_activation (str): Choice of final activation between "sigmoid", "relu" and "softmax".
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.
        n_filters (int):  Number of base filters in the U-Net.

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        out_channel (int): Number of channels in the output image.
        up_path (ModuleList): List of module operations done during decoding.
        last_conv (Conv2d): Last convolution.
        last_film (FiLMlayer): FiLM layer applied to last convolution.
        softmax (Softmax): Softmax layer that can be applied as last layer.
    """

    def __init__(self, out_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1,
                 n_metadata=None, film_layers=None, hemis=False, final_activation="sigmoid", is_2d=True, n_filters=64):
        super(Decoder, self).__init__()
        self.depth = depth
        self.out_channel = out_channel
        self.final_activation = final_activation
        # Up-Sampling path
        self.up_path = nn.ModuleList()
        if hemis:
            in_channel = n_filters * 2 ** self.depth
            self.up_path.append(UpConv(in_channel * 2, n_filters * 2 ** (self.depth - 1), dropout_rate, bn_momentum,
                                       is_2d))
            if film_layers and film_layers[self.depth + 1]:
                self.up_path.append(FiLMlayer(n_metadata, n_filters * 2 ** (self.depth - 1)))
            else:
                self.up_path.append(None)
            # self.depth += 1
        else:
            in_channel = n_filters * 2 ** self.depth

            self.up_path.append(UpConv(in_channel, n_filters * 2 ** (self.depth - 1), dropout_rate, bn_momentum, is_2d))
            if film_layers and film_layers[self.depth + 1]:
                self.up_path.append(FiLMlayer(n_metadata, n_filters * 2 ** (self.depth - 1)))
            else:
                self.up_path.append(None)

        for i in range(1, depth):
            in_channel //= 2

            self.up_path.append(
                UpConv(in_channel + n_filters * 2 ** (self.depth - i - 1 + int(hemis)), n_filters * 2 ** (self.depth - i - 1),
                       dropout_rate, bn_momentum, is_2d))
            if film_layers and film_layers[self.depth + i + 1]:
                self.up_path.append(FiLMlayer(n_metadata, n_filters * 2 ** (self.depth - i - 1)))
            else:
                self.up_path.append(None)

        # Last Convolution
        conv = nn.Conv2d if is_2d else nn.Conv3d
        self.last_conv = conv(in_channel // 2, out_channel, kernel_size=3, padding=1)
        self.last_film = FiLMlayer(n_metadata, self.out_channel) if film_layers and film_layers[-1] else None
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, context=None, w_film=None):
        x = features[-1]

        for i in reversed(range(self.depth)):
            x = self.up_path[-(i + 1) * 2](x, features[i])
            if self.up_path[-(i + 1) * 2 + 1]:
                x, w_film = self.up_path[-(i + 1) * 2 + 1](x, context, w_film)

        # Last convolution
        x = self.last_conv(x)
        if self.last_film:
            x, w_film = self.last_film(x, context, w_film)

        if hasattr(self, "final_activation") and self.final_activation not in ["softmax", "relu", "sigmoid"]:
            raise ValueError("final_activation value has to be either softmax, relu, or sigmoid")
        elif hasattr(self, "final_activation") and self.final_activation == "softmax":
            preds = self.softmax(x)
        elif hasattr(self, "final_activation") and self.final_activation == "relu":
            preds = nn.ReLU()(x) / nn.ReLU()(x).max()
            # If nn.ReLU()(x).max()==0, then nn.ReLU()(x) will also ==0. So, here any zero division will always be 0/0.
            # For float values, 0/0=nan. So, we can handle zero division (without checking data!) by setting nans to 0.
            preds[torch.isnan(preds)] = 0.
            # If model multiclass
            if self.out_channel > 1:
                class_sum = preds.sum(dim=1).unsqueeze(1)
                # Avoid division by zero
                class_sum[class_sum == 0] = 1
                preds /= class_sum
        else:
            preds = torch.sigmoid(x)

        # If model multiclass
        if self.out_channel > 1:
            # Remove background class
            preds = preds[:, 1:, ]

        return preds


class Unet(Module):
    """A reference U-Net model.

    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597

    Args:
        in_channel (int): Number of channels in the input image.
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        final_activation (str): Choice of final activation between "sigmoid", "relu" and "softmax".
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.
        n_filters (int):  Number of base filters in the U-Net.
        **kwargs:

    Attributes:
        encoder (Encoder): U-Net encoder.
        decoder (Decoder): U-net decoder.
    """

    def __init__(self, in_channel=1, out_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1, final_activation='sigmoid',
                 is_2d=True, n_filters=64, **kwargs):
        super(Unet, self).__init__()

        # Encoder path
        self.encoder = Encoder(in_channel=in_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                               is_2d=is_2d, n_filters=n_filters)

        # Decoder path
        self.decoder = Decoder(out_channel=out_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                               final_activation=final_activation, is_2d=is_2d, n_filters=n_filters)

    def forward(self, x):
        features, _ = self.encoder(x)
        preds = self.decoder(features)

        return preds


class FiLMedUnet(Unet):
    """U-Net network containing FiLM layers to condition the model with another data type (i.e. not an image).

    Args:
        n_channel (int): Number of channels in the input image.
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        is_2d (bool): Indicates dimensionality of model.
        n_filters (int):  Number of base filters in the U-Net.
        **kwargs:

    Attributes:
        encoder (Encoder): U-Net encoder.
        decoder (Decoder): U-net decoder.
    """

    def __init__(self, in_channel=1, out_channel=1, depth=3, dropout_rate=0.3,
                 bn_momentum=0.1, n_metadata=None, film_layers=None, is_2d=True, n_filters=64, **kwargs):
        super().__init__(in_channel=in_channel, out_channel=out_channel, depth=depth,
                         dropout_rate=dropout_rate, bn_momentum=bn_momentum)

        # Verify if the length of boolean FiLM layers corresponds to the depth
        if film_layers:
            if len(film_layers) != 2 * depth + 2:
                raise ValueError("The number of FiLM layers {} entered does not correspond to the "
                                 "UNet depth. There should 2 * depth + 2 layers.".format(len(film_layers)))
        else:
            film_layers = [0] * (2 * depth + 2)
        # Encoder path
        self.encoder = Encoder(in_channel=in_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                               n_metadata=n_metadata, film_layers=film_layers, is_2d=is_2d, n_filters=n_filters)
        # Decoder path
        self.decoder = Decoder(out_channel=out_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                               n_metadata=n_metadata, film_layers=film_layers, is_2d=is_2d, n_filters=n_filters)

    def forward(self, x, context=None):
        features, w_film = self.encoder(x, context)
        preds = self.decoder(features, context, w_film)

        return preds


class FiLMgenerator(Module):
    """The FiLM generator processes the conditioning information
    and produces parameters that describe how the target network should alter its computation.

    Here, the FiLM generator is a multi-layer perceptron.

    Args:
        n_features (int): Number of input channels.
        n_channels (int): Number of output channels.
        n_hid (int): Number of hidden units in layer.

    Attributes:
        linear1 (Linear): Input linear layer.
        sig (Sigmoid): Sigmoid function.
        linear2 (Linear): Hidden linear layer.
        linear3 (Linear): Output linear layer.
    """

    def __init__(self, n_features, n_channels, n_hid=64):
        super(FiLMgenerator, self).__init__()
        self.linear1 = nn.Linear(n_features, n_hid)
        self.sig = nn.Sigmoid()
        self.linear2 = nn.Linear(n_hid, n_hid // 4)
        self.linear3 = nn.Linear(n_hid // 4, n_channels * 2)

    def forward(self, x, shared_weights=None):
        if shared_weights is not None:  # weight sharing
            self.linear1.weight = shared_weights[0]
            self.linear2.weight = shared_weights[1]

        x = self.linear1(x)
        x = self.sig(x)
        x = self.linear2(x)
        x = self.sig(x)
        x = self.linear3(x)

        out = self.sig(x)
        return out, [self.linear1.weight, self.linear2.weight]


class FiLMlayer(Module):
    """Applies Feature-wise Linear Modulation to the incoming data
    .. seealso::
        Perez, Ethan, et al. "Film: Visual reasoning with a general conditioning layer."
        Thirty-Second AAAI Conference on Artificial Intelligence. 2018.

    Args:
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        n_channels (int): Number of output channels.

    Attributes:
        batch_size (int): Batch size.
        height (int): Image height.
        width (int): Image width.
        feature_size (int): Number of features in data.
        generator (FiLMgenerator): FiLM network.
        gammas (float): Multiplicative term of the FiLM linear modulation.
        betas (float): Additive term of the FiLM linear modulation.
    """

    def __init__(self, n_metadata, n_channels):
        super(FiLMlayer, self).__init__()

        self.batch_size = None
        self.height = None
        self.width = None
        self.depth = None
        self.feature_size = None
        self.generator = FiLMgenerator(n_metadata, n_channels)
        # Add the parameters gammas and betas to access them out of the class.
        self.gammas = None
        self.betas = None

    def forward(self, feature_maps, context, w_shared):
        data_shape = feature_maps.data.shape
        if len(data_shape) == 4:
            _, self.feature_size, self.height, self.width = data_shape
        elif len(data_shape) == 5:
            _, self.feature_size, self.height, self.width, self.depth = data_shape
        else:
            raise ValueError("Data should be either 2D (tensor length: 4) or 3D (tensor length: 5), found shape: {}".format(data_shape))

        if torch.cuda.is_available():
            context = torch.Tensor(context).cuda()
        else:
            context = torch.Tensor(context)

        # Estimate the FiLM parameters using a FiLM generator from the contioning metadata
        film_params, new_w_shared = self.generator(context, w_shared)

        # FiLM applies a different affine transformation to each channel,
        # consistent accross spatial locations
        if len(data_shape) == 4:
            film_params = film_params.unsqueeze(-1).unsqueeze(-1)
            film_params = film_params.repeat(1, 1, self.height, self.width)
        else:
            film_params = film_params.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            film_params = film_params.repeat(1, 1, self.height, self.width, self.depth)

        self.gammas = film_params[:, :self.feature_size, ]
        self.betas = film_params[:, self.feature_size:, ]

        # Apply the linear modulation
        output = self.gammas * feature_maps + self.betas

        return output, new_w_shared


class HeMISUnet(Module):
    """A U-Net model inspired by HeMIS to deal with missing contrasts.
        1) It has as many encoders as contrasts but only one decoder.
        2) Skip connections are the concatenations of the means and var of all encoders skip connections.

        Param:
        contrasts: list of all the possible contrasts. ['T1', 'T2', 'T2S', 'F']

    .. seealso::
        Havaei, M., Guizard, N., Chapados, N., Bengio, Y.:
        Hemis: Hetero-modal image segmentation.
        ArXiv link: https://arxiv.org/abs/1607.05194

        Reuben Dorent and Samuel Joutard and Marc Modat and Sébastien Ourselin and Tom Vercauteren
        Hetero-Modal Variational Encoder-Decoder for Joint Modality Completion and Segmentation
        ArXiv link: https://arxiv.org/abs/1907.11150

    Args:
        contrasts (list): List of contrasts.
        out_channel (int): Number of output channels.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        **kwargs:

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        contrasts (list): List of contrasts.
        Encoder_mod (ModuleDict): Contains encoder for each modality.
        decoder (Decoder): U-Net decoder.
    """

    def __init__(self, contrasts, out_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1, **kwargs):
        super(HeMISUnet, self).__init__()
        self.depth = depth
        self.contrasts = contrasts

        # Encoder path
        self.Encoder_mod = nn.ModuleDict(
            [['Encoder_{}'.format(Mod), Encoder(in_channel=1, depth=depth, dropout_rate=dropout_rate,
                                                bn_momentum=bn_momentum)] for Mod in self.contrasts])

        # Decoder path
        self.decoder = Decoder(out_channel=out_channel, depth=depth, dropout_rate=dropout_rate,
                               bn_momentum=bn_momentum, hemis=True)

    def forward(self, x_mods, indexes_mod):
        """
        X is list like X = [x_T1, x_T2, x_T2S, x_F]
        indexes_mod: list of arrays like [[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0]]
        N.B. len(list) = number of contrasts.
        len(list[i]) = Batch size
        """
        features_mod = [[] for _ in range(self.depth + 1)]

        # Down-sampling
        for i, Mod in enumerate(self.contrasts):
            features, _ = self.Encoder_mod['Encoder_{}'.format(Mod)](x_mods[i])

            for j in range(self.depth + 1):
                features_mod[j].append(features[j].unsqueeze(0))

        # Abstraction
        for j in range(self.depth + 1):
            features_cat = torch.cat(features_mod[j], 0).transpose(0, 1)

            features_mod[j] = torch.cat([torch.cat([features_cat[i][indexes_mod[i]].squeeze(1).mean(0),
                                                    features_cat[i][indexes_mod[i]].squeeze(1).var(0)], 0).unsqueeze(0)
                                         for i in range(len(indexes_mod))], 0)

        # Up-sampling
        preds = self.decoder(features_mod)

        return preds


class Modified3DUNet(nn.Module):
    """Code from the following repository:
    https://github.com/pykao/Modified-3D-UNet-Pytorch
    The main differences with the original UNet resides in the use of LeakyReLU instead of ReLU, InstanceNormalisation
    instead of BatchNorm due to small batch size in 3D and the addition of segmentation layers in the decoder.

    If attention=True, attention gates are added in the decoder to help focus attention on important features for a
    given task. Code related to the attentions gates is inspired from:
    https://github.com/ozan-oktay/Attention-Gated-Networks

    Args:
        in_channel (int): Number of channels in the input image.
        out_channel (int): Number of channels in the output image.
        n_filters (int): Number of base filters in the U-Net.
        attention (bool): Boolean indicating whether the attention module is on or not.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        final_activation (str): Choice of final activation between "sigmoid", "relu" and "softmax".
        **kwargs:

    Attributes:
        in_channels (int): Number of channels in the input image.
        n_classes (int): Number of channels in the output image.
        base_n_filter (int): Number of base filters in the U-Net.
        attention (bool): Boolean indicating whether the attention module is on or not.
        momentum (float): Batch normalization momentum.
        final_activation (str): Choice of final activation between "sigmoid", "relu" and "softmax".

    Note: All layers are defined as attributes and used in the forward method.
    """

    def __init__(self, in_channel, out_channel, n_filters=16, attention=False, dropout_rate=0.3, bn_momentum=0.1,
                 final_activation="sigmoid", n_metadata=None, film_layers=None, **kwargs):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channel
        self.n_classes = out_channel
        self.base_n_filter = n_filters
        self.attention = attention
        self.momentum = bn_momentum
        self.final_activation = final_activation

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=dropout_rate)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(
            self.in_channels, self.base_n_filter,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.film_layer1 = FiLMlayer(n_metadata, self.base_n_filter) if film_layers and film_layers[0] else None
        self.conv3d_c1_2 = nn.Conv3d(
            self.base_n_filter, self.base_n_filter,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.lrelu_conv_c1 = self.lrelu_conv(
            self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter, momentum=self.momentum)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(
            self.base_n_filter, self.base_n_filter * 2,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.film_layer2 = FiLMlayer(n_metadata, self.base_n_filter * 2) if film_layers and film_layers[1] else None
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(
            self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2, momentum=self.momentum)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(
            self.base_n_filter * 2, self.base_n_filter * 4,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.film_layer3 = FiLMlayer(n_metadata, self.base_n_filter * 4) if film_layers and film_layers[2] else None
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(
            self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4, momentum=self.momentum)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(
            self.base_n_filter * 4, self.base_n_filter * 8,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.film_layer4 = FiLMlayer(n_metadata, self.base_n_filter * 8) if film_layers and film_layers[3] else None
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(
            self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8, momentum=self.momentum)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(
            self.base_n_filter * 8, self.base_n_filter * 16,
            kernel_size=3, stride=2, padding=1, bias=False
        )

        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(
            self.base_n_filter * 16, self.base_n_filter * 16)

        if film_layers and film_layers[4]:
            self.norm_lrelu_0 = self.norm_lrelu(self.base_n_filter * 16)
            self.film_layer5 = FiLMlayer(n_metadata, self.base_n_filter * 16)
            self.upscale_conv_norm_lrelu_0 = self.upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                          self.base_n_filter * 8)
        else:
            self.norm_lrelu_upscale_conv_norm_lrelu_l0 = \
                self.norm_lrelu_upscale_conv_norm_lrelu(
                    self.base_n_filter * 16, self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(
            self.base_n_filter * 8, self.base_n_filter * 8,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.film_layer6 = FiLMlayer(n_metadata, self.base_n_filter * 8) if film_layers and film_layers[5] else None
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8, momentum=self.momentum)

        # Attention UNet
        if self.attention:
            self.gating = UnetGridGatingSignal3(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=(1, 1, 1),
                                                is_batchnorm=True)

            # attention blocks
            self.attentionblock2 = GridAttentionBlockND(in_channels=self.base_n_filter * 2,
                                                        gating_channels=self.base_n_filter * 8,
                                                        inter_channels=self.base_n_filter * 2,
                                                        sub_sample_factor=(2, 2, 2),
                                                        )
            self.attentionblock3 = GridAttentionBlockND(in_channels=self.base_n_filter * 4,
                                                        gating_channels=self.base_n_filter * 8,
                                                        inter_channels=self.base_n_filter * 4,
                                                        sub_sample_factor=(2, 2, 2),
                                                        )
            self.attentionblock4 = GridAttentionBlockND(in_channels=self.base_n_filter * 8,
                                                        gating_channels=self.base_n_filter * 8,
                                                        inter_channels=self.base_n_filter * 8,
                                                        sub_sample_factor=(2, 2, 2),
                                                        )
            self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 16, momentum=self.momentum)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(
            self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(
            self.base_n_filter * 16, self.base_n_filter * 8,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.film_layer7 = FiLMlayer(n_metadata, self.base_n_filter * 4) if film_layers and film_layers[6] else None
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = \
            self.norm_lrelu_upscale_conv_norm_lrelu(
                self.base_n_filter * 8, self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(
            self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(
            self.base_n_filter * 8, self.base_n_filter * 4,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.film_layer8 = FiLMlayer(n_metadata, self.base_n_filter * 2) if film_layers and film_layers[7] else None
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = \
            self.norm_lrelu_upscale_conv_norm_lrelu(
                self.base_n_filter * 4, self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(
            self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(
            self.base_n_filter * 4, self.base_n_filter * 2,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.film_layer9 = FiLMlayer(n_metadata, self.base_n_filter) if film_layers and film_layers[8] else None
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = \
            self.norm_lrelu_upscale_conv_norm_lrelu(
                self.base_n_filter * 2, self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(
            self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(
            self.base_n_filter * 2, self.n_classes,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        # self.film_layer10 = FiLMlayer(n_metadata, ) if film_layers and film_layers[9] else None

        self.ds2_1x1_conv3d = nn.Conv3d(
            self.base_n_filter * 8, self.n_classes,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.ds3_1x1_conv3d = nn.Conv3d(
            self.base_n_filter * 4, self.n_classes,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.film_layer10 = FiLMlayer(n_metadata, self.n_classes) if film_layers and film_layers[9] else None

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out, momentum=self.momentum),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in, momentum=self.momentum),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out,
                      kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out,
                      kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in, momentum=self.momentum),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out, momentum=self.momentum),
            nn.LeakyReLU())

    def norm_lrelu(self, feat_in):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in, momentum=self.momentum),
            nn.LeakyReLU())

    def upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out, momentum=self.momentum),
            nn.LeakyReLU())

    def forward(self, x, context=None, w_film=None):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out

        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        out = self.lrelu(out)
        context_1 = out
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)
        if hasattr(self, 'film_layer1') and self.film_layer1:
            out, w_film = self.film_layer1(out, context, w_film)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        if hasattr(self, 'film_layer2') and self.film_layer2:
            out, w_film = self.film_layer2(out, context, w_film)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        if hasattr(self, 'film_layer3') and self.film_layer3:
            out, w_film = self.film_layer3(out, context, w_film)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        if hasattr(self, 'film_layer4') and self.film_layer4:
            out, w_film = self.film_layer4(out, context, w_film)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5

        if self.attention:
            out = self.inorm3d_l0(out)
            out = self.lrelu(out)

            gating = self.gating(out)
            context_4, att4 = self.attentionblock4(context_4, gating)
            context_3, att3 = self.attentionblock3(context_3, gating)
            context_2, att2 = self.attentionblock2(context_2, gating)

        if hasattr(self, 'film_layer5') and self.film_layer5:
            out = self.norm_lrelu_0(out)
            out, w_film = self.film_layer5(out, context, w_film)
            out = self.upscale_conv_norm_lrelu_0(out)
        else:
            out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)

        out = self.inorm3d_l0(out)
        out = self.lrelu(out)
        if hasattr(self, 'film_layer6') and self.film_layer6:
            out, w_film = self.film_layer6(out, context, w_film)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)
        if hasattr(self, 'film_layer7') and self.film_layer7:
            out, w_film = self.film_layer7(out, context, w_film)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)
        if hasattr(self, 'film_layer8') and self.film_layer8:
            out, w_film = self.film_layer8(out, context, w_film)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        if hasattr(self, 'film_layer9') and self.film_layer9:
            out, w_film = self.film_layer9(out, context, w_film)

        # Level 4 localization pathway
        out = torch.cat([context_1, out], dim=1)
        out = self.conv_norm_lrelu_l4(out)

        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(
            ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        if hasattr(self, 'film_layer10') and self.film_layer10:
            out, w_film = self.film_layer10(out, context, w_film)
        seg_layer = out

        if hasattr(self, "final_activation") and self.final_activation not in ["softmax", "relu", "sigmoid"]:
            raise ValueError("final_activation value has to be either softmax, relu, or sigmoid")
        elif hasattr(self, "final_activation") and self.final_activation == "softmax":
            out = self.softmax(out)
        elif hasattr(self, "final_activation") and self.final_activation == "relu":
            out = nn.ReLU()(seg_layer) / nn.ReLU()(seg_layer).max() if bool(nn.ReLU()(seg_layer).max()) \
                else nn.ReLU()(seg_layer)
        else:
            out = torch.sigmoid(out)

        if self.n_classes > 1:
            # Remove background class
            out = out[:, 1:, ]

        return out


class UNet3D(Modified3DUNet):
    """To ensure retrocompatibility, when calling UNet3D (old model name), Modified3DUNet will be called.
    see Modified3DUNet to learn more about parameters.
    """

    def __init__(self, in_channel, out_channel, n_filters=16, attention=False, dropout_rate=0.3, bn_momentum=0.1,
                 final_activation="sigmoid", n_metadata=None, film_layers=None, **kwargs):
        super(UNet3D, self).__init__()
        Modified3DUNet(in_channel=in_channel, out_channel=out_channel, n_filters=n_filters, attention=attention,
                       dropout_rate=dropout_rate, bn_momentum=bn_momentum, final_activation=final_activation,
                       n_metadata=n_metadata, film_layers=film_layers, **kwargs)


class GridAttentionBlockND(nn.Module):
    """Attention module to focus on important features passed through U-Net's decoder
    Specific to Attention UNet

    .. seealso::
        Oktay, Ozan, et al. "Attention u-net: Learning where to look for the pancreas."
        arXiv preprint arXiv:1804.03999 (2018).

    Args:
        in_channels (int): Number of channels in the input image.
        gating_channels (int): Number of channels in the gating step.
        inter_channels (int): Number of channels in the intermediate gating step.
        dimension (int): Value of 2 or 3 to indicating whether it is used in a 2D or 3D model.
        sub_sample_factor (tuple or list): Convolution kernel size.

    Attributes:
        in_channels (int): Number of channels in the input image.
        gating_channels (int): Number of channels in the gating step.
        inter_channels (int): Number of channels in the intermediate gating step.
        dimension (int): Value of 2 or 3 to indicating whether it is used in a 2D or 3D model.
        sub_sample_factor (tuple or list): Convolution kernel size.
        upsample_mode (str): 'bilinear' or 'trilinear' related to the use of 2D or 3D models.
        W (Sequential): Sequence of convolution and batch normalization layers.
        theta (Conv2d or Conv3d): Convolution layer for gating operation.
        phi (Conv2d or Conv3d): Convolution layer for gating operation.
        psi (Conv2d or Conv3d): Convolution layer for gating operation.

    """

    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3,
                 sub_sample_factor=(2, 2, 2)):
        super(GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplementedError

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                             bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

        # Initialise weights
        for m in self.children():
            m.apply(weights_init_kaiming)

        # Define the operation
        self.operation_function = self._concatenation

    def forward(self, x, g):
        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=True)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


def weights_init_kaiming(m):
    """Initialize weights according to method describe here:
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class UnetGridGatingSignal3(nn.Module):
    """Operation to extract important features for a specific task using 1x1x1 convolution (Gating) which is used in the
    attention blocks.

    Args:
        in_size (int): Number of channels in the input image.
        out_size (int): Number of channels in the output image.
        kernel_size (tuple): Convolution kernel size.
        is_batchnorm (bool): Boolean indicating whether to apply batch normalization or not.

    Attributes:
        conv1 (Sequential): 3D convolution, batch normalization and ReLU activation.
    """

    def __init__(self, in_size, out_size, kernel_size=(1, 1, 1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            weights_init_kaiming(m)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3, stride=1, pad=0, activation=nn.LeakyReLU()):
        """
        Perform convolution, activation and batch normalization.
        Args:
            in_chan (int): number of channels on input
            out_chan (int): number of channel on output
            ksize (int): size of kernel for the 2d convolution
            stride (int): strides for 2d convolution
            pad (int): pad for nn.conv2d
            activation (nn.layers): activation layer. default Leaky ReLu
        """
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride, padding=pad)
        self.activation = activation
        self.batch_norm = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv1(x)))


class SimpleBlock(nn.Module):
    def __init__(self, in_chan, out_chan_1x1, out_chan_3x3, activation=nn.LeakyReLU()):
        """
        Inception module with 3 convolutions with different kernel size with separate activation.
        The 3 outputs are then concatenated. Max pooling performed on concatenation.
        Args:
            in_chan (int): number of channel of input
            out_chan_1x1 (int): number of channel after first convolution block
            out_chan_3x3 (int): number of channel for the other convolution blocks
            activation (nn.layers): activation layer used in convolution block
        """
        super(SimpleBlock, self).__init__()
        self.conv1 = ConvBlock(in_chan, out_chan_1x1, ksize=3, pad=0, activation=activation)
        self.conv2 = ConvBlock(in_chan, out_chan_3x3, ksize=5, pad=1, activation=activation)
        self.conv3 = ConvBlock(in_chan, out_chan_3x3, ksize=9, pad=3, activation=activation)
        self.MP = nn.MaxPool2d(1)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv3_out = self.conv3(x)
        output = torch.cat([conv1_out, conv2_out, conv3_out], 1)
        output = self.MP(output)
        return output


class Countception(Module):
    """Countception model.
    Fully convolutional model using inception module and used for keypoints detection.
    The inception model extracts several patches within each image. Every pixel is therefore processed by the
    network several times, allowing to average multiple predictions and minimize false negatives.

    .. seealso::
        Cohen JP et al. "Count-ception: Counting by fully convolutional redundant counting."
        Proceedings of the IEEE International Conference on Computer Vision Workshops. 2017.

    Args:
        in_channel (int): number of channels on input image
        out_channel (int): number of channels on output image
        use_logits (bool): boolean to change output
        logits_per_output (int): number of outputs of final convolution which will multiplied by the number of channels
        name (str): model's name used for call in configuration file.
    """

    def __init__(self, in_channel=3, out_channel=1, use_logits=False, logits_per_output=12, name='CC', **kwargs):
        super(Countception, self).__init__()

        # params
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.activation = nn.LeakyReLU(0.01)
        self.final_activation = nn.LeakyReLU(0.3)
        self.patch_size = 40
        self.use_logits = use_logits
        self.logits_per_output = logits_per_output

        torch.LongTensor()

        self.conv1 = ConvBlock(self.in_channel, 64, ksize=3, pad=self.patch_size, activation=self.activation)
        self.simple1 = SimpleBlock(64, 16, 16, activation=self.activation)
        self.simple2 = SimpleBlock(48, 16, 32, activation=self.activation)
        self.conv2 = ConvBlock(80, 16, ksize=14, activation=self.activation)
        self.simple3 = SimpleBlock(16, 112, 48, activation=self.activation)
        self.simple4 = SimpleBlock(208, 64, 32, activation=self.activation)
        self.simple5 = SimpleBlock(128, 40, 40, activation=self.activation)
        self.simple6 = SimpleBlock(120, 32, 96, activation=self.activation)
        self.conv3 = ConvBlock(224, 32, ksize=20, activation=self.activation)
        self.conv4 = ConvBlock(32, 64, ksize=10, activation=self.activation)
        self.conv5 = ConvBlock(64, 32, ksize=9, activation=self.activation)

        if use_logits:
            self.conv6 = nn.ModuleList([ConvBlock(
                64, logits_per_output, ksize=1, activation=self.final_activation) for _ in range(out_channel)])
        else:
            self.conv6 = ConvBlock(32, self.out_channel, ksize=20, pad=1, activation=self.final_activation)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('leaky_relu', param=0.01))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        net = self.conv1(x)  # 32
        net = self.simple1(net)
        net = self.simple2(net)
        net = self.conv2(net)
        net = self.simple3(net)
        net = self.simple4(net)
        net = self.simple5(net)
        net = self.simple6(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.conv5(net)

        if self.use_logits:
            net = [c(net) for c in self.conv6]
            [self._print(n) for n in net]
        else:
            net = self.conv6(net)

        return net


def set_model_for_retrain(model_path, retrain_fraction, map_location, reset=True):
    """Set model for transfer learning.

    The first layers (defined by 1-retrain_fraction) are frozen (i.e. requires_grad=False).
    The weights of the last layers (defined by retrain_fraction) are reset unless reset option is False.

    Args:
        model_path (str): Pretrained model path.
        retrain_fraction (float): Fraction of the model that will be retrained, between 0 and 1. If set to 0.3,
            then the 30% last fraction of the model will be re-initalised and retrained.
        map_location (str): Device.
        reset (bool): if the un-frozen weight should be reset or kept as loaded.

    Returns:
        torch.Module: Model ready for retrain.
    """
    # Load pretrained model
    model = torch_load(model_path, map_location=map_location)
    # Get number of layers with learnt parameters
    layer_names = [name for name, layer in model.named_modules() if hasattr(layer, 'reset_parameters')]
    n_layers = len(layer_names)
    # Compute the number of these layers we want to freeze
    n_freeze = int(round(n_layers * (1 - retrain_fraction)))
    # Last frozen layer
    last_frozen_layer = layer_names[n_freeze]

    # Set freeze first layers
    for name, layer in model.named_parameters():
        if not name.startswith(last_frozen_layer):
            layer.requires_grad = False
        else:
            break

    # Reset weights of the last layers
    if reset:
        for name, layer in model.named_modules():
            if name in layer_names[n_freeze:]:
                layer.reset_parameters()

    return model


def get_model_filenames(folder_model):
    """Get trained model filenames from its folder path.

    This function checks if the folder_model exists and get trained model path (.pt or .onnx based on
    model and GPU availability) and its configuration file (.json) from it.

    Args:
        folder_name (str): Path of the model folder.

    Returns:
        str, str: Paths of the model (.pt or .onnx) and its configuration file (.json).
    """
    if Path(folder_model).is_dir():
        prefix_model = Path(folder_model).name
        fname_model_onnx = Path(folder_model, prefix_model + '.onnx')
        fname_model_pt = Path(folder_model, prefix_model + '.pt')
        cuda_available = torch.cuda.is_available()

        # Assign '.pt' or '.onnx' model based on file existence and GPU/CPU device availability
        if not fname_model_pt.is_file() and not fname_model_onnx.is_file():
            raise FileNotFoundError(f"Model files not found in model folder: "
                                    f"'{str(fname_model_onnx)}' or '{str(fname_model_pt)}'")
        # '.pt' is preferred on GPU, or on CPU if '.onnx' doesn't exist
        elif ((cuda_available and fname_model_pt.is_file()) or
              (not cuda_available and not fname_model_onnx.is_file())):
            fname_model = fname_model_pt
        # '.onnx' is preferred on CPU, or on GPU if '.pt' doesn't exist
        elif ((not cuda_available and fname_model_onnx.is_file()) or
              (cuda_available and not fname_model_pt.is_file())):
            fname_model = fname_model_onnx

        fname_model_metadata = Path(folder_model, prefix_model + '.json')
        if not fname_model_metadata.is_file():
            raise FileNotFoundError(f"Model config file not found in model folder: '{str(fname_model_metadata)}'")
    else:
        raise FileNotFoundError(folder_model)

    return str(fname_model), str(fname_model_metadata)
