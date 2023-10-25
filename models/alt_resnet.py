from jax import numpy as jnp
import jax
import equinox as eqx
from equinox import nn as nn
from jax import random as jr
from typing import Optional, Callable, Any, List, Type, Union
from jax import tree_util as jtu
from models.batch_norm import StandardBatchNorm


def tree_split(key: jr.PRNGKey, tree, is_leaf: Optional[Callable] = None):
    leaves, treedef = jtu.tree_flatten(tree, is_leaf=is_leaf)
    key_leaves = jr.split(key, len(leaves))
    key_tree = jtu.tree_unflatten(treedef, key_leaves)
    return key_tree


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    *,
    key: jr.PRNGKey,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        use_bias=False,
        dilation=dilation,
        key=key,
    )


def conv1x1(
    in_planes: int, out_planes: int, stride: int = 1, *, key: jr.PRNGKey
) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, use_bias=False, key=key
    )


class BasicBlock(nn.StatefulLayer):
    expansion: int = eqx.field(default=1, static=True)
    conv1: nn.Conv2d
    bn1: eqx.Module
    conv2: nn.Conv2d
    bn2: eqx.Module
    stride: int = eqx.field(static=True)
    shortcut: Union[callable, eqx.Module]

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., eqx.Module]] = None,
        *,
        key=jr.PRNGKey,
    ) -> None:
        # if norm_layer is None:
        #     norm_layer = lambda *args, **kwargs: nn.BatchNorm(
        #         *args, axis_name="batch", **kwargs
        #     )
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        keys = jr.split(key, 3)
        self.conv1 = conv3x3(in_planes, planes, stride, key=keys[0])
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, key=keys[1])
        self.bn2 = norm_layer(planes)
        self.stride = stride

        shortcut = lambda x, state: (x, state)
        if stride != 1 or in_planes != self.expansion * planes:
            shortcut = nn.Sequential(
                [
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        use_bias=False,
                        key=keys[3],
                    ),
                    norm_layer(self.expansion * planes),
                ]
            )
        self.shortcut = shortcut

    def __call__(
        self, x: jax.Array, state: nn.State, *, key: Optional[jr.PRNGKey] = None
    ) -> jax.Array:
        identity, state = self.shortcut(x, state)

        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = jax.nn.relu(out)

        out = self.conv2(out)
        out, state = self.bn2(out, state)

        out += identity
        out = jax.nn.relu(out)

        return out, state


class Bottleneck(nn.StatefulLayer):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    conv1: nn.Conv2d
    bn: eqx.Module
    conv2: nn.Conv2d
    conv3: nn.Conv2d
    bn3: eqx.Module
    downsample: Optional[eqx.Module]
    stride: int = eqx.field(static=True, default=1)
    expansion: int = eqx.field(default=4, static=True)

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.StatefulLayer] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., eqx.Module]] = None,
        expansion: int = 4,
        *,
        key=jr.PRNGKey,
    ) -> None:
        keys = jr.split(key, 3)
        if norm_layer is None:
            norm_layer = lambda *args, **kwargs: nn.BatchNorm(
                *args, axis_name="batch", **kwargs
            )
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_planes, width, keys[0])
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, keys[1])
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * expansion, keys[3])
        self.bn3 = norm_layer(planes * expansion)
        self.downsample = downsample
        self.stride = stride
        self.expansion = expansion

    def __call__(
        self, x: jax.Array, state: nn.State, *, key: Optional[jr.PRNGKey] = None
    ) -> jax.Array:
        identity = x

        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = self.relu(out)

        out = self.conv2(out)
        out, state = self.bn2(out, state)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity, state = self.downsample(x, state)

        out += identity
        out = jax.nn.relu(out)

        return out, state


def re_initialize_layer(layer: eqx.Module, zero_init_residual: bool, key: jr.PRNGKey):
    is_conv = lambda n: isinstance(n, nn.Conv2d)
    conv_keys = tree_split(key, tree=layer, is_leaf=is_conv)
    layer = jtu.tree_map(
        re_initialize_conv,
        layer,
        conv_keys,
        is_leaf=is_conv,
    )

    if zero_init_residual:
        is_block = lambda n: isinstance(n, Bottleneck) or isinstance(n, BasicBlock)
        layer = jtu.tree_map(
            re_initialize_bn,
            layer,
            is_leaf=is_block,
        )
    return layer


def kaiming_normal(w: jax.Array, key: jr.PRNGKey):
    initializer = jax.nn.initializers.he_normal(in_axis=1, out_axis=0)
    return initializer(key, w.shape, w.dtype)


def re_initialize_conv(module, key):
    if isinstance(module, nn.Conv2d):
        module = eqx.tree_at(
            where=lambda m: m.weight,
            pytree=module,
            replace_fn=lambda w: kaiming_normal(w, key),
        )
    return module


def re_initialize_bn(module):
    if isinstance(module, Bottleneck) and module.bn3.weight is not None:
        module = eqx.tree_at(
            lambda m: m.bn3.weight, module, jnp.zeros_like(module.bn3.weight)
        )
    if isinstance(module, BasicBlock) and module.m.bn2.weight is not None:
        module = eqx.tree_at(
            lambda m: m.bn2.weight, module, jnp.zeros_like(module.bn2.weight)
        )
    return module


class ResNet(nn.StatefulLayer):
    _norm_layer: Optional[Callable[..., eqx.Module]] = eqx.field(static=True)
    in_planes: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)
    groups: int = eqx.field(static=True)
    base_width: int = eqx.field(static=True)
    conv1: nn.Conv2d
    bn1: eqx.Module
    layer1: eqx.Module
    layer2: eqx.Module
    layer3: eqx.Module
    layer4: eqx.Module
    avgpool: nn.AvgPool2d
    fc: nn.Linear
    # block: Union[BasicBlock, Bottleneck]
    # layers: List[int] = eqx.field(static=True)
    # zero_init_residual: bool = eqx.field(static=True)
    # width_per_group: int = eqx.field(static=True)
    # replace_stride_with_dilation: Optional[List[bool]] = eqx.field(static=True)

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        num_blocks: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., eqx.Module]] = None,
        *,
        key: jr.PRNGKey,
    ) -> None:
        if norm_layer is None:
            norm_layer = lambda *args, **kwargs: nn.BatchNorm(
                *args, axis_name="batch", **kwargs
            )
        self._norm_layer = norm_layer

        in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        keys = jr.split(key, 10)
        self.conv1 = nn.Conv2d(
            3,
            in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=keys[0],
        )
        self.bn1 = norm_layer(in_planes)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layer1, in_planes = self._make_layer(
            block, in_planes, 64, num_blocks[0], stride=1, key=keys[1]
        )
        layer2, in_planes = self._make_layer(
            block,
            in_planes,
            128,
            num_blocks[1],
            stride=2,
            key=keys[2],
        )
        layer3, in_planes = self._make_layer(
            block,
            in_planes,
            256,
            num_blocks[2],
            stride=2,
            key=keys[3],
        )
        layer4, in_planes = self._make_layer(
            block,
            in_planes,
            512,
            num_blocks[3],
            stride=2,
            key=keys[4],
        )
        self.in_planes = in_planes

        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512 * block.expansion, num_classes, key=keys[5])

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        # self.layer1 = re_initialize_layer(layer1, zero_init_residual, keys[6])
        # self.layer2 = re_initialize_layer(layer2, zero_init_residual, keys[7])
        # self.layer3 = re_initialize_layer(layer3, zero_init_residual, keys[8])
        # self.layer4 = re_initialize_layer(layer4, zero_init_residual, keys[9])

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck) and m.bn3.weight is not None:
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #         elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        in_planes: int,
        planes: int,
        num_blocks: int,
        stride: int = 1,
        *,
        key: jr.PRNGKey,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        keys = jr.split(key, num_blocks)
        for stride, k in zip(strides, keys):
            layers.append(
                block(in_planes, planes, stride, norm_layer=norm_layer, key=k)
            )
            in_planes = planes * block.expansion

        return nn.Sequential(layers), in_planes

    def __call__(self, x: jax.Array, state: nn.State, *, key: Optional[jax.Array]=None) -> jax.Array:
        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)

        x, state = self.layer1(x, state)
        x, state = self.layer2(x, state)
        x, state = self.layer3(x, state)
        x, state = self.layer4(x, state)

        x = self.avgpool(x)
        x = jnp.reshape(x, -1)
        x = self.fc(x)

        return x, state


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def alt_resnet18(bn_momentum=0.0, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    """

    norm_layer = lambda *args, **kwargs: StandardBatchNorm(
        *args, axis_name="batch", momentum=bn_momentum, **kwargs
    )

    return _resnet(BasicBlock, [8, 8, 8, 8], norm_layer=norm_layer, **kwargs)
