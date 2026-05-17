import math
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from typing import Optional, Callable, List, Dict, Any
from .submodules.layers import LIF, ASLIF, IF, ASIF, Conv, ConvBlock, MSConvBlock
from .submodules.blocks import BasicBlock, BottleneckBlock
from timm.models import register_model
from scipy.stats import norm
from .submodules.initialization import calculate_parameters, calculate_parameters_sim, calculate_alpha, calculate_alpha_sim, spiking_normal_, ding_normal_, fluctuation_normal_, micheli_normal_


class ResNet(nn.Module):
    def __init__(
        self,
        block: Callable[..., Any],
        planes: List[int],
        layers: List[int],
        prologue: nn.Module,
        epilogue: nn.Module,
        T: int = 4,
        groups: int = 1,
        width_per_group: int = 64,
        shortcut: str = 'SEW',
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
        init_method: str = 'spiking',
        zero_init_residual: bool = True,
        downsample_first: bool = False,
        p_init: float = 0.1,
        **kwargs,
    ):
        super(ResNet, self).__init__()
        assert shortcut in ['SEW', 'MS'], "shortcut must be 'SEW' or 'MS'"

        self.shortcut = shortcut
        self.ckwargs = {
            'conv': conv,
            'conv_kwargs': conv_kwargs,
            'activation': activation,
            'activation_kwargs': activation_kwargs}

        is_if = activation == IF or activation == ASIF

        tau = activation_kwargs.get('tau', 2.0)
        kappa = 1.0 - 1.0 / tau if not is_if else 1.0
        theta = activation_kwargs.get('v_threshold', 0.5)
        decay_input = activation_kwargs.get('decay_input', True)
        lmbda = 1.0 / tau if decay_input and not is_if else 1.0
        p_init = p_init

        self.skip = ['prologue.static_conv']

        self.T = T
        self.inplanes = planes[0]

        self.groups = groups
        self.base_width = width_per_group

        self.prologue = prologue
        self.epilogue = epilogue

        self.layers = nn.Sequential()
        for i in range(len(layers)):
            if i == 0 and not downsample_first:
                self.layers.append(self._make_layer(block, planes[i], layers[i]))
            else:
                self.layers.append(self._make_layer(block, planes[i], layers[i], stride=2))
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')

        if init_method == 'spiking':
            self.spiking_init(kappa=kappa, lmbda=lmbda, theta=theta, p_init=p_init)
        elif init_method == 'spiking_sim':
            self.spiking_init_sim(kappa=kappa, lmbda=lmbda, theta=theta, T=T, p_init=p_init)
        elif init_method == 'ding':
            self.ding_init(kappa=kappa, lmbda=lmbda, theta=theta)
        elif init_method == 'fluctuation_driven':
            self.fluctuation_driven_init(kappa=kappa, lmbda=lmbda, theta=theta, p_init=p_init)
        elif init_method == 'micheli':
            self.micheli_init(theta=theta)
        elif init_method == 'kaiming':
            self.kaiming_init()
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        if zero_init_residual:
            self.zero_init_residual(kappa=kappa, lmbda=lmbda, theta=theta, p_init=p_init, T=T)

    def zero_init_residual(self, kappa: float = 0.5, lmbda: float = 1.0, theta: float = 1.0,
                           p_init: float = 0.1, T: int = 4):
        if kappa != 1.0:
            sigma_sew = theta / norm.isf(1e-2) / (lmbda * math.sqrt(0.5) / (1 - kappa))
        else:
            sigma_sew = theta / norm.isf(1e-2) / (lmbda * math.sqrt(0.5) * T)
        for layer in self.layers:
            for block in layer:
                if isinstance(block, (BasicBlock, BottleneckBlock)):
                    block.zero_init_residual(sigma_sew=sigma_sew)
        nn.init.constant_(self.epilogue.classifier.weight, 0)

    def spiking_init(self, kappa: float = 0.5, lmbda: float = 1.0, theta: float = 1.0,
                     p_init: float = 0.1):
        sigma, sigma_static, mu_y, sigma_y = calculate_parameters(kappa, lmbda, theta, p_init)
        alpha = calculate_alpha(mu_y, sigma_y, sigma, kappa, lmbda, theta, p_init)
        self.prologue.init_weight(sigma=sigma, sigma_static=sigma_static)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                spiking_normal_(m.weight, sigma=sigma)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (LIF, IF)):
                m.surrogate_function.alpha = alpha * 2.0
            elif isinstance(m, (ASLIF, ASIF)):
                m.surrogate_function.alpha = alpha * 2.0
                m.base_alpha = alpha * sigma_y
        self.epilogue.init_weight(sigma=sigma)

    def spiking_init_sim(self, kappa: float = 0.5, lmbda: float = 1.0, theta: float = 1.0,
                         T: int = 4, p_init: float = 0.1):
        sigma, sigma_static, mu_y, sigma_y = calculate_parameters_sim(kappa, lmbda, theta, p_init,
                                                                      T)
        alpha = calculate_alpha_sim(sigma, kappa, lmbda, theta, p_init, T)
        self.prologue.init_weight(sigma=sigma, sigma_static=sigma_static)
        for m in self.layers.modules():
            if isinstance(m, nn.Conv2d):
                spiking_normal_(m.weight, sigma=sigma)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (LIF, IF)):
                m.surrogate_function.alpha = alpha * 2.0
            elif isinstance(m, (ASLIF, ASIF)):
                m.surrogate_function.alpha = alpha * 2.0
                m.base_alpha = alpha * sigma_y
        self.epilogue.init_weight(sigma=sigma)

    def ding_init(self, kappa: float = 0.5, lmbda: float = 1.0, theta: float = 1.0):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ding_normal_(m.weight, lmbda=lmbda, theta=theta)
                if m.bias is not None:
                    nn.init.constant_(m.bias, theta * (1 - kappa) / 2.0 / lmbda)

    def fluctuation_driven_init(self, kappa: float = 0.5, lmbda: float = 1.0, theta: float = 1.0,
                                xi: float = 2.0, p_init: float = 0.1):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                fluctuation_normal_(m.weight, kappa=kappa, lmbda=lmbda, theta=theta, xi=xi,
                                    p_init=p_init)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def micheli_init(self, theta: float = 1.0):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                micheli_normal_(m.weight, theta=theta)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Callable[..., Any], planes: int, blocks: int, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.shortcut == 'SEW':
                downsample = ConvBlock(self.inplanes, planes * block.expansion, kernel_size=1,
                                       stride=stride, padding=0, **self.ckwargs)
            else:
                downsample = MSConvBlock(self.inplanes, planes * block.expansion, kernel_size=1,
                                         stride=stride, padding=0, **self.ckwargs)

        layers = nn.Sequential()
        layers.append(
            block(self.inplanes, planes, stride=stride, groups=self.groups,
                  base_width=self.base_width, downsample=downsample, shortcut=self.shortcut,
                  **self.ckwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, stride=1, groups=self.groups,
                      base_width=self.base_width, downsample=None, shortcut=self.shortcut,
                      **self.ckwargs))

        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            assert x.dim() == 5
        else:
            #### [N, T, C, H, W] -> [T, N, C, H, W]
            x = x.transpose(0, 1)
        x = self.prologue(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.epilogue(x)
        return x


class SEWResNetPrologue(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ) -> None:
        super(SEWResNetPrologue, self).__init__()
        self.static_conv = ConvBlock(in_channels, out_channels, kernel_size=7, stride=2, padding=3,
                                     conv=conv, conv_kwargs=conv_kwargs, activation=activation,
                                     activation_kwargs=activation_kwargs)

    def init_weight(self, sigma: float, sigma_static: float):
        for m in self.static_conv.modules():
            if isinstance(m, nn.Conv2d):
                spiking_normal_(m.weight, sigma=sigma_static)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.static_conv(x)
        return x


class SEWResNetEpilogue(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        channels: int = 512,
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ) -> None:
        super(SEWResNetEpilogue, self).__init__()
        self.classifier = layer.Linear(channels, num_classes, step_mode='m')

    def init_weight(self, sigma: float):
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='linear')
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        return x


def _sew_resnet(
    block: Callable[..., Any],
    layers: List[int],
    num_classes: int = 1000,
    in_channels: int = 3,
    conv: Callable[..., Any] = Conv,
    conv_kwargs: Dict = {},
    activation: Callable[..., Any] = LIF,
    activation_kwargs: Dict = {},
    **kwargs,
) -> ResNet:
    ckwargs = {
        'conv': conv,
        'conv_kwargs': conv_kwargs,
        'activation': activation,
        'activation_kwargs': activation_kwargs}

    prologue = SEWResNetPrologue(in_channels, 64, conv, conv_kwargs, activation, activation_kwargs)
    epilogue = SEWResNetEpilogue(num_classes, 512 * block.expansion, conv, conv_kwargs, activation,
                                 activation_kwargs)

    return ResNet(block, [64, 128, 256, 512], layers, prologue, epilogue, shortcut='SEW',
                  downsample_first=True, **ckwargs, **kwargs)


class SEWResNetTinyPrologue(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ) -> None:
        super(SEWResNetTinyPrologue, self).__init__()
        self.static_conv = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                     conv=conv, conv_kwargs=conv_kwargs, activation=activation,
                                     activation_kwargs=activation_kwargs)

    def init_weight(self, sigma: float, sigma_static: float):
        for m in self.static_conv.modules():
            if isinstance(m, nn.Conv2d):
                spiking_normal_(m.weight, sigma=sigma_static)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.static_conv(x)
        return x


def _sew_resnet_tiny(
    block: Callable[..., Any],
    layers: List[int],
    num_classes: int = 1000,
    in_channels: int = 3,
    conv: Callable[..., Any] = Conv,
    conv_kwargs: Dict = {},
    activation: Callable[..., Any] = LIF,
    activation_kwargs: Dict = {},
    **kwargs,
) -> ResNet:
    ckwargs = {
        'conv': conv,
        'conv_kwargs': conv_kwargs,
        'activation': activation,
        'activation_kwargs': activation_kwargs}

    prologue = SEWResNetTinyPrologue(in_channels, 64, conv, conv_kwargs, activation,
                                     activation_kwargs)
    epilogue = SEWResNetEpilogue(num_classes, 512 * block.expansion, conv, conv_kwargs, activation,
                                 activation_kwargs)

    return ResNet(block, [64, 128, 256, 512], layers, prologue, epilogue, shortcut='SEW', **ckwargs,
                  **kwargs)


class MSResNetPrologue(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ) -> None:
        super(MSResNetPrologue, self).__init__()
        self.static_conv = conv(in_channels, out_channels, kernel_size=7, stride=2, padding=3,
                                **conv_kwargs)

    def init_weight(self, sigma: float, sigma_static: float):
        for m in self.static_conv.modules():
            if isinstance(m, nn.Conv2d):
                spiking_normal_(m.weight, sigma=sigma_static)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.static_conv(x)
        return x


class MSResNetEpilogue(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        channels: int = 512,
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ) -> None:
        super(MSResNetEpilogue, self).__init__()
        self.activation = activation(num_features=channels, flatten=True, **activation_kwargs)
        self.classifier = layer.Linear(channels, num_classes, step_mode='m')

    def init_weight(self, sigma: float):
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='linear')
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        x = self.classifier(x)
        return x


def _ms_resnet(
    block: Callable[..., Any],
    layers: List[int],
    num_classes: int = 1000,
    in_channels: int = 3,
    conv: Callable[..., Any] = Conv,
    conv_kwargs: Dict = {},
    activation: Callable[..., Any] = LIF,
    activation_kwargs: Dict = {},
    **kwargs,
) -> ResNet:
    ckwargs = {
        'conv': conv,
        'conv_kwargs': conv_kwargs,
        'activation': activation,
        'activation_kwargs': activation_kwargs}

    prologue = MSResNetPrologue(in_channels, 64, conv, conv_kwargs, activation, activation_kwargs)
    epilogue = MSResNetEpilogue(num_classes, 512 * block.expansion, conv, conv_kwargs, activation,
                                activation_kwargs)
    return ResNet(block, [64, 128, 256, 512], layers, prologue, epilogue, shortcut='MS',
                  downsample_first=True, **ckwargs, **kwargs)


class MSResNetTinyPrologue(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ) -> None:
        super(MSResNetTinyPrologue, self).__init__()
        self.static_conv = conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                **conv_kwargs)

    def init_weight(self, sigma: float, sigma_static: float):
        for m in self.static_conv.modules():
            if isinstance(m, nn.Conv2d):
                spiking_normal_(m.weight, sigma=sigma_static)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.static_conv(x)
        return x


def _ms_resnet_tiny(
    block: Callable[..., Any],
    layers: List[int],
    num_classes: int = 1000,
    in_channels: int = 3,
    conv: Callable[..., Any] = Conv,
    conv_kwargs: Dict = {},
    activation: Callable[..., Any] = LIF,
    activation_kwargs: Dict = {},
    **kwargs,
) -> ResNet:
    ckwargs = {
        'conv': conv,
        'conv_kwargs': conv_kwargs,
        'activation': activation,
        'activation_kwargs': activation_kwargs}

    prologue = MSResNetTinyPrologue(in_channels, 64, conv, conv_kwargs, activation,
                                    activation_kwargs)
    epilogue = MSResNetEpilogue(num_classes, 512 * block.expansion, conv, conv_kwargs, activation,
                                activation_kwargs)
    return ResNet(block, [64, 128, 256, 512], layers, prologue, epilogue, shortcut='MS', **ckwargs,
                  **kwargs)


@register_model
def sew_resnet18(**kwargs):
    return _sew_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


@register_model
def sew_resnet34(**kwargs):
    return _sew_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


@register_model
def sew_resnet50(**kwargs):
    return _sew_resnet(BottleneckBlock, [3, 4, 6, 3], **kwargs)


@register_model
def sew_resnet101(**kwargs):
    return _sew_resnet(BottleneckBlock, [3, 4, 23, 3], **kwargs)


@register_model
def sew_resnet152(**kwargs):
    return _sew_resnet(BottleneckBlock, [3, 8, 36, 3], **kwargs)


@register_model
def sew_resnet18_tiny(**kwargs):
    return _sew_resnet_tiny(BasicBlock, [2, 2, 2, 2], **kwargs)


@register_model
def sew_resnet34_tiny(**kwargs):
    return _sew_resnet_tiny(BasicBlock, [3, 4, 6, 3], **kwargs)


@register_model
def ms_resnet18(**kwargs):
    return _ms_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


@register_model
def ms_resnet34(**kwargs):
    return _ms_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


@register_model
def ms_resnet50(**kwargs):
    return _ms_resnet(BottleneckBlock, [3, 4, 6, 3], **kwargs)


@register_model
def ms_resnet101(**kwargs):
    return _ms_resnet(BottleneckBlock, [3, 4, 23, 3], **kwargs)


@register_model
def ms_resnet152(**kwargs):
    return _ms_resnet(BottleneckBlock, [3, 8, 36, 3], **kwargs)


@register_model
def ms_resnet18_tiny(**kwargs):
    return _ms_resnet_tiny(BasicBlock, [2, 2, 2, 2], **kwargs)


@register_model
def ms_resnet34_tiny(**kwargs):
    return _ms_resnet_tiny(BasicBlock, [3, 4, 6, 3], **kwargs)


def _sew_deep_resnet(
    block: Callable[..., Any],
    layers: List[int],
    num_classes: int = 1000,
    in_channels: int = 3,
    conv: Callable[..., Any] = Conv,
    conv_kwargs: Dict = {},
    activation: Callable[..., Any] = LIF,
    activation_kwargs: Dict = {},
    **kwargs,
) -> ResNet:
    ckwargs = {
        'conv': conv,
        'conv_kwargs': conv_kwargs,
        'activation': activation,
        'activation_kwargs': activation_kwargs}

    prologue = SEWResNetTinyPrologue(in_channels, 16, conv, conv_kwargs, activation,
                                     activation_kwargs)
    epilogue = SEWResNetEpilogue(num_classes, 64 * block.expansion, conv, conv_kwargs, activation,
                                 activation_kwargs)

    return ResNet(block, [16, 32, 64], layers, prologue, epilogue, shortcut='SEW', **ckwargs,
                  **kwargs)


@register_model
def sew_deep_resnet100(**kwargs):
    return _sew_deep_resnet(BasicBlock, [16, 16, 17], **kwargs)


@register_model
def sew_deep_resnet200(**kwargs):
    return _sew_deep_resnet(BasicBlock, [33, 33, 33], **kwargs)


@register_model
def sew_deep_resnet500(**kwargs):
    return _sew_deep_resnet(BasicBlock, [83, 83, 83], **kwargs)


@register_model
def sew_deep_resnet1000(**kwargs):
    return _sew_deep_resnet(BasicBlock, [166, 166, 167], **kwargs)


def _ms_deep_resnet(
    block: Callable[..., Any],
    layers: List[int],
    num_classes: int = 1000,
    in_channels: int = 3,
    conv: Callable[..., Any] = Conv,
    conv_kwargs: Dict = {},
    activation: Callable[..., Any] = LIF,
    activation_kwargs: Dict = {},
    **kwargs,
) -> ResNet:
    ckwargs = {
        'conv': conv,
        'conv_kwargs': conv_kwargs,
        'activation': activation,
        'activation_kwargs': activation_kwargs}

    prologue = MSResNetTinyPrologue(in_channels, 16, conv, conv_kwargs, activation,
                                    activation_kwargs)
    epilogue = MSResNetEpilogue(num_classes, 64 * block.expansion, conv, conv_kwargs, activation,
                                activation_kwargs)

    return ResNet(block, [16, 32, 64], layers, prologue, epilogue, shortcut='MS', **ckwargs,
                  **kwargs)


@register_model
def ms_deep_resnet100(**kwargs):
    return _ms_deep_resnet(BasicBlock, [16, 16, 17], **kwargs)


@register_model
def ms_deep_resnet200(**kwargs):
    return _ms_deep_resnet(BasicBlock, [33, 33, 33], **kwargs)


@register_model
def ms_deep_resnet500(**kwargs):
    return _ms_deep_resnet(BasicBlock, [83, 83, 83], **kwargs)


@register_model
def ms_deep_resnet1000(**kwargs):
    return _ms_deep_resnet(BasicBlock, [166, 166, 167], **kwargs)
