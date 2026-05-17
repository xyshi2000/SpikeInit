import torch
from torch import nn
from .layers import ConvBlock, MSConvBlock, LIF, Conv, ASLIF, ConvA, ConvBN, ConvTEBN, ConvBNTT
from typing import Optional, Callable, Any, Dict
from .initialization import spiking_normal_


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        downsample: Optional[nn.Module] = None,
        shortcut: str = 'SEW',
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ) -> None:
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        assert shortcut in ['SEW', 'MS'], "shortcut must be 'SEW' or 'MS'"
        self.shortcut = shortcut
        convblock = ConvBlock if shortcut == 'SEW' else MSConvBlock
        ckwargs = {
            'conv': conv,
            'conv_kwargs': conv_kwargs,
            'activation': activation,
            'activation_kwargs': activation_kwargs}

        self.downsample = downsample
        self.conv1 = convblock(inplanes, planes, kernel_size=3, stride=stride, padding=1,
                               groups=groups, **ckwargs)
        self.conv2 = convblock(planes, planes, kernel_size=3, stride=1, padding=1, groups=groups,
                               **ckwargs)

    def zero_init_residual(self, sigma_sew: float = None):
        if isinstance(self.conv2.conv, (ConvBN, ConvTEBN)):
            nn.init.constant_(self.conv2.conv.bn.weight, 0)
        elif isinstance(self.conv2.conv, (ConvA, ConvBNTT)):
            nn.init.constant_(self.conv2.conv.gamma, 0)
        elif isinstance(self.conv2.conv, nn.Conv2d):
            if self.shortcut == 'SEW' and sigma_sew is not None:
                spiking_normal_(self.conv2.conv.weight, sigma=sigma_sew)
            else:
                nn.init.constant_(self.conv2.conv.weight, 0)
        if isinstance(self.conv2.activation, ASLIF):
            if self.shortcut == 'SEW' and sigma_sew is not None:
                # calc sigma_y
                sigma_y = self.conv2.activation.base_alpha / self.conv2.activation.surrogate_function.alpha * 2.0
                self.conv2.activation.min_std = sigma_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        downsample: Optional[nn.Module] = None,
        shortcut: str = 'SEW',
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ) -> None:
        super(BottleneckBlock, self).__init__()
        assert shortcut in ['SEW', 'MS'], "shortcut must be 'SEW' or 'MS'"
        self.shortcut = shortcut
        convblock = ConvBlock if shortcut == 'SEW' else MSConvBlock
        ckwargs = {
            'conv': conv,
            'conv_kwargs': conv_kwargs,
            'activation': activation,
            'activation_kwargs': activation_kwargs}
        width = int(planes * (base_width / 64.)) * groups

        self.downsample = downsample
        self.conv1 = convblock(inplanes, width, kernel_size=1, stride=1, padding=0, groups=groups,
                               **ckwargs)
        self.conv2 = convblock(width, width, kernel_size=3, stride=stride, padding=1, groups=groups,
                               **ckwargs)
        self.conv3 = convblock(width, planes * self.expansion, kernel_size=1, stride=1, padding=0,
                               groups=groups, **ckwargs)

    def zero_init_residual(self, sigma_sew: float = None):
        if isinstance(self.conv3.conv, (ConvBN, ConvTEBN)):
            nn.init.constant_(self.conv3.conv.bn.weight, 0)
        elif isinstance(self.conv3.conv, (ConvA, ConvBNTT)):
            nn.init.constant_(self.conv3.conv.gamma, 0)
        elif isinstance(self.conv3.conv, nn.Conv2d):
            if self.shortcut == 'SEW' and sigma_sew is not None:
                spiking_normal_(self.conv3.conv.weight, sigma=sigma_sew)
            else:
                nn.init.constant_(self.conv3.conv.weight, 0)
        if isinstance(self.conv2.activation, ASLIF):
            if self.shortcut == 'SEW' and sigma_sew is not None:
                # calc sigma_y
                sigma_y = self.conv2.activation.base_alpha / self.conv2.activation.surrogate_function.alpha * 2.0
                self.conv2.activation.min_std = sigma_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        return out
