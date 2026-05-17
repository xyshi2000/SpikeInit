import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from .submodules.layers import LIF, ASLIF, IF, ASIF, Conv, ConvBlock
from typing import Callable, Any, Dict
from timm.models import register_model
from .submodules.initialization import calculate_alpha, calculate_parameters, calculate_alpha_sim, calculate_parameters_sim, spiking_normal_, ding_normal_, fluctuation_normal_, micheli_normal_


class VGGSNN(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        classifier: nn.Module,
        T: int = 4,
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
        init_method: str = 'spiking',
        p_init: float = 0.1,
        **kwargs,
    ):
        super(VGGSNN, self).__init__()
        self.skip = ['features.static_conv']
        self.T = T
        ckwargs = {
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

        self.features = features
        self.classifier = classifier
        self.boost = layer.AvgPool1d(10, 10, step_mode='m')

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

    def spiking_init(self, kappa: float = 0.5, lmbda: float = 1.0, theta: float = 1.0,
                     p_init: float = 0.1):
        sigma, sigma_static, mu_y, sigma_y = calculate_parameters(kappa, lmbda, theta, p_init)
        alpha = calculate_alpha(mu_y, sigma_y, sigma, kappa, lmbda, theta, p_init)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                spiking_normal_(m.weight, sigma=sigma)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, )):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (LIF, IF)):
                m.surrogate_function.alpha = alpha * 2.0
            elif isinstance(m, (ASLIF, ASIF)):
                m.surrogate_function.alpha = alpha * 2.0
                m.base_alpha = alpha * sigma_y
        # first conv layer
        spiking_normal_(self.features.static_conv.conv.weight, sigma=sigma_static)

    def spiking_init_sim(self, kappa: float = 0.5, lmbda: float = 1.0, theta: float = 1.0,
                         T: int = 4, p_init: float = 0.1):
        sigma, sigma_static, mu_y, sigma_y = calculate_parameters_sim(kappa, lmbda, theta, p_init,
                                                                      T)
        alpha = calculate_alpha_sim(sigma, kappa, lmbda, theta, p_init, T)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                spiking_normal_(m.weight, sigma=sigma)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (LIF, IF)):
                m.surrogate_function.alpha = alpha * 2.0
            elif isinstance(m, (ASLIF, ASIF)):
                m.surrogate_function.alpha = alpha * 2.0
                m.base_alpha = alpha * sigma_y
        # first conv layer
        spiking_normal_(self.features.static_conv.conv.weight, sigma=sigma_static)

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

    def forward(self, x: torch.Tensor):
        if x.dim() != 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            assert x.dim() == 5
        else:
            #### [N, T, C, H, W] -> [T, N, C, H, W]
            x = x.transpose(0, 1)
        x = self.features(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.classifier(x)
        x = x.unsqueeze(2)
        #### [T, N, L] -> [T, N, C=1, L]
        out = self.boost(x).squeeze(2)
        return out


class VGG11_features(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ):
        super().__init__()

        ckwargs = {
            'conv': conv,
            'conv_kwargs': conv_kwargs,
            'activation': activation,
            'activation_kwargs': activation_kwargs}
        self.static_conv = ConvBlock(in_channels, 64, **ckwargs)
        self.layer2 = ConvBlock(64, 128, **ckwargs)
        self.layer3 = nn.Sequential(
            ConvBlock(128, 256, stride=2, **ckwargs),
            ConvBlock(256, 256, **ckwargs),
        )
        self.layer4 = nn.Sequential(
            ConvBlock(256, 512, stride=2, **ckwargs),
            ConvBlock(512, 512, **ckwargs),
        )
        self.layer5 = nn.Sequential(
            ConvBlock(512, 512, stride=2, **ckwargs),
            ConvBlock(512, 512, **ckwargs),
        )
        self.pool = layer.AvgPool2d(2, 2, step_mode='m')

    def forward(self, x: torch.Tensor):
        x = self.static_conv(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        return x


class VGG16_features(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ):
        super().__init__()

        ckwargs = {
            'conv': conv,
            'conv_kwargs': conv_kwargs,
            'activation': activation,
            'activation_kwargs': activation_kwargs}
        self.static_conv = ConvBlock(in_channels, 64, **ckwargs)
        self.layer1 = ConvBlock(64, 64, **ckwargs)
        self.layer2 = nn.Sequential(
            ConvBlock(64, 128, **ckwargs),
            ConvBlock(128, 128, **ckwargs),
        )
        self.layer3 = nn.Sequential(
            ConvBlock(128, 256, stride=2, **ckwargs),
            ConvBlock(256, 256, **ckwargs),
            ConvBlock(256, 256, **ckwargs),
        )
        self.layer4 = nn.Sequential(
            ConvBlock(256, 512, stride=2, **ckwargs),
            ConvBlock(512, 512, **ckwargs),
            ConvBlock(512, 512, **ckwargs),
        )
        self.layer5 = nn.Sequential(
            ConvBlock(512, 512, stride=2, **ckwargs),
            ConvBlock(512, 512, **ckwargs),
            ConvBlock(512, 512, **ckwargs),
        )
        self.pool = layer.AvgPool2d(2, 2, step_mode='m')

    def forward(self, x: torch.Tensor):
        x = self.static_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        return x


class VGG_classifier(nn.Module):
    def __init__(
        self,
        input_size: int = 32,
        num_classes: int = 10,
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ):
        super().__init__()
        self.fc1 = layer.Linear(512 * (input_size // 16)**2, 2048, step_mode='m')
        self.activation1 = activation(num_features=2048, flatten=True, **activation_kwargs)
        self.fc2 = layer.Linear(2048, 2048, step_mode='m')
        self.activation2 = activation(num_features=2048, flatten=True, **activation_kwargs)
        self.classifier = layer.Linear(2048, num_classes * 10, step_mode='m')

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.classifier(x)
        return x


@register_model
def vgg11snn(
    in_channels: int = 3,
    num_classes: int = 10,
    input_size: int = 32,
    conv: Callable[..., Any] = Conv,
    conv_kwargs: Dict = {},
    activation: Callable[..., Any] = LIF,
    activation_kwargs: Dict = {},
    **kwargs,
) -> VGGSNN:
    ckwargs = {
        'conv': conv,
        'conv_kwargs': conv_kwargs,
        'activation': activation,
        'activation_kwargs': activation_kwargs}

    features = VGG11_features(in_channels=in_channels, **ckwargs)
    classifier = VGG_classifier(input_size=input_size, num_classes=num_classes, **ckwargs)
    model = VGGSNN(features=features, classifier=classifier, **ckwargs, **kwargs)
    return model


@register_model
def vgg16snn(
    in_channels: int = 3,
    num_classes: int = 10,
    input_size: int = 32,
    conv: Callable[..., Any] = Conv,
    conv_kwargs: Dict = {},
    activation: Callable[..., Any] = LIF,
    activation_kwargs: Dict = {},
    **kwargs,
) -> VGGSNN:
    ckwargs = {
        'conv': conv,
        'conv_kwargs': conv_kwargs,
        'activation': activation,
        'activation_kwargs': activation_kwargs}

    features = VGG16_features(in_channels=in_channels, **ckwargs)
    classifier = VGG_classifier(input_size=input_size, num_classes=num_classes, **ckwargs)
    model = VGGSNN(features=features, classifier=classifier, **ckwargs, **kwargs)
    return model


class VGG9_features(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ):
        super().__init__()

        ckwargs = {
            'conv': conv,
            'conv_kwargs': conv_kwargs,
            'activation': activation,
            'activation_kwargs': activation_kwargs}
        self.static_conv = ConvBlock(in_channels, 64, **ckwargs)
        self.layer2 = ConvBlock(64, 128, **ckwargs)
        self.pool1 = layer.AvgPool2d(2, 2, step_mode='m')
        self.layer3 = nn.Sequential(
            ConvBlock(128, 256, **ckwargs),
            ConvBlock(256, 256, **ckwargs),
        )
        self.pool2 = layer.AvgPool2d(2, 2, step_mode='m')
        self.layer4 = nn.Sequential(
            ConvBlock(256, 512, **ckwargs),
            ConvBlock(512, 512, **ckwargs),
        )
        self.pool3 = layer.AvgPool2d(2, 2, step_mode='m')
        self.layer5 = nn.Sequential(
            ConvBlock(512, 512, **ckwargs),
            ConvBlock(512, 512, **ckwargs),
        )
        self.pool4 = layer.AvgPool2d(2, 2, step_mode='m')

    def forward(self, x: torch.Tensor):
        x = self.static_conv(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.pool2(x)
        x = self.layer4(x)
        x = self.pool3(x)
        x = self.layer5(x)
        x = self.pool4(x)
        return x


# Denoted as VGGSNN in previous papers
@register_model
def vgg9snn(
    in_channels: int = 3,
    num_classes: int = 10,
    input_size: int = 32,
    conv: Callable[..., Any] = Conv,
    conv_kwargs: Dict = {},
    activation: Callable[..., Any] = LIF,
    activation_kwargs: Dict = {},
    **kwargs,
) -> VGGSNN:
    ckwargs = {
        'conv': conv,
        'conv_kwargs': conv_kwargs,
        'activation': activation,
        'activation_kwargs': activation_kwargs}

    features = VGG9_features(in_channels=in_channels, **ckwargs)
    classifier = layer.Linear(512 * (input_size // 16)**2, num_classes * 10, step_mode='m')
    model = VGGSNN(features=features, classifier=classifier, **ckwargs, **kwargs)
    return model
