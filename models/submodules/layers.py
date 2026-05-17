import torch
import torch.nn as nn
import math
from spikingjelly.activation_based import layer
from spikingjelly.activation_based import surrogate, neuron
from typing import Callable, Any, Dict, List
from scipy.stats import norm


@torch.jit.script
def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)


@torch.jit.script
def piecewise_exp_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: torch.Tensor):
    return alpha * torch.exp(-2 * alpha * torch.abs(x)) * grad_output


class PiecewiseExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: List[torch.Tensor]):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha[0]
        grad_x = piecewise_exp_backward(grad_output, x, alpha)
        return grad_x, None


class IF(neuron.IFNode):
    def __init__(self, v_threshold=0.5, **kwargs):
        super().__init__(v_threshold=v_threshold, v_reset=0.,
                         surrogate_function=surrogate.PiecewiseExp(alpha=2.0), detach_reset=True,
                         step_mode='m', backend='torch', store_v_seq=False)


class LIF(neuron.LIFNode):
    def __init__(self, tau=2., v_threshold=0.5, decay_input=True, **kwargs):
        super().__init__(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=0.,
                         surrogate_function=surrogate.PiecewiseExp(alpha=2.0), detach_reset=True,
                         step_mode='m', backend='torch', store_v_seq=False)


class ASIF(neuron.IFNode):
    def __init__(self, num_features: int, v_threshold=0.5, flatten: bool = False, base_alpha=1.0,
                 freeze: bool = False, **kwargs):
        super().__init__(v_threshold=v_threshold, v_reset=0.,
                         surrogate_function=surrogate.PiecewiseExp(alpha=2.0), detach_reset=True,
                         step_mode='m', backend='torch', store_v_seq=False)
        self.flatten = flatten
        self.freeze = freeze
        self.base_alpha = base_alpha
        self.register_parameter('gamma', torch.nn.Parameter(torch.ones(num_features)))
        self.register_parameter('beta', torch.nn.Parameter(torch.zeros(num_features)))
        self.min_std = v_threshold / norm.isf(0.01)
        self.instance_alpha = []

    def reset(self):
        self.instance_alpha.clear()
        return super().reset()

    def neuronal_fire(self):
        if self.flatten:
            gamma = self.gamma.view(1, -1)
            beta = self.beta.view(1, -1)
        else:
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
        if self.training and (not self.freeze) and (not self.flatten):
            return PiecewiseExp.apply(self.v * gamma + beta - self.v_threshold, self.instance_alpha)
        else:
            return self.surrogate_function(self.v * gamma + beta - self.v_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        y_seq = []
        v_seq = []
        for t in range(T):
            self.v_float_to_tensor(x[t])
            self.neuronal_charge(x[t])
            if self.training and (not self.freeze):
                with torch.no_grad():
                    v_seq.append(self.v.clone().detach())
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            y_seq.append(spike)

        if self.training and (not self.freeze):
            with torch.no_grad():
                if not self.flatten:
                    instance_std = torch.stack(v_seq).std((0, 3, 4),
                                                          keepdim=True).clamp(min=self.min_std)
                    self.instance_alpha.append(self.base_alpha / instance_std)
        return torch.stack(y_seq)


class ASLIF(neuron.LIFNode):
    def __init__(self, num_features: int, tau=2.0, v_threshold=0.5, decay_input=True,
                 base_alpha=1.0, flatten: bool = False, freeze: bool = False, **kwargs):
        super().__init__(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=0.,
                         surrogate_function=surrogate.PiecewiseExp(alpha=2.0), detach_reset=True,
                         step_mode='m', backend='torch', store_v_seq=False)
        self.flatten = flatten
        self.freeze = freeze
        self.base_alpha = base_alpha
        self.register_parameter('gamma', torch.nn.Parameter(torch.ones(num_features)))
        self.register_parameter('beta', torch.nn.Parameter(torch.zeros(num_features)))
        self.min_std = v_threshold / norm.isf(0.01)
        self.instance_alpha = []

    def reset(self):
        self.instance_alpha.clear()
        return super().reset()

    def neuronal_fire(self):
        if self.flatten:
            gamma = self.gamma.view(1, -1)
            beta = self.beta.view(1, -1)
        else:
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
        if self.training and (not self.freeze) and (not self.flatten):
            return PiecewiseExp.apply(self.v * gamma + beta - self.v_threshold, self.instance_alpha)
        else:
            return self.surrogate_function(self.v * gamma + beta - self.v_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        y_seq = []
        v_seq = []
        for t in range(T):
            self.v_float_to_tensor(x[t])
            self.neuronal_charge(x[t])
            if self.training and (not self.freeze):
                with torch.no_grad():
                    v_seq.append(self.v.clone().detach())
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            y_seq.append(spike)

        if self.training and (not self.freeze):
            with torch.no_grad():
                if not self.flatten:
                    instance_std = torch.stack(v_seq).std((0, 3, 4),
                                                          keepdim=True).clamp(min=self.min_std)
                    self.instance_alpha.append(self.base_alpha / instance_std)
        return torch.stack(y_seq)


class Conv(layer.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                         bias, 'zeros', 'm')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ) -> None:
        super(ConvBlock, self).__init__()
        self.conv = conv(in_channels, out_channels, kernel_size, stride, padding, groups=groups,
                         **conv_kwargs)
        self.activation = activation(num_features=out_channels, **activation_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.activation(out)
        return out


class MSConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        conv: Callable[..., Any] = Conv,
        conv_kwargs: Dict = {},
        activation: Callable[..., Any] = LIF,
        activation_kwargs: Dict = {},
    ) -> None:
        super(MSConvBlock, self).__init__()
        self.activation = activation(num_features=in_channels, **activation_kwargs)
        self.conv = conv(in_channels, out_channels, kernel_size, stride, padding, groups=groups,
                         **conv_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv(out)
        return out


### For ablation Study ###


class ConvN(layer.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                         bias, 'zeros', 'm')
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x_shape = x.shape
        x = x.flatten(0, 1)
        x = torch.nn.functional.batch_norm(x, self.running_mean, self.running_var, None, None,
                                           training=self.training, momentum=0.1, eps=0.00001)
        x = x.view(x_shape)
        return x


class ConvA(layer.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                         True, 'zeros', 'm')
        self.register_parameter('gamma', torch.nn.Parameter(torch.ones(out_channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_shape = [x.shape[0], x.shape[1]]
        y = x.flatten(0, 1)
        weight = self.weight * self.gamma[:, None, None, None]
        y = torch.nn.functional.conv2d(y, weight, self.bias, self.stride, self.padding,
                                       self.dilation, self.groups)
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)


class ConvBN(layer.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                         bias, 'zeros', 'm')
        self.bn = layer.BatchNorm2d(out_channels, step_mode='m')
        nn.init.constant_(self.bn.weight, threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.bn(x)
        return x


class ConvTEBN(layer.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        T: int = 4,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                         bias, 'zeros', 'm')
        self.bn = layer.BatchNorm2d(out_channels, step_mode='m')
        self.register_parameter('p', torch.nn.Parameter(torch.ones(T)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.bn(x)
        x = x * self.p.view(-1, 1, 1, 1, 1)
        return x


class ConvBNTT(layer.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        T: int = 4,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                         bias, 'zeros', 'm')
        self.register_parameter('gamma', torch.nn.Parameter(torch.ones(out_channels * T)))
        self.register_buffer('running_mean', torch.zeros(out_channels * T))
        self.register_buffer('running_var', torch.ones(out_channels * T))
        self.T = T
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        assert (x.dim() == 5) and (x.shape[0] == self.T)
        y_seq = []
        for t in range(self.T):
            y = torch.nn.functional.batch_norm(
                x[t], self.running_mean[t * self.out_channels:(t + 1) * self.out_channels],
                self.running_var[t * self.out_channels:(t + 1) * self.out_channels], None, None,
                training=self.training)
            y_seq.append(
                y * self.gamma[t * self.out_channels:(t + 1) * self.out_channels].view(1, -1, 1, 1))
        return torch.stack(y_seq)


class MPBNLIF(neuron.LIFNode):
    def __init__(self, num_features: int, tau=2., v_threshold=0.5, decay_input=True,
                 flatten: bool = False, **kwargs):
        super().__init__(tau=tau, decay_input=decay_input, v_threshold=v_threshold, v_reset=0.,
                         surrogate_function=surrogate.PiecewiseExp(alpha=2.0), detach_reset=True,
                         step_mode='m', backend='torch', store_v_seq=False)
        self.bn = nn.BatchNorm2d(num_features) if not flatten else nn.Identity()
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        y_seq = []
        for t in range(T):
            self.v_float_to_tensor(x[t])
            self.neuronal_charge(x[t])
            v_bn = self.bn(self.v)
            spike = self.surrogate_function(v_bn - self.v_threshold)
            self.neuronal_reset(spike)
            y_seq.append(spike)
        return torch.stack(y_seq)


class ReLU(nn.ReLU):
    def __init__(self, inplace: bool = False, **kwargs):
        super().__init__(inplace=inplace)
