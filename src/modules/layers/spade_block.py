import torch
from torch import nn
from modules.layers.same_block import SameBlock2d
from torch.nn import functional as F


class SpatiallyAdaptiveNorm(nn.Module):
    r"""Spatially Adaptive Normalization (SPADE) initialization.

    Args:
        num_features (int) : Number of channels in the input tensor.
        cond_dims (int or list of int) : List of numbers of channels
            in the input.
        num_filters (int): Number of filters in SPADE.
        kernel_size (int): Kernel size of the convolutional filters in
            the SPADE layer.

    """

    def __init__(self,
                 num_features,
                 cond_dims,
                 num_filters=128,
                 kernel_size=3,
                 bias_only=False,
                 interpolation='nearest'):
        super().__init__()
        padding = kernel_size // 2
        self.mlps = nn.ModuleList()
        self.gammas = nn.ModuleList()
        self.betas = nn.ModuleList()
        self.bias_only = bias_only
        self.interpolation = interpolation

        # Make cond_dims a list.
        if type(cond_dims) != list:
            cond_dims = [cond_dims]

        # Make num_filters a list.
        if not isinstance(num_filters, list):
            num_filters = [num_filters] * len(cond_dims)
        else:
            assert len(num_filters) >= len(cond_dims)

        for i, cond_dim in enumerate(cond_dims):
            mlp = []
            if num_filters[i] > 0:
                mlp += [SameBlock2d(cond_dim, num_filters[i], kernel_size, padding=padding, padding_mode="reflect",
                                    use_norm=False)]
            mlp_ch = cond_dim if num_filters[i] == 0 else num_filters[i]
            mlp += [nn.Conv2d(mlp_ch, num_features * 2, kernel_size, stride=1, padding=padding, padding_mode="reflect")]
            self.mlps.append(nn.Sequential(*mlp))

        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.conditional = True

    def forward(self, x, *cond_inputs, **_kwargs):
        r"""Spatially Adaptive Normalization (SPADE) forward.

        Args:
            x (N x C1 x H x W tensor) : Input tensor.
            cond_inputs (list of tensors) : Conditional maps for SPADE.
        Returns:
            output (4D tensor) : Output tensor.
        """
        output = self.norm(x) if self.norm is not None else x
        for i in range(len(cond_inputs)):
            if cond_inputs[i] is None:
                continue
            label_map = F.interpolate(cond_inputs[i], size=x.size()[2:], mode=self.interpolation)
            affine_params = self.mlps[i](label_map)
            gamma, beta = affine_params.chunk(2, dim=1)
            if self.bias_only:
                output = output + beta
            else:
                output = output * (1 + gamma) + beta
        return output
