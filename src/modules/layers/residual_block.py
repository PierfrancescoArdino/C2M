import torch.nn as nn
import torch.nn.functional as F
from modules.layers.spade_block import SpatiallyAdaptiveNorm


class ResidualBlock(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_planes, out_planes, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.padding = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=kernel_size,
                               padding=0)
        self.norm1 = nn.BatchNorm2d(in_planes, affine=True)
        self.norm2 = nn.BatchNorm2d(out_planes, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.padding(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.padding(out)
        out = self.conv2(out)
        out += x
        return out


class ResidualSpadeBlock(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, cond_dims, in_planes, out_planes, kernel_size, padding, spade_params):
        super(ResidualSpadeBlock, self).__init__()
        self.cond_dims = cond_dims
        self.spade_params = spade_params
        self.padding = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=kernel_size,
                               padding=0)
        self.norm1 = SpatiallyAdaptiveNorm(in_planes, cond_dims)
        self.norm2 = SpatiallyAdaptiveNorm(out_planes, cond_dims)
        self.learned_shortcut = (in_planes != out_planes)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
            self.norm_s = SpatiallyAdaptiveNorm(in_planes, cond_dims)

    def forward(self, x, *cond_inputs):
        dx = self.norm1(x, *cond_inputs)
        dx = F.leaky_relu(dx, 0.2)
        dx = self.padding(dx)
        dx = self.conv1(dx)
        dx = self.norm2(dx, *cond_inputs)
        dx = F.leaky_relu(dx, 0.2)
        dx = self.padding(dx)
        dx = self.conv2(dx)
        if self.learned_shortcut:
            x_s = self.norm_s(x, *cond_inputs)
            x_s = F.leaky_relu(x_s, 0.2)
            x_s = self.conv_s(x_s)
            out = dx + x_s
        else:
            out = dx
        return out
