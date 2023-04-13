from torch import nn
import torch.nn.functional as F


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=1, padding_mode='zeros',
                 use_norm=True):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, stride=stride,
                              kernel_size=kernel_size, padding=padding, groups=1, padding_mode=padding_mode)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        self.use_norm = use_norm

    def forward(self, x):
        out = self.conv(x)
        if self.use_norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        return out


class SameBlockTwoConv2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=1, padding_mode='zeros',
                 use_norm=True):
        super(SameBlockTwoConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, stride=stride,
                              kernel_size=kernel_size, padding=padding, groups=1, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(in_channels=out_features, out_channels=out_features, stride=stride,
                               kernel_size=kernel_size, padding=padding, groups=1, padding_mode=padding_mode)
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        if self.use_norm:
            out = self.norm(out)
        out = self.conv2(F.leaky_relu(out, 0.2))
        return out


class SameBlock3d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=1, padding_mode='zeros',
                 use_norm=True):
        super(SameBlock3d, self).__init__()
        if padding_mode == "reflect":
            self.pad_conv = nn.ReflectionPad3d(padding)
        elif padding == "replicate":
            self.pad_conv = nn.ReplicationPad3d(padding)
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, stride=stride,
                              kernel_size=kernel_size, padding=0, groups=1, padding_mode=padding_mode)
        self.norm = nn.BatchNorm3d(out_features, affine=True)
        self.use_norm = use_norm

    def forward(self, x):
        out = self.conv(self.pad_conv(x))
        if self.use_norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        return out
