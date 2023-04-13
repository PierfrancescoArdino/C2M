from torch import nn
import torch.nn.functional as F


class DownBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='zeros',
                 use_norm=True):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, stride=stride,
                              kernel_size=kernel_size, padding=padding, groups=1, padding_mode=padding_mode)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.use_norm = use_norm

    def forward(self, x):
        out = self.conv(x)
        if self.use_norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        return out


class DownBlock3d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1],
                 padding_mode='zeros', use_norm=True):
        super(DownBlock3d, self).__init__()
        if padding_mode == "reflect":
            self.pad_conv = nn.ReflectionPad3d(padding)
        elif padding_mode == "replicate":
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
