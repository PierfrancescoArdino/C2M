from torch import nn
import torch


class UpBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=1, padding_mode='zeros',
                 reshape_3d=True, input_2d=False):
        super(UpBlock2d, self).__init__()
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_features, out_features, kernel_size, stride, padding, padding_mode=padding_mode),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.reshape_3d = reshape_3d
        self.input_2d = input_2d

    def forward(self, x):
        if not self.input_2d:
            input_flattened = torch.cat(torch.unbind(x, dim=2), dim=0)
        else:
            input_flattened = x
        input_upsampled = self.main(input_flattened)
        if self.reshape_3d:
            return torch.cat(input_upsampled.unsqueeze(2).chunk(5, 0), 2)
        else:
            return input_upsampled
