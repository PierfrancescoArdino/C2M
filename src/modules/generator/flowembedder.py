import torch
from torch import nn
import torch.nn.functional as F
from modules.layers.same_block import SameBlock2d
from modules.layers.down_block import DownBlock2d
from modules.layers.up_block import UpBlock2d


class FlowEmbedder(nn.Module):
    r"""Embed the generated flow to get embedded features.

    Args:
        model_params (obj): Embed network configuration.
    """

    def __init__(self, model_params):
        super().__init__()
        self.input_channel = model_params["input_channel"]
        self.block_expansion = model_params["block_expansion"]
        self.num_down_blocks = model_params["num_down_blocks"]
        self.max_expansion = model_params["max_expansion"]
        self.padding_mode = model_params["padding_mode"]
        self.use_decoder = model_params["use_decoder"]

        self.conv_first = SameBlock2d(self.input_channel, self.block_expansion, kernel_size=3, padding=1,
                                      padding_mode=self.padding_mode, use_norm=False)

        # Downsample.
        down_blocks = []
        ch = [min(self.max_expansion, self.block_expansion * (2 ** i))
              for i in range(self.num_down_blocks + 1)]
        for i in range(self.num_down_blocks):
            down_blocks.append(DownBlock2d(ch[i], ch[i+1], kernel_size=4, stride=2, padding=1,
                                           padding_mode=self.padding_mode))
        self.down_blocks = nn.ModuleList(down_blocks)

        # Upsample.
        up_blocks = []
        if self.use_decoder:
            for i in reversed(range(self.num_down_blocks)):
                ch_i = ch[i + 1] * (
                    2 if i != self.num_down_blocks - 1 else 1)
                up_blocks.append(UpBlock2d(ch_i, ch[i], kernel_size=3, stride=1, padding=1,
                                           padding_mode=self.padding_mode, reshape_3d=False, input_2d=True))
        self.up_blocks = nn.ModuleList(up_blocks[::-1])

    def forward(self, x):
        r"""Embedding network forward.

        Args:
            x (NxCxHxW tensor): Network input.
        Returns:
            output (list of tensors): Network outputs at different layers.
        """
        if x is None:
            return None
        output = [self.conv_first(x)]

        for i in range(self.num_down_blocks):
            conv = self.down_blocks[i](output[-1])
            output.append(conv)

        if not self.use_decoder:
            return output

        # If the network has a decoder, will use outputs from the decoder
        # layers instead of the encoding layers.

        for i in reversed(range(self.num_down_blocks)):
            input_i = output[-1]
            if i != self.num_down_blocks - 1:
                new_h, new_w = output[i + 1].shape[-2:]
                if list(input_i.shape[-2:]) != [new_h, new_w]:
                    input_i = F.interpolate(input_i, [new_h, new_w], mode="bilinear")
                input_i = torch.cat([input_i, output[i + 1]], dim=1)

            conv = self.up_blocks[i](input_i)
            output.append(conv)

        output = output[self.num_down_blocks:]
        return output[::-1]
