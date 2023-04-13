import torch
from torch import nn
from modules.layers.down_block import DownBlock3d


class SparseMotionFeatureEncoder(nn.Module):
    def __init__(self, params):
        super(SparseMotionFeatureEncoder, self).__init__()
        self.input_channel = params["in_channel"]
        self.block_expansion = params["block_expansion"]
        self.num_down_blocks = params["num_down_blocks"]
        self.max_expansion = params["max_expansion"]
        self.padding_mode = params["padding_mode"]
        down_blocks = []
        for i in range(self.num_down_blocks):
            inplanes = self.input_channel if i == 0 else min(self.max_expansion, self.block_expansion * (2 ** (i - 1)))
            outplanes = min(self.max_expansion, self.block_expansion * (2 ** i))
            down_blocks.append(DownBlock3d(in_features=inplanes, out_features=outplanes,
                                           kernel_size=[3, 4, 4], stride=[1, 2, 2], padding=1,
                                           padding_mode=self.padding_mode))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, sparse_motion):
        out_dict = dict()
        for i in range(len(self.down_blocks)):
            model_input = sparse_motion if i == 0 else out_dict[f"enco_sparse_{i - 1}"]
            out_dict[f"enco_sparse_{i}"] = self.down_blocks[i](model_input)
        return out_dict
