import torch.nn as nn
import torch.nn.functional as F
from modules.layers.residual_block import ResidualBlock, ResidualSpadeBlock
from modules.layers.same_block import SameBlock2d
from modules.layers.down_block import DownBlock2d
from modules.layers.up_block import UpBlock2d
from modules.generator.flowembedder import FlowEmbedder
from utils import resample
import torch


class OcclusionAwareGenerator(nn.Module):
    def __init__(self, model_params, flow_params, input_channel, dataset):
        super(OcclusionAwareGenerator, self).__init__()
        self.input_channel = input_channel
        self.block_expansion = model_params["block_expansion"]
        self.num_down_blocks = model_params["num_down_blocks"]
        self.max_expansion = model_params["max_expansion"]
        self.num_bottleneck_blocks = model_params["num_bottleneck_blocks"]
        self.padding_mode = model_params["padding_mode"]
        self.use_spade = model_params["use_spade"]
        self.use_skip = model_params["use_skip"]
        self.spade_params = None
        self.flow_params = flow_params
        self.dataset = dataset
        # Encoder
        self.first = SameBlock2d(self.input_channel, self.block_expansion, kernel_size=7, padding=3,
                                 padding_mode=self.padding_mode)
        down_blocks = []
        for i in range(self.num_down_blocks):
            in_features = min(self.max_expansion, self.block_expansion * (2 ** i))
            out_features = min(self.max_expansion, self.block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=4, stride=2, padding=1,
                                           padding_mode=self.padding_mode))
        self.down_blocks = nn.ModuleList(down_blocks)

        if "kitti" in self.dataset:
            self.first_warped = SameBlock2d(self.input_channel, self.block_expansion, kernel_size=7, padding=3,
                                            padding_mode=self.padding_mode)
            down_blocks_warped = []
            for i in range(self.num_down_blocks):
                in_features = min(self.max_expansion, self.block_expansion * (2 ** i))
                out_features = min(self.max_expansion, self.block_expansion * (2 ** (i + 1)))
                down_blocks_warped.append(DownBlock2d(in_features, out_features, kernel_size=4, stride=2, padding=1,
                                                      padding_mode=self.padding_mode))
            self.down_blocks_warped = nn.Sequential(*down_blocks_warped)
            self.pre_decode = nn.Sequential(SameBlock2d(out_features * 2, out_features, kernel_size=3, padding=1,
                                                        padding_mode=self.padding_mode))

        up_blocks = []
        for i in range(self.num_down_blocks):
            in_features = min(self.max_expansion, self.block_expansion * (2 ** (self.num_down_blocks - i)))
            out_features = min(self.max_expansion, self.block_expansion * (2 ** (self.num_down_blocks - i - 1)))
            if self.use_spade:
                cond_dims = self.get_cond_dims(self.num_down_blocks - i)
                up_blocks.append(ResidualSpadeBlock(cond_dims=cond_dims, in_planes=in_features, out_planes=out_features,
                                                    kernel_size=3, padding=1, spade_params=self.spade_params))
            else:
                up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=3, padding=1,
                                           padding_mode=self.padding_mode, reshape_3d=False, input_2d=True))
        self.up_blocks = nn.ModuleList(up_blocks)

        # Middle
        blocks = []
        in_features = min(self.max_expansion, self.block_expansion * (2 ** self.num_down_blocks))
        for _ in range(self.num_bottleneck_blocks):
            block = ResidualBlock(in_features, in_features, kernel_size=3, padding=1)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        # Decoder
        self.final = nn.Sequential(
            nn.Conv2d(self.block_expansion, 3, kernel_size=7, padding=3),
            nn.Sigmoid())
        if self.use_spade:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            self.flowembedder = FlowEmbedder(self.flow_params)

    @staticmethod
    def deform_input(inp, optical_flow):
        _, h_old, w_old, _ = optical_flow.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            optical_flow = F.interpolate(optical_flow, size=(h, w), mode='bilinear')
        return resample(inp, optical_flow)

    def apply_optical(self, input_ref=None, optical_flow=None, occlusion_map=None):
        input_skip = self.deform_input(input_ref, optical_flow)
        if occlusion_map is not None:
            if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear')
            out = input_skip * occlusion_map
        else:
            out = input_skip
        return out

    def get_cond_dims(self, num_downs=0):
        r"""Get the dimensions of conditional inputs.

        Args:
           num_downs (int) : How many downsamples at current layer.
        Returns:
           ch (list) : List of dimensions.
        """

        num_filters = self.block_expansion
        num_downs = min(num_downs, self.flow_params["num_down_blocks"])
        ch = [min(self.max_expansion, num_filters * (2 ** num_downs))]
        return ch

    def get_cond_maps(self, label):
        r"""Get the conditional inputs.

        Args:
           label (4D tensor) : Input label tensor.
        Returns:
           cond_maps (list) : List of conditional inputs.
        """
        embedded_label = self.flowembedder(label)
        cond_maps = [embedded_label]
        cond_maps = [[m[i] for m in cond_maps] for i in
                     range(len(cond_maps[0]))]
        return cond_maps

    def forward(self, first_frame, flow, occlusion_map):
        # input mask: 1 for hole, 0 for valid
        if self.use_spade:
            img_warp = self.apply_optical(input_ref=first_frame, optical_flow=flow, occlusion_map=None)
            img_embed = torch.cat([img_warp, flow, occlusion_map], dim=1)
            flow_features = self.get_cond_maps(img_embed)
        out = self.first(first_frame)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        if not self.use_spade:
            out = \
                self.apply_optical(input_ref=out, optical_flow=flow, occlusion_map=occlusion_map)
        out = self.middle(out)
        if "kitti" in self.dataset:
            img_warp = self.apply_optical(input_ref=first_frame, optical_flow=flow, occlusion_map=None)
            x_warped = self.first_warped(img_warp)
            x_warped = self.down_blocks_warped(x_warped)
            if x_warped.shape[2] != occlusion_map.shape[2] or x_warped.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=x_warped.shape[2:], mode='bilinear')
            out = self.pre_decode(torch.cat([out, x_warped * occlusion_map], dim=1))
        for i in range(len(self.up_blocks)):
            if self.use_spade:
                if out.shape[-2:] != flow_features[self.num_down_blocks - i][0].shape[-2:]:
                    out = F.interpolate(out, list(flow_features[self.num_down_blocks - i][0].shape[-2:]),
                                        mode="bilinear")
                out = self.up_blocks[i](out, *flow_features[self.num_down_blocks - i])
                out = self.upsample(out)
            else:
                out = self.up_blocks[i](out)
        if out.shape[-2:] != first_frame.shape[-2:]:
            out = F.interpolate(out, list(first_frame.shape[-2:]), mode="bilinear")
        out = self.final(out)
        return out
