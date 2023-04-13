from torch import nn
import torch
from torchvision.ops import roi_align, roi_pool
from modules.layers.same_block import SameBlock2d
from modules.layers.down_block import DownBlock2d


class AppearanceEncoder(nn.Module):
    def __init__(self, train_params, block_expansion, num_down_blocks,
                 max_expansion, pooling_after, padding_mode, pool_size, scale_factor, image_channel, seg_channel_bg,
                 seg_channel_fg,
                 instance_channel, flow_channel, occlusion_channel):
        super(AppearanceEncoder, self).__init__()
        self.train_params = train_params
        self.pool_size = pool_size
        self.h_appearance_map = int(train_params["input_size"][0] / (2 ** num_down_blocks) * scale_factor)
        self.w_appearance_map = int(train_params["input_size"][1] / (2 ** num_down_blocks) * scale_factor)

        down_blocks = []
        roi_align_blocks = []
        for i in range(num_down_blocks):
            if i == 0:
                inplanes = (image_channel + seg_channel_bg + seg_channel_fg +
                            instance_channel) * train_params["num_input_frames"] +\
                           (flow_channel + occlusion_channel) * (train_params["num_input_frames"] - 1)

                outplanes = block_expansion * train_params["num_input_frames"]
            elif i == num_down_blocks - 1:
                inplanes = min(max_expansion, block_expansion * (2 ** (i - 1))) * train_params["num_input_frames"]
                outplanes = min(max_expansion, block_expansion * (2 ** i))
            else:
                inplanes = min(max_expansion, block_expansion * (2 ** (i - 1))) * train_params["num_input_frames"]
                outplanes = min(max_expansion, block_expansion * (2 ** i)) * train_params["num_input_frames"]
            down_blocks.append(DownBlock2d(in_features=inplanes,
                                           out_features=outplanes,
                                           kernel_size=4, stride=2, padding=1,
                                           padding_mode=padding_mode, use_norm=True))
        self.h_flatten_appearance = self.h_appearance_map * self.w_appearance_map * outplanes
        roi_inplanes = block_expansion * (2 ** (pooling_after-1))
        roi_outplanes = block_expansion * (2 ** pooling_after)
        roi_align_blocks.append(SameBlock2d(in_features=roi_inplanes, out_features=roi_outplanes * 2,
                                            kernel_size=self.pool_size, stride=1, padding=0,
                                            padding_mode=padding_mode, use_norm=False))
        roi_align_blocks.extend([nn.Flatten(),
                                 nn.Linear(in_features=roi_outplanes * 2, out_features=roi_outplanes * 2)])
        self.roi_align_regressor = nn.Linear(in_features=roi_outplanes * 2, out_features=roi_outplanes)
        self.fuse_appearance_roi = nn.Linear(in_features=roi_outplanes + self.h_flatten_appearance,
                                             out_features=roi_outplanes)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.roi_align_blocks = nn.Sequential(*roi_align_blocks)
        self.spatial_scale = (1/scale_factor) * 2 ** pooling_after
        self.pooling_after = pooling_after

    def forward(self, input_dict):
        out_dict = dict()

        boxes = \
            torch.cat([input_dict["tracking_gnn"].batch.unsqueeze(1).repeat_interleave(self.train_params["num_input_frames"], dim=0),
                       torch.cat(torch.unbind(input_dict["tracking_gnn"].source_frames_nodes_roi_padded,
                                              dim=1),
                                 dim=0)],
                      dim=1)
        for i in range(len(self.down_blocks)):
            conv_input = input_dict["first_frame"] if i == 0 else out_dict[f"enco{i-1}"]
            feature_key = "app_encoded" if i == (len(self.down_blocks) - 1) else f"enco{i}"
            out_dict[feature_key] = self.down_blocks[i](conv_input)
        object_features =\
            roi_align(torch.cat(out_dict[f"enco{self.pooling_after-1}"].chunk(self.train_params["num_input_frames"], 1),
                                dim=0), boxes, self.pool_size, spatial_scale=1 / self.spatial_scale)
        object_features = self.roi_align_regressor(self.roi_align_blocks(object_features))
        out_dict["objects_feature"] = \
            torch.cat(self.fuse_appearance_roi(torch.cat([torch.repeat_interleave(out_dict["app_encoded"].flatten(1),
                                                                                  input_dict[
                                                                                      "tracking_gnn"].num_real_nodes *
                                                                                  self.train_params["num_input_frames"],
                                                                                  dim=0),
                                                          object_features], dim=1)).unsqueeze(1).chunk(
                self.train_params["num_input_frames"], 0), 1)
        return out_dict
