from torch import nn
import torch
import torch.nn.functional as F
import utils
from modules.layers.same_block import SameBlock3d, SameBlock2d
from modules.layers.down_block import DownBlock3d
from modules.layers.up_block import UpBlock2d
from utils import resample


class DenseMotionEncoder(nn.Module):
    def __init__(self, model_params, input_channel, output_channel):
        super(DenseMotionEncoder, self).__init__()

        self.scale_factor = model_params["scale_factor"]
        self.input_size = model_params["input_size"]
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.block_expansion = model_params["block_expansion"]
        self.num_down_blocks = model_params["num_down_blocks"]
        self.down_factor = 2 ** self.num_down_blocks
        self.max_expansion = model_params["max_expansion"]
        self.padding_mode = model_params["padding_mode"]
        self.h_appearance_map = int(self.input_size[0] / self.down_factor * self.scale_factor)
        self.w_appearance_map = int(self.input_size[1] / self.down_factor * self.scale_factor)
        self.t_s = model_params["t_stride"]
        self.h_s = model_params["h_stride"]
        self.w_s = model_params["w_stride"]
        self.t_k = model_params["t_kernel"]
        self.h_k = model_params["h_kernel"]
        self.w_k = model_params["w_kernel"]
        self.t_p = model_params["t_padding"]
        self.h_p = model_params["h_padding"]
        self.w_p = model_params["w_padding"]
        down_blocks = []

        for i in range(len(self.w_p)):
            padding = [self.w_p[i]] * 2 + [self.h_p[i]] * 2 + [self.t_p[i]] * 2
            inplanes = self.input_channel if i == 0 else min(self.max_expansion, self.block_expansion * (2 ** (i - 1)))
            outplanes = min(self.max_expansion, self.block_expansion * (2 ** i))
            down_blocks.append(DownBlock3d(in_features=inplanes, out_features=outplanes,
                                           kernel_size=[self.t_k[i], self.h_k[i], self.w_k[i]],
                                           stride=[self.t_s[i], self.h_s[i], self.w_s[i]],
                                           padding=padding,
                                           padding_mode=self.padding_mode))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.fc1 = nn.Linear(self.h_appearance_map * self.w_appearance_map * outplanes, self.output_channel)
        self.fc2 = nn.Linear(self.h_appearance_map * self.w_appearance_map * outplanes, self.output_channel)

    def forward(self, video):
        out_dict = dict()
        for i in range(len(self.w_p)):
            conv_input = video if i == 0 else out
            out = self.down_blocks[i](conv_input)
        temp = out.view(video.shape[0], -1)
        out_dict["mu"] = self.fc1(temp)
        # print 'mu: '+str(mu.size())
        out_dict["logvar"] = self.fc2(temp)
        return out_dict


class DenseMotionDecoder(nn.Module):
    def __init__(self, model_params):
        super(DenseMotionDecoder, self).__init__()
        self.scale_factor = model_params["scale_factor"]
        self.input_size = model_params["input_size"]
        self.input_channel = model_params["in_channel"]
        self.out_channel = model_params["out_channel"]
        self.num_input_frames = model_params["num_input_frames"]
        self.num_predicted_frames = model_params["num_predicted_frames"]
        self.block_expansion = model_params["block_expansion"]
        self.num_up_blocks = model_params["num_up_blocks"]
        self.up_factor = 2 ** self.num_up_blocks
        self.max_expansion = model_params["max_expansion"]
        self.padding_mode = model_params["padding_mode"]
        self.num_down_block_sparse_encoder = model_params["sparse_down"]
        self.use_feature_resample = model_params["use_feature_resample"]
        self.use_appearance_feature = model_params["use_appearance_feature"]
        out_features = min(self.max_expansion, self.block_expansion * (2 ** self.num_up_blocks))
        self.first = SameBlock3d(self.input_channel, out_features, 3, 1, 1, padding_mode=self.padding_mode)
        up_blocks = []
        fuse_convs = []
        flow_predictors = []
        occlusion_predictors = []
        for i in range(self.num_up_blocks):
            if i == 0:
                in_features = min(self.max_expansion, self.block_expansion * (2 ** (self.num_up_blocks - i)))
            else:
                in_features = min(self.max_expansion, self.block_expansion * (2 ** (self.num_up_blocks - i)))
                in_features = in_features * (self.num_input_frames + 1) if self.use_appearance_feature else in_features
            out_features = min(self.max_expansion, self.block_expansion * (2 ** (self.num_up_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, padding_mode=self.padding_mode))
            flow_predictors.append(FlowPredictor(output_channel=2, input_channel=out_features))
            occlusion_predictors.append(OcclusionPredictor(input_channel_features=out_features,
                                                           input_conditioning=0))
            if i >= (self.num_up_blocks - self.num_down_block_sparse_encoder):
                fuse_convs.append(SameBlock3d(out_features * 2, out_features, 3, 1, 1, padding_mode=self.padding_mode))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.fuse_convs = nn.ModuleList(fuse_convs)
        self.flow_predictors = nn.ModuleList(flow_predictors)
        self.occlusion_predictors = nn.ModuleList(occlusion_predictors)
        self.final_up_block = UpBlock2d(out_features, self.out_channel, padding_mode=self.padding_mode)
        self.final_fuse = SameBlock3d(out_features + 2, out_features, 3, 1, 1, padding_mode=self.padding_mode)
        self.flow = FlowPredictor(output_channel=2, input_channel=out_features)
        self.occlusion = OcclusionPredictor(input_channel_features=out_features, input_conditioning=0)

    def forward(self, appearance_features, sparse_features, sparse_motion, sparse_occlusion, z):
        out_dict = dict()
        out = self.first(z)
        idx = 0
        for i in range(len(self.up_blocks)):
            if i == 0:
                out = self.up_blocks[i](out)
            else:
                if self.use_appearance_feature:
                    app_key = f"enco{self.num_up_blocks - i}"
                    app_repeated = torch.cat(torch.unbind(torch.unsqueeze(appearance_features[app_key],
                                                                          2).repeat(1, 1, self.num_predicted_frames, 1, 1),
                                                          dim=2), dim=0)
                    if self.use_feature_resample:
                        new_h, new_w = app_repeated.shape[-2:]
                        obj_motion = utils.resize_flow(torch.cat(torch.unbind(sparse_motion, 2), 0), [new_h, new_w])
                        obj_occlusion = F.interpolate(torch.cat(torch.unbind(sparse_occlusion, 2), 0), size=[new_h, new_w],
                                                      mode="bilinear")
                        app_resampled = resample(app_repeated, obj_motion) * obj_occlusion
                    else:
                        new_h, new_w = app_repeated.shape[-2:]
                        app_resampled = app_repeated
                    if list(out.shape[-2:]) != [new_h, new_w]:
                        out = utils.resize_video(out, [new_h, new_w], mode="bilinear")
                    up_input = torch.cat([out,
                                          torch.cat(app_resampled.unsqueeze(2).chunk(self.num_predicted_frames, 0), 2)],
                                         1)
                else:
                    up_input = out
                out = self.up_blocks[i](up_input)
            if i >= (self.num_up_blocks - self.num_down_block_sparse_encoder):
                new_h, new_w = sparse_features[f"enco_sparse_{self.num_up_blocks-i - 1}"].shape[-2:]
                if list(out.shape[-2:]) != [new_h, new_w]:
                    out = utils.resize_video(out, [new_h, new_w], mode="bilinear")
                fused = torch.cat([out, sparse_features[f"enco_sparse_{self.num_up_blocks-i - 1}"]], 1)
                out = self.fuse_convs[idx](fused)
                idx += 1
        out = self.final_up_block(out)
        out = torch.cat(torch.unbind(self.final_fuse(torch.cat([out, sparse_motion], dim=1)), dim=2), dim=0)
        dense_motion = self.flow(out)
        out_dict["dense_motion"] = torch.cat(dense_motion.unsqueeze(2).chunk(self.num_predicted_frames, 0), 2)
        out_dict["occlusion"] = torch.cat(self.occlusion(out).unsqueeze(2).chunk(self.num_predicted_frames, 0), 2)
        return out_dict


class FlowPredictor(nn.Module):
    def __init__(self, output_channel=2, input_channel=64):
        super(FlowPredictor, self).__init__()
        self.flow_predictor = []
        self.flow_predictor.append(SameBlock2d(in_features=input_channel, out_features=32,
                                               kernel_size=3, stride=1, padding=1, padding_mode="reflect"))
        self.flow_predictor.extend([
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, output_channel, 3, 1, 0)]
        )
        self.flow_predictor = nn.Sequential(*self.flow_predictor)

    def forward(self, x):
        return self.flow_predictor(x)


class OcclusionPredictor(nn.Module):
    def __init__(self, input_channel_features=64, input_conditioning=2):
        super(OcclusionPredictor, self).__init__()
        self.occlusion_predictor = []
        self.occlusion_predictor.append(SameBlock2d(in_features=input_channel_features, out_features=32,
                                                    kernel_size=3, stride=1, padding=1, padding_mode="reflect"))
        self.occlusion_predictor.extend([
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 1, 3, 1, 0), nn.Sigmoid()]
        )
        self.occlusion_predictor = nn.Sequential(*self.occlusion_predictor)

    def forward(self, x):
        return self.occlusion_predictor(x)
