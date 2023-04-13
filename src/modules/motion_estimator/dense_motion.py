from torch import nn
import torch.nn.functional as F
import torch

import utils
from modules.motion_estimator.sparse_motion_estimator import SparseMotionGenerator
from modules.motion_estimator.motion_autoencoder import DenseMotionEncoder, DenseMotionDecoder
from modules.motion_estimator.sparse_encoder import SparseMotionFeatureEncoder
from modules.layers.same_block import SameBlockTwoConv2d


class DenseMotionNetwork(nn.Module):
    """
    Module that predicts a dense motion starting from user controlled sparse motion
    """
    def __init__(self, train_params, model_params):
        super(DenseMotionNetwork, self).__init__()
        self.train_params = train_params
        self.model_params = model_params
        self.num_frames = self.train_params["num_input_frames"] + self.train_params["num_predicted_frames"]
        self.dense_motion_params = model_params["motion_estimator"]
        self.scale_factor = model_params["common_params"]["scale_factor"]
        self.down_factor = 2 ** model_params["appearance_encoder"]["num_down_blocks"]
        self.h_appearance_map = int(train_params["input_size"][0] / self.down_factor * self.scale_factor)
        self.w_appearance_map = int(train_params["input_size"][1] / self.down_factor * self.scale_factor)
        self.h_scene_feature = model_params["appearance_encoder"]["block_expansion"] * \
            (2 ** model_params["appearance_encoder"]["pooling_after"])
        self.sparse_motion_estimator = \
            SparseMotionGenerator(**model_params["motion_estimator"]["sparse_motion_estimator"],
                                  input_scene_features=self.h_scene_feature, h_scene_features=self.h_scene_feature,
                                  num_predicted_frames=self.train_params["num_predicted_frames"],
                                  num_input_frames=self.train_params["num_input_frames"])
        self.sparse_feature_encoder = SparseMotionFeatureEncoder(self.dense_motion_params["sparse_motion_encoder"])
        motion_encoder_params = self.dense_motion_params["dense_motion_encoder"]
        motion_encoder_params.update({"scale_factor": self.scale_factor,
                                      "input_size": train_params["input_size"]})
        z_conv_app_inplanes = min(model_params["appearance_encoder"]["block_expansion"]
                                  * (2 ** model_params["appearance_encoder"]["num_down_blocks"]),
                                  model_params["appearance_encoder"]["max_expansion"])
        motion_encoder_fg_input_channels =\
            (self.model_params["common_params"]["image_channel"] +
             self.model_params["common_params"]["seg_channel_fg"] +
             self.model_params["common_params"]["instance_channel"]) * train_params["num_input_frames"] + \
            (self.model_params["common_params"]["flow_channel"] +
             self.model_params["common_params"]["occlusion_channel"]) + \
            (self.model_params["common_params"]["image_channel"] +
             self.model_params["common_params"]["seg_channel_fg"] +
             self.model_params["common_params"]["instance_channel"])

        self.motion_encoder_fg =\
            DenseMotionEncoder(motion_encoder_params, input_channel=motion_encoder_fg_input_channels,
                               output_channel=self.dense_motion_params["dense_motion_encoder"]["out_channel_fg"])
        motion_encoder_bg_input_channels =\
            (self.model_params["common_params"]["image_channel"] +
             self.model_params["common_params"]["seg_channel_bg"]) * train_params["num_input_frames"] + \
            (self.model_params["common_params"]["flow_channel"] +
             self.model_params["common_params"]["occlusion_channel"]) + \
            (self.model_params["common_params"]["image_channel"] +
             self.model_params["common_params"]["seg_channel_bg"])
        self.motion_encoder_bg = \
            DenseMotionEncoder(motion_encoder_params, input_channel=motion_encoder_bg_input_channels,
                               output_channel=self.dense_motion_params["dense_motion_encoder"]["out_channel_bg"])

        dense_generator_params = self.dense_motion_params["dense_motion_decoder"]
        dense_generator_params.update(
            {"num_input_frames": train_params["num_input_frames"],
             "num_predicted_frames": train_params["num_predicted_frames"],
             "scale_factor": self.scale_factor, "input_size": train_params["input_size"],
             "sparse_down": self.dense_motion_params["sparse_motion_encoder"]["num_down_blocks"]})
        self.dense_generator_bw = DenseMotionDecoder(dense_generator_params)
        if self.train_params["use_fw_of"]:
            self.dense_generator_fw = DenseMotionDecoder(dense_generator_params)

        self.zconv = SameBlockTwoConv2d(z_conv_app_inplanes + 64, 16 * self.train_params["num_predicted_frames"],
                                        3, 1, 1, padding_mode="reflect")
        self.fc = nn.Linear(self.dense_motion_params["dense_motion_encoder"]["out_channel_bg"] +
                            self.dense_motion_params["dense_motion_encoder"]["out_channel_fg"],
                            64 * self.h_appearance_map * self.w_appearance_map)

    def get_parameters(self):
        params = list(self.sparse_feature_encoder.parameters()) + list(self.motion_encoder_fg.parameters()) +\
                 list(self.motion_encoder_bg.parameters()) + list(self.dense_generator_bw.parameters()) +\
                 list(self.zconv.parameters()) + list(self.fc.parameters())
        if self.train_params["use_fw_of"]:
            params += list(self.dense_generator_fw.parameters())
        return params

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate_sparse_motion(self, tracking_gnn, sparse_motion_dict, source_instance, use_gt=False):
        """
        Generate objects sparse/dense motion representation from object's affine transformation
        :param tracking_gnn: object's info
        :param sparse_motion_dict: object's affine transformation
        :param source_instance: instance map
        :param use_gt: use gt theta instead of predicted from gnn
        :return: objects sparse/dense motion representation
        """
        out_dict = dict()
        sparse_motion_bw = torch.zeros(
            size=(source_instance.shape[0], 2, self.train_params["num_predicted_frames"], source_instance.shape[-2],
                  source_instance.shape[-1])).to(
            source_instance.device).float()
        sparse_motion_fw = torch.zeros(
            size=(source_instance.shape[0], 2, self.train_params["num_predicted_frames"], source_instance.shape[-2],
                  source_instance.shape[-1])).to(
            source_instance.device).float()
        sparse_motion_bin = torch.zeros(
            size=(source_instance.shape[0], 1, self.train_params["num_predicted_frames"], source_instance.shape[-2],
                  source_instance.shape[-1])).to(
            source_instance.device).float()
        b, _, h, w = source_instance.size()
        base_grid = torch.zeros([1, h, w, 2]).to(
            source_instance.device)
        linear_points = torch.linspace(-1, 1, w) if w > 1 else torch.Tensor([-1])
        base_grid[:, :, :, 0] = torch.ger(torch.ones(h), linear_points).expand_as(base_grid[:, :, :, 0])
        linear_points = torch.linspace(-1, 1, h) if h > 1 else torch.Tensor([-1])
        base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(w)).expand_as(base_grid[:, :, :, 1])
        for idx, [inst_id,
                  batch_id,
                  affine_gt] in enumerate(zip(tracking_gnn.source_frames_nodes_instance_ids[:, -1].long(),
                                              tracking_gnn.batch.long(), tracking_gnn.targets_theta)):
            if inst_id == 0:
                continue
            obj_mask = (source_instance[batch_id] == torch.squeeze(inst_id)).float()
            for t in range(self.train_params["num_predicted_frames"]):
                if use_gt:
                    warped_obj, obj_flow = self.warp(affine_gt[t].view(2, 3), obj_mask.unsqueeze(0), base_grid)
                else:
                    warped_obj, obj_flow = self.warp(sparse_motion_dict[f"theta_{t}"][idx].view(2, 3),
                                                     obj_mask.unsqueeze(0), base_grid)

                sparse_motion_bw[batch_id, :, t, ...] = torch.where(warped_obj == 1, obj_flow,
                                                                    sparse_motion_bw[batch_id, :, t, ...])
                sparse_motion_fw[batch_id, :, t, ...] = torch.where(obj_mask == 1, obj_flow * -1,
                                                                    sparse_motion_fw[batch_id, :, t, ...])
                sparse_motion_bin[batch_id, :, t, ...] = torch.where(warped_obj == 1, warped_obj,
                                                                     sparse_motion_bin[batch_id, :, t, ...])
        out_dict["sparse_motion_bw"] = sparse_motion_bw.detach()
        if self.train_params["use_fw_of"]:
            out_dict["sparse_motion_fw"] = sparse_motion_fw.detach()
        out_dict["sparse_motion_bin"] = sparse_motion_bin
        out_dict["sparse_occ_bw"] =\
            torch.cat([torch.unsqueeze(self.clip_mask(utils.get_occlusion_map(sparse_motion_fw[:, :, i, ...])),
                                       2) for i in range(self.train_params["num_predicted_frames"])], 2)
        out_dict["sparse_occ_fw"] =\
            torch.cat([torch.unsqueeze(self.clip_mask(utils.get_occlusion_map(sparse_motion_bw[:, :, i, ...])),
                                       2) for i in range(self.train_params["num_predicted_frames"])], 2)
        return out_dict

    @staticmethod
    def clip_mask(mask):
        one_ = torch.ones_like(mask)
        zero_ = torch.zeros_like(mask)
        return torch.where(mask > 0.5, one_, zero_)

    @staticmethod
    def warp(affine_matrix, x, base_grid):
        grid = F.affine_grid(affine_matrix.unsqueeze(0), x.size())
        b, _, h, w = x.size()
        flow = grid - base_grid
        flow = torch.cat([flow[:, :, :, 0:1] * ((w - 1.0) / 2.0), flow[:, :, :, 1:2] * ((h - 1.0) / 2.0)], dim=-1)
        t_x = F.grid_sample(x, grid)
        return t_x, flow.permute(0, 3, 1, 2)

    def forward(self, app_features, model_input):
        out_dict = dict()
        # FG
        flattened_frames = \
            torch.cat([torch.cat(torch.unbind(model_input["frames"][:, :, :self.train_params["num_input_frames"], ...],
                                              dim=2),
                                 dim=1).unsqueeze(2).repeat(1, 1, self.train_params["num_predicted_frames"], 1, 1),
                       model_input["frames"][:, :, self.train_params["num_input_frames"]:, ...]], dim=1)
        flattened_bg_mask = \
            torch.cat([torch.cat(torch.unbind(model_input["bg_mask"][:, :, :self.train_params["num_input_frames"], ...],
                                              dim=2),
                                 dim=1).unsqueeze(2).repeat(1, 1, self.train_params["num_predicted_frames"], 1, 1),
                       model_input["bg_mask"][:, :, self.train_params["num_input_frames"]:, ...]], dim=1)
        flattened_fg_mask = \
            torch.cat([torch.cat(torch.unbind(model_input["fg_mask"][:, :, :self.train_params["num_input_frames"], ...],
                                              dim=2),
                                 dim=1).unsqueeze(2).repeat(1, 1, self.train_params["num_predicted_frames"], 1, 1),
                       model_input["fg_mask"][:, :, self.train_params["num_input_frames"]:, ...]], dim=1)
        flattened_instance = \
            torch.cat([torch.cat(torch.unbind(model_input["instance"][:, :, :self.train_params["num_input_frames"], ...],
                                              dim=2),
                                 dim=1).unsqueeze(2).repeat(1, 1, self.train_params["num_predicted_frames"], 1, 1),
                       model_input["instance"][:, :, self.train_params["num_input_frames"]:, ...]], dim=1)
        flattened_flows = torch.cat([model_input["target_bw_of"], model_input["target_bw_occ"]], dim=1)

        bg_out =\
            self.motion_encoder_bg(torch.cat([flattened_frames,
                                              flattened_bg_mask, flattened_flows], 1).contiguous())
        # FG
        fg_out =\
            self.motion_encoder_fg(torch.cat([flattened_frames,
                                              flattened_fg_mask,
                                              flattened_instance,
                                              flattened_flows], 1).contiguous())

        out_dict.update({"mu": torch.cat([bg_out["mu"], fg_out["mu"]], 1),
                         "logvar": torch.cat([bg_out["logvar"], fg_out["logvar"]], 1)})
        z_m = self.reparameterize(out_dict["mu"], out_dict["logvar"])
        sparse_motion_dict = self.sparse_motion_estimator(model_input["tracking_gnn"], app_features["objects_feature"],
                                                          model_input["latent"])
        out_dict.update(sparse_motion_dict)
        sparse_motion =\
            self.generate_sparse_motion(model_input["tracking_gnn"], sparse_motion_dict,
                                        model_input["instance"][:, :, self.train_params["num_input_frames"] - 1, ...].float(),
                                        self.train_params["use_gt_training"])
        sparse_encoded_bw = self.sparse_feature_encoder(sparse_motion["sparse_motion_bw"])
        if self.train_params["use_fw_of"]:
            sparse_encoded_fw = self.sparse_feature_encoder(sparse_motion["sparse_motion_fw"])
        code_motion = self.zconv(torch.cat([self.fc(z_m).view(-1, 64, self.h_appearance_map,
                                                              self.w_appearance_map), app_features["app_encoded"]], 1))
        codex = torch.unsqueeze(app_features["app_encoded"], 2).repeat(1, 1,
                                                                       self.train_params["num_predicted_frames"], 1, 1)
        code_motion = torch.cat(torch.chunk(code_motion.unsqueeze(2), self.train_params["num_predicted_frames"], 1), 2)
        z = torch.cat([codex, code_motion], 1)  # (256L, 272L, 8L, 8L)   272-256=16
        if self.train_params["use_fw_of"]:
            dense_fw = self.dense_generator_fw(app_features, sparse_encoded_fw, sparse_motion["sparse_motion_fw"],
                                               sparse_motion["sparse_occ_fw"], z)
        dense_bw = self.dense_generator_bw(app_features, sparse_encoded_bw, sparse_motion["sparse_motion_bw"],
                                           sparse_motion["sparse_occ_bw"], z)
        out_dict.update(sparse_motion)
        out_dict["dense_motion_bw"] = dense_bw["dense_motion"]
        out_dict["occlusion_bw"] = dense_bw["occlusion"]
        if self.train_params["use_fw_of"]:
            out_dict["dense_motion_fw"] = dense_fw["dense_motion"]
            out_dict["occlusion_fw"] = dense_fw["occlusion"]
        return out_dict

    def inference(self, app_features, model_input):
        out_dict = dict()
        sparse_motion_dict = self.sparse_motion_estimator.inference(model_input["tracking_gnn"],
                                                                    model_input["latent_traj"],
                                                                    model_input["index_user_guidance"],
                                                                    app_features["objects_feature"])
        out_dict.update(sparse_motion_dict)
        sparse_motion =\
            self.generate_sparse_motion(
                model_input["tracking_gnn"], sparse_motion_dict,
                model_input["instance"][:, :, self.train_params["num_input_frames"] - 1, ...].float(),
                self.train_params["use_gt_eval"])
        sparse_encoded_bw = self.sparse_feature_encoder(sparse_motion["sparse_motion_bw"])
        if self.train_params["use_fw_of"]:
            sparse_encoded_fw = self.sparse_feature_encoder(sparse_motion["sparse_motion_fw"])
        code_motion = self.zconv(torch.cat([self.fc(model_input["z_m"]).view(-1, 64, self.h_appearance_map,
                                                                             self.w_appearance_map),
                                            app_features["app_encoded"]], 1))
        codex = torch.unsqueeze(app_features["app_encoded"], 2).repeat(1, 1, self.train_params["num_predicted_frames"],
                                                                       1, 1)
        code_motion = torch.cat(torch.chunk(code_motion.unsqueeze(2), self.train_params["num_predicted_frames"], 1), 2)
        z = torch.cat([codex, code_motion], 1)  # (256L, 272L, 8L, 8L)   272-256=16
        if self.train_params["use_fw_of"]:
            dense_fw = self.dense_generator_fw(app_features, sparse_encoded_fw, sparse_motion["sparse_motion_fw"],
                                               sparse_motion["sparse_occ_fw"], z)
        dense_bw = self.dense_generator_bw(app_features, sparse_encoded_bw, sparse_motion["sparse_motion_bw"],
                                           sparse_motion["sparse_occ_bw"], z)
        out_dict.update(sparse_motion)
        out_dict["dense_motion_bw"] = dense_bw["dense_motion"]
        out_dict["occlusion_bw"] = dense_bw["occlusion"]
        if self.train_params["use_fw_of"]:
            out_dict["dense_motion_fw"] = dense_fw["dense_motion"]
            out_dict["occlusion_fw"] = dense_fw["occlusion"]
        out_dict["index_user_guidance"] = model_input["index_user_guidance"]
        return out_dict
