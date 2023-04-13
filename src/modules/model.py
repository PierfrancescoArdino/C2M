import torch
from torch.autograd import Variable as Vb
import torch.nn as nn
import torch.optim as optim
from modules.appearance_encoder.appearance_encoder import AppearanceEncoder
from modules.motion_estimator.dense_motion import DenseMotionNetwork
from modules.generator.generator import OcclusionAwareGenerator
from modules.discriminator import discriminator
from modules.layers.utils import init_weights
import losses.losses as losses
import utils
import numpy as np
import functools


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


mean = Vb(torch.FloatTensor([0.485, 0.456, 0.406])).view([1, 3, 1, 1])
std = Vb(torch.FloatTensor([0.229, 0.224, 0.225])).view([1, 3, 1, 1])


class GeneratorFullModel(nn.Module):

    def __init__(self, train_params=None, model_params=None, is_inference=False, dataset="cityscape"):
        super(GeneratorFullModel, self).__init__()
        self.train_params = train_params
        self.model_params = model_params
        self.num_frames = self.train_params["num_input_frames"] + self.train_params["num_predicted_frames"]
        self.appearance_encoder = AppearanceEncoder(self.train_params,
                                                    **self.model_params["appearance_encoder"],
                                                    **self.model_params["common_params"])
        self.motion_encoder = DenseMotionNetwork(self.train_params, self.model_params)
        # if not (self.train_params["do_not_use_vgg_loss"] and self.train_params["do_not_use_style_loss"]):
        #    self.vgg_net = Vgg19()
        self.criterionGAN = discriminator.GANLoss()
        self.criterionFeat = torch.nn.L1Loss()
        self.generator = OcclusionAwareGenerator(self.model_params["generator"],
                                                 self.model_params["flow_embedder"],
                                                 input_channel=self.model_params["common_params"]["image_channel"],
                                                 dataset=dataset)
        if not is_inference:
            self.objective_func = losses.TrainingLosses(self.train_params, self.model_params)

            self.model_parameters = list(self.appearance_encoder.parameters()) +\
                list(self.motion_encoder.get_parameters()) + list(self.generator.parameters())
            self.optimizer = optim.Adam(self.model_parameters, lr=self.train_params["lr_rate_g"],
                                        betas=(self.train_params["beta1"], self.train_params["beta2"]),
                                        eps=float(self.train_params["eps"]))
            self.optimizer_gnn = optim.Adam(list(self.motion_encoder.sparse_motion_estimator.parameters()),
                                            lr=self.train_params["lr_rate_gnn"], betas=(self.train_params["beta1"],
                                                                                        self.train_params["beta2"]),
                                            eps=float(self.train_params["eps"]))
            milestones = [i for i in range(self.train_params["milestone_start"],
                                           self.train_params["milestone_end"],
                                           self.train_params["milestone_every"])]
            self.scheduler_g = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                    milestones=milestones,
                                                                    gamma=self.train_params["gamma_g"])
            self.scheduler_gnn = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_gnn,
                                                                      milestones=milestones,
                                                                      gamma=self.train_params["gamma_gnn"])
            if self.train_params["use_image_discriminator"]:
                self.netD_image = \
                    discriminator.define_d(self.model_params["discriminator"]["in_channel"],
                                           self.model_params["discriminator"]["ndf"],
                                           self.model_params["discriminator"]["n_layers_D"],
                                           self.model_params["discriminator"]["num_D"],
                                           self.model_params["discriminator"]["padding_mode"])

                self.d_optimizer_image = optim.Adam(list(self.netD_image.parameters()),
                                                    lr=self.train_params["lr_rate_d"],
                                                    betas=(self.train_params["beta1"], self.train_params["beta2"]),
                                                    eps=float(self.train_params["eps"]))
                self.scheduler_d_image = torch.optim.lr_scheduler.MultiStepLR(self.d_optimizer_image,
                                                                        milestones=milestones,
                                                                        gamma=self.train_params["gamma_d"])
            if self.train_params["use_video_discriminator"]:
                in_channel = self.num_frames * self.model_params["discriminator"]["in_channel"]
                self.netD_video = \
                    discriminator.define_d(in_channel, self.model_params["discriminator"]["ndf"],
                                           self.model_params["discriminator"]["n_layers_D"],
                                           self.model_params["discriminator"]["num_D"],
                                           self.model_params["discriminator"]["padding_mode"])

                self.d_optimizer_video = optim.Adam(list(self.netD_video.parameters()),
                                                    lr=self.train_params["lr_rate_d"],
                                                    betas=(self.train_params["beta1"], self.train_params["beta2"]),
                                                    eps=float(self.train_params["eps"]))
                self.scheduler_d_video = torch.optim.lr_scheduler.MultiStepLR(self.d_optimizer_video,
                                                                              milestones=milestones,
                                                                              gamma=self.train_params["gamma_d"])

    def compute_loss_d(self, net_d, gt, fake, dis_type="image"):
        pred_real = net_d.forward(gt)
        pred_fake = net_d.forward(fake.detach())
        key = 'prediction_map_%s' % 0
        loss_d_real = self.criterionGAN(pred_real[key], True)
        loss_d_fake = self.criterionGAN(pred_fake[key], False)
        pred_fake = net_d.forward(fake)
        loss_g_gan, loss_g_gan_feat = self.gan_and_fm_loss(pred_real, pred_fake, dis_type)
        return loss_d_real, loss_d_fake, loss_g_gan, loss_g_gan_feat

    def gan_and_fm_loss(self, pred_real, pred_fake, dis_type):
        # GAN loss
        key = 'prediction_map_%s' % 0
        loss_g_gan = self.criterionGAN(pred_fake[key], True)
        # GAN feature matching loss
        key = 'feature_maps_%s' % 0
        loss_g_gan_feat = 0
        if self.train_params["loss_weights"][f"feature_matching_{dis_type}"] > 0:
            for i, (a, b) in enumerate(zip(pred_real[key], pred_fake[key])):
                value = torch.abs(a.detach() - b).mean()
                loss_g_gan_feat += value
        return loss_g_gan, loss_g_gan_feat

    def forward(self, data_batch):
        output_dict = {}
        frames = utils.resize_video(data_batch.get("video", None),
                                    scale_factor=self.model_params["common_params"]["scale_factor"], mode="bilinear")
        bg_mask = utils.resize_video(data_batch.get("bg_mask", None),
                                     scale_factor=self.model_params["common_params"]["scale_factor"], mode="nearest")
        fg_mask = utils.resize_video(data_batch.get("fg_mask", None),
                                     scale_factor=self.model_params["common_params"]["scale_factor"], mode="nearest")
        instance = utils.resize_video(data_batch.get("instance_mask", None).float(),
                                      scale_factor=self.model_params["common_params"]["scale_factor"],
                                      mode="nearest").int()

        input_of = utils.resize_video(data_batch.get("input_of", None),
                                      scale_factor=self.model_params["common_params"]["scale_factor"],
                                      mode="bilinear", is_flow=True)
        input_occ = utils.resize_video(data_batch.get("input_occ", None),
                                       scale_factor=self.model_params["common_params"]["scale_factor"],
                                       mode="bilinear")
        target_bw_of = utils.resize_video(data_batch.get("target_bw_of", None),
                                          scale_factor=self.model_params["common_params"]["scale_factor"],
                                          mode="bilinear", is_flow=True)
        target_bw_occ = utils.resize_video(data_batch.get("target_bw_occ", None),
                                           scale_factor=self.model_params["common_params"]["scale_factor"],
                                           mode="bilinear")
        target_fw_of = utils.resize_video(data_batch.get("target_fw_of", None),
                                          scale_factor=self.model_params["common_params"]["scale_factor"],
                                          mode="bilinear", is_flow=True)
        target_fw_occ = utils.resize_video(data_batch.get("target_fw_occ", None),
                                           scale_factor=self.model_params["common_params"]["scale_factor"],
                                           mode="bilinear")
        source_full_seg_mask = torch.cat([bg_mask[:, :, :self.train_params["num_input_frames"], ...],
                                          fg_mask[:, :, :self.train_params["num_input_frames"], ...]], 1)

        latent_traj = torch.FloatTensor(data_batch["tracking_gnn"].x.shape[0],
                                        self.train_params["num_predicted_frames"],
                                        self.model_params["motion_estimator"]["sparse_motion_estimator"][
                                            "z_dim"]).normal_(0, 1).to(data_batch["tracking_gnn"].x.device)
        # Encoder Network --> encode input frames
        input_frames = \
            torch.cat([torch.cat(torch.unbind(frames[:, :, :self.train_params["num_input_frames"], ...], dim=2), dim=1),
                       torch.cat(torch.unbind(source_full_seg_mask, dim=2), dim=1),
                       torch.cat(torch.unbind(instance[:, :, :self.train_params["num_input_frames"], ...], dim=2),
                                 dim=1)],
                      1)
        if input_of is not None:
            input_frames = \
                torch.cat([input_frames,
                           torch.cat(torch.unbind(input_of[:, :, :self.train_params["num_input_frames"], ...], dim=2),
                                     dim=1),
                           torch.cat(torch.unbind(input_occ[:, :, :self.train_params["num_input_frames"], ...], dim=2),
                                     dim=1)],
                          1)

        appearance_input = {"first_frame": input_frames,
                            "tracking_gnn": data_batch["tracking_gnn"]}
        app_features = self.appearance_encoder(appearance_input)
        # Motion Network --> compute latent vector
        motion_input = {"frames": frames,
                        "bg_mask": bg_mask,
                        "fg_mask": fg_mask,
                        "instance": instance,
                        "input_of": input_of,
                        "input_occ": input_occ,
                        "target_bw_of": target_bw_of,
                        "target_bw_occ": target_bw_occ,
                        "target_fw_of": target_fw_of,
                        "target_fw_occ": target_fw_occ,
                        "tracking_gnn": data_batch["tracking_gnn"],
                        "latent": latent_traj}
        output_dict.update(self.motion_encoder(app_features, motion_input))

        output_dict.update(
            {"generated": torch.cat(self.generator(
                torch.cat(torch.unbind(torch.unsqueeze(frames[:, :, self.train_params["num_input_frames"] - 1, :, :],
                                                       2).repeat(1, 1, self.train_params["num_predicted_frames"], 1, 1),
                                       dim=2), dim=0),
                torch.cat(torch.unbind(output_dict["dense_motion_bw"], 2), 0),
                torch.cat(torch.unbind(output_dict["occlusion_bw"], 2), 0))
                                    .unsqueeze(2).chunk(self.train_params["num_predicted_frames"], 0), 2)})
        output_dict.update({"generated_sparse": torch.cat(
            [torch.unsqueeze(utils.resample(frames[:, :, self.train_params["num_input_frames"] - 1, :, :],
                                            output_dict["sparse_motion_bw"][:, :, i, ...].detach()),
                             2) for i in range(self.train_params["num_predicted_frames"])], 2).detach()})
        output_dict.update({"generated_sparse_occ": torch.cat(
            [torch.unsqueeze(utils.resample(frames[:, :, self.train_params["num_input_frames"] - 1, :, :],
                                            output_dict["sparse_motion_bw"][:, :, i, ...].detach()) *
                             output_dict["sparse_occ_bw"][:, :, i, ...],
                             2) for i in range(self.train_params["num_predicted_frames"])], 2)})
        loss_dict = self.objective_func(data_batch["video"], frames, target_bw_of, target_fw_of,
                                        target_bw_occ, target_fw_occ, output_dict, data_batch["tracking_gnn"])
        loss_dict_dis_image = {}
        loss_dict_dis_video = {}
        if self.train_params["use_image_discriminator"]:
            loss_d_real, loss_d_fake, loss_g_gan, loss_g_gan_feat =\
                self.compute_loss_d(
                    self.netD_image, torch.cat(torch.unbind(data_batch["video"][:, :,
                                                            self.train_params["num_input_frames"]:, :, :], 2), 0),
                    torch.cat(torch.unbind(output_dict["generated"], 2), 0), "image")

            loss_dict["g_gan_image"] = loss_g_gan
            loss_dict["feature_matching_image"] = loss_g_gan_feat
            loss_dict_dis_image["d_real"] = loss_d_real
            loss_dict_dis_image["d_fake"] = loss_d_fake
        if self.train_params["use_video_discriminator"]:
            loss_d_real, loss_d_fake, loss_g_gan, loss_g_gan_feat =\
                self.compute_loss_d(
                    self.netD_video,
                    torch.cat(torch.unbind(frames, dim=2), dim=1),
                    torch.cat([torch.cat(torch.unbind(frames[:, :, :self.train_params["num_input_frames"], ...], dim=2),
                                         dim=1), torch.cat(torch.unbind(output_dict["generated"], 2), 1)], dim=1),
                    "video")
            loss_dict["g_gan_video"] = loss_g_gan
            loss_dict["feature_matching_video"] = loss_g_gan_feat
            loss_dict_dis_video["d_real"] = loss_d_real
            loss_dict_dis_video["d_fake"] = loss_d_fake
        return output_dict, loss_dict, loss_dict_dis_image, loss_dict_dis_video

    def inference(self, video, bg_mask, fg_mask, instance_mask, input_of, input_occ, tracking_gnn=None,
                  index_user_guidance=None,
                  z_m=None):

        output_dict = {}
        latent_traj = torch.FloatTensor(tracking_gnn.x.shape[0], self.train_params["num_predicted_frames"],
                                        self.model_params["motion_estimator"]["sparse_motion_estimator"][
                                            "z_dim"]).normal_(0, 1).to(video.device)
        self.motion_encoder.sparse_motion_estimator.eval()
        num_real_nodes = torch.LongTensor(
            [tracking_gnn.num_real_nodes]) if isinstance(
            tracking_gnn.num_real_nodes, int) else tracking_gnn.num_real_nodes
        total_number_nodes = 0
        if index_user_guidance is None:
            index_user_guidance = []
            for num_real_node in num_real_nodes:
                index_user_guidance.append(
                    np.random.random_integers(0, int(num_real_node) - 1) + total_number_nodes)
                total_number_nodes += num_real_node
            index_user_guidance = torch.LongTensor(index_user_guidance).to(video.device)
        frames = utils.resize_video(video, scale_factor=self.model_params["common_params"]["scale_factor"],
                                    mode="bilinear")
        bg_mask = utils.resize_video(bg_mask, scale_factor=self.model_params["common_params"]["scale_factor"],
                                     mode="nearest")
        fg_mask = utils.resize_video(fg_mask, scale_factor=self.model_params["common_params"]["scale_factor"],
                                     mode="nearest")
        input_of = utils.resize_video(input_of,
                                      scale_factor=self.model_params["common_params"]["scale_factor"],
                                      mode="bilinear", is_flow=True)
        input_occ = utils.resize_video(input_occ,
                                       scale_factor=self.model_params["common_params"]["scale_factor"],
                                       mode="bilinear")
        instance = utils.resize_video(instance_mask.float(),
                                      scale_factor=self.model_params["common_params"]["scale_factor"],
                                      mode="nearest").int()
        source_full_seg_mask = torch.cat([bg_mask[:, :, :self.train_params["num_input_frames"], ...],
                                          fg_mask[:, :, :self.train_params["num_input_frames"], ...]], 1)

        # Encoder Network --> encode input frames
        input_frames = \
            torch.cat([torch.cat(torch.unbind(frames[:, :, :self.train_params["num_input_frames"], ...], dim=2), dim=1),
                       torch.cat(torch.unbind(source_full_seg_mask, dim=2), dim=1),
                       torch.cat(torch.unbind(instance[:, :, :self.train_params["num_input_frames"], ...], dim=2),
                                 dim=1)],
                      1)
        if input_of is not None:
            input_frames = \
                torch.cat([input_frames,
                           torch.cat(torch.unbind(input_of[:, :, :self.train_params["num_input_frames"], ...], dim=2),
                                     dim=1),
                           torch.cat(torch.unbind(input_occ[:, :, :self.train_params["num_input_frames"], ...], dim=2),
                                     dim=1)],
                          1)

        appearance_input = {"first_frame": input_frames,
                            "tracking_gnn": tracking_gnn}

        # Encoder Network --> encode input frames
        app_features = self.appearance_encoder(appearance_input)
        motion_input = {"instance": instance,
                        "latent_traj": latent_traj,
                        "z_m": z_m,
                        "index_user_guidance": index_user_guidance,
                        "tracking_gnn": tracking_gnn}
        dense_motion_dict = self.motion_encoder.inference(app_features, motion_input)
        output_dict.update(dense_motion_dict)
        output_dict.update(
            {"generated": torch.cat(self.generator(
                torch.cat(torch.unbind(torch.unsqueeze(frames[:, :, self.train_params["num_input_frames"] - 1, :, :],
                                                       2).repeat(1, 1, self.train_params["num_predicted_frames"], 1, 1),
                                       dim=2), dim=0),
                torch.cat(torch.unbind(output_dict["dense_motion_bw"], 2), 0),
                torch.cat(torch.unbind(output_dict["occlusion_bw"], 2), 0))
                                    .unsqueeze(2).chunk(self.train_params["num_predicted_frames"], 0), 2)})
        output_dict.update({"generated_sparse": torch.cat(
            [torch.unsqueeze(utils.resample(video[:, :, self.train_params["num_input_frames"] - 1, :, :],
                                            dense_motion_dict["sparse_motion_bw"][:, :, i, ...]),
                             2) for i in range(self.train_params["num_predicted_frames"])], 2)})
        output_dict.update({"generated_sparse_occ": torch.cat(
            [torch.unsqueeze(utils.resample(video[:, :, self.train_params["num_input_frames"] - 1, :, :],
                                            dense_motion_dict["sparse_motion_bw"][:, :, i, ...]) *
                             dense_motion_dict["sparse_occ_bw"][:, :, i, ...],
                             2) for i in range(self.train_params["num_predicted_frames"])], 2)})
        return output_dict
