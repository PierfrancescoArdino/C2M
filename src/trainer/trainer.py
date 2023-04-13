from trainer.base import BaseTrainer
import numpy as np
import torch
import utils


class Trainer(BaseTrainer):
    r"""Initialize c2m trainer.
        Args:
            cfg (obj): Global configuration.
            c2m (obj): C2M network.
            optimizer_vae (obj): Optimizer for the generator network.
            optimizer_gnn
            optimizer_d_image (obj): Optimizer for the discriminator network.
            train_data_loader (obj): Train data loader.
            val_data_loader (obj): Validation data loader.
            local_rank (int): local rank of gpu
        """
    def __init__(self, cfg, opt, c2m, flownet, optimizer_vae, optimizer_gnn, optimizer_d_image, optimizer_d_video,
                 scheduler_vae, scheduler_gnn, scheduler_d_image, scheduler_d_video, train_data_loader, val_data_loader,
                 local_rank):
        super(Trainer, self).__init__(cfg, opt, c2m, flownet, optimizer_vae, optimizer_gnn, optimizer_d_image,
                                      optimizer_d_video, scheduler_vae, scheduler_gnn, scheduler_d_image,
                                      scheduler_d_video, train_data_loader, val_data_loader, local_rank)
        self.cfg = cfg
        self.opt = opt
        self.c2m = c2m
        self.flownet = flownet
        self.optimizer_vae = optimizer_vae
        self.optimizer_gnn = optimizer_gnn
        self.scheduler_vae = scheduler_vae
        self.scheduler_gnn = scheduler_gnn
        self.scheduler_d_image = scheduler_d_image
        self.scheduler_d_video = scheduler_d_video
        self.optimizer_d_image = optimizer_d_image
        self.optimizer_d_video = optimizer_d_video
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.local_rank = local_rank
        self.loss_dict_gen = self.loss_dict_dis = {}

    def compute_flow(self, train_batch):
        """
        compute optical flow and occlusion mask using pre-trained network
        :param train_batch: input video
        :return: optical flow and occlusion map
        """
        out_dict = {}
        t_in = self.train_params["num_input_frames"]
        t_out = self.train_params["num_predicted_frames"]
        input_optical_flows = []
        input_occlusion_masks = []
        target_bw_optical_flows = []
        target_bw_occlusion_masks = []
        target_fw_optical_flows = []
        target_fw_occlusion_masks = []
        b, _, _, h, w = train_batch["video"].size()
        # Compute forward optical flow between consecutive frames in input frames
        for i in range(t_in - 1):
            input_image_a = train_batch["video"][:, :, i, ...] * 2 - 1
            input_image_b = train_batch["video"][:, :, i + 1, ...] * 2 - 1
            # Forward flow
            fw_flow, bw_conf = self.flownet(input_image_a, input_image_b)
            fw_flow = fw_flow.unsqueeze(2)
            # Backward Flow
            bw_flow, fw_conf = self.flownet(input_image_b, input_image_a)
            fw_conf = fw_conf.unsqueeze(2)
            input_optical_flows.append(fw_flow)
            input_occlusion_masks.append(fw_conf)

        # Compute forward and backward optical flow between last input frame and label frames
        for i in range(t_out):
            input_image_a = train_batch["video"][:, :, t_in-1, ...] * 2 - 1
            input_image_b = train_batch["video"][:, :, t_in + i, ...] * 2 - 1
            # Forward flow
            fw_flow, bw_conf = self.flownet(input_image_a, input_image_b)
            fw_flow = fw_flow.unsqueeze(2)
            bw_conf = bw_conf.unsqueeze(2)
            # Backward Flow
            bw_flow, fw_conf = self.flownet(input_image_b, input_image_a)
            bw_flow = bw_flow.unsqueeze(2)
            fw_conf = fw_conf.unsqueeze(2)
            target_fw_optical_flows.append(fw_flow)
            target_fw_occlusion_masks.append(fw_conf)

            target_bw_optical_flows.append(bw_flow)
            target_bw_occlusion_masks.append(bw_conf)

        out_dict["input_of"] = torch.cat(input_optical_flows, dim=2) if len(input_optical_flows) > 0 else None
        out_dict["input_occ"] =\
            torch.cat(input_occlusion_masks, dim=2) if len(input_occlusion_masks) > 0 else None

        out_dict["target_bw_of"] = torch.cat(target_bw_optical_flows, dim=2)
        out_dict["target_bw_occ"] = torch.cat(target_bw_occlusion_masks, dim=2)
        if self.train_params["use_fw_of"]:
            out_dict["target_fw_of"] = torch.cat(target_fw_optical_flows, dim=2)
            out_dict["target_fw_occ"] = torch.cat(target_fw_occlusion_masks, dim=2)
        return out_dict

    def _start_of_iteration(self, train_batch):
        r"""Things to do before an iteration.
        Args:
            train_batch (dict): Data used for the current iteration.
        """
        train_batch = train_batch.data
        del train_batch['complete_list']
        for k, v in train_batch.items():
            train_batch[k] = v.to(self.local_rank, non_blocking=True)
        if not self.train_params["use_pre_processed_of"]:
            train_batch.update(self.compute_flow(train_batch))
        else:
            train_batch["input_of"] = None
            train_batch["input_occ"] = None
        self.c2m.train()
        return train_batch

    def load_checkpoint(self):
        if self.train_params["continue_train"]:
            model_name_c2m = self.sampledir + f'/{self.train_params["which_epoch"]}_c2m_model.pth.tar'
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
            print("loading model from {}".format(model_name_c2m))
            state_dict = torch.load(model_name_c2m, map_location=map_location)
            self.c2m.load_state_dict(state_dict['c2m'], strict=False)
            self.optimizer_vae.load_state_dict(state_dict['optimizer'])
            if self.train_params["use_image_discriminator"]:
                self.optimizer_d_image.load_state_dict(state_dict['optimizer_d_image'])
            if self.train_params["use_video_discriminator"]:
                self.optimizer_d_video.load_state_dict(state_dict['optimizer_d_video'])
            try:
                start_epoch, epoch_iter = np.loadtxt(self.iter_path, delimiter=',',
                                                     dtype=int)
            except FileNotFoundError:
                start_epoch, epoch_iter = 1, 0
        else:
            start_epoch, epoch_iter = 1, 0
        return start_epoch, epoch_iter

    def update_model(self, data):
        self.optimizer_vae.zero_grad(set_to_none=True)
        self.optimizer_gnn.zero_grad(set_to_none=True)
        if self.train_params["use_image_discriminator"]:
            self.optimizer_d_image.zero_grad(set_to_none=True)
        if self.train_params["use_video_discriminator"]:
            self.optimizer_d_video.zero_grad(set_to_none=True)
        self.generated_data, self.loss_dict_gen, self.loss_dict_dis_image, self.loss_dict_dis_video = self.c2m(data)
        loss = torch.tensor(0., device=self.generated_data["generated"].device)
        loss_weights = self.train_params["loss_weights"]
        for key in self.loss_dict_gen:
            loss += self.loss_dict_gen[key] * loss_weights[key]
        self.loss_dict_gen['total_gen'] = loss
        if self.train_params["use_image_discriminator"]:
            loss_d_image = (self.loss_dict_dis_image.get("d_real", 0) + self.loss_dict_dis_image.get("d_fake", 0)) * 0.5
            self.loss_dict_dis["total_image_dis"] = loss_d_image
            loss_d_image.backward()
        if self.train_params["use_video_discriminator"]:
            loss_d_video = (self.loss_dict_dis_video.get("d_real", 0) + self.loss_dict_dis_video.get("d_fake", 0)) * 0.5
            self.loss_dict_dis["total_video_dis"] = loss_d_video
            loss_d_video.backward()
        loss.backward()
        self.optimizer_vae.step()
        self.optimizer_gnn.step()
        if self.train_params["use_image_discriminator"]:
            self.optimizer_d_image.step()
        if self.train_params["use_video_discriminator"]:
            self.optimizer_d_video.step()
        self._detach_losses()
        self._detach_generated()
        self.loss_dict_gen.update(self.loss_dict_dis)

    def _detach_losses(self):
        r"""Detach all logging variables to prevent potential memory leak."""
        for loss_name in self.loss_dict_gen:
            self.loss_dict_gen[loss_name] = self.loss_dict_gen[loss_name].detach()
        if self.train_params["use_image_discriminator"]:
            for loss_name in self.loss_dict_dis:
                self.loss_dict_dis[loss_name] = self.loss_dict_dis[loss_name].detach()
        if self.train_params["use_video_discriminator"]:
            for loss_name in self.loss_dict_dis:
                self.loss_dict_dis[loss_name] = self.loss_dict_dis[loss_name].detach()

    def _detach_generated(self):
        r"""Detach all logging variables to prevent potential memory leak."""
        for tensor_name in self.generated_data:
            self.generated_data[tensor_name] = self.generated_data[tensor_name].detach().contiguous()

    def _generate_eval(self):
        """
        Compute evaluation
        :return: input frames and generated frames
        """
        with torch.no_grad():
            self.c2m.eval()
            val_batch = iter(self.val_data_loader).next()
            # Read data
            val_batch = val_batch.data
            del val_batch['complete_list']
            for k, v in val_batch.items():
                val_batch[k] = v.to(self.local_rank)
            if not self.train_params["use_pre_processed_of"]:
                val_batch.update(self.compute_flow(val_batch))
            else:
                val_batch["input_of"] = None
                val_batch["input_occ"] = None
            val_batch_size = val_batch["video"].shape[0]
            z_m = torch.FloatTensor(val_batch_size, 1024).normal_(0, 1).to(self.local_rank)
            output_dict_eval = self.c2m.inference(val_batch["video"], val_batch["bg_mask"], val_batch["fg_mask"],
                                                  val_batch["instance_mask"], val_batch["input_of"],
                                                  val_batch["input_occ"], val_batch["tracking_gnn"], None, z_m)
        return output_dict_eval, val_batch

    def _gather_data(self):
        """
        fetch all tensor into master node
        :return:
        """
        generated_data = {"generated": utils.dist_all_gather_tensor(self.generated_data["generated"].detach()),
                          "generated_sparse": utils.dist_all_gather_tensor(
                              self.generated_data["generated_sparse"].detach()),
                          "generated_sparse_occ": utils.dist_all_gather_tensor(
                              self.generated_data["generated_sparse_occ"].detach()),
                          "dense_motion_bw": utils.dist_all_gather_tensor(
                              self.generated_data["dense_motion_bw"].detach().contiguous()),
                          "dense_motion_fw": utils.dist_all_gather_tensor(
                              self.generated_data["dense_motion_fw"].detach().contiguous())
                          if self.train_params["use_fw_of"] else None,
                          "occlusion_bw": utils.dist_all_gather_tensor(
                              self.generated_data["occlusion_bw"].detach().contiguous()),
                          "occlusion_fw": utils.dist_all_gather_tensor(
                              self.generated_data["occlusion_fw"].detach().contiguous())
                          if self.train_params["use_fw_of"] else None,
                          "sparse_occ_bw": utils.dist_all_gather_tensor(
                              self.generated_data["sparse_occ_bw"].detach().contiguous()),
                          "sparse_occ_fw": utils.dist_all_gather_tensor(
                              self.generated_data["sparse_occ_fw"].detach().contiguous())
                          if self.train_params["use_fw_of"] else None,
                          "sparse_motion_bw": utils.dist_all_gather_tensor(
                              self.generated_data["sparse_motion_bw"].detach().contiguous()),
                          "sparse_motion_fw": utils.dist_all_gather_tensor(
                              self.generated_data["sparse_motion_fw"].detach().contiguous())
                          if self.train_params["use_fw_of"] else None,
                          "sparse_motion_bin": utils.dist_all_gather_tensor(
                              self.generated_data["sparse_motion_bin"].detach().contiguous())}
        return generated_data

    def save_checkpoint(self):
        print(
            'saving the model at the end of epoch %d, iters %d' % (
                self.current_epoch, self.epoch_iter))
        checkpoint = {'c2m': self.c2m.state_dict(), 'optimizer_gnn': self.optimizer_gnn.state_dict(),
                      'optimizer': self.optimizer_vae.state_dict()}
        if self.train_params["use_image_discriminator"]:
            checkpoint.update({'optimizer_d_image': self.optimizer_d_image.state_dict()})
        if self.train_params["use_video_discriminator"]:
            checkpoint.update({'optimizer_d_video': self.optimizer_d_video.state_dict()})
        checkpoint_path_c2m = self.sampledir + '/latest_c2m_model.pth.tar'
        print("c2m model saved to {}".format(checkpoint_path_c2m))
        torch.save(checkpoint,
                   checkpoint_path_c2m)
        np.savetxt(self.iter_path, (self.current_epoch + 1, self.epoch_iter), delimiter=',',
                   fmt='%d')
