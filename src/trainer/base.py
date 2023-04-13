import os
from utils.visualizer import Visualizer
import utils
import time
import torch.distributed as dist
from collections import OrderedDict
from yaml import dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class BaseTrainer(object):
    def __init__(self, cfg, opt, c2m, flownet, optimizer_vae, optimizer_gnn, optimizer_d_image, optimizer_d_video,
                 scheduler_vae, scheduler_gnn, scheduler_d_image, scheduler_d_video, train_data_loader, val_data_loader,
                 local_rank):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.c2m = c2m
        self.flownet = flownet
        self.opt = opt
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
        self.dataset_params = self.cfg["dataset_params"]
        self.train_params = self.cfg["train_params"]
        self.model_params = self.cfg["model_params"]
        self.visualizer_params = self.cfg["visualizer_params"]
        self.checkpoint_params = self.cfg["checkpoint_params"]
        self.workspace = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')
        self.dataset = self.dataset_params["dataset"]
        self.jobname = self.dataset + f'_{self.cfg["name"]}'
        self.sampledir = os.path.join(self.workspace, self.jobname)
        self.iter_path = os.path.join(self.sampledir, "iter.txt")
        self.parameterdir = self.sampledir + '/params'
        if len(opt.device_ids) > 1:
            self.multiplier = dist.get_world_size()
        else:
            self.multiplier = 1
        if self.local_rank == 0:
            self.visualizer = Visualizer(self.sampledir, self.visualizer_params)
            if not os.path.exists(self.parameterdir):
                os.makedirs(self.parameterdir)
            # Write parameters setting file
            if os.path.exists(self.parameterdir):
                utils.save_parameters(self.parameterdir, self.jobname, self.opt, self.train_params["continue_train"],
                                      None)
                with open(os.path.join(self.parameterdir, "config.txt"), 'w') as yaml_file:
                    dump(self.cfg, yaml_file, default_flow_style=False, Dumper=Dumper)
        self.dataset_size = len(self.train_data_loader) * self.train_params["batch_size"]
        self.total_steps = 0
        self.epoch_iter = 0
        self.display_delta = None
        self.print_delta = None
        self.eval_delta = None
        self.current_iteration = 0
        self.current_epoch = 0
        self.start_iteration_time = None
        self.start_epoch_time = None
        self.elapsed_iteration_time = 0
        self.time_iteration = None
        self.time_epoch = None
        self.loss_dict_gen = self.loss_dict_dis_image = self.loss_dict_dis_video = {}
        self.generated_data = {}

    def initialize_deltas(self, start_epoch, epoch_iter):
        self.total_steps = (start_epoch - 1) * self.dataset_size + epoch_iter
        self.display_delta = self.total_steps % self.visualizer_params["display_freq"]
        self.print_delta = self.total_steps % self.visualizer_params["print_freq"]
        self.eval_delta = self.total_steps % self.train_params["eval_freq"]

    def start_of_epoch(self, current_epoch):
        r"""Things to do before an epoch.
        Args:
            current_epoch (int): Current number of epoch.
        """
        self.current_epoch = current_epoch
        self.start_epoch_time = time.time()

    def start_of_iteration(self, data):
        data = self._start_of_iteration(data)
        # torch.cuda.synchronize()
        self.start_iteration_time = time.time()
        return data

    def end_of_iteration(self, data):
        self.total_steps += (self.train_params["batch_size"] * self.multiplier)
        self.epoch_iter += (self.train_params["batch_size"] * self.multiplier)
        if self.total_steps % self.visualizer_params["print_freq"] == self.print_delta:
            errors = {k: utils.dist_all_reduce_tensor(v).item() for k, v in self.loss_dict_gen.items()}
            if self.local_rank == 0:
                t = (time.time() - self.start_iteration_time)
                self.visualizer.print_current_errors(self.current_epoch, self.epoch_iter, errors, t)
                self.visualizer.print_current_pred(self.current_epoch, self.epoch_iter, self.generated_data, t,
                                                   self.train_params["num_predicted_frames"],
                                                   data["tracking_gnn"], "train", self.train_params["input_size"])
                self.visualizer.plot_current_errors(errors, self.total_steps)
        if self.total_steps % self.visualizer_params["display_freq"] == self.display_delta:
            for k, v in data.items():
                if k not in ["tracking_gnn", "complete_list"]:
                    if v is not None:
                        data[k] = utils.dist_all_gather_tensor(v.detach()).cpu()
            generated_data = self._gather_data()
            if self.local_rank == 0:
                self.save_prediction(generated_data, data, "train", self.visualizer_params["grid_size"])
        if self.total_steps % self.train_params["eval_freq"] == self.eval_delta:
            if self.local_rank == 0:
                eval_prediction, val_batch = self._generate_eval()
                self.save_prediction(eval_prediction, val_batch, "eval", [4, 4])
                self.visualizer.print_current_pred(self.current_epoch, self.epoch_iter, eval_prediction, 0,
                                                   self.train_params["num_predicted_frames"],
                                                   val_batch["tracking_gnn"], "eval", self.train_params["input_size"])

    def end_of_epoch(self):
        r"""Things to do after an epoch.
        """
        self.scheduler_vae.step()
        self.scheduler_gnn.step()
        if self.train_params["use_image_discriminator"]:
            self.scheduler_d_image.step()
        if self.train_params["use_video_discriminator"]:
            self.scheduler_d_video.step()
        if self.current_epoch % self.checkpoint_params["save_epoch_freq"] == 0:
            if self.local_rank == 0:
                self.save_checkpoint()
        if self.local_rank == 0:
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (self.current_epoch, self.train_params["num_epochs"],
                   time.time() - self.start_epoch_time))
            print(f"Learning rates: \n VAE {self.scheduler_vae.get_last_lr()} "
                  f"\n GNN {self.scheduler_gnn.get_last_lr()}")
            if self.train_params["use_image_discriminator"]:
                print(f"\n Dis {self.scheduler_d_image.get_last_lr()}")
            if self.train_params["use_video_discriminator"]:
                print(f"\n Dis {self.scheduler_d_video.get_last_lr()}")
        self.epoch_iter = self.epoch_iter % self.dataset_size

    def save_prediction(self, generated_data, data, phase, grid_size):
        """
        Save prediction as html
        :param generated_data: output from generator
        :param data: input data
        :param phase: train or test
        :param grid_size: html grid size
        :return:
        """
        visual_list = [('source_frames',
                        utils.tensor2im(data["video"][:, :, :self.train_params["num_input_frames"], ...].contiguous(),
                                        normalize=False, size=grid_size)),
                       ('target_frames',
                        utils.tensor2im(data["video"][:, :, self.train_params["num_input_frames"]:, ...].contiguous(),
                                        normalize=False, size=grid_size)),
                       ('predicted_frames', utils.tensor2im(generated_data["generated"], normalize=False,
                                                            size=grid_size)),
                       ('predicted_frames_sparse', utils.tensor2im(generated_data["generated_sparse"], normalize=False,
                                                                   size=grid_size)),
                       ('predicted_frames_sparse_occ', utils.tensor2im(generated_data["generated_sparse_occ"],
                                                                       normalize=False, size=grid_size)),
                       ('gt_target_bw_of', utils.tensor2flow(data["target_bw_of"], size=grid_size)),
                       ('gt_target_fw_of', utils.tensor2flow(data["target_fw_of"], size=grid_size))
                       if self.train_params["use_fw_of"] else None,
                       ('pred_dense_motion_bw', utils.tensor2flow(generated_data["dense_motion_bw"], size=grid_size)),
                       ('pred_dense_motion_fw', utils.tensor2flow(generated_data["dense_motion_fw"], size=grid_size))
                       if self.train_params["use_fw_of"] else None,
                       ('gt_target_bw_occ', utils.tensor2occ(data["target_bw_occ"], size=grid_size)),
                       ('gt_target_fw_occ', utils.tensor2occ(data["target_fw_occ"], size=grid_size))
                       if self.train_params["use_fw_of"] else None,
                       ('pred_occlusion_bw', utils.tensor2occ(generated_data["occlusion_bw"], size=grid_size)),
                       ('pred_occlusion_fw', utils.tensor2occ(generated_data["occlusion_fw"], size=grid_size))
                       if self.train_params["use_fw_of"] else None,
                       ('pred_sparse_motion_bw', utils.tensor2flow(generated_data["sparse_motion_bw"], size=grid_size)),
                       ('pred_sparse_motion_fw', utils.tensor2flow(generated_data["sparse_motion_fw"], size=grid_size))
                       if self.train_params["use_fw_of"] else None,
                       ('pred_sparse_occ_bw', utils.tensor2occ(generated_data["sparse_occ_bw"], size=grid_size)),
                       ('pred_sparse_occ_fw', utils.tensor2occ(generated_data["sparse_occ_fw"], size=grid_size))
                       if self.train_params["use_fw_of"] else None,
                       ('pred_sparse_motion_bin', utils.tensor2occ(generated_data["sparse_motion_bin"], size=grid_size))
                       ]
        visuals = OrderedDict(visual_list)
        self.visualizer.display_current_results(visuals, self.current_epoch, phase)

    def save_checkpoint(self):
        pass

    def _start_of_iteration(self, data):
        r"""Operations to do before starting an iteration.
        Args:
            data (dict): Data used for the current iteration.
        Returns:
            (dict): Data used for the current iteration. They might be
                processed by the custom _start_of_iteration function.
        """
        return data

    def _end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Operations to do after an iteration.
        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current epoch number.
        """
        pass

    def _end_of_epoch(self, data, current_epoch, current_iteration):
        r"""Operations to do after an epoch.
        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current epoch number.
        """
        pass

    def _generate_eval(self):
        pass

    def _gather_data(self):
        pass
