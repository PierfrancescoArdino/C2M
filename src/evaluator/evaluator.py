import torch
from utils import utils
import modules.networks.yolo_v3.models as yolo_model
import modules.networks.yolo_v3.utils.utils as yolo_utils
from utils.utils_yolov3 import compute_detection
import matplotlib.pyplot as plt
import os
import numpy as np
import statistics
from utils.fid import calculate_fid
from utils.fvd.score import fvd as fvd_score
from sklearn.metrics import f1_score, accuracy_score
from utils.visualizer import EvaluatorVisualizer
from collections import OrderedDict


class Evaluator(object):
    def __init__(self, cfg, opt, c2m, flownet, val_data_loader, local_rank):
        self.cfg = cfg
        self.c2m = c2m
        self.flownet = flownet
        self.opt = opt
        self.val_data_loader = val_data_loader
        self.local_rank = local_rank
        self.dataset_params = self.cfg["dataset_params"]
        self.train_params = self.cfg["train_params"]
        self.model_params = self.cfg["model_params"]
        self.visualizer_params = self.cfg["visualizer_params"]
        self.checkpoint_params = self.cfg["checkpoint_params"]
        self.test_params = self.cfg["test_params"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_yolo = yolo_model.Darknet("modules/networks/yolo_v3/config/yolov3.cfg",
                                             img_size=self.test_params["input_size"]).to(self.device)
        self.model_yolo.load_darknet_weights("modules/networks/yolo_v3/weights/yolov3.weights")
        self.model_yolo.eval()
        # Bounding-box colors
        self.cmap = plt.get_cmap("tab20b")
        self.colors = [self.cmap(i) for i in np.linspace(0, 1, 20)]
        self.classes = yolo_utils.load_classes("modules/networks/yolo_v3/data/coco.names")

        self.dataset = self.dataset_params["dataset"]
        self.suffix = '_' + self.cfg["suffix"]
        self.jobname = self.dataset + f'_{self.cfg["name"]}'
        self.which_epoch = self.test_params["which_epoch"]
        # whether to start training from an existing snapshot
        self.load = True
        if self.suffix != "_":
            self.sampledir = os.path.join(f'../{self.dataset}_test_results', self.jobname + self.suffix,
                                          str(self.which_epoch)+'_'+str(self.test_params["seed"]))
            self.visualizer_dir = os.path.join(f'../{self.dataset}_test_results', self.jobname + self.suffix)
        else:
            self.sampledir = os.path.join(f'../{self.dataset}_test_results', self.jobname,
                                          str(self.which_epoch)+'_'+str(self.test_params["seed"]))
            self.visualizer_dir = os.path.join(f'../{self.dataset}_test_results', self.jobname)

        if not os.path.exists(self.sampledir):
            os.makedirs(self.sampledir)
        self.visualizer = EvaluatorVisualizer(self.visualizer_dir, self.visualizer_params)
        # Create Folder for test images.
        self.output_image_dir = self.sampledir + '_images'
        self.output_image_dir_w_bbox = self.sampledir + '_images_w_bbox'
        self.output_gt_detector_dir = self.sampledir + '_images_detector_gt'
        self.output_pred_detector_dir = self.sampledir + '_images_detector_pred'
        self.output_image_dir_before = self.sampledir + '_images_before'
        self.output_bw_flow_dir = self.sampledir + '_bw_flow'
        self.output_fw_flow_dir = self.sampledir + '_fw_flow'

        self.output_bw_mask_dir = self.sampledir + '_bw_mask'
        self.output_fw_mask_dir = self.sampledir + '_fw_mask'
        self.gt_samples_fvd = None
        self.pred_samples_fvd = None
        self.gt_samples_fid = None
        self.pred_samples_fid = None
        self.batch_index_user_guidance_tensor = None
        self.complete_list = None
        self.mse_batches = []
        self.mse_normalized_batches = []
        self.gt_detected_images_batches = []
        self.pred_detected_images_batches = []
        self.batch_index_user_guidance = []
        self.iteration = 0
        if self.test_params["load_index_user_guidance"] and not self.test_params["custom_test"]:
            self.batch_index_user_guidance_tensor = torch.load(f"index_user_guidance_{self.dataset}.pt")

        utils.mkdirs([self.output_image_dir, self.output_image_dir_w_bbox, self.output_image_dir_before,
                      self.output_gt_detector_dir, self.output_pred_detector_dir, self.output_bw_flow_dir,
                      self.output_fw_flow_dir, self.output_fw_mask_dir, self.output_bw_mask_dir])

    def load_checkpoint(self):
        if self.load:
            model_name = f'../{self.jobname}/{self.which_epoch}_c2m_model.pth.tar'
            print("loading model from {}".format(model_name))
            state_dict = torch.load(model_name)
            test_dict =\
                {k: v for k, v in state_dict["c2m"].items()
                 if any(substring in k for substring in ["motion_encoder", "appearance_encoder", "generator"])}
            self.c2m.load_state_dict(test_dict, strict=True)

    def save_user_guidance(self):
        torch.save(torch.stack(self.batch_index_user_guidance), f"index_user_guidance_{self.dataset}.pt")

    def evaluate(self, idx, batch):
        if self.batch_index_user_guidance_tensor is not None:
            index_user_guidance = self.batch_index_user_guidance_tensor[idx].to(self.local_rank)
        else:
            index_user_guidance = None
        val_batch_size = batch["video"].shape[0]
        z_m = torch.FloatTensor(val_batch_size, 1024).normal_(0, 1).to(self.local_rank)
        output_dict_eval = self.c2m.inference(batch["video"], batch["bg_mask"], batch["fg_mask"],
                                              batch["instance_mask"], batch["input_of"], batch["input_occ"],
                                              batch["tracking_gnn"], index_user_guidance, z_m)
        #output_dict_eval.update({"index_user_guidance": index_user_guidance})
        self.batch_index_user_guidance.append(output_dict_eval["index_user_guidance"])
        return output_dict_eval

    def set_eval(self):
        self.c2m.eval()

    def compute_flow(self, batch):
        out_dict = {}
        t_in = self.test_params["num_input_frames"]
        t_out = self.test_params["num_predicted_frames"]
        input_optical_flows = []
        input_occlusion_masks = []
        target_bw_optical_flows = []
        target_bw_occlusion_masks = []
        target_fw_optical_flows = []
        target_fw_occlusion_masks = []
        b, _, _, h, w = batch["video"].size()
        # Compute forward optical flow between consecutive frames in input frames
        for i in range(t_in - 1):
            input_image_a = batch["video"][:, :, i, ...] * 2 - 1
            input_image_b = batch["video"][:, :, i + 1, ...] * 2 - 1
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
            input_image_a = batch["video"][:, :, t_in-1, ...] * 2 - 1
            input_image_b = batch["video"][:, :, t_in + i, ...] * 2 - 1
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
        out_dict["target_fw_of"] = torch.cat(target_fw_optical_flows, dim=2)
        out_dict["target_fw_occ"] = torch.cat(target_fw_occlusion_masks, dim=2)
        return out_dict

    def start_iteration(self, data, iteration):
        self.iteration = iteration
        self.complete_list = data['complete_list']
        del data["complete_list"]
        for k, v in data.items():
            data[k] = v.to(self.local_rank)
        if not self.test_params["use_pre_processed_of"]:
            data.update(self.compute_flow(data))
        else:
            data["input_of"] = None
            data["input_occ"] = None
        self.c2m.eval()
        return data

    def save_samples(self, data, generated_data, sample_number):
        if sample_number == 0:
            self.save_prediction_html(generated_data, data, [self.test_params["batch_size"] // 4, 4])
        self.visualizer.print_current_pred(self.iteration, generated_data, self.test_params["num_predicted_frames"],
                                           data["tracking_gnn"], self.test_params["input_size"])
        utils.save_samples(data["video"].cpu(), generated_data["generated"].cpu(),
                           generated_data["dense_motion_bw"].cpu(),
                           generated_data["dense_motion_fw"].cpu(),
                           generated_data["sparse_motion_bw"].cpu(),
                           generated_data["sparse_motion_fw"].cpu(),
                           generated_data["sparse_motion_bin"].cpu(),
                           generated_data["occlusion_bw"].cpu(),
                           generated_data["occlusion_fw"].cpu(),
                           data["target_bw_of"].cpu(),
                           data["target_fw_of"].cpu(),
                           data["target_bw_occ"].cpu(),
                           data["target_fw_occ"].cpu(),
                           self.iteration,
                           self.sampledir, self.cfg,
                           is_eval=True, use_mask=True, sample_number=sample_number)

        utils.save_images(self.output_image_dir, data["video"].cpu(), generated_data["generated"].cpu(),
                          self.complete_list, sample_number=sample_number)
        utils.save_images_w_bbox(self.output_image_dir_w_bbox, data["video"].cpu(),
                                 generated_data["generated"].cpu(), self.complete_list,
                                 self.cfg, data["tracking_gnn"], 0)
        video = data["video"].cpu().cpu().data.permute(0, 2, 3, 4, 1).numpy()
        utils.save_gif(video * 255, video.shape[1], [4, 4], self.sampledir + '/{:06d}_real.gif'.format(self.iteration))

        '''save flows'''
        utils.save_flows(self.output_fw_flow_dir, generated_data["dense_motion_fw"].cpu(), self.complete_list,
                         sample_number=sample_number)
        utils.save_flows(self.output_bw_flow_dir, generated_data["dense_motion_bw"].cpu(), self.complete_list,
                         sample_number=sample_number)

        '''save occlusion maps'''
        utils.save_occ_map(self.output_fw_mask_dir, generated_data["occlusion_fw"].cpu(), self.complete_list,
                           sample_number=sample_number)
        utils.save_occ_map(self.output_bw_mask_dir, generated_data["occlusion_bw"].cpu(), self.complete_list,
                           sample_number=sample_number)

    def save_prediction_html(self, generated_data, data, grid_size):
        visual_list = [('source_frames',
                        utils.tensor2im(data["video"][:, :, :self.test_params["num_input_frames"], ...].contiguous(),
                                        normalize=False, size=grid_size)),
                       ('target_frames',
                        utils.tensor2im(data["video"][:, :, self.test_params["num_input_frames"]:, ...].contiguous(),
                                        normalize=False, size=grid_size)),
                       ('predicted_frames', utils.tensor2im(generated_data["generated"], normalize=False,
                                                            size=grid_size)),
                       ('predicted_frames_sparse', utils.tensor2im(generated_data["generated_sparse"], normalize=False,
                                                                   size=grid_size)),
                       ('predicted_frames_sparse_occ', utils.tensor2im(generated_data["generated_sparse_occ"],
                                                                       normalize=False, size=grid_size)),
                       ('gt_target_bw_of', utils.tensor2flow(data["target_bw_of"], size=grid_size)),
                       ('gt_target_fw_of', utils.tensor2flow(data["target_fw_of"], size=grid_size)),
                       ('pred_dense_motion_bw', utils.tensor2flow(generated_data["dense_motion_bw"], size=grid_size)),
                       ('pred_dense_motion_fw', utils.tensor2flow(generated_data["dense_motion_fw"], size=grid_size)),
                       ('gt_target_bw_occ', utils.tensor2occ(data["target_bw_occ"], size=grid_size)),
                       ('gt_target_fw_occ', utils.tensor2occ(data["target_fw_occ"], size=grid_size)),
                       ('pred_occlusion_bw', utils.tensor2occ(generated_data["occlusion_bw"], size=grid_size)),
                       ('pred_occlusion_fw', utils.tensor2occ(generated_data["occlusion_fw"], size=grid_size)),
                       ('pred_sparse_motion_bw', utils.tensor2flow(generated_data["sparse_motion_bw"], size=grid_size)),
                       ('pred_sparse_motion_fw', utils.tensor2flow(generated_data["sparse_motion_fw"], size=grid_size)),
                       ('pred_sparse_occ_bw', utils.tensor2occ(generated_data["sparse_occ_bw"], size=grid_size)),
                       ('pred_sparse_occ_fw', utils.tensor2occ(generated_data["sparse_occ_fw"], size=grid_size)),
                       ('pred_sparse_motion_bin', utils.tensor2occ(generated_data["sparse_motion_bin"], size=grid_size))
                       ]
        visuals = OrderedDict(visual_list)
        self.visualizer.display_current_results(visuals, self.iteration)

    def compute_detection(self, batch, generated_data, sample_number):
        if self.test_params["custom_test"]:
            generated_data["index_user_guidance"] = torch.LongTensor(range(len(batch["tracking_gnn"].batch)))
        mse_batch, mse_normalized_batch, gt_detected_images, pred_detected_images =\
            compute_detection(batch["video"], generated_data["generated"], self.model_yolo, batch["tracking_gnn"],
                              self.classes, self.device, generated_data["index_user_guidance"],
                              self.complete_list, self.output_gt_detector_dir, self.output_pred_detector_dir)
        self.mse_batches.extend(mse_batch)
        self.mse_normalized_batches.extend(mse_normalized_batch)
        self.gt_detected_images_batches.extend(gt_detected_images)
        self.pred_detected_images_batches.extend(pred_detected_images)

    def fetch_metrics_data(self, batch, generated_data, sample_number):
        generated = torch.cat(
            [batch["video"][:, :, 0].unsqueeze(2).cpu(), generated_data["generated"].cpu()], dim=2)

        fake_fid = generated.permute(0, 2, 3, 4, 1).contiguous().view(
            generated.shape[0] * generated.shape[2], generated.shape[3],
            generated.shape[4], 3).cpu().numpy()
        fake_fvd = torch.cat([generated.permute(0, 2, 1, 4, 3).cpu(),
                              generated.permute(0, 2, 1, 4, 3).cpu().flip([2])], dim=1)
        gt_fid = batch["video"].permute(0, 2, 3, 4, 1).contiguous().view(
            batch["video"].shape[0] * batch["video"].shape[2], batch["video"].shape[3],
            batch["video"].shape[4], 3).cpu().numpy()
        gt_fvd = torch.cat([batch["video"].permute(0, 2, 1, 4, 3).cpu(),
                            batch["video"].permute(0, 2, 1, 4, 3).cpu().flip(
                                [2])], dim=1)

        if self.gt_samples_fvd is None:
            self.gt_samples_fvd = gt_fvd
            self.pred_samples_fvd = fake_fvd
            self.gt_samples_fid = gt_fid
            self.pred_samples_fid = fake_fid
        else:
            self.gt_samples_fvd = torch.cat([self.gt_samples_fvd, gt_fvd], dim=0)
            self.gt_samples_fid = np.concatenate([self.gt_samples_fid, gt_fid], axis=0)
            self.pred_samples_fvd = torch.cat([self.pred_samples_fvd, fake_fvd], dim=0)
            self.pred_samples_fid = np.concatenate([self.pred_samples_fid, fake_fid], axis=0)

    def generate_metrics(self):
        self.pred_detected_images_batches.extend([0 for _ in range(len(self.gt_detected_images_batches) - len(self.pred_detected_images_batches))])
        f1_score_detection = f1_score(self.gt_detected_images_batches, self.pred_detected_images_batches)
        accuracy_detection = accuracy_score(self.gt_detected_images_batches, self.pred_detected_images_batches)
        print('Computing FID...')
        fid_score = calculate_fid(self.gt_samples_fid, self.pred_samples_fid, False,
                                  self.cfg["test_params"]["batch_size"])
        print(f"fid_score: {fid_score}")
        print('Computing FVD...')
        print(self.gt_samples_fvd.shape)
        print(self.pred_samples_fvd.shape)
        fvd = fvd_score(self.gt_samples_fvd, self.pred_samples_fvd)
        print(f"fvd: {fvd}")
        self.write_metrics(f1_score_detection, accuracy_detection, fid_score, fvd)

    def write_metrics(self, f1_score_detection, accuracy_detection, fid_score, fvd):
        with open(os.path.join(self.sampledir, "results.txt"), "a") as f_out:
            print(f"f1 score {f1_score_detection}")
            f_out.write(f"f1 score {f1_score_detection}\n")
            print(f"accuracy score {accuracy_detection} gt_detection {sum(self.gt_detected_images_batches)}"
                  f" pred_detection{sum(self.pred_detected_images_batches)}")
            f_out.write(f"accuracy score {accuracy_detection} gt_detection {sum(self.gt_detected_images_batches)}"
                        f" pred_detection{sum(self.pred_detected_images_batches)}\n")
            print(f"mse_traj_loss {statistics.mean(self.mse_batches)}")
            f_out.write(f"mse_traj_loss {statistics.mean(self.mse_batches)}\n")
            print(f"mse_normalized_traj_loss {statistics.mean(self.mse_normalized_batches)}")
            f_out.write(f"mse_normalized_traj_loss {statistics.mean(self.mse_normalized_batches)}\n")
            print(f"fid_score: {fid_score}")
            f_out.write(f"fid_score: {fid_score} \n")
            print(f"fvd: {fvd}")
            f_out.write(f"fvd_score: {fvd} \n\n\n")
