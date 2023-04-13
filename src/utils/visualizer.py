import os
import time
import numpy as np
import scipy.misc
from io import BytesIO
from . import utils
from . import html


class Visualizer:
    def __init__(self, name, visualized_params):
        # self.opt = opt
        self.tf_log = visualized_params["tf_log"]
        self.win_size = visualized_params["display_winsize"]
        self.name = name
        self.base_folder = None
        self.use_html = visualized_params["use_html"]
        if self.tf_log:
            from torch.utils.tensorboard import SummaryWriter
            self.log_dir = os.path.join(self.name, 'logs')
            self.writer = SummaryWriter(self.log_dir)
        if self.use_html:
            self.web_dir_train = os.path.join(self.name, 'train', 'web')
            self.web_dir_eval = os.path.join(self.name, 'eval', 'web')
            self.img_dir_train = os.path.join(self.web_dir_train, 'images')
            self.img_dir_eval = os.path.join(self.web_dir_eval, 'images')
            print('create web directory %s...' % self.web_dir_train)
            print('create web directory %s...' % self.web_dir_eval)
            utils.mkdirs([self.web_dir_train, self.img_dir_train, self.web_dir_eval, self.img_dir_eval])
        self.log_name = os.path.join(self.name, 'loss_log.txt')
        self.gnn_name_train = os.path.join(self.name, 'gnn_log_train.txt')
        self.gnn_name_eval = os.path.join(self.name, 'gnn_log_eval.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        with open(self.gnn_name_train, "a") as gnn_file:
            now = time.strftime("%c")
            gnn_file.write('================ Training Trajectories (%s) ================\n' % now)

        with open(self.gnn_name_eval, "a") as gnn_file:
            now = time.strftime("%c")
            gnn_file.write('================ Eval Trajectories (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, phase="train"):
        img_dir = self.img_dir_train if phase == "train" else self.img_dir_eval
        web_dir = self.web_dir_train if phase == "train" else self.web_dir_eval
        if self.use_html:
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(img_dir, 'epoch%.3d_%s_%d.jpg' % (epoch, label, i))
                        utils.save_image(image_numpy[i], img_path)
                elif image_numpy is not None:
                    img_path = os.path.join(img_dir, 'epoch%.3d_%s.gif' % (epoch, label))
                    frames_fo_save = [np.uint8(image_numpy[frame_id]) for frame_id in
                                      range(image_numpy.shape[0])]
                    utils.save_image(frames_fo_save, img_path)
                else:
                    continue

            # update website
            webpage = html.HTML(web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            max_epoch = max(epoch - 10, 0) if phase == "train" else 0
            for n in range(epoch, max_epoch, -1):
                if not os.path.exists(os.path.join(img_dir, "epoch%.3d_%s.gif" % (n, list(visuals.keys())[0]))):
                    continue
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if image_numpy is None:
                        continue
                    img_path = 'epoch%.3d_%s.gif' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                if len(ims) < 6:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    def display_current_model(self, net, input_to_model):
        self.writer.add_graph(net, input_to_model)

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                value = float(value)
                self.writer.add_scalar(f"Train/{tag}", value, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.7f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.7f ' % (k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_gnn(self, traj, tracking_gnn, train_params, device, val_batch_size):
        with open(self.gnn_name, "a") as traj_file:
            import torch
            trajectory_normalized = (traj + 1) / 2
            num_real_nodes = torch.LongTensor(
                [tracking_gnn.num_real_nodes]) if isinstance(
                tracking_gnn.num_real_nodes, int) else tracking_gnn.num_real_nodes
            trajectory_real = trajectory_normalized * torch.LongTensor(
                [train_params["input_size"][0], train_params["input_size"][1]]).to(device[0])
            y = ((tracking_gnn.y_n + 1) / 2) * torch.LongTensor(
                [train_params["input_size"][0], train_params["input_size"][1]]).to(device[0])
            x = ((tracking_gnn.x[:, :2] + 1) / 2) * torch.LongTensor(
                [train_params["input_size"][0], train_params["input_size"][1]]).to(device[0])
            total_number_nodes = 0
            for index, num_real_node in enumerate(num_real_nodes):
                print(f"\n\n sample [{index}/{val_batch_size}] \n")
                traj_file.write(f"\n\n sample [{index}/{val_batch_size}] \n")
                for i in range(total_number_nodes, total_number_nodes + num_real_node):
                    message = f"x:{x[i].cpu().numpy()} -> real: {y[i].cpu().numpy()} \t pred:{trajectory_real[i].cpu().numpy()}"
                    print(message)
                    traj_file.write(f"{message}\n")
                total_number_nodes += int(num_real_node)
                traj_file.write("------------- \n\n\n\n")
            print("------------- \n\n\n\n")

    def print_current_pred(self, epoch, epoch_iter, output_dict, t=0, num_predicted_frames=5, tracking_gnn=None,
                          phase="train", size=None):
        file_name = self.gnn_name_train if phase == "train" else self.gnn_name_eval
        with open(file_name, "a") as traj_file:
            import torch
            num_real_nodes = torch.LongTensor(
                [tracking_gnn.num_real_nodes]) if isinstance(
                tracking_gnn.num_real_nodes, int) else tracking_gnn.num_real_nodes
            total_number_nodes = 0
            message = '(epoch: %d, iters: %d, time: %.7f) ' % (epoch, epoch_iter, t)
            traj_file.write(f"{message}\n")
            object_id = tracking_gnn.source_frames_nodes_instance_ids[:, -1]
            for index, num_real_node in enumerate(num_real_nodes):
                for i in range(total_number_nodes, total_number_nodes + num_real_node):
                    for t in range(num_predicted_frames):
                        message = f"sample:{index} object_id: {object_id[i]} t: {t} -> " \
                                  f"affine_pred: {[output_dict[f'theta_{t}'][i].data.cpu().numpy().tolist()]} " \
                                  f"affine_gt: {[tracking_gnn.targets_theta[i][t].data.cpu().numpy().tolist()]}"
                        traj_file.write(f"{message}\n")
                total_number_nodes += int(num_real_node)

    def print_results(self, results):
        message = '(test epoch)'
        for k, v in results.items():
            if v != 0:
                message += '%s: %.7f ' % (k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_gradient(self, epoch, i, gradients, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in gradients.items():
            if v:
                message += '%s: %.8f \n ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk


class EvaluatorVisualizer:
    def __init__(self, name, visualized_params):
        # self.opt = opt
        self.win_size = visualized_params["display_winsize"]
        self.name = name
        self.base_folder = None
        self.use_html = visualized_params["use_html"]
        if self.use_html:
            self.web_dir = os.path.join(self.name, 'test', 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            utils.mkdirs([self.web_dir, self.img_dir])
        self.traj_name = os.path.join(self.name, 'traj_log_test.txt')
        with open(self.traj_name, "w") as gnn_file:
            now = time.strftime("%c")
            gnn_file.write('================ Test Trajectories (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, batch):
        if self.use_html:
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.jpg' % (batch, label, i))
                        utils.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'batch%.3d_%s.gif' % (batch, label))
                    frames_fo_save = [np.uint8(image_numpy[frame_id]) for frame_id in
                                      range(image_numpy.shape[0])]
                    utils.save_image(frames_fo_save, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(0, batch):
                webpage.add_header('batch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'batch%.3d_%s.gif' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                if len(ims) < 6:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    def print_current_pred(self, batch, output_dict, num_predicted_frames=5, tracking_gnn=None, size=None):
        with open(self.traj_name, "a") as traj_file:
            import torch
            num_real_nodes = torch.LongTensor(
                [tracking_gnn.num_real_nodes]) if isinstance(
                tracking_gnn.num_real_nodes, int) else tracking_gnn.num_real_nodes
            total_number_nodes = 0
            message = '(batch: %d) ' % batch
            traj_file.write(f"{message}\n")
            object_id = tracking_gnn.source_frames_nodes_instance_ids[:, -1]
            for index, num_real_node in enumerate(num_real_nodes):
                for i in range(total_number_nodes, total_number_nodes + num_real_node):
                    for t in range(num_predicted_frames):
                        message = f"sample:{index} object_id: {object_id[i]} t: {t} -> " \
                                  f"affine_pred: {[output_dict[f'theta_{t}'][i].data.cpu().numpy().tolist()]} " \
                                  f"affine_gt: {[tracking_gnn.targets_theta[i][t].data.cpu().numpy().tolist()]}"
                        traj_file.write(f"{message}\n")
                total_number_nodes += int(num_real_node)
