import torch
import numpy as np
import os
from torch.utils.data import Dataset
import torchvision.transforms as transform
from PIL import Image
import glob
from itertools import permutations
from torch_geometric.data import Data
from utils import read_flow
from pathlib import Path


input_transform = transform.Compose([
    transform.ToTensor()])

image_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def replace_index_and_read_frame(frame_dir, frame_name, size, frame_type, frame_suffix= None,
                                 scene_info=None, tracking=False):
    if frame_type == "image":
        new_frame_dir = os.path.join(frame_dir, frame_name)
        load_type = Image.BICUBIC
    elif frame_type == "seg_mask":
        new_frame_dir = os.path.join(frame_dir, frame_name[:-4] + frame_suffix)
        load_type = Image.NEAREST
    else:
        new_frame_dir = os.path.join(frame_dir, frame_name[:-4] + frame_suffix)
        load_type = Image.NEAREST
    try:
        frame = Image.open(new_frame_dir)
        if size is not None:
            frame = frame.resize((size[1], size[0]), load_type)
        if frame_type == "image":
            return input_transform(frame)
        elif frame_type == "seg_mask":
            seg_mask = input_transform(frame) * 255
            seg_mask_fg = torch.cat([seg_mask == i for i in range(11, 20)], 0)
            seg_mask_bg = torch.cat([seg_mask == i for i in range(0, 11)], 0)
            seg_mask_bg = seg_mask_bg.contiguous().type(torch.FloatTensor)
            seg_mask_fg = seg_mask_fg.contiguous().type(torch.FloatTensor)
            return seg_mask_fg, seg_mask_bg
        else:
            inst_mask = input_transform(np.array(frame))
            if tracking:
                mask_traj = torch.zeros(size=(size[0], size[1])).float()
                for instance_id in scene_info:
                    mask_traj = torch.where(torch.BoolTensor(inst_mask == instance_id),
                                            torch.ones(size=(size[0], size[1])).float(),
                                            mask_traj)
                return mask_traj
            else:
                return inst_mask

    except FileNotFoundError as ex:
        print(f'Exception: {ex} \n origin_dir: {frame_dir} \n new_dir: {new_frame_dir}')


def read_video(frame_dir, size, num_frames, frame_type, video_images, frame_suffix):
    video_images = list(map(lambda x: x.replace('0/seq3-drone_0000999.jpg', '1000/seq3-drone_0001000.jpg'),
                            video_images))

    if frame_type == "image":
        return {"video": torch.stack([replace_index_and_read_frame(frame_dir, frame_path, size, frame_type,
                                                                   frame_suffix) for frame_path in video_images],
                                     dim=1)}
    elif frame_type == "seg_mask":
        fg_samples = []
        bg_samples = []
        for frame_path in video_images:
            mask_volume_fg, mask_volume_bg = replace_index_and_read_frame(frame_dir, frame_path, size, frame_type,
                                                                          frame_suffix)
            fg_samples.append(mask_volume_fg)
            bg_samples.append(mask_volume_bg)
        return {"bg_mask": torch.stack(bg_samples, dim=1), "fg_mask": torch.stack(fg_samples, dim=1)}
    elif frame_type == "inst_mask":
        return {"instance_mask": torch.stack([replace_index_and_read_frame(frame_dir, frame_path, size, frame_type,
                                                                           frame_suffix, None,
                                                                           False) for frame_path in video_images],
                                             dim=1)}
    else:
        print(f"frame type error. {frame_type} is not a valid frame type")
        exit(0)


def load_scene_info(scene, num_frames, size, config):
    source_frames_nodes_features = []
    source_frames_nodes_instance_ids = []
    target_frames_nodes_instance_ids = []
    target_frames_nodes_roi = []
    source_frames_nodes_roi = []
    source_frames_nodes_roi_padded = []
    source_frames_nodes_barycenter = []
    target_frames_nodes_barycenter = []
    target_frames_nodes_displacement = []
    source_frames_nodes_size = []
    target_frames_nodes_size = []
    targets_theta = []
    for scene_instance in glob.glob(scene + "*.txt"):
        with open(scene_instance, "r") as file_instance:
            source_frames_node_instance_ids = []
            source_frames_node_size = []
            target_frames_node_size = []
            source_frames_node_features = []
            target_frames_node_barycenter = []
            target_frames_node_displacement = []
            source_frames_node_barycenter = []
            source_frames_node_roi_padded = []
            source_frames_node_roi = []
            target_frames_node_roi = []
            inst_info = file_instance.read().splitlines()[:num_frames]
            target_frames_node_instance_ids = []
            frames_theta = []
            for idx, frame_info in enumerate(inst_info):
                frame_info = frame_info.split(",")
                x_l = float(frame_info[0]) / 1920 * size[1]
                x_l_padded = max(x_l - 15, 0)
                x_r = (float(frame_info[0]) + float(frame_info[2])) / 1920 * size[1]
                x_r_padded = min(x_r + 15, size[1])
                y_t = float(frame_info[1]) / 1080 * size[0]
                y_t_padded = max(y_t - 10, 0)
                y_b = (float(frame_info[1]) + float(frame_info[3])) / 1080 * size[0]
                y_b_padded = min(y_b + 10, size[0])
                bbox_size = np.array([float(frame_info[3]) / 1080, float(frame_info[2]) / 1920])  # y,x
                if 0 in bbox_size:
                    break
                roi = [x_l, x_r, y_t, y_b]
                roi_padded = [x_l_padded, x_r_padded, y_t_padded, y_b_padded]
                x_center = (x_l + x_r) / 2
                x_center_norm = x_center / size[1]
                y_center = (y_t + y_b) / 2
                y_center_norm = y_center / size[0]
                roi_barycenter = np.array([y_center_norm * 2 - 1, x_center_norm * 2 - 1])
                if idx < config["train_params"]["num_input_frames"]:
                    source_frames_node_features.append([y_center_norm * 2 - 1, x_center_norm * 2 - 1,
                                                        bbox_size[0], bbox_size[1]] +
                                                       list(np.eye(19)[int(frame_info[-1]) // 1000]))
                    source_frames_node_instance_ids.append(int(frame_info[-1]))
                    source_frames_node_barycenter.append(roi_barycenter)
                    source_frames_node_roi.append(roi)
                    source_frames_node_roi_padded.append(roi_padded)
                    source_frames_node_size.append(bbox_size)
                else:
                    target_frames_node_size.append(bbox_size)
                    displacement = (source_frames_node_barycenter[-1] - roi_barycenter)
                    target_frames_node_barycenter.append(roi_barycenter)
                    target_frames_node_displacement.append(displacement)
                    target_frames_node_roi.append(roi)
                    target_frames_node_instance_ids.append(int(frame_info[-1]))
                    target_scale = source_frames_node_size[-1] / bbox_size
                    frames_theta.append([target_scale[1], 0, displacement[1], 0, target_scale[0], displacement[0]])
            if len(frames_theta) != 5:
                continue
            target_frames_nodes_roi.append(target_frames_node_roi)
            target_frames_nodes_instance_ids.append(target_frames_node_instance_ids)
            source_frames_nodes_barycenter.append(source_frames_node_barycenter)
            source_frames_nodes_roi_padded.append(source_frames_node_roi_padded)
            source_frames_nodes_roi.append(source_frames_node_roi)
            target_frames_nodes_barycenter.append(target_frames_node_barycenter)
            target_frames_nodes_displacement.append(target_frames_node_displacement)
            source_frames_nodes_features.append(source_frames_node_features)
            source_frames_nodes_instance_ids.append(source_frames_node_instance_ids)
            source_frames_nodes_size.append(source_frames_node_size)
            target_frames_nodes_size.append(target_frames_node_size)
            targets_theta.append(frames_theta)
    len_pre_pad = len(source_frames_nodes_features)
    edge_index = list(permutations(range(len_pre_pad), 2))
    if not edge_index:
        edge_index = [[0, 0]]
    num_nodes = torch.IntTensor([len_pre_pad])
    source_frames_nodes_instance_ids = torch.LongTensor(source_frames_nodes_instance_ids)
    target_frames_nodes_instance_ids = torch.LongTensor(target_frames_nodes_instance_ids)
    tracking_ids = torch.cat([source_frames_nodes_instance_ids,
                              target_frames_nodes_instance_ids], dim=1).permute(1, 0)
    data = Data(x=torch.FloatTensor(source_frames_nodes_features),
                y=torch.FloatTensor(np.array(target_frames_nodes_barycenter)),
                num_real_nodes=num_nodes,
                source_frames_nodes_roi=torch.FloatTensor(source_frames_nodes_roi),
                source_frames_nodes_roi_padded=torch.FloatTensor(source_frames_nodes_roi_padded),
                target_frames_nodes_roi=torch.FloatTensor(target_frames_nodes_roi),
                source_frames_nodes_instance_ids=source_frames_nodes_instance_ids,
                target_frames_nodes_instance_ids=target_frames_nodes_instance_ids,
                targets_barycenter=torch.FloatTensor(np.array(target_frames_nodes_barycenter)),
                targets_displacement=torch.FloatTensor(np.array(target_frames_nodes_displacement)),
                targets_theta=torch.FloatTensor(targets_theta),
                edge_index=torch.tensor(edge_index, dtype=torch.long).permute(1, 0))
    return tracking_ids, data


def load_tracking_mask(instance_dir, tracking_dir, size, num_frames, config, video_images, frame_suffix):
    video_images = list(map(lambda x: x.replace('0/seq3-drone_0000999.jpg', '1000/seq3-drone_0001000.jpg'),
                            video_images))
    scene_info, gnn_data = load_scene_info(tracking_dir, num_frames, size, config)
    instance_frames = torch.stack([replace_index_and_read_frame(instance_dir, frame_path, size, "inst_mask",
                                                                frame_suffix, scene_info[i],
                                                                True) for i, frame_path in enumerate(video_images)], dim=1)
    return {"tracking_mask": instance_frames, "tracking_gnn": gnn_data}


def complete_full_list(video_images, num_frames, output_name):
    dir_list = [video_images[0][:-4] + "_" + output_name]
    dir_list.extend([video_images[0][:-4] + video_images[i][-12:-4] + "_" + output_name for i in range(1, len(video_images))])
    return dir_list


def load_instance(mask_dir, size):
    mask = Image.open(mask_dir)
    mask = mask.resize((size[1], size[0]), Image.NEAREST)
    mask = input_transform(np.array(mask))
    return mask


def load_optical_flow(optical_flow_dir, size):
    optical_flow = read_flow(optical_flow_dir)
    optical_flow = torch.from_numpy(optical_flow)
    h, w, c = optical_flow.size()
    if [h, w] != size:
        resize = transform.Resize((size[0], size[1]))
        flow = resize(optical_flow.permute(2, 0, 1)) * size[0] / h
    else:
        flow = optical_flow.permute(2, 0, 1)
    # c, h, w = flow.size()
    # flow = torch.cat([flow[0:1, :, :] / ((w - 1.0) / 2.0), flow[1:2, :, :] / ((h - 1.0) / 2.0)], dim=0)
    return flow


def load_optical_flow_occlusion_mask(bw_occlusion_dir, bw_occlusion_suffix, bw_optical_flow_dir, bw_optical_flow_suffix,
                                     video_images, size, num_frame):
    backward_optical_flows = []
    backward_occlusion_masks = []
    for i in range(1, num_frame):
        new_dir_backward_optical_flow = bw_optical_flow_dir + "/" + video_images[0][:-4] +\
                                        video_images[i][-12:-4] + bw_optical_flow_suffix
        new_dir_backward_occlusion_mask = bw_occlusion_dir + "/" + video_images[0][:-4] +\
                                          video_images[i][-12:-4] + bw_occlusion_suffix
        backward_optical_flows.append(load_optical_flow(new_dir_backward_optical_flow, size))
        backward_occlusion_masks.append(load_instance(new_dir_backward_occlusion_mask, size))

    return {"target_bw_of": torch.stack(backward_optical_flows, dim=1),
            "target_bw_occ": torch.stack(backward_occlusion_masks, dim=1)}


def clip_mask(mask):
    one_ = torch.ones_like(mask)
    zero_ = torch.zeros_like(mask)
    return torch.where(mask > 0.5, one_, zero_)


def load_all_image_paths(image_dir, imgs_names):
    file_list =\
        sorted(set(["/".join(fname.split("/")[-2:]) for fname in glob.glob(f"{str(image_dir)}/*/*")
                    if fname.split("/")[-1][-4:] == ".jpg"]))
    image_dir = Path(image_dir)
    videos = {}
    for img_name in imgs_names:
        j = file_list.index(img_name.strip())
        image = []
        for k in range(j, j + 18, 3):
            if k >= len(file_list):
                break
            full_image_path = image_dir / file_list[k].split("/")[0]/file_list[k].split("/")[1]
            if not os.path.isfile(full_image_path):
                break
            image.append(file_list[k])
        if len(image) != 6:
            continue
        videos[image[0]] = image
    return videos


class Mvtid(Dataset):
    def __init__(self, dataset_type=None, images_path=None, segmasks_path=None, instances_path=None, tracking_path=None,
                 bw_occ_path=None, bw_of_path=None, datalist=None, size=(128, 128), split="train",
                 segmask_suffix='gtFine_labelIds.png', instance_suffix="gtFine_instanceIds.png", bw_occ_suffix="",
                 bw_of_suffix="", config=None):
        self.dataset_type = dataset_type
        self.num_frames = config["train_params"]["num_input_frames"] + config["train_params"]["num_predicted_frames"]
        self.num_input_frames = config["train_params"]["num_input_frames"]
        self.num_predicted_frames = config["train_params"]["num_predicted_frames"]
        self.dataroot = config["dataset_params"]["root"]
        self.split = split
        self.images_root = os.path.join(self.dataroot, self.dataset_type, images_path, self.split)
        self.datalist = open(os.path.join(self.dataroot, datalist)).readlines()
        self.size = size
        self.segmask_root = os.path.join(self.dataroot, self.dataset_type, segmasks_path, self.split)
        self.segmask_suffix = segmask_suffix
        self.instance_root = os.path.join(self.dataroot, self.dataset_type, instances_path, self.split)
        self.instance_suffix = instance_suffix
        self.tracking_root = os.path.join(self.dataroot, self.dataset_type, tracking_path, self.split)
        self.bwd_occlusion_root = os.path.join(self.dataroot, self.dataset_type, bw_occ_path, self.split)
        self.bwd_occlusion_suffix = bw_occ_suffix
        self.bwd_optical_flow_root = os.path.join(self.dataroot, self.dataset_type, bw_of_path, self.split)
        self.bwd_optical_flow_suffix = bw_of_suffix
        self.config = config
        self.all_image_paths = load_all_image_paths(self.images_root, self.datalist)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        out_dict = {}
        out_dict.update(read_video(self.images_root, self.size, self.num_frames, "image",
                                   self.all_image_paths[self.datalist[idx].strip()], None))
        tracking_dir = os.path.join(self.tracking_root, self.datalist[idx].strip()[0:-4])
        out_dict.update(read_video(self.segmask_root, self.size, self.num_frames, "seg_mask",
                                   self.all_image_paths[self.datalist[idx].strip()], self.segmask_suffix))
        out_dict.update(read_video(self.instance_root, self.size, self.num_frames, "inst_mask",
                                   self.all_image_paths[self.datalist[idx].strip()], self.instance_suffix))
        out_dict.update(load_tracking_mask(self.instance_root, tracking_dir, self.size, self.num_frames, self.config,
                                           self.all_image_paths[self.datalist[idx].strip()], self.instance_suffix))

        if self.config["train_params"]["use_pre_processed_of"]:
            out_dict.update(load_optical_flow_occlusion_mask(self.bwd_occlusion_root, self.bwd_occlusion_suffix,
                                                             self.bwd_optical_flow_root, self.bwd_optical_flow_suffix,
                                                             self.all_image_paths[self.datalist[idx].strip()],
                                                             self.size, self.num_frames))
        out_dict["complete_list"] = complete_full_list(self.all_image_paths[self.datalist[idx].strip()], self.num_frames, 'gt.png')
        return out_dict

    def get_scene_num_instances(self):
        scenes_num_instances = []
        files = glob.glob(self.tracking_root + "/*/*.txt")
        files = [file[:-10] for file in files]
        files = list(set(files))
        for scene in files:
            scenes_num_instances.append(len(glob.glob(scene + "*.txt")))
        return scenes_num_instances
