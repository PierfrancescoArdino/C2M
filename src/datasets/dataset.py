from datasets.dataset_path import *


def get_training_set(config):
    dataset = config["dataset_params"]["dataset"]
    assert dataset in ['cityscapes', 'kitti', 'mvtid']

    if dataset == 'cityscapes':
        from datasets.cityscapes import Cityscapes
        train_dataset = Cityscapes(images_path=config["dataset_params"].get("train_images_path",""),
            segmasks_path=config["dataset_params"].get("train_segmasks_path",""),
            instances_path=config["dataset_params"].get("train_instances_path",""),
            tracking_path=config["dataset_params"].get("train_tracking_path",""),
            bw_occ_path=config["dataset_params"].get("train_bw_occ_path",""),
            bw_of_path=config["dataset_params"].get("train_bw_of_path",""),
            fw_occ_path=config["dataset_params"].get("train_fw_occ_path",""),
            fw_of_path=config["dataset_params"].get("train_fw_of_path",""),
            datalist=config["dataset_params"]["train_data_list"],
            size=config["train_params"]["input_size"], split='train',
            segmask_suffix=config["dataset_params"]["segmask_suffix"],
            instance_suffix=config["dataset_params"]["instance_suffix"],
            bw_occ_suffix=config["dataset_params"]["bw_occ_suffix"],
            bw_of_suffix=config["dataset_params"]["bw_of_suffix"],
            fw_occ_suffix=config["dataset_params"]["fw_occ_suffix"],
            fw_of_suffix=config["dataset_params"]["fw_of_suffix"],
            config=config)

    elif dataset == 'kitti':
        from datasets.kitti import Kitti
        train_dataset = Kitti(images_path=config["dataset_params"].get("train_images_path",""),
            segmasks_path=config["dataset_params"].get("train_segmasks_path",""),
            instances_path=config["dataset_params"].get("train_instances_path",""),
            tracking_path=config["dataset_params"].get("train_tracking_path",""),
            bw_occ_path=config["dataset_params"].get("train_bw_occ_path",""),
            bw_of_path=config["dataset_params"].get("train_bw_of_path",""),
            fw_occ_path=config["dataset_params"].get("train_fw_occ_path",""),
            fw_of_path=config["dataset_params"].get("train_fw_of_path",""),
            datalist=config["dataset_params"]["train_data_list"],
            size=config["train_params"]["input_size"], split='train',
            segmask_suffix=config["dataset_params"]["segmask_suffix"],
            instance_suffix=config["dataset_params"]["instance_suffix"],
            bw_occ_suffix=config["dataset_params"]["bw_occ_suffix"],
            bw_of_suffix=config["dataset_params"]["bw_of_suffix"],
            config=config)
    elif dataset == 'mvtid':
        from datasets.mvtid import Mvtid
        dataset_type = config["dataset_params"]["dataset_type"]
        assert dataset_type in ['Drone', 'Infrastructure']
        train_dataset = Mvtid(
            dataset_type=dataset_type,
            images_path=config["dataset_params"].get("train_images_path", ""),
            segmasks_path=config["dataset_params"].get("train_segmasks_path", ""),
            instances_path=config["dataset_params"].get("train_instances_path", ""),
            tracking_path=config["dataset_params"].get("train_tracking_path", ""),
            bw_occ_path=config["dataset_params"].get("train_bw_occ_path", ""),
            bw_of_path=config["dataset_params"].get("train_bw_of_path", ""),
            datalist=config["dataset_params"]["train_data_list"],
            size=config["train_params"]["input_size"], split='train',
            segmask_suffix=config["dataset_params"]["segmask_suffix"],
            instance_suffix=config["dataset_params"]["instance_suffix"],
            bw_occ_suffix=config["dataset_params"]["bw_occ_suffix"],
            bw_of_suffix=config["dataset_params"]["bw_of_suffix"],
            config=config)

    return train_dataset


def get_test_set(config):
    dataset = config["dataset_params"]["dataset"]
    assert dataset in ['cityscapes', 'kitti', 'mvtid']

    if dataset == 'cityscapes':
        from datasets.cityscapes import Cityscapes
        test_dataset = Cityscapes(
            images_path=config["dataset_params"].get("val_images_path", ""),
            segmasks_path=config["dataset_params"].get("val_segmasks_path", ""),
            instances_path=config["dataset_params"].get("val_instances_path", ""),
            tracking_path=config["dataset_params"].get("val_tracking_path", ""),
            bw_occ_path=config["dataset_params"].get("val_bw_occ_path", ""),
            bw_of_path=config["dataset_params"].get("val_bw_of_path", ""),
            fw_occ_path=config["dataset_params"].get("val_fw_occ_path", ""),
            fw_of_path=config["dataset_params"].get("val_fw_of_path", ""),
            datalist=config["dataset_params"]["val_data_list"],
            size=config["train_params"]["input_size"], split='test',
            segmask_suffix=config["dataset_params"]["segmask_suffix"],
            instance_suffix=config["dataset_params"]["instance_suffix"],
            bw_occ_suffix=config["dataset_params"]["bw_occ_suffix"],
            bw_of_suffix=config["dataset_params"]["bw_of_suffix"],
            fw_occ_suffix=config["dataset_params"]["fw_occ_suffix"],
            fw_of_suffix=config["dataset_params"]["fw_of_suffix"],
            config=config)
    elif dataset == 'kitti':
        from datasets.kitti_dataset import KITTI
        test_dataset = KITTI(
            images_path=config["dataset_params"].get("val_images_path", ""),
            segmasks_path=config["dataset_params"].get("val_segmasks_path", ""),
            instances_path=config["dataset_params"].get("val_instances_path", ""),
            tracking_path=config["dataset_params"].get("val_tracking_path", ""),
            bw_occ_path=config["dataset_params"].get("val_bw_occ_path", ""),
            bw_of_path=config["dataset_params"].get("val_bw_of_path", ""),
            fw_occ_path=config["dataset_params"].get("val_fw_occ_path", ""),
            fw_of_path=config["dataset_params"].get("val_fw_of_path", ""),
            datalist=config["dataset_params"]["val_data_list"],
            size=config["train_params"]["input_size"], split='test',
            segmask_suffix=config["dataset_params"]["segmask_suffix"],
            instance_suffix=config["dataset_params"]["instance_suffix"],
            bw_occ_suffix=config["dataset_params"]["bw_occ_suffix"],
            bw_of_suffix=config["dataset_params"]["bw_of_suffix"],
            config=config)
    elif dataset == 'mvtid':
        from datasets.mvtid import Mvtid
        dataset_type = config["dataset_params"]["dataset_type"]
        assert dataset_type in ['Drone', 'Infrastructure']
        test_dataset = Mvtid(
            dataset_type=dataset_type,
            images_path=config["dataset_params"].get("val_images_path", ""),
            segmasks_path=config["dataset_params"].get("val_segmasks_path", ""),
            instances_path=config["dataset_params"].get("val_instances_path", ""),
            tracking_path=config["dataset_params"].get("val_tracking_path", ""),
            bw_occ_path=config["dataset_params"].get("val_bw_occ_path", ""),
            bw_of_path=config["dataset_params"].get("val_bw_of_path", ""),
            fw_occ_path=config["dataset_params"].get("val_fw_occ_path", ""),
            fw_of_path=config["dataset_params"].get("val_fw_of_path", ""),
            datalist=config["dataset_params"]["val_data_list"],
            size=config["train_params"]["input_size"], split='val',
            segmask_suffix=config["dataset_params"]["segmask_suffix"],
            instance_suffix=config["dataset_params"]["instance_suffix"],
            bw_occ_suffix=config["dataset_params"]["bw_occ_suffix"],
            bw_of_suffix=config["dataset_params"]["bw_of_suffix"],
            config=config)
    return test_dataset
