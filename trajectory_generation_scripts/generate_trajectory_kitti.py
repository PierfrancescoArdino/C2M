### Script used for generating script for cityscapes dataset to gen training data for foreground object prediction
import numpy as np
import pandas as pd
import os
from PIL import Image
from shutil import copyfile

ImagesRoot = "/home/pardino/dataset_kitti_video/leftImg8bit_sequence/"

import json
from multiprocessing import Pool


def load_all_image_paths(image_dir, phase):
	image_dir = image_dir + phase + "/"
	day_dir = os.listdir(image_dir)
	day_dir.sort()
	video = []
	for i in range(len(day_dir)):
		video_dir = image_dir + day_dir[i]
		video_list = os.listdir(video_dir)
		video_list.sort()
		for j in range(len(video_list)):
			frame_dir = video_dir + "/" + video_list[j] + "/image_00/data_rect"
			frame_list = os.listdir(frame_dir)
			frame_list.sort()
			image = []
			for k in range(len(frame_list)):
				full_image_path = frame_dir + "/" + frame_list[k]
				assert os.path.isfile(full_image_path)
				image.append(full_image_path)
			video.append((image))
	return video


# Utils for reading in upsnet instance result
# Include mask/class/score
class upsnet_instance():
	def __init__(self):
		pass

	def get_file_name(self, full_path):
		name_split = full_path.split("/")
		image_name = name_split[-1]
		city_dir = "/".join(full_path.split("/")[6:-3])
		return city_dir, image_name

	def load_txt(self, txt_name):
		lineList = [line.rstrip('\n') for line in open(txt_name)]
		info_dict = []
		for i in range(len(lineList)):
			line = lineList[i].split(" ")
			full_name = line[0].split("/")[-1]
			cls = int(line[1])
			score = float(line[2])
			info_dict.append((full_name, cls, score))
		return info_dict

	def compute_bbox(self, mask):
		y, x = np.where(mask == 1)
		# print("y = ", y)
		# print("x = ", x )
		# try:
		st_x = np.min(x)
		# except:
		# return None
		st_y = np.min(y)
		tx = np.max(x) - np.min(x)
		ty = np.max(y) - np.min(y)
		return [float(st_x), float(st_y), float(tx), float(ty)]

	# return [float(st_y), float(st_x), float(ty), float(tx)]

	def readio_upsnet_instance(self, InstanceRoot, phase, file_name):
		# Filename for rgb image name
		# Upsnet instance root = /disk1/yue/cityscapes/cityscapes/instance_upsnet/origin_result/
		city, img_name = self.get_file_name(file_name)
		#img_name = img_name.replace(".png", "_gtFine_instanceIds.png")
		segs_name = InstanceRoot + "/" + city + "/image_02/data/" + img_name
		instance_mask = np.array(Image.open(segs_name))
		info_dict_more = []
		instance_ids = filter(lambda x: x > 1000, np.unique(instance_mask))
		for k in instance_ids:
			mask = (np.squeeze(instance_mask) == np.squeeze(k)).astype(float)
			bbox = self.compute_bbox(mask)
			if bbox is None:
				return None
			info_dict_more.append((img_name, mask, bbox, k // 1000, k))
		# print(info_dict)
		return info_dict_more


def tracking_list(i, j, initial_instance, image_paths, dict_all):
	# Each object is represented by a dict
	# Name video_%d_frame_%d_object_%d

	for k in range(len(initial_instance)):
		dict = {}
		dict['video_dir'] = 'video_%04d_frame_%02d_object_%02d' % (i, j, k)
		dict['init_rect'] = initial_instance[k][2]
		dict['img_names'] = image_paths
		dict_all['video_%04d_frame_%02d_object_%02d' % (i, j, k)] = dict
	return dict_all


def track_txt_reader(txt_name):
	if os.path.exists(txt_name):
		lineList = [line.rstrip('\n') for line in open(txt_name)]
		info_dict = []
		valid_sample = True
		for i in range(len(lineList)):
			line = lineList[i].split(",")
			st_x = float(line[0])
			st_y = float(line[1])
			tx = float(line[2])
			ty = float(line[3])
			score = float(line[4])
			if score < 0.95:
				valid_sample = False
			# print([st_x, st_y, tx, ty])
			info_dict.append([st_x, st_y, tx, ty])
		return info_dict, valid_sample
	else:
		return False, False


def bbox2mask(bbox,size):
	# print("bbox = ", bbox)
	st_x = bbox[0]
	st_y = bbox[1]
	tx = bbox[2]
	ty = bbox[3]
	mask = np.zeros((size))
	mask[int(st_y):int(st_y + ty), int(st_x):int(st_x + tx)] = 1
	return mask


def check_single_track(bbox_src, mask_src, cls_src, bbox_tgt, cls_tgt):
	# Check bbox
	flag = True

	# check bbox and mask iou
	mask_tgt = bbox2mask(bbox_tgt, mask_src.shape)
	if np.sum(mask_tgt * mask_src) / np.sum(mask_src) < 0.8:
		flag = False
	if cls_src != cls_tgt:
		flag = False
	return flag


def match_instance_bbox(instance, bbox_tgt, cls_tgt):
	for i in range(len(instance)):
		bbox_src = instance[i][2]
		mask_src = instance[i][1]
		cls_src = instance[i][3]
		flag = instance[i][-1]
		check = check_single_track(bbox_src, mask_src, cls_src, bbox_tgt,
								   cls_tgt)
		if check is True:
			# instance[i][-1] = True
			return instance, i
	return instance, -1


phase = 'train'
InstanceRoot = f"/home/pardino/dataset_kitti_video/leftImg8bit_sequence/{phase}_instance"
upsnet_instance_readio = upsnet_instance()
all_image_paths = load_all_image_paths(ImagesRoot, phase)
print("Loaded %d image paths = " % len(all_image_paths))
dict_all = {}


def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


txt_root = f"/home/pardino/trajectory_generation_scripts/results_{phase}_kitti360/kitti/model/"

result_dir = f"/home/pardino/dataset_kitti_video/leftImg8bit_sequence/{phase}_instance_tracking_test/"
mkdir(result_dir)

def track_instance(i):
	print("video %04d" % i)
	video_paths = all_image_paths[i]
	#mkdir(result_dir)
	for j in range(0, len(video_paths) - 5, 6):
		cnt_video = video_paths[j:j + 6]
		init_video_frame = video_paths[j]
		Instances_list = [None] * 6
		for k in range(6):
			Instances_list[k] = upsnet_instance_readio.readio_upsnet_instance(
				InstanceRoot, phase, video_paths[j + k])
		# Instance is a list of image name, mask, bbox, cls, score
		initial_instance = Instances_list[0]
		if initial_instance is not None:
			for tmp in range(len(initial_instance)):
				# Load tracker for this instance
				root_result_path = os.path.join(txt_root, "/".join(init_video_frame.split("/")[6:-1])) + "/"

				txt_file = f'{init_video_frame.split("/")[-1][:-4]}_{initial_instance[tmp][4]}.txt'
				track_txt = root_result_path + txt_file
				tracker, flag = track_txt_reader(track_txt)

				if flag is True:
					masks_list = []
					mask_initial = initial_instance[tmp][1]
					cls_initial = initial_instance[tmp][3]
					# print("class", cls_initial)
					masks_list.append([mask_initial, initial_instance[tmp][-1]])
					# Test whether this track has correspondense in future frames one by one
					for f in range(1, 6):
						# bbox of tracker
						bbox = tracker[f]

						# instance this frame
						instance = Instances_list[f]

						Instances_list[f], idx = match_instance_bbox(instance, bbox,
																	 cls_initial)
						if idx == -1:
							# Don't have correspoding mask
							break
						else:
							masks_list.append([instance[idx][1], instance[idx][-1]])

					#### Save here
					# Save masks
					if len(masks_list) == 6:
						result_object_path = result_dir + "/".join(root_result_path.split("/")[7:-1]) + "/"
						mkdir(result_object_path)
						with open(result_object_path + txt_file, "w") as writer:
							writer.write("\n".join([",".join(
								map(str, tracker[i] + [masks_list[i][-1]])) for i in
													range(len(masks_list))]))
				# copyfile(track_txt, result_object_path + txt_file)


if __name__=="__main__":
	pool = Pool(processes=4)
	pool.map(track_instance, np.arange(0,len(all_image_paths))) # range(0,1000) if you want to replicate your example
	pool.close()
	pool.join()
	#for i in np.arange(0,len(all_image_paths)):
	#		track_instance(i)
