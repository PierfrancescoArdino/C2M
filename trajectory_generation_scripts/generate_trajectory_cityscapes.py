### Script used for generating script for cityscapes dataset to gen training data for foreground object prediction
import numpy as np
import pandas as pd
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='generate json')
parser.add_argument('--phase', default='', type=str,
        help='phase')
parser.add_argument('--images_root', default='', type=str,
        help='images_root', required=True)
parser.add_argument('--instance_root', default='', type=str,
        help='instance_root', required=True)
parser.add_argument('--tracker_root', default='', type=str,
        help='tracker_root', required=True)
parser.add_argument('--result_dir', default='', type=str,
        help='result_dir', required=True)
parser.add_argument('--object_instance_dir', default='', type=str,
        help='object_instance_dir', required=True)
parser.add_argument('--n_processes', default=1, type=int)

args = parser.parse_args()
phase = args.phase

import json
from multiprocessing import Pool


## Load all image_paths
def load_all_image_paths(image_dir, phase):
	image_dir = image_dir + phase
	city_dir = os.listdir(image_dir)
	city_dir.sort()
	video = []
	for i in range(len(city_dir)):
		frame_dir = image_dir + "/" + city_dir[i]
		frame_list = os.listdir(frame_dir)
		frame_list.sort()
		for j in range(len(frame_list)//30):
			image = []
			for k in range(j*30, (j+1)*30):
				full_image_path = frame_dir + "/" + frame_list[k]
				assert os.path.isfile(full_image_path)
				image.append(full_image_path)
			video.append(image)
	return video


# Utils for reading in upsnet instance result
# Include mask/class/score
class upsnet_instance():
	def __init__(self):
		pass

	def get_file_name(self, full_path):
		name_split = full_path.split("/")
		image_name = name_split[-1]
		city_dir = name_split[-2]
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
		img_name = img_name.replace("_leftImg8bit.png",
									"_gtFine_instanceIds.png")
		segs_name = InstanceRoot + "/" + phase + "/" + city + "/" + img_name
		instance_mask = np.array(Image.open(segs_name))
		instance_mask[800:, :] = 0
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


def tracking_list(i, j, initial_instance, image_paths, frames_info):
	# Each object is represented by a dict
	# Name video_%d_frame_%d_object_%d

	for k in range(len(initial_instance)):
		object_info = {'video_dir': 'video_%04d_frame_%02d_object_%02d' % (i, j, k),
					   'init_rect': initial_instance[k][2],
					   'img_names': image_paths}
		frames_info['video_%04d_frame_%02d_object_%02d' % (i, j, k)] = object_info
	return frames_info


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


def bbox2mask(bbox):
	# print("bbox = ", bbox)
	st_x = bbox[0]
	st_y = bbox[1]
	tx = bbox[2]
	ty = bbox[3]
	mask = np.zeros((1024, 2048))
	mask[int(st_y):int(st_y + ty), int(st_x):int(st_x + tx)] = 1
	return mask


def check_single_track(bbox_src, mask_src, cls_src, bbox_tgt, cls_tgt):
	if not np.all(np.array(bbox_src[2:]) > 0):
		return False, None, None
	if cls_src != cls_tgt:
		return False, None, None
	# check bbox and mask iou
	mask_tgt = bbox2mask(bbox_tgt)
	if np.sum(mask_tgt * mask_src) / np.sum(mask_src) < 0.8:
		return False, None, None
	return True, np.sum(mask_tgt * mask_src) / np.sum(mask_src), np.sum(mask_tgt * mask_src)


def match_instance_bbox(instance, bbox_tgt, cls_tgt):
	instances = []
	found = False
	for i in range(len(instance)):
		bbox_src = instance[i][2]
		mask_src = instance[i][1]
		cls_src = instance[i][3]
		flag = instance[i][-1]
		check, iou, area = check_single_track(bbox_src, mask_src, cls_src, bbox_tgt,
								   cls_tgt)
		if check is True:
			# instance[i][-1] = True
			instances.append([iou, area, instance[i][-1]])
			found = True
			#return instance, i
		else:
			instances.append([-1, -1, -1])
	return instance, max(enumerate(instances), key=lambda elem: max(elem[1]))[0] if found else -1


upsnet_instance_readio = upsnet_instance()
all_image_paths = load_all_image_paths(args.images_root, phase)
print("Loaded %d image paths = " % len(all_image_paths))
frames_info = {}


def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


txt_root = f"{args.tracker_root}/results_{phase}/cityscape/model/"

result_dir = f"{args.result_dir}/{phase}"
mask_result_dir = f"{args.object_instance_dir}/{phase}"


def track_instance(i):
	print("video %04d" % i)
	video_paths = all_image_paths[i]
	mkdir(result_dir)
	mkdir(mask_result_dir)
	j = 0
	cnt_video = video_paths[j:j + 9]
	init_video_frame = video_paths[j]
	instances_list = [None] * 9
	for k in range(9):
		instances_list[k] = upsnet_instance_readio.readio_upsnet_instance(
			args.instance_root, phase, video_paths[j + k])
	# Instance is a list of image name, mask, bbox, cls, score
	initial_instance = instances_list[0]
	if initial_instance is not None:
		for tmp in range(len(initial_instance)):
			# Load tracker for this instance
			txt_file = f'{init_video_frame.split("/")[-1][:-16]}_{initial_instance[tmp][4]}.txt'
			track_txt = txt_root + txt_file
			tracker, flag = track_txt_reader(track_txt)

			if flag is True:
				masks_list = []
				mask_initial = initial_instance[tmp][1]
				cls_initial = initial_instance[tmp][3]
				# print("class", cls_initial)
				masks_list.append([initial_instance[tmp][0],
								   mask_initial, initial_instance[tmp][2],
								   initial_instance[tmp][-1]])
				# Test whether this track has correspondense in future frames one by one
				for f in range(1, 9):
					# bbox of tracker
					bbox = tracker[f]

					# instance this frame
					instance = instances_list[f]

					instances_list[f], idx = match_instance_bbox(instance, bbox,
																 cls_initial)
					if idx == -1:
						# Don't have correspoding mask
						break
					else:
						masks_list.append([instance[idx][0], instance[idx][1], instance[idx][2], instance[idx][-1]])

				#### Save here
				# Save masks
				if len(masks_list) == 9:
					result_object_path = result_dir + "/" + txt_file.split("_")[
						0] + "/"

					mkdir(result_object_path)
					with open(result_object_path + txt_file, "w") as writer:
						for obj in masks_list:
							base_name = obj[0].replace("_gtFine_instanceIds.png", "")
							result_mask_object_path = mask_result_dir + "/" + txt_file.split("_")[
								0] + "/" + base_name + "/"
							mkdir(result_mask_object_path)
							writer.write(f"{','.join(map(str, obj[2] + [obj[3]]))}\n")
							Image.fromarray((obj[1] * 255).astype(np.uint8)).save(
								result_mask_object_path + f"{'_'.join([base_name, str(obj[3])])}.png")


if __name__=="__main__":
	pool = Pool(processes=args.n_processes)
	pool.map(track_instance, np.arange(0,len(all_image_paths))) # range(0,1000) if you want to replicate your example
	pool.close()
	pool.join()