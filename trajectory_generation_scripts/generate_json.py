### Script used for generating script for cityscapes dataset to gen training data for foreground object prediction
import numpy as np
import pandas as pd
import os
from PIL import Image
import json
import argparse

parser = argparse.ArgumentParser(description='generate json')
parser.add_argument('--phase', default='', type=str,
        help='phase')
args = parser.parse_args()
phase = args.phase

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
		y, x  = np.where(mask == 1)
		#print("y = ", y)
		#print("x = ", x )
		#try:
		st_x = np.min(x)
		#except:
		#return None
		st_y = np.min(y)
		tx = np.max(x) - np.min(x)
		ty = np.max(y) - np.min(y)
		if tx > 0 and ty > 0:
			return [float(st_x), float(st_y), float(tx), float(ty)]
		else:
			return None
		#return [float(st_y), float(st_x), float(ty), float(tx)]

	def readio_upsnet_instance(self, InstanceRoot, phase, file_name):
		# Filename for rgb image name
		# Upsnet instance root = /disk1/yue/cityscapes/cityscapes/instance_upsnet/origin_result/
		city, img_name = self.get_file_name(file_name)
		img_name = img_name.replace("_leftImg8bit.png", "_gtFine_instanceIds.png")
		segs_name = InstanceRoot + "/" + phase + "/" + city + "/" + img_name
		instance_mask = np.array(Image.open(segs_name))
		instance_mask[800:,:] = 0
		info_dict_more = []
		instance_ids = filter(lambda x: x > 1000, np.unique(instance_mask))
		for k in instance_ids:
			mask = (np.squeeze(instance_mask) == np.squeeze(k)).astype(float)
			bbox = self.compute_bbox(mask)
			if bbox is None:
				continue
			info_dict_more.append((img_name, mask, bbox, k // 1000, k))
		#print(info_dict)
		return info_dict_more


def tracking_list(i,j,initial_instance, image_paths, dict_all):
	# Each object is represented by a dict
	# Name video_%d_frame_%d_object_%d
	
	for k in range(len(initial_instance)):
		dict = {}
		dict['video_dir'] = 'video_%04d_frame_%02d_object_%02d'%(i,j,k)
		dict['init_rect'] = initial_instance[k][2]
		dict['img_names'] = image_paths
		dict["instance_id"] = str(initial_instance[k][-1])
		dict_all['video_%04d_frame_%02d_object_%02d'%(i,j,k)] = dict
	return dict_all


ImagesRoot = "/home/ardino/dataset_cityscape_video/leftImg8bit_sequence/leftImg8bit_sequence/"
InstanceRoot = f"/home/ardino/dataset_cityscape_video/leftImg8bit_sequence/instances"
upsnet_instance_readio = upsnet_instance()
all_image_paths = load_all_image_paths(ImagesRoot, phase)
print("Loaded %d image paths = "%len(all_image_paths))

st = 0
dict_all = {}
for st in range(len(all_image_paths)):
	if st == 2232:
		continue
	print("video %04d" % st)
	j = 0
	video_paths = all_image_paths[st]
	cnt_video = video_paths[j:j+9]
	init_video_frame = video_paths[j]
	instance_io = upsnet_instance_readio.readio_upsnet_instance(InstanceRoot, phase, init_video_frame)
	#for k in range(len(instance_io)):
	#print(instance_io[k])
	if instance_io is None:
		continue
	dict_all = tracking_list(st, j, instance_io, cnt_video, dict_all)
		

with open(f"cityscapes_{0}_{len(all_image_paths)}_{phase}.json", 'w') as fp:
	json.dump(dict_all, fp)

#save_dir = "/disk1/yue/cityscapes/cityscapes/instance_upsnet/dict/"

#rgb_full_name = ImageRoot + "aachen/aachen_000000_000000_leftImg8bit.png"
