# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--snapshot', default='', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
parser.add_argument('--phase', default='', type=str,
        help='phase')
parser.add_argument('--start_from', default=0, type=int)
parser.add_argument('--end_to', default=1000000000000000000, type=int)
args = parser.parse_args()
phase= args.phase
start_from = args.start_from
end_to = args.end_to
torch.set_num_threads(10)

def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir)

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)
    num_images = 1 if phase=="val" else 8
    # create dataset
    if not os.path.exists(dataset_root + f"/dataset_{phase}_kitti360.pickle"):
        dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False, num_images=num_images, phase=phase)

        import pickle
        outfile = open(dataset_root + f"/dataset_{phase}_kitti360.pickle",'wb')
        pickle.dump(dataset,outfile)
        outfile.close()
    else:
        import pickle
        infile = open(dataset_root + f"/dataset_{phase}_kitti360.pickle",'rb')
        dataset = pickle.load(infile)
        infile.close()
    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    for v_idx, video in enumerate(dataset):
        if v_idx < start_from:
            continue
        if v_idx > end_to:
            continue
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        skip = False
        for idx, (img, init_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                #cx, cy, w, h = get_axis_aligned_bbox(np.array(init_bbox))
                #gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                gt_bbox_ = np.array(init_bbox)
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(1)
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
                if outputs['best_score'] < 0.95 or np.all(np.array(pred_bbox) <= 0):
                    skip = True
                    break

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())

        if not skip:
            toc /= cv2.getTickFrequency()

            model_path = os.path.join(f'results_{phase}_kitti360', args.dataset, model_name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            root_result_path = os.path.join(model_path, "/".join(video.img_names[0].split("/")[6:-1]))
            if not os.path.isdir(root_result_path):
                os.makedirs(root_result_path)

            result_path = os.path.join(root_result_path, f'{video.img_names[0].split("/")[-1][:-4]}_{video.instance_id}.txt')
            with open(result_path, 'w') as f:
                for x in range(len(pred_bboxes)):
                    cnt_bbox = pred_bboxes[x]
                    cnt_score = scores[x]
                    f.write(','.join([str('%.2f'%i) for i in cnt_bbox]))
                    f.write(',')
                    f.write(str('%.3f'%cnt_score)+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
