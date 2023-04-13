import os
import glob
from shutil import copy2
from PIL import Image
import json
import numpy as np
import argparse
import json
import threading

def size(s):
    try:
        x, y = map(int, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Size must be x, y")


def copy_file(src, src_ext, dst, type, size):
    # find all files ends up with ext
    flist = sorted(glob.glob(os.path.join(src, '*', src_ext)))
    for fname in flist:
        src_path = fname
        if not os.path.exists(os.path.join(dst, fname.split("/")[-2])):
            os.makedirs(os.path.join(dst, fname.split("/")[-2]))
        img = Image.open(src_path)
        if type == "img":
            img = img.resize((size[1],size[0]), Image.BICUBIC)
        elif type in ["inst", "label"]:
            img = img.resize((size[1],size[0]), Image.NEAREST)
        img.save(os.path.join(os.path.join(dst, fname.split("/")[-2]), src_path.split("/")[-1]))
        print('copied %s to %s' % (src_path, os.path.join(dst, fname.split("/")[-2])))


# organize image
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str,
                        default='../datasets')
    parser.add_argument('--input_size', default=(64, 128), type=size, help='input image size')

    opt = parser.parse_args()
    folder_name = opt.dataroot
    train_img_dst = os.path.join(folder_name, f'leftImg8bit_sequence_{opt.input_size[0]}x{opt.input_size[1]}/train')
    train_label_dst = os.path.join(folder_name, f'segmasks_{opt.input_size[0]}x{opt.input_size[1]}/train')
    train_inst_dst = os.path.join(folder_name, f'instances_{opt.input_size[0]}x{opt.input_size[1]}/train')
    val_img_dst = os.path.join(folder_name, f'leftImg8bit_sequence_{opt.input_size[0]}x{opt.input_size[1]}/val')
    val_label_dst = os.path.join(folder_name, f'segmasks_{opt.input_size[0]}x{opt.input_size[1]}/val')
    val_inst_dst = os.path.join(folder_name, f'instances_{opt.input_size[0]}x{opt.input_size[1]}/val')
    if not os.path.exists(train_img_dst):
        os.makedirs(train_img_dst)
    if not os.path.exists(train_label_dst):
        os.makedirs(train_label_dst)
    if not os.path.exists(train_inst_dst):
        os.makedirs(train_inst_dst)
    if not os.path.exists(val_img_dst):
        os.makedirs(val_img_dst)
    if not os.path.exists(val_label_dst):
        os.makedirs(val_label_dst)
    if not os.path.exists(val_inst_dst):
        os.makedirs(val_inst_dst)

t1 = threading.Thread( target=copy_file, args=(os.path.join(opt.dataroot,'leftImg8bit_sequence/train'),
              '*.jpg', train_img_dst,"img", opt.input_size) )
t2 = threading.Thread( target=copy_file, args=(os.path.join(opt.dataroot,'segmasks/train'),
              '*_gtFine_labelIds.png', train_label_dst, "label", opt.input_size) )
t3 = threading.Thread( target=copy_file, args=(os.path.join(opt.dataroot,'instances/train'),
            '*_gtFine_instanceIds.png', train_inst_dst, "inst", opt.input_size) )
t4 = threading.Thread( target=copy_file, args=(os.path.join(opt.dataroot,'leftImg8bit_sequence/val'),
            '*.jpg', val_img_dst, "img", opt.input_size) )
t5 = threading.Thread( target=copy_file, args=(os.path.join(opt.dataroot,'segmasks/val'),
            '*_gtFine_labelIds.png', val_label_dst, "label", opt.input_size) )
t6 = threading.Thread( target=copy_file, args=(os.path.join(opt.dataroot,'instances/val'),
            '*_gtFine_instanceIds.png', val_inst_dst, "inst", opt.input_size) )
t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()