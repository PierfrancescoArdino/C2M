import os
import glob
from shutil import copy2
from PIL import Image
import json
import numpy as np
import argparse
import json
import threading

def copy_file(src, src_ext, dst, type):
    # find all files ends up with ext
    flist = sorted(glob.glob(os.path.join(src, '*', src_ext)))
    for fname in flist:
        src_path = fname
        if not os.path.exists(os.path.join(dst, fname.split("/")[-2])):
            os.makedirs(os.path.join(dst, fname.split("/")[-2]))
        img = Image.open(src_path)
        if type == "img":
            img = img.resize((256,128), Image.BICUBIC)
        elif type in ["inst", "label"]:
            img = img.resize((256,128), Image.NEAREST)
        img.save(os.path.join(os.path.join(dst, fname.split("/")[-2]), src_path.split("/")[-1]))
        print('copied %s to %s' % (src_path, os.path.join(dst, fname.split("/")[-2])))


# organize image
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str,
                        default='../datasets')

    opt = parser.parse_args()
    folder_name = opt.dataroot
    train_img_dst = os.path.join(folder_name, 'train256_128')
    train_label_dst = os.path.join(folder_name, 'train_semantic_segmask256x128')
    train_inst_dst = os.path.join(folder_name, 'train_instance256x128')
    val_img_dst = os.path.join(folder_name, 'val256_128')
    val_label_dst = os.path.join(folder_name, 'val_semantic_segmask256x128')
    val_inst_dst = os.path.join(folder_name, 'val_instance256x128')
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
              '*_leftImg8bit.png', train_img_dst,"img") )
t2 = threading.Thread( target=copy_file, args=(os.path.join(opt.dataroot,'train_semantic_segmask'),
              '*_ssmask.png', train_label_dst, "label") )
t3 = threading.Thread( target=copy_file, args=(os.path.join(opt.dataroot,'train_instance'),
            '*_instanceIds.png', train_inst_dst, "inst") )
t4 = threading.Thread( target=copy_file, args=(os.path.join(opt.dataroot,'leftImg8bit_sequence/val'),
            '*_leftImg8bit.png', val_img_dst, "img") )
t5 = threading.Thread( target=copy_file, args=(os.path.join(opt.dataroot,'val_semantic_segmask/'),
            '*_ssmask.png', val_label_dst, "label") )
t6 = threading.Thread( target=copy_file, args=(os.path.join(opt.dataroot,'val_instance'),
            '*_instanceIds.png', val_inst_dst, "inst") )
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