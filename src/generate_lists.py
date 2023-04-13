import os
import glob
import argparse

parser = argparse.ArgumentParser(description='generate json')
parser.add_argument('--phase', default='', type=str,
                    help='phase')
parser.add_argument('--dataset_root', default='', type=str,
                    help='phase')
args = parser.parse_args()
phase = args.phase
root = args.dataset_root

file_list = sorted(
    set(["/".join(fname.split("/")[-2:])[:-10] + "_leftImg8bit.png" for fname in glob.glob(f"{root}/instance_tracking/{phase}/*/*")]))

with open(f"{root}/cityscapes_{phase}_new.txt", "w") as writer:
    writer.write("\n".join(file_list))
