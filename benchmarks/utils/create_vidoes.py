import imageio
import numpy as np
import os
import yaml
import random
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from benchmarks.SalsaNext.train.common.laserscan import SemLaserScan


EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']
EXTENSIONS_NPZ = ['.npz']


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def get_files_list(dataset_path):
    rawdata_path = dataset_path
    if os.path.isdir(rawdata_path):
        print("Raw data folder exists! Using raw data from %s" % rawdata_path)
    else:
        raise ValueError(
            "Raw data folder doesn't exist! Exiting... %s" % rawdata_path)
    rawdata_list = []

    seq_rawdata_path = rawdata_path
    rawdata_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(seq_rawdata_path)) for f in fn if is_scan(f)]

    rawdata_list.extend(rawdata_files)

    rawdata_list.sort()

    print("Using {} data from sequences".format(len(rawdata_list)))

    return rawdata_list


def idlabel2colorlabel(id_label,lut):
    width,height = id_label.shape
    color_label = np.zeros((width,height,3),np.uint8)
    for key in lut:
        color_label[(id_label==key)] = lut[key]
    return color_label

color_dict = {0: "000000",
              3: "006600",
              4: "00ff00",
              5: "009999",
              6: "0080ff",
              7: "0000ff",
              8: "ffff00",
              9: "ff007f",
              10: "404040",
              12: "ff0000",
              15: "660000",
              17: "cc99ff",
              19: "ff99cc",
              23: "aaaaaa",
              27: "2979FF",
              29: "651FFF",
              30: "899509",
              31: "86ffef",
              32: "1d6988",
              33: "634222",
              34: "6e168a"}

def hex2rgb(s):
    return [np.int(s[:2],16),int(s[2:4],16),int(s[4:6],16)]

color_dict = {i[0]: hex2rgb(i[1]) for i in color_dict.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",'-p')
    parser.add_argument("--sequence",'-s')
    args = parser.parse_args()
    print(color_dict)
    mapper = SemLaserScan(sem_color_dict=color_dict, project=True, H=64, W=1024, fov_up=2.0,
                          fov_down=-24.33, max_classes=300, DA=False, flip_sign=False, drop_points=False)
    w = imageio.get_writer(f'{args.sequence}.mp4', format='FFMPEG', mode='I', fps=10)

    dataset_path = f'{args.path}/{args.sequence}/os1_cloud_node_kitti_bin'

    label_list = get_files_list(dataset_path)
    for scanfile in tqdm(label_list):
        mapper.open_scan(scanfile)
        labelfile = scanfile.replace("os1_cloud_node_kitti_bin", "os1_cloud_node_semantickitti_label_id")
        labelfile = labelfile.replace("bin", "label")
        mapper.open_label(labelfile)
        mapper.colorize()
        proj_sem_color = mapper.proj_sem_color*255
        w.append_data(proj_sem_color.astype(np.uint8))

    w.close()