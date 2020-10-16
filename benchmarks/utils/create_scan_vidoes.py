import imageio
import numpy as np
import os
import yaml
import random
from tqdm import tqdm
from skimage import color
import argparse
import matplotlib.pyplot as plt
from benchmarks.SalsaNext.train.common.laserscan import SemLaserScan
import random


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
    filename = "pt_test"
    root_path = "/home/usl/Datasets/rellis"
    list_path = f"/home/usl/Datasets/rellis/{filename}.lst"
    mapper = SemLaserScan(sem_color_dict=color_dict, project=True, H=64, W=1024, fov_up=22.5,
                          fov_down=-22.5, max_classes=300, DA=False, flip_sign=False, drop_points=False)
    w = imageio.get_writer(f'{filename}s.mp4', format='FFMPEG', mode='I', fps=10)

    label_list = [line.strip().split() for line in open(list_path)]
    #img_list = get_files_list(dataset_path)
    max_num = 100
    count = 0
    #random.shuffle(label_list)
    #print(label_list)
    for scanfile in tqdm(label_list):
        #print(scanfile)
        scanfile = os.path.join(root_path,scanfile[0])
        mapper.open_scan(scanfile)
        labelfile = scanfile.replace("os1_cloud_node_kitti_bin", "os1_cloud_node_semantickitti_label_id")
        labelfile = labelfile.replace("bin", "label")
        proj_range = (mapper.proj_range+1)/2*255
        proj_range = color.gray2rgb(proj_range,None)
        proj_range[:,:,0] = proj_range[:,:,0] * 0.2989
        proj_range[:,:,1] = proj_range[:,:,1] * 0.5870
        proj_range[:,:,2] = proj_range[:,:,2] * 0.1140
        proj_range = proj_range.astype(np.uint8)
        mapper.open_label(labelfile)
        mapper.colorize()
        proj_sem_color = mapper.proj_sem_color*255
        salsa_labelfile = labelfile.replace("rellis","salsa")


        mapper.open_label(salsa_labelfile)
        mapper.colorize()
        proj_sem_color_salsa = mapper.proj_sem_color*255

        conv_labelfile = labelfile.replace("rellis","kpconv")
        label = np.fromfile(conv_labelfile, dtype=np.int32)
        label = label.reshape((-1))
        #print(np.unique(label))
        mapper.open_label(conv_labelfile)
        mapper.colorize()
        proj_sem_color_kpconv = mapper.proj_sem_color*255



        cat_img = np.concatenate((proj_range,proj_sem_color,proj_sem_color_kpconv,proj_sem_color_salsa),axis=0).astype(np.uint8)
        w.append_data(cat_img)
        # if count == max_num:
        #     break
        # else:
        #     count=count+1
        #break

    w.close()