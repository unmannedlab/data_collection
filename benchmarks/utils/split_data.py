import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import yaml
from PIL import Image, ImageOps
from scipy import misc, ndimage, special

import torch
import time
import pickle
import json

import argparse

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']
EXTENSIONS_NPZ = ['.npz']


def is_scan(filename,extensions):
    return any(filename.endswith(ext) for ext in extensions)


def get_files_list(root_path, sequences, datafolder_name, extensions):
    if os.path.isdir(root_path):
        print("Sequences folder exists! Using sequences from %s" % root_path)
    else:
        raise ValueError("Sequences folder doesn't exist! Exiting...")

    files_list = []

    for seq in sequences:


        print("parsing seq {}".format(seq))

        scan_path = os.path.join(root_path, seq, datafolder_name)

        # get files
        # scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        #     os.path.expanduser(scan_path)) for f in fn if is_scan(f,extensions)]
        scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(scan_path)) for f in fn if is_scan(f,extensions)]

        # extend list
        files_list.extend(scan_files)

    # sort for correspondance
    files_list.sort()

    print("Using {} scans from sequences {}".format(len(files_list),
                                                    sequences))

    return files_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path")
    parser.add_argument("--sequences", nargs='+', type=str)
    parser.add_argument("--df_name")
    parser.add_argument("--dl_name")
    parser.add_argument("--data_ext", nargs='+', type=str)
    parser.add_argument("--label_ext", nargs='+', type=str)
    parser.add_argument("--output_name")
    args = parser.parse_args()
    print(args)
    root_path = args.root_path
    sequences = args.sequences
    datafolder_name = args.df_name
    labelfolder_name = args.dl_name
    data_ext = args.data_ext
    label_ext = args.label_ext
    #data_list = get_files_list(root_path,sequences,datafolder_name,data_ext)
    label_list = get_files_list(root_path,sequences,labelfolder_name,label_ext) 
    datafilename = os.path.join(root_path,args.output_name)
    #labelfilename = os.path.join(root_path,"train_label.txt")
    print(datafilename)
    path_len = len(root_path)
    with open(datafilename,'w') as f:
        for label_path in label_list:
            datapath = label_path[path_len:].copy()
            datapath.replace(labelfolder_name,datafolder_name)
            datapath.replace("png",'jpg')
            f.write(f'{datapath} {label_path[path_len:]}\n')
    # with open(labelfilename,'w') as f:
    #     for i in label_list:
    #         f.write(i+'\n')
