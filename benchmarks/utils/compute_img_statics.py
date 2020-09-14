import numpy as np
from PIL import Image
import numpy as np
from PIL import Image
import argparse
import multiprocessing
import matplotlib.pyplot as plt
import os
import time
import matplotlib.pyplot as plt 

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
        scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(scan_path)) for f in fn if is_scan(f, extensions)]

        # extend list
        files_list.extend(scan_files)

    # sort for correspondance
    files_list.sort()

    print("Using {} scans from sequences {}".format(len(files_list),
                                                    sequences))
    return files_list


def compute_meanstd():
    pass

label_dict = {0: "0_uknown",
              3: "3_grass",
              4: "4_tree",
              5: "5_pole",
              6: "6_water",
              7: "7_sky",
              8: "8_vehicle",
              9: "9_object",
              10: "10_asphalt",
              12: "12_building",
              15: "15_log",
              17: "17_person",
              19: "19_bush",
              23: "23_concrete",
              27: "27_barrier",
              29: "29_uphill",
              30: "30_downhill",
              31: "31_puddle",
              32: "32_deepwater",
              33: "33_mud",
              34: "34_rubble"}

def count_labels(labellist,num_worker):
    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_worker)
    res = pool.map(single_count, labellist)
    pool.close()
    count_dict = {}
    for elm in res:
        for key in elm.keys():
            if key in count_dict.keys():
                count_dict[key]=count_dict[key]+elm[key]
            else:
                count_dict[key] = elm[key]
    print("Compute time:",time.time()-start_time)
    return count_dict

def read_img(img_label_path):
    img_label = np.array(Image.open(img_label_path))
    img_label = img_label[:, :, 0]
    return img_label

def single_count(file_path):
    label = np.array(Image.open(file_path))
    label = label[:, :, 0]
    unique, unique_counts = np.unique(label,return_counts = True)
    #print(unique,unique_counts)
    return {i:j for i,j in zip(unique,unique_counts)}

def count_label(label_paths,num_workers):
    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_workers)
    res = pool.map(single_count, label_paths)
    pool.close()
    count_dict = {}
    for elm in res:
        for key in elm.keys():
            if key in count_dict.keys():
                count_dict[key]=count_dict[key]+elm[key]
            else:
                count_dict[key] = elm[key]
    print("Compute time:",time.time()-start_time)
    return count_dict




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", '-p')
    parser.add_argument("--sequence", '-s', nargs='+')
    parser.add_argument("--num_workers",type=int)
    parser.add_argument("--name",type=str)
    args = parser.parse_args()

    rootpath = args.path
    seq = args.sequence
    labellist = get_files_list(rootpath,seq, "pylon_camera_node_label_id",[".png"])
    #labellist = get_files_list(rootpath,seq, "F8_annotation_files",[".png"])
    imglist = get_files_list(rootpath,seq, "pylon_camera_node_label_color",[".png"])
    count_dict = count_label(labellist,args.num_workers)
    x = list(range(len(label_dict)))
    for i in label_dict.keys(): 
        count_dict[i]=count_dict[i] if i in count_dict.keys() else 0 
    fig, ax = plt.subplots(figsize=(40,10))
    count_dict = dict(sorted(count_dict.items()))
    print(count_dict)
    #fig,ax = plt.figure(figsize=(20,20))
    print(count_dict.keys(),x)
    plt.title(f"{args.name}",fontsize = 20.0)
    plt.rcParams.update({'font.size': 40})
    rects = ax.bar(x,count_dict.values())
    ax.set_xticks(x)
    ax.set_xticklabels(label_dict.values(),fontsize = 15.0)
    #plt.show()
    plt.savefig(f"{args.name}.png")
    #plt.show()

    