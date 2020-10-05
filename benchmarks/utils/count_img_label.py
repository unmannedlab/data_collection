import numpy as np
from PIL import Image
import multiprocessing
import argparse
import time
import os
import numpy.linalg as lg
import pickle

m = 64 * 2048
label_table = {11:33, 13:33, 21:34, 22:31}

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


def single_count(file_path):
    label = np.array(Image.open(file_path))
    #print(label.shape)
    #label = label[:, :, 0]
    unique, unique_counts = np.unique(label, return_counts=True)
    if 1 in unique:
        colorid_path = file_path.replace("pylon_camera_node_label_id","pylon_camera_node_label_color")
        img_path = file_path.replace("pylon_camera_node_label_id","pylon_camera_node")
        img_path = img_path.replace("png","jpg")
        colorid = np.array(Image.open(colorid_path))
        img = np.array(Image.open(img_path))
        sv_img = np.concatenate((img,colorid),axis=1)
        filename = os.path.split(img_path)[1]
        #print(filename)
        sv_img = Image.fromarray(sv_img)
        sv_img.save(filename)
        print(filename)
    return {i: j for i, j in zip(unique, unique_counts)}

def count_label(label_paths,num_workers):
    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_workers)
    res = pool.map(single_count, [i for i in label_paths])
    pool.close()
    count_dict = {}
    for elm in res:
        for key in elm.keys():
            if key in count_dict.keys():
                count_dict[key]=count_dict[key]+elm[key]
            else:
                count_dict[key] = elm[key]
    print("Compute time:",time.time()-start_time)
    sum_count = 0
    for v in count_dict.values():
        sum_count+=v 
    # for k,v in count_dict.items():
    #     count_dict[k] = v/sum_count
    count_dict = sorted(count_dict.items())
    print(count_dict)
    print(len(count_dict))
    return count_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path",default="/media/maskjp/Datasets1/data_collection/20200213/trail_2/sequences")
    parser.add_argument("--sequences", nargs='+', type=str,default=["00002"])#default=["00000","00001","00002","00003","00004"])
    parser.add_argument("--dl_name",default="labels")
    parser.add_argument("--label_ext", type=str,default="label")
    parser.add_argument("--save_name",type=str)
    args = parser.parse_args()
    root_path = args.root_path
    sequences = args.sequences
    labelfolder_name = args.dl_name
    label_ext = args.label_ext
    label_list = get_files_list(root_path,sequences,labelfolder_name,label_ext) 
    count_dict = count_label(label_list,7)
    with open(f"{args.save_name}.pkl",'wb') as f:
        pickle.dump(count_dict,f)


