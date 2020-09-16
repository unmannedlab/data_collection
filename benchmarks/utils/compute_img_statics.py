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
from sklearn.utils.class_weight import compute_class_weight


def compute_var(sq_num, el_num, b, m, axis=None):
    print(b*m*sq_num.sum(axis=axis), m*(el_num.sum(axis=axis))**2)
    return (-sq_num.sum(axis=axis)+m*(el_num.sum(axis=axis))**2)/(b*(m*b-1))


def is_scan(filename, extensions):
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

label_mapping = {0: 0,
                 3: 1,
                 4: 2,
                 5: 3,
                 6: 4,
                 7: 5,
                 8: 6,
                 9: 7,
                 10: 8,
                 12: 9,
                 15: 10,
                 17: 11,
                 19: 12,
                 23: 13,
                 27: 14,
                 29: 1,
                 30: 1,
                 31: 4,
                 32: 4,
                 33: 15,
                 34: 16}


def count_labels(labellist, num_worker):
    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_worker)
    res = pool.map(single_count, labellist)
    pool.close()
    count_dict = {}
    for elm in res:
        for key in elm.keys():
            if key in count_dict.keys():
                count_dict[key] = count_dict[key]+elm[key]
            else:
                count_dict[key] = elm[key]
    print("Compute time:", time.time()-start_time)
    return count_dict


def read_img(img_label_path):
    img_label = np.array(Image.open(img_label_path))
    img_label = img_label[:, :, 0]
    return img_label


def single_count(file_path):
    label = np.array(Image.open(file_path))
    label = label[:, :, 0]
    unique, unique_counts = np.unique(label, return_counts=True)
    return {i: j for i, j in zip(unique, unique_counts)}


def single_map(file_path):
    label = np.array(Image.open(file_path))
    label = label[:, :, 0]
    tmp = label.copy()
    for k,v in label_mapping.items():
        label[tmp==k] = v
    label = label.flatten()
    return label


def single_compute(file_path):
    img = np.array(Image.open(file_path))
    img_mean = np.mean(img, axis=(0,1))
    img_sq_mean = np.mean(np.square(img), axis=(0,1))
    img_min = np.min(img, axis=(0,1))
    img_max = np.max(img, axis=(0,1))
    # print(scan_mean,scan_max,scan_max,scan_sq_mean,scan_count)
    res = {
        "img_mean": img_mean,
        "img_sq_mean": img_sq_mean,
        "img_min": img_min,
        "img_max": img_max
    }
    return res


def compute_meanstd(imglist, num_worker):
    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_worker)
    res = pool.map(single_compute, imglist)
    pool.close()
    b = len(imglist)
    img = np.array(Image.open(imglist[0]))
    img_size = img.shape
    m = img_size[1]*img_size[2]

    img_mean_arr = np.array([ent['img_mean'] for ent in res])
    img_sq_mean_arr = np.array([ent['img_sq_mean'] for ent in res])
    img_min_arr = np.array([ent['img_min'] for ent in res])
    img_max_arr = np.array([ent['img_max'] for ent in res])
    img_mean = np.mean(img_mean_arr, axis=0)
    img_var = compute_var(img_sq_mean_arr, img_mean_arr, b, m)
    img_min = np.min(img_min_arr, axis=0)
    img_max = np.max(img_max_arr, axis=0)
    return {"img_mean": img_mean, "img_var": img_var, "img_min": img_min, "img_max": img_max}


def calculateWeights(target,num_classes,norm = True,upper_bound=1.0):
    hist = np.histogram(target.flatten(), range(num_classes+1), normed=True)[0]
    print(hist)
    if norm:
        hist = ((hist != 0) * upper_bound * (1 / hist)) + 1
    else:
        hist = ((hist != 0) * upper_bound * (1 - hist)) + 1
    return hist

def count_weight(label_paths, num_workers):
    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_workers)
    res = pool.map(single_map, label_paths)
    pool.close()
    label = np.array(res)
    print(label.shape)
    label = label.flatten()
    classes = [i for i in label_mapping.values()]

    real_classes = np.unique(classes)
    classes = np.unique(label)
    print(classes,label)
    weight = calculateWeights(label,len(classes))
    return weight


def count_label(label_paths, num_workers):
    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_workers)
    res = pool.map(single_count, label_paths)
    pool.close()
    count_dict = {}
    for elm in res:
        for key in elm.keys():
            if key in count_dict.keys():
                count_dict[key] = count_dict[key]+elm[key]
            else:
                count_dict[key] = elm[key]
    print("Compute time:", time.time()-start_time)
    return count_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", '-p')
    parser.add_argument("--sequence", '-s', nargs='+')
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()

    rootpath = args.path
    seq = args.sequence
    labellist = get_files_list(
        rootpath, seq, "pylon_camera_node_label_id", [".png"])
    imglist = get_files_list(
        rootpath, seq, "pylon_camera_node", [".jpg"])
    count_dict = count_label(labellist, args.num_workers)
    img_statics = compute_meanstd(imglist, args.num_workers)
    x = list(range(len(label_dict)))
    for i in label_dict.keys():
        count_dict[i] = count_dict[i] if i in count_dict.keys() else 0
    fig, ax = plt.subplots(figsize=(40, 10))
    count_dict = dict(sorted(count_dict.items()))
    plt.title(f"{args.name}", fontsize=20.0)
    plt.rcParams.update({'font.size': 40})
    rects = ax.bar(x, count_dict.values())
    ax.set_xticks(x)
    ax.set_xticklabels(label_dict.values(), fontsize=15.0)
    plt.savefig(f"{args.name}.png")
    weight = count_weight(labellist,args.num_workers)
    with open(f"{args.name}_statics.txt", "w") as f:
        for k in count_dict:
            f.write(f"{label_dict[k]}: {count_dict[k]}\n")
        f.write(f"mean: {img_statics['img_mean']}\n")
        f.write(f"var: {img_statics['img_var']}")
        f.write(f"std: {np.sqrt(img_statics['img_var'])}\n")
        f.write(f"min: {img_statics['img_min']}\n")
        f.write(f"max: {img_statics['img_max']}\n")
        f.write(f"weight: {weight}")


