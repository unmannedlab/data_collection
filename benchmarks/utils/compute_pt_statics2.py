import numpy as np
from PIL import Image
import multiprocessing
import argparse
import time
import os

m = 64 * 2048


label_dict = {0: "0_uknown",
              1: "1_dirt",
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
              18: "18_fence",
              19: "19_bush",
              23: "23_concrete",
              27: "27_barrier",
              31: "31_puddle",
              33: "33_mud",
              34: "34_rubble"}

label_mapping = {0: 0,
                 1: 0,
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
                 18: 12,
                 19: 13,
                 23: 14,
                 27: 15,
                 31: 16,
                 33: 17,
                 34: 18}

def single_compute(file_path):
    scan = np.fromfile(file_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    
    scan_range = np.linalg.norm(scan[:,:-1],axis=1)
    scan = np.concatenate([np.expand_dims(scan_range,axis=1),scan],axis=1)
    scan_mean = np.sum(scan, axis=0)/m
    scan_sq_mean = np.sum(np.square(scan), axis=0)/m
    scan_min = np.min(scan, axis=0)
    scan_max = np.max(scan, axis=0)
    scan_count = scan.shape[0]
    #print(scan_mean,scan_max,scan_max,scan_sq_mean,scan_count)
    res = {
        "scan_mean": scan_mean,
        "scan_sq_mean": scan_sq_mean,
        "scan_min": scan_min,
        "scan_max": scan_max,
        "scan_count": scan_count
    }
    return res


def compute_meanstd(data_path, num_workers):
    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_workers)
    res = pool.map(single_compute, [i for i in data_path])
    pool.close()
    scans_mean = np.stack([elm["scan_mean"] for elm in res])
    scans_sq_mean = np.stack([elm["scan_sq_mean"] for elm in res])
    scans_min = np.stack([elm["scan_min"] for elm in res])
    scans_max = np.stack([elm["scan_max"] for elm in res])
    scans_count = np.array([elm['scan_count'] for elm in res])
    res_min = np.min(scans_min, axis=0)
    res_max = np.max(scans_max, axis=0)
    res_mean = np.mean(np.multiply(scans_mean,np.expand_dims(m/scans_count,axis=1)), axis=0)
    N = np.sum(scans_count)
    #print(np.sum(scans_sq_mean, axis=0),np.square(np.sum(scans_mean, axis=0)))
    res_var = np.sum(scans_sq_mean, axis=0)*(m/((N-1))) - \
        (np.sum(scans_mean, axis=0)**2)*(m**2/(N*(N-1)))
    #print(res_var)
    print("Compute time:",time.time()-start_time)
    return res_min,res_max,res_mean,res_var

def single_count(file_path):
    label = np.fromfile(file_path, dtype=np.int32)
    label = label.reshape((-1))
    label = label & 0xFFFF
    unique, unique_counts = np.unique(label,return_counts = True)
    #print(unique,unique_counts)
    return {i:j for i,j in zip(unique,unique_counts)}

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
    return count_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_rootpath")
    parser.add_argument("--data_file")
    parser.add_argument("--num_workers",type=int)
    parser.add_argument("--sv_name")
    #parser.add_argument("--label_ids")

    args = parser.parse_args()
    data_rootpath = args.data_rootpath
    img_list = [line.strip().split() for line in open(args.data_file)]
    data_paths = []
    label_paths = []
    for em in img_list:
        data_paths.append(os.path.join(data_rootpath,em[0]))
        label_paths.append(os.path.join(data_rootpath,em[1]))

    num_workers = args.num_workers
    res_min,res_max,res_mean,res_var = compute_meanstd(data_paths, args.num_workers)
    count_dict = count_label(label_paths,num_workers)
    res_std = np.sqrt(res_var)
    
    log_path = os.path.join(args.data_rootpath,f"{args.sv_name}.txt")

    with open(log_path,'w') as f:
        f.write(f"res_min:{res_min}\n")
        f.write(f"res_max:{res_max}\n")
        f.write(f"res_mean:{res_mean}\n")
        f.write(f"res_var:{res_var}\n")
        f.write(f"res_std:{res_std}\n")
        for k in count_dict:
            f.write(f"{k}: {count_dict[k]}\n")

