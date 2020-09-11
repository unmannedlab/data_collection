import numpy as np
from PIL import Image
import multiprocessing
import argparse
import time
import os

m = 64 * 2048


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
    res = pool.map(single_compute, [i[:-1] for i in data_path])
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
    res = pool.map(single_count, [i[:-1] for i in label_paths])
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
    parser.add_argument("--label_file")
    parser.add_argument("--num_workers",type=int)
    #parser.add_argument("--label_ids")

    args = parser.parse_args()
    with open(args.data_file, 'r') as f:
        data_paths = f.readlines()

    with open(args.label_file, 'r') as f:
        label_paths = f.readlines()

    num_workers = args.num_workers
    res_min,res_max,res_mean,res_var = compute_meanstd(data_paths, args.num_workers)
    count_dict = count_label(label_paths,num_workers)
    res_std = np.sqrt(res_var)
    
    log_path = os.path.join(args.data_rootpath,"statics.txt")

    with open(log_path,'w') as f:
        f.write(f"res_min:{res_min}\n")
        f.write(f"res_max:{res_max}\n")
        f.write(f"res_mean:{res_mean}\n")
        f.write(f"res_var:{res_var}\n")
        f.write(f"res_std:{res_std}\n")
        for k in count_dict:
            f.write(f"{k}: {count_dict[k]}\n")

