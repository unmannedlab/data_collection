import numpy as np 
from PIL import Image
import os 
import imageio
from tqdm import tqdm
import pickle
import time
import multiprocessing 

def single_count(file_path):
    label = np.array(Image.open(file_path))
    #print(file_path)
    label = label[:, :]
    unique, unique_counts = np.unique(label, return_counts=True)
    if 1 in unique:
        print(file_path)
        colorid_path = file_path.replace("pylon_camera_node_label_id","pylon_camera_node_label_color")
        img_path = file_path.replace("pylon_camera_node_label_id","pylon_camera_node")
        img_path = img_path.replace("png","jpg")
        colorid = np.array(Image.open(colorid_path))
        img = np.array(Image.open(img_path))
        sv_img = np.concatenate((colorid,img),axis=1)
        sv_img = Image.fromarray(sv_img)
        sv_name = os.path.split(img_path)[1]
        sv_img.save(sv_name)
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

filename = "all"
root_path = "/home/usl/Datasets/rellis"
list_path = f"/home/usl/Datasets/rellis/{filename}.lst"
img_list = [line.strip().split() for line in open(list_path)]

label_list = [os.path.join(root_path,i[1]) for i in img_list]
label_list = [i.replace("pylon_camera_node_label_color","pylon_camera_node_label_id") for i in label_list]
print(label_list[0])
count_dict = count_label(label_list,7)
with open(f"{filename}.pkl",'wb') as f:
    pickle.dump(count_dict,f)