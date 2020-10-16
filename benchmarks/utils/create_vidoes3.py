import numpy as np
from PIL import Image
import os
import imageio
from tqdm import tqdm

filename = "test"
root_path = "/home/usl/Datasets/rellis"
list_path = f"/home/usl/Datasets/rellis/split/{filename}.lst"
img_list = [line.strip().split() for line in open(list_path)]
w = imageio.get_writer(f'{filename}.mp4', format='FFMPEG', mode='I', fps=10)

label_mapping = {0: 0,
                 1: 3,
                 2: 4,
                 3: 5,
                 4: 6,
                 5: 7,
                 6: 8,
                 7: 9,
                 8: 10,
                 9: 12,
                 10: 15,
                 11: 17,
                 12: 18,
                 13: 19,
                 14: 23,
                 15: 27,
                 16: 31,
                 17: 33,
                 18: 34}
color_map = {
    0: [0, 0, 0],
    1: [108, 64, 20],
    3: [0, 102, 0],
    4: [0, 255, 0],
    5: [0, 153, 153],
    6: [0, 128, 255],
    7: [0, 0, 255],
    8: [255, 255, 0],
    9: [255, 0, 127],
    10: [64, 64, 64],
    12: [255, 0, 0],
    15: [102, 0, 0],
    17: [204, 153, 255],
    18: [102, 0, 204],
    19: [255, 153, 204],
    23: [170, 170, 170],
    27: [41, 121, 255],
    31: [134, 255, 239],
    33: [99, 66, 34],
    34: [110, 22, 138]
}

def convert_label(label, label_mapping, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label

def colorize_label(label,color_map):
    shape = label.shape
    color_label = np.zeros(shape+(3,))
    for k, v in color_map.items():
        color_label[label==k,:]= v
    return color_label

for item in tqdm(img_list):
    image_path, label_path = item
    image_path = os.path.join(root_path, image_path)
    label_path = os.path.join(root_path, label_path)
    hrnet_path = label_path.replace("rellis", "hrnet")
    gscnn_path = label_path.replace("rellis", 'gscnn')
    img = np.array(Image.open(image_path))
    label = Image.open(label_path)
    label = np.array(label)
    label_shape = label.shape
    hrnet_pred = Image.open(hrnet_path)
    if label_shape[0] != hrnet_pred.size[0] or label_shape[1] != hrnet_pred.size[1]:
        hrnet_pred = hrnet_pred.resize((label_shape[1],label_shape[0]),Image.NEAREST)
    hrnet_pred = np.array(hrnet_pred)[:,:,0]
    hrnet_pred = convert_label(hrnet_pred,label_mapping)
    gscnn_pred = Image.open(gscnn_path)
    if label_shape[0] != gscnn_pred.size[0] or label_shape[1] != gscnn_pred.size[1]:
        gscnn_pred = gscnn_pred.resize((label_shape[1],label_shape[0]),Image.NEAREST)
    gscnn_pred = np.array(gscnn_pred)[:,:,0]
    gscnn_pred = convert_label(gscnn_pred,label_mapping)
    label = colorize_label(label,color_map)
    gscnn_pred = colorize_label(gscnn_pred,color_map)
    hrnet_pred = colorize_label(hrnet_pred,color_map)

    gtlabel = np.concatenate((img, label), axis=1)
    pred = np.concatenate((gscnn_pred, hrnet_pred), axis=1)
    catimg = np.concatenate((gtlabel, pred), axis=0)
    # print(catimg.shape)
    w.append_data(catimg.astype(np.uint8))

w.close()
