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

for item in tqdm(img_list):
    image_path, label_path = item
    image_path = os.path.join(root_path,image_path)
    label_path = os.path.join(root_path,label_path)
    img = np.array(Image.open(image_path))
    label = Image.open(label_path)
    #print(label.size)
    label = np.array(label)
    #print(img.shape,label.shape)
    #print(label_path,image_path)
    catimg = np.concatenate((img,label),axis=1)
    #catimg = Image.fromarray(catimg)
    #print(catimg.shape)
    w.append_data(catimg.astype(np.uint8))

w.close()