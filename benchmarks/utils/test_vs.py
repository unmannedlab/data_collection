import numpy as np
import open3d as o3d
import yaml
import matplotlib.pyplot as plt

def read_kitti(file_path):
    f = open(file_path,'rb')
    data = np.fromfile(f,'<f4')
    data = data.reshape((int(len(data)/4),4))
    return data

def open_label(filename,lut):

    label = np.fromfile(filename, dtype=np.int32)
    label = label.reshape((-1))
    rgb_label = np.zeros((len(label),3))
    for k,v in lut.items():
        rgb_label[label==k]=v
    return rgb_label


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def save(vis):
    # This function is called within the o3d.visualization.Visualizer::run() loop
    # The run loop calls the function, then re-render
    # So the sequence in this function is to:
    # 1. Capture frame
    # 2. index++, check ending criteria
    # 3. Set camera
    # 4. (Re-render)
    ctr = vis.get_view_control()
    image = vis.capture_screen_float_buffer(False)
    plt.imsave("test.png", np.asarray(image), dpi = 1)
    print("test")
    file_path = "/media/maskjp/Datasets/data_collection/20200213/trail_2/sequences/00003/os1_cloud_node_kitti_bin/001000.bin"
    label_path = file_path.replace("os1_cloud_node_kitti_bin","os1_cloud_node_semantickitti_label_id")
    label_path = label_path.replace("bin","label")
    xyz = read_kitti(file_path)[:,:3]
    rgb = open_label(label_path,lut_dict)/255
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    image2 = vis.capture_screen_float_buffer(False)
    plt.imsave("test2.png", np.asarray(image2), dpi = 1)    
    print("test2")
    vis.register_animation_callback(None)
    return False

def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.register_animation_callback(save)
    vis.run()
    vis.destroy_window()

label_config = "/media/maskjp/Datasets/Code/data_collection/benchmarks/SalsaNext/train/tasks/semantic/config/labels/rellis.yaml"
with open(label_config) as f:
    label_config = yaml.load(f, Loader=yaml.FullLoader)

lut_dict = label_config['color_map']

file_path = "/media/maskjp/Datasets/data_collection/20200213/trail_2/sequences/00003/os1_cloud_node_kitti_bin/000300.bin"
label_path = file_path.replace("os1_cloud_node_kitti_bin","os1_cloud_node_semantickitti_label_id")
label_path = label_path.replace("bin","label")
xyz = read_kitti(file_path)[:,:3]
rgb = open_label(label_path,lut_dict)/255

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)
o3d.io.write_point_cloud("test.ply", pcd)
#save_view_point(pcd,"viewpoint.json")
load_view_point(pcd,"viewpoint.json")
#o3d.visualization.draw_geometries([pcd])

