#!/bin/bash
python split_data.py --root_path /home/usl/Datasets/rellis/ \
    --sequences 00000 \
    --df_name pylon_camera_node \
    --dl_name pylon_camera_node_label_color \
    --data_ext jpg \
    --label_ext png \
    --output_name 00000.lst
