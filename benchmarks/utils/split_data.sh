#!/bin/bash
python split_data.py --root_path /home/usl/Datasets/rellis/ \
    --sequences 00000 00001 00002 00003 00004 \
    --df_name pylon_camera_node \
    --dl_name pylon_camera_node_label_id \
    --data_ext jpg \
    --label_ext png \
    --output_name all.lst
