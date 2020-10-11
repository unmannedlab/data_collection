#!/bin/bash
python split_data.py --root_path /home/usl/Datasets/rellis/ \
    --sequences 00004 \
    --df_name os1_cloud_node_kitti_bin \
    --dl_name os1_cloud_node_semantickitti_label_id \
    --data_ext bin \
    --label_ext label \
    --output_name 00004.lst
