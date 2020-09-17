#!/bin/bash
python split_data.py --root_path /media/maskjp/Datasets1/SemanticKITTI/dataset/sequences/ \
    --sequences 01 02 03 04 05 06 07 09 10 \
    --df_name velodyne \
    --dl_name labels \
    --data_ext bin \
    --label_ext label
