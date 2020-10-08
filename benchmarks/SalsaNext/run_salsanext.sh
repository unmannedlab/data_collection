#!/bin/sh
export CUDA_VISIBLE_DEVICES="0,1"
cd ./train/tasks/semantic;  ./train.py -d /media/maskjp/Datasets4/data_collection/20200213/trail_2/sequences  -ac ./config/arch/salsanext_ouster.yml -dc ./config/labels/rellis.yaml -n rellis -l ./logs -p ""