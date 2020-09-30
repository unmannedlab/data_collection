#!/bin/sh
export CUDA_VISIBLE_DEVICES="$c"
cd ./train/tasks/semantic;  ./train.py -d rellis  -ac ./salsanext_ouster.yml -dc ./config/labels/rellis.yaml -n rellis