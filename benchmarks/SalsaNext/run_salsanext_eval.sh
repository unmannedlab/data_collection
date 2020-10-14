#!/bin/sh
export CUDA_VISIBLE_DEVICES="0,1"
cd ./train/tasks/semantic
python infer2.py -d /home/usl/Datasets/rellis -l /home/usl/Datasets -s test -m /home/usl/Code/Peng/data_collection/benchmarks/SalsaNext/train/tasks/semantic/logs/logs/2020-10-11-17_03rellis