#!/bin/bash
export PYTHONPATH=/home/usl/Code/Peng/data_collection/benchmarks/GSCNN-master/:$PYTHONPATH
echo $PYTHONPATH
python train.py --dataset rellis --bs_mult 2 --lr 0.0005 --exp norm3 --scale_max 1.0 --scale_min 1.0