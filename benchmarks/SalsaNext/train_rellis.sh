#!/bin/sh
export CUDA_VISIBLE_DEVICES="$c"
cd ./train/tasks/semantic;  ./train.py -d "rellis"  -ac "$a" -l "$l" -n "$n" -p "$p" -u "$u"