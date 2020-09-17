#!/bin/bash
export PYTHONPATH=/home/usl/Code/Peng/data_collection/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/:$PYTHONPATH
echo $PYTHONPATH
PY_CMD="python -m torch.distributed.launch --nproc_per_node=2"
python /home/usl/Code/Peng/data_collection/benchmarks/scripts/test.py
$PY_CMD tools/train.py --cfg experiments/rellis/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
