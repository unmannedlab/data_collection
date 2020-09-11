#/bin/bash
python compute_pt_statics.py --data_rootpath /media/maskjp/Datasets1/SemanticKITTI/dataset/sequences \
    --data_file /media/maskjp/Datasets1/SemanticKITTI/dataset/sequences/train_data.txt \
    --label_file /media/maskjp/Datasets1/SemanticKITTI/dataset/sequences/train_label.txt \
    --num_workers 5
