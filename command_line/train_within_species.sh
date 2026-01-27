taskset -c 0 python main.py \
    --train \
    --rbp_name PUM2 \
    --smooth_rate 1 \
    --file_path ./data/within_species \
    --source_species HUMAN \
    --target_species HUMAN \
    --gpuid 0 \
    --batch_size 64 \
    --pretrain_RNA_model RiNALMo

taskset -c 0 python main.py \
    --train \
    --rbp_name TRA2A \
    --smooth_rate 1 \
    --file_path ./data/within_species \
    --source_species HUMAN \
    --target_species HUMAN \
    --gpuid 0 \
    --batch_size 64 \
    --pretrain_RNA_model RiNALMo