taskset -c 0 python main.py \
    --train \
    --rbp_name FUS_HITS-CLIP_Human \
    --smooth_rate 0.8725 \
    --file_path ./data/cross_species \
    --source_species HUMAN \
    --target_species MOUSE \
    --gpuid 0 \
    --batch_size 64 \
    --pretrain_RNA_model RiNALMo

taskset -c 0 python main.py \
    --train \
    --rbp_name UPF1_Human \
    --smooth_rate 0.7824 \
    --file_path ./data/cross_species \
    --source_species HUMAN \
    --target_species DROME \
    --gpuid 0 \
    --batch_size 64 \
    --pretrain_RNA_model RiNALMo