taskset -c 1 python main.py \
    --har \
    --infer_fasta_path data/predict_data/mouse_test.fa \
    --rbp_name FUS_HITS-CLIP_Human \
    --smooth_rate 0.8725 \
    --source_species HUMAN \
    --target_species MOUSE \
    --gpuid 0 \
    --batch_size 64 \
    --pretrain_RNA_model RiNALMo

taskset -c 1 python main.py \
    --har \
    --infer_fasta_path data/predict_data/fly_test.fa \
    --rbp_name UPF1_Human \
    --smooth_rate 0.7824 \
    --source_species HUMAN \
    --target_species DROME \
    --gpuid 0 \
    --batch_size 64 \
    --pretrain_RNA_model RiNALMo
