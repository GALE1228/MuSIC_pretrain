taskset -c 1 python main.py \
     --gerenate_embeddingh5 --infer_embedding_data_process \
    --infer_fasta_path data/predict_data/mouse_test.fa \
    --gpuid 0 \
    --batch_size 64 \
    --pretrain_RNA_model RiNALMo

taskset -c 1 python main.py \
     --gerenate_embeddingh5 --infer_embedding_data_process \
    --infer_fasta_path data/predict_data/mouse_test.fa \
    --gpuid 0 \
    --batch_size 64 \
    --pretrain_RNA_model RiNALMo
