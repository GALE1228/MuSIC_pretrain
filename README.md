# 🧬 MuSIC

MuSIC is a deep learning toolkit for predicting RNA-binding protein (RBP) interactions with RNA across multiple species, leveraging both sequence and secondary structure information, and evolutionary conservation. It supports within-species and cross-species prediction and high-attention region analysis.

**Authors:**  
Jiale He*, Tong Zhou*, Lufeng Hu*, Yuhua Jiao, Junhao Wang, Shengwen Yan, Siyao Jia, Qiuzhen Chen, Yangming Wang, Yucheng T. Yang, Lei Sun  

*Equal contribution

---

### 🧩 MuSIC Framework
![MuSIC](fig/music.png)

- [Getting Started](#getting-started)
- [Datasets](#datasets)
- [Usage](#usage)
- [Output Directory Structure](#output-directory-structure)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
---

## Getting Started

### 1. Environment Setup

Create environment
```bash
conda env create -f environment.yml
conda activate MuSIC_pretrain
```

- **CUDA:**  
  GPU acceleration is recommended. Ensure CUDA 11.8 and compatible PyTorch are installed.

### 2. Pretrained Model Installation and Environment Setup

See the [./pretrained_model/README.md](./pretrained_model/README.md).

### 3. RNAfold Installation

See the [./RNAtools/README.md](./RNAtools/README.md).

### 4. Requirements

- Python 3.11.5
- PyTorch 2.1.0 (CUDA recommended)
- See `environment.yml` for all dependencies.

---

## Datasets 

### Directory Structure

```text
data/
├── within_species/      # Within-species test datasets
├── cross_species/       # Cross-species test datasets
├── predict_data/             # Example FASTA files for prediction
├── protein_fasta_embedding/           # RBP sequence and embedding from ProT5
```
### Data Preprocessing

Convert FASTA to H5 (with structure prediction and sequence embedding):

```bash
taskset -c 0 python main.py --gerenate_embeddingh5 \
    --infer_embedding_data_process \
    --infer_fasta_path ./data/predict_data/mouse_test.fa \
    --gpuid 0 \
    --batch_size 64 \
    --pretrain_RNA_model RiNALMo
```
**Options:**
- `--train_embedding_data_process`: Use this option for training datasets.
- `--validate_embedding_data_process`: Use this option for validation datasets.
- `--infer_embedding_data_process`: Use this option for inference datasets.

**Output:**  
The resulting file, `mouse_test_RiNALMo_rnaembedding.h5`, will be saved in the same directory as the input FASTA file.
### Data Format

- ***.fa**: Input RNA sequences
- ***_annotation.tsv**: Structural annotation files for datasets
- ***_RiNALMo_rnaembedding.h5**: Combined sequence embeddings and structure features for model input

---

## Usage

### Cross-Species Training & Validation

```bash
# Cross-Species Training
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
```

```bash
# Cross-Species Validation
taskset -c 0 python main.py \
    --validate \
    --rbp_name FUS_HITS-CLIP_Human \
    --smooth_rate 0.8725 \
    --file_path ./data/cross_species \
    --source_species HUMAN \
    --target_species MOUSE \
    --gpuid 0 \
    --batch_size 64 \
    --pretrain_RNA_model RiNALMo
```

**Additional Arguments:**
- `--train`: Initiates the model training process.
- `--validate`: Initiates the model validation process.
- `--rbp_name`: Specifies the name of the RBP.
- `--smooth_rate`: Defines the label smoothing rate, which reflects the conservation between the target RBP and the source RBP.
- `--file_path`: Specifies the path to the dataset directory.
- `--source_species`: Indicates the source species for the dataset, used to retrieve the RBP sequence embeddings for the source species.
- `--target_species`: Indicates the target species for the dataset, used to retrieve the RBP sequence embeddings for the target species.
- `--gpuid`: Specifies the GPU device ID to be used for training or validation.
- `--batch_size`: Defines the batch size for training or validation.
- `--pretrain_RNA_model`: Specifies the name of the pretrained RNA model to be used.

---

### Within-Species Training & Validation

```bash
# Within-Species Training
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
```

```bash
# Within-Species Validation
taskset -c 0 python main.py \
    --validate \
    --rbp_name PUM2 \
    --smooth_rate 1 \
    --file_path ./data/within_species \
    --source_species HUMAN \
    --target_species HUMAN \
    --gpuid 0 \
    --batch_size 64 \
    --pretrain_RNA_model RiNALMo
```

**Arguments:**
- `--train`: Start the model training process.
- `--validate`: Start the model validation process.
- `--rbp_name`: Name of the RBP.
- `--smooth_rate`: Label smoothing rate. For within-species training, this value is set to 1.
- `--file_path`: Path to the dataset directory.
- `--source_species`: The source species for the dataset.
- `--target_species`: The target species for the dataset. For within-species tasks, this should be the same as the source species.
- `--gpuid`: GPU device ID to use for training or validation.
- `--batch_size`: Batch size for training or validation.
- `--pretrain_RNA_model`: Name of the pretrained RNA model to use.

---

### Inference (Prediction)

```bash
taskset -c 1 python main.py \
    --infer \
    --infer_fasta_path data/predict_data/mouse_test.fa \
    --rbp_name FUS_HITS-CLIP_Human \
    --smooth_rate 0.8725 \
    --source_species HUMAN \
    --target_species MOUSE \
    --gpuid 0 \
    --batch_size 64 \
    --pretrain_RNA_model RiNALMo
```
**Output:**  
Inference results are saved as `.inference` files in `music/out/infer/`.

---

### High Attention Region (HAR) Computation

```bash
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
```
**Output:**  
HAR results are stored in `music/out/har/`.

---

### Motif Resource

Predicted RNA binding motifs for 184 RBPs across 11 species are available for download:

- **PNG:** [MuSIC 11 Species Motif Collection (PNG)](https://github.com/GALE1228/music_11species_motif_png2026)

---

## Output Directory Structure

- `music/out/model/`: Trained model weights (`.pth`)
- `music/out/logs/`: Training and validation logs (`.txt`)
- `music/out/infer/`: Inference results (`.inference`)
- `music/out/har/`: High Attention Region results (`.txt`)
- `music/out/evals/`: Evaluation metrics (`.metrics`, `.probs`)


## Citation

If you use MuSIC in your research, please cite:

```bibtex
@article{xxx,
  title={xxx},
  author={xxx},
  year={xxx},
  doi={xxx},
  journal={xxx}
}
```

---

## License

This project is covered under the MIT License.

---

## Contact

Thank you for using MuSIC! For questions, bug reports, or contributions, please contact the authors or open an issue on GitHub.

---
