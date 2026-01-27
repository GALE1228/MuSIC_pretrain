# Deploying Pretrained Model Environment

# Install RiNALMo
1. Activate the `MuSIC_pretrain` environment:
    ```bash
    conda activate MuSIC_pretrain
    ```
2. Navigate to the RNA directory:
    ```bash
    cd ./pretrained_model/RNA
    ```
3. Clone the RiNALMo repository:
    ```bash
    git clone https://github.com/lbcb-sci/RiNALMo
    cd RiNALMo
    ```

### Note:
In our deployment tests, using the `./rinalmo/pretrained.py` script to download and load the pretrained `.pt` model file directly may result in errors.  
We recommend replacing the `./rinalmo/pretrained.py` file with the `pretrained_model/RNA/pretrained.py` file provided in this repository.  
You will also need to manually download the pretrained model weights.

---

## Download RiNALMo Weights
1. Install `gdown`:
    ```bash
    pip install -U gdown
    ```
2. Create the weights directory:
    ```bash
    mkdir -p ./pretrained_model/RNA/RiNALMo/weights
    ```
3. Download the pretrained weights using `gdown`:
    ```bash
    gdown 1-E2Ziu2VFDAgwCmQvVeAviGtsQQ94L3L -O ./pretrained_model/RNA/RiNALMo/weights/rinalmo_giga_pretrained.pt
    ```
    Alternatively, you can download the weights from the following URL:  
    [Download Pretrained Weights](https://drive.usercontent.google.com/download?id=1-E2Ziu2VFDAgwCmQvVeAviGtsQQ94L3L&export=download&authuser=0)  
    Save the file to weights.

---

## Install RiNALMo
Install the RiNALMo package:
```bash
pip install ./pretrained_model/RNA/RiNALMo/.
```

---

## Install Flash-Attention for Acceleration
1. Install `flash-attn`:
    ```bash
    pip install flash-attn==2.3.2
    ```
2. If the above method fails, download and install the `.whl` file manually:
    ```bash
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.2/flash_attn-2.3.2+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    pip install flash_attn-2.3.2+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    ```

---

## Test Installation
Run the following script to verify the installation:
```bash
python ./pretrained_model/test_rna_RiNALMo.py
```

---

# Install proT5 and Embed RBP Sequences
1. Navigate to the protein directory:
    ```bash
    cd ./pretrained_model/protein
    ```
2. Download the `prot_t5_xl_uniref50` model:
    ```bash
    curl -L -o prot_t5_xl_uniref50.zip https://zenodo.org/records/4644188/files/prot_t5_xl_uniref50.zip
    unzip prot_t5_xl_uniref50.zip
    ```

---

## Create Protein Embedding Environment
1. Create and activate a new Conda environment:
    ```bash
    conda env create -f ./pretrained_model/environment.yml
    conda activate MuSIC_prot5
    ```
2. Requirements:
    ```bash
    transformers
    sentencepiece
    torch==2.6.0
    pandas
    protobuf
    google
    sentencepiece
    ```

---

## Test Protein Embedding
Run the following script to test the embedding process and embed all test RBP sequences:
```bash
python ./RBP_embedding.py
```

For more details, refer to the [proT5 GitHub page](https://github.com/agemagician/ProtTrans).
