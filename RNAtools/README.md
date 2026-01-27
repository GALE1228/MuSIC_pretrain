# Install RNAfold

MuSIC uses RNAfold for RNA secondary structure prediction.  
Please ensure you are in the `MuSIC_pretrain` environment before proceeding.

## Installation Steps
1. Navigate to the `RNAtools` directory:
    ```bash
    cd ./RNAtools
    ```
2. Download the ViennaRNA source code:
    ```bash
    wget https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_7_x/ViennaRNA-2.7.2.tar.gz
    ```
3. Extract the downloaded archive:
    ```bash
    tar -zxvf ViennaRNA-2.7.2.tar.gz
    ```
4. Navigate to the extracted directory:
    ```bash
    cd ViennaRNA-2.7.2
    ```
5. Configure the installation:
    ```bash
    ./configure
    ```
6. Compile the source code:
    ```bash
    make
    make install
    ```

For more details, refer to the [official installation guide](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/install.html).