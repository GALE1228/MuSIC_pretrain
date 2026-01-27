import argparse
from .annotation_tools import *
import os

def process_train_rnafold_data(data_path, rbp_name):
    """
    Process RNAfold data, perform folding, and annotation.
    
    Parameters:
        data_path (str): The path where the data is stored.
        rbp_name (str): The RBP name, such as "AGO2_MiClip".
    """
    # Data types (positive and negative) and task types (train and test)
    dts = ["positive", "negative"]

    # Loop through all combinations of data type and task type
    for dt in dts:
        # Perform RNAfold
        rnafold_result = run_rnafold(data_path, dt, rbp_name, tt = "train")
        # Generate the annotation file path
        seq_str_anno_file = f"{data_path}/{rbp_name}/{dt}_data/train_annotation.tsv"
        # Process RNAfold result and annotate
        process_rnafold_and_annotate(rnafold_result, seq_str_anno_file)

    print("RNAfold processing, annotation complete and generate h5 file for MuSIC.")

def process_validation_rnafold_data(data_path, rbp_name):
    """
    Process RNAfold data, perform folding, and annotation.
    
    Parameters:
        data_path (str): The path where the data is stored.
        rbp_name (str): The RBP name, such as "AGO2_MiClip".
    """
    # Data types (positive and negative) and task types (train and test)
    dts = ["positive", "negative"]

    # Loop through all combinations of data type and task type
    for dt in dts:
        # Perform RNAfold
        rnafold_result = run_rnafold(data_path, dt, rbp_name ,tt = "test")
        # Generate the annotation file path
        seq_str_anno_file = f"{data_path}/{rbp_name}/{dt}_data/test_annotation.tsv"

        print("Begin to process RNAfold result and generate annotation file.")

        # Process RNAfold result and annotate
        process_rnafold_and_annotate(rnafold_result, seq_str_anno_file)

    print("RNAfold processing, annotation complete and generate h5 file for MuSIC.")

def process_rnafold_infer_data(fasta_filepath):
    """
    Process an unlabeled dataset, perform inference or calculate HAR, etc.
    """
    # Perform RNAfold
    rnafold_result = run_infer_rnafold(fasta_filepath)
    # Generate the annotation file path
    seq_str_anno_file = os.path.splitext(fasta_filepath)[0] + "_annotation.tsv"

    print("Begin to process RNAfold result and generate annotation file.")

    # Process RNAfold result and annotate
    process_rnafold_and_annotate(rnafold_result, seq_str_anno_file)

    print("RNAfold processing, annotation complete and generate h5 file for MuSIC.")