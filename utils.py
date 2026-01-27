import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import h5py
import numpy as np
from data_gerenate.RNAfold_annotation_gerenate_h5 import *
from train_code.rna_embedding_RiNALMo import RiNALMo_Embedder
from data_gerenate.one_hot_encode_decode import *

from tqdm import tqdm
import gc

def is_bad_file(path, min_size_mb=1):
    """Check if a file is missing, empty, or smaller than a minimum size."""
    if not os.path.exists(path):
        return True
    size = os.path.getsize(path)
    if size == 0:
        return True
    if size < min_size_mb * 1024 * 1024:
        return True
    return False

def init_fasta_headers(fasta_path, replacement='|'):
    """Initialize FASTA headers by replacing whitespace with a given character."""
    import re

    lines = []
    with open(fasta_path, 'r') as infile:
        for line in infile:
            if line.startswith('>'):
                clean_line = re.sub(r'[\s\t]+', replacement, line.strip())
                lines.append(clean_line + '\n')
            else:
                lines.append(line)

    with open(fasta_path, 'w') as outfile:
        outfile.writelines(lines)

    return fasta_path

def load_embedding_h5file(h5_file):
    """Load RNA names and combined features from an h5 file."""
    with h5py.File(h5_file, 'r') as f:
        rna_names = f['rna_names'][:]
        combined_features = f['combined_features'][:]
        
    return rna_names, combined_features

def load_tsv_file(device, tsv_file, max_length, embedder_name, save_path, rna_embedding_path):
    """Load RNA sequences from a TSV file, generate embeddings, and save as h5."""
    chunk_size = 10000

    total_lines = sum(1 for _ in open(tsv_file, "r"))
    print(f"Total RNA lines: {total_lines}")

    df_iter = pd.read_csv(tsv_file, sep="\t", header=None, chunksize=chunk_size)

    if embedder_name == "RiNALMo":
        embedder = RiNALMo_Embedder(
            model_path=rna_embedding_path,
            device=device
        )
        embed_dim = 1280
    else:
        raise ValueError(f"Unsupported embedder: {embedder_name}")

    with h5py.File(save_path, "w") as h5f:

        name_ds = h5f.create_dataset(
            "rna_names",
            shape=(0,), maxshape=(None,), dtype="S200"
        )
        feature_ds = h5f.create_dataset(
            "combined_features",
            shape=(0, max_length, embed_dim + 2),
            maxshape=(None, max_length, embed_dim + 2),
            dtype=np.float32
        )

        total_count = 0

        with tqdm(total=total_lines, desc="Processing RNA") as pbar:

            for chunk_df in df_iter:

                batch_names = []
                batch_features = []

                for index, row in chunk_df.iterrows():

                    rna_name = row.iloc[0]
                    seq = row.iloc[1].replace("T", "U")
                    structure = row.iloc[2]

                    embedding = embedder.embed([seq])

                    embedding = embedding.cpu().squeeze(0).numpy()
                    L = embedding.shape[0]

                    if L > max_length:
                        extra = L - max_length
                        left = extra // 2
                        right = extra - left
                        embedding = embedding[left:L-right, :]
                    else:
                        pad = max_length - L
                        embedding = np.vstack([embedding, np.zeros((pad, embedding.shape[1]), dtype=np.float32)])

                    str_2_oh = convert_one_hot_str_2(structure, max_length).T

                    combined = np.concatenate((embedding, str_2_oh), axis=1)

                    batch_names.append(rna_name.encode())
                    batch_features.append(combined)

                    pbar.update(1)

                batch_features = np.stack(batch_features, axis=0)

                old_size = name_ds.shape[0]
                new_size = old_size + len(batch_names)

                name_ds.resize((new_size,))
                feature_ds.resize((new_size, max_length, embed_dim + 2))

                name_ds[old_size:new_size] = batch_names
                feature_ds[old_size:new_size] = batch_features

                total_count = new_size
                h5f.flush()
                print(f"Completed {total_count} RNA")

    print(f"\nTotally {total_count} RNA.")
    h5f.close()

def load_tsv_file_infer(device, tsv_file, max_length, embedder_name, save_dir, rna_embedding_path):
    """Load RNA sequences from a TSV file for inference and save embeddings."""
    chunk_size = 5000
    total_lines = sum(1 for _ in open(tsv_file, "r"))
    print(f"Total RNA lines: {total_lines}")

    embedder = RiNALMo_Embedder(model_path=rna_embedding_path, device=device)
    embed_dim = 1280

    total_count = 0
    num_parts = (total_lines // chunk_size) + (1 if total_lines % chunk_size > 0 else 0)

    with tqdm(total=total_lines, desc="Processing RNA") as pbar:
        for part in range(num_parts):
            start_idx = part * chunk_size
            end_idx = min((part + 1) * chunk_size, total_lines)

            process_part(device, tsv_file, start_idx, end_idx, max_length, embedder_name, save_dir, embedder, embed_dim, pbar)

    print(f"\nTotally {total_count} RNA processed.")

def load_inferh5(tsv_file, embedding_dir, embedder_name):
    """Load inferred RNA embeddings from an h5 file for inference."""
    chunk_size = 5000
    total_lines = sum(1 for _ in open(tsv_file, "r"))
    print(f"Total RNA lines: {total_lines}")

    num_parts = (total_lines // chunk_size) + (1 if total_lines % chunk_size > 0 else 0)

    for part in range(num_parts):
        start_idx = part * chunk_size
        end_idx = min((part + 1) * chunk_size, total_lines)

        embedding_file = f"{embedding_dir}/{os.path.splitext(os.path.basename(tsv_file))[0]}_{embedder_name}_rnaembedding_{start_idx + 1}_{end_idx}.h5"
        print("load embedding feature from", embedding_file)
        rna_names, combined_features = load_embedding_h5file(embedding_file)

        yield rna_names, combined_features

def process_part(device, tsv_file, start_idx, end_idx, max_length, embedder_name, save_dir, embedder, embed_dim, pbar):
    """Process a part of the RNA data and save the embeddings to an h5 file."""
    embedding_file = f"{save_dir}/{os.path.splitext(os.path.basename(tsv_file))[0]}_{embedder_name}_rnaembedding_{start_idx + 1}_{end_idx}.h5"
    print(f"Processing RNA part: {start_idx + 1} to {end_idx}")

    if os.path.exists(embedding_file):
        print(f"{embedding_file} already exists, skipping.")
        return

    chunk_df = pd.read_csv(tsv_file, sep="\t", header=None, skiprows=range(1, start_idx + 1), nrows=(end_idx - start_idx))

    batch_names = []
    batch_features = []

    part_pbar = tqdm(total=end_idx - start_idx, desc=f"Processing RNA part {start_idx + 1} to {end_idx}", position=1, leave=False)

    for index, row in chunk_df.iterrows():
        rna_name = row.iloc[0]
        seq = row.iloc[1].replace("T", "U")
        structure = row.iloc[2]

        embedding = embedder.embed([seq])

        embedding = embedding.cpu().squeeze(0).numpy()
        L = embedding.shape[0]

        if L > max_length:
            extra = L - max_length
            left = extra // 2
            right = extra - left
            embedding = embedding[left:L-right, :]
        else:
            pad = max_length - L
            embedding = np.vstack([embedding, np.zeros((pad, embedding.shape[1]), dtype=np.float32)])

        str_2_oh = convert_one_hot_str_2(structure, max_length).T
        combined = np.concatenate((embedding, str_2_oh), axis=1)

        batch_names.append(rna_name.encode())
        batch_features.append(combined)

        part_pbar.update(1)

    batch_features = np.stack(batch_features, axis=0)

    with h5py.File(embedding_file, "w") as h5f:
        name_ds = h5f.create_dataset("rna_names", shape=(len(batch_names),), dtype="S200")
        feature_ds = h5f.create_dataset("combined_features", shape=(len(batch_features), max_length, embed_dim + 2), dtype=np.float32)

        name_ds[:] = batch_names
        feature_ds[:] = batch_features

    print(f"Completed RNA part {start_idx + 1} to {end_idx}, saved to {embedding_file}")

    del batch_names, batch_features
    gc.collect()

    pbar.update(end_idx - start_idx)

def gerenate_RNAembedding_h5(device, file_path, rbp_name, pretrain_RNA_model, data_process,rna_embedding_path):
    """Generate RNA embedding h5 files for training, testing, or inference."""
    if data_process == "train":
        # -------- Positive Embedding --------
        train_positive_embedding = f"{file_path}/{rbp_name}/positive_data/train_{pretrain_RNA_model}_rnaembedding.h5"
        bad = is_bad_file(train_positive_embedding)
        print("train_positive_embedding bad:", bad)

        if bad:
            print(f"{train_positive_embedding} missing or empty, generating h5 files.")
            train_positive_annotation_file = f"{file_path}/{rbp_name}/positive_data/train_annotation.tsv"
            if not os.path.exists(train_positive_annotation_file):
                process_train_rnafold_data(file_path, rbp_name)
                load_tsv_file(device, train_positive_annotation_file, max_length=200,
                            embedder_name=pretrain_RNA_model, save_path=train_positive_embedding, rna_embedding_path=rna_embedding_path)
            else:
                load_tsv_file(device, train_positive_annotation_file, max_length=200,
                            embedder_name=pretrain_RNA_model, save_path=train_positive_embedding, rna_embedding_path=rna_embedding_path)

        # -------- Negative Embedding --------
        train_negative_embedding = f"{file_path}/{rbp_name}/negative_data/train_{pretrain_RNA_model}_rnaembedding.h5"
        bad = is_bad_file(train_negative_embedding)
        print("train_negative_embedding bad:", bad)

        if bad:
            print(f"{train_negative_embedding} missing or empty, generating h5 files.")
            train_negative_annotation_file = f"{file_path}/{rbp_name}/negative_data/train_annotation.tsv"
            if not os.path.exists(train_negative_annotation_file):
                process_train_rnafold_data(file_path, rbp_name)
                load_tsv_file(device, train_negative_annotation_file, max_length=200,
                            embedder_name=pretrain_RNA_model, save_path=train_negative_embedding, rna_embedding_path=rna_embedding_path)
            else:
                load_tsv_file(device, train_negative_annotation_file, max_length=200,
                            embedder_name=pretrain_RNA_model, save_path=train_negative_embedding, rna_embedding_path=rna_embedding_path)

    elif data_process == "test":
        # -------- Positive Embedding --------
        test_positive_embedding = f"{file_path}/{rbp_name}/positive_data/test_{pretrain_RNA_model}_rnaembedding.h5"
        bad = is_bad_file(test_positive_embedding)
        print("test_positive_embedding bad:", bad)

        if bad:
            print(f"{test_positive_embedding} missing or empty, generating h5 files.")
            test_positive_annotation_file = f"{file_path}/{rbp_name}/positive_data/test_annotation.tsv"
            if not os.path.exists(test_positive_annotation_file):
                process_validation_rnafold_data(file_path, rbp_name)
                load_tsv_file(device, test_positive_annotation_file, max_length=200,
                            embedder_name=pretrain_RNA_model, save_path=test_positive_embedding, rna_embedding_path=rna_embedding_path)
            else:
                load_tsv_file(device, test_positive_annotation_file, max_length=200,
                            embedder_name=pretrain_RNA_model, save_path=test_positive_embedding, rna_embedding_path=rna_embedding_path)

        # -------- Negative Embedding --------
        test_negative_embedding = f"{file_path}/{rbp_name}/negative_data/test_{pretrain_RNA_model}_rnaembedding.h5"
        bad = is_bad_file(test_negative_embedding)
        print("test_negative_embedding bad:", bad)

        if bad:
            print(f"{test_negative_embedding} missing or empty, generating h5 files.")
            test_negative_annotation_file = f"{file_path}/{rbp_name}/negative_data/test_annotation.tsv"
            if not os.path.exists(test_negative_annotation_file):
                process_validation_rnafold_data(file_path, rbp_name)
                load_tsv_file(device, test_negative_annotation_file, max_length=200,
                            embedder_name=pretrain_RNA_model, save_path=test_negative_embedding, rna_embedding_path=rna_embedding_path)
            else:
                load_tsv_file(device, test_negative_annotation_file, max_length=200,
                            embedder_name=pretrain_RNA_model, save_path=test_negative_embedding, rna_embedding_path=rna_embedding_path)

    elif data_process == "infer":
        file_dir = os.path.dirname(file_path)
        file_prefix = os.path.splitext(os.path.basename(file_path))[0]
        embedding_dir = f"{file_dir}/{file_prefix}_{pretrain_RNA_model}_rnaembedding"
        embedding_annotation_file = f"{file_dir}/{file_prefix}_annotation.tsv"

        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)

        print("Checking if embeddings already exist...")

        if not os.path.exists(embedding_annotation_file):
            process_rnafold_infer_data(file_path)
            load_tsv_file_infer(device, tsv_file=embedding_annotation_file, max_length=200,
                                embedder_name=pretrain_RNA_model, save_dir=embedding_dir, rna_embedding_path=rna_embedding_path)

        else:
            load_tsv_file_infer(device, tsv_file=embedding_annotation_file, max_length=200,
                                embedder_name=pretrain_RNA_model, save_dir=embedding_dir, rna_embedding_path=rna_embedding_path)

def create_dataloader(X, y, y_smooth, batch_size):
    """Create a DataLoader for training with smooth labels."""
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    tensor_y_smooth = torch.Tensor(y_smooth)
    dataset = TensorDataset(tensor_x, tensor_y, tensor_y_smooth)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def create_dataloader_within(X, y, batch_size):
    """Create a DataLoader for training without smooth labels."""
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def create_infer_dataloader(X, y, rna_names, batch_size=32):
    """Create a DataLoader for inference."""
    tensor_x = torch.Tensor(X)  # Convert to Tensor
    tensor_y = torch.Tensor(y)  # Convert to Tensor
    dataset = TensorDataset(tensor_x, tensor_y)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Return DataLoader and rna_names
    return dataloader, rna_names

def split_dataset(data ,labels ,test_size):
    """Split dataset into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42 ,stratify=labels)
    return X_train, X_test, y_train, y_test

def load_model(model, model_path, device):
    """Load a pre-trained model from the specified path."""
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def make_directory(path, foldername, verbose=1):
    """Create a directory if it does not exist."""
    if not os.path.isdir(path):
        os.mkdir(path)
        print("making directory: " + path)

    outdir = os.path.join(path, foldername)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print("making directory: " + outdir)
    return outdir

def smooth_onehot_label(one_hot_label, smooth_rate):
    """Smooth the one-hot encoded labels using a specified smoothing rate."""
    total_count = one_hot_label.size(0)
    class_1_distribution = (one_hot_label == 1).sum().item() / total_count
    smoothed_labels = torch.zeros_like(one_hot_label)
    for i in range(one_hot_label.size(0)):
        if one_hot_label[i] == 1:
            smoothed_labels[i] = smooth_rate * one_hot_label[i] + (1 - smooth_rate) * class_1_distribution
        else:
            smoothed_labels[i] = smooth_rate * one_hot_label[i] + (1 - smooth_rate) * (1 - class_1_distribution)

    smoothed_labels = smoothed_labels.cpu().numpy()
    return smoothed_labels

def train_dataset_RNAembedding(device, file_path, rbp_name, batch_size, smooth_rate, pretrain_RNA_model, rna_embedding_path):
    """Generate and load training dataset with RNA embeddings."""
    train_positive_embedding = f"{file_path}/{rbp_name}/positive_data/train_{pretrain_RNA_model}_rnaembedding.h5"
    bad = is_bad_file(train_positive_embedding)
    print("train_positive_embedding bad:", bad)
    if bad:
        print(f"{train_positive_embedding} missing or empty, generating h5 files.")
        train_positive_annotation_file = f"{file_path}/{rbp_name}/positive_data/train_annotation.tsv"
        if not os.path.exists(train_positive_annotation_file):
            process_train_rnafold_data(file_path, rbp_name)
            load_tsv_file(device, train_positive_annotation_file, max_length=200, embedder_name=pretrain_RNA_model, save_path=train_positive_embedding, rna_embedding_path=rna_embedding_path)
        else:
            load_tsv_file(device, train_positive_annotation_file, max_length=200, embedder_name=pretrain_RNA_model, save_path=train_positive_embedding, rna_embedding_path=rna_embedding_path)

    train_negative_embedding = f"{file_path}/{rbp_name}/negative_data/train_{pretrain_RNA_model}_rnaembedding.h5"
    bad = is_bad_file(train_negative_embedding)
    print("train_positive_embedding bad:", bad)
    if bad:
        print(f"{train_negative_embedding} missing or empty, generating h5 files.")
        train_negative_annotation_file = f"{file_path}/{rbp_name}/negative_data/train_annotation.tsv"
        if not os.path.exists(train_negative_annotation_file):
            process_train_rnafold_data(file_path, rbp_name)
            load_tsv_file(device, train_negative_annotation_file, max_length=200, embedder_name=pretrain_RNA_model, save_path=train_negative_embedding, rna_embedding_path=rna_embedding_path)
        else:
            load_tsv_file(device, train_negative_annotation_file, max_length=200, embedder_name=pretrain_RNA_model, save_path=train_negative_embedding, rna_embedding_path=rna_embedding_path)
    
    print(f"loading the train data from {train_positive_embedding}")
    train_p_rna_names, train_p_embeddings_stru = load_embedding_h5file(train_positive_embedding)

    print(f"loading the train data from {train_negative_embedding}")
    train_n_rna_names, train_n_embeddings_stru = load_embedding_h5file(train_negative_embedding)

    print("Complete data loading")

    train_data = np.concatenate((train_p_embeddings_stru, train_n_embeddings_stru), axis=0)
    
    train_labels = np.concatenate((np.ones(len(train_p_embeddings_stru)), np.zeros(len(train_n_embeddings_stru))), axis=0)

    smoothed_label = smooth_onehot_label(torch.tensor(train_labels) , smooth_rate)
    assert smoothed_label.shape == train_labels.shape

    X_train, X_test, y_train, y_test, y_train_smooth, y_test_smooth = train_test_split(
    train_data, train_labels, smoothed_label, test_size=0.2, random_state=42
    )

    train_loader = create_dataloader(X_train, y_train, y_train_smooth, batch_size)
    test_loader = create_dataloader(X_test, y_test, y_test_smooth, batch_size)

    return train_loader, test_loader

def validation_dataset_RNAembedding(device, file_path, rbp_name, batch_size, smooth_rate, pretrain_RNA_model, rna_embedding_path):
    """Generate and load validation dataset with RNA embeddings."""
    
    test_positive_embedding = f"{file_path}/{rbp_name}/positive_data/test_{pretrain_RNA_model}_rnaembedding.h5"
    bad = is_bad_file(test_positive_embedding)
    print("test_positive_embedding bad:", bad)
    if bad:
        print(f"{test_positive_embedding} missing or empty, generating h5 files.")
        test_positive_annotation_file = f"{file_path}/{rbp_name}/positive_data/test_annotation.tsv"
        if not os.path.exists(test_positive_annotation_file):
            process_validation_rnafold_data(file_path, rbp_name)
            load_tsv_file(device, test_positive_annotation_file, max_length=200, embedder_name=pretrain_RNA_model, save_path=test_positive_embedding, rna_embedding_path=rna_embedding_path)
        else:
            load_tsv_file(device, test_positive_annotation_file, max_length=200, embedder_name=pretrain_RNA_model, save_path=test_positive_embedding, rna_embedding_path=rna_embedding_path)

    test_negative_embedding = f"{file_path}/{rbp_name}/negative_data/test_{pretrain_RNA_model}_rnaembedding.h5"
    bad = is_bad_file(test_negative_embedding)
    print("train_positive_embedding bad:", bad)
    if bad:
        print(f"{test_negative_embedding} missing or empty, generating h5 files.")
        test_negative_annotation_file = f"{file_path}/{rbp_name}/negative_data/test_annotation.tsv"
        if not os.path.exists(test_negative_annotation_file):
            process_validation_rnafold_data(file_path, rbp_name)
            load_tsv_file(device, test_negative_annotation_file, max_length=200, embedder_name=pretrain_RNA_model, save_path=test_negative_embedding, rna_embedding_path=rna_embedding_path)
        else:
            load_tsv_file(device, test_negative_annotation_file, max_length=200, embedder_name=pretrain_RNA_model, save_path=test_negative_embedding, rna_embedding_path=rna_embedding_path)


    print(f"loading the train data from {test_positive_embedding}")
    test_p_rna_names, test_p_embeddings_stru = load_embedding_h5file(test_positive_embedding)

    print(f"loading the train data from {test_negative_embedding}")
    # print(test_p_embeddings_stru.shape)
    test_n_rna_names, test_n_embeddings_stru = load_embedding_h5file(test_negative_embedding)

    print("Complete data loading")

    test_data = np.concatenate((test_p_embeddings_stru, test_n_embeddings_stru), axis=0)
    
    # print(len(train_data))
    test_labels = np.concatenate((np.ones(len(test_p_embeddings_stru)), np.zeros(len(test_n_embeddings_stru))), axis=0)

    smoothed_label = smooth_onehot_label(torch.tensor(test_labels) , smooth_rate)
    assert smoothed_label.shape == test_labels.shape

    data_loader = create_dataloader(test_data, test_labels, smoothed_label, batch_size)

    return data_loader

def save_validations(out_dir, filename, dataname, predictions, label, met):
    """Save validation results and metrics."""
    evals_dir = make_directory(out_dir, f"out/evals")
    metrics_path = os.path.join(evals_dir, filename+'.metrics')
    probs_path = os.path.join(evals_dir, filename+'.probs')
    with open(metrics_path,"w") as f:
        print("{:s}\t{:.3f}\t{:.3f}\t{:.3f}\t{:d}\t{:d}\t{:d}\t{:d}".format(
            dataname,
            met.acc,
            met.auc,
            met.prc,
            met.tp,
            met.tn,
            met.fp,
            met.fn,
        ), file=f)
    with open(probs_path,"w") as f:
        for i in range(len(predictions)):
            print("{:.3f}\t{}".format(predictions[i], label[i]), file=f)
    print("Evaluation file:", metrics_path)
    print("Prediction file:", probs_path)

def save_infers(out_dir, filename, rna_names_all, y_all, p_all):
    """Save inference results."""
    evals_dir = make_directory(out_dir, "out/infer")
    probs_path = os.path.join(evals_dir, filename + '.inference')
    
    # Open the file for writing
    with open(probs_path, "w") as f:
        for i in range(len(y_all)):
            rna_name = rna_names_all[i]
            
            # Check if it's a byte string, and decode if so
            if isinstance(rna_name, bytes):
                rna_name = rna_name.decode('utf-8')
            
            y_line = "{:f}".format(y_all[i])
            p_line = "{:f}".format(p_all[i])
            f.write(f"{rna_name}\t{y_line}\t{p_line}\n")

    print(f"Prediction file saved to: {probs_path}")