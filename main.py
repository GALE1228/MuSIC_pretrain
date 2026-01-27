from utils import *
import numpy as np
import argparse
import torch
import torch.nn as nn
from model_code.model import MuSIC
from train_code.train_loop import *
from model_code.GradualWarmupScheduler import GradualWarmupScheduler
import logging
import os
import sys
from Bio import SeqIO

# Load the RBP embedding from .npz file
def load_RBP_embedding(embedding_dir, rbp_name):

    embedding_path = os.path.join(embedding_dir, f"{rbp_name}_protein_features.npz")
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found for {rbp_name} at {embedding_path}")
    
    data = np.load(embedding_path)

    emb_key = list(data.keys())[0]
    embedding = data[emb_key]
    
    return torch.tensor(embedding, dtype=torch.float32)

def create_rna_name_to_sequence(file_path):
    """
    Create a dictionary mapping RNA names to their corresponding sequences from a FASTA file.
    """
    rna_name_to_sequence = {}
    with open(file_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            rna_name_to_sequence[record.id] = str(record.seq)
    return rna_name_to_sequence

def main(args):
    """
    Main function to handle training, validation, inference, and high attention region computation.
    It processes command-line arguments and performs tasks like model training, validation, and inference.
    """

    # Extract input arguments
    pretrain_RNA_model = args.pretrain_RNA_model
    file_path = args.file_path
    out_dir = args.out_dir
    best_model_dir = args.best_model_dir
    rbp = args.rbp_name
    source_s = args.source_species.upper() if args.source_species else None
    target_s = args.target_species.upper() if args.target_species else None

    # Extract training hyperparameters
    batch_size = args.batch_size
    gpuid = args.gpuid
    nepochs = args.num_epochs
    weight_decay = args.weight_decay
    pos_weight = args.pos_weight
    early_stopping = args.early_stopping
    exp_name = args.exp_name
    learn_rate = args.learn_rate
    smooth_rate = args.smooth_rate
    RBP_embedding_path = args.RBP_embedding_path
    rna_embedding_path = args.rna_embedding_path

    # Set device (GPU or CPU)
    device = torch.device(f"cuda:{gpuid}" if torch.cuda.is_available() else "cpu")
    print("device gpu ID is", gpuid)

    # Define experiment identity
    if target_s is None:
        identity = f"{rbp}_{exp_name}_{source_s}_{pretrain_RNA_model}"
    else:
        identity = f"{rbp}_{exp_name}_{source_s}_{target_s}_{pretrain_RNA_model}"
    print("rbp_clip information is ", identity)

    # Define path to save the best model
    best_model_path = f"{best_model_dir}/out/model/{identity}_best.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # Generate RNA embeddings
    if args.gerenate_embeddingh5:
        if args.train_embedding_data_process:
            data_process = "train"
            gerenate_RNAembedding_h5(device, file_path , rbp , pretrain_RNA_model, data_process, rna_embedding_path)
        elif args.validate_embedding_data_process:
            data_process = "test"
            gerenate_RNAembedding_h5(device, file_path , rbp , pretrain_RNA_model, data_process, rna_embedding_path)
        elif args.infer_embedding_data_process:
            data_process = "infer"
            fasta_path = args.infer_fasta_path
            rbp = None
            gerenate_RNAembedding_h5(device, fasta_path , rbp , pretrain_RNA_model, data_process, rna_embedding_path)

    # Train the cross-species model with RNA and RBP embeddings
    if args.train:
        if os.path.exists(best_model_path):
            # Load existing model if available
            print(f"Found existing model at {best_model_path}, skip training.")
            model = MuSIC(dim=1024, n_heads=8).to(device)
            model.load_state_dict(torch.load(best_model_path))
            print("Model loaded.")
        else:
            # Load RBP embeddings for source and target species
            parts = rbp.split("_")
            RBP = parts[0]
            source_RBP = source_s + "_" + RBP
            target_RBP = target_s + "_" + RBP
            print("source RBP is ", source_RBP, "target RBP is ", target_RBP)

            try:
                source_RBP_emb = load_RBP_embedding(RBP_embedding_path, source_RBP)
                target_RBP_emb = load_RBP_embedding(RBP_embedding_path, target_RBP)
            except Exception as e:
                print(f"[ERROR] loading RBP embedding ERROR: {e}")
                sys.exit(1)

            # Prepare data for training
            train_loader, test_loader = train_dataset_RNAembedding(device, file_path , rbp , batch_size ,smooth_rate, pretrain_RNA_model, rna_embedding_path)

            # Define the model and training setup
            model = MuSIC(dim=source_RBP_emb.shape[1], n_heads=8).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=float(nepochs), after_scheduler=None)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

            best_auc = 0
            best_acc = 0
            best_epoch = 0

            # Set up logging for training and validation metrics
            log_dir = f"{out_dir}/out/logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{identity}.txt")
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                                handlers=[logging.FileHandler(log_file , mode='w'), logging.StreamHandler()])

            COLOR_GREEN = '\033[92m'
            COLOR_RED = '\033[91m'
            COLOR_RESET = '\033[0m'

            # Training loop
            for epoch in range(1, nepochs + 1):
                t_met = train(model, device, train_loader, source_RBP_emb, target_RBP_emb, criterion, optimizer, batch_size, smooth_rate)
                v_met, _, _ = validate(model, device, test_loader, source_RBP_emb, target_RBP_emb, criterion , smooth_rate)
                scheduler.step()
                lr = scheduler.get_lr()[0]
                
                # Check if validation results are the best
                color_best = 'green'
                if best_auc < v_met.auc:
                    best_auc = v_met.auc
                    best_acc = v_met.acc
                    best_epoch = epoch
                    color_best = 'red'
                    torch.save(model.state_dict(), best_model_path)

                # Early stopping condition
                if epoch - best_epoch > early_stopping:
                    print(f"Early stop at {epoch}, {exp_name}")
                    break

                color = COLOR_RED if color_best == 'red' else COLOR_GREEN
                t_avg_loss = sum(t_met.other) / len(t_met.other)
                logging.info(f'{color}Train Epoch: {epoch} avg.loss: {t_avg_loss:.4f} '
                             f'Acc: {t_met.acc:.2f}, AUC: {t_met.auc:.4f}, lr: {lr:.6f}{COLOR_RESET}')
                v_avg_loss = sum(v_met.other) / len(v_met.other)
                logging.info(f'{color}Test Epoch: {epoch} avg.loss: {v_avg_loss:.4f} '
                             f'Acc: {v_met.acc:.2f}, AUC: {v_met.auc:.4f} '
                             f'({best_auc:.4f} best){COLOR_RESET}')
                
            logging.info("%s auc: %.4f acc: %.4f", "TEST", best_auc, best_acc)

            # Load the best model
            filename = best_model_path.format("best")
            model.load_state_dict(torch.load(filename))

    # Validate the cross-species model with RNA and RBP embeddings
    if args.validate:
        
        # get rbp and species information
        parts = rbp.split("_")
        RBP = parts[0]
        source_RBP = source_s + "_" + RBP
        target_RBP = target_s + "_" + RBP
        print("source RBP is ", source_RBP, "target RBP is ", target_RBP)
        
        # load RBP embeddings and RNA embeddings
        source_RBP_emb, target_RBP_emb = load_RBP_embedding(RBP_embedding_path, source_RBP), load_RBP_embedding(RBP_embedding_path, target_RBP) # (L_p, 1024)
        data_loader = validation_dataset_RNAembedding(device, file_path , rbp , batch_size ,smooth_rate, pretrain_RNA_model, rna_embedding_path)
        print("Test  set:", len(data_loader.dataset))

        best_model = MuSIC(dim=1024, n_heads=8).to(device)
        best_model = load_model(best_model, best_model_path, device)
        print("load best model path is ", best_model_path)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        met, y_all, p_all = validate(best_model, device, data_loader, source_RBP_emb, target_RBP_emb, criterion , smooth_rate)

        p_name = identity

        print("> eval {} auc: {:.4f} acc: {:.4f}".format(p_name, met.auc, met.acc))

        save_validations(out_dir, identity, p_name, p_all, y_all, met)

    # Inference on RNA sequences
    if args.infer:

        # Set the file path for the input RNA sequence
        file_path = args.infer_fasta_path
        identity = f"{identity}_{os.path.basename(file_path).replace('.fa', '')}"
        evals_dir = make_directory(out_dir, "out/infer")
        probs_path = os.path.join(evals_dir, identity + '.inference')
        print(probs_path)

        # Check if the inference results already exist, skip if they do
        if os.path.exists(probs_path):
            print(f"Probs file already exists at {probs_path}, skipping inference.")
            return

        # Prepare the file directory and embedding paths
        file_dir = os.path.dirname(file_path)
        file_prefix = os.path.splitext(os.path.basename(file_path))[0]
        embedding_dir = f"{file_dir}/{file_prefix}_{pretrain_RNA_model}_rnaembedding"
        print("fasta file embedding dir is", embedding_dir)
        os.makedirs(embedding_dir, exist_ok=True)
        embedding_annotation_file = f"{file_dir}/{file_prefix}_annotation.tsv"
        if not os.path.exists(embedding_annotation_file):
            process_rnafold_infer_data(file_path)
            load_tsv_file_infer(device, tsv_file=embedding_annotation_file, max_length=200,
                    embedder_name=pretrain_RNA_model, save_dir=embedding_dir, rna_embedding_path=rna_embedding_path)
        else:
            load_tsv_file_infer(device, tsv_file=embedding_annotation_file, max_length=200,
                    embedder_name=pretrain_RNA_model, save_dir=embedding_dir, rna_embedding_path=rna_embedding_path)

        # Load the pre-trained model
        best_model = MuSIC(dim=1024, n_heads=8).to(device)
        best_model = load_model(best_model, best_model_path, device)
        print("load best model path is ", best_model_path)

        # Get RBP and species information from the input
        parts = rbp.split("_")
        RBP = parts[0]
        source_RBP = source_s + "_" + RBP
        target_RBP = target_s + "_" + RBP
        print("source RBP is ", source_RBP, "target RBP is ", target_RBP)

        try:
            source_RBP_emb = load_RBP_embedding(RBP_embedding_path, source_RBP)
            target_RBP_emb = load_RBP_embedding(RBP_embedding_path, target_RBP)
        except Exception as e:
            print(f"[ERROR] loading RBP embedding ERROR: {e}")
            sys.exit(1)

        # Initialize lists to store final results
        p_all_final, y_all_final, rna_names_out_final = [], [], []

        # Perform inference on RNA sequences in batches
        for rna_names, combined_features in load_inferh5(embedding_annotation_file, embedding_dir, pretrain_RNA_model):
            test_labels = np.ones(len(combined_features))
            data_loader, rna_names_all = create_infer_dataloader(combined_features, test_labels, rna_names, batch_size)
            p_all, y_all, rna_names_out = inference(args, best_model, device, data_loader, rna_names_all, smooth_rate, source_RBP_emb, target_RBP_emb)

            p_all_final.append(p_all)
            y_all_final.append(y_all)
            rna_names_out_final.extend(rna_names_out)

        p_all_final = np.concatenate(p_all_final)
        y_all_final = np.concatenate(y_all_final)

        # Save the inference results to the output directory
        save_infers(out_dir, identity, rna_names_out_final, y_all_final, p_all_final)
        print("Inference results saved successfully.")
    
    # Compute high attention regions
    if args.har:
        # Set the file path for the RNA sequence input file
        file_path = args.infer_fasta_path
        print("Inference fasta file path :", file_path)
        
        # Modify the identity for saving the output file
        identity = identity + "_" + os.path.basename(file_path).replace(".fa", "")
        har_dir = make_directory(out_dir, "out/har")
        har_path = os.path.join(har_dir, identity + '.txt')
        if os.path.exists(har_path):
            print(f"Probs file already exists at {har_path}, skipping inference.")
            return

        # Prepare paths for the embedding directory and annotation file
        file_dir = os.path.dirname(file_path)
        file_prefix = os.path.splitext(os.path.basename(file_path))[0]
        embedding_dir = f"{file_dir}/{file_prefix}_{pretrain_RNA_model}_rnaembedding"
        print("fasta file embedding dir is", embedding_dir)
        embedding_annotation_file = f"{file_dir}/{file_prefix}_annotation.tsv"

        # Load the pre-trained best model
        best_model = MuSIC(dim=1024, n_heads=8).to(device)
        best_model = load_model(best_model, best_model_path, device)
        print("load best model path is ", best_model_path)

        # Get RBP and species information from the input
        parts = rbp.split("_")
        RBP = parts[0]
        source_RBP = source_s + "_" + RBP
        target_RBP = target_s + "_" + RBP
        print("source RBP is ", source_RBP, "target RBP is ", target_RBP)

        # Load RBP embeddings for source and target species
        try:
            source_RBP_emb = load_RBP_embedding(RBP_embedding_path, source_RBP)
            target_RBP_emb = load_RBP_embedding(RBP_embedding_path, target_RBP)
        except Exception as e:
            print(f"[ERROR] loading RBP embedding ERROR: {e}")
            sys.exit(1)

        # Process the data in chunks, compute high attention regions for each batch
        hars_final = []
        for rna_names, combined_features in load_inferh5(embedding_annotation_file, embedding_dir, pretrain_RNA_model):
            test_labels = np.ones(len(combined_features))
            data_loader, rna_names_all = create_infer_dataloader(combined_features, test_labels, rna_names, batch_size)
            hars = compute_high_attention_region(args, best_model, device, data_loader, rna_names_all, target_RBP_emb)
            hars_final.append(hars)

        hars_final = np.concatenate(hars_final)
        with open(har_path, 'w') as f:
            f.writelines(hars_final)

        print(f"High attention regions saved to {har_path}")

if __name__ == '__main__':
    """
    Entry point for the script, parses command-line arguments and calls the main function.
    """
    parser = argparse.ArgumentParser(description="Process RNA and RBP embeddings for cross-species model training.")
    
    # File paths and model configurations
    parser.add_argument('--file_path', type=str, help="Path to the dataset folder.")
    parser.add_argument('--infer_fasta_path', type=str, help="Path to the fasta file for inference.")
    parser.add_argument('--rbp_name', type=str, help="Name of the RNA binding protein (RBP).")
    parser.add_argument('--source_species', type=str, default="HUMAN", help="Source species name.")
    parser.add_argument('--target_species', type=str, help="Target species name (optional).")
    parser.add_argument('--RBP_embedding_path', type=str, default="data/protein_fasta_embedding/embedding/protT5", help="Path to the RBP embedding folder.")
    parser.add_argument('--pretrain_RNA_model', type=str, default="RiNALMo", help="Pre-trained RNA model to use.")
    parser.add_argument('--rna_embedding_path', type=str, default="./pretrained_model/RNA/RiNALMo/weights/rinalmo_giga_pretrained.pt", help="Pre-trained RNA model to use.")
    parser.add_argument('--out_dir', type=str, default="music", help="Directory to save the results.")
    parser.add_argument('--best_model_dir', type=str, default="music", help="Directory to save the best model.")

    # Training, validation, and inference options
    parser.add_argument('--train', action='store_true', help='Flag to train the model for cross-species tasks.')
    parser.add_argument('--validate', action='store_true', help='Flag to validate the model for cross-species tasks.')
    parser.add_argument('--infer', action='store_true', help='Run inference mode on given RNA sequences.')
    parser.add_argument('--har', action='store_true', help='Compute the highest attention region.')

    # Data processing options
    parser.add_argument('--gerenate_embeddingh5', action='store_true', help='Generate RNA embedding files.')
    parser.add_argument('--train_embedding_data_process', action='store_true', help='Process training dataset for embeddings.')
    parser.add_argument('--validate_embedding_data_process', action='store_true', help='Process validation dataset for embeddings.')
    parser.add_argument('--infer_embedding_data_process', action='store_true', help='Process inference dataset for embeddings.')

    # Hyperparameters and configuration
    parser.add_argument('--gpuid', type=int, default=1, help="GPU ID to use for training.")
    parser.add_argument('--smooth_rate', type=float, default=0.8, help="Label smoothing rate.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs for training.")
    parser.add_argument('--weight_decay', type=float, default=1e-6, help="Weight decay factor.")
    parser.add_argument('--pos_weight', type=float, default=1, help="Class weight for the positive class.")
    parser.add_argument('--learn_rate', type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument('--early_stopping', type=int, default=20, help="Early stopping patience.")
    parser.add_argument('--exp_name', type=str, default="music", help="Experiment name for logging and results.")
    
    args = parser.parse_args()
    main(args)
