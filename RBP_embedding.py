import os
import re
import torch
import numpy as np
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel

class ProteinEmbedder:

    def __init__(self, model_path, device=None):
        
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False, legacy=False)
        self.model = T5EncoderModel.from_pretrained(model_path, ignore_mismatched_sizes=True).to(self.device)
        self.model.eval()

    def embed_batch(self, protein_sequences, protein_ids):


        protein_sequences = [
            " ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in protein_sequences
        ]

        ids = self.tokenizer.batch_encode_plus(
            protein_sequences, add_special_tokens=True, padding="longest"
        )
        
        input_ids = torch.tensor(ids["input_ids"]).to(self.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self.device)


        with torch.no_grad():
            embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


        features_dict = {}
        for i in range(len(protein_sequences)):
            col = torch.sum(attention_mask[i])
            emb = embedding_repr[i, :col - 1]
            features_dict[protein_ids[i]] = emb

        return features_dict

    def embed_all(self, tsv_file, out_dir, batch_size=1):

        os.makedirs(out_dir, exist_ok=True)


        existing_ids = {
            fname.split("_protein_features_")[0].strip("'")
            for fname in os.listdir(out_dir) if fname.endswith(".npz")
        }

        df_data = pd.read_csv(tsv_file, sep='\t')

        df_data = df_data[~df_data["p_id"].astype(str).str.strip("'").isin(existing_ids)]
        if df_data.empty:
            print("All protein features have been generated.")
            return

        num_sequences = len(df_data)
        num_batches = (num_sequences + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_sequences)

            protein_sequences = df_data.iloc[start_idx:end_idx]["protein"].tolist()
            protein_ids = df_data.iloc[start_idx:end_idx]["p_id"].tolist()

            protein_name = protein_ids[0].strip("'")


            features_dict = self.embed_batch(protein_sequences, protein_ids)


            for key, value in features_dict.items():
                features_dict[key] = value.cpu().numpy()

            save_path = os.path.join(out_dir, f"{protein_name}_protein_features.npz")
            np.savez(save_path, **features_dict)

            print(f"{protein_ids} done, saved to {save_path}")

        print("All protein embeddings have been saved.")

    def embed_one(self, p_id, tsv_file, out_dir):

            os.makedirs(out_dir, exist_ok=True)

            df = pd.read_csv(tsv_file, sep='\t')

            entry = df[df["p_id"] == p_id]
            if entry.empty:
                raise ValueError(f"No entry found for p_id: {p_id}")

            protein_sequence = entry.iloc[0]["protein"]
            protein_id = entry.iloc[0]["p_id"]

            protein_name = protein_id.split("_")[1]
            print(f"Embedding single protein: {protein_name}")

            # 计算embedding
            features_dict = self.embed_batch([protein_sequence], [protein_id])
            for key, value in features_dict.items():
                features_dict[key] = value.cpu().numpy()

            save_path = os.path.join(out_dir, f"{protein_id}_protein_features.npz")
            np.savez(save_path, **features_dict)

            print(f"Saved {p_id} embedding to {save_path}")

if __name__ == "__main__":

    model_path = "pretrained_model/protein/prot_t5_xl_uniref50"
    tsv_file = "data/protein_fasta_embedding/rbp_sequences_test.tsv"
    out_dir = "data/protein_fasta_embedding/embedding/protT5/"

    embedder = ProteinEmbedder(model_path)

    ### all proteins embedding
    embedder.embed_all(tsv_file, out_dir, batch_size=1)

    ### single protein embedding
    # embedder.embed_one("HUMAN_TRA2A", tsv_file, out_dir)
    