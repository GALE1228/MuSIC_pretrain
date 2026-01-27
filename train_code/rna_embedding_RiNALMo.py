import torch
import sys
from rinalmo.pretrained import get_pretrained_model

class RiNALMo_Embedder:
    def __init__(self, model_path="weights/rinalmo_giga_pretrained.pt", device=None):
        self.device = device or ("cuda:2" if torch.cuda.is_available() else "cpu")
        
        self.model, self.alphabet = get_pretrained_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def embed(self, seqs):
        
        tokens = torch.tensor(
            self.alphabet.batch_tokenize(seqs),
            dtype=torch.int64,
            device=self.device,
        )
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.model(tokens)
        
        return outputs["representation"]
