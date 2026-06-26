import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    """
    Loads pre-extracted float16 feature embeddings from disk directly into RAM.
    Converts them back to float32 on-the-fly for stable classifier training.
    """
    def __init__(self, pt_file):
        print(f"Loading {pt_file} into system RAM...")
        data = torch.load(pt_file)
        self.embeddings = data['embeddings']
        self.labels = data['labels']
        self.concat_dim = data['concat_dim']
        print(f"Loaded {len(self.labels)} samples. Dimension: {self.concat_dim}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = self.embeddings[idx].to(torch.float32)
        lbl = self.labels[idx]
        return emb, lbl
