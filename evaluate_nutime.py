import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from embedding_dataset import EmbeddingDataset

torch.set_float32_matmul_precision('high')

class ResidualBlock(nn.Module):
    def __init__(self, channels, drop):
        super().__init__()
        self.b = nn.Sequential(nn.Linear(channels, channels), nn.LayerNorm(channels), nn.GELU(), nn.Dropout(drop),
                               nn.Linear(channels, channels), nn.LayerNorm(channels))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.b(x))

class NuTimeMultiBranchHead(nn.Module):
    def __init__(self, nutime_dim, total_dim, num_classes=8, drop=0.65):
        super().__init__()
        self.n_dim = nutime_dim
        self.f_dim = 2787
        self.s_dim = 45   
        
        self.n_norm = nn.BatchNorm1d(self.n_dim)
        self.n_proj = nn.Sequential(nn.Linear(self.n_dim, 128), nn.LayerNorm(128), nn.GELU(), ResidualBlock(128, drop))

        self.f_norm = nn.BatchNorm1d(self.f_dim)
        self.f_proj = nn.Sequential(nn.Linear(self.f_dim, 256), nn.LayerNorm(256), nn.GELU(), ResidualBlock(256, drop))

        self.s_norm = nn.BatchNorm1d(self.s_dim)
        self.s_proj = nn.Sequential(nn.Linear(self.s_dim, 128), nn.LayerNorm(128), nn.GELU(), ResidualBlock(128, drop))

        self.gate = nn.Sequential(nn.Linear(512, 3), nn.Softmax(dim=-1))
        self.trunk = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, num_classes))

    def forward(self, x):
        feat_n = self.n_proj(self.n_norm(x[:, :self.n_dim]))
        feat_f = self.f_proj(self.f_norm(x[:, self.n_dim : self.n_dim + self.f_dim]))
        feat_s = self.s_proj(self.s_norm(x[:, self.n_dim + self.f_dim:]))

        gates = self.gate(torch.cat([feat_n, feat_f, feat_s], dim=-1)) 
        g_n, g_f, g_s = feat_n * gates[:, 0:1], feat_f * gates[:, 1:2], feat_s * gates[:, 2:3]
        return self.trunk(torch.cat([g_n, g_f, g_s], dim=-1))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Locked dimensions from the Memmap builder
    TOTAL_DIM = 3088
    NUTIME_DIM = 256
    
    print("Loading 2020 Judge Embeddings into RAM...")
    val_dataset = EmbeddingDataset('nutime_val2020.pt')
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True)
    
    print("Initializing Architecture and Restoring SOTA Weights...")
    model = NuTimeMultiBranchHead(NUTIME_DIM, TOTAL_DIM).to(device)
    
    # Load the golden checkpoint
    model.load_state_dict(torch.load('nutime_best.pth', map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Executing Blind Inference...")
    with torch.no_grad():
        for emb, lbl in val_loader:
            emb = emb.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(emb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbl.numpy())
            
    # Calculate SHL Challenge Metrics
    classes = ['Still', 'Walk', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\n" + "="*50)
    print(f"SHL 2020 BLIND TEST RESULTS")
    print("="*50)
    print(f"OVERALL ACCURACY : {acc*100:.4f}%")
    print(f"MACRO F1 SCORE   : {f1*100:.4f}%")
    print("="*50 + "\n")
    
    print("--- Detailed Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    
    print("\nRendering Graphical Confusion Matrix...")
    cm_raw = confusion_matrix(all_labels, all_preds)
    cm_perc = confusion_matrix(all_labels, all_preds, normalize='true') * 100
    
    box_labels = np.asarray([f"{count}\n({perc:.1f}%)" 
                             for count, perc in zip(cm_raw.flatten(), cm_perc.flatten())]).reshape(8, 8)
    
    plt.figure(figsize=(12, 9))
    
    sns.heatmap(cm_raw, annot=box_labels, fmt='', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 11}, cbar_kws={'label': 'Number of Frames'})
    
    plt.title(f'NuTime + SOTA Fusion - 2020 Judge Validation\nAccuracy: {acc*100:.2f}% | F1: {f1:.4f}', fontsize=16, pad=15)
    plt.ylabel('True Target Class', fontsize=14, labelpad=10)
    plt.xlabel('Model Prediction', fontsize=14, labelpad=10)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    
    save_path = 'nutime_2020_confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f">>> Graphic saved successfully to: {save_path}")
