import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler
import torch.nn.functional as F
import numpy as np
import sys

from embedding_dataset import EmbeddingDataset

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

def verify_cuda():
    if not torch.cuda.is_available(): sys.exit(1)
    return torch.device("cuda")

class MemmapDataset(Dataset):
    def __init__(self, emb_path, lbl_path, num_samples, dim):
        self.emb_path = emb_path
        self.lbl_path = lbl_path
        self.num_samples = num_samples
        self.dim = dim

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        if not hasattr(self, 'emb_fp'):
            self.emb_fp = np.memmap(self.emb_path, dtype='float16', mode='r', shape=(self.num_samples, self.dim))
            self.lbl_fp = np.memmap(self.lbl_path, dtype='int64', mode='r', shape=(self.num_samples,))
        emb = torch.from_numpy(np.copy(self.emb_fp[idx])).to(torch.float32)
        lbl = torch.tensor(self.lbl_fp[idx], dtype=torch.long)
        return emb, lbl

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.5, label_smoothing=0.1): 
        super().__init__()
        self.alpha, self.gamma, self.ls = alpha, gamma, label_smoothing
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

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
        self.n_dim, self.f_dim, self.s_dim = nutime_dim, 2787, 45
        
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

if __name__ == "__main__":
    device = verify_cuda()
    
    # -------------------------------------------------------------
    # TiRex / A
    # -------------------------------------------------------------
    TOTAL_SAMPLES = 2573208
    TOTAL_DIM = 3088      
    NUTIME_DIM = 256
    
    train_dataset = MemmapDataset('nutime_train.dat', 'nutime_train.lbl', TOTAL_SAMPLES, TOTAL_DIM)
    val_dataset = EmbeddingDataset('nutime_val2020.pt')
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    
    model = NuTimeMultiBranchHead(NUTIME_DIM, TOTAL_DIM).to(device)
    cw = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.5, 1.5, 1.8, 1.8], device=device)
    criterion = FocalLoss(alpha=cw, gamma=2.5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
    
    epochs = 40
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')
    
    best_val = 0.0
    alpha = 0.2 
    
    print("\n--- BEGIN NUTIME TRAINING ---")
    for epoch in range(epochs):
        model.train()
        total_loss, correct, samples = 0, 0, 0
        
        for i, (emb, lbl) in enumerate(train_loader):
            emb, lbl = emb.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True) 
            
            lam = max(torch.distributions.beta.Beta(alpha, alpha).sample().item(), 1 - torch.distributions.beta.Beta(alpha, alpha).sample().item()) 
            idx = torch.randperm(emb.size(0)).to(device)
            m_emb = lam * emb + (1 - lam) * emb[idx]
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(m_emb)
                loss = lam * criterion(logits, lbl) + (1 - lam) * criterion(logits, lbl[idx])
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            
            total_loss += loss.item()
            if i % 100 == 0: print(f"E{epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f}")
                
        scheduler.step()
        
        model.eval()
        v_corr, v_samp = 0, 0
        with torch.no_grad():
            for emb, lbl in val_loader:
                emb, lbl = emb.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16): logits = model(emb)
                v_samp += lbl.size(0)
                v_corr += (logits.argmax(1) == lbl).sum().item()
                
        v_acc = 100 * v_corr / v_samp
        print(f"[E{epoch+1}] Val Acc: {v_acc:.2f}%")
        
        if v_acc > best_val:
            best_val = v_acc
            torch.save(model.state_dict(), 'nutime_best.pth')
            print(">>> Saved New Best!")
