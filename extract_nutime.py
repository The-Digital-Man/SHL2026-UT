import os
import sys
import gc
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import LMDBDataset

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# 1. Inject NuTime repository into the Python path
sys.path.insert(0, os.path.abspath('NuTime'))

def verify_cuda():
    if not torch.cuda.is_available(): sys.exit(1)
    return torch.device("cuda")

def load_academic_checkpoint(ckpt_path, device):
    print("Parsing NuTime academic configuration...")
    
    with open('NuTime/configs/default_ssl.json', 'r') as f:
        cfg_dict = json.load(f)
        
    # --- GEOMETRY & ARCHITECTURE LOCKS ---
    cfg_dict['num_channels'] = 128       # Transformer Latent Dim
    cfg_dict['num_bias'] = 9             # IMU Input Channels
    cfg_dict['window_emb_dim'] = 128     # Latent projection size
    cfg_dict['mb_emb_dim'] = 32          # Multi-bias expansion
    cfg_dict['model_series_size'] = 496  # Truncated
    cfg_dict['window_size'] = 16         # Checkpoint window size
    cfg_dict['model'] = 'wint'
    cfg_dict['pe_mode'] = 'learnable'    # Bypass buggy sincos math
    cfg_dict['transformer_heads'] = 8
    
    if cfg_dict.get('task') == 'cls':
        cfg_dict['num_classes'] = 8 
        
    config = argparse.Namespace(**cfg_dict)
    
    from encoders.build import get_encoder
    from models.build import get_model
    
    # --- MOCK DATASET FOR ENCODER INIT ---
    class MockDataset:
        samples = torch.zeros(2, 9, 496)
        targets = torch.zeros(2)
        
    print("Building NuTime WindowNormEncoder (backbone.0)...")
    encoder = get_encoder(config, MockDataset())
    
    # Handoff 128 channels to the Transformer
    config.num_channels = 128 
    
    print("Building NuTime Transformer (backbone.1)...")
    transformer = get_model(config)
    
    # Combine the split architecture
    model = torch.nn.Sequential(encoder, transformer)
    
    print(f"Loading SOTA weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Strip the SSL wrapper prefix
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('backbone.', '') if k.startswith('backbone.') else k
        clean_state_dict[new_key] = v
        
    # =========================================================
    # POSITIONAL EMBEDDING SLICE
    # (31 windows + 1 CLS token). Slice the tensor to match
    # =========================================================
    if '1.pos_embed' in clean_state_dict:
        model_shape = model[1].pos_embed.shape
        ckpt_shape = clean_state_dict['1.pos_embed'].shape
        if ckpt_shape != model_shape:
            print(f"Adapting pos_embed from {ckpt_shape[1]} tokens down to {model_shape[1]} tokens...")
            clean_state_dict['1.pos_embed'] = clean_state_dict['1.pos_embed'][:, :model_shape[1], :]
    # =========================================================
            
    model.load_state_dict(clean_state_dict, strict=False)
    return model.to(device).eval()

def apply_random_3d_rotation(frames):
    batch_size = frames.size(0)
    device = frames.device
    rotated = frames.clone()
    for i in range(batch_size):
        a, b, c = torch.rand(3) * 2 * torch.pi
        ca, sa, cb, sb, cg, sg = torch.cos(a), torch.sin(a), torch.cos(b), torch.sin(b), torch.cos(c), torch.sin(c)
        R = torch.tensor([
            [ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg],
            [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg],
            [-sb,   cb*sg,            cb*cg]
        ], device=device, dtype=frames.dtype)
        rotated[i, 0:3, :] = torch.matmul(R, frames[i, 0:3, :])
        rotated[i, 3:6, :] = torch.matmul(R, frames[i, 3:6, :])
        rotated[i, 6:9, :] = torch.matmul(R, frames[i, 6:9, :])
    return rotated

def process_through_nutime(frames, model, batch_size):
    device = frames.device
    
    # 1. FOUNDATION PASS
    frames_nutime = frames[:, :, :496]
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(frames_nutime)
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        nutime_embeddings = h.mean(dim=1) if h.dim() == 3 else h
        nutime_embeddings = nutime_embeddings.view(batch_size, -1).float()

    # 2. FREQUENCY & SPECTROGRAM (Using full 500 frames)
    fft_features = torch.log1p(torch.abs(torch.fft.rfft(frames, dim=-1))).view(batch_size, -1).float()
    accel_mag = torch.sqrt(torch.sum(frames[:, 0:3, :]**2, dim=1))
    stft_out = torch.stft(accel_mag, n_fft=64, hop_length=32, window=torch.hann_window(64, device=device), return_complex=True)
    stft_flattened = torch.abs(stft_out).reshape(batch_size, -1).float()

    # 3. STATISTICAL ISOLATION (Using full 500 frames)
    mean, var = frames.mean(dim=-1), frames.var(dim=-1)   
    z_scores = (frames - mean.unsqueeze(-1)) / torch.sqrt(var + 1e-6).unsqueeze(-1)
    skewness = torch.mean(z_scores ** 3, dim=-1) 
    kurtosis = torch.mean(z_scores ** 4, dim=-1) - 3.0 
    
    cross_corr = []
    for idx1, idx2 in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (6, 7), (7, 8), (6, 8)]:
        cross_corr.append(torch.mean(z_scores[:, idx1, :] * z_scores[:, idx2, :], dim=-1).unsqueeze(-1))
    cross_corr = torch.cat(cross_corr, dim=-1) 
    stat_features = torch.cat([mean, var, skewness, kurtosis, cross_corr], dim=-1).float()

    # 4. SOTA FUSION
    return torch.cat([nutime_embeddings, fft_features, stft_flattened, stat_features], dim=1).cpu().to(torch.float16)

def run_extraction_pass(loader, model, device, name, aug_type="none"):
    all_emb, all_lbl = [], []
    print(f"\n--- Extracting {name} ---")
    with torch.no_grad():
        for i, (frames, labels) in enumerate(loader):
            frames = frames.to(device, non_blocking=True)
            bs = frames.size(0)
            
            if aug_type == "rail":
                mask = (labels == 6) | (labels == 7)
                if mask.any():
                    frames[mask, 0:6, :] = 0.0
                    all_emb.append(process_through_nutime(frames, model, bs)[mask.cpu()])
                    all_lbl.append(labels[mask].cpu())
                continue
                
            if aug_type == "loc": frames = frames * (torch.rand(bs, 9, 1, device=device) > 0.25)
            if aug_type == "rot": frames = apply_random_3d_rotation(frames)
            
            all_emb.append(process_through_nutime(frames, model, bs))
            all_lbl.append(labels.cpu())
            
            if i % 100 == 0: print(f"[{name}] Batch {i}/{len(loader)}")
            
    return torch.cat(all_emb, dim=0), torch.cat(all_lbl, dim=0)

if __name__ == "__main__":
    device = verify_cuda()
    
    train_ds = LMDBDataset('shl2026_dataset.lmdb', mode='train', simulate_test_set=False)
    val2020_ds = LMDBDataset('shl2020_judge_test.lmdb', mode='test', simulate_test_set=True)
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
    val2020_loader = DataLoader(val2020_ds, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
    
    ckpt_file = 'NuTime/ckpt/checkpoint_bias9.pth'
    nutime_model = load_academic_checkpoint(ckpt_file, device)
    
    print("\nCalculating Final Geometry...")
    dummy = torch.zeros(1, 9, 500, device=device)
    with torch.no_grad(): 
        concat_dim = process_through_nutime(dummy, nutime_model, 1).size(-1)
        nutime_dim = concat_dim - 2787 - 528 - 45 
    print(f"Detected NuTime Latent Dim: {nutime_dim} | Total Fused Dim: {concat_dim}")
    
    passes = [("Clean Baseline", "none", "nutime_train_clean.pt"),
              ("Rail Drop", "rail", "nutime_train_rail.pt"),
              ("Loc Drop", "loc", "nutime_train_loc.pt"),
              ("3D Rotation", "rot", "nutime_train_rot.pt")]
              
    for name, aug, file in passes:
        emb, lbl = run_extraction_pass(train_loader, nutime_model, device, name, aug)
        torch.save({'embeddings': emb, 'labels': lbl, 'concat_dim': concat_dim, 'nutime_dim': nutime_dim}, file)
        del emb, lbl; gc.collect()
        
    emb_val, lbl_val = run_extraction_pass(val2020_loader, nutime_model, device, "2020 Judge Validation", "none")
    torch.save({'embeddings': emb_val, 'labels': lbl_val, 'concat_dim': concat_dim, 'nutime_dim': nutime_dim}, 'nutime_val2020.pt')
    
    print("\nNuTime SOTA Extraction Complete!")
