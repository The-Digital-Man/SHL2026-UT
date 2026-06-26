import torch
import numpy as np
import gc

files = ['nutime_train_clean.pt', 'nutime_train_rail.pt', 'nutime_train_loc.pt', 'nutime_train_rot.pt']

print("Calculating NuTime dynamic size...")
total_samples = 0
dim = 0
nutime_dim = 0

for pt_file in files:
    data = torch.load(pt_file)
    total_samples += data['labels'].size(0)
    dim = data['concat_dim']
    nutime_dim = data['nutime_dim']
    del data; gc.collect()

print(f"TOTAL_SAMPLES = {total_samples}")
print(f"TOTAL_DIM = {dim}")
print(f"NUTIME_DIM = {nutime_dim}")
print("==========================================\n")

emb_fp = np.memmap('nutime_train.dat', dtype='float16', mode='w+', shape=(total_samples, dim))
lbl_fp = np.memmap('nutime_train.lbl', dtype='int64', mode='w+', shape=(total_samples,))

current_idx = 0
for pt_file in files:
    print(f"Streaming {pt_file}...")
    data = torch.load(pt_file)
    emb, lbl = data['embeddings'].numpy(), data['labels'].numpy()
    n = emb.shape[0]
    
    emb_fp[current_idx : current_idx + n] = emb
    lbl_fp[current_idx : current_idx + n] = lbl
    current_idx += n
    emb_fp.flush(); lbl_fp.flush()
    del data, emb, lbl; gc.collect()

print("NuTime Memmap Complete!")
