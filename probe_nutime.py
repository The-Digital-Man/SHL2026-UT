import torch
import sys
import os

sys.path.insert(0, os.path.abspath('NuTime'))

ckpt_path = 'NuTime/ckpt/checkpoint_bias9.pth'

if not os.path.exists(ckpt_path):
    print(f"CRITICAL: {ckpt_path} not found. Did the 'git lfs pull' succeed?")
    sys.exit(1)

print(f"Loading {ckpt_path} into RAM...")
checkpoint = torch.load(ckpt_path, map_location='cpu')

print("\n=== CHECKPOINT ROOT KEYS ===")
print(list(checkpoint.keys()))

for key in ['args', 'config', 'hyper_parameters', 'cfg']:
    if key in checkpoint:
        print(f"\n=== EMBEDDED {key.upper()} ===")
        print(checkpoint[key])

print("\n=== NETWORK TENSOR SIGNATURE (First 15 Layers) ===")
state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

if isinstance(state_dict, dict):
    for i, (k, v) in enumerate(state_dict.items()):
        print(f"{k}: {v.shape}")
        if i >= 14: 
            print("... (remaining layers truncated)")
            break
else:
    print("Warning: State dict is not a standard dictionary.")
