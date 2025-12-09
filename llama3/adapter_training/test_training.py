#!/usr/bin/env python
"""Quick test to see if training runs with pre-encoded tokens."""

import sys
import time

print("=" * 50)
print("STARTING TRAINING TEST")
print("=" * 50)
sys.stdout.flush()

# Import and run just the setup part
print("1. Importing modules...")
sys.stdout.flush()

import torch
print(f"   - PyTorch {torch.__version__}")
sys.stdout.flush()

# Check for pre-encoded tokens
import pathlib
pre_encoded_dir = pathlib.Path("pre_encoded_audio")
if pre_encoded_dir.exists():
    npy_files = list(pre_encoded_dir.glob("*.npy"))
    print(f"   - Found {len(npy_files)} pre-encoded files")
else:
    print("   - No pre-encoded directory found!")
sys.stdout.flush()

print("\n2. Loading dataset...")
sys.stdout.flush()

from train_adapter import MusicCapsDataset
dataset = MusicCapsDataset(max_samples=5, pre_encoded_dir=str(pre_encoded_dir))
print(f"   - use_pre_encoded: {dataset.use_pre_encoded}")
sys.stdout.flush()

print("\n3. Testing iteration...")
sys.stdout.flush()

# Try to get one sample
for i, sample in enumerate(dataset):
    if i == 0:
        if sample is None:
            print("   - Got None sample")
        else:
            print(f"   - Got sample with keys: {sample.keys()}")
            if 'audio_codes' in sample:
                print(f"   - Audio codes shape: {sample['audio_codes'].shape}")
    break

print("\n4. Testing collate_fn...")
sys.stdout.flush()

from train_adapter import collate_fn
batch = []
for i, sample in enumerate(dataset):
    if sample is not None:
        batch.append(sample)
    if len(batch) >= 1:
        break

if batch:
    print(f"   - Batch size: {len(batch)}")
    result = collate_fn(batch, device="cpu", use_pre_encoded=True)
    print(f"   - Collate result keys: {result.keys()}")
    print(f"   - input_ids shape: {result['input_ids'].shape}")
    print(f"   - input_ids range: [{result['input_ids'].min()}, {result['input_ids'].max()}]")
else:
    print("   - No valid samples to collate")

print("\n✅ Test completed successfully!")
print("=" * 50)