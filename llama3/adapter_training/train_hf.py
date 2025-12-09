#!/usr/bin/env python
'''Training script for MoT adapter using HuggingFace pre-encoded dataset.'''

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import time
from typing import Optional
from dataclasses import dataclass
from transformers import AutoTokenizer

# Import model components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

# Fairscale imports
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)

@dataclass
class TrainingArgs:
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_steps: int = 100
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 10
    max_seq_len: int = 1024
    text_max_len: int = 32
    checkpoint_dir: str = './checkpoints'

class HFMusicCapsDataset(Dataset):
    '''Dataset using pre-encoded tokens from HuggingFace.'''
    
    def __init__(self, split='train', max_seq_len=1024, text_max_len=32):
        print(f'📦 Loading pre-encoded dataset from HuggingFace...')
        self.dataset = load_dataset('LeeHarrold/musiccaps-mot-tokens', split=split)
        print(f'   ✅ Loaded {len(self.dataset)} samples')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_seq_len = max_seq_len
        self.text_max_len = text_max_len
        self.audio_offset = 128256  # Shift audio tokens to expanded vocabulary
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Text prompt
        prompt = f'Generate music: {item["caption"]}'
        text_tokens = self.tokenizer(
            prompt,
            max_length=self.text_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).input_ids.squeeze()
        
        # Audio tokens (already encoded)
        audio_codes = torch.tensor(np.array(item['audio_codes']), dtype=torch.long)
        
        # Ensure shape is [4, seq_len]
        if len(audio_codes.shape) != 2 or audio_codes.shape[0] != 4:
            # Skip malformed samples
            return self.__getitem__((idx + 1) % len(self.dataset))
        
        # Interleave codebooks: [c0_t0, c1_t0, c2_t0, c3_t0, c0_t1, ...]
        audio_interleaved = audio_codes.T.reshape(-1)  # [seq_len, 4] -> [seq_len*4]
        
        # Shift to audio vocabulary range
        audio_interleaved = audio_interleaved + self.audio_offset
        
        # Truncate if needed
        max_audio_len = self.max_seq_len - self.text_max_len
        if len(audio_interleaved) > max_audio_len:
            audio_interleaved = audio_interleaved[:max_audio_len]
            
        # Combine text and audio
        input_ids = torch.cat([text_tokens, audio_interleaved])
        
        # Create labels (shift by 1 for next-token prediction)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.tokenizer.eos_token_id
        
        # Mask text tokens in loss (-100 = ignore)
        labels[:self.text_max_len] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }

def setup_model(llama_ckpt_dir, device='cuda'):
    '''Initialize MoT model.'''
    
    # Load model args
    with open(os.path.join(llama_ckpt_dir, 'params.json'), 'r') as f:
        params = json.load(f)
    
    # Create model args
    model_args = ModelArgs(
        dim=params['dim'],
        n_layers=params['n_layers'],
        n_heads=params['n_heads'],
        n_kv_heads=params.get('n_kv_heads', params['n_heads']),
        vocab_size=130304,  # Expanded vocabulary
        ffn_dim_multiplier=params.get('ffn_dim_multiplier', 1.0),
        multiple_of=params['multiple_of'],
        norm_eps=params['norm_eps'],
        rope_theta=params.get('rope_theta', 500000.0),
        max_batch_size=32,
        max_seq_len=1024,
    )
    
    # Build model
    model = Transformer(model_args)
    
    # Load checkpoint
    checkpoint = torch.load(
        os.path.join(llama_ckpt_dir, 'consolidated.00.pth'),
        map_location='cpu'
    )
    model.load_state_dict(checkpoint, strict=False)
    
    # Freeze base model, only train adapters
    for name, param in model.named_parameters():
        if 'adapter' not in name.lower():
            param.requires_grad = False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'   Trainable parameters: {trainable_params/1e6:.1f}M / {total_params/1e6:.1f}M')
    
    return model.to(device)

def train():
    '''Main training function.'''
    
    args = TrainingArgs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'🚀 Starting MoT Training on {device}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name()}')
        print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    
    # Initialize model parallel
    # initialize_model_parallel(1)  # Skip for single GPU
    
    # Load datasets
    print('\n📦 Loading datasets...')
    train_dataset = HFMusicCapsDataset(split='train')
    val_dataset = HFMusicCapsDataset(split='test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    print('\n🤖 Loading MoT model...')
    llama_ckpt_dir = Path(__file__).parent.parent / 'Llama-3.2-1B'
    model = setup_model(str(llama_ckpt_dir), device)
    
    # Setup optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Training loop
    print(f'\n🚀 Starting training for {args.max_steps} steps...')
    global_step = 0
    model.train()
    
    pbar = tqdm(total=args.max_steps, desc='Training')
    
    for epoch in range(100):  # Max epochs
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= args.max_steps:
                break
                
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, 0)  # start_pos=0
            logits = outputs
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )
            
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.update(1)
                
                # Logging
                if global_step % args.logging_steps == 0:
                    print(f'\n   Step {global_step}: Loss = {loss.item() * args.gradient_accumulation_steps:.4f}')
                
                # Checkpointing
                if global_step % args.save_steps == 0:
                    save_path = Path(args.checkpoint_dir) / f'checkpoint_step_{global_step}.pt'
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                    }, save_path)
                    print(f'   💾 Saved checkpoint to {save_path}')
                
                if global_step >= args.max_steps:
                    break
        
        if global_step >= args.max_steps:
            break
    
    pbar.close()
    print(f'\n✅ Training completed! Trained for {global_step} steps.')
    
    # Save final model
    final_path = Path(args.checkpoint_dir) / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f'   💾 Final model saved to {final_path}')

if __name__ == '__main__':
    train()
