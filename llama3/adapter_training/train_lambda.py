#!/usr/bin/env python
"""
Lambda Labs training script for MoT adapter using pre-encoded MusicCaps tokens.
Optimized for cloud GPU training with the uploaded HuggingFace dataset.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, MusicgenForConditionalGeneration
from datasets import load_dataset
import numpy as np
from pathlib import Path
import wandb
from tqdm import tqdm
import sys
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from llama.model import MoT

class CloudMusicCapsDataset(torch.utils.data.Dataset):
    """Dataset for pre-encoded MusicCaps tokens from HuggingFace."""

    def __init__(self, split="train", max_seq_len=1024, text_max_len=32):
        """Initialize dataset from HuggingFace."""
        print(f"📦 Loading pre-encoded dataset from HuggingFace...")

        # Load from HuggingFace
        self.dataset = load_dataset("LeeHarrold/musiccaps-mot-tokens", split=split)
        print(f"   ✅ Loaded {len(self.dataset)} samples from split '{split}'")

        # Initialize tokenizer for text prompts
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_seq_len = max_seq_len
        self.text_max_len = text_max_len
        self.audio_offset = 128256  # Shift audio tokens to expanded vocabulary

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a single sample."""
        item = self.dataset[idx]

        # Text prompt
        prompt = f"Generate music: {item['caption']}"
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
            'labels': labels,
            'ytid': item['ytid']
        }

def setup_wandb(config):
    """Initialize Weights & Biases tracking."""
    wandb.init(
        project="mot-musiccaps",
        name=f"lambda-training-{config['run_name']}",
        config=config
    )

def train_on_lambda():
    """Main training function for Lambda Labs GPUs."""

    # Training configuration
    config = {
        'batch_size': 16,  # Larger batch size for V100/A100
        'learning_rate': 1e-4,
        'num_epochs': 3,
        'gradient_accumulation_steps': 4,
        'max_seq_len': 1024,
        'text_max_len': 32,
        'warmup_steps': 500,
        'eval_steps': 100,
        'save_steps': 500,
        'logging_steps': 10,
        'fp16': True,  # Use mixed precision on GPU
        'gradient_checkpointing': True,
        'run_name': 'mot-adapter-v1',
        'output_dir': './checkpoints',
        'resume_from_checkpoint': None,
    }

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Initialize W&B
    setup_wandb(config)

    # Load datasets
    print("\n📦 Loading datasets...")
    train_dataset = CloudMusicCapsDataset(split="train", max_seq_len=config['max_seq_len'])
    val_dataset = CloudMusicCapsDataset(split="test", max_seq_len=config['max_seq_len'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    print("\n🤖 Loading MoT model...")
    model = MoT(
        llama_checkpoint="meta-llama/Llama-3.2-1B",
        musicgen_checkpoint="facebook/musicgen-small",
        adapter_dim=512,
        adapter_type="feedforward",
        device=device
    )

    if config['gradient_checkpointing']:
        model.gradient_checkpointing_enable()

    # Freeze base models, only train adapter
    for param in model.llama.parameters():
        param.requires_grad = False
    for param in model.audio_model.parameters():
        param.requires_grad = False

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable parameters: {trainable_params/1e6:.1f}M / {total_params/1e6:.1f}M")

    model = model.to(device)

    # Setup optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=0.01
    )

    # Training loop
    print("\n🚀 Starting training...")
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\n📈 Epoch {epoch + 1}/{config['num_epochs']}")

        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc="Training")

        for batch_idx, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss / config['gradient_accumulation_steps']

            # Backward
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % config['logging_steps'] == 0:
                    wandb.log({
                        'train/loss': loss.item() * config['gradient_accumulation_steps'],
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                        'train/global_step': global_step,
                    })

                # Evaluation
                if global_step % config['eval_steps'] == 0:
                    model.eval()
                    val_loss = 0

                    with torch.no_grad():
                        for val_batch in tqdm(val_loader, desc="Validation", leave=False):
                            val_input = val_batch['input_ids'].to(device)
                            val_labels = val_batch['labels'].to(device)

                            val_outputs = model(val_input, labels=val_labels)
                            val_loss += val_outputs.loss.item()

                    avg_val_loss = val_loss / len(val_loader)

                    wandb.log({
                        'val/loss': avg_val_loss,
                        'val/global_step': global_step,
                    })

                    print(f"   Step {global_step}: Val Loss = {avg_val_loss:.4f}")

                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save_path = Path(config['output_dir']) / f"best_model_step_{global_step}.pt"
                        save_path.parent.mkdir(parents=True, exist_ok=True)

                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'global_step': global_step,
                            'val_loss': avg_val_loss,
                            'config': config,
                        }, save_path)

                        print(f"   💾 Saved best model to {save_path}")

                    model.train()

                # Regular checkpointing
                if global_step % config['save_steps'] == 0:
                    save_path = Path(config['output_dir']) / f"checkpoint_step_{global_step}.pt"
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                        'config': config,
                    }, save_path)

            train_pbar.set_postfix({'loss': loss.item() * config['gradient_accumulation_steps']})
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"   Epoch {epoch + 1} - Avg Train Loss: {avg_train_loss:.4f}")

    print("\n✅ Training completed!")
    print(f"   Best validation loss: {best_val_loss:.4f}")

    # Save final model
    final_path = Path(config['output_dir']) / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_path)
    print(f"   💾 Final model saved to {final_path}")

    wandb.finish()

if __name__ == "__main__":
    train_on_lambda()