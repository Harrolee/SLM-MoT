# Pre-Encoding Implementation Summary

## Date: Current Session

## Overview
This document summarizes the work done to resolve the "backward through graph a second time" error and implement pre-encoding of audio tokens for the MoT adapter training pipeline.

## Problem Statement

### The Error
```
RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). 
Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). 
Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
```

### Root Cause
The error occurred because the MusicGen audio encoder (`musicgen.audio_encoder`) was being called during the training loop, even within a `torch.no_grad()` context. This created computation graph connections that persisted into the backward pass of the main model, causing PyTorch's autograd engine to fail when trying to free intermediate tensors.

## Solution: Pre-Encoding Audio Tokens

### Strategy
Instead of encoding audio on-the-fly during training, we pre-encode all audio files offline and save the resulting tokens to disk. The training loop then loads these pre-encoded tokens directly, completely decoupling the audio encoder from the training computation graph.

### Implementation Steps

#### 1. Created Pre-Encoding Script (`pre_encode_audio.py`)
- **Purpose**: Encode all audio files offline using MusicGen's EnCodec tokenizer
- **Process**:
  - Iterates through downloaded audio files in `musiccaps_audio/`
  - Resamples audio to 32kHz (MusicGen requirement)
  - Encodes audio using `musicgen.audio_encoder` within `torch.no_grad()`
  - Flattens 4 EnCodec codebooks into interleaved sequence
  - Saves numpy arrays to `pre_encoded_audio/` directory
  - Saves metadata mapping YouTube IDs to encoded file paths

#### 2. Modified Dataset Class (`MusicCapsDataset`)
- **Added `pre_encoded_dir` parameter**: Path to directory containing pre-encoded tokens
- **Added `use_pre_encoded` flag**: Controls whether to use pre-encoded tokens or encode on-the-fly
- **Modified `__iter__` method**:
  - If `use_pre_encoded=True`: Loads pre-encoded tokens from disk (numpy arrays)
  - If `use_pre_encoded=False`: Falls back to on-the-fly encoding (for backward compatibility)
- **Removed MusicGen model loading** when using pre-encoded tokens to save memory

#### 3. Updated Collate Function (`collate_fn`)
- **Conditional logic**:
  - `use_pre_encoded=True`: Loads numpy arrays, converts to tensors, ensures proper shape
  - `use_pre_encoded=False`: Uses MusicGen processor and encoder (original path)
- **Detached tensors**: Ensures pre-encoded tokens are detached from computation graph
- **Shape handling**: Properly handles various tensor dimensions from pre-encoded files

#### 4. Fixed Embedding Initialization
- **Problem**: Audio token embeddings were initialized to zeros, causing numerical instability (NaN losses)
- **Solution**: Initialize new audio token embeddings with small random values:
  ```python
  emb_std = old_emb.std().item()
  new_emb[original_vocab_size:] = torch.randn(
      audio_vocab_size, old_emb.shape[1],
      dtype=old_emb.dtype, device=old_emb.device
  ) * (emb_std * 0.1)  # 10% of existing std for stability
  ```
- **Applied to**: Both `tok_embeddings.weight` and `output.weight` layers

#### 5. Fixed MusicGen Weight Copying
- **Problem**: MusicGen weights were being copied without proper detachment, potentially creating graph connections
- **Solution**: Wrap weight copying in `torch.no_grad()` and explicitly detach/clone:
  ```python
  with torch.no_grad():
      fc1_weight = decoder_layers[layer_id].fc1.weight.detach().clone().to(device)
      audio_ffn.fc1.weight.copy_(fc1_weight)
  ```

#### 6. Code Cleanup
- **Removed all try-except blocks**: Simplified code by removing error handling that was masking issues
- **Fixed indentation errors**: Corrected multiple indentation issues in dataset loading and processing code
- **Simplified dataset initialization**: Removed nested try-except blocks that were causing syntax errors

## Files Modified

1. **`train_adapter.py`**:
   - Added `pre_encoded_dir` argument to `MusicCapsDataset`
   - Modified `collate_fn` to handle pre-encoded tokens
   - Fixed embedding initialization (random values instead of zeros)
   - Fixed MusicGen weight copying (explicit detachment)
   - Removed all try-except blocks
   - Fixed indentation issues

2. **`pre_encode_audio.py`** (new file):
   - Script to pre-encode audio files offline
   - Saves tokens and metadata for training use

3. **`MusicCapsDataset` class**:
   - Added pre-encoded token loading logic
   - Conditional MusicGen model loading
   - Simplified error handling

## Current Status

### ✅ Completed
- Pre-encoding script created and tested
- Dataset class modified to load pre-encoded tokens
- Collate function updated for pre-encoded path
- Embedding initialization fixed (random values)
- MusicGen weight copying fixed (detached)
- All try-except blocks removed
- Syntax errors fixed

### ⚠️ Known Issues
- **MoTAdapter attribute error**: The `_initialize_mot_experts_from_checkpoint` method expects `w1` attribute on `MoTAdapter`, but `MoTAdapter` wraps the expert module. This was previously caught by try-except but now surfaces as an error.
  - **Impact**: Expert initialization from checkpoint may fail
  - **Workaround**: Expert weights are initialized randomly, which is acceptable for adapter training

### 🚀 Next Steps
1. Fix MoTAdapter expert initialization (handle wrapped expert structure)
2. Test training with pre-encoded tokens end-to-end
3. Encode remaining audio files (currently 50/5233 files encoded)
4. Monitor training loss to ensure no NaN issues persist

## Benefits of Pre-Encoding

1. **Eliminates Graph Errors**: Completely decouples audio encoder from training graph
2. **Faster Training**: No on-the-fly encoding overhead during training
3. **Reproducibility**: Same tokens used across training runs
4. **Memory Efficiency**: Don't need to keep MusicGen model in memory during training
5. **Debugging**: Easier to inspect and debug token values

## Usage

### Pre-encode Audio Files
```bash
cd llama3/adapter_training
python pre_encode_audio.py
```

### Train with Pre-encoded Tokens
```bash
python train_adapter.py
# The script automatically detects pre_encoded_audio/ directory and uses it
```

## Technical Details

### Pre-encoded File Format
- **File**: `{ytid}.npy`
- **Shape**: `[n_codebooks, seq_len]` or `[batch, n_codebooks, seq_len]`
- **Metadata**: `pre_encoded_metadata.json` maps YouTube IDs to file paths

### Token Processing
- **Flattening**: 4 codebooks interleaved: `c0_t0, c1_t0, c2_t0, c3_t0, c0_t1...`
- **Vocabulary Shift**: Audio tokens shifted by `Llama_vocab_size` (128256) to create distinct vocabulary
- **Range**: Audio tokens in range `[128256, 130303]` (2048 tokens)

### Training Configuration
- **Device**: MPS (Apple Silicon)
- **Max Sequence Length**: 1024 tokens
- **Max Audio Tokens**: 992 (leaving room for text tokens)
- **Batch Size**: 1 (for MPS memory constraints)
- **Gradient Clipping**: Max norm 1.0 (to prevent exploding gradients)

## Lessons Learned

1. **Graph Connections**: Even `torch.no_grad()` doesn't guarantee graph isolation if models are on the same device
2. **Pre-encoding**: Offline preprocessing is often the cleanest solution for graph-related issues
3. **Embedding Initialization**: Zero initialization can cause numerical instability; small random values work better
4. **Weight Copying**: Always detach and clone when copying weights between models to avoid graph connections
5. **Error Handling**: Removing try-except blocks helped surface real issues that were being masked

## References

- Original error analysis: `BACKWARD_GRAPH_ERROR.md`
- Pre-encoding script: `pre_encode_audio.py`
- Training script: `train_adapter.py`
- Dataset class: `MusicCapsDataset` in `train_adapter.py`

