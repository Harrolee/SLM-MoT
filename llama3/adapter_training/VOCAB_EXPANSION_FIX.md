# Vocabulary Expansion Fix

## Problem
The model's output layer needs to be expanded to accommodate audio tokens (128256-130303), but the training script wasn't doing this expansion.

## What Was Fixed

### 1. Training Script (`train_adapter.py`)
**Before**: Model was built with original vocab_size (128256), couldn't predict audio tokens.

**After**: 
- ✅ Expands `vocab_size` from 128256 → 130304 before building model
- ✅ Loads Llama checkpoint and expands `tok_embeddings` and `output` layers
- ✅ Loads MusicGen weights into audio expert FFN layers
- ✅ Initializes MoT experts from checkpoint

### 2. Inference Script (`inference_with_adapters.py`)
**Already correct**: 
- ✅ Expands vocab_size before building model
- ✅ Expands embeddings and output layers when loading checkpoint
- ✅ Loads MusicGen weights into audio expert

## Architecture Clarification

### Why Both Experts Share One Output Layer?

**Question**: "Why will both llama and musicgen output tokens in the same response? I expected it to be either or"

**Answer**: 
- **Expert routing** (FFN level): Text positions → text expert, Audio positions → audio expert ✅
- **Output layer**: Shared, predicts over entire vocabulary (text + audio tokens)
- **Token filtering**: Enforced during training (loss masking) and inference (logit filtering)

See `MOT_ARCHITECTURE_EXPLAINED.md` for full details.

## Key Changes

### `train_adapter.py`
```python
# 1. Expand vocabulary BEFORE building model
original_vocab_size = model_args.vocab_size  # 128256
audio_vocab_size = 2048  # EnCodec codebook size
model_args.vocab_size = original_vocab_size + audio_vocab_size  # 130304

# 2. Build model with expanded vocab
model = Transformer(model_args)

# 3. Load checkpoint and expand layers
checkpoint = torch.load(ckpt_path)
# Expand tok_embeddings: [128256, dim] → [130304, dim]
# Expand output: [128256, dim] → [130304, dim]
model.load_state_dict(checkpoint, strict=False)

# 4. Load MusicGen weights into audio expert
musicgen = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
# Copy FFN weights to audio expert FFN layers
```

## Verification

To verify vocabulary expansion is working:

1. **Check model output layer size**:
   ```python
   print(model.output.weight.shape)  # Should be [130304, 2048]
   ```

2. **Check embeddings size**:
   ```python
   print(model.tok_embeddings.weight.shape)  # Should be [130304, 2048]
   ```

3. **During inference, check logits range**:
   ```python
   logits = model(input_ids, modality_masks=masks)
   print(logits.shape)  # Should be [batch, seq_len, 130304]
   audio_logits = logits[0, -1, 128256:130304]
   print(audio_logits.shape)  # Should be [2048]
   ```

## Next Steps

1. ✅ Vocabulary expansion fixed in training script
2. ✅ MusicGen weights loading added to training script
3. ✅ Architecture explanation documented
4. 🔄 Ready to retrain adapters with correct vocab size
5. 🔄 Ready to run inference with trained adapters

