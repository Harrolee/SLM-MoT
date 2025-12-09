# MoT Architecture: Why Both Experts Share One Output Layer

## Your Question
> "In the MoT model, why will both llama and musicgen output tokens in the same response? I expected it to be either or"

This is a great question! Let me explain the architecture choice.

## The Architecture

### What Happens During Forward Pass

1. **Input Sequence**: `[text_tokens] + [audio_tokens]`
   - Example: `[128000, 128001, ..., 128256, 128257, ...]`
   - Text tokens: 0-128255
   - Audio tokens: 128256-130303

2. **Modality Routing** (Inside Each Layer):
   - **Text positions** → routed to **text expert** (Llama FFN)
   - **Audio positions** → routed to **audio expert** (MusicGen FFN via adapter)
   - Each expert processes its tokens independently

3. **Shared Output Layer**:
   - Both experts' outputs feed into the **SAME** `output` layer
   - This layer maps `[hidden_dim] → [vocab_size]` where `vocab_size = 130304`
   - So it can predict **any token** (text OR audio) at **any position**

## Why This Design?

### Option A: Shared Output (What We're Doing)
```
Text Expert → ──┐
                ├─→ Shared Output Layer → [130304 logits] → Can predict text OR audio
Audio Expert → ─┘
```

**Pros:**
- Simpler architecture (one output layer)
- Model learns to predict the right token type based on context
- Cross-modal understanding (text expert can "see" audio tokens in vocabulary)

**Cons:**
- Model CAN technically predict text tokens at audio positions (we filter this out)
- Requires manual filtering during inference

### Option B: Separate Output Layers (What You Expected)
```
Text Expert → Text Output Layer → [128256 logits] → Only text tokens
Audio Expert → Audio Output Layer → [2048 logits] → Only audio tokens
```

**Pros:**
- Explicit separation - each expert can only output its modality
- No filtering needed

**Cons:**
- More complex architecture
- Less parameter sharing
- Harder to learn cross-modal relationships

## How We Ensure "Either/Or" Behavior

### During Training (`train_adapter.py`)
```python
# Create labels
labels = input_ids.clone()
labels[:, :text_len] = -100  # Mask text positions - no loss computed
# Only audio positions contribute to loss
```

The model learns: "At audio positions, predict audio tokens (or get penalized)".

### During Inference (`inference_with_adapters.py`)
```python
# Forward pass - model outputs logits for ALL tokens
logits = model(seq_tensor, start_pos=0, modality_masks=modality_masks)

# Filter to ONLY audio tokens
audio_logits = next_token_logits[audio_token_min:audio_token_max + 1]

# Sample from filtered logits
audio_code = torch.multinomial(probs, num_samples=1).item()
```

We manually filter logits to enforce "audio positions → audio tokens only".

## The Key Insight

**The routing happens at the EXPERT level (FFN), not at the OUTPUT level.**

- **Expert routing**: Text positions → text expert, Audio positions → audio expert ✅
- **Output layer**: Shared, predicts over entire vocabulary
- **Token filtering**: Enforced during training (loss masking) and inference (logit filtering)

## Visual Flow

```
Input: [text_tok1, text_tok2, audio_tok1, audio_tok2]
         ↓
Modality Masks: [T, T, F, F], [F, F, T, T]
         ↓
Layer 1:
  - Text expert processes positions 0,1
  - Audio expert processes positions 2,3
         ↓
... (through all layers) ...
         ↓
Final Hidden States: [h1, h2, h3, h4]
         ↓
Shared Output Layer: [logits1, logits2, logits3, logits4]
  Each logits_i is shape [130304] - can predict ANY token
         ↓
Training: Loss only computed on positions 2,3 (audio)
Inference: Filter logits2, logits3 to only audio tokens [128256:130304]
```

## Could We Change It?

Yes! If you want strict "either/or" behavior, we could:

1. **Add separate output layers**:
   ```python
   self.text_output = ColumnParallelLinear(dim, text_vocab_size)
   self.audio_output = ColumnParallelLinear(dim, audio_vocab_size)
   ```

2. **Route based on modality mask**:
   ```python
   if modality_mask[i] == text:
       logits = self.text_output(hidden[i])
   else:
       logits = self.audio_output(hidden[i])
   ```

But the current design is simpler and works well with proper filtering!

