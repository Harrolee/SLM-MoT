# Training Analysis: Your Questions Answered

## 1. Is Loss of 5.4 Good? What's the Goal?

### Loss Context

**Cross-Entropy Loss on 130304-token vocabulary:**
- **Theoretical minimum**: 0 (perfect prediction)
- **Random baseline**: ~11.6 (log(130304) ≈ 11.6)
- **Your current loss**: ~5.4
- **Improvement**: 9.76 → 5.4 (45% reduction from start)

### Is 5.4 Good?

**For audio token prediction, 5.4 is reasonable but could be better:**
- ✅ Much better than random (11.6)
- ✅ Loss is decreasing (good sign)
- ⚠️ Still relatively high (could be 2-4 for well-trained models)
- ⚠️ Plateauing suggests we might need more training or better data

### What's a Good Target?

- **Excellent**: 2-3 (model is very confident)
- **Good**: 3-4 (model is learning well)
- **Acceptable**: 4-6 (model is learning but needs work)
- **Poor**: >6 (model struggling)

**Your 5.4 is in the "acceptable" range** - the model is learning but has room for improvement.

## 2. Not Improving Between Epochs 2-3 - Increase Learning Rate?

### Current Progress

```
Epoch 1: 6.01 → 5.76 (avg loss)
Epoch 2: 5.76 → 5.75 (avg loss)  ← Plateauing!
Epoch 3: Currently ~5.75
```

### Why It Might Be Plateauing

1. **Dummy Data**: You're training on random dummy audio! Real data would help.
2. **Learning Rate**: Current LR is `1e-4` - might be too conservative
3. **Limited Training**: Only 300 steps so far
4. **Output Layer**: Large layer (267M params) might need more steps

### Should You Increase LR?

**Current**: `lr=1e-4`

**Options**:
- **Try `lr=5e-4`**: 5x increase - more aggressive learning
- **Try `lr=1e-3`**: 10x increase - very aggressive (might overshoot)
- **Try learning rate schedule**: Start high, decay over time

**Recommendation**: Try `lr=5e-4` first. If loss spikes or becomes unstable, reduce it.

## 3. Overfitting Concerns

### You're Right!

**Overfitting risks are LOW here because:**
- ✅ Only training adapters + output layer (~334M params)
- ✅ Backbone is frozen (not overfitting)
- ✅ You'll discover overfitting quickly when testing
- ✅ Even overfit adapters might work reasonably well

**Signs of overfitting:**
- Training loss keeps decreasing but validation loss plateaus/increases
- Model memorizes training examples
- Poor generalization to new prompts

**Since you're using dummy data**, overfitting isn't really a concern yet - you need real data first!

## 4. Are We Training the Entire MoT?

### What's Trainable

```python
# Frozen (NOT trained):
- Llama backbone layers (all transformer blocks)
- Text expert FFN weights
- Audio expert FFN weights (MusicGen)
- Attention layers
- Embeddings (except new audio positions)

# Trainable (IS trained):
- Adapter down_proj layers (~33M params)
- Adapter up_proj layers (~33M params)
- Output layer (~267M params) ← NEW!
```

**Total trainable**: ~334M params out of ~1.9B total (18%)

### Does Loss Come from Text + Audio Tokens?

**NO! Loss ONLY comes from audio tokens:**

```python
# In collate_fn:
labels = input_ids.clone()
labels[:, :text_len] = -100  # Mask text positions - NO LOSS!

# CrossEntropyLoss ignores -100 labels
loss = loss_fn(logits, labels)  # Only audio positions contribute
```

**Key Point**: 
- Text positions: `labels = -100` → **ignored in loss**
- Audio positions: `labels = actual token IDs` → **loss computed here**

So the loss of 5.4 is **purely from audio token prediction**, not text!

## 5. How Do We Distinguish Audio vs Text Tokens?

### During Training

**Labels distinguish them:**
```python
Sequence: [text_tok1, text_tok2, audio_tok1, audio_tok2]
Labels:    [-100,      -100,      128256,      128257]
           ↑           ↑          ↑            ↑
        Ignored    Ignored    Loss here   Loss here
```

- Text positions: `-100` → CrossEntropyLoss ignores them
- Audio positions: Actual token IDs → Loss computed

### During Inference

**Logit filtering distinguishes them:**
```python
# Model outputs logits for ALL tokens [0:130304]
logits = model(input_ids, modality_masks=masks)  # [batch, seq_len, 130304]

# Filter to ONLY audio tokens
audio_logits = logits[0, -1, 128256:130304]  # Only audio range

# Sample from filtered logits
next_token = sample(audio_logits) + 128256  # Convert back to full vocab ID
```

**Key**: 
- Modality masks route to correct expert (text vs audio)
- Labels mask out text positions during training
- Logit filtering restricts to audio tokens during inference

## Recommendations

### Immediate Actions

1. **Increase Learning Rate**: Try `lr=5e-4` or `lr=1e-3`
2. **Train Longer**: Let it run to 500-1000 steps
3. **Use Real Data**: Replace dummy data with actual MusicCaps samples

### Monitoring

- **Watch loss trend**: Should decrease steadily
- **Check checkpoint size**: Should be ~1GB+ (includes output layer)
- **Test inference**: Try generating audio after training

### Expected Behavior

- **Loss should decrease**: 5.4 → 4.5 → 3.5 → 2.5 (ideally)
- **If loss plateaus**: Increase LR or train longer
- **If loss spikes**: LR too high, reduce it
- **If loss decreases slowly**: Normal for large output layer

## Summary

✅ **Loss 5.4**: Acceptable, room for improvement
✅ **Plateauing**: Normal, try higher LR or more training
✅ **Overfitting**: Low risk, not a concern with dummy data
✅ **Training scope**: Only adapters + output layer (334M params)
✅ **Loss source**: ONLY audio tokens (text masked out)
✅ **Token distinction**: Labels mask text, logit filtering restricts audio

**Next Step**: Increase LR to `5e-4` and train longer, or get real data!

