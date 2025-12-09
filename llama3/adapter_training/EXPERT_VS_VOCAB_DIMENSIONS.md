# Expert Output Dimensions vs. Vocabulary Sizes

## Your Question

> "It's not a matter of making the output token dimensions of all experts match - it's a matter of **summing the dimensions** of the output shapes of all experts."

## The Key Distinction

### Expert Output Dimensions (Hidden States) - **NOT Summed**

```
Text Expert:  [batch, seq_len, 2048] → [batch, seq_len, 2048]  (hidden states)
Audio Expert: [batch, seq_len, 2048] → [batch, seq_len, 2048]  (hidden states)
```

**Both experts output the SAME dimension**: 2048-dim hidden states
- These are **NOT summed** - they're the same!
- Experts don't output tokens - they output hidden representations

### Vocabulary Sizes (Token Vocabularies) - **THIS is Summed**

```
Text Vocabulary:  128256 tokens  (IDs: 0-128255)
Audio Vocabulary: 2048 tokens    (IDs: 128256-130303)
─────────────────────────────────────────────────
Total Vocabulary: 130304 tokens (IDs: 0-130303)
```

**Vocabulary size IS the sum**: 128256 + 2048 = 130304
- This is what the output layer maps to
- Each token ID is unique across all modalities

## Visual Flow

```
Input Tokens: [text_tok, text_tok, audio_tok, audio_tok]
                    ↓
Expert Processing (same dimension):
  Text Expert  → [h1, h2]  (2048-dim hidden states)
  Audio Expert → [h3, h4]  (2048-dim hidden states)
                    ↓
Merged Hidden States: [h1, h2, h3, h4]  (all 2048-dim)
                    ↓
Output Layer (summed vocabulary):
  [h1, h2, h3, h4] → [logits1, logits2, logits3, logits4]
  Each logits_i: [130304] = [128256 text + 2048 audio]
```

## Answer to Your Question

**What gets summed?**
- ✅ **Vocabulary sizes**: 128256 + 2048 = 130304
- ❌ **NOT expert output dimensions**: Both are 2048 (same, not summed)

**Why?**
- Experts output **hidden states** (same dimension: 2048)
- Output layer maps to **vocabulary** (sum of vocabularies: 130304)

## Code Reference

```python
# Expert output dimensions (same for all)
text_expert_output_dim = 2048
audio_expert_output_dim = 2048  # Same!

# Vocabulary sizes (summed)
text_vocab_size = 128256
audio_vocab_size = 2048
total_vocab_size = text_vocab_size + audio_vocab_size  # 130304

# Output layer
output_layer = Linear(2048, total_vocab_size)  # Maps to summed vocabulary
```

## Summary

✅ **Expert dimensions**: Same (2048-dim hidden states) - **NOT summed**
✅ **Vocabulary size**: Sum of modality vocabularies (130304) - **THIS is summed**

So yes, you're right about summing - but it's **vocabulary sizes** that are summed, not expert output dimensions!

