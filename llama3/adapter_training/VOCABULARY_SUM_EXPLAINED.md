# Vocabulary Size = Sum of All Modality Vocabularies

## Your Insight

> "It's not a matter of making the output token dimensions of all experts match - it's a matter of **summing the dimensions** of the output shapes of all experts."

**Almost! Let me clarify the distinction:** 🎯

**Important Distinction:**
- **Expert output dimensions**: Hidden states (same for all: 2048-dim) ✅
- **Vocabulary sizes**: Token vocabularies (different: 128256 + 2048 = 130304) ✅

So it's about **summing vocabulary sizes**, not expert output dimensions!

## The Architecture

### Experts Output Hidden States (Same Dimension)

```
Text Expert:  [batch, seq_len, 2048] → [batch, seq_len, 2048]  (hidden states)
Audio Expert: [batch, seq_len, 2048] → [batch, seq_len, 2048]  (hidden states)
```

**Key Point**: Both experts output the **same dimension** (2048), but they're **specialized** for different modalities.

### Output Layer Maps to Vocabulary (Sum of All Vocabularies)

```
Shared Output Layer: [batch, seq_len, 2048] → [batch, seq_len, vocab_size]
```

Where `vocab_size = sum of all modality vocabularies`:

```
vocab_size = text_vocab_size + audio_vocab_size
           = 128256 + 2048
           = 130304
```

## Visual Breakdown

### Token ID Ranges

```
Text Tokens:  0        → 128255  (128256 tokens)
Audio Tokens: 128256   → 130303  (2048 tokens)
─────────────────────────────────────────────
Total:        0        → 130303  (130304 tokens)
```

### Expert Processing

```
Input Sequence: [text_tok1, text_tok2, audio_tok1, audio_tok2]
                    ↓
Modality Routing:
  - Position 0,1 → Text Expert → [h1, h2] (2048-dim hidden states)
  - Position 2,3 → Audio Expert → [h3, h4] (2048-dim hidden states)
                    ↓
Merged Hidden States: [h1, h2, h3, h4] (all 2048-dim)
                    ↓
Shared Output Layer: [logits1, logits2, logits3, logits4]
  Each logits_i: [130304] = [text_logits (128256) + audio_logits (2048)]
                    ↓
Token Prediction:
  - Position 0,1: Filter to text range [0:128256]
  - Position 2,3: Filter to audio range [128256:130304]
```

## Why Sum, Not Match?

### ❌ Wrong Understanding
"Each expert outputs tokens in its own vocabulary size"
- Text Expert → 128256 tokens
- Audio Expert → 2048 tokens
- Output layer size = max(128256, 2048) = 128256

### ✅ Correct Understanding
"All experts output hidden states (same dimension), output layer maps to combined vocabulary"
- Text Expert → 2048-dim hidden states
- Audio Expert → 2048-dim hidden states
- Output layer size = 128256 + 2048 = 130304 (sum of vocabularies)

## The Key Insight

**Experts don't output tokens - they output hidden states!**

- **Expert level**: Modality-specific processing → hidden states (same dimension)
- **Output level**: Shared mapping → vocabulary (sum of all vocabularies)

## Why This Design?

1. **Shared representation space**: Both experts output to same 2048-dim space
2. **Unified vocabulary**: Output layer predicts over all possible tokens
3. **Modality routing**: Happens at expert level (which expert processes which tokens)
4. **Token filtering**: Happens at output level (which tokens are valid for which positions)

## Summary

✅ **Experts**: Output hidden states (same dimension: 2048) - **NOT summed**
✅ **Output Layer**: Maps to vocabulary (sum of vocabularies: 130304) - **THIS is summed**
✅ **Vocabulary Size**: 128256 (text) + 2048 (audio) = 130304 (total) - **Sum of vocabularies**

**Key Point**: 
- Expert output dimensions are the **same** (2048-dim hidden states)
- Vocabulary size is the **sum** of all modality vocabularies (130304 tokens)

So yes, you're right - it's about **summing vocabulary sizes**, not expert output dimensions!

