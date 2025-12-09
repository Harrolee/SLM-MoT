---
language:
- en
task_categories:
- audio-to-text
- text-to-audio
tags:
- music
- audio
- musicgen
- encodec
- llama
- mixture-of-transformers
pretty_name: MusicCaps MoT Tokens
size_categories:
- 1K<n<10K
---

# MusicCaps Pre-Encoded Tokens for Mixture-of-Transformers (MoT)

## Dataset Description

This dataset contains pre-encoded audio tokens from the [MusicCaps dataset](https://huggingface.co/datasets/google/MusicCaps),
processed through Meta's MusicGen EnCodec tokenizer for use in Mixture-of-Transformers (MoT) training.

### Dataset Summary

- **5,233 music clips** encoded as discrete tokens
- **4 codebook layers** from MusicGen's EnCodec
- **~500 tokens per 10-second clip**
- **Compressed from ~12GB audio to 82MB tokens**
- Ready for multimodal language model training

## Intended Use

This dataset is designed for:
- Training Mixture-of-Transformers (MoT) models that combine Llama and MusicGen
- Research in multimodal language models with audio understanding
- Experiments in music captioning and generation
- Efficient training without on-the-fly audio encoding

## Dataset Structure

### Data Fields

- `ytid`: YouTube video ID
- `caption`: Human-written music description
- `aspect_list`: Musical aspects mentioned in caption
- `audioset_positive_labels`: AudioSet labels
- `audio_codes`: Pre-encoded tokens shape `[4, ~500]` (4 codebooks, ~500 time steps)
- `n_codebooks`: Number of codebooks (always 4)
- `seq_length`: Sequence length of tokens
- `start_s`: Start time in original video
- `end_s`: End time in original video
- `author_id`: Caption author ID
- `is_balanced_subset`: Whether part of balanced subset
- `is_audioset_eval`: Whether part of AudioSet eval

### Data Splits

- `train`: 4,710 examples (90%)
- `test`: 523 examples (10%)

## Pre-Encoding Details

### Encoding Process

1. **Audio Loading**: 10-second clips from MusicCaps
2. **Resampling**: All audio resampled to 32kHz (MusicGen requirement)
3. **Tokenization**: MusicGen EnCodec with 4 codebooks @ 50Hz
4. **Vocabulary**: 2048 tokens per codebook
5. **Compression**: ~12GB audio → 82MB tokens

### Token Format

```python
# Shape: [4, ~500]
# - 4 codebooks (hierarchical encoding)
# - ~500 time steps (50Hz * 10 seconds)
# Each value in range [0, 2047]
```

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("YOUR_USERNAME/musiccaps-mot-tokens")

# Access pre-encoded tokens
sample = dataset['train'][0]
audio_codes = np.array(sample['audio_codes'])  # Shape: [4, ~500]
caption = sample['caption']
```

### Using with MoT Training

```python
# Shift tokens for combined vocabulary
# Llama uses tokens [0, 128255]
# Audio uses tokens [128256, 130303]
audio_tokens = audio_codes + 128256

# Interleave codebooks for sequence modeling
# [c0_t0, c1_t0, c2_t0, c3_t0, c0_t1, ...]
b, k, t = 1, audio_codes.shape[0], audio_codes.shape[1]
interleaved = audio_codes.transpose(1, 0).reshape(-1)
```

## Training Configuration

Recommended settings for MoT adapter training:

- **Model**: Llama 3.2 1B + MusicGen Small
- **Adapter dims**: 67M parameters
- **Batch size**: 8-16 (on GPU)
- **Learning rate**: 1e-4
- **Max sequence**: 1024 tokens (32 text + 992 audio)

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{musiccaps_mot_tokens,
  title={MusicCaps Pre-Encoded Tokens for MoT},
  author={Your Name},
  year={2024},
  publisher={HuggingFace}
}

@article{musiccaps,
  title={MusicCaps: Music Audio Captioning with Text-Audio Retrieval},
  author={Agostinelli et al.},
  year={2023}
}
```

## Acknowledgments

- Google for the original MusicCaps dataset
- Meta for MusicGen and EnCodec
- The Mixture-of-Transformers paper authors

## License

This dataset inherits licenses from:
- MusicCaps: [Research use]
- Encoded representations are derivative works

Please ensure compliance with original dataset licenses.
