# Upload to HuggingFace

## Quick Setup

1. **Get your HuggingFace token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with write permissions
   - Copy the token

2. **Set environment variable and upload:**
   ```bash
   export HUGGING_FACE_HUB_TOKEN="your_token_here"
   python upload_to_hf.py
   ```

## Alternative: Use HF CLI

1. **Install and login:**
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```

2. **Then run:**
   ```bash
   python upload_to_hf.py
   ```

## Dataset will be available at:
`https://huggingface.co/datasets/YOUR_USERNAME/musiccaps-mot-tokens`

## Lambda Labs Usage

Once uploaded, on Lambda Labs:

```python
from datasets import load_dataset
import numpy as np

# Load dataset
dataset = load_dataset("YOUR_USERNAME/musiccaps-mot-tokens")

# Get a sample
sample = dataset['train'][0]
audio_codes = np.array(sample['audio_codes'])  # [4, ~500]
caption = sample['caption']

print(f"Caption: {caption}")
print(f"Audio tokens shape: {audio_codes.shape}")
```

## Training on Lambda Labs

1. **Clone your training repo**
2. **Install dependencies:**
   ```bash
   pip install torch transformers datasets accelerate
   ```

3. **Modify train_adapter.py to use HF dataset:**
   ```python
   from datasets import load_dataset
   dataset = load_dataset("YOUR_USERNAME/musiccaps-mot-tokens")
   ```

4. **Run training with GPU power!**
   ```bash
   python train_adapter.py
   ```

Expected speedup: **10-50x faster than M2!**