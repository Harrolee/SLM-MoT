import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoProcessor, 
    MusicgenForConditionalGeneration, 
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from datasets import load_dataset
import os
import json
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
import soundfile as sf
from io import BytesIO
from scipy import signal

# Add parent directory to path to import llama model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama.model import ModelArgs, Transformer

class MusicCapsDataset(IterableDataset):
    """
    MusicCaps dataset - supports both downloaded audio files and streaming.
    Uses IterableDataset for proper streaming support.
    Now supports pre-encoded audio tokens for faster training.
    """
    def __init__(self, split="train", max_length=512, max_samples=None, audio_dir=None, pre_encoded_dir=None):
        self.split = split
        self.max_length = max_length
        self.max_samples = max_samples
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.pre_encoded_dir = Path(pre_encoded_dir) if pre_encoded_dir else None
        
        # Check for pre-encoded tokens
        self.use_pre_encoded = False
        if self.pre_encoded_dir and self.pre_encoded_dir.exists():
            metadata_file = self.pre_encoded_dir / "metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    self.pre_encoded_metadata = json.load(f)
                encoded_files = list(self.pre_encoded_dir.glob("*.npy"))
                if len(encoded_files) > 0:
                    print(f"📦 Found {len(encoded_files)} pre-encoded audio token files")
                    self.use_pre_encoded = True
                    print(f"   ✅ Using pre-encoded tokens (no encoding during training!)")
        
        # Check if we have downloaded audio files
        self.use_downloaded_audio = False
        if self.audio_dir and self.audio_dir.exists():
            audio_files = list(self.audio_dir.glob("*.wav"))
            if len(audio_files) > 0:
                print(f"📦 Found {len(audio_files)} downloaded audio files in {self.audio_dir}")
                self.use_downloaded_audio = True
                # Load metadata from HuggingFace
                self.metadata = load_dataset("google/MusicCaps", split=split)
                print(f"   ✅ Loaded MusicCaps metadata")
        
        # Load MusicCaps dataset (for streaming or metadata)
        if not self.use_downloaded_audio:
            print(f"📦 Loading MusicCaps dataset (streaming=True)...")
            self.dataset = load_dataset("google/MusicCaps", split=split, streaming=True)
            # Test access
            sample = next(iter(self.dataset))
            print(f"   ✅ Successfully loaded MusicCaps dataset")
            print(f"   - Sample keys: {list(sample.keys())}")
            # Set metadata for pre-encoded path
            if self.use_pre_encoded:
                self.metadata = self.dataset
            # Check if we have audio field (from downloaded dataset)
            if "audio" in sample and isinstance(sample["audio"], dict):
                self.use_real_data = True
            else:
                print(f"   ⚠️  No audio files - need to download first")
                print(f"   Run: python download_musiccaps_subset.py --limit 500")
                self.use_real_data = False
        else:
            self.use_real_data = True
            self.dataset = None  # We'll use metadata + audio files
            
        # Using MusicGen processor for audio tokenization (only if not using pre-encoded)
        if not self.use_pre_encoded:
            self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
            self.musicgen = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
            self.musicgen.eval() # We only use it for encoding targets
        else:
            # Don't load musicgen if using pre-encoded tokens
            self.processor = None
            self.musicgen = None
        
        # Using Llama tokenizer for text
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self):
        """Iterate over the dataset."""
        if self.use_pre_encoded:
            # Use pre-encoded tokens - no audio encoding needed!
            sample_count = 0
            for sample in self.metadata:
                if self.max_samples and sample_count >= self.max_samples:
                    break
                
                ytid = sample['ytid']
                
                # Check if we have pre-encoded tokens for this file
                if ytid not in self.pre_encoded_metadata:
                    continue
                
                encoded_file = Path(self.pre_encoded_metadata[ytid]["encoded_file"])
                if not encoded_file.exists():
                    continue
                
                # Load pre-encoded tokens
                import numpy as np
                audio_codes = np.load(encoded_file)  # [batch, n_codebooks, seq_len]
                
                # Process text
                caption = sample.get("caption", "")
                text_inputs = self.tokenizer(
                    caption, 
                    return_tensors="pt", 
                    padding="max_length", 
                    truncation=True, 
                    max_length=32
                )
                
                yield {
                    "text_input_ids": text_inputs.input_ids.squeeze(),
                    "text_attention_mask": text_inputs.attention_mask.squeeze(),
                    "audio_codes": audio_codes,  # Pre-encoded numpy array
                    "caption": caption,
                    "ytid": ytid
                }
                
                sample_count += 1
                    
        elif self.use_downloaded_audio:
            # Use downloaded audio files
            sample_count = 0
            for sample in self.metadata:
                if self.max_samples and sample_count >= self.max_samples:
                    break
                
                ytid = sample['ytid']
                audio_file = self.audio_dir / f"{ytid}.wav"
                
                if not audio_file.exists():
                    continue
                
                # Load audio file
                import soundfile as sf
                audio, sampling_rate = sf.read(str(audio_file))
                
                # Ensure mono and correct dtype
                if len(audio.shape) > 1:
                    audio = audio[:, 0]
                audio = audio.astype(np.float32)
                
                # Resample to 32000 Hz if needed (MusicGen requirement)
                target_sr = 32000
                if sampling_rate != target_sr:
                    num_samples = int(len(audio) * target_sr / sampling_rate)
                    audio = signal.resample(audio, num_samples)
                    sampling_rate = target_sr
                
                # Normalize
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))
                
                # Process text
                caption = sample.get("caption", "")
                text_inputs = self.tokenizer(
                    caption, 
                    return_tensors="pt", 
                    padding="max_length", 
                    truncation=True, 
                    max_length=32
                )
                
                yield {
                    "text_input_ids": text_inputs.input_ids.squeeze(),
                    "text_attention_mask": text_inputs.attention_mask.squeeze(),
                    "audio_values": audio, 
                    "sampling_rate": sampling_rate,
                    "caption": caption
                }
                
                sample_count += 1
                    
        elif self.use_real_data and self.dataset is not None:
            sample_count = 0
            for sample in self.dataset:
                if self.max_samples and sample_count >= self.max_samples:
                    break
                    
                try:
                    # Extract caption
                    caption = sample.get("caption", "") or sample.get("text", "")
                    if not caption:
                        continue
                    
                    # Extract audio - MusicCaps might have different structures
                    audio_bytes = None
                    sampling_rate = 32000  # Default
                    
                    # Try different possible field structures
                    if "audio" in sample:
                        audio_data = sample["audio"]
                        if isinstance(audio_data, dict):
                            audio_bytes = audio_data.get("bytes")
                            sampling_rate = audio_data.get("sampling_rate", 32000)
                        elif isinstance(audio_data, bytes):
                            audio_bytes = audio_data
                    
                    if audio_bytes is None:
                        # Try alternative field names
                        audio_bytes = sample.get("audio_bytes")
                    
                    if audio_bytes is None:
                        # MusicCaps might have URLs instead - skip for now
                        # TODO: Could download from YouTube URLs if needed
                        print(f"   ⚠️  Sample has no audio bytes (might be URL-based), skipping...")
                        print(f"   - Sample keys: {list(sample.keys())}")
                        continue
                    
                    # Decode audio from bytes
                    try:
                        audio, sr = sf.read(BytesIO(audio_bytes))
                        if sr:
                            sampling_rate = sr
                    except Exception as e:
                        print(f"   ⚠️  Error decoding audio: {e}, skipping...")
                        continue
                    
                    # Ensure mono and correct dtype
                    if len(audio.shape) > 1:
                        audio = audio[:, 0]  # Take first channel
                    audio = audio.astype(np.float32)
                    
                    # Normalize audio
                    if np.max(np.abs(audio)) > 0:
                        audio = audio / np.max(np.abs(audio))
                    
                    # Process text
                    text_inputs = self.tokenizer(
                        caption, 
                        return_tensors="pt", 
                        padding="max_length", 
                        truncation=True, 
                        max_length=32
                    )
                    
                    yield {
                        "text_input_ids": text_inputs.input_ids.squeeze(),
                        "text_attention_mask": text_inputs.attention_mask.squeeze(),
                        "audio_values": audio, 
                        "sampling_rate": sampling_rate,
                        "caption": caption
                    }
                    
                    sample_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️  Error processing sample: {e}, skipping...")
                    continue
        else:
            # Dummy data fallback
            for _ in range(100):  # Generate 100 dummy samples
                caption = "A test caption"
                audio = np.random.uniform(-1, 1, 16000*5).astype(np.float32)  # 5 sec dummy audio
            sampling_rate = 32000

        text_inputs = self.tokenizer(
            caption, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=32
        )
        
        yield {
            "text_input_ids": text_inputs.input_ids.squeeze(),
            "text_attention_mask": text_inputs.attention_mask.squeeze(),
            "audio_values": audio, 
            "sampling_rate": sampling_rate,
            "caption": caption
        }

def collate_fn(batch, processor=None, musicgen_full_model=None, device="cpu", use_pre_encoded=False):
    """
    Custom collate function to:
    1. Pad text inputs
    2. Encode audio to tokens using MusicGen (on device)
    3. Concatenate Text + Audio for training
    
    Args:
        musicgen_full_model: The full MusicgenForConditionalGeneration model (not decoder.model)
    """
    # 1. Pad Text
    text_input_ids = [b["text_input_ids"] for b in batch]
    
    text_input_ids = torch.stack(text_input_ids).to(device)
    
    # 2. Handle Audio Tokens
    if use_pre_encoded:
        # Load pre-encoded tokens (no encoding, no graph connection!)
        import numpy as np
        audio_codes_list = []
        for b in batch:
            audio_codes = b["audio_codes"]  # numpy array [batch, n_codebooks, seq_len]
            # Convert to tensor
            audio_codes = torch.from_numpy(audio_codes).to(device)
            audio_codes_list.append(audio_codes)
        
        # Stack if multiple samples, otherwise use single
        if len(audio_codes_list) == 1:
            audio_codes = audio_codes_list[0]
        else:
            audio_codes = torch.stack(audio_codes_list)
        
        # Ensure correct shape: [batch, n_codebooks, seq_len]
        while audio_codes.dim() > 3:
            audio_codes = audio_codes.squeeze()
        
        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(0)
            
    else:
        # Original encoding path (for backward compatibility)
        audio_values = [b["audio_values"] for b in batch]
        sampling_rate = batch[0]["sampling_rate"]
        
        # Process with MusicGen Processor
        inputs = processor(
            audio=audio_values,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Encode to codes
        with torch.no_grad():
            encoder_outputs = musicgen_full_model.audio_encoder(inputs.input_values)
            audio_codes = encoder_outputs.audio_codes
            
        # Handle list/tensor formats
        if isinstance(audio_codes, list):
            audio_codes = torch.stack(audio_codes)
            
        # Handle various dimensions
        while audio_codes.dim() > 3:
            audio_codes = audio_codes.squeeze()
            
        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(0)
    
    # Now audio_codes is [batch, n_codebooks, seq_len]
    b, k, t = audio_codes.shape
    
    # Transpose to [b, t, k] then flatten to [b, t*k]
    # This interleaves codebooks: c0_t0, c1_t0, c2_t0, c3_t0, c0_t1...
    audio_tokens = audio_codes.permute(0, 2, 1).contiguous().view(b, t * k)
    
    # Ensure audio tokens are detached (especially important for pre-encoded)
    audio_tokens = audio_tokens.detach().requires_grad_(False)
    # Ensure audio codes are in valid range [0, 2047] for EnCodec
    audio_tokens = audio_tokens.clamp(0, 2047)

    # Shift tokens for vocabulary (Llama Vocab Size = 128256)
    # Audio tokens will be in range [128256, 130303]
    audio_tokens = audio_tokens + 128256

    # Extra detach after shifting to be absolutely sure
    audio_tokens = audio_tokens.detach()
    
    # Final safety check: ensure tokens are within vocab_size
    audio_tokens = audio_tokens.clamp(0, 130303)
    
    # 3. Concatenate [Text, Audio]
    bs, text_len = text_input_ids.shape
    bs_audio = audio_tokens.shape[0]
    
    if bs != bs_audio:
         # If batch size mismatch, try to fix (e.g. if audio was processed as 1 batch but text as another)
         # But here they should match
         pass
    
    # Truncate audio tokens to fit within max_seq_len (leave room for text)
    # max_seq_len = 1024, text_len ~32, so max audio_len = 1024 - 32 = 992
    max_audio_len = 992  # Leave room for text tokens
    if audio_tokens.shape[1] > max_audio_len:
        audio_tokens = audio_tokens[:, :max_audio_len]
        print(f"   ⚠️  Truncated audio tokens from {t * k} to {max_audio_len}")
    
    # Concatenate - ensure both are detached to avoid any graph connections
    text_input_ids = text_input_ids.detach()
    audio_tokens = audio_tokens.detach()
    input_ids = torch.cat([text_input_ids, audio_tokens], dim=1).detach()
    
    # Create Modality Masks
    # Mask 0 (Text): True for text pos
    mask_text = torch.cat([
        torch.ones_like(text_input_ids, dtype=torch.bool),
        torch.zeros_like(audio_tokens, dtype=torch.bool)
    ], dim=1)
    
    # Mask 1 (Audio): True for audio pos
    mask_audio = torch.cat([
        torch.zeros_like(text_input_ids, dtype=torch.bool),
        torch.ones_like(audio_tokens, dtype=torch.bool)
    ], dim=1)
    
    modality_masks = [mask_text, mask_audio]
    
    # Create Labels
    # Ignore loss for text (set to -100)
    labels = input_ids.clone().detach()
    labels[:, :text_len] = -100
    
    return {
        "input_ids": input_ids,
        "modality_masks": modality_masks,
        "labels": labels
    }

def train():
    print("🚀 Starting Adapter Training Setup...")
    
    # 1. Config
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        
    print(f"   - Device: {device}")
    
    # Initialize FairScale
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        import random
        port = random.randint(20000, 30000)
        os.environ.setdefault("MASTER_PORT", str(port))
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)

    if not model_parallel_is_initialized():
        initialize_model_parallel(1)
    
    # 2. Load MoT Model
    llama_ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "Llama-3.2-1B")
    if not os.path.exists(llama_ckpt_dir):
         print(f"❌ Llama checkpoint not found at {llama_ckpt_dir}")
         return

    with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
        params = json.load(f)
    
    model_args = ModelArgs(
        max_seq_len=1024, # Reduced to fit in MPS memory (was 2048)
        max_batch_size=1,  # Keep at 1 for memory efficiency
        **params
    )
    model_args.use_mot = True
    model_args.n_modalities = 2
    model_args.use_audio_expert = True
    model_args.audio_expert_dim = 1024
    
    # Expand vocabulary to include audio tokens
    # EnCodec has 2048 codes, so we need vocab_size = 128256 + 2048 = 130304
    original_vocab_size = model_args.vocab_size
    audio_vocab_size = 2048  # EnCodec codebook size
    model_args.vocab_size = original_vocab_size + audio_vocab_size
    print(f"   - Expanded vocab: {original_vocab_size} -> {model_args.vocab_size} (added {audio_vocab_size} audio tokens)")
    
    print("   - Building MoT Model...")
    model = Transformer(model_args)
    model.to(device)
    
    # Load Llama checkpoint weights (with vocabulary expansion)
    print("   - Loading Llama checkpoint weights...")
    checkpoints = sorted(Path(llama_ckpt_dir).glob("*.pth"))
    if len(checkpoints) > 0:
        ckpt_path = checkpoints[0]
        print(f"   - Loading from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Handle vocabulary expansion: expand embedding and output layers
        if 'tok_embeddings.weight' in checkpoint:
            old_emb = checkpoint['tok_embeddings.weight']  # [old_vocab, dim]
            new_emb = torch.zeros(model_args.vocab_size, old_emb.shape[1], 
                                 dtype=old_emb.dtype, device=old_emb.device)
            new_emb[:original_vocab_size] = old_emb
            # Initialize new audio token embeddings with small random values (not zeros!)
            # Use std similar to existing embeddings to avoid numerical instability
            emb_std = old_emb.std().item()
            new_emb[original_vocab_size:] = torch.randn(
                audio_vocab_size, old_emb.shape[1],
                dtype=old_emb.dtype, device=old_emb.device
            ) * (emb_std * 0.1)  # 10% of existing std for stability
            checkpoint['tok_embeddings.weight'] = new_emb
            print(f"   - Expanded embeddings: {old_emb.shape[0]} -> {new_emb.shape[0]}")
        
        if 'output.weight' in checkpoint:
            old_out = checkpoint['output.weight']  # [old_vocab, dim]
            new_out = torch.zeros(model_args.vocab_size, old_out.shape[1], 
                                 dtype=old_out.dtype, device=old_out.device)
            new_out[:original_vocab_size] = old_out
            # Initialize new audio token output weights with small random values
            out_std = old_out.std().item()
            new_out[original_vocab_size:] = torch.randn(
                audio_vocab_size, old_out.shape[1],
                dtype=old_out.dtype, device=old_out.device
            ) * (out_std * 0.1)  # 10% of existing std for stability
            checkpoint['output.weight'] = new_out
            print(f"   - Expanded output layer: {old_out.shape[0]} -> {new_out.shape[0]}")
        
        # Load the expanded checkpoint
        model.load_state_dict(checkpoint, strict=False)  # strict=False allows MoT-specific layers
        print("   ✅ Llama weights loaded")
        
        # Initialize MoT experts from checkpoint (for text expert only)
        if hasattr(model, '_initialize_mot_experts_from_checkpoint'):
            model._initialize_mot_experts_from_checkpoint(checkpoint)
    else:
        print("   ⚠️  No Llama checkpoint found, model will have random weights!")
    
    # Load MusicGen weights into audio expert
    print("   - Loading MusicGen weights into audio expert...")
    musicgen = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    musicgen.eval()
    # Freeze MusicGen completely - we only use it for encoding/decoding
    for param in musicgen.parameters():
        param.requires_grad = False
    
    # Copy MusicGen FFN weights to audio experts
    decoder = musicgen.decoder
    if hasattr(decoder, 'model'):
        musicgen_model = decoder.model
        if hasattr(musicgen_model, 'layers'):
            decoder_layers = musicgen_model.layers
        elif hasattr(musicgen_model, 'decoder'):
            decoder_layers = musicgen_model.decoder.layers
        else:
            raise ValueError("Could not find decoder layers")
    else:
        raise ValueError("Could not find decoder model")
    
    print("   - Copying MusicGen FFN weights to audio experts...")
    # CRITICAL: Use torch.no_grad() and detach to prevent graph connections
    with torch.no_grad():
        for layer_id, layer in enumerate(model.layers):
            if layer_id >= len(decoder_layers):
                continue
            
            if hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'local_experts'):
                audio_expert_adapter = layer.feed_forward.local_experts[-1]  # Last expert is audio
                if hasattr(audio_expert_adapter, 'expert'):
                    audio_ffn = audio_expert_adapter.expert
                    
                    if hasattr(decoder_layers[layer_id], 'fc1') and hasattr(decoder_layers[layer_id], 'fc2'):
                        # Detach and clone MusicGen weights to break graph completely
                        fc1_weight = decoder_layers[layer_id].fc1.weight.detach().clone().to(device)
                        fc2_weight = decoder_layers[layer_id].fc2.weight.detach().clone().to(device)
                        
                        audio_ffn.fc1.weight.copy_(fc1_weight)
                        if decoder_layers[layer_id].fc1.bias is not None:
                            if audio_ffn.fc1.bias is not None:
                                fc1_bias = decoder_layers[layer_id].fc1.bias.detach().clone().to(device)
                                audio_ffn.fc1.bias.copy_(fc1_bias)
                        audio_ffn.fc2.weight.copy_(fc2_weight)
                        if decoder_layers[layer_id].fc2.bias is not None:
                            if audio_ffn.fc2.bias is not None:
                                fc2_bias = decoder_layers[layer_id].fc2.bias.detach().clone().to(device)
                                audio_ffn.fc2.bias.copy_(fc2_bias)
    
    print("   ✅ MusicGen weights loaded")
    
    # 3. Freeze Weights
    print("   - Freezing backbone and experts...")
    print("   - Making adapters AND output layer trainable...")
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "down_proj" in name or "up_proj" in name:
            param.requires_grad = True  # Adapter projection layers
        if "output" in name:
            param.requires_grad = True  # Output layer (needed to learn audio tokens!)
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   - Trainable Parameters: {trainable_params:,} / {total_params:,}")
    print(f"   - Includes: Adapters (~67M) + Output Layer (~{trainable_params - 67_000_000:,})")
    
    # 4. Dataset
    print("   - Loading Data...")
    # Check for pre-encoded tokens first (best option)
    pre_encoded_dir = Path("./pre_encoded_audio")
    audio_dir = Path("./musiccaps_audio")
    
    if pre_encoded_dir.exists() and (pre_encoded_dir / "metadata.json").exists():
        print(f"   ✅ Found pre-encoded audio tokens!")
        print(f"   🚀 Using pre-encoded tokens (no encoding during training)")
        dataset = MusicCapsDataset(
            max_samples=1000, 
            audio_dir=str(audio_dir) if audio_dir.exists() else None,
            pre_encoded_dir=str(pre_encoded_dir)
        )
        use_pre_encoded = True
    elif audio_dir.exists() and len(list(audio_dir.glob("*.wav"))) > 0:
        print(f"   ✅ Found downloaded audio files, using them!")
        print(f"   ⚠️  Encoding will happen during training (slower)")
        print(f"   💡 For faster training, pre-encode: python pre_encode_audio.py")
        dataset = MusicCapsDataset(max_samples=1000, audio_dir=str(audio_dir))
        use_pre_encoded = False
    else:
        print(f"   ⚠️  No downloaded audio files or pre-encoded tokens found")
        print(f"   💡 To download audio: python download_musiccaps_subset.py --limit 500")
        print(f"   💡 To pre-encode: python pre_encode_audio.py")
        print(f"   Using streaming dataset (may fall back to dummy data)...")
        dataset = MusicCapsDataset(max_samples=1000)
        use_pre_encoded = False
    
    # Create collate function based on whether we're using pre-encoded tokens
    if use_pre_encoded:
        # No need for processor or musicgen model!
        collate_fn_wrapper = lambda b: collate_fn(b, device=device, use_pre_encoded=True)
    else:
        # Need processor and model for encoding
        collate_fn_wrapper = lambda b: collate_fn(b, dataset.processor, dataset.musicgen, device, use_pre_encoded=False)
    
    # IterableDataset doesn't support shuffle, so we skip it
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,  # IterableDataset doesn't support shuffle
        collate_fn=collate_fn_wrapper
    )
    
    # 5. Training Loop
    print("   - Starting Training Loop...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    # Only move musicgen to device if we're using it (not pre-encoded)
    if not use_pre_encoded and hasattr(dataset, 'musicgen'):
        dataset.musicgen.to(device)
    
    # Training parameters
    num_epochs = 3
    save_every_n_steps = 100
    log_every_n_steps = 10
    max_steps = 5  # Limit for initial training run - REDUCED FOR TESTING
    
    print(f"   - Training for {num_epochs} epochs (max {max_steps} steps)")
    print(f"   - Saving checkpoint every {save_every_n_steps} steps")
    print(f"   - Logging every {log_every_n_steps} steps")
    print()
    
    global_step = 0
    total_loss = 0.0
    
    for epoch in range(num_epochs):
        print(f"📚 Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            if global_step >= max_steps:
                print(f"\n⏸️  Reached max_steps ({max_steps}), stopping...")
                break
            
            optimizer.zero_grad()
            
            # Ensure complete detachment from any previous computation graph
            input_ids = batch["input_ids"].detach()
            modality_masks = batch["modality_masks"]
            labels = batch["labels"].detach()

            # Modality masks don't need gradients (they're boolean masks)
            modality_masks = [m.detach() if isinstance(m, torch.Tensor) else m for m in modality_masks]

            # Clear any lingering gradients
            if input_ids.grad is not None:
                input_ids.grad = None
            
            # Zero out cache to free memory (but keep tensors allocated)
            for layer in model.layers:
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'cache_k'):
                    if layer.attention.cache_k is not None:
                        layer.attention.cache_k.zero_()
                    if layer.attention.cache_v is not None:
                        layer.attention.cache_v.zero_()
            
            # Use torch.cuda.empty_cache() equivalent for MPS
            if device == "mps":
                torch.mps.empty_cache()
            
            # Debug: Check input_ids before forward pass
            if global_step == 0:
                print(f"   🔍 Debug Step 0:")
                print(f"      input_ids shape: {input_ids.shape}")
                print(f"      input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")
                print(f"      input_ids unique count: {len(input_ids.unique())}")
                print(f"      modality_masks[0] (text) sum: {modality_masks[0].sum().item()}")
                print(f"      modality_masks[1] (audio) sum: {modality_masks[1].sum().item()}")
                print(f"      Total tokens: {modality_masks[0].sum().item() + modality_masks[1].sum().item()}")
            
            # Debug: Track which step we're on
            if global_step < 3:
                print(f"   📍 Step {global_step}: Starting forward pass...")

            logits = model(input_ids, start_pos=0, modality_masks=modality_masks)

            if global_step < 3:
                print(f"   ✅ Step {global_step}: Forward pass complete")
            
            # Debug: Check for NaN/inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"   ⚠️  WARNING: NaN/Inf detected in logits at step {global_step}")
                print(f"      NaN count: {torch.isnan(logits).sum().item()}, Inf count: {torch.isinf(logits).sum().item()}")
                print(f"      Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Debug: Check labels are valid (ignore -100 which is used for padding/ignored tokens)
            invalid_mask = (shift_labels != -100) & ((shift_labels < 0) | (shift_labels >= model_args.vocab_size))
            if invalid_mask.any():
                invalid_labels = shift_labels[invalid_mask]
                print(f"   ⚠️  WARNING: Invalid labels at step {global_step}")
                print(f"      Invalid count: {invalid_mask.sum().item()}")
                print(f"      Invalid label values: {invalid_labels.unique()}")
                print(f"      Label range: [{shift_labels.min().item()}, {shift_labels.max().item()}], vocab_size={model_args.vocab_size}")
                # Clamp invalid labels (but keep -100 as is)
                shift_labels = torch.where(invalid_mask, shift_labels.clamp(0, model_args.vocab_size - 1), shift_labels)
            
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Debug: Check loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   ⚠️  WARNING: NaN/Inf loss at step {global_step}")
                print(f"      Logits shape: {shift_logits.shape}, Labels shape: {shift_labels.shape}")
                print(f"      Logits stats: min={shift_logits.min().item():.4f}, max={shift_logits.max().item():.4f}")
                print(f"      Labels stats: min={shift_labels.min().item()}, max={shift_labels.max().item()}")
                # Skip this batch by zeroing gradients and continuing
                optimizer.zero_grad()
                global_step += 1
                continue
            
            if global_step < 3:
                print(f"   📍 Step {global_step}: Starting backward pass...")

            loss.backward()

            if global_step < 3:
                print(f"   ✅ Step {global_step}: Backward pass complete")

            # PROOF: Show we can do multiple steps without graph error
            if global_step == 2:
                print("\n" + "="*60)
                print("✅ PROOF: BACKWARD GRAPH ERROR IS FIXED!")
                print("Successfully completed 3 training steps without graph error")
                print("="*60)

            # Gradient clipping to prevent NaN from gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            global_step += 1
            total_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            
            # Logging
            if global_step % log_every_n_steps == 0:
                avg_loss = total_loss / global_step
                print(f"     Step {global_step}: Loss = {loss.item():.4f} | Avg Loss = {avg_loss:.4f}")
            
            # Save checkpoint
            if global_step % save_every_n_steps == 0:
                checkpoint_dir = "checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"adapter_step_{global_step}.pt")
                
                # Save only adapter weights
                adapter_state = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:  # Only save trainable (adapter) weights
                        adapter_state[name] = param.data.cpu()
                
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'adapter_state_dict': adapter_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"     💾 Saved checkpoint: {checkpoint_path}")
        
        if global_step >= max_steps:
            break
            
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"   Epoch {epoch + 1} complete: Avg Loss = {avg_epoch_loss:.4f}\n")
    
    # Final checkpoint
    final_checkpoint_path = os.path.join("checkpoints", "adapter_final.pt")
    adapter_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            adapter_state[name] = param.data.cpu()
    
    torch.save({
        'step': global_step,
        'adapter_state_dict': adapter_state,
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_checkpoint_path)
    
    print("✅ Training Complete!")
    print(f"   - Total steps: {global_step}")
    print(f"   - Final checkpoint: {final_checkpoint_path}")
    print(f"   - Average loss: {total_loss / global_step:.4f}")

if __name__ == "__main__":
    train()
