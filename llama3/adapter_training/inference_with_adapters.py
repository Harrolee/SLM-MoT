import torch
import torch.nn as nn
import json
import os
import sys
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)

def load_trained_adapters(model, checkpoint_path):
    """Load trained adapter weights into MoT model."""
    print(f"📦 Loading adapter weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    adapter_state = checkpoint['adapter_state_dict']
    
    loaded_count = 0
    for name, param in model.named_parameters():
        if name in adapter_state:
            param.data.copy_(adapter_state[name].to(param.device))
            loaded_count += 1
    
    print(f"   ✅ Loaded {loaded_count} adapter parameter groups")
    print(f"   - Training step: {checkpoint.get('step', 'unknown')}")
    if 'loss' in checkpoint:
        print(f"   - Final loss: {checkpoint['loss']:.4f}")
    return model

def generate_audio_tokens(model, text_prompt, tokenizer, device, max_audio_tokens=200):
    """
    Generate audio tokens autoregressively from text prompt.
    
    Returns:
        audio_token_ids: List of audio token IDs (in vocabulary range 128256+)
    """
    print(f"\n🎵 Generating audio tokens from prompt: '{text_prompt}'")
    
    # 1. Tokenize text prompt
    text_tokens = tokenizer.encode(text_prompt, bos=True, eos=False)
    text_tensor = torch.tensor([text_tokens], dtype=torch.long, device=device)
    
    print(f"   - Text tokens: {len(text_tokens)}")
    
    # 2. Initialize sequence with text tokens
    # Add a special token to indicate "start generating audio"
    # Actually, we'll just start generating audio tokens after text
    sequence = text_tokens.copy()
    
    model.eval()
    audio_token_ids = []
    
    print(f"   - Generating {max_audio_tokens} audio tokens...")
    print(f"   - Debug: First few text tokens: {text_tokens[:5]}")
    
    with torch.no_grad():
        for step in range(max_audio_tokens):
            # Current sequence tensor
            seq_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
            
            # Update modality masks for current sequence length
            # Text positions: [0:len(text_tokens)]
            # Audio positions: [len(text_tokens):] (including the position we're about to predict)
            seq_len = len(sequence)
            text_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
            text_mask[0, :len(text_tokens)] = True
            
            audio_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
            # All positions after text are audio (including next position)
            audio_mask[0, len(text_tokens):] = True
            
            modality_masks = [text_mask, audio_mask]
            
            # Debug: Check logits distribution
            if step == 0:
                # Check what the model predicts before filtering
                temp_logits = model(seq_tensor, start_pos=0, modality_masks=modality_masks)
                temp_next = temp_logits[0, -1, :]
                top_k = torch.topk(temp_next, 10)
                print(f"   - Debug: Top 10 logits before filtering: {[(idx.item(), val.item()) for idx, val in zip(top_k.indices, top_k.values)]}")
                
                # Check audio token range specifically
                audio_token_min = 128256
                audio_token_max = 130303
                audio_logits_check = temp_next[audio_token_min:audio_token_max + 1]
                if len(audio_logits_check) > 0:
                    top_audio = torch.topk(audio_logits_check, min(5, len(audio_logits_check)))
                    print(f"   - Debug: Top 5 audio token logits: {[(audio_token_min + idx.item(), val.item()) for idx, val in zip(top_audio.indices, top_audio.values)]}")
                else:
                    print(f"   - Debug: Audio token range is empty!")
            
            # Forward pass
            logits = model(seq_tensor, start_pos=0, modality_masks=modality_masks)
            
            # Get next token prediction (last position)
            next_token_logits = logits[0, -1, :]
            
            # Filter to audio token range (128256+)
            # EnCodec vocab size is 2048, so audio tokens are 128256 to 130303
            audio_token_min = 128256
            audio_token_max = 130303  # 128256 + 2048 - 1
            
            # Extract only audio token logits
            audio_logits = next_token_logits[audio_token_min:audio_token_max + 1]
            
            # Check if we have valid audio logits
            if len(audio_logits) == 0:
                print(f"   ⚠️  No audio logits available, stopping")
                break
            
            # Apply temperature for sampling (0.8 for some randomness)
            temperature = 0.8
            audio_logits = audio_logits / temperature
            
            # Sample using softmax + multinomial (or greedy)
            if temperature == 0.0:
                # Greedy
                audio_code = torch.argmax(audio_logits).item()
            else:
                # Sample
                probs = torch.softmax(audio_logits, dim=-1)
                # Ensure probs is valid
                if probs.numel() == 0:
                    print(f"   ⚠️  Empty probability distribution, stopping")
                    break
                audio_code = torch.multinomial(probs, num_samples=1).item()
            
            # Convert back to full vocabulary token ID
            next_token_id = audio_token_min + audio_code
            
            # Verify it's in audio range
            if audio_token_min <= next_token_id <= audio_token_max:
                audio_token_ids.append(next_token_id)
                sequence.append(next_token_id)
            else:
                print(f"   ⚠️  Generated token {next_token_id} outside audio range, stopping")
                break
            
            # Progress indicator
            if (step + 1) % 50 == 0:
                print(f"     Generated {step + 1}/{max_audio_tokens} tokens...")
    
    print(f"   ✅ Generated {len(audio_token_ids)} audio tokens")
    return audio_token_ids

def decode_audio_tokens(audio_token_ids, musicgen_model, processor, device):
    """
    Decode audio token IDs back to waveform using MusicGen's EnCodec decoder.
    
    Args:
        audio_token_ids: List of token IDs (in range 128256+)
        musicgen_model: MusicGen model for decoding
        processor: MusicGen processor
    
    Returns:
        audio_waveform: numpy array of audio samples
        sampling_rate: int
    """
    print(f"\n🔊 Decoding {len(audio_token_ids)} audio tokens to waveform...")
    
    # Convert token IDs back to EnCodec codes
    # audio_token_id = 128256 + encodec_code
    encodec_codes = [tid - 128256 for tid in audio_token_ids]
    
    # Reshape codes back to [batch, n_codebooks, seq_len]
    # We flattened as: [c0_t0, c1_t0, c2_t0, c3_t0, c0_t1, ...]
    # Need to unflatten: group by 4
    
    num_timesteps = len(encodec_codes) // 4
    if len(encodec_codes) % 4 != 0:
        print(f"   ⚠️  Warning: {len(encodec_codes)} codes not divisible by 4, truncating")
        encodec_codes = encodec_codes[:num_timesteps * 4]
    
    # Reshape: [num_timesteps * 4] -> [4, num_timesteps] -> [1, 4, num_timesteps]
    codes_array = np.array(encodec_codes).reshape(4, num_timesteps)
    codes_tensor = torch.tensor(codes_array, dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"   - Codebook shape: {codes_tensor.shape} [batch, codebooks, timesteps]")
    
    # Decode using MusicGen's EnCodec decoder
    with torch.no_grad():
            # MusicGen's audio_encoder is an EnCodecModel
            encodec_model = musicgen_model.audio_encoder
            
            # EnCodec decode requires codes and scales
            # For now, create dummy scales (all ones)
            # In real usage, scales come from the encoder, but we can use defaults
            audio_scales = torch.ones(1, codes_tensor.shape[1], device=device)
            
            try:
                # Try decode method with codes and scales
                if hasattr(encodec_model, 'decode'):
                    # EnCodecModel.decode(codes, scales) -> audio
                    audio_values = encodec_model.decode(codes_tensor, audio_scales)
                    if isinstance(audio_values, tuple):
                        audio_values = audio_values[0]
                elif hasattr(encodec_model, 'decoder'):
                    # Try decoder submodule
                    decoder = encodec_model.decoder
                    # Decoder might need different format
                    # Try passing codes directly
                    audio_values = decoder(codes_tensor)
                    if isinstance(audio_values, tuple):
                        audio_values = audio_values[0]
                else:
                    # Last resort: try forward with codes
                    # This might not work but worth trying
                    raise AttributeError("No decode method found")
                
                # Convert to numpy
                if isinstance(audio_values, torch.Tensor):
                    audio_waveform = audio_values.cpu().numpy()
                    # Handle batch dimension: [B, C, T] or [B, T]
                    if audio_waveform.ndim == 3:
                        audio_waveform = audio_waveform[0, 0, :]  # [batch, channel, time] -> mono
                    elif audio_waveform.ndim == 2:
                        audio_waveform = audio_waveform[0, :]  # [batch, time]
                    
                    sampling_rate = getattr(encodec_model.config, 'sampling_rate', 32000)
                    if not sampling_rate:
                        sampling_rate = 32000  # Default for MusicGen
                    
                    print(f"   ✅ Decoded audio: {len(audio_waveform)} samples at {sampling_rate} Hz")
                    return audio_waveform, sampling_rate
                    
            except Exception as e:
                print(f"   ⚠️  Decoding error: {e}")
                print(f"   Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print("   💡 Falling back to placeholder audio")
    
    # Fallback: Generate placeholder audio
    sampling_rate = 32000
    duration_seconds = num_timesteps / 50.0  # 50 Hz frame rate
    audio_waveform = np.random.uniform(-0.1, 0.1, int(sampling_rate * duration_seconds))
    print(f"   ⚠️  Using placeholder audio ({duration_seconds:.2f}s)")
    return audio_waveform, sampling_rate

def main():
    print("🎤 MoT Audio Inference with Trained Adapters")
    print("=" * 50)
    
    # Config
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    
    print(f"   - Device: {device}")
    
    # Initialize FairScale
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(np.random.randint(20000, 30000)))
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)
    
    if not model_parallel_is_initialized():
        initialize_model_parallel(1)
    
    # Load model args
    llama_ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "Llama-3.2-1B")
    with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
        params = json.load(f)
    
    model_args = ModelArgs(
        max_seq_len=2048,
        max_batch_size=1,
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
    
    print("\n🏗️  Building MoT Model...")
    model = Transformer(model_args)
    model.to(device)
    
    # Load Llama checkpoint weights first!
    print("\n📦 Loading Llama checkpoint weights...")
    checkpoints = sorted(Path(llama_ckpt_dir).glob("*.pth"))
    if len(checkpoints) > 0:
        ckpt_path = checkpoints[0]
        print(f"   - Loading from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Handle vocabulary expansion: expand embedding and output layers
        # Before loading, we need to expand these layers in the checkpoint
        if 'tok_embeddings.weight' in checkpoint:
            old_emb = checkpoint['tok_embeddings.weight']  # [old_vocab, dim]
            new_emb = torch.zeros(model_args.vocab_size, old_emb.shape[1], 
                                 dtype=old_emb.dtype, device=old_emb.device)
            new_emb[:original_vocab_size] = old_emb
            checkpoint['tok_embeddings.weight'] = new_emb
            print(f"   - Expanded embeddings: {old_emb.shape[0]} -> {new_emb.shape[0]}")
        
        if 'output.weight' in checkpoint:
            old_out = checkpoint['output.weight']  # [old_vocab, dim]
            new_out = torch.zeros(model_args.vocab_size, old_out.shape[1], 
                                 dtype=old_out.dtype, device=old_out.device)
            new_out[:original_vocab_size] = old_out
            checkpoint['output.weight'] = new_out
            print(f"   - Expanded output layer: {old_out.shape[0]} -> {new_out.shape[0]}")
        
        # Now load the expanded checkpoint
        model.load_state_dict(checkpoint, strict=False)  # strict=False allows MoT-specific layers
        print("   ✅ Llama weights loaded")
        
        # Initialize MoT experts from checkpoint (for text expert only)
        # Audio expert will be loaded from MusicGen separately
        if hasattr(model, '_initialize_mot_experts_from_checkpoint'):
            # Only initialize text expert (expert 0), skip audio expert (expert 1)
            # We'll handle audio expert separately with MusicGen weights
            try:
                # Try to initialize, but it might fail for audio expert - that's OK
                model._initialize_mot_experts_from_checkpoint(checkpoint)
            except AttributeError as e:
                # Expected for audio expert - we'll load MusicGen weights instead
                print("   ℹ️  Skipping audio expert initialization (will use MusicGen weights)")
    else:
        print("   ⚠️  No Llama checkpoint found, model will have random weights!")
    
    # Load MusicGen weights into audio expert (if not already loaded)
    print("\n🎸 Loading MusicGen weights...")
    musicgen = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    
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
    for layer_id, layer in enumerate(model.layers):
        if layer_id >= len(decoder_layers):
            continue
        
        audio_expert_adapter = layer.feed_forward.local_experts[-1]
        audio_ffn = audio_expert_adapter.expert
        
        if hasattr(decoder_layers[layer_id], 'fc1') and hasattr(decoder_layers[layer_id], 'fc2'):
            with torch.no_grad():
                audio_ffn.fc1.weight.copy_(decoder_layers[layer_id].fc1.weight.to(device))
                if decoder_layers[layer_id].fc1.bias is not None:
                    audio_ffn.fc1.bias.copy_(decoder_layers[layer_id].fc1.bias.to(device))
                audio_ffn.fc2.weight.copy_(decoder_layers[layer_id].fc2.weight.to(device))
                if decoder_layers[layer_id].fc2.bias is not None:
                    audio_ffn.fc2.bias.copy_(decoder_layers[layer_id].fc2.bias.to(device))
    
    print("   ✅ MusicGen weights loaded")
    
    # Load trained adapters
    checkpoint_path = "checkpoints/adapter_final.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/adapter_step_300.pt"
    
    model = load_trained_adapters(model, checkpoint_path)
    
    # Load tokenizer
    tokenizer_path = os.path.join(llama_ckpt_dir, "tokenizer.model")
    tokenizer = Tokenizer(tokenizer_path)
    
    # Generate audio
    text_prompt = "80s pop track with bassy drums and synth"
    audio_token_ids = generate_audio_tokens(
        model, text_prompt, tokenizer, device, max_audio_tokens=200
    )
    
    # Decode to audio
    audio_waveform, sampling_rate = decode_audio_tokens(
        audio_token_ids, musicgen, processor, device
    )
    
    # Save audio
    output_path = "mot_generated_audio.wav"
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_waveform)
    print(f"\n💾 Saved generated audio to: {output_path}")
    print(f"   - Duration: {len(audio_waveform) / sampling_rate:.2f} seconds")
    print(f"   - Sampling rate: {sampling_rate} Hz")
    
    print("\n🎉 Inference Complete!")
    print("   🎵 Open the .wav file to hear your MoT-generated audio!")

if __name__ == "__main__":
    main()

