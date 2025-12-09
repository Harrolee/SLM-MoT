#!/usr/bin/env python3
"""Fix collate_fn indentation"""

with open('train_adapter.py', 'r') as f:
    content = f.read()

# Replace the problematic section
old_section = """    else:
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
        # musicgen.audio_encoder is EnCodecModel
        encoder_outputs = musicgen_full_model.audio_encoder(inputs.input_values)
        audio_codes = encoder_outputs.audio_codes
        
        # Flatten Strategy: Interleave all 4 codebooks
        # audio_codes: [batch, n_codebooks, seq_len] or [batch, n_codebooks, 1, seq_len]
    
        # Handle list output if any
        if isinstance(audio_codes, list):
        audio_codes = torch.stack(audio_codes)
        
        # Handle various dimensions
        while audio_codes.dim() > 3:
        # Squeeze any singleton dimensions
        audio_codes = audio_codes.squeeze()
        
        # Now should be [batch, n_codebooks, seq_len] or [n_codebooks, seq_len] if batch_size=1
        if audio_codes.dim() == 3:
        b, k, t = audio_codes.shape
        elif audio_codes.dim() == 2:
        # If 2D, assume [n_codebooks, seq_len] - add batch dimension
        k, t = audio_codes.shape
        audio_codes = audio_codes.unsqueeze(0)  # [1, n_codebooks, seq_len]
        b = 1
    else:
        raise ValueError(f"Unexpected audio_codes dimensions: {audio_codes.dim()}, shape: {audio_codes.shape}")
    
        # Transpose to [b, t, k] then flatten to [b, t*k]
        # This interleaves codebooks: c0_t0, c1_t0, c2_t0, c3_t0, c0_t1...
        audio_tokens = audio_codes.permute(0, 2, 1).contiguous().view(b, t * k)"""

new_section = """    else:
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
    audio_tokens = audio_tokens.detach().requires_grad_(False)"""

content = content.replace(old_section, new_section)

with open('train_adapter.py', 'w') as f:
    f.write(content)

print("Fixed collate_fn!")

